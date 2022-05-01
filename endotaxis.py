import numpy as np
import matplotlib.pyplot as plt

class Endotaxis:
    def __init__(self, params, graph):
        self.graph = graph

        self.nP = params['nP'] # number of places (point cells), place/point used interchangeably
        self.nG = params['nG'] # number of goals
        self.goals = np.array(params['goals']) # an array whose elements are the node numbers of goal cells (0, 1, ... nG-1) in the graph
        self.w = params['w'] # decay constant, neighboring point cells fire ar rate w
        self.beta_mm = params['beta_mm']
        self.alpha_mm = params['alpha_mm']
        self.beta_gm = params['beta_gm']
        self.alpha_gm = params['alpha_gm']

        self.F = np.zeros((self.nG, self.nP)) # amount of resource--feature cells' input to goal cells, fixed as a table
        self.init_feature()

        self.G = np.zeros((self.nG, 1)) # goal cells' activity
        self.P = np.zeros((self.nP, 1)) # point cells' activity
        self.M = np.zeros((self.nP, 1)) # map cells' activity; the num of map cells is the same as that of point cells

        self.MM = np.zeros((self.nP, self.nP)) # recurrent connections among map cells
        self.GM = np.zeros((self.nG, self.nP))

        self.curr_node = -1

    def init_feature(self):
        if len(self.goals) != self.nG:
            print('Number of goals not equal to the number of goal cells!')
            return
        else:
            self.F[:, self.goals] = np.eye(self.nG)

    def gen_trajs(self):
        if self.graph.get_graph_name() == 'mat1':
            traj = [8,4,5,9,8,9,5,1,2,3,7,11,10,9] * 10
        else:
            traj = []
        return traj

    def get_point_cells_activity(self, node):
        P = np.zeros((self.nP, 1))
        P[node, 0] = 1
        for adj_node in self.graph.adj_sparse.getrow(node).nonzero()[1]:
            if adj_node != node:
                P[adj_node, 0] = self.w
        return P

    def update_point_cells(self):
        self.P = self.get_point_cells_activity(self.curr_node)

    def get_map_cells_activity(self, node):
        P = self.get_point_cells_activity(node)
        try:
            K = np.linalg.inv(np.eye(self.nP) - self.MM)
            M = K @ P
            return M
        except np.linalg.LinAlgError as err:
            print('Matrix inversion fails in method "get_map_cells_activity" of class "Endotaxis" with error:', err)
            raise

    def update_map_cells(self):
        self.M = self.get_map_cells_activity(self.curr_node)

    def get_goal_cells_activity(self, node):
        M = self.get_map_cells_activity(node)
        G = self.F[:, self.curr_node].reshape(-1, 1) + self.GM @ M
        return G

    def update_goal_cells(self):
        self.G = self.get_goal_cells_activity(self.curr_node)

    def train(self):
        log = []
        traj = self.gen_trajs()
        for self.curr_node in traj:
            self.update_point_cells()
            try:
                self.update_map_cells()
            except np.linalg.LinAlgError:
                print('Training not finished due to matrix inversion error')
                return
            self.update_goal_cells()
            self.MM += self.beta_mm * (self.alpha_mm * self.M @ self.M.T - self.MM * self.M ** 2)
            self.GM += self.beta_gm * (self.alpha_gm * self.G @ self.M.T - self.GM * self.G ** 2)
            self.MM = np.where(self.MM < 0, 0, self.MM)
            self.GM = np.where(self.GM < 0, 0, self.GM)
            np.fill_diagonal(self.MM, 0)
            log.append({'MM': self.MM,
                        'GM': self.GM})
        return log

    def navigatability(self):
        goal_cells_resp = np.zeros((self.nG, self.nP)) # goal cell's activities at all places, for subsequent gradient-ascent
        for node in range(self.nP):
            goal_cells_resp[:, node] = self.get_goal_cells_activity(node).reshape(3)

        next_node_mat = np.zeros((self.nP, self.nG))
        for start in range(self.nP):
            for goal_idx, goal in zip(range(self.nG), self.goals):
                adj_nodes = self.graph.adj_sparse.getrow(start).nonzero()[1]
                if len(adj_nodes) == 0:
                    next_node = -1
                elif np.max(goal_cells_resp[goal_idx, adj_nodes]) <= goal_cells_resp[goal_idx, start]:
                    next_node = start # nowhere else to go, it's either a local maximum or already at goal loc'n
                else:
                    next_node = adj_nodes[np.argmax(goal_cells_resp[goal_idx, adj_nodes])]
                next_node_mat[start, goal_idx] = next_node
        return next_node_mat

    def comp_vect_field(self):
        locns = np.asarray(
            ((0, 2), (1, 2), (2, 2), (3, 2), (0, 1), (1, 1), (2, 1), (3, 1), (0, 0), (1, 0), (2, 0), (3, 0)))
        goal_dominance_factor = 40
        learning_rate_scale = 10

        fig, axs = plt.subplots(1, 5)
        goal_cell_idx, goal = 0, 3
        K = np.linalg.inv(np.eye(self.nP) - self.MM)

        current_node = 8
        axs = axs.flatten()
        ax_idx = 0
        while current_node != goal:
            P = self.get_point_cells_activity(current_node)
            M = K @ (P + goal_dominance_factor * self.GM[0,:].reshape(-1,1))
            # print(self.GM[0,[1,4,5,9]])
            MM = self.MM + learning_rate_scale * self.beta_mm * (self.alpha_mm * M @ M.T)# - self.MM * M ** 2)

            axs[ax_idx].scatter(locns[:, 0], locns[:, 1], s=100, marker='o', color='black')
            axs[ax_idx].scatter(locns[goal, 0], locns[goal, 1], s=100, marker='o', color='green')  # mark the goal location
            axs[ax_idx].set_title(f'Goal {goal}')
            axs[ax_idx].axis('equal')

            for node in range(self.nP):
                adj_nodes = self.graph.adj_sparse.getrow(node).nonzero()[1]
                # print(adj_nodes)
                for adj_node in adj_nodes:
                    axs[ax_idx].plot([locns[node,0], locns[adj_node,0]], [locns[node,1], locns[adj_node,1]], color='orange', linewidth = MM[node, adj_node] * 500)
            adj_nodes = self.graph.adj_sparse.getrow(current_node).nonzero()[1]
            next_node = np.argmax(MM[current_node, adj_nodes])
            current_node = next_node
            ax_idx += 1
        plt.show()
