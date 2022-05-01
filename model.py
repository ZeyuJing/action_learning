import numpy as np
import matplotlib.pyplot as plt

class Model:
    def __init__(self, graph, params):
        self.graph = graph

        #dimension parameters
        self.nG = params['nG']
        self.nP = params['nP']
        self.nA = params['nA']
        self.nS = params['nS']

        #learning parameters
        self.beta_sg, self.alpha_sg  = params['beta_sg'], params['alpha_sg']
        self.beta_sp, self.alpha_sp = params['beta_sp'], params['alpha_sp']
        self.beta_ss, self.alpha_ss = params['beta_ss'], params['alpha_ss']
        self.beta_ps, self.alpha_ps = params['beta_ps'], params['alpha_ps']
        self.beta_as, self.alpha_as = params['beta_as'], params['alpha_as']

        #synaptic weight matrices
        self.SG = np.abs(np.random.randn(self.nS, self.nG)) * 0.2
        self.SP = np.abs(np.random.randn(self.nS, self.nP)) * 0.2
        self.SS = np.zeros((self.nS, self.nS))
        # self.SS = -np.ones((self.nS, self.nS))*1
        # np.fill_diagonal(self.SS, 0.5)

        self.PS = np.zeros((self.nP, self.nS))
        self.AS = np.zeros((self.nA, self.nS))

        #sigmoid / F
        self.k = params['k']
        self.theta = params['theta']


    def F(self, x):
        """sigmoid activation function for state cells' activity"""
        return 1 / (1 + np.exp(-self.k * (x - self.theta)))

    def gen_uniform_seqn(self, goals, num_nodes, num_runs):
        return np.asarray(np.concatenate((np.random.choice(goals, num_runs).reshape(-1,1),
                                         np.random.choice(num_nodes, num_runs).reshape(-1,1)),
                                         axis=1))

    def gen_sqn(self, predecessors, goals, num_nodes, num_runs):
        """
        Generate a sequence of training runs for state cells' sparse representation of (goal, location) pairing. For each
        run, it picks a goal from the list 'goals' and then randomly picks a starting point from the rest of nodes.
        Each run consists of a list of nodes defined the matrix predecessors.
        :param predecessors: matrix of predecessor for each (start, goal) pair
        :param goals: goal indices
        :param num_nodes: number of nodes in the graph
        :param num_runs: number of runs to generate
        :return sqn: list of runs
        """
        goals = np.asarray(goals)
        sqn = []
        i = 0
        while i < num_runs:
            goal = goals[np.random.randint(0, len(goals))]
            start = np.random.randint(0, num_nodes - 1)
            if start >= goal:
                start += 1
            ls = [goal]
            x = predecessors[start, goal]
            if x < 0:
                continue
            while x >= 0:
                ls.append(x)
                x = predecessors[start, x]
            ls.reverse()
            sqn.append(ls)
            i += 1
        return sqn

    def solve_dyn_sys(self, F, M, x, SS):
        """
        ds/dt = -s + F(nG*x + SS*s)
        Solve the above dynamical system using 1st order Euler method to get a self-consistent solution for the firing rates of state cells
        :param F: non-linear function, sigmoid
        :param M: syn weight matrix, from inputs to state cells
        :param x: inputs
        :param SS: recurrent anti-hebbian inhibitory weights
        :return s: state cell activity, which has to go through a step function later
        """
        dt = 0.1
        s = np.zeros((1, SS.shape[0]))  # all rate array is a 1-by-n array
        dsdt = None
        for _ in range(1000):
            dsdt = -s.T + F(M @ x.T + SS @ s.T)
            s += dt * dsdt.T
        # print(np.abs(dsdt).max())
        return s

    def display_mat_ctrl(self, fig, axs, row_idx, run_idx):
        mats = self.sort_mat()[0:3]  # sorted matrices
        for idx, ax, mat in zip(range(len(mats)), axs[row_idx, :], mats):
            if idx == 0:
                aspect = 0.1
                ax.text(-4, 0.5, f'Run: {run_idx}', va='center', ha='right', fontsize=12, transform=ax.transAxes)
            else:
                aspect = 1
            ax.imshow(mat, aspect=aspect, interpolation='none')
            ax.set_yticks(range(0, mat.shape[0], 20))
            if idx == 0:
                ax.set_xticks(range(0, mat.shape[1], 1))
            else:
                ax.set_xticks(range(0, mat.shape[1], 20))
        if row_idx == axs.shape[0] - 1:  # if last row
            for ax, ax_name in zip(axs[0, :], ('SG', 'SP', 'SS')):
                ax.set_title(ax_name)

    def display_mat(self, fig, axs, row_idx, run_idx):
        """
        Display the five matrices SG, SP, SS, PS, AS.
        :param fig: figure to be plotted on
        :param axs: axs to be plotted on
        :param row_idx: current row idx of axes
        :param run_idx: current index of run
        """
        mats = self.sort_mat() # sorted matrices
        images = []
        for idx, ax, mat in zip(range(len(mats)), axs[row_idx,:], mats):
            if idx == 0:
                aspect = 0.1
                ax.text(-4, 0.5, f'Run: {run_idx}', va='center', ha='right', fontsize=12, transform=ax.transAxes)
            elif idx == 4:
                aspect = 10
            else:
                aspect = 1
            img = ax.imshow(mat, aspect=aspect, interpolation='none')
            if row_idx == axs.shape[0] - 1: #last row
                images.append(img)
            if idx != 4:
                ax.set_yticks(range(0, mat.shape[0], 20))
            else:
                ax.set_yticks(range(0, mat.shape[0], 1))
            if idx == 0:
                ax.set_xticks(range(0, mat.shape[1], 1))
            else:
                ax.set_xticks(range(0, mat.shape[1], 20))

        if row_idx == axs.shape[0]-1: # if last row
            for ax, ax_name in zip(axs[0,:], ('SG','SP','SS','PS','AS')):
                ax.set_title(ax_name)
            # for im, ax in zip(images, axs[-1,:]):
            #     fig.colorbar(mappable=im, ax=ax, orientation='horizontal')

    def sort_mat(self):
        """
        Sort state cells according to 1) which goal it's encoding 2) which place it's encoding in lexicographic order.
        :return: sorted matrices, note that the sorted matrices are copies
        """
        indices= np.lexsort(np.concatenate([np.argmax(self.SP, axis=1).reshape(1, -1),
                                            np.argmax(self.SG, axis=1).reshape(1, -1)], axis=0))
        SG = self.SG[indices, :]
        SP = self.SP[indices, :]
        SS = self.SS[indices, :]
        SS = SS[:,indices]
        PS = self.PS[:, indices]
        AS = self.AS[:, indices]
        return [SG, SP, SS, PS, AS]

    def train_ctrl(self, goals, num_runs=2001, display_mat=True, num_plot_steps=1000):
        Gc = np.zeros((1, self.nG))  # goal cell firing rates; row vector
        Pc = np.zeros((1, self.nP))  # point cell firing rates; row vector
        Sc = np.zeros((1, self.nS))  # state cell firing rates; goal * map

        sqn = self.gen_uniform_seqn(goals, self.nP, num_runs)  # sequence of shortest-path runs for training
        x = -1  # agent's location

        if display_mat:
            num_rows = int(num_runs / num_plot_steps) + 1
            fig, axs = plt.subplots(num_rows, 3, figsize=(4 * 3, 3 * num_rows))

        for run, idx in zip(sqn, range(len(sqn))):
            goal, x = run[0], run[1]
            Gc *= 0
            Gc[0, np.nonzero(goals == goal)] = 1
            Pc *= 0
            Pc[0, x] = 1
            Sc *= 0

            if display_mat and idx % num_plot_steps == 0:
                self.display_mat_ctrl(fig, axs, row_idx=int(idx / (num_plot_steps - 1)), run_idx=idx)

            # update state cell activity
            Sc = self.solve_dyn_sys(self.F, np.concatenate((self.SG, self.SP), axis=1),
                                    np.concatenate((Gc, Pc), axis=1), self.SS)
            Sc = np.where(Sc > 0.5, 1, 0)

            # weight updates for goal->state, place->state, state->state
            self.SG += self.beta_sg * (Sc.T @ Gc - self.alpha_sg * (Sc.T ** 2) * self.SG)
            # for each postdynaptic neuron, the depression term is nonzero when it fires, compare with action learning above
            self.SP += self.beta_sp * (Sc.T @ Pc - self.alpha_sp * (Sc.T ** 2) * self.SP)
            self.SS -= self.beta_ss * ((Sc.T @ Sc) + self.alpha_ss * self.SS)  # anti-hebbian
            np.fill_diagonal(self.SS, 0)  # make sure there is no self connection
            print(f'Progress: {idx / len(sqn) * 100}% finished.')
        if display_mat:
            plt.show()

    def train(self, goals, num_runs=2000, display_mat = True, num_plot_steps=1000):
        """
        Train the model using Hebbian learning. The desired result is that the each state cell will encode one and only
        one combination of goal and location, or (goal, location) pair. Also, each state cell will be associated with an
        action that brings the agent closer to the goal according to the shortest path.
        :param goals: list of goal locations
        :param num_runs: number of training runs
        :param display_mat: display the connectivity matrices
        :param num_plot_steps: plot every __ steps
        """
        #initialize neurons
        Gc = np.zeros((1, self.nG))  # goal cell firing rates; row vector
        Pc = np.zeros((1, self.nP))  # point cell firing rates; row vector
        Sc = np.zeros((1, self.nS))  # state cell firing rates; goal * map
        Ac = np.zeros((1, self.nA))  # action cell firing rates; row vector

        sqn = self.gen_sqn(self.graph.predecessors, goals, self.nP, num_runs) # sequence of shortest-path runs for training
        x=-1 # agent's location

        if display_mat:
            num_rows = int(num_runs/num_plot_steps) + 1
            fig, axs = plt.subplots(num_rows, 5, figsize=(4*5, 3*num_rows))

        for run, idx in zip(sqn, range(len(sqn))):
            start, end = run[0], run[-1]
            Gc *= 0
            Gc[0, np.nonzero(goals == end)] = 1
            Sc *= 0
            Pc *= 0
            Ac *= 0
            act = -1 # action taken

            if display_mat and idx % num_plot_steps == 0:
                self.display_mat(fig, axs, row_idx=int(idx/(num_plot_steps-1)), run_idx=idx)

            for x in run:
                prev_node = self.graph.predecessors[start, x] # previous node visited
                if prev_node >= 0:
                    act = int(self.graph.get_action_idx(self.graph.actions[prev_node, x]))
                Pc *= 0
                Pc[0, x] = 1 # update point cell activity according to the current node
                Ac *= 0
                if act >= 0:  # -9999 is used by csgraph as N/A
                    Ac[0, act] = 1 # update action cell activity according to previous action taken

                # weight updates for state->place and state->action
                self.PS += self.beta_ps * (Pc.T @ Sc - self.alpha_ps * self.PS * Sc)
                # for each presynaptic neuron, the depression term is nonzero when it fires
                self.AS += self.beta_as * (Ac.T @ Sc - self.alpha_as * self.AS * Sc)

                # update state cell activity
                Sc = self.solve_dyn_sys(self.F, np.concatenate((self.SG, self.SP), axis=1), np.concatenate((Gc, Pc), axis=1), self.SS)
                Sc = np.where(Sc > 0.5, 1, 0)

                # weight updates for goal->state, place->state, state->state
                self.SG += self.beta_sg * (Sc.T @ Gc - self.alpha_sg * (Sc.T ** 2) * self.SG)
                # for each postdynaptic neuron, the depression term is nonzero when it fires, compare with action learning above
                self.SP += self.beta_sp * (Sc.T @ Pc - self.alpha_sp * (Sc.T ** 2) * self.SP)
                self.SS -= self.beta_ss * ((Sc.T @ Sc) + self.alpha_ss * self.SS) # anti-hebbian
                np.fill_diagonal(self.SS, 0)  # make sure there is no self connection
            # print(f'Progress: {idx/len(sqn) * 100 :.2f}% finished.')
        if display_mat:
            plt.show()

    def performance(self, goals):
        goals = np.asarray(goals)
        Gc = np.zeros((1, self.nG))  # goal cell firing rates; row vector
        Pc = np.zeros((1, self.nP))  # map cell firing rates; row vector
        Sc = np.zeros((1, self.nS))  # state cell firing rates; goal * map
        Sc_indices = []
        expected_num_of_states = 0
        for goal in goals:
            for start in range(self.nP):
                if self.graph.predecessors[start, goal] < 0 or start == goal: # no vectors for start and goal locations
                    continue
                Gc *= 0
                Pc *= 0
                Sc *= 0
                Gc[0, np.nonzero(goals == goal)] = 1 # set goal cells' activities
                Pc[0, start] = 1 # set place cells' activities
                Sc = self.solve_dyn_sys(self.F, np.concatenate((self.SG, self.SP), axis=1),
                                              np.concatenate((Gc, Pc), axis=1), self.SS)
                Sc = np.where(Sc > 0.5, 1, 0)
                indices = np.nonzero(Sc==1)[1]
                if len(indices) > 1:
                    print(f'Warning: goal locn: {goal}, start locn: {start} activate {indices}. Simultaneous firing of multiple state cells!')
                if len(indices) != 0:
                    for k in range(len(indices)):
                        Sc_indices.append(indices[k])
                else:
                    print(f'Warning: no state cell for goal locn: {goal}, start locn: {start}!')
                expected_num_of_states += 1
        performance = len(set(Sc_indices)) / expected_num_of_states
        return performance

    def accuracy(self, goals):
        goals = np.asarray(goals)
        Gc = np.zeros((1, self.nG))  # goal cell firing rates; row vector
        Pc = np.zeros((1, self.nP))  # map cell firing rates; row vector
        Sc = np.zeros((1, self.nS))  # state cell firing rates; goal * map

        num_total_action, num_invalid_action = 0, 0
        for goal in goals:
            for start in range(self.nP):
                if self.graph.predecessors[start, goal] < 0 or start == goal:  # no vector for start and goal locations
                    continue
                else:
                    num_total_action += 1
                Gc *= 0
                Pc *= 0
                Sc *= 0
                Gc[0, np.nonzero(goals == goal)] = 1  # set goal cells' activity
                Pc[0, start] = 1  # set place cells' activity
                Sc = self.solve_dyn_sys(self.F, np.concatenate((self.SG, self.SP), axis=1),
                                        np.concatenate((Gc, Pc), axis=1), self.SS)
                Sc = np.where(Sc > 0.5, 1, 0)

                if np.max(self.AS @ Sc.T) < 0.5:
                    action = None
                    num_invalid_action += 1
                    continue
                else:
                    action = self.graph.get_action(np.argmax(self.AS @ Sc.T))
                    if action == self.graph.action_from_to(start, goal):
                        correct = True
                    else:
                        correct = False
                        num_invalid_action += 1
        return 1-num_invalid_action/num_total_action


    def plot_vec_field(self, goals):
        """
        Plot the action vector field at each location for each goal. The vector field describes which direction to goal
        according to the shortest path strategy. Note that this function is only valid for 3 x 4 rectangle graphs such
        as the one in the Endotaxis paper.
        :param goals: list of goal locations
        """

        if self.graph.get_graph_name() in ('mat1', 'mat2'):
            locns = np.asarray([[i,j] for j in range(2,-1,-1) for i in range(0,4,1)])
        elif self.graph.get_graph_name() == 'binary 3 level':
            locns = np.asarray([[i,j] for j in range(3,-1,-1) for i in range(0,7,1)])
        else:
            print(f'Error: Vector field plotting for {self.graph.get_graph_name()} not implemented yet!')
            return

        goals = np.asarray(goals)
        Gc = np.zeros((1, self.nG))  # goal cell firing rates; row vector
        Pc = np.zeros((1, self.nP))  # map cell firing rates; row vector
        Sc = np.zeros((1, self.nS))  # state cell firing rates; goal * map

        fig, axs = plt.subplots(1, len(goals), figsize=(np.max(locns[:,0]) * len(goals), np.max(locns[:,1])))

        for goal, ax in zip(goals, axs.flatten()):
            ax.scatter(locns[:,0], locns[:,1], s=100, marker='o', color='black')
            ax.scatter(locns[goal,0], locns[goal,1], s=100, marker='o', color='blue') # mark the goal location
            ax.set_title(f'Goal {goal}')
            ax.axis('equal')
            for start in range(self.nP):
                if self.graph.predecessors[start, goal] < 0 or start == goal: # no vector for start and goal locations
                    continue
                Gc *= 0
                Pc *= 0
                Sc *= 0
                Gc[0, np.nonzero(goals == goal)] = 1 # set goal cells' activity
                Pc[0, start] = 1 # set place cells' activity
                Sc = self.solve_dyn_sys(self.F, np.concatenate((self.SG, self.SP), axis=1),
                                              np.concatenate((Gc, Pc), axis=1), self.SS)
                Sc = np.where(Sc > 0.5, 1, 0)
                print(f'goal locn: {goal}, start locn: {start}, firing state cells: {str(np.nonzero(Sc==1)[1])}')

                if np.max(self.AS @ Sc.T) < 0.5:
                    action = None
                    continue
                else:
                    action = self.graph.get_action(np.argmax(self.AS @ Sc.T))
                    correct = True if action == self.graph.action_from_to(start, goal) else False

                if action == 2:
                    dx, dy = 0, -0.25
                elif action == 4:
                    dx, dy = -0.25, 0
                elif action == 8:
                    dx, dy = 0, 0.25
                elif action == 6:
                    dx, dy = 0.25, 0
                else:
                    dx, dy = 0, 0

                # plot arrows as vector field
                arrow_color = 'green' if correct else 'red'
                ax.arrow(locns[start, 0], locns[start, 1], dx, dy, color=arrow_color, head_width=0.1, linewidth=2)

                # plot the graph using gray lines to connect adjacent locations
                for i in range(len(self.graph.adj_dense)):
                    for j in range(i + 1, len(self.graph.adj_dense)):
                        if self.graph.adj_dense[i][j] == 1:  # if i and j are connected, plot a grey line
                            ax.plot([locns[i, 0], locns[j, 0]], [locns[i, 1], locns[j, 1]], color='gray', linewidth=1)
        plt.show()

