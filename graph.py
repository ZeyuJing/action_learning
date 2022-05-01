import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse.csgraph as csgraph

class Graph:
    def __init__(self, graph_stored='mat1'):
        """
        Create a graph with adjacency matrix, actions, and shortest path information.
        :param graph_stored: the graph to be created
        """
        self.graph_stored = graph_stored
        self.adj_dense, self.actions, self.action_dict = self.load_graph_info(self.graph_stored)
        self.adj_sparse = csr_matrix(self.adj_dense) # get sparse representation
        self.dist_matrix, self.predecessors = csgraph.floyd_warshall(csgraph=self.adj_sparse, directed=False, return_predecessors=True)

    def get_graph_name(self):
        return self.graph_stored

    def get_action_idx(self, act):
        """
        Get the index of the corresponding action for each graph. It's primarily used for representing each action using
        action ncells from index 0 to n-1 where n is the number of actions.
        :param act: action's informatio stored in the matrix self.actions
        :return idx: the index of action 'act'
        """
        idx = -9999
        if self.graph_stored in ('mat1', 'mat2', 'binary 3 level'):
            idx = act/ 2 - 1
        elif self.graph_stored == 'hanoi':
            m, n = int(act / 10), act % 10 # decimal decomposition
            idx = 3 * m + n # ternary conversion; note: (0,0), (1,1), (2,2) are prohibited in the game but we'll leave them here
        elif self.graph_stored == 'binary':
            idx = act - 1
        return idx

    def get_action(self, act_idx):
        """
        A complementary implementation of self.get_action_idx, i.e, return the corresponding action given the index of
        action cell.
        :param act_idx: index of the action
        :return act: action
        """
        act = -1
        if self.graph_stored in ('mat1', 'mat2', 'binary 3 level'):
            act = 2 * (act_idx + 1)
        elif self.graph_stored == 'hanoi':
            act = int(act_idx/3) * 10 + (act % 3)
        elif self.graph_stored == 'binary':
            act = act_idx + 1
        return act

    def action_from_to(self, start, end):
        if self.predecessors[start, end] < 0:
            return None
        next = end
        while self.predecessors[start, next] != start:
            next = self.predecessors[start, next]
        return self.actions[start, next]

    def load_graph_info(self, graph='mat1'):
        """
        Load stored graphs' information.
        :param graph: name of the graph
        :return adj_dense: dense representation of the graph
        :return actions: matrix of actions connecting adjacent nodes
        :return action_dict: dictionary of the verbal interpretation of each action
        """
        adj_dense, actions, actions_dict = [], [], []
        if graph == 'mat1':
            adj_dense = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
                         [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 1
                         [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
                         [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 3
                         [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],  # 4
                         [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # 5
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
                         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],  # 7
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # 8
                         [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],  # 9
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],  # 10
                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # 11
                         ]
            actions = np.asarray([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
                                  [0, 0, 6, 0, 0, 2, 0, 0, 0, 0, 0, 0],  # 1
                                  [0, 4, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
                                  [0, 0, 4, 0, 0, 0, 0, 2, 0, 0, 0, 0],  # 3
                                  [0, 0, 0, 0, 0, 6, 0, 0, 2, 0, 0, 0],  # 4
                                  [0, 8, 0, 0, 4, 0, 0, 0, 0, 2, 0, 0],  # 5
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
                                  [0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 2],  # 7
                                  [0, 0, 0, 0, 8, 0, 0, 0, 0, 6, 0, 0],  # 8
                                  [0, 0, 0, 0, 0, 8, 0, 0, 4, 0, 6, 0],  # 9
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 6],  # 10
                                  [0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 4, 0],  # 11
                                  ])
            actions_dict = {'2': 'south',
                            '4': 'west',
                            '6': 'east',
                            '8': 'north'}
        elif graph == 'mat2':
            adj_dense = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
                         [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 1
                         [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
                         [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 3
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 4
                         [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 5
                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],  # 6
                         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],  # 7
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # 8
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],  # 9
                         [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1],  # 10
                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # 11
                         ]
            actions = np.asarray([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
                                  [0, 0, 6, 0, 0, 2, 0, 0, 0, 0, 0, 0],  # 1
                                  [0, 4, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
                                  [0, 0, 4, 0, 0, 0, 0, 2, 0, 0, 0, 0],  # 3
                                  [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],  # 4
                                  [0, 8, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0],  # 5
                                  [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 2, 0],  # 6
                                  [0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 2],  # 7
                                  [0, 0, 0, 0, 8, 0, 0, 0, 0, 6, 0, 0],  # 8
                                  [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 6, 0],  # 9
                                  [0, 0, 0, 0, 0, 0, 8, 0, 0, 4, 0, 6],  # 10
                                  [0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 4, 0],  # 11
                                  ])
            actions_dict = {'2': 'south',
                            '4': 'west',
                            '6': 'east',
                            '8': 'north'}
        elif graph == 'binary 3 level':
            #             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27
            adj_dense = [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
                         [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
                         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
                         [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5
                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
                         [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 9
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 11
                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 12
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 13
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 14
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 15
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 16
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 17
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 18
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 19
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 20
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 21
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 22
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 23
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 24
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 25
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 26
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 27
                         ]

            #     0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27
            actions = np.asarray(
                [[0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
                 [4, 0, 6, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
                 [0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
                 [0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
                 [0, 0, 0, 0, 4, 0, 6, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5
                 [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
                 [0, 8, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
                 [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 9
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 6, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 11
                 [0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],  # 12
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 13
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 14
                 [0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 4, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 15
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 16
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],  # 17
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0],  # 18
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 4, 0, 6, 0, 0, 0, 0, 0, 0, 0],  # 19
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0],  # 20
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 21
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 22
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 23
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 24
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 25
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 26
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 27
                 ])
            actions_dict = {'2': 'south',
                            '4': 'west',
                            '6': 'east',
                            '8': 'north'}
        elif graph == 'hanoi':
            adj_dense, actions, actions_dict = self.get_adj_hanoi(3)
        elif graph == 'binary':
            adj_dense, actions, actions_dict = self.get_adj_binary(6)
        else:
            adj_dense, actions, actions_dict = [], [], {}
            print(f'Graph: {self.graph_stored} not implemented yet.')

        return adj_dense, actions, actions_dict

    def get_adj_hanoi(self, n=3):
        '''
        By Tony Zhang et al, 2021
        Returns adjacency matrix for the Tower of Hanoi graph with n disks
        Modified by Zeyu Jing, including renamings, decimal transformation, and action dictionary.
        '''
        states, adjacency_n_action = self.create_hanoi_graph(n)
        m = len(states)
        adj_dense = np.zeros((m, m))
        actions = np.zeros((m, m))
        actions_dict= {}
        for (i, j), (m,n) in adjacency_n_action:
            adj_dense[i, j] = 1
            actions[i, j] = m*10+n # decimal conversion of the index of action
            actions_dict.update({f'{actions[i, j]}': f'from pole {m} to {n}'})
        return adj_dense, actions, actions_dict

    def create_hanoi_graph(self, n):
        '''
        By Tony Zhang et al, 2021
        Finds all the states and their adjacency matrix for the Tower of Hanoi with n disks.
        Returns:
        s = a list of states, each is a list containing the stack for each of the 3 posts.
        The disks are numbered 0,...,n-1.
        c = a sparse connectivity matrix with corresponding actions taken among those states, given as a list of pairs [i,j]
        for all the connected states, where i and j are indices into s.
        '''
        st = [list(range(n)), [], []]  # initialize starting state
        s = [st]  # initialize list of all states
        ca = []  # initialize set of connections and actions
        k = 0  # index of state currently being considered
        while k < len(s):  # check if there are more states to handle
            st = s[k]  # current state
            for i in range(3):  # try moving a ring from each post i
                if len(st[i]) > 0:  # if there is a ring
                    x = st[i][0]  # top ring on post i
                    for j in range(3):  # consider each of the other posts
                        if j != i:  # other posts j
                            if len(st[j]) == 0 or x < st[j][0]:  # if x is smaller than top ring on j
                                sta = st.copy()  # move the ring
                                sta[i] = st[i][1:]  # from post i
                                sta[j] = [x] + st[j]  # to post j
                                if sta in s:  # have already found this state
                                    ca.append([[k, s.index(sta)], [i, j]])  # add to set of edges
                                else:  # if this is a new state
                                    s += [sta]  # add to list of states
                                    ca.append([[k, len(s) - 1], [i, j]])  # add to set of edges
            k += 1  # done with this state
        return s, ca

    def get_adj_binary(self, n=5):  # n must be even for square maze
        ''' Construct adjacency matrix for a binary maze with n levels. From MM. Modified by ZJ.
        Args:
          n: number of levels
        Returns:
          The adjacency matrix
        '''
        pa = []  # records the parent for each node
        for i in range(n + 1):  # i is the level of the binary tree, with the root branch point at i=0
            for j in range(2 ** i):  # j is the index of the branch points within a level
                if i == 0:  # this node is the first branch point
                    pa.append(-1)  # this node has no parent
                else:
                    k = 2 ** (i - 1) - 1 + j // 2  # parent node is one level up in the tree
                    pa.append(k)
        # print(f'parents: {pa}')
        m = len(pa)  # number of nodes
        adj_dense = np.zeros((m, m))  # adjacency matrix
        actions = np.zeros((m, m))
        for i in range(1, m):  # connect every node to its parent
            adj_dense[i, pa[i]] = 1
            actions[i, pa[i]] = 2  # move up to the parent

            adj_dense[pa[i], i] = 1
            if i % 2 == 1:
                actions[pa[i], i] = 1  # move down to the left child
            else:
                actions[pa[i], i] = 3  # move down to the right child
        actions_dict = {'1': 'move down to left child',
                        '2': 'move up to parent',
                        '3': 'move down to right child'}
        return adj_dense, actions, actions_dict