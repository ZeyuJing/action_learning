import numpy as np


def gen_sqn(predecessors, goals, num_nodes, num_runs):
    sqn = []
    i = 0
    while i < num_runs:
        goal = goals[np.random.randint(0,len(goals))]
        start = np.random.randint(0,num_nodes-1)
        if start >= goal:
            start += 1
        ls = [goal]
        x = predecessors[start, goal]
        if x < 0:
            continue
        while x >= 0:
            ls.append(x)
            x = predecessors[start,x]
        ls.reverse()
        sqn.append(ls)
        i += 1
    return sqn

def solve_dyn_sys(F, M, x, SS):
    """
    ds/dt = -s + F(nG*x + SS*s)
    Solve a dynamical system to get a self-consistent solutio for the firing rates of state cells
    :param F: non-linear function, sigmoid
    :param M: syn weight matrix, from inputs to state cells
    :param x: inputs
    :param SS: recurrent anti-hebbian inhibitory weights
    :return: non-stepped state cells activity, which has to go through a step function later
    """
    dt = 0.1
    s = np.zeros((1, SS.shape[0])) # all rate array is a 1-by-n array
    for _ in range(400):
        dsdt = -s.T + F(M @ x.T + SS @ s.T)
        s += dt * dsdt.T
    return s, dsdt


