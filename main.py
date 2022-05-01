from graph import Graph
from model import Model
from endotaxis import Endotaxis

if __name__ == '__main__':
    # Action learning
    params={
        'nG': 2,
        'nP': 28,
        'nA': 4,
        'nS': 60,
        'beta_sg': 0.2,
        'alpha_sg': 1,
        'beta_sp': 0.2,
        'alpha_sp': 1,
        'beta_ss': 0.4, #0.5
        'alpha_ss': 0,
        'beta_ps': 0.8,
        'alpha_ps': 1,
        'beta_as': 0.8,
        'alpha_as': 1,
        'k': 10,
        'theta': 0.4
    }

    if params['nG'] * params['nP'] != params['nS']:
        print('Warning: nG*N not equal to Q, more state cells than needed!')
    graph = Graph('binary 3 level')

    accuracy = []
    goals = [4, 24]
    print_step, num_rep = 1, 10
    for i in range(num_rep):
        model = Model(graph, params)
        model.train(goals, num_runs=1501, display_mat=False, num_plot_steps=500)
        accuracy.append(model.accuracy(goals))
        if i % print_step == 0:
            print(f'Progress: {i+1} out of {num_rep} repeats completed.')
    print(f'Average accuracy of the model (measured by the ratio of correct actions versus all required actions): {sum(accuracy)/len(accuracy):.2f}')
        # model.plot_vec_field(goals)
        # performance = model.performance(goals)
        # print(f'performance: {performance}')

    ##Endotaxis
    # params_endotaxis = {'nG': 3,
    #                     'goals': [3, 8, 11],
    #                     'nP': 12,
    #                     'w': 0.3,
    #                     'beta_mm':0.02,
    #                     'alpha_mm':0.05,
    #                     'beta_gm': 0.3,
    #                     'alpha_gm': 0.025}
    # graph = Graph('mat1')
    # endotaxis_model = Endotaxis(params_endotaxis, graph)
    # endotaxis_model.train()
    # # nav = endotaxis_model.navigatability()
    # endotaxis_model.comp_vect_field()



