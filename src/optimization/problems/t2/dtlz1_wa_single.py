import numpy as np

from src.models.moo.utils.deap.weight_avg.harness import run_wa
from src.optimization.functions.mop import dtlz1

if __name__ == '__main__':
    # %%
    problem_name = 'DTLZ1'
    problem = dtlz1
    weights = np.array([.6, .2, .2])
    prob_cfg = {'n_obj': 3, 'n_variables': 12, 'bounds_low': 0, 'bounds_up': 1}
    algo_cfg = {'algorithm': 'WEIGHTED AVERAGE', 'max_gen': 100, 'pop_size': 100, 'cross_prob': 0.8, 'weights': weights}

    result = run_wa(problem, prob_cfg, algo_cfg, weights, step_size=algo_cfg['max_gen'] // 20,
                    plot_=True, verbose=True, save_plots=False, problem_name=problem_name)

    print('Best Individual: {}({}) = {} ~ {}'.format(problem_name,
                                                list(np.round(result['best_ind'], 3)),
                                                list(np.round(problem(result['best_ind']), 3)),
                                                round(result['best_fitness'], 4)))
