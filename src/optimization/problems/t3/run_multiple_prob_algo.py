import time

import seaborn as sns

from src.optimization.harness.moo import run_multiple_problems

sns.set_theme('poster')

if __name__ == '__main__':
    '''
    - is_reproductible: sets a consecutive seed for each of the executions.
    - use_date saves the files using a datetime suffix
    - probs: specifies the problems to solve with their respective k number of objectives
    - termination can be specified as ('n_gen', #) or as ('n_eval', #) 
    '''

    general_cfg = {'is_reproductible': True,
                   'n_repeat': 20,
                   'plot_ind_algorithms': True,
                   'save_ind_algorithms': True,
                   'save_comparison_plot': True,
                   'save_stat_plots': True,
                   'save_stats': True,
                   'use_date': False}

    folder_cfg = {'results': 'results',
                  'images': 'img',
                  'experiment': 'woSmsEmoa_n_gen'}

    # problem name and number of objectives
    probs = [('wfg1', 3), ('wfg2', 3), ('wfg3', 3), ('wfg4', 5),
             ('wfg4', 6), ('wfg4', 7), ('wfg4', 8), ('wfg4', 9), ('wfg4', 10)]
    algos = ['NSGA2', 'NSGA3', 'MOEAD'] #, 'SMSEMOA']

    prob_cfg = {'n_variables': 24}

    algo_cfg = {'termination': ('n_gen', 200), 'pop_size': 60}
    params = {'algo_cfg': algo_cfg, 'prob_cfg': prob_cfg, 'verbose': 0}
    t0 = time.time()
    run_multiple_problems(probs, algos, general_cfg, params, algo_cfg, prob_cfg, folder_cfg, parallel=True)
    print('Total time: {} min'.format(round((time.time() - t0) / 60), 1))
