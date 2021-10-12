from src.optimization.harness.moo import run_moo_problem
from src.utils.util import save_vars

if __name__ == '__main__':
    # %%
    general_cfg = {'save_plots': True,
                   'save_stats': True,
                   'use_date': False}

    folder_cfg = {'results': 'results',
                  'images': 'img',
                  'experiment': 'plot_all'}

    prob_cfg = {'n_variables': 24,
                'hv_ref': [5, 5, 5]}

    algo_cfg = {'termination': ('n_eval', 2000),
                'pop_size': 60,
                'hv_ref': [10, 10, 10],
                'verbose': 2}

    probs = [('wfg4', 5), ('wfg4', 6), ('wfg4', 7), ('wfg4', 8), ('wfg4', 9), ('wfg4', 10)]
    # algos = ['NSGA2', 'NSGA3', 'MOEAD']  # , 'SMSEMOA']
    algos = ['SMSEMOA']

    # %%
    for problem, k in probs:
        print('\nRunning Optimization for problem: {} k={}'.format(problem, k))
        prob_cfg['name'], prob_cfg['n_obj'] = problem, k
        prob_cfg['hv_ref'], algo_cfg['hv_ref'] = [5] * k, [10] * k
        algos_run = []
        for algo in algos:
            algos_run.append(run_moo_problem(algo,
                                             algo_cfg,
                                             prob_cfg,
                                             plot=k == 3,
                                             verbose=algo_cfg['verbose'],
                                             file_path=['output',
                                                        folder_cfg['experiment'],
                                                        folder_cfg['images'],
                                                        '{}_{}_k{}'.format(algo, problem, k)],
                                             save_plots=general_cfg['save_plots'],
                                             use_date=general_cfg['use_date']))

        if general_cfg['save_stats']:
            pops = [run['result']['res'].pop for run in algos_run]
            algo_hvs = [run['hv_pop'] for run in algos_run]
            sv = {'algos': algos, 'problem': problem, 'k': k, 'prob_cfg': prob_cfg, 'pops': pops,
                  'algo_cfg': algo_cfg, 'algo_hvs': algo_hvs}
            save_vars(sv,
                      file_path=['output',
                                 folder_cfg['experiment'],
                                 folder_cfg['results'],
                                 '{}_k{}_res'.format(problem, k)],
                      use_date=general_cfg['use_date'])
