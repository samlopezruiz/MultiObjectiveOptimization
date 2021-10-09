import numpy as np
import seaborn as sns

from src.models.moo.utils.indicators import hv_hist_from_runs
from src.models.moo.utils.utils import get_hv_hist_vs_n_evals
from src.optimization.harness.moo import repeat_moo
from src.optimization.harness.plot import plot_runs, plot_histogram
from src.utils.util import array_from_lists, mean_std_from_array, save_df

sns.set_context('poster')

if __name__ == '__main__':
    # %%
    general_cfg = {'is_reproductible': True,
                   'n_repeat': 20,
                   'plot_ind_algorithms': True,
                   'save_ind_algorithms': True,
                   'save_comparison_plot': True,
                   'save_stat_plots': True,
                   'save_stats': True}

    # problem name and number of objectives
    probs = [('wfg1', 3), ('wfg2', 3), ('wfg3', 3)]
    algos = ['NSGA2', 'NSGA3', 'MOEAD', 'SMSEMOA']

    # probs = [('wfg4', 5), ('wfg4', 6), ('wfg4', 7), ('wfg4', 8), ('wfg4', 9), ('wfg4', 10)]
    # algos = ['NSGA2', 'NSGA3', 'MOEAD']

    prob_cfg = {'n_variables': 24}
    algo_cfg = {'termination': ('n_eval', 5000), 'pop_size': 100}
    params = {'algo_cfg': algo_cfg, 'prob_cfg': prob_cfg}

    result = {}
    for prob in probs:
        print('Running Optimization for problem: {}'.format(prob))
        prob_cfg['name'], prob_cfg['n_obj'] = prob
        prob_cfg['hv_ref'] = [5] * prob_cfg['n_obj']
        algo_cfg['hv_ref'] = [10] * prob_cfg['n_obj']

        algos_runs, algos_hv_hist_runs = [], []
        for algo in algos:
            params['name'] = algo
            runs = repeat_moo(params, general_cfg['n_repeat'], general_cfg['is_reproductible'])
            algos_runs.append(runs)
            algos_hv_hist_runs.append(hv_hist_from_runs(runs, ref=prob_cfg['hv_ref']))
        result[prob[0]] = {'algos_runs': algos_runs, 'algos_hv_hist_runs': algos_hv_hist_runs}

    if general_cfg['plot_ind_algorithms']:
        for problem in result:
            for i, hv_hist_runs in enumerate(result[problem]['algos_hv_hist_runs']):
                plot_runs(hv_hist_runs, np.mean(hv_hist_runs, axis=0),
                          x_label='Generations',
                          y_label='Hypervolume',
                          title='{} convergence plot with {}. Ref={}'.format(problem,
                                                                             algos[i],
                                                                             str(prob_cfg['hv_ref'])),
                          size=(15, 9),
                          file_path=['img', algos[i] + '_' + problem],
                          save=general_cfg['save_ind_algorithms'])

    for problem in result:
        if algo_cfg['termination'][0] == 'n_eval':
            algos_mean_hv_hist = get_hv_hist_vs_n_evals(result[problem]['algos_runs'],
                                                        result[problem]['algos_hv_hist_runs'])
        else:
            algos_mean_hv_hist = array_from_lists([np.mean(hv_hist_runs, axis=0)
                                                   for hv_hist_runs in result[problem]['algos_hv_hist_runs']])

        plot_runs(algos_mean_hv_hist,
                  x_label='Function Evaluations' if algo_cfg['termination'][0] == 'n_eval' else 'Generations',
                  y_label='Hypervolume',
                  title='{} convergence plot. Ref={}'.format(problem, str(prob_cfg['hv_ref'])),
                  size=(15, 9),
                  file_path=['img', 'Compare_' + problem],
                  save=general_cfg['save_comparison_plot'],
                  legend_labels=algos)

    # %%
    stats = {}
    for problem, k in probs:
        hvs_algos = array_from_lists([hv_hist_runs[:, -1] for hv_hist_runs in result[problem]['algos_hv_hist_runs']])
        hvs_stats = mean_std_from_array(hvs_algos, labels=algos)
        exec_times = array_from_lists([[algo_run['result']['opt_time'] for algo_run in algo_runs]
                                       for algo_runs in result[problem]['algos_runs']])
        exec_times_stats = mean_std_from_array(exec_times, labels=algos)
        if general_cfg['save_stats']:
            save_df(exec_times_stats, file_path=['results', '{}_k{}_exec_t'.format(problem, k)])
            save_df(hvs_stats, file_path=['results', '{}_k{}_hv'.format(problem, k)])

        plot_histogram(hvs_algos,
                       algos,
                       x_label='Algorithm',
                       y_label='Hypervolume',
                       title=problem + ' hypervolume history',
                       size=(15, 9),
                       file_path=['img', 'HVs_' + problem],
                       save=general_cfg['save_stat_plots'],
                       show_grid=False)

        plot_histogram(exec_times,
                       algos,
                       x_label='Algorithm',
                       y_label='Seconds',
                       title=problem + ' execution times',
                       size=(15, 9),
                       file_path=['img', 'Exec_Times_' + problem],
                       save=general_cfg['save_stat_plots'],
                       show_grid=False)
