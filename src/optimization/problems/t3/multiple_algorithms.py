import numpy as np
import pandas as pd
import seaborn as sns

from src.models.moo.utils.indicators import hv_hist_from_runs
from src.optimization.harness.moo import repeat_moo
from src.optimization.harness.plot import plot_runs, plot_histogram
from src.utils.util import array_from_lists, mean_std_from_array

sns.set_context('poster')

if __name__ == '__main__':
    # %%
    general_cfg = {'is_reproductible': True,
                   'n_repeat': 5,
                   'plot_ind_algorithms': True,
                   'save_ind_algorithms': False,
                   'save_comparison_plot': False,
                   'save_stat_plots': False}

    algos = ['NSGA2', 'NSGA3', 'MOEAD', 'SMSEMOA']

    prob_cfg = {'name': 'wfg2',
                'n_obj': 3,
                'n_variables': 24,
                'hv_ref': [5, 5, 5]}

    algo_cfg = {'termination': ('n_gens', 100),
                'pop_size': 100,
                'hv_ref': [10, 10, 10]}

    params = {'algo_cfg': algo_cfg,
              'prob_cfg': prob_cfg}

    algos_runs, algos_hv_hist_runs = [], []
    for algo in algos:
        params['name'] = algo
        runs = repeat_moo(params, general_cfg['n_repeat'], general_cfg['is_reproductible'])
        algos_runs.append(runs)
        algos_hv_hist_runs.append(hv_hist_from_runs(runs, ref=prob_cfg['hv_ref']))

    if general_cfg['plot_ind_algorithms']:
        for i, hv_hist_runs in enumerate(algos_hv_hist_runs):
            plot_runs(hv_hist_runs, np.mean(hv_hist_runs, axis=0),
                      x_label='Generations',
                      y_label='Hypervolume',
                      title='{} convergence plot with {}. Ref={}'.format(prob_cfg['name'],
                                                                         algos[i],
                                                                         str(prob_cfg['hv_ref'])),
                      size=(15, 9),
                      file_path=['img', algos[i] + '_' + prob_cfg['name']],
                      save=general_cfg['save_ind_algorithms'])

    algos_mean_hv_hist = array_from_lists([np.mean(hv_hist_runs, axis=0)
                                           for hv_hist_runs in algos_hv_hist_runs])
    plot_runs(algos_mean_hv_hist,
              x_label='Generations',
              y_label='Hypervolume',
              title='{} convergence plot. Ref={}'.format(prob_cfg['name'], str(prob_cfg['hv_ref'])),
              size=(15, 9),
              file_path=['img', 'Compare_' + prob_cfg['name']],
              save=general_cfg['save_comparison_plot'],
              legend_labels=algos)

    # %%
    hvs_algos = array_from_lists([hv_hist_runs[:, -1] for hv_hist_runs in algos_hv_hist_runs])
    hvs_stats = mean_std_from_array(hvs_algos, labels=algos)
    exec_times = array_from_lists([[algo_run['result']['opt_time'] for algo_run in algo_runs]
                                   for algo_runs in algos_runs])
    exec_times_stats = mean_std_from_array(exec_times, labels=algos)

    plot_histogram(hvs_algos,
                   algos,
                   x_label='Algorithm',
                   y_label='Hypervolume',
                   title='Hypervolume',
                   size=(15, 9),
                   file_path=['img', 'HVs_' + prob_cfg['name']],
                   save=general_cfg['save_stat_plots'],
                   show_grid=False)

    plot_histogram(exec_times,
                   algos,
                   x_label='Algorithm',
                   y_label='Seconds',
                   title='Execution Times',
                   size=(15, 9),
                   file_path=['img', 'Exec_Times_' + prob_cfg['name']],
                   save=general_cfg['save_stat_plots'],
                   show_grid=False)
