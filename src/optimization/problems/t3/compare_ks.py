import os
from os.path import isfile

import joblib
import numpy as np
import seaborn as sns
from src.optimization.harness.plot import plot_histogram
from src.utils.util import array_from_lists, reshape_01axis, mean_std_from_array, save_df, files_with_substring, \
    unpack_results, write_text_file, latex_table

sns.set_theme('poster')

if __name__ == '__main__':
    # %%
    general_cfg = {'save_plots': True,
                   'save_stats': True,
                   'use_date': False,
                   'save_latex': True}

    in_folder_cfg = {'results': 'results',
                     'images': 'img',
                     'experiment': 'all_n_gen',
                     'file': 'res'}

    out_folder_cfg = {'results': 'results',
                      'images': 'img',
                      'experiment': 'k_compare'}

    probs = [('wfg4', 5), ('wfg4', 6), ('wfg4', 7), ('wfg4', 8), ('wfg4', 9), ('wfg4', 10)]

    algos = ['NSGA2', 'NSGA3', 'MOEAD', 'SMSEMOA']
    in_file_path = ['output', in_folder_cfg['experiment'], in_folder_cfg['results']]
    out_file_path = ['output', out_folder_cfg['experiment'], out_folder_cfg['results']]

    problem_algos_hv = []
    for problem, k in probs:
        file_ss = '{}_k{}_{}'.format(problem, k, in_folder_cfg['file'])
        files = files_with_substring(in_file_path, file_ss)
        if len(files) > 0:
            res = unpack_results(in_file_path + [files[0]])
            problem_algos_hv.append(array_from_lists([hv_hist[:, -1] for hv_hist in res['algos_hv_hist_runs']]))
        else:
            raise Exception('No file found with substring {}'.format(file_ss))

    problems_algos_hvs = [algos_hv[np.newaxis, :, :] for algos_hv in problem_algos_hv]
    problems_algos_hvs = np.concatenate(problems_algos_hvs)
    algos_problems_hvs = reshape_01axis(problems_algos_hvs)

    # %%
    probs_labels = ['{}_k={}'.format(p, k) for p, k in probs]
    for a, algo_problem_hvs in enumerate(algos_problems_hvs):
        algo_problem_stats = mean_std_from_array(algo_problem_hvs, labels=probs_labels)
        if general_cfg['save_stats']:
            save_df(algo_problem_stats,
                    file_path=out_file_path+['{}_compK'.format(algos[a])],
                    use_date=general_cfg['use_date'])

        if general_cfg['save_latex']:
            write_text_file(out_file_path + ['{}_compK_latex'.format(algos[a])],
                            latex_table('{} Hypervolume Results'.format(algos[a]), algo_problem_stats.to_latex()))

        plot_histogram(algo_problem_hvs,
                       probs_labels,
                       x_label='Problem',
                       y_label='Hypervolume',
                       title='{} hypervolume comparison'.format(algos[a]),
                       size=(15, 9),
                       file_path=['output',
                                  out_folder_cfg['experiment'],
                                  out_folder_cfg['images'],
                                  '{}_compK'.format(algos[a])],
                       save=general_cfg['save_plots'],
                       show_grid=False,
                       use_date=general_cfg['use_date'])
