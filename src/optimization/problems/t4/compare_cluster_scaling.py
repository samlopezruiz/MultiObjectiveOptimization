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
                   'save_stats': False,
                   'use_date': False,
                   'save_latex': False}

    in_folder_cfg = {'results': 'results',
                     'images': 'img',
                     'experiment': 'clusters',
                     'file': 'res'}

    out_folder_cfg = {'results': 'results',
                      'images': 'img',
                      'experiment': 'cluster_compare'}

    probs = [('wfg2', 3), ('wfg2', 5), ('wfg2', 7), ('wfg2', 9), ('wfg2', 11), ('wfg2', 13), ('wfg2', 15)]

    algos = ['pNSGA3']
    in_file_path = ['output', in_folder_cfg['experiment'], in_folder_cfg['results']]
    out_file_path = ['output', out_folder_cfg['experiment'], out_folder_cfg['results']]

    vars = ['op', 'hv', 'time']
    problem_algos_vars = {}
    for var in vars:
        problem_algos_vars[var] = []

    for problem, k in probs:
        file_ss = '{}_eps2_m{}'.format(problem, k, in_folder_cfg['file'])
        files = files_with_substring(in_file_path, file_ss)
        if len(files) > 0:
            runs = unpack_results(in_file_path + [files[0]])
            for var in vars:
                problem_algos_vars[var].append([run[var] for run in runs])

        else:
            raise Exception('No file found with substring {}'.format(file_ss))

    algos_problems_vars = {}
    for var in vars:
        problem_algos_vars[var] = array_from_lists(problem_algos_vars[var])

    # algos_problems_vars[var] = reshape_01axis(problem_algos_vars[var])

    # problems_algos_hvs = [algos_hv[np.newaxis, :, :] for algos_hv in problem_algos_hv]
    # problems_algos_hvs = np.concatenate(problems_algos_hvs)


    # %%
    probs_labels = ['{}_m={}'.format(p, k) for p, k in probs]

    algo_problem_stats = {}

    for var in vars:
        algo_problem_stats[var] = mean_std_from_array(problem_algos_vars[var], labels=probs_labels)
        if general_cfg['save_stats']:
            save_df(algo_problem_stats[var],
                    file_path=out_file_path+['{}_compClusters'.format(var)],
                    use_date_suffix=general_cfg['use_date'])

        if general_cfg['save_latex']:
            write_text_file(out_file_path + ['{}_compClusters_latex'.format(var)],
                            latex_table('{} Results'.format(var), algo_problem_stats[var].to_latex()))

        plot_histogram(problem_algos_vars[var],
                       probs_labels,
                       x_label='Problem',
                       y_label=var,
                       title='{} comparison'.format(var),
                       size=(15, 9),
                       file_path=['output',
                                  out_folder_cfg['experiment'],
                                  out_folder_cfg['images'],
                                  '{}_compClusters'.format(var)],
                       save=general_cfg['save_plots'],
                       show_grid=False,
                       use_date=general_cfg['use_date'])
