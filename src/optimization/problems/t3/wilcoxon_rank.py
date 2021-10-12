import os
from os.path import isfile

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from src.optimization.harness.plot import plot_histogram
from src.utils.stats import wilcoxon_rank
from src.utils.util import array_from_lists, reshape_01axis, mean_std_from_array, save_df, files_with_substring, \
    unpack_results, latex_table, write_text_file

sns.set_theme('poster')

if __name__ == '__main__':
    # %%
    general_cfg = {'save_stats': True,
                   'use_date': False,
                   'save_latex': True}

    in_folder_cfg = {'results': 'results',
                     'images': 'img',
                     'experiment': 'wfg123_n_eval',
                     'file': 'res'}

    out_folder_cfg = {'results': 'results',
                      'images': 'img',
                      'experiment': 'wilcoxon_rank_n_eval'}

    probs = [('wfg1', 3), ('wfg2', 3), ('wfg3', 3)]

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

    for p, problem_algos_hvs in enumerate(problems_algos_hvs):
        wr = wilcoxon_rank(problem_algos_hvs, algos)
        if general_cfg['save_stats']:
            save_df(wr,
                    file_path=out_file_path+['{}_k{}_wilcoxon_ranks'.format(probs[p][0], probs[p][1])],
                    use_date=general_cfg['use_date'])

        if general_cfg['save_latex']:
            write_text_file(out_file_path + ['{}_k{}_wilcoxon_ranks_latex'.format(probs[p][0], probs[p][1])],
                            latex_table('{}_k={} Wilcoxon Ranks'.format(probs[p][0], probs[p][1]), wr.round(4).to_latex()))
