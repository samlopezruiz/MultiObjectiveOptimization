import os

import numpy as np
import pandas as pd
import seaborn as sns

from src.models.compare.winners import Winners
from src.utils.util import files_with_substring, \
    latex_table, write_text_file, unpack_results, array_from_lists

sns.set_theme('poster')

if __name__ == '__main__':
    # %%
    general_cfg = {'save_stats': False,
                   'use_date': False,
                   'save_latex': False}

    in_folder_cfg = {'results': 'results',
                     'images': 'img',
                     'experiment': 'wfg123_n_eval',
                     'hv_file': 'hv',
                     'res_file': 'res'}

    out_folder_cfg = {'results': 'results',
                      'images': 'img',
                      'experiment': 'winners_n_eval'}

    probs = [('wfg1', 3), ('wfg2', 3), ('wfg3', 3)]

    algos = ['NSGA2', 'NSGA3', 'MOEAD', 'SMSEMOA']
    in_file_path = ['output', in_folder_cfg['experiment'], in_folder_cfg['results']]
    out_file_path = ['output', out_folder_cfg['experiment'], out_folder_cfg['results']]

    # Get mean and std dev results
    std_algos_hv, mean_algos_hv, latex_algos_hv, problem_algos_hv = [], [], [], []
    for problem, k in probs:
        hv_file_ss = '{}_k{}_{}'.format(problem, k, in_folder_cfg['hv_file'])
        hv_files = files_with_substring(in_file_path, hv_file_ss)
        res_file_ss = '{}_k{}_{}'.format(problem, k, in_folder_cfg['res_file'])
        res_files = files_with_substring(in_file_path, res_file_ss)
        if len(hv_files) > 0 and len(res_files) > 0:
            res = unpack_results(in_file_path + [res_files[0]])
            problem_algos_hv.append(array_from_lists([hv_hist[:, -1] for hv_hist in res['algos_hv_hist_runs']]))

            df = pd.read_csv(os.path.join(*(in_file_path + [hv_files[0]])), index_col=0)
            latex_df = pd.DataFrame()
            latex_df['{} k={}'.format(problem, k)] = df['mean'].round(2).astype(str) + \
                                                     ' (' + df['std'].round(2).astype(str) + ')'
            df.columns = ['{} {} k={}'.format(problem, col, k) for col in df.columns]
            mean_algos_hv.append(df.iloc[:, 0])
            std_algos_hv.append(df.iloc[:, 1])
            latex_algos_hv.append(latex_df)

        else:
            raise Exception('No file found with substring {}'.format(hv_file_ss))

    mean_algos_hv = pd.concat(mean_algos_hv, axis=1)
    std_algos_hv = pd.concat(std_algos_hv, axis=1)
    problems_algos_hvs = [algos_hv[np.newaxis, :, :] for algos_hv in problem_algos_hv]
    problems_algos_hvs = np.concatenate(problems_algos_hvs)

    if general_cfg['save_latex']:
        latex_df = pd.concat(latex_algos_hv, axis=1)
        write_text_file(out_file_path + ['hv_latex'], latex_table('Hypervolume Results', latex_df.T.to_latex()))

    # %%
    metric = np.negative(mean_algos_hv.values).T
    winners = Winners(metric, algos)
    scores = winners.score(problems_algos_hvs, alternative='greater')

    if general_cfg['save_latex']:
        output_text = ''
        for key in scores:
            if isinstance(scores[key], pd.DataFrame):
                table_text = latex_table(key, scores[key].to_latex())
                output_text += table_text + '\n\n'

        write_text_file(out_file_path + ['scores_latex'], output_text)
