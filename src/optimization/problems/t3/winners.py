import os

import numpy as np
import pandas as pd
import seaborn as sns

from src.models.compare.winners import Winners
from src.utils.util import files_with_substring, \
    latex_table, write_text_file

sns.set_theme('poster')

if __name__ == '__main__':
    # %%
    general_cfg = {'save_stats': True,
                   'use_date': False,
                   'save_latex': True}

    in_folder_cfg = {'results': 'results',
                     'images': 'img',
                     'experiment': 'all_n_eval',
                     'file': 'hv'}

    out_folder_cfg = {'results': 'results',
                      'images': 'img',
                      'experiment': 'winners_n_gen'}

    probs = [('wfg1', 3), ('wfg2', 3), ('wfg3', 3), ('wfg4', 5),
             ('wfg4', 6), ('wfg4', 7), ('wfg4', 8), ('wfg4', 9), ('wfg4', 10)]

    algos = ['NSGA2', 'NSGA3', 'MOEAD', 'SMSEMOA']
    in_file_path = ['output', in_folder_cfg['experiment'], in_folder_cfg['results']]
    out_file_path = ['output', out_folder_cfg['experiment'], out_folder_cfg['results']]

    # Get mean and std dev results
    std_algos_hv, mean_algos_hv, latex_algos_hv = [], [], []
    for problem, k in probs:
        file_ss = '{}_k{}_{}'.format(problem, k, in_folder_cfg['file'])
        files = files_with_substring(in_file_path, file_ss)
        if len(files) > 0:
            df = pd.read_csv(os.path.join(*(in_file_path + [files[0]])), index_col=0)
            latex_df = pd.DataFrame()
            latex_df['{} k={}'.format(problem, k)] = df['mean'].round(2).astype(str) + \
                                                     ' (' + df['std'].round(2).astype(str) + ')'
            df.columns = ['{} {} k={}'.format(problem, col, k) for col in df.columns]
            mean_algos_hv.append(df.iloc[:, 0])
            std_algos_hv.append(df.iloc[:, 1])
            latex_algos_hv.append(latex_df)

        else:
            raise Exception('No file found with substring {}'.format(file_ss))

    mean_algos_hv = pd.concat(mean_algos_hv, axis=1)
    std_algos_hv = pd.concat(std_algos_hv, axis=1)

    if general_cfg['save_latex']:
        latex_df = pd.concat(latex_algos_hv, axis=1)
        write_text_file(out_file_path + ['hv_latex'], latex_table('Hypervolume Results', latex_df.T.to_latex()))

    # %%
    metric = np.negative(mean_algos_hv.values).T
    winners = Winners(metric, algos)
    scores = winners.score()

    if general_cfg['save_latex']:
        output_text = ''
        for key in scores:
            if isinstance(scores[key], pd.DataFrame):
                table_text = latex_table(key, scores[key].to_latex())
                output_text += table_text + '\n\n'

        write_text_file(out_file_path + ['scores_latex'], output_text)
