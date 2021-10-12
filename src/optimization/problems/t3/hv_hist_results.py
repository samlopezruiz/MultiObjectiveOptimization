import os
from os.path import isfile

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from src.optimization.harness.plot import plot_histogram
from src.utils.util import array_from_lists, reshape_01axis, mean_std_from_array, save_df, files_with_substring, \
    unpack_results, write_text_file, latex_table, create_dir, get_new_file_path

sns.set_theme('poster')

if __name__ == '__main__':
    # %%
    general_cfg = {'save_plots': True,
                   'save_stats': False,
                   'use_date': False,
                   'save_latex': False,
                   'save_excel': True}

    in_folder_cfg = {'results': 'results',
                     'images': 'img',
                     'experiment': 'all_n_gen',
                     'file': 'res'}

    out_folder_cfg = {'results': 'results',
                      'images': 'img',
                      'experiment': 'hv_hist_n_gen'}

    probs = [('wfg1', 3), ('wfg2', 3), ('wfg3', 3), ('wfg4', 5),
             ('wfg4', 6), ('wfg4', 7), ('wfg4', 8), ('wfg4', 9), ('wfg4', 10)]

    algos = ['NSGA2', 'NSGA3', 'MOEAD', 'SMSEMOA']
    in_file_path = ['output', in_folder_cfg['experiment'], in_folder_cfg['results']]
    out_file_path = ['output', out_folder_cfg['experiment'], out_folder_cfg['results']]

    problem_algos_hv = []
    for problem, k in probs:
        file_ss = '{}_k{}_{}'.format(problem, k, in_folder_cfg['file'])
        files = files_with_substring(in_file_path, file_ss)
        if len(files) > 0:
            res = unpack_results(in_file_path + [files[0]])
            if general_cfg['save_excel']:
                file_path = out_file_path + ['Hypervolume_History_{}_k{}'.format(problem, k)]
                create_dir(file_path)
                path = get_new_file_path(file_path, '.xlsx', use_date=False)
                writer = pd.ExcelWriter(os.path.join(*path))
            for i, hv_hist in enumerate(res['algos_hv_hist_runs']):
                df = pd.DataFrame(hv_hist,
                                  columns=['Generation {}'.format(r) for r in range(hv_hist.shape[1])],
                                  index=['Execution {}'.format(r) for r in range(hv_hist.shape[0])])
                if general_cfg['save_excel']:
                    df.T.to_excel(writer, sheet_name=algos[i])
                if general_cfg['save_latex']:
                    write_text_file(out_file_path + ['{}_k{}_{}'.format(problem, k, algos[i])],
                                    latex_table('{} k={} Hypervolume with {}'.format(problem, k, algos[i]),
                                                df.T.iloc[::2, :].round(1).to_latex()))
            if general_cfg['save_excel']:
                writer.save()

        else:
            raise Exception('No file found with substring {}'.format(file_ss))
