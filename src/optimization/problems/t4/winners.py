import numpy as np
import pandas as pd
import seaborn as sns

from src.models.compare.winners import Winners
from src.utils.util import array_from_lists, files_with_substring, \
    unpack_results, write_text_file, latex_table, write_latex_from_scores

sns.set_theme('poster')

if __name__ == '__main__':
    # %%
    general_cfg = {'save_plots': False,
                   'save_stats': False,
                   'use_date': False,
                   'save_latex': False}

    in_folder_cfg = {'results': 'results',
                     'images': 'img',
                     'experiment': 'repeat',
                     'file': 'res'}

    out_folder_cfg = {'results': 'results',
                      'images': 'img',
                      'experiment': 'winners'}

    probs = ['dtlz2', 'wfg1', 'wfg2', 'wfg3']

    algos = ['NSGA2', 'NSGA3', 'MOEAD', 'SMSEMOA']
    in_file_path = ['output', in_folder_cfg['experiment'], in_folder_cfg['results']]
    out_file_path = ['output', out_folder_cfg['experiment'], out_folder_cfg['results']]

    latex_df={}
    problem_algos_latex = []
    problems_algos_hvs, problems_algos_ts, problems_algos_ops = [], [], []
    for problem in probs:
        problem_algos_hv, problem_algos_t, problem_algos_op = [], [], []
        file_ss = '{}'.format(problem)
        files = files_with_substring(in_file_path, file_ss)
        labels = [f.replace('_res.z', '') for f in files]
        # labels = ['pNSGA-III', 'pWoArNSGA-III', 'NSGA-III']
        if len(files) == 0:
            raise Exception('No file found with substring {}'.format(file_ss))

        for file in files:
            runs = unpack_results(in_file_path + [file])
            problem_algos_hv.append([run['hv'] for run in runs])
            problem_algos_t.append([run['time'] for run in runs])
            problem_algos_op.append([run['op'] for run in runs])

        algos_hvs = array_from_lists(problem_algos_hv)
        algos_ops = array_from_lists(problem_algos_op)
        algos_ts = array_from_lists(problem_algos_t)

        latex_df = pd.DataFrame()
        latex_df['Hypervolume'] = pd.Series(np.mean(algos_hvs, axis=1)).round(2).astype(str) + ' (' + \
                                  pd.Series(np.std(algos_hvs, axis=1)).round(2).astype(str) + ')'
        latex_df['Optimum %'] = pd.Series(np.mean(algos_ops, axis=1)).round(2).astype(str) + ' (' + \
                                  pd.Series(np.std(algos_ops, axis=1)).round(2).astype(str) + ')'
        latex_df['Cpu time (s)'] = pd.Series(np.mean(algos_ts, axis=1)).round(2).astype(str) + ' (' + \
                                  pd.Series(np.std(algos_ts, axis=1)).round(2).astype(str) + ')'
        latex_df.index = labels
        problems_algos_hvs.append(algos_hvs)
        problems_algos_ts.append(algos_ts)
        problems_algos_ops.append(algos_ops)

        if general_cfg['save_latex']:
            write_text_file(out_file_path + [problem+'_latex'],
                            latex_table(problem + ' Results', latex_df.T.to_latex()))


    if len(probs) == 1:
        problems_algos_hvs = problems_algos_hvs[0][np.newaxis, :, :]
        problems_algos_ts = problems_algos_ts[0][np.newaxis, :, :]
        problems_algos_ops = problems_algos_ops[0][np.newaxis, :, :]
    else:
        problems_algos_hvs = np.stack(problems_algos_hvs) # np.concatenate(
        problems_algos_ts = np.stack(problems_algos_ts)
        problems_algos_ops = np.stack(problems_algos_ops)

    #%%
    metric = np.negative(np.mean(problems_algos_hvs, axis=2))
    winners = Winners(metric, labels)
    scores = winners.score(problems_algos_hvs, alternative='greater')

    if general_cfg['save_latex']:
        write_latex_from_scores(scores, out_file_path, file_name='hv_scores_latex')


    metric = np.negative(np.mean(problems_algos_ops, axis=2))
    winners = Winners(metric, labels)
    scores = winners.score(problems_algos_ops, alternative='greater')

    if general_cfg['save_latex']:
        write_latex_from_scores(scores, out_file_path, file_name='op_scores_latex')

    metric = np.mean(problems_algos_ts, axis=2)
    winners = Winners(metric, labels)
    scores = winners.score(problems_algos_ts, alternative='less')

    if general_cfg['save_latex']:
        write_latex_from_scores(scores, out_file_path, file_name='t_scores_latex')



