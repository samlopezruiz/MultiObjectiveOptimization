import copy
from datetime import datetime
import os
from os.path import isfile

import joblib
import numpy as np
import pandas as pd


def get_type_str(obj):
    return str(type(obj)).split("'")[1].split('.')[-1]


def array_from_lists(lists):
    max_shape = max([len(a) for a in lists])
    arr = np.zeros((len(lists), max_shape))
    arr.fill(np.nan)
    for i, a in enumerate(lists):
        arr[i, :len(a)] = a
    return arr

def mean_std_from_array(arr, labels):
    df = pd.DataFrame()
    df['mean'] = np.mean(arr, axis=1)
    df['std'] = np.std(arr, axis=1)
    df.index = labels
    return df


def get_new_file_path(file_path, extension, use_date_suffix):
    if not isinstance(file_path, list):
        path = os.path.dirname(file_path)
        filename = file_path.split('\\')[-1]
    else:
        path = os.path.join(*file_path[:-1])
        filename = file_path[-1]

    ex = len(extension)
    if use_date_suffix:
        filename = filename + '_' + datetime.now().strftime("%d_%m_%Y %H-%M") + extension
    else:
        filename = filename + extension
        if os.path.exists(os.path.join(path, filename)):
            counter = 1
            filename = '{}_1{}'.format(filename[:-ex], extension)
            while True:
                filename = '{}{}{}'.format(filename[:-(ex + 1)],
                                           str(counter),
                                           extension)
                if not os.path.exists(os.path.join(path, filename)):
                    return os.path.join(path, filename)
                else:
                    counter += 1
        else:
            return os.path.join(path, filename)
    return os.path.join(path, filename)


def save_df(df, file_path, use_date_suffix=False):
    create_dir(file_path)
    path = os.path.join(get_new_file_path(file_path, '.csv', use_date_suffix))
    print('Saving Dataframe to: \n{}'.format(path))
    df.to_csv(path)


def save_vars(vars, file_path, extension='.z', use_date_suffix=False):
    create_dir(file_path)
    path = get_new_file_path(file_path, extension, use_date_suffix)
    print('Saving Vars to: \n{}'.format(path))
    joblib.dump(vars, path)


def create_dir(file_path, filename_included=True):
    if not isinstance(file_path, list):
        path = os.path.dirname(file_path) if filename_included else file_path
        if not os.path.exists(path):
            os.makedirs(path)
    else:
        for i in range(1, len(file_path)):
            path = os.path.join(*file_path[:i])
            if not os.path.exists(path):
                os.makedirs(path)


def reshape_01axis(arr_in):
    s = arr_in.shape
    arr_out = np.zeros((s[1], s[0], s[2]))
    for p in range(s[0]):
        for a in range(s[1]):
            arr_out[a, p, :] = arr_in[p, a, :]

    return arr_out


def files_with_substring(file_path, substring):
    path = os.path.join(*file_path)
    files = [f for f in os.listdir(path) if (isfile(os.path.join(path, f)) and substring in f)]
    return files


def unpack_results(file_path):
    print('Loading file: {}'.format(file_path[-1]))
    result = joblib.load(os.path.join(*file_path))
    if isinstance(result, dict):
        return result
    else:
        return result
        # algos, problem, k, prob_cfg, prob_cfg, algos_hv_hist_runs = result
        # return {'algos': algos, 'problem': problem, 'k': k,
        #         'prob_cfg': prob_cfg, 'algos_hv_hist_runs': algos_hv_hist_runs}


def latex_table(title, tabbular_text):
    table_str = '\\begin{table}[h] \n\\begin{center}\n'
    table_str += '\\caption{{{0}}}\\label{{tbl:{1}}}\n'.format(title.replace('_', ' '),
                                                               title.lower().replace(' ', '_'))
    table_str += tabbular_text
    table_str += '\\end{center} \n\\end{table}\n'
    return table_str


def write_text_file(file_path, text, extension='.txt', use_date_suffix=False):
    create_dir(file_path)
    path = get_new_file_path(file_path, extension, use_date_suffix)
    with open(os.path.join(path), "w") as text_file:
        text_file.write(text)


def write_latex_from_scores(scores, out_file_path, file_name='scores_latex'):
    output_text = ''
    for key in scores:
        if isinstance(scores[key], pd.DataFrame):
            table_text = latex_table(key, scores[key].to_latex())
            output_text += table_text + '\n\n'

    write_text_file(out_file_path + [file_name], output_text)