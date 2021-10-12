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


def create_dir(file_path):
    if not isinstance(file_path, list):
        raise Exception('file path is not a list: {}'.format(file_path))

    for i in range(1, len(file_path)):
        path = os.path.join(*file_path[:i])
        if not os.path.exists(path):
            os.makedirs(path)


def mean_std_from_array(arr, labels):
    df = pd.DataFrame()
    df['mean'] = np.mean(arr, axis=1)
    df['std'] = np.std(arr, axis=1)
    df.index = labels
    return df


def save_df(df, file_path=['results', 'res'], use_date=False):
    create_dir(file_path)
    file_path = get_new_file_path(file_path, '.csv', use_date)
    df.to_csv(os.path.join(*file_path))


def save_vars(vars, file_path=['results', 'res'], use_date=False):
    create_dir(file_path)
    file_path = get_new_file_path(file_path, '.z', use_date)
    joblib.dump(vars, os.path.join(*file_path))


def get_new_file_path(file_path, extension, use_date):
    ex = len(extension)
    new_file_path = copy.copy(file_path)
    if use_date:
        new_file_path[-1] = new_file_path[-1] + '_' + datetime.now().strftime("%d_%m_%Y %H-%M") + extension
    else:
        new_file_path[-1] = new_file_path[-1] + extension
        if os.path.exists(os.path.join(*new_file_path)):
            counter = 1
            new_file_path[-1] = '{}_1{}'.format(new_file_path[-1][:-ex], extension)
            while True:
                new_file_path[-1] = '{}{}{}'.format(new_file_path[-1][:-(ex + 1)],
                                                    str(counter),
                                                    extension)
                if not os.path.exists(os.path.join(*new_file_path)):
                    return new_file_path
                else:
                    counter += 1
        else:
            return new_file_path
    return new_file_path


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
        algos, problem, k, prob_cfg, prob_cfg, algos_hv_hist_runs = result
        return {'algos': algos, 'problem': problem, 'k': k,
                'prob_cfg': prob_cfg, 'algos_hv_hist_runs': algos_hv_hist_runs}


def latex_table(title, tabbular_text):
    table_str = '\\begin{table} \n\\begin{center}\n'
    table_str += '\\caption{{{0}}}\\label{{tbl:{1}}}\n'.format(title.upper().replace('_', ' '),
                                                               title.lower().replace(' ', '_'))
    table_str += tabbular_text
    table_str += '\\end{center} \n\\end{table}\n'
    return table_str


def write_text_file(file_path, text, extension='.txt', use_date=False):
    create_dir(file_path)
    path = get_new_file_path(file_path, extension, use_date=use_date)
    with open(os.path.join(*path), "w") as text_file:
        text_file.write(text)
