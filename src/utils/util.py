import copy
from datetime import datetime
import os

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
    if use_date:
        file_path[-1] = file_path[-1] + '_' + datetime.now().strftime("%d_%m_%Y %H-%M") + extension
    else:
        file_path[-1] = file_path[-1] + extension
        if os.path.exists(os.path.join(*file_path)):
            test_file_path = copy.copy(file_path)
            counter = 1
            test_file_path[-1] = '{}_1{}'.format(test_file_path[-1][:-ex], extension)
            while True:
                test_file_path[-1] = '{}{}{}'.format(test_file_path[-1][:-(ex + 1)],
                                                     str(counter),
                                                     extension)
                if not os.path.exists(os.path.join(*test_file_path)):
                    return test_file_path
                else:
                    counter += 1
        else:
            return file_path
    return file_path
