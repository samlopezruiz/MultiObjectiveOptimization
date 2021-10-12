import numpy as np
import pandas as pd
from scipy.stats import ranksums


def wilcoxon_rank(arr, labels):
    '''
    :param labels: list of test labels
    :param arr: rows are tests and columns are samples
    :return: matrix with wilcoxon p values
    '''
    n_tests = arr.shape[0]
    wilcoxon_rank = np.zeros((n_tests, n_tests))
    wilcoxon_rank.fill(np.nan)
    for i in range(n_tests):
        for j in range(n_tests):
            x, y = arr[i, :], arr[j, :]
            wilcoxon_rank[i, j] = ranksums(x, y).pvalue
    WR = pd.DataFrame(wilcoxon_rank, index=labels, columns=labels)
    return WR