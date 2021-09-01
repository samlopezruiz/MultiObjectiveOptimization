import pandas as pd
from matplotlib import pyplot as plt


def plot_log(log, ylabel='F(x)', title='F(x) vs Generation'):
    f, ax = plt.subplots(figsize=(13, 9))
    log = log.copy()
    log.drop(['worst', 'mean'], axis=1, inplace=True)
    log.plot(ax=ax)
    plt.xlabel('GEN')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()