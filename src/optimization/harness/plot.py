import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os
from datetime import datetime

from src.utils.util import create_dir, get_new_file_path


def plot_runs(y_runs,
              mean_run=None,
              x_label=None,
              y_label=None,
              title=None,
              size=(15, 9),
              file_path=None,
              save=False,
              legend_labels=None,
              show_grid=True,
              use_date=False):
    x = np.arange(y_runs.shape[1]).astype(int) + 1

    fig, ax = plt.subplots(figsize=size)
    for y in y_runs:
        sns.lineplot(x=x, y=y, ax=ax, color='gray' if mean_run is not None else None, alpha=0.8, linewidth=4)
        # sns.scatterplot(x=x, y=y, ax=ax, color='black', alpha=0.3)
    if mean_run is not None:
        sns.lineplot(x=x, y=mean_run, ax=ax, color='blue', linewidth=6)

    plt.xlim([0, y_runs.shape[1]])
    ax.set(xlabel='x' if x_label is None else x_label,
           ylabel='x' if y_label is None else y_label)
    ax.set_title('' if title is None else title)

    if legend_labels is not None:
        plt.legend(labels=legend_labels)
    if show_grid:
        plt.grid()
    plt.show()

    if save:
        save_fig(fig, file_path, use_date)


def plot_histogram(Y,
                   x_labels,
                   x_label=None,
                   y_label=None,
                   title=None,
                   size=(15, 9),
                   file_path=None,
                   save=False,
                   show_grid=True,
                   use_date=False):
    Y = pd.DataFrame(Y.T, columns=x_labels)
    df = Y.melt()
    df.columns = ['variable' if x_label is None else x_label,
                  'value' if y_label is None else y_label]
    fig, ax = plt.subplots(figsize=size)
    if show_grid:
        plt.grid()
    # for i, y in enumerate(Y):
    sns.barplot(data=df, x=x_label, y=y_label, capsize=.2)
    ax.set_title('' if title is None else title)
    plt.show()

    if save:
        save_fig(fig, file_path, use_date)


def plot_gs_stats(stats, gs_cfg=None, title=None, rmargin=0.8, figsize=(20, 9), save=False, save_folder='images'):
    df = pd.DataFrame(stats, columns=['cfg', 'gen', 'eval'])
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    sns.barplot(data=df, y='gen', x='cfg', capsize=.2, ax=axes[0]).set(xticklabels=[])
    sns.barplot(data=df, y='eval', x='cfg', capsize=.2, ax=axes[1], hue='cfg', dodge=False,
                ).set(xticklabels=[])
    plt.subplots_adjust(right=rmargin)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if gs_cfg is not None:
        fig.suptitle('Grid Search for: ' + str(list(gs_cfg.keys())))
    if title is not None:
        fig.suptitle(title)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if save:
        fig.savefig(os.path.join(save_folder, title + '_' + datetime.now().strftime("%d_%m_%Y %H-%M") + '.png'))
    plt.show()


def plot_gs_gp_stats(stats, gs_cfg=None, title=None, rmargin=0.7, figsize=(15, 9), save=False, save_folder='images'):
    df = pd.DataFrame(stats, columns=['cfg', 'eval'])
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=df, y='eval', x='cfg', capsize=.2, ax=ax, hue='cfg', dodge=False,
                ).set(xticklabels=[])
    plt.subplots_adjust(right=rmargin)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if gs_cfg is not None:
        fig.suptitle('Grid Search for: ' + str(list(gs_cfg.keys())))
    if title is not None:
        fig.suptitle(title)
    plt.show()

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if save:
        fig.savefig(os.path.join(save_folder, title + '_' + datetime.now().strftime("%d_%m_%Y %H-%M") + '.png'))


def save_fig(fig, file_path, use_date):
    file_path = ['img', 'plot'] if file_path is None else file_path
    create_dir(file_path)
    file_path = get_new_file_path(file_path, '.png', use_date)
    fig.savefig(os.path.join(*file_path))
