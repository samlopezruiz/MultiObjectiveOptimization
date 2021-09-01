import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os
from datetime import datetime


def plot_runs(logs, var='best', title='', save=False, save_folder='images'):
    fig, ax = plt.subplots(figsize=(15, 9))
    max_n_gens = 0
    all_runs = []
    for log in logs:
        run = log[2]
        max_n_gens = max(max_n_gens, run.shape[0])
        sns.lineplot(x=run.index, y=var, data=run, ax=ax, color='gray', alpha=0.5)
        sns.scatterplot(x=run.index, y=var, data=run, ax=ax, color='black', alpha=0.3)
        all_runs.append(run[var])

    padded_runs = []
    for run in all_runs:
        padded_runs.append(np.pad(run, (0, max_n_gens - len(run)), 'edge'))
    runs = np.array(padded_runs)
    mean_run = np.mean(runs, axis=0)
    sns.lineplot(x=list(range(max_n_gens)), y=mean_run, ax=ax, color='red')
    plt.xlim([0, max_n_gens])
    ax.set_title('{}: F(x) vs. Generations @ {} runs'.format(title, len(logs)))
    plt.show()

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if save:
        fig.savefig(os.path.join(save_folder, title + '_' + datetime.now().strftime("%d_%m_%Y %H-%M") + '.png'))


def plot_runs(logs, var='best', title='', save=False, save_folder='images'):
    fig, ax = plt.subplots(figsize=(15, 9))
    max_n_gens = 0
    all_runs = []
    for log in logs:
        run = log['log']
        max_n_gens = max(max_n_gens, run.shape[0])
        sns.lineplot(x=run.index, y=var, data=run, ax=ax, color='gray', alpha=0.5)
        sns.scatterplot(x=run.index, y=var, data=run, ax=ax, color='black', alpha=0.3)
        all_runs.append(run[var])

    padded_runs = []
    for run in all_runs:
        padded_runs.append(np.pad(run, (0, max_n_gens - len(run)), 'edge'))
    runs = np.array(padded_runs)
    mean_run = np.mean(runs, axis=0)
    sns.lineplot(x=list(range(max_n_gens)), y=mean_run, ax=ax, color='blue')
    plt.xlim([0, max_n_gens])
    ax.set_title('{}: F(x) vs. Generations @ {} runs'.format(title, len(logs)))
    plt.show()

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if save:
        fig.savefig(os.path.join(save_folder, title + '_' + datetime.now().strftime("%d_%m_%Y %H-%M") + '.png'))


def plot_gp_runs(res, title, metric='min'):
    fig, ax1 = plt.subplots(figsize=(15, 9))
    ax2 = ax1.twinx()
    fits, sizes = [], []
    for run in res:
        log = run[2]
        gen = log.select("gen")
        fit_mins = log.chapters["fitness"].select(metric)
        size_avgs = log.chapters["size"].select("avg")
        sns.lineplot(x=gen, y=fit_mins, ax=ax1, color='blue', alpha=0.3)
        sns.lineplot(x=gen, y=size_avgs, ax=ax2, color='red', alpha=0.3)
        fits.append(fit_mins)
        sizes.append(size_avgs)

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Best Individual Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")
    ax2.set_ylabel("Average Individual Size", color="r")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    mean_fits = np.mean(np.array(fits), axis=0)
    mean_sizes = np.mean(np.array(sizes), axis=0)
    sns.lineplot(x=gen, y=mean_fits, ax=ax1, color='blue', linewidth=4)
    sns.lineplot(x=gen, y=mean_sizes, ax=ax2, color='red', linewidth=4)
    ax1.grid(False)
    ax2.grid(False)
    ax1.set_title('{}: F(x) vs. Generations @ {} runs'.format(title, len(res)))

    plt.show()


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
