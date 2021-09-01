import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from src.models.ga.codification.binary import get_ga_binary_problem
from src.optimization.harness.gridsearch import gs_search
from src.optimization.harness.plot import plot_gs_stats

sns.set_theme()
sns.set_context("poster")  # 'talk'

if __name__ == '__main__':
    model_cfg = {'n_gens': 50, 'n_pop': 300, 'cx_pb': 0.8, 'elitism_p': 0.02, 'mx_pb': 0.01,
                 'selec': 'roullete'}
    name = 'himmelblau'
    n_repeat = 5
    verbose = 1
    get_ga = get_ga_binary_problem(name)
    gs_cfg = {'n_pop': list(range(50, 200, 50)), 'n_gens': list(range(10, 100, 20))}

    gs_logs, stats = gs_search(name, gs_cfg, model_cfg, get_ga, n_repeat=n_repeat, verbose=verbose)
    plot_gs_stats(stats, gs_cfg, rmargin=0.75, figsize=(25, 9))
