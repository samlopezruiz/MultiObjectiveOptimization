import numpy as np
from src.models.ga.codification.real import real_create_funcs, real_create_sbx_funcs
from src.models.ga.ga import GA
from src.models.ga.plot import plot_log
from src.optimization.functions.funcs import problems
from src.optimization.harness.plot import plot_runs
from src.optimization.harness.repeat import repeat, consolidate_results
from src.optimization.plot.func import plot_contour, plot_contour_path
import seaborn as sns
sns.set_theme()
sns.set_context("poster")

if __name__ == '__main__':
    #%% CONFIGURE GA
    cfg = {'n_gens': 50, 'n_pop': 100, 'cx_pb': 0.8, 'elitism_p': 0, 'mx_pb': 0.05,
           'selec': 'tournament'}
    name = 'f1_t1'
    fx, bounds, bits = problems[name]

    # ind_gen_fn, obj_fn, mut_fn, cross_fn = real_create_funcs(fx, bounds, cfg['mx_pb'], op=1, isint=False)
    ind_gen_fn, obj_fn, mut_fn, cross_fn = real_create_sbx_funcs(fx, bounds, cfg['mx_pb'], eta=0.5, isint=False)
    ga = GA(ind_gen_fn, cross_fn, mut_fn, obj_fn, cfg['selec'], verbose=0)
    ga.set_params(cfg['n_pop'], cfg['cx_pb'], 1, cfg['elitism_p'])

    #%% RUN GA AND PLOT
    results = repeat(ga.run, [cfg['n_gens']], n_repeat=16)
    plot_runs(results, title=name, save=False)

    consolidated_res = consolidate_results(results)
    plot_contour_path(fx, bounds, consolidated_res['best_inds'], delta=1, solid_color=True)

#%%
    print('f(x) = {} +-{} @{} gens'.format(round(np.mean(consolidated_res['best_evals']), 4),
                                           round(np.std(consolidated_res['best_evals']),4),
                                           round(np.mean(consolidated_res['best_gens']),0)))

