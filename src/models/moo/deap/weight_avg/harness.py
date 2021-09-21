import array
import copy
import time
from functools import partial

import numpy as np
from deap import base, creator, tools
from pymoo.factory import get_performance_indicator

from src.models.moo.deap.harness import prepare_toolbox, run_algorithm
from src.models.moo.deap.weight_avg.util import weighted_avg
from src.models.moo.utils.indicators import get_hypervolume
from src.models.moo.utils.plot import plot_gen_progress, plot_pop
from src.optimization.functions.mop import dtlz2, dtlz1
from src.utils.plot.plot import plot_hv


def run_wa(problem_function, prob_cfg, algo_cfg, weights, step_size=5,
           verbose=False, save_plots=False, problem_name='', plot_=False):
    problem = partial(weighted_avg, function=problem_function, weights=weights)

    # Preparing an instance of `toolbox`:
    selec_func = partial(tools.selTournament, tournsize=5)
    creator.create("FitnessMin3", base.Fitness, weights=(-1.0,))
    creator.create("Individual3", array.array, typecode='d', fitness=creator.FitnessMin3)
    toolbox = prepare_toolbox(problem, selec_func, prob_cfg['n_variables'], prob_cfg['bounds_low'],
                              prob_cfg['bounds_up'], creator, algo_cfg['pop_size'], algo_cfg['max_gen'],
                              algo_cfg['cross_prob'])

    # %%
    # Store the populations in every iteration.
    stats = tools.Statistics()
    stats.register('pop', copy.deepcopy)
    t0 = time.time()
    res, logbook = run_algorithm(toolbox, stats=stats, verbose=verbose, type='Simple')

    run_time = round(time.time() - t0, 4)
    if verbose:
        print('Algorithm finished in {}s'.format(round(time.time() - t0, 4)))

    ix_best = np.argmin(np.array([ind.fitness.values[0] for ind in res]))
    best_ind, best_fitness = res[ix_best], res[ix_best].fitness.values[0]
    # %%
    # Plotting hypervolume
    pops = logbook.select('pop')
    pop_hist = [np.array([problem_function(ind) for ind in pop]) for pop in pops]
    ref = np.max(np.vstack(pop_hist), axis=0)
    hv = get_performance_indicator("hv", ref_point=ref)
    hypervols = [hv.calc(f) for f in pop_hist]
    hv = get_performance_indicator("hv", ref_point=np.array([1., 1., 1.]))
    hypervol = hv.calc(pop_hist[-1])

    if verbose:
        print('Hyper-Volume in last generation: {}'.format(round(hypervols[-1], 4)))

    if plot_:
        plot_hv(hypervols, title='Hypervolume Weighted Average Fitness')
        # Plot Generation Progress
        plot_gen_progress(pop_hist, step_size=step_size, save=save_plots, file_path=['img', problem_name + '_gen_hist'],
                          title=problem_name + ' Generation Progress' + '<br>CFG: ' + str(algo_cfg))

        # Plot Optimum Solutions
        plot_pop(pop_hist[-1], save=save_plots, file_path=['img', problem_name + '_optimum'], plot_norm=False,
                 title=problem_name + ' Optimum Solutions' + '<br>CFG: ' + str(algo_cfg))

    result = {'pop': res, 'logbook': logbook, 'pop_hist': pop_hist, 'hypervol': hypervol,
              'hypervols': hypervols, 'best_ind': best_ind, 'best_fitness': best_fitness}
    return result
