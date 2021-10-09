import array
import copy
import time
from functools import partial

import numpy as np
from deap import base, creator, tools
from pymoo.factory import get_performance_indicator

from src.models.moo.utils.deap.harness import prepare_toolbox, run_algorithm
from src.models.moo.utils.deap.weight_avg.util import weighted_avg
from src.models.moo.utils.indicators import get_hypervolume
from src.models.moo.utils.plot import plot_gen_progress, plot_pop
from src.optimization.functions.mop import dtlz1
from src.utils.plot.plot import plot_hv

if __name__ == '__main__':
    # %%
    problem_name = 'DTLZ2'
    prob_cfg = {'n_obj': 3, 'n_variables': 12, 'bounds_low': 0, 'bounds_up': 1}
    algo_cfg = {'algorithm': 'WEIGHTED AVGERAGE', 'max_gen': 100, 'pop_size': 100, 'cross_prob': 0.5}
    verbose = False
    save_plots = False
    custom_problem = False

    problem_function = dtlz1
    weights = np.array([.2, .3, .5])
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

    print('Algorithm finished in {}s'.format(round(time.time() - t0, 4)))

    # %%
    # Plotting hypervolume
    pops = logbook.select('pop')
    pop_hist = [np.array([problem_function(ind) for ind in pop]) for pop in pops]
    ref = np.max(np.vstack(pop_hist), axis=0)
    hv = get_performance_indicator("hv", ref_point=ref)
    hypervols = [hv.calc(f) for f in pop_hist]
    plot_hv(hypervols, title='Hypervolume Weighted Average Fitness')

    hypervol = get_hypervolume(pop_hist[-1])
    print('Hyper-Volume in last generation: {}'.format(round(hypervol, 4)))

    # %% Plot Generation Progress
    plot_gen_progress(pop_hist, step_size=5, save=save_plots, file_path=['img', problem_name + '_gen_hist'],
                      title=problem_name + ' Generation Progress' + '<br>CFG: ' + str(algo_cfg))

    # Plot Optimum Solutions
    plot_pop(pop_hist[-1], save=save_plots, file_path=['img', problem_name + '_optimum'], plot_norm=False,
             title=problem_name + ' Optimum Solutions' + '<br>CFG: ' + str(algo_cfg))
