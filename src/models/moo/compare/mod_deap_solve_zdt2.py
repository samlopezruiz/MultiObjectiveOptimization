import array
import copy
import time

import numpy as np
from deap import base, creator, tools, benchmarks
from pymoo.algorithms.nsga3 import ReferenceDirectionSurvival
from pymoo.factory import get_reference_directions

from src.models.moo.deap.harness import prepare_toolbox, nsga_iii
from src.models.moo.deap.nsgaiii import sel_nsga_iii, ind_to_fitness_array, generate_reference_points, ReferencePoint, \
    get_sel_nsga_iii_pymoo
from src.models.moo.deap.plot import plot_objective_space, plot_pop_obj_space, plot_pop_obj_hist
from src.models.moo.deap.survive import MyReferenceDirectionSurvival
from src.models.moo.deap.utils import get_deap_pop_hist
from src.optimization.functions.mop import dtlz2
from src.utils.plot.plot import plot_hist_hv

if __name__ == '__main__':
    # %%
    custom_problem = False
    prob_cfg = {'n_obj': 3, 'n_variables': 30, 'bounds_low': 0, 'bounds_up': 1}
    algo_cfg = {'algorithm': 'NSGA III', 'max_gen': 100, 'pop_size': 150}
    verbose = False
    save_plots = True

    # Use PYMOOS selection function or custom function
    selec = 'custom'
    sel_func = get_sel_nsga_iii_pymoo() if selec == 'pymoo' else sel_nsga_iii

    # Use DEAP benchmark function or custom function
    problem = dtlz2 if custom_problem else lambda ind: benchmarks.dtlz2(ind, prob_cfg['n_obj'])

    # Preparing an instance of `toolbox`:
    creator.create("FitnessMin3", base.Fitness, weights=(-1.0,) * prob_cfg['n_obj'])
    creator.create("Individual3", array.array, typecode='d', fitness=creator.FitnessMin3)
    toolbox = prepare_toolbox(problem, sel_func, prob_cfg['n_variables'], prob_cfg['bounds_low'],
                              prob_cfg['bounds_up'], creator, algo_cfg['pop_size'], algo_cfg['max_gen'])

    # %%
    # Store the populations in every iteration.
    stats = tools.Statistics()
    stats.register('pop', copy.deepcopy)
    t0 = time.time()
    res, logbook = nsga_iii(toolbox, stats=stats, verbose=verbose)
    print('Algorithm finished in {}s'.format(round(time.time() - t0, 4)))

    # Get optimum individuals form resulting population.
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
    survive = MyReferenceDirectionSurvival(ref_dirs)
    _ = survive.do(res, len(res))
    plot_pop_obj_space(ind_to_fitness_array(survive.opt), save=True,
                       file_path=['img', 'opt_deap_res'], title='ZDT2' + '<br>CFG: ' + str(algo_cfg))

    # Plotting hypervolume
    plot_hist_hv(logbook, lib='deap')

    # %%
    pop_hist = get_deap_pop_hist(logbook)
    plot_pop_obj_hist(pop_hist, save=save_plots,
                      file_path=['img', 'opt_deap_pop_hist'], title='ZDT2' + '<br>CFG: ' + str(algo_cfg))
