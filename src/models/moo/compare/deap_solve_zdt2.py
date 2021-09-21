import array
import copy
import time

from deap import base, creator, tools, benchmarks

from src.models.moo.deap.harness import prepare_toolbox, run_algorithm
from src.models.moo.deap.nsga3.nsgaiii import sel_nsga_iii, ind_to_fitness_array
from src.models.moo.utils.plot import plot_pop_obj_space, plot_pop_obj_hist
from src.optimization.functions.mop import dtlz2
from src.utils.plot.plot import plot_hist_hv

if __name__ == '__main__':
    #%%
    custom_problem = False
    prob_cfg = {'n_obj': 3, 'n_variables': 30, 'bounds_low': 0, 'bounds_up': 1}
    algo_cfg = {'max_gen': 100, 'pop_size': 100}
    verbose = False

    # Preparing an instance of `toolbox`:
    problem = dtlz2 if custom_problem else lambda ind: benchmarks.dtlz2(ind, prob_cfg['n_obj'])
    creator.create("FitnessMin3", base.Fitness, weights=(-1.0,) * prob_cfg['n_obj'])
    creator.create("Individual3", array.array, typecode='d', fitness=creator.FitnessMin3)
    toolbox = prepare_toolbox(problem, sel_nsga_iii, prob_cfg['n_variables'], prob_cfg['bounds_low'],
                              prob_cfg['bounds_up'], creator, algo_cfg['pop_size'], algo_cfg['max_gen'])

    # Store the populations in every iteration.
    stats = tools.Statistics()
    stats.register('pop', copy.deepcopy)
    t0 = time.time()
    res, logbook = run_algorithm(toolbox, stats=stats, verbose=verbose)
    print('Algorithm finished in {}s'.format(round(time.time() - t0, 4)))

    # Resulting population.
    plot_pop_obj_space(ind_to_fitness_array(res))

    # Plotting hypervolume
    plot_hist_hv(logbook, lib='deap')

    #%%
    import numpy as np
    pop_hist = []
    for gen in logbook:
        pop = gen['pop']
        pop_hist.append(np.array([ind.fitness.values for ind in gen['pop']]))

    plot_pop_obj_hist(pop_hist)


