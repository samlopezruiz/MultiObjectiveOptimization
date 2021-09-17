import array
import copy

import numpy as np
from deap import base, benchmarks, creator, tools

from src.models.moo.deap.harness import prepare_toolbox, nsga_iii
from src.models.moo.deap.indicators import hypervolume
from src.models.moo.deap.nsgaiii import sel_nsga_iii
from src.models.moo.deap.plot import plot_objective_space
from src.utils.plot.plot import plot_hv

if __name__ == '__main__':
    # %%
    n_obj = 3
    number_of_variables = 30
    max_gen = 100
    pop_size = 50
    bounds_low, bounds_up = 0, 1

    # Preparing an instance of `toolbox`:
    creator.create("FitnessMin3", base.Fitness, weights=(-1.0,) * n_obj)
    creator.create("Individual3", array.array, typecode='d',
                   fitness=creator.FitnessMin3)
    toolbox = prepare_toolbox(lambda ind: benchmarks.dtlz2(ind, 3),
                              sel_nsga_iii, number_of_variables,
                              bounds_low, bounds_up, creator, pop_size, max_gen)

    # Store the populations in every iteration.
    stats = tools.Statistics()
    stats.register('pop', copy.deepcopy)
    res, logbook = nsga_iii(toolbox, stats=stats, verbose=True)

    # Resulting population.
    plot_objective_space(res, plot_norm=False)
    print('Algorithm Finished')

    #%% Plotting hypervolume
    pops = logbook.select('pop')
    pops_obj = [np.array([ind.fitness.wvalues for ind in pop]) * -1 for pop in pops]
    ref = np.max([np.max(wobjs, axis=0) for wobjs in pops_obj], axis=0) + 1
    hypervols = [hypervolume(pop, ref) for pop in pops]
    plot_hv(hypervols)

