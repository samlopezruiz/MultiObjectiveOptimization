# %%
import array
import copy
import matplotlib.pyplot as plt
import numpy as np
from deap import algorithms, base, benchmarks, creator
from deap import tools

from src.models.moo.deap.harness import prepare_toolbox, nsga_iii
from src.models.moo.deap.nsgaiii import sel_nsga_iii

# # Understanding NSGA-III selection
# In order to understand the NSGA-III selection mechanism we will make an example iteration
# of solving a three-objectives DTLZ2 problem.
if __name__ == '__main__':
    # %%
    creator.create("FitnessMin3", base.Fitness, weights=(-1.0,) * 3)
    creator.create("Individual3", array.array, typecode='d',
                   fitness=creator.FitnessMin3)

    # Preparing an instance of `toolbox`:
    number_of_variables = 30
    bounds_low, bounds_up = 0, 1
    toolbox = prepare_toolbox(lambda ind: benchmarks.dtlz2(ind, 3),
                              sel_nsga_iii, number_of_variables,
                              bounds_low, bounds_up, creator)

    # # Implementing NSGA-III
    # Having the selection mechanism the implementantion of NSGA-III is very simple thanks to DEAP.
    # We create a `Statistics` instance to store the populations in every iteration.
    stats = tools.Statistics()
    stats.register('pop', copy.deepcopy)

    # Running NSGA-III...
    res, logbook = nsga_iii(toolbox, stats=stats)

    # Resulting population.
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    for ind in res:
        ax.scatter(ind.fitness.values[0],
                   ind.fitness.values[1],
                   ind.fitness.values[2], c='purple', marker='o')

    ax.set_xlabel('$f_1()$', fontsize=15)
    ax.set_ylabel('$f_2()$', fontsize=15)
    ax.set_zlabel('$f_3()$', fontsize=15)
    ax.view_init(elev=11, azim=-21)
    plt.autoscale(tight=True)
    plt.show()

    # ### Plotting hypervolume
    pops = logbook.select('pop')
    from deap.tools._hypervolume import hv

    def hypervolume(individuals, ref=None):
        front = tools.sortLogNondominated(individuals, len(individuals), first_front_only=True)
        wobjs = np.array([ind.fitness.wvalues for ind in front]) * -1
        if ref is None:
            ref = np.max(wobjs, axis=0) + 1
        return hv.hypervolume(wobjs, ref)

    pops_obj = [np.array([ind.fitness.wvalues for ind in pop]) * -1 for pop in pops]
    ref = np.max([np.max(wobjs, axis=0) for wobjs in pops_obj], axis=0) + 1
    hypervols = [hypervolume(pop, ref) for pop in pops]

    plt.plot(hypervols)
    plt.xlabel('Iterations (t)')
    plt.ylabel('Hypervolume')
    plt.show()
