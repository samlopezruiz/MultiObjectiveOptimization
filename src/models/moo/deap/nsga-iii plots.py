import array
from deap import base, benchmarks, creator
from src.models.moo.deap.harness import prepare_toolbox
from src.models.moo.deap.nsgaiii import sel_nsga_iii
from src.models.moo.deap.plot import plot_objective_space, \
    plot_norm_objective_space

if __name__ == '__main__':
    # %%
    creator.create("FitnessMin3", base.Fitness, weights=(-1.0,) * 3)
    creator.create("Individual3", array.array, typecode='d',
                   fitness=creator.FitnessMin3)

    number_of_variables = 30
    bounds_low, bounds_up = 0, 1
    toolbox = prepare_toolbox(lambda ind: benchmarks.dtlz2(ind, 3),
                              sel_nsga_iii, number_of_variables,
                              bounds_low, bounds_up, creator)

    pop = toolbox.population(n=20)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    plot_objective_space(pop, label_scale=1)
    plot_norm_objective_space(pop, label_scale=1)

