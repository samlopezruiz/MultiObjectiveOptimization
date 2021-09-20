import array

from deap import base, benchmarks, creator, tools
from deap.algorithms import varOr

from src.models.moo.deap.harness import prepare_toolbox
from src.models.moo.deap.nsga3.nsgaiii import get_sel_nsga_iii_pymoo, get_sel_nsga_iii_deap
from src.models.moo.deap.nsga3.nsgaiii_survive import selection_NSGA3
from src.models.moo.utils.plot import plot_multiple_pop

if __name__ == '__main__':
    # %%
    creator.create("FitnessMin3", base.Fitness, weights=(-1.0,) * 3)
    creator.create("Individual3", array.array, typecode='d',
                   fitness=creator.FitnessMin3)

    number_of_variables = 30
    bounds_low, bounds_up = 0, 1
    toolbox = prepare_toolbox(lambda ind: benchmarks.dtlz2(ind, 3),
                              get_sel_nsga_iii_deap(), number_of_variables,
                              bounds_low, bounds_up, creator)

    toolbox_pymoo = prepare_toolbox(lambda ind: benchmarks.dtlz2(ind, 3),
                                    get_sel_nsga_iii_pymoo(), number_of_variables,
                                    bounds_low, bounds_up, creator)

    population = toolbox.population(n=100)
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    offspring = varOr(population, toolbox, lambda_=toolbox.pop_size, cxpb=toolbox.cross_prob, mutpb=toolbox.mut_prob)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # # Select the next generation population
    ref_points = tools.uniform_reference_points(nobj=3, p=12)
    next_population_deap = selection_NSGA3(population + offspring, 100, ref_points)
    # next_population_deap = toolbox.select(population + offspring, toolbox.pop_size)
    next_population_pymoo = toolbox_pymoo.select(population + offspring, toolbox.pop_size)


    plot_multiple_pop([population, offspring, next_population_deap], labels=['population', 'offspring', 'survivals'],
                         label_scale=1, save=False, opacity=0.6,
                         file_path=['img', 'ini_pop'], title='for Random Initial Population')

    plot_multiple_pop([next_population_deap, next_population_pymoo], labels=['survivals_deap', 'survivals_pymoo'],
                         label_scale=1, save=False, opacity=1, change_markers=True,
                         file_path=['img', 'ini_pop'], title='for Random Initial Population')

