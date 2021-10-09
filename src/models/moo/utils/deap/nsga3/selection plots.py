import array
from functools import partial

from deap import base, benchmarks, creator, tools
from deap.algorithms import varOr
from deap.tools import sortLogNondominated

from src.models.moo.utils.deap.harness import prepare_toolbox
from src.models.moo.utils.deap.nsga3.nsgaiii_survive import selection_NSGA3
from src.models.moo.utils.plot import plot_nitching_selec_deap, plot_multiple_pop, plot_pop, get_fitnesses

if __name__ == '__main__':
    # %%
    problem_name = 'DTLZ2'
    save_plots = True

    creator.create("FitnessMin3", base.Fitness, weights=(-1.0,) * 3)
    creator.create("Individual3", array.array, typecode='d',
                   fitness=creator.FitnessMin3)

    number_of_variables = 30
    bounds_low, bounds_up = 0, 1
    ref_points = tools.uniform_reference_points(nobj=3, p=12)
    selec_func = partial(selection_NSGA3, ref_points=ref_points)
    toolbox = prepare_toolbox(lambda ind: benchmarks.dtlz2(ind, 3),
                              selec_func, number_of_variables,
                              bounds_low, bounds_up, creator, pop_size=50)

    population = toolbox.population(n=50)
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    offspring = varOr(population, toolbox, lambda_=toolbox.pop_size, cxpb=toolbox.cross_prob, mutpb=toolbox.mut_prob)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    next_population = toolbox.select(population + offspring, toolbox.pop_size)
    individuals = population + offspring

    plot_multiple_pop([population, offspring], labels=['population', 'offspring'], save=save_plots,
                      file_path=['img', problem_name + '_pop_off'],
                      title=problem_name + ' Random Population and Offspring')

    plot_pop(individuals, file_path=['img', problem_name + '_normalized'], save=save_plots,
             title=problem_name + ' Total Population (mu + lambda) Normalized')

    pareto_fronts = sortLogNondominated(individuals, len(individuals))
    Fs = [get_fitnesses(front) for front in pareto_fronts]
    plot_multiple_pop(Fs[::-1], prog_color=False, prog_opacity=True, legend_label='front', plot_border=True,
                      file_path=['img', problem_name + '_pareto'], save=save_plots,
                      title=problem_name + ' Pareto Fronts')

    plot_nitching_selec_deap(individuals, 50, file_path=['img', problem_name + '_niching'], save=save_plots,
                             title=problem_name + ' NSGA3 Niching Selection')

    plot_multiple_pop([population, offspring, next_population], labels=['population', 'offspring', 'next_pop'],
                      file_path=['img', problem_name + '_selection'], save=save_plots,
                      title=problem_name + ' Next Generation After NSGA Nitching Selection')
