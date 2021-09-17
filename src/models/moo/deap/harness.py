import random
from deap import base, algorithms, tools


def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]


def prepare_toolbox(problem_instance, selection_func, number_of_variables, bounds_low, bounds_up, creator,
                    pop_size=100, max_gen=200):
    toolbox = base.Toolbox()
    toolbox.register('evaluate', problem_instance)
    toolbox.register('select', selection_func)
    toolbox.register("attr_float", uniform, bounds_low, bounds_up, number_of_variables)
    toolbox.register("individual", tools.initIterate, creator.Individual3, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                     low=bounds_low, up=bounds_up, eta=20.0)
    toolbox.register("mutate", tools.mutPolynomialBounded,
                     low=bounds_low, up=bounds_up, eta=20.0,
                     indpb=1.0 / number_of_variables)

    toolbox.pop_size = pop_size  # population size
    toolbox.max_gen = max_gen  # max number of iteration
    toolbox.mut_prob = 1 / number_of_variables
    toolbox.cross_prob = 0.3

    return toolbox


def nsga_iii(toolbox, stats=None, verbose=False):
    population = toolbox.population(n=toolbox.pop_size)

    return algorithms.eaMuPlusLambda(population, toolbox,
                                     mu=toolbox.pop_size,
                                     lambda_=toolbox.pop_size,
                                     cxpb=toolbox.cross_prob,
                                     mutpb=toolbox.mut_prob,
                                     ngen=toolbox.max_gen,
                                     stats=stats, verbose=verbose)
