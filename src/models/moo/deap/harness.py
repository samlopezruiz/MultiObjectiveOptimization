import array
import copy
import random
import time
from functools import partial

import numpy as np
from deap import base, algorithms, tools, creator
from pymoo.factory import get_performance_indicator

from src.models.moo.deap.nsga3.nsgaiii_survive import selection_NSGA3, get_optimum_pop
from src.models.moo.deap.utils import get_deap_pop_hist
from src.models.moo.utils.indicators import get_hypervolume, get_hvs_from_log
from src.models.moo.utils.plot import plot_gen_progress, plot_multiple_pop, get_fitnesses
from src.utils.plot.plot import plot_hist_hv


def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]


def prepare_toolbox(problem_instance, selection_func, number_of_variables, bounds_low, bounds_up, creator,
                    pop_size=100, max_gen=200, cross_prob=0.9):
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
    toolbox.cross_prob = cross_prob

    return toolbox


def run_algorithm(toolbox, stats=None, verbose=False, type='MuPlusLambda'):
    population = toolbox.population(n=toolbox.pop_size)

    if type == 'MuPlusLambda':
        res = algorithms.eaMuPlusLambda(population, toolbox,
                                        mu=toolbox.pop_size,
                                        lambda_=toolbox.pop_size,
                                        cxpb=toolbox.cross_prob,
                                        mutpb=toolbox.mut_prob,
                                        ngen=toolbox.max_gen,
                                        stats=stats, verbose=verbose)
    elif type == 'Simple':
        res = algorithms.eaSimple(population, toolbox,
                                  cxpb=toolbox.cross_prob,
                                  mutpb=toolbox.mut_prob,
                                  ngen=toolbox.max_gen,
                                  stats=stats, verbose=verbose)
    else:
        raise Exception('type {} is not valid'.format(type))

    return res


def run_nsga3(problem, prob_cfg, algo_cfg, step_size=10, plot_=False, verbose=False,
              save_plots=False, problem_name='', ref_point=np.array([1., 1., 1.])):
    # Preparing an instance of `toolbox`:
    ref_points = tools.uniform_reference_points(nobj=3, p=12)
    selec_func = partial(selection_NSGA3, ref_points=ref_points)
    creator.create("FitnessMin3", base.Fitness, weights=(-1.0,) * prob_cfg['n_obj'])
    creator.create("Individual3", array.array, typecode='d', fitness=creator.FitnessMin3)
    toolbox = prepare_toolbox(problem, selec_func, prob_cfg['n_variables'], prob_cfg['bounds_low'],
                              prob_cfg['bounds_up'], creator, algo_cfg['pop_size'],
                              algo_cfg['max_gen'], algo_cfg['cross_prob'])

    # Store the populations in every iteration.
    stats = tools.Statistics()
    stats.register('pop', copy.deepcopy)
    t0 = time.time()
    res, logbook = run_algorithm(toolbox, stats=stats, verbose=verbose)

    if verbose:
        print('Algorithm finished in {}s'.format(round(time.time() - t0, 4)))

    hvs = get_hvs_from_log(logbook)
    hv = get_performance_indicator("hv", ref_point=ref_point)
    hypervol = hv.calc(get_fitnesses(res))

    pop_hist = get_deap_pop_hist(logbook)

    if verbose:
        print('Hyper-Volume in last generation: {}'.format(round(hypervol, 4)))

    #  Plot
    if plot_:
        #  Plotting hypervolume
        plot_hist_hv(logbook, lib='deap')
        plot_gen_progress(pop_hist, step_size=step_size, save=save_plots, file_path=['img', problem_name + '_gen_hist'],
                          title=problem_name + ' Generation Progress' + '<br>CFG: ' + str(algo_cfg))

        # Plot Optimum Solutions
        optimum = get_optimum_pop(res, ref_points)
        plot_multiple_pop([res, optimum], labels=['population', 'optimum'], save=save_plots, opacities=[0.4, 1],
                          plot_border=True, file_path=['img', problem_name + '_optimum'],
                          title=problem_name + ' Optimum Solutions' + '<br>CFG: ' + str(algo_cfg))

    result = {'pop': res, 'logbook': logbook, 'pop_hist': pop_hist, 'hypervol': hypervol, 'hypervols': hvs}
    return result
