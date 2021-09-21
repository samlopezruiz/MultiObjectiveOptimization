import array
import copy
import time
from functools import partial

from deap import base, creator, tools, benchmarks

from src.models.moo.deap.harness import prepare_toolbox, run_algorithm
from src.models.moo.deap.nsga3.nsgaiii_survive import get_optimum_pop, selection_NSGA3
from src.models.moo.utils.indicators import get_hypervolume
from src.models.moo.utils.plot import plot_multiple_pop
from src.optimization.functions.mop import dtlz1
from src.utils.plot.plot import plot_hist_hv

if __name__ == '__main__':
    # %%
    problem_name = 'DTLZ1'
    prob_cfg = {'n_obj': 3, 'n_variables': 12, 'bounds_low': 0, 'bounds_up': 1}
    algo_cfg = {'algorithm': 'NSGA III', 'max_gen': 1000, 'pop_size': 100}
    verbose = False
    save_plots = True
    custom_problem = True

    # Use DEAP benchmark function or custom function
    problem = dtlz1 if custom_problem else lambda ind: benchmarks.dtlz1(ind, prob_cfg['n_obj'])

    # Preparing an instance of `toolbox`:
    ref_points = tools.uniform_reference_points(nobj=3, p=12)
    selec_func = partial(selection_NSGA3, ref_points=ref_points)
    creator.create("FitnessMin3", base.Fitness, weights=(-1.0,) * prob_cfg['n_obj'])
    creator.create("Individual3", array.array, typecode='d', fitness=creator.FitnessMin3)
    toolbox = prepare_toolbox(problem, selec_func, prob_cfg['n_variables'], prob_cfg['bounds_low'],
                              prob_cfg['bounds_up'], creator, algo_cfg['pop_size'], algo_cfg['max_gen'])

    # %%
    # Store the populations in every iteration.
    stats = tools.Statistics()
    stats.register('pop', copy.deepcopy)
    t0 = time.time()
    res, logbook = run_algorithm(toolbox, stats=stats, verbose=verbose)

    print('Algorithm finished in {}s'.format(round(time.time() - t0, 4)))

    # %% Plotting hypervolume
    plot_hist_hv(logbook, lib='deap')
    hypervol = get_hypervolume(res)
    print('Hyper-Volume in last generation: {}'.format(round(hypervol, 4)))

    # %% Plot Generation Progress
    # pop_hist = get_deap_pop_hist(logbook)
    # plot_gen_progress(pop_hist, step_size=5, save=save_plots, file_path=['img', problem_name+'_gen_hist'],
    #                   title=problem_name+' Generation Progress' + '<br>CFG: ' + str(algo_cfg))

    # Plot Optimum Solutions
    optimum = get_optimum_pop(res, ref_points)
    plot_multiple_pop([res, optimum], labels=['population', 'optimum'], save=save_plots, opacities=[0.4, 1],
                      plot_border=True, file_path=['img', problem_name+'_optimum'],
                      title=problem_name+' Optimum Solutions' + '<br>CFG: ' + str(algo_cfg))
