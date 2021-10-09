import time

import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3

from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination, get_problem, \
    get_reference_directions
from pymoo.optimize import minimize

from src.models.moo.utils.plot import plot_multiple_pop, plot_gen_progress
from src.utils.plot.plot import plot_hist_hv

if __name__ == '__main__':

    prob_cfg = {'n_obj': 3, 'n_variables': 30, 'bounds_low': 0, 'bounds_up': 1}
    algo_cfg = {'algorithm': 'NSGA III', 'max_gen': 100, 'pop_size': 100}
    save_plots = False

    problem = get_problem("dtlz2")
    problem.n_var = prob_cfg['n_variables']
    problem.n_obj = prob_cfg['n_obj']
    problem.xl = np.ones(prob_cfg['n_variables']) * prob_cfg['bounds_low']
    problem.xu = np.ones(prob_cfg['n_variables']) * prob_cfg['bounds_up']

    # create the reference directions to be used for the optimization
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

    algorithm = NSGA3(
        pop_size=algo_cfg['pop_size'],
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True,
        ref_dirs=ref_dirs
    )

    termination = get_termination("n_gen", algo_cfg['max_gen'])

    t0 = time.time()
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=True,
                   verbose=True)
    print('Algorithm finished in {}s'.format(round(time.time() - t0, 4)))

    # %%
    # Plotting hypervolume
    plot_hist_hv(res, lib='py_moo')

    # %% Plot Generation Progress
    pop_hist = [gen.pop.get('F') for gen in res.history]
    plot_gen_progress(pop_hist, step_size=5, save=save_plots,
                      file_path=['img', 'opt_deap_res'],
                      title='DTLZ2 Generation Progress' + '<br>CFG: ' + str(algo_cfg))
    #%%
    # Plot Optimum Solutions
    plot_multiple_pop([pop_hist[-1], res.opt.get('F')], labels=['population', 'optimum'], save=save_plots,
                      opacities=[.2, 1], plot_border=True, file_path=['img', 'opt_deap_res'],
                      title='DTLZ2 Optimum Solutions' + '<br>CFG: ' + str(algo_cfg))
