import time

import numpy as np
from pymoo.algorithms.nsga3 import NSGA3
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination, get_problem, \
    get_reference_directions
from pymoo.optimize import minimize

from src.models.moo.deap.plot import plot_pop_obj_space, plot_pop_obj_hist
from src.utils.plot.plot import plot_hist_hv

if __name__ == '__main__':

    prob_cfg = {'n_obj': 3, 'n_variables': 30, 'bounds_low': 0, 'bounds_up': 1}
    algo_cfg = {'algorithm': 'NSGA III', 'max_gen': 100, 'pop_size': 100}

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

    plot_pop_obj_space(res.F, save=True, file_path=['img', 'opt_pymoo_res'],
                       title='ZDT2' + '<br>CFG: ' + str(algo_cfg))
    plot_hist_hv(res, lib='pymoo')

    pop_hist = [gen.pop.get('F') for gen in res.history]
    plot_pop_obj_hist(pop_hist, save=True, file_path=['img', 'opt_pymoo_pop_hist'],
                      title='ZDT2' + '<br>CFG: ' + str(algo_cfg))


