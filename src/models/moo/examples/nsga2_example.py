import time

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination, get_problem, \
    get_reference_directions
from pymoo.optimize import minimize

from src.models.moo.utils.indicators import get_hypervolume
from src.models.moo.utils.plot import plot_multiple_pop
from src.utils.plot.plot import plot_hist_hv

if __name__ == '__main__':

    prob_cfg = {'name': 'wfg2', 'n_obj': 3, 'n_variables': 24, 'hv_ref': [5, 5, 5]}
    algo_cfg = {'algorithm': 'NSGA II', 'max_gen': 200, 'pop_size': 200}
    save_plots = False
    problem = get_problem(prob_cfg['name'], n_var=prob_cfg['n_variables'], n_obj=prob_cfg['n_obj'])

    # create the reference directions of size equal to population size
    ref_dirs = get_reference_directions("energy", 3, algo_cfg['pop_size'])

    algorithm = NSGA2(
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
    plot_hist_hv(res)

    # Plot Optimum Solutions
    pop_hist = [gen.pop.get('F') for gen in res.history]
    plot_multiple_pop([pop_hist[-1], res.opt.get('F')], labels=['population', 'optimum'], save=save_plots,
                      opacities=[.2, 1], plot_border=True, file_path=['img', 'opt_deap_res'],
                      title=prob_cfg['name']+' Optimum Solutions' + '<br>CFG: ' + str(algo_cfg))

    #%%
    hv_pop = get_hypervolume(pop_hist[-1], prob_cfg['hv_ref'])
    hv_opt = get_hypervolume(res.opt.get('F'), prob_cfg['hv_ref'])
    print('Hypervolume from all population for reference point {}: {}'.format(str(prob_cfg['hv_ref']), round(hv_pop, 4)))
    print('Hypervolume from opt population for reference point {}: {}'.format(str(prob_cfg['hv_ref']), round(hv_opt, 4)))
    print('Optimum population {}/{}'.format(len(res.opt), len(pop_hist[-1])))
