import gc
import time

import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_termination, get_problem, get_sampling, get_crossover, get_mutation, \
    get_reference_directions
from pymoo.optimize import minimize

from src.models.moo.smsemoa import SMSEMOA
from src.models.moo.utils.indicators import get_hypervolume
from src.models.moo.utils.utils import get_moo_args
from src.optimization.harness.repeat import repeat_different_args, repeat
from src.utils.plotly.plot import plot_results_moo
from src.utils.util import get_type_str
from src.models.moo.utils.indicators import hv_hist_from_runs
from src.models.moo.utils.utils import get_hv_hist_vs_n_evals
from src.optimization.harness.plot import plot_runs, plot_histogram
from src.utils.util import array_from_lists, mean_std_from_array, save_df, save_vars


def run_moo(problem, algorithm, algo_cfg, verbose=1, seed=None):
    termination = get_termination(algo_cfg['termination'][0], algo_cfg['termination'][1])

    t0 = time.time()
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=seed,
                   save_history=True,
                   verbose=verbose >= 2)

    opt_time = time.time() - t0
    if verbose >= 1:
        print('{} with {} finished in {}s'.format(get_type_str(problem), get_type_str(algorithm), round(opt_time, 4)))

    pop_hist = [gen.pop.get('F') for gen in res.history]
    result = {'res': res, 'pop_hist': pop_hist, 'opt_time': opt_time}
    return result


def create_title(prob_cfg, algo_cfg, algorithm):
    return prob_cfg['name'] + ' Optimum Solutions' + '<br>' + \
           get_type_str(algorithm) + 'CFG: ' + str(algo_cfg)


def get_algorithm(name, algo_cfg, n_obj=3):
    algo_options = ['SMSEMOA', 'MOEAD', 'NSGA2', 'NSGA3']
    if name not in algo_options:
        raise Exception('Algorithm {} not valid. Options are: {}'.format(name, str(algo_options)))

    cbx_pb = algo_cfg.get('cbx_pb', 0.9)
    cbx_eta = algo_cfg.get('cbx_eta', 15)
    mut_eta = algo_cfg.get('mut_eta', 20)
    ref_dirs = get_reference_directions("energy", n_obj, algo_cfg['pop_size'])
    if name == 'SMSEMOA':
        algorithm = SMSEMOA(
            ref_point=algo_cfg['hv_ref'],
            pop_size=algo_cfg['pop_size'],
            sampling=get_sampling("real_random"),
            crossover=get_crossover("real_sbx", prob=cbx_pb, eta=cbx_eta),
            mutation=get_mutation("real_pm", eta=mut_eta),
            eliminate_duplicates=True
        )
    elif name == 'MOEAD':
        algorithm = MOEAD(
            sampling=get_sampling("real_random"),
            crossover=get_crossover("real_sbx", prob=cbx_pb, eta=cbx_eta),
            mutation=get_mutation("real_pm", eta=mut_eta),
            ref_dirs=ref_dirs,
            n_neighbors=15,
            prob_neighbor_mating=0.7,
        )
    elif name == 'NSGA2':
        algorithm = NSGA2(
            pop_size=algo_cfg['pop_size'],
            sampling=get_sampling("real_random"),
            crossover=get_crossover("real_sbx", prob=cbx_pb, eta=cbx_eta),
            mutation=get_mutation("real_pm", eta=mut_eta),
            eliminate_duplicates=True,
            ref_dirs=ref_dirs
        )
    elif name == 'NSGA3':
        algorithm = NSGA3(
            pop_size=algo_cfg['pop_size'],
            sampling=get_sampling("real_random"),
            crossover=get_crossover("real_sbx", prob=cbx_pb, eta=cbx_eta),
            mutation=get_mutation("real_pm", eta=mut_eta),
            eliminate_duplicates=True,
            ref_dirs=ref_dirs
        )

    return algorithm


def run_moo_problem(name,
                    algo_cfg,
                    prob_cfg,
                    problem=None,
                    plot=True,
                    file_path=['img', 'opt_res'],
                    save_plots=False,
                    verbose=1,
                    seed=None):
    problem = get_problem(prob_cfg['name'],
                          n_var=prob_cfg['n_variables'],
                          n_obj=prob_cfg['n_obj']) if problem is None else problem

    algorithm = get_algorithm(name, algo_cfg, n_obj=prob_cfg['n_obj'])

    algo_cfg['name'] = get_type_str(algorithm)
    result = run_moo(problem, algorithm, algo_cfg, verbose=verbose, seed=seed)

    if plot:
        plot_results_moo(result['res'],
                         file_path=file_path,
                         title=create_title(prob_cfg, algo_cfg, algorithm),
                         save_plots=save_plots)

    hv_pop = get_hypervolume(result['pop_hist'][-1], prob_cfg['hv_ref'])
    hv_opt = get_hypervolume(result['res'].opt.get('F'), prob_cfg['hv_ref'])

    problem_result = {'result': result, 'hv_pop': hv_pop, 'hv_opt': hv_opt}
    return problem_result


def repeat_moo(params, n_repeat, is_reproductible=False):
    if is_reproductible:
        args = []
        for i in range(n_repeat):
            # Specify seed for each run for reproducibility
            params['seed'] = i
            args.append(get_moo_args(params))
        runs = repeat_different_args(run_moo_problem, args)
    else:
        runs = repeat(run_moo_problem, get_moo_args(params), n_repeat=n_repeat)
    return runs


def run_multiple_problems(probs, algos, general_cfg, params, algo_cfg, prob_cfg, folder_cfg):
    for problem, k in probs:
        print('\nRunning Optimization for problem: {}'.format(problem))
        prob_cfg['name'], prob_cfg['n_obj'] = problem, k
        prob_cfg['hv_ref'] = [5] * prob_cfg['n_obj']
        algo_cfg['hv_ref'] = [10] * prob_cfg['n_obj']

        algos_runs, algos_hv_hist_runs = [], []
        for algo in algos:
            t0 = time.time()
            params['name'] = algo
            runs = repeat_moo(params, general_cfg['n_repeat'], general_cfg['is_reproductible'])
            algos_runs.append(runs)
            algos_hv_hist_runs.append(hv_hist_from_runs(runs, ref=prob_cfg['hv_ref']))
            print('\t{} with {} finished in {}s'.format(problem, algo, round(time.time() - t0, 2)))

        if general_cfg['save_stats']:
            save_vars([algo, problem, k, prob_cfg, prob_cfg, algos_hv_hist_runs],
                      file_path=['output',
                                 folder_cfg['experiment'],
                                 folder_cfg['results'], '{}_k{}_res'.format(problem, k)],
                      use_date=general_cfg['use_date'])

        if general_cfg['plot_ind_algorithms']:
            for i, hv_hist_runs in enumerate(algos_hv_hist_runs):
                plot_runs(hv_hist_runs, np.mean(hv_hist_runs, axis=0),
                          x_label='Generations',
                          y_label='Hypervolume',
                          title='{} convergence plot with {}. Ref={}'.format(problem,
                                                                             algos[i],
                                                                             str(prob_cfg['hv_ref'])),
                          size=(15, 9),
                          file_path=['output',
                                     folder_cfg['experiment'],
                                     folder_cfg['images'],
                                     algos[i] + '_' + problem],
                          save=general_cfg['save_ind_algorithms'])

        if algo_cfg['termination'][0] == 'n_eval':
            algos_mean_hv_hist = get_hv_hist_vs_n_evals(algos_runs, algos_hv_hist_runs)
        else:
            algos_mean_hv_hist = array_from_lists([np.mean(hv_hist_runs, axis=0)
                                                   for hv_hist_runs in algos_hv_hist_runs])

        plot_runs(algos_mean_hv_hist,
                  x_label='Function Evaluations' if algo_cfg['termination'][0] == 'n_eval' else 'Generations',
                  y_label='Hypervolume',
                  title='{} convergence plot. Ref={}'.format(problem, str(prob_cfg['hv_ref'])),
                  size=(15, 9),
                  file_path=['output', folder_cfg['experiment'], folder_cfg['images'], 'Compare_' + problem],
                  save=general_cfg['save_comparison_plot'],
                  legend_labels=algos)

        hvs_algos = array_from_lists([hv_hist_runs[:, -1] for hv_hist_runs in algos_hv_hist_runs])
        hvs_stats = mean_std_from_array(hvs_algos, labels=algos)
        exec_times = array_from_lists([[algo_run['result']['opt_time'] for algo_run in algo_runs]
                                       for algo_runs in algos_runs])
        exec_times_stats = mean_std_from_array(exec_times, labels=algos)
        if general_cfg['save_stats']:
            save_df(exec_times_stats, file_path=['output',
                                                 folder_cfg['experiment'],
                                                 folder_cfg['results'],
                                                 '{}_k{}_exec_t'.format(problem, k)],
                    use_date=general_cfg['use_date'])
            save_df(hvs_stats, file_path=['output',
                                          folder_cfg['experiment'],
                                          folder_cfg['results'],
                                          '{}_k{}_hv'.format(problem, k)],
                    use_date=general_cfg['use_date'])

        plot_histogram(hvs_algos,
                       algos,
                       x_label='Algorithm',
                       y_label='Hypervolume',
                       title=problem + ' hypervolume history',
                       size=(15, 9),
                       file_path=['output',
                                  folder_cfg['experiment'],
                                  folder_cfg['images'],
                                  'HVs_' + problem],
                       save=general_cfg['save_stat_plots'],
                       show_grid=False,
                       use_date=general_cfg['use_date'])

        plot_histogram(exec_times,
                       algos,
                       x_label='Algorithm',
                       y_label='Seconds',
                       title=problem + ' execution times',
                       size=(15, 9),
                       file_path=['output',
                                  folder_cfg['experiment'],
                                  folder_cfg['images'],
                                  'Exec_Times_' + problem],
                       save=general_cfg['save_stat_plots'],
                       show_grid=False,
                       use_date=general_cfg['use_date'])

        gc.collect()