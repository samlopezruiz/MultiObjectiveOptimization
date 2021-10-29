import copy
import time

import numpy as np
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from pymoo.core.population import Population
from pymoo.factory import get_problem, get_reference_directions
from tqdm import tqdm

from src.models.moo.external_archive import ArchiveEps2
from src.models.moo.utils.indicators import get_hypervolume
from src.models.moo.utils.plot import plot_multiple_pop, plot_pop
from src.models.parallel.utils import cluster_reference_directions, ClusterNitching, get_optimum, ini_algorithm, \
    setup_algorithm, run_parallel_algos, results_cluster_nitching, reset_algorithms, get_parallel_algorithms
from src.optimization.harness.moo import get_algorithm, run_moo, create_title
from src.optimization.harness.repeat import repeat_different_args
from src.utils.plot.plot import plot_hv
from src.utils.plotly.plot import plot_results_moo

if __name__ == '__main__':
    # %%
    n_clusters = 15
    ref_dirs = get_reference_directions("uniform", 3, n_partitions=17)

    prob_cfg = {'name': 'dtlz2',
                'n_obj': 3,
                'n_variables': 24,
                'hv_ref': [5] * 3  # used for HV history
                }
    algo_cfg = {'termination': ('n_gen', 100),
                'max_gen': 150,
                'pop_size': ref_dirs.shape[0] // n_clusters * 6,
                'hv_ref': [10] * 3,  # used only for SMS-EMOA,
                'strategy': 'keep_outside_cluster',
                }

    problem = get_problem(prob_cfg['name'],
                          n_var=prob_cfg['n_variables'],
                          n_obj=prob_cfg['n_obj'])

    seed = None

      # cpu_count() - 6
    cluster_ref_dir = cluster_reference_directions(ref_dirs,
                                                   n_clusters,
                                                   z_score_thold=0,
                                                   plot_distances=True)
    algo_cfg['pop_size'] *= n_clusters

    print(cluster_ref_dir['data_structure']['neighbors_thold'])

    #%%
    # print('solving with original algorithm')
    # orig_algo_cfg = copy.copy(algo_cfg)
    # # orig_algo_cfg['pop_size'] = 100
    # orig_algo = get_algorithm('NSGA3', orig_algo_cfg, prob_cfg['n_obj'])
    # orig_result = run_moo(problem, orig_algo, orig_algo_cfg, verbose=2)
    #
    # plot_results_moo(orig_result['res'],
    #                  file_path=['img', 'opt_deap_res'],
    #                  title=create_title(prob_cfg, algo_cfg, orig_algo_cfg),
    #                  save_plots=False)
    #
    # print('Optimum population {}/{}'.format(len(orig_result['res'].opt),
    #                                         len(orig_result['pop_hist'][-1])))
    #
    # hv_pop = get_hypervolume(orig_result['pop_hist'][-1], prob_cfg['hv_ref'])
    # hv_opt = get_hypervolume(orig_result['res'].opt.get('F'), prob_cfg['hv_ref'])
    # print('Hypervolume from all population for reference point {}: {}'.format(str(prob_cfg['hv_ref']),
    #                                                                           round(hv_pop, 4)))
    # print('Hypervolume from opt population for reference point {}: {}'.format(str(prob_cfg['hv_ref']),
    #                                                                           round(hv_opt, 4)))
    # %% Initial Population

    print('Creating Initial Population')
    algo_name = 'NSGA3'
    algorithm = get_algorithm('NSGA3', algo_cfg, n_obj=3, save_history=False)
    algorithm = ini_algorithm(algorithm, problem, algo_cfg)
    pop = algorithm.infill()
    algorithm.evaluator.eval(algorithm.problem, pop, algorithm)
    algorithm.advance(infills=pop)

    F = pop.get('F')
    F_cluster = cluster_ref_dir['clustering'].predict(F)
    centroids = cluster_ref_dir['centroids']

    cluster_nitching = ClusterNitching(prob_cfg['n_obj'], centroids)

    niche_of_individuals = cluster_nitching.get_nitches(F)

    partial_pops = [pop[niche_of_individuals == k] for k in range(n_clusters)]
    partial_pops_f = [p.get('F') for p in partial_pops]

    # plot_multiple_pop(partial_pops_f + [cluster_ref_dir['centroids']],
    #                   pair_lines=cluster_ref_dir['data_structure']['pairs_thold'])

    algo_cfg['pop_size'] = algo_cfg['pop_size'] // n_clusters

    #%%
    # ref_dirs_clustered = [ref_dirs[cluster_ref_dir['labels'] == k] for k in range(n_clusters)]
    # plot_multiple_pop(ref_dirs_clustered, legend_label='cluster')

    #%%
    # k = 0
    # ref_dirs_partial = np.concatenate([ref_dirs[cluster_ref_dir['labels'] == k, :]] +
    #                                   [ref_dirs[cluster_ref_dir['labels'] == n]
    #                                    for n in cluster_ref_dir['data_structure']['neighbors'][k]
    #                                    ], axis=0)
    #
    # plot_pop(ref_dirs_partial)

    # %%
    algo_cfg['termination'] = ('n_gen', 10)
    algo_cfg['global_steps'] = 5
    epsilon_archive = 0.05
    print('initializing...')
    t0 = time.time()
    # algorithms = [setup_algorithm(name, n_obj, algo_cfg, p) for p in partial_pops]
    algorithms = get_parallel_algorithms(algo_cfg,
                                         cluster_ref_dir,
                                         ref_dirs,
                                         n_clusters,
                                         save_history=False,
                                         ref_dirs_with_neighbors=True,
                                         archive_type='eps2',
                                         epsilon_archive=epsilon_archive)

    args = [(algo, problem, algo_cfg, p) for algo, p in zip(algorithms, partial_pops)]
    algorithms = repeat_different_args(setup_algorithm, args, parallel=True)
    print('Initialization time: {} s'.format(round(time.time() - t0, 2)))

    pop_size = algo_cfg['pop_size']
    merge = True
    global_archive = ArchiveEps2(epsilon=epsilon_archive)
    rds = ReferenceDirectionSurvival(ref_dirs)

    t0 = time.time()
    hvs, overhead = [], []
    for i in tqdm(range(algo_cfg['global_steps'])):
        partial_results = run_parallel_algos(algorithms, parallel=True, n_jobs=n_clusters)
        global_pop = np.concatenate([r.pop for r, a in partial_results]).view(Population)

        tg = time.time()
        archives = np.concatenate([a for r, a in partial_results], axis=0).view(Population)
        global_archive.update(archives)
        # global_archive.update(global_pop)
        survival = rds.do(problem, global_pop, n_survive=ref_dirs.shape[0]) #pop_size*n_clusters)
        overhead.append(time.time() - tg)

        if merge:
            global_pop = np.concatenate([survival, global_archive.archive]).view(Population)
            # global_pop = np.concatenate([global_pop, global_archive.archive]).view(Population)
        partial_pops = results_cluster_nitching(global_pop, cluster_nitching, n_clusters)
        reset_algorithms(partial_pops, algorithms)


        hv_pop = get_hypervolume(global_pop.get('F'), prob_cfg['hv_ref'])
        hvs.append(hv_pop)

    plot_multiple_pop([global_pop.get('F'), global_archive.archive.get('F')],
                      colors_ix=[0, 2],
                      prog_color=False,
                      labels=['population', 'archive'])

    # plot_pop(global_archive.archive.get('F'), title='Archive')
    plot_multiple_pop([r.F for r, a in partial_results])
    # plot_multiple_pop([p.get('F') for p in partial_pops])

    print('Overhead time: {} s'.format(round(np.mean(overhead), 4)))
    print('Exec time: {} s'.format(round(time.time() - t0, 2)))

    plot_hv(hvs, title='Hypervolume History')

    opt = get_optimum(global_pop, ref_dirs)

    plot_pop(opt.get('F'), title='Optimum')
    plot_multiple_pop([global_pop.get('F'), opt.get('F')],
                      labels=['population', 'optimum'],
                      opacities=[0.2, 1],
                      plot_border=True)

    print('Optimum population {}/{}'.format(len(opt), len(global_pop)))

    hv_pop = get_hypervolume(global_pop.get('F'), prob_cfg['hv_ref'])
    hv_opt = get_hypervolume(opt.get('F'), prob_cfg['hv_ref'])
    print('Hypervolume from all population for reference point {}: {}'.format(str(prob_cfg['hv_ref']),
                                                                              round(hv_pop, 4)))
    print('Hypervolume from opt population for reference point {}: {}'.format(str(prob_cfg['hv_ref']),
                                                                              round(hv_opt, 4)))
