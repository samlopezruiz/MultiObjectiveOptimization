import copy
import time

from pymoo.factory import get_sampling, get_crossover, get_mutation, get_problem, get_reference_directions, \
    get_termination
from pymoo.util.misc import termination_from_tuple

from src.models.moo.nsga3 import NSGA3_parallel
from src.models.moo.utils.indicators import get_hypervolume
from src.models.moo.utils.plot import plot_multiple_pop
from src.models.parallel.utils import cluster_reference_directions, ini_algorithm, ClusterNitching, \
    get_parallel_algorithms
from src.optimization.harness.moo import create_title, get_algorithm
from src.utils.plotly.plot import plot_results_moo
from src.utils.util import get_type_str

if __name__ == '__main__':
    # %%
    prob_cfg = {'name': 'dtlz2',
                'n_obj': 3,
                'n_variables': 24,
                'hv_ref': [5] * 3  # used for HV history
                }
    algo_cfg = {'termination': ('n_gen', 100),
                'max_gen': 150,
                'pop_size': 100,
                'hv_ref': [10] * 3,  # used only for SMS-EMOA
                'strategy': 'keep_outside_cluster'
                }

    problem = get_problem(prob_cfg['name'],
                          n_var=prob_cfg['n_variables'],
                          n_obj=prob_cfg['n_obj'])

    n_clusters = 6

    # %% Initial Population
    algo_cfg['pop_size'] *= n_clusters
    ref_dirs = get_reference_directions("uniform", 3, n_partitions=17)
    cluster_ref_dir = cluster_reference_directions(ref_dirs,
                                                   n_clusters,
                                                   z_score_thold=1,
                                                   plot_distances=False)

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

    # %%
    algo_cfg['pop_size'] = algo_cfg['pop_size'] // n_clusters
    algo_cfg['pop_size'] = 50
    # %%
    cluster_id = 1

    termination = get_termination(algo_cfg['termination'][0], algo_cfg['termination'][1])

    cbx_pb = algo_cfg.get('cbx_pb', 0.9)
    cbx_eta = algo_cfg.get('cbx_eta', 15)
    mut_eta = algo_cfg.get('mut_eta', 20)

    algorithm = NSGA3_parallel(
        ref_dirs=ref_dirs[cluster_ref_dir['labels'] == cluster_id, :],
        clustering=cluster_ref_dir,
        strategy=algo_cfg['strategy'],
        cluster_id=cluster_id,
        pop_size=algo_cfg['pop_size'],
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=cbx_pb, eta=cbx_eta),
        mutation=get_mutation("real_pm", eta=mut_eta),
        eliminate_duplicates=True,
        save_history=True
    )

    algo_cfg['name'] = get_type_str(algorithm)
    algorithm = copy.deepcopy(algorithm)

    # initialize the algorithm object given a problem - if not set already
    if algorithm.problem is None:
        algorithm.setup(problem, termination=termination, seed=None, save_history=True, )
    else:
        if termination is not None:
            algorithm.termination = termination_from_tuple(termination)

    termination = algorithm.termination
    termination = copy.deepcopy(termination)
    algorithm.termination = termination

    # initialize algo
    partial_pop = partial_pops[cluster_id]

    _ = algorithm.infill()
    algorithm.pop = partial_pop
    algorithm.history = []
    algorithm.n_gen = 0
    algorithm.evaluator.eval(algorithm.problem, partial_pop, algorithm)
    algorithm.advance(infills=partial_pop)

    # %%
    t0 = time.time()
    res = algorithm.run()

    opt_time = time.time() - t0
    print('{} with {} finished in {}s'.format(get_type_str(problem), get_type_str(algorithm), round(opt_time, 4)))

    pop_hist = [gen.pop.get('F') for gen in res.history]
    result = {'res': res, 'pop_hist': pop_hist, 'opt_time': opt_time}

    plot_results_moo(result['res'],
                     file_path=['img', 'opt_deap_res'],
                     title=create_title(prob_cfg, algo_cfg, algorithm),
                     save_plots=False)

    hv_pop = get_hypervolume(result['pop_hist'][-1], prob_cfg['hv_ref'])
    hv_opt = get_hypervolume(result['res'].opt.get('F'), prob_cfg['hv_ref'])
    print('Hypervolume from all population for reference point {}: {}'.format(str(prob_cfg['hv_ref']),
                                                                              round(hv_pop, 4)))
    print('Hypervolume from opt population for reference point {}: {}'.format(str(prob_cfg['hv_ref']),
                                                                              round(hv_opt, 4)))
    print('Optimum population {}/{}'.format(len(result['res'].opt),
                                            len(result['pop_hist'][-1])))
