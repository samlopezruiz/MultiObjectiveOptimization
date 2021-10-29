from pymoo.algorithms.moo.moead import MOEAD
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_problem, get_reference_directions

from src.models.moo.nsga3 import NSGA3_parallel
from src.models.moo.utils.indicators import get_hypervolume
from src.models.moo.utils.plot import plot_multiple_pop
from src.models.parallel.utils import ClusterNitching, cluster_reference_directions, ini_algorithm
from src.optimization.harness.moo import run_moo, create_title, get_algorithm
from src.utils.plotly.plot import plot_results_moo
from src.utils.util import get_type_str

if __name__ == '__main__':
    prob_cfg = {'name': 'dtlz2',
                'n_obj': 3,
                'n_variables': 24,
                'hv_ref': [5]*3  # used for HV history
                }
    algo_cfg = {'termination': ('n_gen', 100),
                'max_gen': 150,
                'pop_size': 100,
                'hv_ref': [10]*3  # used only for SMS-EMOA
                }

    problem = get_problem(prob_cfg['name'],
                          n_var=prob_cfg['n_variables'],
                          n_obj=prob_cfg['n_obj'])



    #%%
    n_clusters = 4
    ref_dirs = get_reference_directions("uniform", 3, n_partitions=12)
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

    #%%
    cbx_pb = algo_cfg.get('cbx_pb', 0.9)
    cbx_eta = algo_cfg.get('cbx_eta', 15)
    mut_eta = algo_cfg.get('mut_eta', 20)
    algorithm = NSGA3_parallel(
        clustering=cluster_ref_dir,
        cluster_id=0,
        strategy='keep_outside_cluster',
        pop_size=algo_cfg['pop_size'],
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=cbx_pb, eta=cbx_eta),
        mutation=get_mutation("real_pm", eta=mut_eta),
        eliminate_duplicates=True,
        ref_dirs=ref_dirs,
        save_history=True,
        epsilon_archive=0.08,
        archive_type='eps1'
    )

    algo_cfg['name'] = get_type_str(algorithm)
    result = run_moo(problem, algorithm, algo_cfg, verbose=2)

    plot_results_moo(result['res'],
                     file_path=['img', 'opt_deap_res'],
                     title=create_title(prob_cfg, algo_cfg, algorithm),
                     save_plots=False)

    plot_multiple_pop([result['res'].pop.get('F'), result['res'].opt.get('F'), result['archive'].get('F')],
                      labels=['population', 'optimum', 'archive'],
                      save=False,
                      opacities=[.2, 1, 0.5],
                      plot_border=True)

    hv_pop = get_hypervolume(result['pop_hist'][-1], prob_cfg['hv_ref'])
    hv_opt = get_hypervolume(result['res'].opt.get('F'), prob_cfg['hv_ref'])
    print('Hypervolume from all population for reference point {}: {}'.format(str(prob_cfg['hv_ref']),
                                                                              round(hv_pop, 4)))
    print('Hypervolume from opt population for reference point {}: {}'.format(str(prob_cfg['hv_ref']),
                                                                              round(hv_opt, 4)))
    print('Optimum population {}/{}'.format(len(result['res'].opt),
                                            len(result['pop_hist'][-1])))
