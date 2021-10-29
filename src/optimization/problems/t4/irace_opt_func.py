import sys

from pymoo.factory import get_problem, get_reference_directions

from src.models.moo.parallel import ParallelNSGA3


def opt_func(epsilon_archive, increase_pop_factor):
    problem = get_problem('dtlz2',
                          n_var=24,
                          n_obj=3)
    n_clusters = 5
    ref_dirs = get_reference_directions("uniform", 3, n_partitions=17)

    parallel_pop_size = ref_dirs.shape[0] // n_clusters * increase_pop_factor
    initial_pop_size = parallel_pop_size * n_clusters

    parallel_nsga3 = ParallelNSGA3(problem,
                                   n_clusters=n_clusters,
                                   ref_dirs=ref_dirs,
                                   initial_pop_size=initial_pop_size,
                                   epsilon_archive=epsilon_archive,
                                   archive_type='eps2',
                                   hv_ref=5,
                                   z_score_thold=0,
                                   strategy='keep_outside_cluster',
                                   verbose=0,
                                   ref_dirs_with_neighbors=True)

    res = parallel_nsga3.run(parallel_pop_size=parallel_pop_size,
                             global_steps=5,
                             n_parallel_gen=20,
                             merge_archive=True)

    score = res['hv_history'][-1] * len(res['opt']) / ref_dirs.shape[0]

    return score


if __name__ == '__main__':
    print(opt_func(epsilon_archive=0.02, increase_pop_factor=2))
    # sys.exit()
