from pymoo.factory import get_problem, get_reference_directions

from src.models.moo.parallel import ParallelNSGA3
from src.models.moo.utils.plot import plot_multiple_pop
from src.utils.plot.plot import plot_hv
from src.utils.util import save_vars

if __name__ == '__main__':
    # %%
    seed = None
    problem_name = 'dtlz2'
    problem = get_problem(problem_name,
                          n_var=24,
                          n_obj=3)

    ref_dirs = get_reference_directions("uniform", 3, n_partitions=17)

    global_results = {}
    for n_clusters in range(3, 16, 2):
        increase_pop_factor = 3
        parallel_pop_size = ref_dirs.shape[0] // n_clusters * increase_pop_factor
        initial_pop_size = parallel_pop_size * n_clusters

        parallel_nsga3 = ParallelNSGA3(problem,
                                       n_clusters=n_clusters,
                                       ref_dirs=ref_dirs,
                                       initial_pop_size=initial_pop_size,
                                       epsilon_archive=0.04,
                                       archive_type='eps2',
                                       hv_ref=5,
                                       z_score_thold=0,
                                       strategy='keep_outside_cluster',
                                       verbose=2,
                                       ref_dirs_with_neighbors=True)

        results = []
        for _ in range(20):
            res = parallel_nsga3.run(parallel_pop_size=parallel_pop_size,
                                     global_steps=5,
                                     n_parallel_gen=20,
                                     merge_archive=True)

            results.append({'res': res,
                            'hv': res['hv_history'][-1],
                            'op': len(res['opt']) / ref_dirs.shape[0]})

        global_results[n_clusters] = results

    save_vars(global_results,
              file_path=['output',
                         'iterate_clusters',
                         'results',
                         '{}_res'.format(problem_name)])


