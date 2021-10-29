from pymoo.factory import get_problem, get_reference_directions

from src.models.moo.parallel import ParallelNSGA3
from src.utils.util import save_vars

if __name__ == '__main__':
    # %%
    seed = None
    plot_ = False
    problem_name = 'wfg2'
    archive_type = 'eps2'
    problem = get_problem(problem_name,
                          n_var=24,
                          n_obj=3)
    n_clusters = 15
    ref_dirs = get_reference_directions("uniform", 3, n_partitions=17)

    increase_pop_factor = 4
    parallel_pop_size = ref_dirs.shape[0] // n_clusters * increase_pop_factor
    initial_pop_size = parallel_pop_size * n_clusters

    results = []
    for i in range(20):
        parallel_nsga3 = ParallelNSGA3(problem,
                                       n_clusters=n_clusters,
                                       ref_dirs=ref_dirs,
                                       initial_pop_size=initial_pop_size,
                                       epsilon_archive=0.03,
                                       archive_type=archive_type,
                                       hv_ref=10,
                                       z_score_thold=0,
                                       strategy='keep_outside_cluster',
                                       verbose=0,
                                       ref_dirs_with_neighbors=True)

        print('Running experiment {}'.format(i))
        try:
            res = parallel_nsga3.run(parallel_pop_size=parallel_pop_size,
                                     global_steps=3,
                                     n_parallel_gen=30,
                                     merge_archive=True)

            results.append({'result': res,
                            'hv': res['hv_history'][-1],
                            'op': len(res['opt']) / ref_dirs.shape[0],
                            'time': res['times'].loc['time_parallel_execution', 0]})
        except:
            print('Error')
            results.append({'result': None,
                            'hv': results[-1]['hv'],
                            'op': results[-1]['op'],
                            'time': results[-1]['time']})

    for res in results:
        del res['result']

    save_vars(results,
              file_path=['output',
                         'clusters',
                         'results',
                         '{}_{}_m{}_res'.format(problem_name, archive_type, n_clusters)])


