from pymoo.factory import get_problem, get_reference_directions

from src.models.moo.parallel import ParallelNSGA3
from src.models.moo.utils.plot import plot_multiple_pop
from src.utils.plot.plot import plot_hv

if __name__ == '__main__':
    # %%
    seed = None
    plot_ = True
    problem_name = 'wfg3'
    problem = get_problem(problem_name,
                          n_var=24,
                          n_obj=3)
    n_clusters = 5
    ref_dirs = get_reference_directions("uniform", 3, n_partitions=17)

    increase_pop_factor = 4
    parallel_pop_size = ref_dirs.shape[0] // n_clusters * increase_pop_factor
    initial_pop_size = parallel_pop_size * n_clusters

    parallel_nsga3 = ParallelNSGA3(problem,
                                   n_clusters=n_clusters,
                                   ref_dirs=ref_dirs,
                                   initial_pop_size=initial_pop_size,
                                   epsilon_archive=0.03,
                                   archive_type='eps2',
                                   hv_ref=10,
                                   z_score_thold=0,
                                   strategy='keep_outside_cluster',
                                   verbose=2,
                                   ref_dirs_with_neighbors=True)

    res = parallel_nsga3.run(parallel_pop_size=parallel_pop_size,
                             global_steps=3,
                             n_parallel_gen=30,
                             merge_archive=True)

    if plot_:

        if res['archive'] is not None:
            plot_multiple_pop([res['pop'].get('F'), res['archive'].get('F')],
                              colors_ix=[0, 2],
                              prog_color=False,
                              labels=['population', 'archive'])

        plot_multiple_pop([r.F for r, a in res['partial_results']])

        print('\nExec times: {}'.format(res['times']))

        plot_hv(res['hv_history'], title='Hypervolume History')

        plot_multiple_pop([res['pop'].get('F'), res['opt'].get('F')],
                          labels=['population', 'optimum'],
                          opacities=[0.2, 1],
                          plot_border=True)

        print('\nOptimum population {}/{} out of {} ref dirs'.format(len(res['opt']),
                                                                     len(res['pop']),
                                                                     ref_dirs.shape[0]))

        print('\nHypervolume {}'.format(round(res['hv_history'][-1], 4)))

        score = res['hv_history'][-1] * len(res['opt']) / ref_dirs.shape[0]
