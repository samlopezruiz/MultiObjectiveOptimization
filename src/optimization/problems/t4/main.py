import argparse
import logging
import sys

from pymoo.factory import get_reference_directions, get_problem

from src.models.moo.parallel import ParallelNSGA3
# from src.optimization.problems.t4.irace_opt_func import opt_func

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

def main(epsilon_archive, increase_pop_factor, DATFILE):
    score = opt_func(epsilon_archive, increase_pop_factor)

    # save the fo values in DATFILE
    with open(DATFILE, 'w') as f:
        f.write(str(score))


if __name__ == "__main__":
    # just check if args are ok
    with open('args.txt', 'w') as f:
        f.write(str(sys.argv))

    # loading example arguments
    ap = argparse.ArgumentParser(description='Feature Selection using Parallel NSGA3')
    ap.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    # 3 args to test values
    ap.add_argument('--epsilon_archive', dest='epsilon_archive', type=float, required=True)
    ap.add_argument('--increase_pop_factor', dest='increase_pop_factor', type=int, required=True)
    # ap.add_argument('--mut', dest='mut', type=float, required=True, help='Mutation probability')
    # 1 arg file name to save and load fo value
    ap.add_argument('--datfile', dest='datfile', type=str, required=True,
                    help='File where it will be save the score (result)')

    args = ap.parse_args()
    logging.debug(args)
    # call main function passing args
    main(args.epsilon_archive, args.increase_pop_factor, args.datfile)
