import argparse
import logging
import sys

from pymoo.factory import get_reference_directions, get_problem

from src.models.moo.parallel import ParallelNSGA3
import warnings
warnings.filterwarnings("ignore")

# from src.optimization.problems.t4.irace_opt_func import opt_func

def opt_func(epsilon_archive, increase_pop_factor, global_steps, n_parallel_gen):
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
                             global_steps=global_steps,
                             n_parallel_gen=n_parallel_gen,
                             merge_archive=True)

    # add a minus since IRACE looks to minimze the cost
    score = - round(res['hv_history'][-1] * len(res['opt']) / ref_dirs.shape[0], 4)

    return score


def main(epsilon_archive, increase_pop_factor, global_steps, n_parallel_gen, DATFILE=None):
    score = opt_func(epsilon_archive, increase_pop_factor, global_steps, n_parallel_gen)

    # save the fo values in DATFILE
    if DATFILE is not None:
        with open(DATFILE, 'w') as f:
            f.write(str(score))


if __name__ == "__main__":
    # just check if args are ok
    with open('args.txt', 'w') as f:
        f.write(str(sys.argv))

    # loading example arguments
    ap = argparse.ArgumentParser(description='Feature Selection using Parallel NSGA3')
    ap.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")

    # custom args
    ap.add_argument('--epsilon_archive', dest='epsilon_archive', type=float, required=True)
    ap.add_argument('--increase_pop_factor', dest='increase_pop_factor', type=int, required=True)
    ap.add_argument('--global_steps', dest='global_steps', type=int, required=True)
    ap.add_argument('--n_parallel_gen', dest='n_parallel_gen', type=int, required=True)

    # 1 arg file name to save and load fo value
    ap.add_argument('--datfile', dest='datfile', type=str, required=True)

    args = ap.parse_args()
    logging.debug(args)

    # call main function passing args
    # main(args.epsilon_archive, args.increase_pop_factor, args.global_steps, args.n_parallel_gen)
    main(args.epsilon_archive, args.increase_pop_factor, args.global_steps, args.n_parallel_gen, args.datfile)
