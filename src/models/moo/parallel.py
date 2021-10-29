import time

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.core.evaluator import Evaluator
from pymoo.core.initialization import Initialization
from pymoo.core.population import Population
from pymoo.core.repair import NoRepair
from pymoo.factory import get_sampling, get_crossover, get_mutation, \
    get_termination
from pymoo.operators.sampling.rnd import FloatRandomSampling
from tqdm import tqdm

from src.models.moo.external_archive import ArchiveEps2, ArchiveEps1
from src.models.moo.nsga3 import NSGA3_parallel
from src.models.moo.utils.indicators import get_hypervolume
from src.models.parallel.utils import cluster_reference_directions, ClusterNitching, get_optimum, setup_algorithm, \
    run_parallel_algos, results_cluster_nitching, reset_algorithms
from src.optimization.harness.repeat import repeat_different_args


class ParallelNSGA3:
    def __init__(self,
                 problem,
                 n_clusters,
                 ref_dirs,
                 initial_pop_size=500,
                 epsilon_archive=0.05,
                 archive_type='eps2',
                 hv_ref=5,
                 z_score_thold=0,
                 strategy='keep_outside_cluster',
                 verbose=1,
                 ref_dirs_with_neighbors=False,
                 ):

        self.ref_dirs_partial = None
        self.partial_results = None
        self.opt = None
        self.archive = None
        self.pop = None
        archive_options = ['eps1', 'eps2']
        if archive_type is not None and archive_type not in archive_options:
            raise Exception('{} not a valid option. Options: {}'.format(archive_type, archive_options))

        if archive_type:
            if archive_type == 'eps1':
                self.global_archive = ArchiveEps1(epsilon=epsilon_archive)
            elif archive_type == 'eps2':
                self.global_archive = ArchiveEps2(epsilon=epsilon_archive)
        else:
            self.global_archive = None

        self.hv_history = []
        self.rds = ReferenceDirectionSurvival(ref_dirs)
        self.algorithms = []
        self.problem = problem
        self.n_clusters = n_clusters
        self.ref_dirs = ref_dirs
        self.initial_pop_size = initial_pop_size
        self.epsilon_archive = epsilon_archive
        self.archive_type = archive_type
        self.z_score_thold = z_score_thold
        self.strategy = strategy
        self.verbose = verbose
        self.ref_dirs_with_neighbors = ref_dirs_with_neighbors
        self.exec_times = {}
        self.n_obj = problem.n_obj
        self.hv_ref = [hv_ref] * self.n_obj

        # attributes
        self.cluster_nitching = None
        self.ini_partial_pops = None

        self.termination = None
        self.cluster_ref_dirs()
        self.create_initial_clustered_population()

    def run(self,
            parallel_pop_size,
            n_parallel_gen=10,
            global_steps=5,
            merge_archive=True):

        self.initialize_parallel_algorithms(parallel_pop_size, n_parallel_gen)
        self.run_parallel_algorithms(global_steps,
                                     parallel_pop_size,
                                     merge_archive=merge_archive)

        return {
            'times': pd.DataFrame.from_dict(self.exec_times, orient='index'),
            'pop': self.pop,
            'opt': self.opt,
            'archive': self.archive,
            'hv_history': self.hv_history,
            'partial_results': self.partial_results,
        }

    def cluster_ref_dirs(self):
        t0 = time.time()
        self.cluster_ref_dir = cluster_reference_directions(self.ref_dirs,
                                                            self.n_clusters,
                                                            z_score_thold=self.z_score_thold,
                                                            plot_distances=False)
        self.exec_times['time_cluster_ref_dir'] = time.time() - t0

    def create_initial_clustered_population(self):
        if self.verbose > 1:
            print('Creating initial population')

        t0 = time.time()
        # create and eval new population
        initialization = Initialization(FloatRandomSampling(),
                                        repair=NoRepair(),
                                        eliminate_duplicates=DefaultDuplicateElimination())

        pop = initialization.do(self.problem, self.initial_pop_size)
        Evaluator().eval(self.problem, pop)

        # assign the population to each cluster
        cluster_nitching = ClusterNitching(self.n_obj, self.cluster_ref_dir['centroids'])
        niche_of_individuals = cluster_nitching.get_nitches(pop.get('F'))
        partial_pops = [pop[niche_of_individuals == k] for k in range(self.n_clusters)]

        t1 = time.time() - t0
        self.exec_times['time_ini_pop'] = t1

        if self.verbose > 1:
            print('Initial population time: {} s'.format(round(t1, 2)))

        self.ini_partial_pops = partial_pops
        self.cluster_nitching = cluster_nitching

    def initialize_parallel_algorithms(self, parallel_pop_size, n_parallel_gen):

        self.termination = get_termination('n_gen', n_parallel_gen)
        if self.verbose > 1:
            print('Initializing algorithms...')

        t0 = time.time()
        self.get_parallel_algorithms(parallel_pop_size, save_history=False)

        args = [(algo, self.problem, self.termination, p) for algo, p in zip(self.algorithms, self.ini_partial_pops)]
        self.algorithms = repeat_different_args(setup_algorithm, args, parallel=True, verbose=self.verbose)

        t1 = time.time() - t0
        self.exec_times['time_initialization_algos'] = t1

        if self.verbose > 1:
            print('Initialization time: {} s'.format(round(t1, 2)))

    def get_parallel_algorithms(self,
                                parallel_pop_size,
                                save_history=False):

        for k in range(self.n_clusters):
            if self.ref_dirs_with_neighbors:
                ref_dirs_partial = np.concatenate([self.ref_dirs[self.cluster_ref_dir['labels'] == k, :]] +
                                                  [self.ref_dirs[self.cluster_ref_dir['labels'] == n]
                                                   for n in self.cluster_ref_dir['data_structure']['neighbors'][k]
                                                   ], axis=0)
            else:
                ref_dirs_partial = self.ref_dirs[self.cluster_ref_dir['labels'] == k, :]

            # print(ref_dirs_partial.shape[0])
            algorithm = NSGA3_parallel(
                ref_dirs=ref_dirs_partial,
                clustering=self.cluster_ref_dir,
                strategy=self.strategy,
                cluster_id=k,
                pop_size=parallel_pop_size,
                sampling=get_sampling("real_random"),
                crossover=get_crossover("real_sbx", prob=0.9, eta=15),
                mutation=get_mutation("real_pm", eta=20),
                eliminate_duplicates=True,
                save_history=save_history,
                archive_type=self.archive_type,
                epsilon_archive=self.epsilon_archive,
                verbose=False,
            )
            self.algorithms.append(algorithm)
            self.ref_dirs_partial = ref_dirs_partial


    def run_parallel_algorithms(self, global_steps, parallel_pop_size, merge_archive=True):

        t0 = time.time()
        overhead = []
        for i in (tqdm(range(global_steps)) if self.verbose > 0 else range(global_steps)):
            partial_results = run_parallel_algos(self.algorithms, parallel=True, n_jobs=self.n_clusters, verbose=self.verbose)
            global_pop = np.concatenate([r.pop for r, a in partial_results]).view(Population)

            tg = time.time()
            if self.global_archive is not None:
                archives = np.concatenate([a for r, a in partial_results], axis=0).view(Population)
                self.global_archive.update(archives)

            # global_archive.update(global_pop)
            survival = self.rds.do(self.problem,
                                   global_pop,
                                   # n_survive=parallel_pop_size * self.n_clusters)
                                   n_survive=self.ref_dirs.shape[0])
            overhead.append(time.time() - tg)

            if merge_archive and self.global_archive is not None:
                global_pop = np.concatenate([survival, self.global_archive.archive]).view(Population)
                # global_pop = np.concatenate([global_pop, global_archive.archive]).view(Population)

            partial_pops = results_cluster_nitching(global_pop,
                                                    self.cluster_nitching,
                                                    self.n_clusters)

            if np.any(np.array([len(p) for p in partial_pops]) == 0):
                print('zero partial population')

            reset_algorithms(partial_pops, self.algorithms)

            hv_pop = get_hypervolume(global_pop.get('F'), self.hv_ref)
            self.hv_history.append(hv_pop)

        self.partial_results = partial_results
        self.pop = global_pop
        if self.global_archive is not None:
            self.archive = self.global_archive.archive
        self.opt = get_optimum(global_pop, self.ref_dirs)


        t1 = time.time() - t0
        self.exec_times['time_parallel_execution'] = t1
        self.exec_times['time_overhead'] = np.mean(overhead)

        if self.verbose > 1:
            print('Execution time: {} s'.format(round(t1, 2)))
