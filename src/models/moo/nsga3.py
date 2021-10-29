import warnings

import numpy as np
from numpy.linalg import LinAlgError

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.moo.nsga3 import comp_by_cv_then_random, HyperplaneNormalization, associate_to_niches, \
    calc_niche_count, niching, ReferenceDirectionSurvival
from pymoo.docs import parse_doc_string
from pymoo.core.survival import Survival
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection, compare
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.function_loader import load_function
from pymoo.util.misc import intersect, has_feasible
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from src.models.moo.external_archive import GeneticAlgorithmWithArchive


class NSGA3_parallel(GeneticAlgorithmWithArchive):

    def __init__(self,
                 ref_dirs,
                 clustering,
                 strategy,
                 cluster_id,
                 pop_size=None,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(func_comp=comp_by_cv_then_random),
                 crossover=SimulatedBinaryCrossover(eta=30, prob=1.0),
                 mutation=PolynomialMutation(eta=20, prob=None),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 display=MultiObjectiveDisplay(),
                 archive_type=None,
                 epsilon_archive=None,
                 **kwargs):
        """

        Parameters
        ----------

        ref_dirs : {ref_dirs}
        pop_size : int (default = None)
            By default the population size is set to None which means that it will be equal to the number of reference
            line. However, if desired this can be overwritten by providing a positive number.
        sampling : {sampling}
        selection : {selection}
        crossover : {crossover}
        mutation : {mutation}
        eliminate_duplicates : {eliminate_duplicates}
        n_offsprings : {n_offsprings}

        """
        options = ['kill_outside_cluster', 'keep_outside_cluster']
        if strategy not in options:
            raise Exception('strategy "{}" is not a valid option. Options: {}'.format(strategy, options))

        self.clustering = clustering
        self.ref_dirs = ref_dirs
        self.cluster_id = cluster_id

        # in case of R-NSGA-3 they will be None - otherwise this will be executed
        if self.ref_dirs is not None:

            if pop_size is None:
                pop_size = len(self.ref_dirs)

            if pop_size < len(self.ref_dirs):
                pass
                # print(
                #     f"WARNING: pop_size={pop_size} is less than the number of reference directions ref_dirs={len(self.ref_dirs)}.\n"
                #     "This might cause unwanted behavior of the algorithm. \n"
                #     "Please make sure pop_size is equal or larger than the number of reference directions. ")

        if 'survival' in kwargs:
            survival = kwargs['survival']
            del kwargs['survival']
        else:
            survival = ReferenceDirectionSurvivalCluster(ref_dirs, clustering, cluster_id, strategy)

        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=survival,
                         eliminate_duplicates=eliminate_duplicates,
                         n_offsprings=n_offsprings,
                         display=display,
                         advance_after_initial_infill=True,
                         archive_type=archive_type,
                         epsilon_archive=epsilon_archive,
                         **kwargs)

    def _setup(self, problem, **kwargs):

        if self.ref_dirs is not None:
            if self.ref_dirs.shape[1] != problem.n_obj:
                raise Exception(
                    "Dimensionality of reference points must be equal to the number of objectives: %s != %s" %
                    (self.ref_dirs.shape[1], problem.n_obj))

    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            self.opt = self.survival.opt

class ReferenceDirectionSurvivalCluster(Survival):

    def __init__(self, ref_dirs, clustering, cluster_id, strategy):
        super().__init__(filter_infeasible=True)
        self.ref_dirs = ref_dirs
        self.opt = None
        self.norm = HyperplaneNormalization(ref_dirs.shape[1])
        self.clustering = clustering
        self.cluster_id = cluster_id
        self.strategy = strategy

    def _do(self, problem, pop, n_survive, D=None, **kwargs):

        # attributes to be set after the survival
        F = pop.get("F")

        cluster_classification = np.array([self.clustering['clustering'].predict(f.reshape(1, -1))[0] for f in F])


        if self.strategy == 'keep_outside_cluster':
            pass
        else:
            inside_cluster = cluster_classification == self.cluster_id
            outside_cluster = cluster_classification != self.cluster_id

            if self.strategy == 'kill_outside_cluster':
                F = F[inside_cluster, Ellipsis]
            elif self.strategy == 'keep_outside_cluster':
                pass
            elif self.strategy == 'last_front_outside_cluster':
                F_out = F[outside_cluster, :]
                F = F[inside_cluster, :]


        # calculate the fronts of the population
        fronts, rank = NonDominatedSorting().do(F, return_rank=True, n_stop_if_ranked=n_survive)
        non_dominated, last_front = fronts[0], fronts[-1]

        if self.strategy == 'last_front_outside_cluster':
            last_front = np.concatenate([last_front, F_out], axis=0)

        # update the hyperplane based boundary estimation
        hyp_norm = self.norm
        hyp_norm.update(F, nds=non_dominated)
        ideal, nadir = hyp_norm.ideal_point, hyp_norm.nadir_point

        #  consider only the population until we come to the splitting front
        I = np.concatenate(fronts)
        pop, rank, F = pop[I], rank[I], F[I]

        # update the front indices for the current population
        counter = 0
        for i in range(len(fronts)):
            for j in range(len(fronts[i])):
                fronts[i][j] = counter
                counter += 1
        last_front = fronts[-1]

        # associate individuals to niches
        niche_of_individuals, dist_to_niche, dist_matrix = \
            associate_to_niches(F, self.ref_dirs, ideal, nadir)

        # attributes of a population
        pop.set('rank', rank,
                'niche', niche_of_individuals,
                'dist_to_niche', dist_to_niche)

        # set the optimum, first front and closest to all reference directions
        closest = np.unique(dist_matrix[:, np.unique(niche_of_individuals)].argmin(axis=0))
        self.opt = pop[intersect(fronts[0], closest)]

        # if we need to select individuals to survive
        if len(pop) > n_survive:

            # if there is only one front
            if len(fronts) == 1:
                n_remaining = n_survive
                until_last_front = np.array([], dtype=int)
                niche_count = np.zeros(len(self.ref_dirs), dtype=int)

            # if some individuals already survived
            else:
                until_last_front = np.concatenate(fronts[:-1])
                niche_count = calc_niche_count(len(self.ref_dirs), niche_of_individuals[until_last_front])
                n_remaining = n_survive - len(until_last_front)

            S = niching(pop[last_front], n_remaining, niche_count, niche_of_individuals[last_front],
                        dist_to_niche[last_front])

            survivors = np.concatenate((until_last_front, last_front[S].tolist()))
            pop = pop[survivors]

        return pop
