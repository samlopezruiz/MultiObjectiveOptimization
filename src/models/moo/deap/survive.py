import numpy as np
from pymoo.algorithms.nsga3 import get_extreme_points_c, get_nadir_point, \
    associate_to_niches, calc_niche_count, niching
from pymoo.util.misc import intersect
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


class MyReferenceDirectionSurvival():

    def __init__(self, ref_dirs):
        self.ref_dirs = ref_dirs
        self.extreme_points = None
        self.intercepts = None
        self.nadir_point = None
        self.opt = None
        self.ideal_point = np.full(ref_dirs.shape[1], np.inf)
        self.worst_point = np.full(ref_dirs.shape[1], -np.inf)

    def do(self, pop, n_survive, D=None, **kwargs):
        # attributes to be set after the survival
        F = np.array([ind.fitness.values for ind in pop])

        # find or usually update the new ideal point - from feasible solutions
        self.ideal_point = np.min(np.vstack((self.ideal_point, F)), axis=0)
        self.worst_point = np.max(np.vstack((self.worst_point, F)), axis=0)

        # calculate the fronts of the population
        fronts, rank = NonDominatedSorting().do(F, return_rank=True, n_stop_if_ranked=n_survive)
        non_dominated, last_front = fronts[0], fronts[-1]

        # find the extreme points for normalization
        self.extreme_points = get_extreme_points_c(F[non_dominated, :], self.ideal_point,
                                                   extreme_points=self.extreme_points)

        # find the intercepts for normalization and do backup if gaussian elimination fails
        worst_of_population = np.max(F, axis=0)
        worst_of_front = np.max(F[non_dominated, :], axis=0)

        self.nadir_point = get_nadir_point(self.extreme_points, self.ideal_point, self.worst_point,
                                           worst_of_population, worst_of_front)

        #  consider only the population until we come to the splitting front
        I = np.concatenate(fronts)
        rank, F = rank[I], F[I]
        pop = [pop[i] for i in I]
        # update the front indices for the current population
        counter = 0
        for i in range(len(fronts)):
            for j in range(len(fronts[i])):
                fronts[i][j] = counter
                counter += 1
        last_front = fronts[-1]

        # associate individuals to niches
        niche_of_individuals, dist_to_niche, dist_matrix = \
            associate_to_niches(F, self.ref_dirs, self.ideal_point, self.nadir_point)

        # attributes of a population
        for i, ind in enumerate(pop):
            ind.fitness.rank = rank[i]
            ind.fitness.niche = niche_of_individuals[i]
            ind.fitness.dist_to_niche = dist_to_niche[i]

        # set the optimum, first front and closest to all reference directions
        closest = np.unique(dist_matrix[:, np.unique(niche_of_individuals)].argmin(axis=0))
        self.opt = [pop[i] for i in intersect(fronts[0], closest)]

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

            pop_last_front = [pop[i] for i in last_front]
            S = niching(pop_last_front, n_remaining, niche_count, niche_of_individuals[last_front],
                        dist_to_niche[last_front])

            survivors = np.concatenate((until_last_front, last_front[S].tolist()))
            pop = [pop[i] for i in survivors]

        return pop