import copy
import numpy as np
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population, pop_from_array_or_individual
from pymoo.optimize import default_termination
from pymoo.util.misc import termination_from_tuple
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from src.models.moo.utils.plot import plot_multiple_pop


class ExternalArchive:

    def __init__(self, epsilon):
        self.archive = None
        self.epsilon = epsilon
        self.dim = None
        self.dim_set = False
        # self.nds = NonDominatedSorting(epsilon=epsilon)

    def update(self, pop):
        if not self.dim_set:
            F = pop.get('F')
            self.dim = F.shape[1]
            self.archive = np.empty((0,)).view(Population)
            self.dim_set = True
        self._update(pop)

    def _dominated_by(self, f, archive_F):
        if archive_F.shape[0] > 0:
            diff = ((archive_F - self.epsilon) - f).reshape(-1, self.dim)
            dominance = np.sum(diff < 0, axis=1) == self.dim
            different = np.sum(diff == 0, axis=1) != self.dim
            e_dominance = np.bitwise_and(dominance, different)

            return e_dominance
        else:
            return np.empty((0, self.dim))

    def _are_dominated(self, f, archive_F):
        if archive_F.shape[0] > 0:
            diff = (archive_F - f).reshape(-1, self.dim)
            dominance = np.sum(diff > 0, axis=1) == self.dim
            different = np.sum(diff == 0, axis=1) != self.dim
            dominates = np.bitwise_and(dominance, different)
            return dominates
        else:
            return np.empty((0, self.dim))

    def _update(self, F):
        pass


class ArchiveEps1(ExternalArchive):

    def __init__(self, epsilon=0.05, **kwargs):
        super().__init__(epsilon=epsilon, **kwargs)

    def _update(self, pop):
        archive_F = self.archive.get('F').reshape(-1, self.dim)
        F = pop.get('F').reshape(-1, self.dim)

        for ind, f in zip(pop, F):
            f = f.reshape(-1, self.dim)
            dominated_by_f = self._dominated_by(f, archive_F)
            if np.sum(dominated_by_f) > 0:
                continue

            f_dominates = self._are_dominated(f, archive_F)
            if f_dominates.shape[0] > 0:
                archive_F = np.delete(archive_F, f_dominates, axis=0)
                self.archive = np.delete(self.archive, f_dominates, axis=0).view(Population)

            archive_F = np.concatenate([archive_F, f], axis=0).reshape(-1, self.dim)
            self.archive = np.concatenate([self.archive, pop_from_array_or_individual(ind)], axis=0).view(Population)


class ArchiveEps2(ExternalArchive):

    def __init__(self, epsilon=0.05, **kwargs):
        super().__init__(epsilon=epsilon, **kwargs)

    def _update(self, pop):
        archive_F = self.archive.get('F').reshape(-1, self.dim)

        F = pop.get('F').reshape(-1, self.dim)
        for ind, f in zip(pop, F):
            added = False
            f = f.reshape(-1, self.dim)
            dominated_by_f = self._dominated_by(f, archive_F)
            if np.sum(dominated_by_f) == 0:
                archive_F = np.concatenate([archive_F, f], axis=0).reshape(-1, self.dim)
                self.archive = np.concatenate([self.archive, pop_from_array_or_individual(ind)], axis=0).view(
                    Population)
                added = True

            f_dominates = self._are_dominated(f, archive_F)
            if np.sum(f_dominates) > 0:
                archive_F = np.delete(archive_F, f_dominates, axis=0)
                self.archive = np.delete(self.archive, f_dominates, axis=0).view(Population)
                if not added:
                    archive_F = np.concatenate([archive_F, f], axis=0).reshape(-1, self.dim)
                    self.archive = np.concatenate([self.archive, pop_from_array_or_individual(ind)], axis=0).view(
                        Population)


class GeneticAlgorithmWithArchive(GeneticAlgorithm):

    def __init__(self, archive_type, epsilon_archive, *args, **kwargs):

        archive_options = ['eps1', 'eps2']
        if archive_type is not None and archive_type not in archive_options:
            raise Exception('{} not a valid option. Options: {}'.format(archive_type, archive_options))

        if archive_type:
            if archive_type == 'eps1':
                self.archive = ArchiveEps1(epsilon=epsilon_archive)
            elif archive_type == 'eps2':
                self.archive = ArchiveEps2(epsilon=epsilon_archive)
        else:
            self.archive = None

        super().__init__(*args, **kwargs)

    def _advance(self, infills=None, **kwargs):

        # merge the offsprings with the current population
        if infills is not None:
            self.pop = Population.merge(self.pop, infills)

        if self.archive is not None:
            self.archive.update(self.pop)

        # execute the survival to find the fittest solutions
        self.pop = self.survival.do(self.problem, self.pop, n_survive=self.pop_size, algorithm=self)


def minimize_problem(problem,
                     algorithm,
                     termination=None,
                     copy_algorithm=True,
                     copy_termination=True,
                     **kwargs):
    # create a copy of the algorithm object to ensure no side-effects
    if copy_algorithm:
        algorithm = copy.deepcopy(algorithm)

    # initialize the algorithm object given a problem - if not set already
    if algorithm.problem is None:
        algorithm.setup(problem, termination=termination, **kwargs)
    else:
        if termination is not None:
            algorithm.termination = termination_from_tuple(termination)

    # if no termination could be found add the default termination either for single or multi objective
    termination = algorithm.termination
    if termination is None:
        termination = default_termination(problem)

    if copy_termination:
        termination = copy.deepcopy(termination)
    algorithm.termination = termination

    # actually execute the algorithm
    res = algorithm.run()

    # store the deep copied algorithm in the result object
    res.algorithm = algorithm

    if hasattr(algorithm, 'archive'):
        print(len(algorithm.archive.archive))
        archive = copy.deepcopy(algorithm.archive.archive)
    else:
        archive = None

    return res, archive


if __name__ == '__main__':
    # %%
    epsilon = 0.05
    dim = 3

    external_archive = ArchiveEps2(dim)
    pop = np.random.random((50, dim))
    pop2 = np.random.random((50, dim))

    external_archive.update(pop)
    archive1 = copy.copy(external_archive.archive)
    plot_multiple_pop([pop, archive1],
                      opacities=[.2, 1])

    external_archive.update(pop2)
    archive2 = copy.copy(external_archive.archive)
    plot_multiple_pop([archive1, archive2],
                      opacities=[.2, 1])
