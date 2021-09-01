import numpy as np
from copy import copy, deepcopy

import pandas as pd
from numpy.random import rand
from functools import partial
from src.models.ga.selection_func import selection_sus, selection_tournament, selection_roullete


def stats(data):
    return round(min(data), 2), round(max(data), 2), round(np.mean(data), 2)


def get_ix(scores, n_elitism, reverse):
    ix = []
    unique_scores = np.unique(scores)
    unique_scores = np.flip(unique_scores) if reverse else unique_scores

    for score in unique_scores:
        ix += list(np.where(np.array(scores) == score)[0])
        if len(ix) > n_elitism:
            break
    return ix


def log_to_df(log):
    stat = pd.DataFrame(log, columns=['gen', 'best_ind', 'best', 'worst', 'mean'])
    stat.set_index('gen', inplace=True)
    return stat


class GA:
    def __init__(self, ind_gen_fn, cross_fn, mut_fn, objective_fn, selec='tournament',
                 min_prob=True, verbose=1, tour_size=5, early_stop=(False, 0), tol=0.01):
        selec_methods = ['sus', 'tournament', 'roullete']
        if selec not in selec_methods:
            print('Selection method must be one of: {}'.format(selec_methods))
        self.log = []
        self.population = []
        self.tol = tol
        self.best_gen = 0
        self.tour_size = tour_size
        self.selec_fn = selec
        self.ind_gen_fn = ind_gen_fn
        self.cross_fn = cross_fn
        self.mut_fn = mut_fn
        self.objective_fn = objective_fn
        self.min_prob = min_prob
        self.best_ind = None
        self.best_eval = None
        self.verbose = verbose
        self.early_stop = early_stop
        self.gen = 0
        self.set_params()
        self.set_selec_fn()

    def set_selec_fn(self):
        if self.selec_fn == 'sus':
            self.selection = selection_sus
        elif self.selec_fn == 'tournament':
            self.selection = partial(selection_tournament, tour_size=self.tour_size)
        elif self.selec_fn == 'roullete':
            self.selection = selection_roullete

    def set_params(self, n_pop=100, cx_pb=0.6, mx_pb=1, elitism_p=0.01):
        self.n_pop = n_pop
        self.cx_pb = cx_pb
        self.mx_pb = mx_pb
        self.elitism_p = elitism_p

    def run(self, n_gens):
        self.algorithm(n_gens)
        solution = {'best_ind': self.best_ind, 'best_gen': self.best_gen, 'best_eval': self.best_eval,
                    'log': log_to_df(self.log)}
        # return self.best_ind, self.best_gen, self.best_eval, log_to_df(self.log)
        return solution

    def algorithm(self, n_gens):
        self.initialize()
        for _ in range(n_gens):
            self.step()
            if self.early_stop[0] and self.best_eval <= self.early_stop[1] * (1 + self.tol):
                return

    def step(self):
        self.gen += 1
        scores = self.score_pop(self.population)
        best_s, worst_s, mean_s = stats(scores)
        self.log.append((self.gen, copy(self.best_ind), best_s, worst_s, mean_s))
        self.print_progress()
        self.replace_best_individual(self.population, scores)
        selected = self.selection(self.population, scores)
        children = self.cross_and_mut(selected)
        children = self.apply_elitism(children, self.population, scores)
        self.population = children

    def initialize(self):
        self.population = self.generate_population()
        self.best_ind, self.best_eval = None, self.objective_fn(self.population[0])
        self.log = []
        self.gen = 0

    def apply_elitism(self, children, pop, scores):
        if self.elitism_p > 0:
            n_elitism = round(self.n_pop * self.elitism_p)
            # sorted_score = sorted(scores)
            children_scores = self.score_pop(children)
            # ordenar en orden descendiente a los hijos
            # children_sorted_scores = sorted(children_scores, reverse=True)

            # reemplaza los peores n individuos por los mejores n individuos
            # de la población inicial (elitismo)
            ix_pop = get_ix(scores, n_elitism, reverse=False)
            ix_child = get_ix(children_scores, n_elitism, reverse=True)

            for i in range(n_elitism):
                children[ix_child[i]] = copy(pop[ix_pop[i]])

            return children
        else:
            return children
        pass

    def cross_and_mut(self, selected):
        children = list()
        for i in range(0, self.n_pop, 2):
            p1, p2 = selected[i], selected[i + 1]
            for c in self.crossover(p1, p2):
                self.mutation(c)
                children.append(c)
        return children

    def mutation(self, ind):
        if rand() < self.mx_pb:
            return self.mut_fn(ind)
        else:
            return ind

    def crossover(self, p1, p2):
        c1, c2 = copy(p1), copy(p2)
        if rand() < self.cx_pb:
            c1, c2 = self.cross_fn(c1, c2)
        return [c1, c2]

    def generate_population(self):
        population = [self.ind_gen_fn() for _ in range(self.n_pop)]
        return population

    def print_progress(self):
        if self.verbose > 2:
            gen, best_ind, best_s, worst_s, mean_s = self.log[-1]
            print('{}, best: {}, worst:{}, avg: {}'.format(gen, best_s, worst_s, mean_s))

    def score_pop(self, pop):
        return [self.objective_fn(p) for p in pop]

    def replace_best_individual(self, pop, scores):
        for i in range(self.n_pop):
            if scores[i] < self.best_eval:
                self.best_ind, self.best_eval, self.best_gen = copy(pop[i]), scores[i], self.gen
                if self.verbose > 1:
                    print(">%d, new best f(x) = %f" % (self.gen, scores[i]))
