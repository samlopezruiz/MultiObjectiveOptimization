import random

import numpy as np
from numpy.random import randint
from numpy.random import rand
from random import choice
from functools import partial

from copy import copy, deepcopy

from src.models.ga.codification.real import real_objective_fn
from src.models.ga.ga import GA
from src.models.ga.plot import plot_log
from src.optimization.functions.funcs import problems


def perm_cross_fn(p1, p2):
    c1, c2 = deepcopy(p1), deepcopy(p2)
    pos = sorted(random.sample(range(1, len(p1)), 2))
    extract_p1 = p1[pos[0]:pos[1]]
    extract_p2 = p2[pos[0]:pos[1]]
    c2[pos[0]:pos[1]] = extract_p1
    c1[pos[0]:pos[1]] = extract_p2

    for i in range(len(p1)):
        if i < pos[0] or i >= pos[1]:
            while c1[i] in extract_p2:
                c1[i] = extract_p1[np.argwhere(extract_p2 == c1[i])[0][0]]
            while c2[i] in extract_p1:
                c2[i] = extract_p2[np.argwhere(extract_p1 == c2[i])[0][0]]
    return [c1, c2]


def perm_mut_fn(x):
    ix = random.sample(range(1, len(x)), 2)
    x[ix[0]], x[ix[1]] = x[ix[1]], x[ix[0]]
    return x


def perm_ind_gen_fn(n):
    return np.random.permutation(n)


def perm_create_funcs(fx, n):
    obj_fn = partial(real_objective_fn, fx=fx)
    ind_gen_fn = partial(perm_ind_gen_fn, n=n)
    mut_fn = perm_mut_fn
    cross_fn = perm_cross_fn
    return ind_gen_fn, obj_fn, mut_fn, cross_fn


if __name__ == '__main__':
    cfg = {'n_gens': 100, 'n_pop': 200, 'cx_pb': 0.5, 'elitism_p': 0.02, 'mx_pb': 0.8,
           'selec': 'tournament'}
    name = 'eggholder'
    fx, bounds, bits = problems[name]
    ind_gen_fn, obj_fn, mut_fn, cross_fn = perm_create_funcs(fx, n=8)
    ga = GA(ind_gen_fn, cross_fn, mut_fn, obj_fn, cfg['selec'], verbose=2)
    ga.set_params(cfg['n_pop'], cfg['cx_pb'], cfg['mx_pb'], cfg['elitism_p'])
    best_ind, best_gen, best_eval, log = ga.run(cfg['n_gens'])

    print('%s @ %s gens: f(%s) = %f' % (name, best_gen, best_ind, best_eval))
    plot_log(log, title=name + ': F(x) vs Generation')