import numpy as np
from numpy.random import randint
from numpy.random import rand
from random import choice
from functools import partial
from copy import copy
from src.models.ga.ga import GA
from src.models.ga.plot import plot_log
from src.optimization.functions.funcs import problems


def real_cross_fn(c1, c2, bounds, op=1, isint=False):
    z1, z2 = [], []
    i = 0
    for xi, yi in zip(c1, c2):
        if op == 0:
            z1i = xi if rand() > 0.5 else yi
            z2i = xi if rand() > 0.5 else yi
        elif op == 1:
            z1i = xi + (rand() * (1.25 + 0.25) - 0.25) * (yi - xi)
            z2i = xi + (rand() * (1.25 + 0.25) - 0.25) * (yi - xi)
        else:
            z1i = xi + choice([-0.25, 1.25]) * (yi - xi)
            z2i = xi + choice([-0.25, 1.25]) * (yi - xi)
        z1i = min(max(z1i, bounds[i][0]), bounds[i][1])
        z2i = min(max(z2i, bounds[i][0]), bounds[i][1])
        z1.append(int(round(z1i)) if isint else z1i)
        z2.append(int(round(z2i)) if isint else z2i)
        i += 1
    return [z1, z2]


def sbx_cross_fn(c1, c2, bounds, isint=False, eta=1):
    z1, z2 = [], []
    i = 0
    mu = rand()
    if mu < 0.5:
        beta = np.power(2 * mu, 1 / (eta + 1.0))
    else:
        beta = np.power((1.0 / (2.0 - mu)), (1.0 / (eta + 1.0)))

    for xi, yi in zip(c1, c2):
        z1i = 0.5 * (xi + yi) - beta * (yi - xi)
        z2i = 0.5 * (xi + yi) + beta * (yi - xi)
        z1i = min(max(z1i, bounds[i][0]), bounds[i][1])
        z2i = min(max(z2i, bounds[i][0]), bounds[i][1])
        z1.append(int(round(z1i)) if isint else z1i)
        z2.append(int(round(z2i)) if isint else z2i)
        i += 1
    return [z1, z2]


def real_mut_fn(x, bounds, mx_pb=None, isint=False):
    if mx_pb is None:
        mx_pb = 1 / len(x)

    z = []
    for i, xi in enumerate(x):
        if rand() < mx_pb:
            mut_range = 0.1 * (bounds[i][1] - bounds[i][0])
            delta = sum([(1 if rand() < 1 / 16 else 0) * 2 ** (-j) for j in range(16)])
            zi = xi + mut_range * delta if rand() > 0.5 else xi - mut_range * delta
            zi = min(max(zi, bounds[i][0]), bounds[i][1])
            z.append(int(round(zi)) if isint else zi)
        else:
            z.append(xi)
    return z


def real_poly_mut_fn(x, bounds, mx_pb=None, isint=False, eta=50):
    if mx_pb is None:
        mx_pb = 1 / len(x)
    z = []
    for i, xi in enumerate(x):
        if rand() < mx_pb:
            mu = rand()
            if mu <= 0.5:
                delta = np.power(2 * mu, 1 / (1 + eta)) - 1
                zi = xi + delta * (xi - bounds[i][0])
            else:
                delta = 1 - np.power(2 * (1 - mu), 1 / (1 + eta))
                zi = xi + delta * (bounds[i][1] - xi)
            zi = min(max(zi, bounds[i][0]), bounds[i][1])
            z.append(int(round(zi)) if isint else zi)
        else:
            z.append(xi)
    return z

def real_ind_gen_fn(bounds, isint=False):
    x = []
    for bound in bounds:
        if isint:
            x.append(int(round(rand() * (bound[1] - bound[0]) + bound[0])))
        else:
            x.append(rand() * (bound[1] - bound[0]) + bound[0])
    return x


def real_objective_fn(ind, fx):
    return fx(ind)


def real_create_funcs(fx, bounds, mx_pb, op=1, isint=False):
    obj_fn = partial(real_objective_fn, fx=fx)
    ind_gen_fn = partial(real_ind_gen_fn, bounds=bounds, isint=isint)
    mut_fn = partial(real_mut_fn, mx_pb=mx_pb, bounds=bounds, isint=isint)
    cross_fn = partial(real_cross_fn, op=op, bounds=bounds, isint=isint)
    return ind_gen_fn, obj_fn, mut_fn, cross_fn


def real_create_sbx_funcs(fx, bounds, mx_pb, eta=1, isint=False):
    obj_fn = partial(real_objective_fn, fx=fx)
    ind_gen_fn = partial(real_ind_gen_fn, bounds=bounds, isint=isint)
    mut_fn = partial(real_poly_mut_fn, mx_pb=mx_pb, bounds=bounds, isint=isint)
    cross_fn = partial(sbx_cross_fn, eta=eta, bounds=bounds, isint=isint)
    return ind_gen_fn, obj_fn, mut_fn, cross_fn


def real_ga(cfg, fx, bounds, op=1, isint=False):
    ind_gen_fn, obj_fn, mut_fn, cross_fn = real_create_funcs(fx, bounds, cfg['mx_pb'], op, isint)
    ga = GA(ind_gen_fn, cross_fn, mut_fn, obj_fn, cfg['selec'], early_stop=(cfg.get('early_stop', False),
                                                                            cfg.get('target', 0)), verbose=0)
    ga.set_params(cfg['n_pop'], cfg['cx_pb'], cfg['mx_pb'], cfg['elitism_p'])
    return ga


def get_ga_contf_problem(name, op, isint=False):
    fx, bounds, bits = problems[name]
    bounds=bounds[:2]
    get_ga = partial(real_ga, fx=fx, bounds=bounds, op=op, isint=isint)
    return get_ga


if __name__ == '__main__':
    cfg = {'n_gens': 100, 'n_pop': 200, 'cx_pb': 0.5, 'elitism_p': 0.02, 'mx_pb': 0.8,
           'selec': 'tournament'}
    name = 'eggholder'
    fx, bounds, bits = problems[name]
    ind_gen_fn, obj_fn, mut_fn, cross_fn = real_create_funcs(fx, bounds, cfg['mx_pb'], op=1, isint=True)
    ga = GA(ind_gen_fn, cross_fn, mut_fn, obj_fn, cfg['selec'], verbose=2)
    ga.set_params(cfg['n_pop'], cfg['cx_pb'], 1, cfg['elitism_p'])
    best_ind, best_gen, best_eval, log = ga.run(cfg['n_gens'])

    print('%s @ %s gens: f(%s) = %f' % (name, best_gen, best_ind, best_eval))
    plot_log(log, title=name + ': F(x) vs Generation')