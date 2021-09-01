from numpy.random import randint
from numpy.random import rand
from functools import partial
from copy import copy
from src.models.ga.ga import GA
from src.models.ga.plot import plot_log
from src.optimization.functions.funcs import problems


def decode(bitstring, bounds, n_bits):
    decoded = list()
    largest = 2 ** n_bits
    for i in range(len(bounds)):
        # extraer la subcadena por variable
        start, end = i * n_bits, (i * n_bits) + n_bits
        substring = bitstring[start:end]
        # subcadena a cadena
        chars = ''.join([str(s) for s in substring])
        # cadena a entero
        integer = int(chars, 2)
        # escalar entero a rango
        value = bounds[i][0] + (integer / largest) * (bounds[i][1] - bounds[i][0])
        decoded.append(value)
    return decoded


def binary_cross_fn(c1, c2):
    c1_, c2_ = copy(c1), copy(c2)
    pt = randint(1, len(c1_) - 2)
    # realizar la cruza
    c1 = c1_[:pt] + c2_[pt:]
    c2 = c2_[:pt] + c1_[pt:]
    return [c1, c2]


def binary_mut_fn(bitstring, mx_pb=0.01):
    for i in range(len(bitstring)):
        # revisar probabilidad de mutación
        if rand() < mx_pb:
            # cambiar el bit
            bitstring[i] = 1 - bitstring[i]
    return bitstring


def binary_objective_fn(ind, fx, bounds, n_bits):
    return fx(decode(ind, bounds, n_bits))


def binary_ind_gen_fn(n_bits, bounds):
    return randint(0, 2, n_bits * len(bounds)).tolist()


def binary_create_funcs(fx, bounds, bits, mx_pb):
    decode_fn = partial(decode, bounds=bounds, n_bits=bits)
    objective_fn = partial(binary_objective_fn, fx=fx, bounds=bounds, n_bits=bits)
    ind_gen_fn = partial(binary_ind_gen_fn, n_bits=bits, bounds=bounds)
    mut_fn = partial(binary_mut_fn, mx_pb=mx_pb)
    return ind_gen_fn, objective_fn, mut_fn, decode_fn


def binary_ga(cfg, fx, bounds, bits):
    ind_gen_fn, obj_fn, mut_fn, decode_fn = binary_create_funcs(fx, bounds, bits, cfg['mx_pb'])
    ga = GA(ind_gen_fn, binary_cross_fn, mut_fn, obj_fn, cfg['selec'], verbose=0)
    ga.set_params(cfg['n_pop'], cfg['cx_pb'], cfg['mx_pb'], cfg['elitism_p'])
    return ga


def get_ga_binary_problem(name):
    fx, bounds, bits = problems[name]
    get_ga = partial(binary_ga, fx=fx, bounds=bounds, bits=bits)
    return get_ga

if __name__ == '__main__':
    cfg = {'n_gens': 100, 'n_pop': 500, 'cx_pb': 0.8, 'elitism_p': 0.02, 'mx_pb': 0.01,
           'selec': 'roullete'}
    name = 'himmelblau'
    fx, bounds, bits = problems[name]
    ind_gen_fn, obj_fn, mut_fn, decode_fn = binary_create_funcs(fx, bounds, bits, cfg['mx_pb'])
    ga = GA(ind_gen_fn, binary_cross_fn, mut_fn, obj_fn, cfg['selec'], verbose=2)
    ga.set_params(cfg['n_pop'], cfg['cx_pb'], cfg['mx_pb'], cfg['elitism_p'])
    best_ind, best_gen, best_eval, log = ga.run(cfg['n_gens'])

    print('%s @ %s gens: f(%s) = %f' % (name, best_gen, decode_fn(best_ind), best_eval))
    plot_log(log, title=name + ': F(x) vs Generation')