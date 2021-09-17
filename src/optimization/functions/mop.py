from math import sqrt, cos, pi, sin

import numpy as np

zdt_bounds = [[0, 1]]


def g1(x):
    return 1 + (9 / (len(x) - 1)) * sum(x[1:])


def zdt1(x):
    f1 = x[0]
    f2 = g1(x) * (1 - sqrt(f1 / g1(x)))
    return [f1, f2]


def zdt2(x):
    f1 = x[0]
    f2 = g1(x) * (1 - (f1 / g1(x)) ** 2)
    return [f1, f2]


def dtlz2(x):
    x = np.array(x)
    g = np.sum(np.power(x[2:] - 0.5, 2))
    f1 = cos(x[0] * pi / 2) * cos(x[1] * pi / 2) * (1 + g)
    f2 = cos(x[0] * pi / 2) * sin(x[1] * pi / 2) * (1 + g)
    f3 = sin(x[0] * pi / 2) * (1 + g)
    return [f1, f2, f3]


funcs = [zdt1, zdt2]


def zdt(f, dim):
    return funcs[f - 1], zdt_bounds * dim
