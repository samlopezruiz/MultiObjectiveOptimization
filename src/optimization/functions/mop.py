from math import sqrt, cos, pi, sin

import numpy as np

zdt_bounds = [[0, 1]]


def g1(x):
    return 1 + (9 / (len(x) - 1)) * sum(x[1:])


def g2(self, X_M):
    return np.sum(np.square(X_M - 0.5), axis=1)


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


def dtlz1(x):
    x = np.array(x)
    g = 100 * (10 + np.sum(np.power(x[2:] - 0.5, 2) - np.cos(20 * pi * (x[2:] - 0.5))))
    f1 = 0.5 * x[0] * x[1] * (1 + g)
    f2 = 0.5 * x[0] * (1 - x[1]) * (1 + g)
    f3 = 0.5 * (1 - x[0]) * (1 + g)
    return [f1, f2, f3]


funcs = [zdt1, zdt2]


def zdt(f, dim):
    return funcs[f - 1], zdt_bounds * dim
