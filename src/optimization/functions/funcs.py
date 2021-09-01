import numpy as np
from math import sin, sqrt, pi, cos

# rango de las variables
import torch

beale_bounds = [[-4.5, 4.5]] * 2
himmelblau_bounds = [[-5.0, 5.0]] * 2
eggholder_bounds = [[-512.0, 512.0]] * 2
rastrigin_bounds = [[-5.12, 5.12]] * 4
f1_t1_bounds = [[-10.0, 10.0]] * 2
f2_t1_bounds = [[-500.0, 500.0]] * 2


# funciones objetivo
def beale(x):
    return (1.5 - x[0] + x[0] * x[1]) ** 2 + (2.25 - x[0] + x[0] * (x[1] ** 2)) ** 2 + \
           (2.625 - x[0] + x[0] * (x[1] ** 3)) ** 2


def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def eggholder(x):
    return -(x[1] + 47) * sin(sqrt(abs(x[0] / 2 + x[1] + 47))) - x[0] * \
           sin(sqrt(abs(x[0] - (x[1] + 47))))


def rastrigin(x):
    n = len(x)
    x = np.array(x)
    return 10 * n + np.sum(np.power(x, 2) - 10 * np.cos(2 * pi * x))


def f1_t1(x):
    return x[0] ** 2 + x[1] ** 2


def f2_t1(x):
    return 418.9829 * 2 - x[0] * sin(sqrt(abs(x[0]))) - x[1] * sin(sqrt(abs(x[1])))


def f2_t1_torch(x):
    # se convierten a tensores los componentes de 'x' que no lo son
    for i in range(len(x)):
        if not isinstance(x[i], torch.Tensor):
            x[i] = torch.tensor(x[i])

    return 418.9829 * 2 - x[0] * torch.sin(torch.sqrt(torch.abs(x[0]))) - x[1] * torch.sin(torch.sqrt(torch.abs(x[1])))

def gf1(X):
    return np.array([2. * X[0], 2. * X[1]])


def gf2(X):
    return np.array([(-X[0] ** 2 * cos(sqrt(abs(X[0])))) / (2 * abs(X[0] * sqrt(abs(X[0])))) - sin(sqrt(abs(X[0]))),
                     (-X[1] ** 2 * cos(sqrt(abs(X[1])))) / (2 * abs(X[1] * sqrt(abs(X[1])))) - sin(sqrt(abs(X[1])))])


# Diccionario con los problemas
bounds = [beale_bounds, himmelblau_bounds, eggholder_bounds, rastrigin_bounds, f1_t1_bounds, f2_t1_bounds, f2_t1_bounds]
functions = [beale, himmelblau, eggholder, rastrigin, f1_t1, f2_t1, f2_t1_torch]
n_bits = [12, 12, 12, 8, 8, 8, 8]
names = ['beale', 'himmelblau', 'eggholder', 'rastrigin', 'f1_t1', 'f2_t1', 'f2_t1_torch']

problems = {}
for i in range(len(bounds)):
    problems[names[i]] = [functions[i], bounds[i], n_bits[i]]
