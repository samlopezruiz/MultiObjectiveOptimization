import time
import torch
import numpy as np
from math import sin, sqrt, pi, cos


def gradient_descent_torch(x0, fx, n_iter, alpha, epsilon=0.0001, early_break=True):
    # se crea una solución inicial x0 usando tensores
    x_tensor = [torch.tensor(xi, requires_grad=True) for xi in x0]

    # se almacenan las soluciones 'xs', las evaluaciones f(x) 'fxs',
    # los gradientes 'grads' a lo largo de las iteraciones
    xs, fxs, grads = [], [], []

    for i in range(n_iter):
        # se realiza la auto diferenciación de la función
        f = fx(x_tensor)
        f.backward()
        grads.append([xi.grad.detach().numpy().item() for xi in x_tensor])

        # se actualiza la solución usando el gradiente calculado
        with torch.no_grad():
            x_tensor = [xi - alpha * xi.grad for xi in x_tensor]
        for xi in x_tensor:
            xi.requires_grad = True

        # se almacena x y f(x)
        xs.append([xi.detach().numpy().item() for xi in x_tensor])
        fxs.append(f.detach().numpy().item())

        # en caso de no haber disminución en la evaluación de la función
        # usando un umbral = epsilon, se detiene prematuramente el bucle
        if i > 2 and early_break:
            if abs(fxs[-1] - fxs[-2]) < epsilon:
                break

    return xs, fxs, grads


def gradient_descent(x0, fx, gfx, n_iter, alpha_n=2, epsilon=0.0001, early_break=True):
    x = np.array(x0)
    # se almacenan las soluciones 'xs', las evaluaciones f(x) 'fxs',
    # los gradientes 'grads' y las alphas 'alphas' a lo largo de las iteraciones
    xs, fxs, grads, alphas = [], [], [], []
    for i in range(n_iter):
        gdx = gfx(x)
        grads.append(gdx)

        # se calcula un tamaño de paso óptimo usando backtracking
        alpha = backtrack_alpha(fx, gfx, -gdx, x, alpha_n, c=0.0001, rho=0.9)

        # se actualiza la solución usando el gradiente
        x = x - alpha * np.array(gfx(x))

        # se almacena x, f(x), y alpha
        xs.append(list(x))
        fxs.append(fx(x))
        alphas.append(alpha)

        # en caso de no haber disminución en la evaluación de la función
        # usando un umbral = epsilon, se detiene prematuramente el bucle
        if i > 2 and early_break:
            if abs(fxs[-1] - fxs[-2]) < epsilon:
                break

    return xs, fxs, grads, alphas




def backtrack_alpha(fx, gfx, p_k, x_k, alpha_n, c=0.0001, rho=0.9):
    '''
    :param fx: función objetivo
    :param gfx: derivada de la función objetivo
    :param p_k: gradiente negativo de la función en x
    :param x_k: punto a evaluar
    :param alpha_n: tamaño de paso inicial
    :param c: constante
    :param rho: decremento para el tamaño de paso
    :return: el tamaño de paso que garantiza una disminución en la evaluación f(x)
    '''
    alpha = alpha_n
    while fx(x_k + alpha * p_k) > fx(x_k) + c * alpha * gfx(x_k).dot(p_k):
        alpha = rho * alpha

    return alpha

