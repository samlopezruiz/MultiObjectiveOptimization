import numpy as np

from src.optimization.plot.func import plot_fx


def f(X):
    return X[0] ** 2 + X[1] ** 2 + X[2] ** 2


def gf(X):
    return np.array([2. * X[0], 2. * X[1], 2. * X[2]])


# %%
def backtrack_alpha(p_k, x_k, alpha_n, c=0.0001, rho=0.9):
    alpha = alpha_n
    while f(x_k + alpha * p_k) > f(x_k) + c * alpha * gf(x_k).dot(p_k):
        # fx = f(x_k + alpha * p_k)
        # b = f(x_k) + c * alpha * gf(x_k).dot(p_k)
        # print('{} <= {}'.format(round(fx, 4), round(b, 4)))
        alpha = rho * alpha

    return alpha

import seaborn as sns
sns.set_context("poster")

if __name__ == '__main__':
    #%%
    alpha_n = 2
    c = 0.0001
    x_k = np.array((2., 2., 2.))
    p_k = - gf(x_k)

    fxs, alphas = [], []
    for i in range(20):
        p_k = - gf(x_k)
        # print('p_k = {}'.format(str(p_k)))
        alpha = backtrack_alpha(p_k, x_k, alpha_n, rho=0.9)
        alphas.append(alpha)
        fxs.append(f(x_k))
        # print('alpha = {}'.format(str(alpha)))
        x_k = x_k + alpha * p_k
        # print('x_k = {}'.format(str(x_k)))
        # print('f({}) = {}'.format(str(x_k), str(f(x_k))))

    plot_fx(fxs)
    plot_fx(alphas)
