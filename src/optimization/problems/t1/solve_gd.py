import time
import numpy as np
from src.models.gd.gd_torch import gradient_descent_torch, gradient_descent
from src.optimization.plot.func import plot_contour_path, plot_fx
from src.optimization.functions.funcs import problems, gf2, gf1
import seaborn as sns

sns.set_theme()
sns.set_context("poster")

if __name__ == '__main__':
    # %% Parámetros
    alpha_n = .1
    n_iter = 100

    #%% Método por gradiente
    fx, bounds, bits = problems['f1_t1']
    x0 = [bounds[i][0] + np.random.rand() * (bounds[i][1] - bounds[i][0]) for i in range(len(bounds))]
    # x0 = [-380., -65.]
    print('x0={}'.format(x0))

    gfx = gf1
    t0 = time.time()
    xs, fxs, grads, alphas = gradient_descent(x0, fx, gfx, n_iter, alpha_n, early_break=True)
    print('gradient method time = {} ms'.format(round((time.time() - t0) * 100, 4)))
    print('f({}) = {}'.format(xs[-1], fxs[-1]))
    plot_fx(fxs)
    plot_fx(alphas, y_label='alpha')

    #%% Método usando auto-diff
    # fx, bounds, bits = problems['f2_t1_torch']
    # t0 = time.time()
    # xs, fxs, grads = gradient_descent_torch(x0, fx, n_iter, alpha_n, early_break=False)
    # print('gradient method time = {} ms'.format(round((time.time() - t0) * 100, 4)))
    # print('f({}) = {}'.format(xs[-1], fxs[-1]))
    # plot_fx(fxs)

    plot_contour_path(fx, bounds, xs, delta=1)


