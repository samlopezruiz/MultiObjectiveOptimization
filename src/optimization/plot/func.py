import numpy as np
from matplotlib import pyplot as plt

from src.optimization.functions.funcs import problems


def mesh_contour(fx, bounds, delta=0.025):
    x = np.arange(bounds[0][0], bounds[0][1], delta)
    y = np.arange(bounds[1][0], bounds[1][1], delta)
    X, Y = np.meshgrid(x, y)

    @np.vectorize
    def func(x, y):
        # some arbitrary function
        return fx([x, y])

    X = X.T
    Y = Y.T
    Z = func(X, Y)
    return X, Y, Z


def plot_contour(fx, bounds, delta=0.25, n_levels=None, solid_color=True):
    X, Y, Z = mesh_contour(fx, bounds, delta=delta)
    if n_levels is not None:
        levels = np.arange(np.min(Z), np.max(Z), n_levels)
    else:
        levels = None
    fig, ax = plt.subplots(figsize=(12, 10))
    CS = ax.contourf(X, Y, Z, levels=levels, cmap='viridis')
    if solid_color:
        plt.colorbar(CS)
    ax.set_title('Contour Plot')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    plt.show()


def plot_contour_path(fx, bounds, x_path, delta=0.25, solid_color=True):
    X, Y, Z = mesh_contour(fx, bounds, delta=delta)
    n_levels = delta
    # if n_levels is not None:
    levels = np.arange(np.min(Z), np.max(Z), n_levels)
    # else:
    #     levels = None
    fig, ax = plt.subplots(figsize=(12, 10))
    CS = ax.contourf(X, Y, Z, levels=levels, cmap='viridis')
    if solid_color:
        plt.colorbar(CS)
    for x in x_path:
        plt.plot(*x, color='red', marker='o', markersize=10)
    ax.set_title('Contour Plot')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    plt.tight_layout()
    plt.show()


def plot_fx(fxs, y_label='f(x)'):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title('Gradient Descent')
    ax.set_xlabel('iteration')
    ax.set_ylabel(y_label)
    plt.grid()
    plt.plot(fxs)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    #%%
    name = 'eggholder'
    fx, bounds, bits = problems[name]
    plot_contour(fx, bounds, delta=1, n_levels=10)
