from functools import partial

import numpy as np
from deap import benchmarks
from matplotlib import pyplot as plt
from pymoo.factory import get_problem

from src.optimization.functions.mop import zdt2, dtlz2

if __name__ == '__main__':
    # %%
    n_obj = 3
    number_of_variables = 30
    X = np.random.random((100, number_of_variables))
    obj_fun_deap = partial(benchmarks.dtlz2, obj=n_obj)

    problem = get_problem("dtlz2")
    problem.n_var = number_of_variables

    fx1s = np.array([obj_fun_deap(x) for x in X])
    fx2s = np.array([problem.evaluate(x) for x in X])
    fx3s = np.array([dtlz2(x) for x in X])

    fig, axes = plt.subplots(n_obj, 1, figsize=(10, 5))
    for i in range(n_obj):
        axes[i].plot(fx1s[:, i])
        axes[i].plot(fx2s[:, i])
        axes[i].plot(fx3s[:, i])
    plt.show()

    print('diff: {}'.format(np.sum(np.sum(fx1s - fx2s))))



