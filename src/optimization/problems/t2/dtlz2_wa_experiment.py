import numpy as np

from src.models.moo.utils.deap.weight_avg.harness import run_wa
from src.optimization.functions.mop import dtlz2
from src.optimization.harness.repeat import repeat
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # %%
    problem_name = 'DTLZ2'
    problem = dtlz2
    weights = np.array([.6, .2, .2])
    prob_cfg = {'n_obj': 3, 'n_variables': 12, 'bounds_low': 0, 'bounds_up': 1}
    algo_cfg = {'algorithm': 'WEIGHTED AVERAGE', 'max_gen': 100, 'pop_size': 100, 'cross_prob': 0.8, 'weights': weights}

    n_repeat = 20
    results = repeat(run_wa, [problem, prob_cfg, algo_cfg, weights], n_repeat, parallel=False)

    #%%
    fitnesses = np.array([res['best_fitness'] for res in results])
    hvs = np.array([res['hypervol'] for res in results])

    sns.boxplot(x=fitnesses)
    plt.title('Best Fitness @ {} runs'.format(n_repeat))
    plt.xlabel('Weighted Average Fitness')
    plt.show()


