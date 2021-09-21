import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from src.models.moo.deap.harness import run_nsga3
from src.optimization.functions.mop import dtlz1, dtlz2
from src.optimization.harness.repeat import repeat

if __name__ == '__main__':
    # %%
    problem_name = 'DTLZ2'
    problem = dtlz2
    prob_cfg = {'n_obj': 3, 'n_variables': 12, 'bounds_low': 0, 'bounds_up': 1}
    algo_cfg = {'algorithm': 'NSGA III', 'max_gen': 100, 'pop_size': 100, 'cross_prob': 0.8}

    n_repeat = 20
    results = repeat(run_nsga3, [problem, prob_cfg, algo_cfg], n_repeat, parallel=False)

    #%% Plot Hypervolume per Run
    hvs = np.array([res['hypervol'] for res in results])
    sns.boxplot(x=hvs)
    plt.title('Hypervolume @ {} runs'.format(n_repeat))
    plt.show()

