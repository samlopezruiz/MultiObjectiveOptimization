import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import normalize

from src.models.moo.utils.deap.weight_avg.harness import run_wa
from src.models.moo.utils.plot import plot_pop
from src.optimization.functions.mop import dtlz1

if __name__ == '__main__':
    # %%
    problem_name = 'DTLZ1'
    problem = dtlz1
    save_plots = False

    # Create 20 Random Weights
    weights = np.random.random((17, 3))
    weights = normalize(np.vstack([weights, np.eye(3)]), axis=1, norm='l1')

    prob_cfg = {'n_obj': 3, 'n_variables': 12, 'bounds_low': 0, 'bounds_up': 1}
    algo_cfg = {'algorithm': 'WEIGHTED AVERAGE', 'max_gen': 100, 'pop_size': 100, 'cross_prob': 0.8, 'weights': weights}

    # Execute algorithm for each weight
    results = []
    for w in weights:
        print('Weight: {}'.format(list(w)))
        results.append(run_wa(problem, prob_cfg, algo_cfg, weights=w, step_size=algo_cfg['max_gen'] // 20,
                       plot_=False, verbose=False, save_plots=save_plots, problem_name=problem_name))

    best_obj_space = np.array([problem(res['best_ind']) for res in results])

    # Plot the best individual optained for each weight
    plot_pop(best_obj_space, save=save_plots, file_path=['img', problem_name + '_weights'], plot_norm=False,
             title=problem_name + ' Optimum Solutions')

    # Pearson correlation between objective and weight
    pearson_corr, _ = pearsonr(best_obj_space.ravel(), weights.ravel())
    print('Pearson Correlation between Objective and Weight: {}'.format(round(pearson_corr, 4)))

