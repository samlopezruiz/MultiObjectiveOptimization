import numpy as np
import seaborn as sns

from src.models.moo.utils.indicators import hv_hist_from_runs
from src.optimization.harness.moo import repeat_moo
from src.optimization.harness.plot import plot_runs

sns.set_context('poster')

if __name__ == '__main__':
    # %%
    prob_cfg = {'name': 'wfg4',
                'n_obj': 5,
                'n_variables': 24,
                'hv_ref': [5]*5}

    algo_cfg = {'termination': ('n_gens', 100),
                'pop_size': 100,
                'hv_ref': [10]*5}

    params = {'name': 'SMSEMOA',
              'algo_cfg': algo_cfg,
              'prob_cfg': prob_cfg,
              'verbose': 2,
              'seed': None}

    is_reproductible = True
    n_repeat = 1

    runs = repeat_moo(params, n_repeat, is_reproductible)
    hv_hist_runs = hv_hist_from_runs(runs, ref=prob_cfg['hv_ref'])

    plot_runs(hv_hist_runs, np.mean(hv_hist_runs, axis=0),
              x_label='Generations',
              y_label='Hypervolume',
              title='{} convergence plot with {}. Ref={}'.format(prob_cfg['name'],
                                                                 params['name'],
                                                                 str(prob_cfg['hv_ref'])),
              size=(15, 9),
              file_path=['img', params['name']+'_'+prob_cfg['name']],
              save=True)


