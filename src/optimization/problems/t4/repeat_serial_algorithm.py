import time

import numpy as np
import seaborn as sns
from pymoo.factory import get_reference_directions

from src.models.moo.utils.indicators import hv_hist_from_runs
from src.optimization.harness.moo import repeat_moo
from src.optimization.harness.plot import plot_runs
from src.utils.util import save_vars

sns.set_context('poster')

if __name__ == '__main__':
    # %%
    ref_dirs = get_reference_directions("uniform", 3, n_partitions=17)


    prob_cfg = {'name': 'wfg3',
                'n_obj': 3,
                'n_variables': 24,
                'hv_ref': [10]*3}

    algo_cfg = {'termination': ('n_gens', 90),
                'pop_size': 171,
                'hv_ref': [10]*3}

    params = {'name': 'NSGA3',
              'algo_cfg': algo_cfg,
              'prob_cfg': prob_cfg,
              'verbose': 1,
              'seed': None,
              'ref_dirs': ref_dirs}

    is_reproductible = True
    n_repeat = 20

    t0 = time.time()
    runs = repeat_moo(params, n_repeat, is_reproductible)

    for run in runs:
        del run['result']
    save_vars(runs,
              file_path=['output',
                         'repeat',
                         'results',
                         '{}_res'.format(prob_cfg['name'])])

    print('Total time: {} s'.format(round((time.time() - t0)), 1))


