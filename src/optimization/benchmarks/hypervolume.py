import time

from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_problem, get_reference_directions

from src.models.moo.smsemoa import SMSEMOA
from src.models.moo.utils.indicators import get_hypervolume
from src.optimization.harness.moo import run_moo, create_title
from src.utils.plotly.plot import plot_results_moo
from src.utils.util import get_type_str
from deap.tools._hypervolume import hv

if __name__ == '__main__':
    prob_cfg = {'name': 'wfg4',
                'n_obj': 10,
                'n_variables': 24,
                'hv_ref': [5]*10  # used for HV history
                }
    algo_cfg = {'termination': ('n_gen', 100),
                'max_gen': 150,
                'pop_size': 100,
                'hv_ref': [10]*10  # used only for SMS-EMOA
                }

    problem = get_problem(prob_cfg['name'],
                          n_var=prob_cfg['n_variables'],
                          n_obj=prob_cfg['n_obj'])

    ref_dirs = get_reference_directions("energy", prob_cfg['n_obj'], algo_cfg['pop_size'])
    algorithm = NSGA3(
        pop_size=algo_cfg['pop_size'],
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True,
        ref_dirs=ref_dirs
    )

    algo_cfg['name'] = get_type_str(algorithm)
    result = run_moo(problem, algorithm, algo_cfg, verbose=2)

    t0 = time.time()
    hv_pop = get_hypervolume(result['pop_hist'][-1], prob_cfg['hv_ref'])
    print('PYMOO Hypervolume execution time: {}'.format(round(time.time() - t0, 4)))
    print('Hypervolume from all population for reference point {}: {}'.format(str(prob_cfg['hv_ref']),
                                                                              round(hv_pop, 4)))

    t0 = time.time()
    hv_pop = hv.hypervolume(result['pop_hist'][-1],  prob_cfg['hv_ref'])
    print('DEAP Hypervolume execution time: {}'.format(round(time.time() - t0, 4)))
    print('Hypervolume from all population for reference point {}: {}'.format(str(prob_cfg['hv_ref']),
                                                                              round(hv_pop, 4)))
