import time

from pymoo.factory import get_problem, get_reference_directions

from src.models.moo.utils.indicators import get_hypervolume
from src.optimization.harness.moo import run_moo, create_title, get_algorithm
from src.utils.plotly.plot import plot_results_moo
from src.utils.util import get_type_str

if __name__ == '__main__':
    ref_dirs = get_reference_directions("uniform", 3, n_partitions=17)

    prob_cfg = {'name': 'wfg1',
                'n_obj': 3,
                'n_variables': 24,
                'hv_ref': [10]*3
                }
    algo_cfg = {'termination': ('n_gen', 200),
                'pop_size': ref_dirs.shape[0],
                }

    problem = get_problem(prob_cfg['name'],
                          n_var=prob_cfg['n_variables'],
                          n_obj=prob_cfg['n_obj'])

    cbx_pb = algo_cfg.get('cbx_pb', 0.9)
    cbx_eta = algo_cfg.get('cbx_eta', 15)
    mut_eta = algo_cfg.get('mut_eta', 20)

    exec_times = {}

    t0 = time.time()
    algorithm = get_algorithm('NSGA3', algo_cfg, n_obj=3, ref_dirs=ref_dirs, save_history=False)
    exec_times['time_ini_algo'] = time.time() - t0

    algo_cfg['name'] = get_type_str(algorithm)
    t0 = time.time()
    result = run_moo(problem, algorithm, algo_cfg, verbose=2)
    exec_times['time_serial'] = time.time() - t0

    plot_results_moo(result['res'],
                     file_path=['img', 'opt_deap_res'],
                     title=create_title(prob_cfg, algo_cfg, algorithm),
                     save_plots=False)

    hv_pop = get_hypervolume(result['pop_hist'][-1], prob_cfg['hv_ref'])
    hv_opt = get_hypervolume(result['res'].opt.get('F'), prob_cfg['hv_ref'])
    print('Hypervolume from all population for reference point {}: {}'.format(str(prob_cfg['hv_ref']),
                                                                              round(hv_pop, 4)))
    print('Hypervolume from opt population for reference point {}: {}'.format(str(prob_cfg['hv_ref']),
                                                                              round(hv_opt, 4)))
    print('Optimum population {}/{}'.format(len(result['res'].opt),
                                            len(result['pop_hist'][-1])))
