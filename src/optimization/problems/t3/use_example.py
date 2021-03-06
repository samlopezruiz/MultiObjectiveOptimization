from pymoo.algorithms.moo.moead import MOEAD
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_problem, get_reference_directions

from src.models.moo.smsemoa import SMSEMOA
from src.models.moo.utils.indicators import get_hypervolume
from src.optimization.harness.moo import run_moo, create_title
from src.utils.plotly.plot import plot_results_moo
from src.utils.util import get_type_str

if __name__ == '__main__':
    prob_cfg = {'name': 'dtlz2',
                'n_obj': 3,
                'n_variables': 24,
                'hv_ref': [5]*3  # used for HV history
                }
    algo_cfg = {'termination': ('n_gen', 100),
                'max_gen': 150,
                'pop_size': 100,
                'hv_ref': [10]*3  # used only for SMS-EMOA
                }

    problem = get_problem(prob_cfg['name'],
                          n_var=prob_cfg['n_variables'],
                          n_obj=prob_cfg['n_obj'])

    cbx_pb = algo_cfg.get('cbx_pb', 0.9)
    cbx_eta = algo_cfg.get('cbx_eta', 15)
    mut_eta = algo_cfg.get('mut_eta', 20)
    ref_dirs = get_reference_directions("energy", prob_cfg['n_obj'], algo_cfg['pop_size'])
    algorithm = MOEAD(
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=cbx_pb, eta=cbx_eta),
        mutation=get_mutation("real_pm", eta=mut_eta),
        ref_dirs=ref_dirs,
        n_neighbors=15,
        prob_neighbor_mating=0.7,
    )

    algo_cfg['name'] = get_type_str(algorithm)
    result = run_moo(problem, algorithm, algo_cfg, verbose=2)

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
