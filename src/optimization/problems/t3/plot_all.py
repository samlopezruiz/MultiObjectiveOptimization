from src.optimization.harness.moo import run_moo_problem

if __name__ == '__main__':
    # %%
    prob_cfg = {'name': 'wfg2',
                'n_obj': 3,
                'n_variables': 24,
                'hv_ref': [5, 5, 5]}

    algo_cfg = {'termination': ('n_eval', 2000),
                'pop_size': 100,
                'hv_ref': [10, 10, 10]}

    prob_cfgs = [prob_cfg]

    algos = ['NSGA3']
    for prob_cfg in prob_cfgs:
        for algo in algos:
            result = run_moo_problem(algo,
                                     algo_cfg,
                                     prob_cfg,
                                     plot=True,
                                     verbose=1,
                                     file_path=['img', algo+'_'+prob_cfg['name']],
                                     save_plots=True)

