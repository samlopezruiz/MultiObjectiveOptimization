from itertools import product
from copy import copy

from src.optimization.harness.repeat import repeat_eval, print_eval


def gs_get_cfgs(gs_cfg, model_cfg, comb=True):
    keys = list(gs_cfg.keys())
    ranges = list(gs_cfg.values())
    combinations = list(product(*ranges))
    name_cfg = copy(gs_cfg)

    cfgs_gs, names = [], []
    if comb:
        for comb in combinations:
            new_cfg = copy(model_cfg)
            for k, key in enumerate(keys):
                new_cfg[key] = comb[k]
                name_cfg[key] = comb[k]
            cfgs_gs.append(new_cfg)
            names.append(str(name_cfg))
    else:
        for i in range(len(gs_cfg[keys[0]])):
            new_cfg = copy(model_cfg)
            for key in keys:
                new_cfg[key] = gs_cfg[key][i]
                name_cfg[key] = gs_cfg[key][i]
            cfgs_gs.append(new_cfg)
            names.append(str(name_cfg))

    return cfgs_gs, names


def gs_search(name, gs_cfg, model_cfg, get_ga, n_repeat=1, verbose=0):
    cfgs, names = gs_get_cfgs(gs_cfg, model_cfg)
    gs_logs = []
    for i, cfg in enumerate(cfgs):
        ga = get_ga(cfg=cfg)
        logs = repeat_eval(ga, n_repeat, cfg)
        if verbose > 0:
            print_eval(name, logs)
        gs_logs.append((names[i], logs))

    stats = consolidate_log(gs_logs)
    return gs_logs, stats



# def gs_gp_search(x, y, gs_cfg, model_cfg, n_repeat=1, verbose=0):
#     cfgs, names = gs_get_cfgs(gs_cfg, model_cfg, comb=True)
#
#     gs_logs = []
#     for c, (cfg, name) in enumerate(zip(cfgs, names)):
#         print(name)
#         res = repeat_eval_gp(n_repeat, cfg, x, y, c * n_repeat)
#         gs_logs.append((name, res))
#
#     stats = consolidate_gp_log(gs_logs)
#     return gs_logs, stats


def consolidate_log(gs_logs):
    stats = []
    for name, log_cfg in gs_logs:
        gens, evals = [], []
        for best_gen, best_eval, log, best_ind in log_cfg:
            gens.append(best_gen)
            evals.append(best_eval)
            stats.append((str(name), best_gen, best_eval))
    return stats


def consolidate_gp_log(gs_logs):
    stats = []
    for name, log_cfg in gs_logs:
        evals = []
        for best_ind, best_eval, log, pset in log_cfg:
            evals.append(best_eval)
            stats.append((str(name), best_eval))
    return stats

