import time

import numpy as np
from deap.tools._hypervolume import hv
from pymoo.factory import get_performance_indicator

from src.models.moo.utils.deap.utils import get_deap_pops_obj, get_pymoo_pops_obj
from src.models.moo.utils.plot import get_fitnesses
from src.utils.util import array_from_lists


def hypervolume(individuals, ref=None):
    # front = tools.sortLogNondominated(individuals, len(individuals), first_front_only=True)
    wobjs = np.array([ind.fitness.wvalues for ind in individuals]) * -1
    if ref is None:
        ref = np.max(wobjs, axis=0)  # + 1
    return hv.hypervolume(wobjs, ref)


def get_hypervolume(pop, ref=None):
    F = pop if isinstance(pop, np.ndarray) else get_fitnesses(pop)
    ref = np.max(F, axis=0) if ref is None else np.array(ref)
    hypervol = hv.hypervolume(F, ref)
    # hv = get_performance_indicator("hv", ref_point=ref)
    # hypervol = hv.do(F)
    return hypervol


def get_hvs_from_log(hist, lib='deap'):
    pops_obj, ref = get_deap_pops_obj(hist) if lib == 'deap' else get_pymoo_pops_obj(hist)
    hv = get_performance_indicator("hv", ref_point=ref)
    hypervols = [hv.do(pop_obj) for pop_obj in pops_obj]
    return hypervols


def hv_hist_from_runs(runs, ref=None):
    pop_hist_runs = [run['result']['pop_hist'] for run in runs]
    hv_hist_runs = []
    for pop_hist in pop_hist_runs:
        hv_hist_runs.append([get_hypervolume(pop, ref=ref) for pop in pop_hist])
    return array_from_lists(hv_hist_runs)
