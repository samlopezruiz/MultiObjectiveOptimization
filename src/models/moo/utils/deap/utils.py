import numpy as np


def get_deap_pops_obj(logbook):
    pops = logbook.select('pop')
    pops_obj = [np.array([ind.fitness.values for ind in pop]) for pop in pops]
    ref = np.max([np.max(wobjs, axis=0) for wobjs in pops_obj], axis=0) #+ 1
    return pops_obj, ref

def get_pymoo_pops_obj(res):
    pops = [pop.pop for pop in res.history]
    pops_obj = [np.array([ind.F for ind in pop]) for pop in pops]
    ref = np.max([np.max(wobjs, axis=0) for wobjs in pops_obj], axis=0) #+ 1
    return pops_obj, ref

def get_deap_pop_hist(logbook):
    pop_hist = []
    for gen in logbook:
        pop = gen['pop']
        pop_hist.append(np.array([ind.fitness.values for ind in gen['pop']]))
    return pop_hist