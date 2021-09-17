import numpy as np
from deap import tools
from deap.tools._hypervolume import hv


def hypervolume(individuals, ref=None):
    # front = tools.sortLogNondominated(individuals, len(individuals), first_front_only=True)
    wobjs = np.array([ind.fitness.wvalues for ind in individuals]) * -1
    if ref is None:
        ref = np.max(wobjs, axis=0) + 1
    return hv.hypervolume(wobjs, ref)