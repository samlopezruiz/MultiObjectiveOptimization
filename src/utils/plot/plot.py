import numpy as np
from deap.tools._hypervolume import hv
from matplotlib import pyplot as plt
from pymoo.factory import get_performance_indicator

from src.models.moo.deap.indicators import hypervolume
from src.models.moo.deap.utils import get_deap_pops_obj, get_pymoo_pops_obj


def plot_hv(hypervols, title=''):
    plt.plot(hypervols)
    plt.xlabel('Iterations (t)')
    plt.ylabel('Hypervolume')
    plt.title(title)
    plt.show()


def plot_hist_hv(hist, lib='deap'):
    pops_obj, ref = get_deap_pops_obj(hist) if lib == 'deap' else get_pymoo_pops_obj(hist)
    # hypervols = [hypervolume(pop, ref) for pop in pops]
    hv = get_performance_indicator("hv", ref_point=ref)
    hypervols = [hv.calc(pop_obj) for pop_obj in pops_obj]
    plot_hv(hypervols, title=lib)



