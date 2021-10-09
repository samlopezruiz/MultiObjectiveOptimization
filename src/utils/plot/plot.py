import pymoo
from matplotlib import pyplot as plt
from pymoo.factory import get_performance_indicator

from src.models.moo.utils.deap.utils import get_pymoo_pops_obj, get_deap_pops_obj


def plot_hv(hypervols, title='', save=False):
    plt.plot(hypervols)
    plt.xlabel('Iterations (t)')
    plt.ylabel('Hypervolume')
    plt.title(title)
    plt.show()


def plot_hist_hv(hist, save=False):
    if isinstance(hist, pymoo.core.result.Result):
        pops_obj, ref = get_pymoo_pops_obj(hist)
    else:
        pops_obj, ref = get_deap_pops_obj(hist)

    hv = get_performance_indicator("hv", ref_point=ref)
    hypervols = [hv.do(pop_obj) for pop_obj in pops_obj]
    plot_hv(hypervols, title='Hypervolume History', save=save)
    return hypervols



