import math
import time

import numpy as np
import plotly.io as pio
from deap import tools
from deap.tools import sortLogNondominated
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from src.models.moo.utils.deap.nsga3.nsgaiii_survive import find_extreme_points, find_intercepts, associate_to_niche, niching
from src.utils.plotly.utils import plotly_save

pio.renderers.default = "browser"

colors = [
    '#1f77b4',  # muted blue
    '#2ca02c',  # cooked asparagus green
    '#ff7f0e',  # safety orange
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'  # blue-teal
    'rgba(0, 0, 0, 0)' #transparent ix=
]

marker_symbols = ['circle', 'x', 'cross', 'circle-open']

def plane_from_intercepts(intercepts, mesh_size=0.01):
    xrange = np.arange(0, math.ceil(intercepts[0]) + mesh_size, mesh_size)
    yrange = np.arange(0, math.ceil(intercepts[1]) + mesh_size, mesh_size)
    xx, yy = np.meshgrid(xrange, yrange)
    xx, yy = xx.flatten(), yy.flatten()
    A = np.vstack([xx, yy]).T
    coeff = 1 / intercepts
    zz = (1 / coeff[2]) * (1 - np.sum(np.multiply(A, coeff[:2]), axis=1))
    z = zz.reshape((yrange.shape[0], xrange.shape[0]))
    z[z < 0] = np.nan

    return xrange, yrange, z

def get_fitnesses(pop):
    fitnesses = np.array([ind.fitness.wvalues for ind in pop])
    fitnesses *= -1
    return fitnesses


def calc_geo_points(pop):
    best_point = np.min(pop, axis=0)
    worst_point = np.max(pop, axis=0)

    extreme_points = find_extreme_points(pop, best_point)
    intercepts = find_intercepts(extreme_points, best_point, worst_point, worst_point)

    return best_point, worst_point, extreme_points, intercepts


def add_3d_scatter_trace(data, name=None, color_ix=0, markersize=5, marker_symbol='circle',
                         mode='markers', legend=True, opacity=1, line_width=None, transparent_fill=False):
    return go.Scatter3d(
        x=data[:, 0],
        y=data[:, 1],
        z=data[:, 2],
        visible=True,
        showlegend=legend,
        name=name,
        mode=mode,
        opacity=opacity,
        marker_symbol=marker_symbol,
        marker=dict(size=markersize,
                    color='rgba(0, 0, 0, 0)' if transparent_fill else colors[color_ix],
                    line=dict(
                        color=colors[color_ix],
                        width=line_width
                    ) if line_width is not None else None)
    )


def add_3d_surface_trace(xrange, yrange, z, opacity=0.1):
    return go.Surface(
        x=xrange,
        y=yrange,
        z=z,
        showlegend=False,
        opacity=opacity,
        surfacecolor=np.zeros(z.shape),
        colorscale='Greens',
        showscale=False,
    )


def plot_pop(pop, save=False, file_path=None, label_scale=1, size=(1980, 1080),
             plot_norm=False, save_png=False, title='', opacity=1):

    fitnesses = get_fitnesses(pop) if not isinstance(pop, np.ndarray) else pop
    best_point, worst_point, extreme_points, intercepts = calc_geo_points(fitnesses)

    fig = make_subplots(rows=1, cols=1)
    traces = []

    # Population
    traces.append(add_3d_scatter_trace(fitnesses, name='population', color_ix=0, opacity=opacity, markersize=5))

    # Ideal and worst point
    traces.append(
        add_3d_scatter_trace(best_point.reshape(1, -1), name='ideal_point', color_ix=2, markersize=10,
                             marker_symbol='cross'))
    traces.append(
        add_3d_scatter_trace(worst_point.reshape(1, -1), name='ndir_point', color_ix=5, markersize=10,
                             marker_symbol='cross'))

    # Normalized Objectives
    if plot_norm:
        ref_points = tools.uniform_reference_points(nobj=3, p=12)
        fn = (fitnesses - best_point) / (intercepts - best_point)
        traces.append(add_3d_scatter_trace(fn, name='normalized', color_ix=0, markersize=4,
                                           marker_symbol='circle-open'))
        traces.append(add_3d_scatter_trace(extreme_points, name='extremes', color_ix=3, markersize=5))
        verts = np.array([(intercepts[0], 0, 0), (0, intercepts[1], 0), (0, 0, intercepts[2])])
        traces.append(add_3d_scatter_trace(verts, name='intercepts', color_ix=1, markersize=4, marker_symbol='x'))
        xrange, yrange, z = plane_from_intercepts(intercepts, mesh_size=0.01)
        traces.append(add_3d_surface_trace(xrange, yrange, z))

        # Reference Points
        traces.append(add_3d_scatter_trace(ref_points, name='reference points', color_ix=1, markersize=2,
                                           marker_symbol='circle'))

    fig = go.Figure(data=traces)
    fig.update_layout(scene=dict(
        xaxis_title='f1(x)',
        yaxis_title='f2(x)',
        zaxis_title='f3(x)'))
    fig.update_layout(title=title, legend_itemsizing='constant',
                      legend=dict(font=dict(size=18 * label_scale)))
    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.show()

    time.sleep(1.5)
    if file_path is not None and save is True:
        plotly_save(fig, file_path, size, save_png)


def plot_multiple_pop(pops, labels=None, legend_label='population', save=False, file_path=None, colors_ix=None,
                      label_scale=1, size=(1980, 1080), color_default=0, save_png=False, title='', opacities=None,
                      opacity=1, prog_markers=False, prog_opacity=False, prog_color=True, plot_border=False):

    colors_ix = np.ones(len(pops), dtype=int) * color_default if colors_ix is None else colors_ix

    Fs = [get_fitnesses(pop) for pop in pops] if not isinstance(pops[0], np.ndarray) else pops
    all_F = np.vstack(Fs)
    best_point = np.min(all_F, axis=0)
    worst_point = np.max(all_F, axis=0)

    fig = make_subplots(rows=1, cols=1)
    traces = []

    # Population
    for i, F in enumerate(Fs):
        op = i / len(Fs) if prog_opacity else (opacity if opacities is None else opacities[i])
        traces.append(add_3d_scatter_trace(F, color_ix=i if prog_color else colors_ix[i], markersize=5,
                                           opacity=op,
                                           marker_symbol=marker_symbols[i] if prog_markers else 'circle',
                                           name=legend_label + ' ' + str(i) if labels is None else labels[i]))
        if plot_border:
            traces.append(add_3d_scatter_trace(F, color_ix=i if prog_color else colors_ix[i], markersize=5,
                                               transparent_fill=True, opacity=opacity, line_width=0.5,
                                               marker_symbol='circle', legend=False))
    # Ideal and worst point
    traces.append(
        add_3d_scatter_trace(best_point.reshape(1, -1), name='ideal_point', color_ix=2, markersize=10,
                             marker_symbol='cross'))
    traces.append(
        add_3d_scatter_trace(worst_point.reshape(1, -1), name='ndir_point', color_ix=5, markersize=10,
                             marker_symbol='cross'))

    fig = go.Figure(data=traces)
    fig.update_layout(scene=dict(
        xaxis_title='f1(x)',
        yaxis_title='f2(x)',
        zaxis_title='f3(x)'))
    fig.update_layout(title=title, legend_itemsizing='constant',
                      legend=dict(font=dict(size=18 * label_scale)))
    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.show()

    if file_path is not None and save is True:
        time.sleep(1.5)
        plotly_save(fig, file_path, size, save_png)


def plot_gen_progress(pop_hist, step_size, file_path=None, title=None, save=False, color_ix=1, last_color_ix=0):
    pop_hist = [pop_hist[i] for i in range(0, len(pop_hist), step_size)]
    colors_ix = np.ones(len(pop_hist), dtype=int) * color_ix
    colors_ix[-1] = last_color_ix
    labels = ['gen ' + str(i * step_size) for i in range(len(pop_hist))]
    plot_multiple_pop(pop_hist, labels=labels, save=save, colors_ix=colors_ix,
                      prog_opacity=True, prog_color=False, plot_border=True,
                      file_path=file_path, title=title)


def plot_nitching_selec_deap(pop, k, save=False, file_path=None, label_scale=1, size=(1980, 1080),
                             save_png=False, title=''):
    ref_points = tools.uniform_reference_points(nobj=3, p=12)
    pareto_fronts = sortLogNondominated(pop, k)

    # Extract fitnesses as a np array in the nd-sort order
    # Use wvalues * -1 to tackle always as a minimization problem
    fitnesses = np.array([ind.fitness.wvalues for f in pareto_fronts for ind in f])
    fitnesses *= -1

    # Get best and worst point of population, contrary to py_moo
    best_point = np.min(fitnesses, axis=0)
    worst_point = np.max(fitnesses, axis=0)

    extreme_points = find_extreme_points(fitnesses, best_point)
    front_worst = np.max(fitnesses[:sum(len(f) for f in pareto_fronts), :], axis=0)
    intercepts = find_intercepts(extreme_points, best_point, worst_point, front_worst)
    niches, dist = associate_to_niche(fitnesses, ref_points, best_point, intercepts)
    fn = (fitnesses - best_point) / (intercepts - best_point)

    # Get counts per niche for individuals in all front but the last
    niche_counts = np.zeros(len(ref_points), dtype=np.int64)
    index, counts = np.unique(niches[:-len(pareto_fronts[-1])], return_counts=True)
    niche_counts[index] = counts

    pairs = []
    for i, ix in enumerate(niches):
        pairs.append(np.array([(fn[i, :], ref_points[ix])]).squeeze(axis=0))

    # Choose individuals from all fronts but the last
    chosen = fn[:-len(pareto_fronts[-1])]
    selection = fn[-len(pareto_fronts[-1]):]

    sel_count = len(chosen)
    n = k - sel_count
    ind_fn = [sel for sel in selection]
    selected = np.array(niching(ind_fn, n, niches[sel_count:], dist[sel_count:], niche_counts))

    # PLOT
    fig = make_subplots(rows=1, cols=1)
    traces = []

    # Ideal and worst point
    traces.append(
        add_3d_scatter_trace(best_point.reshape(1, -1), name='ideal_point', color_ix=2, markersize=10,
                             marker_symbol='cross'))

    # Normalized Objectives
    traces.append(add_3d_scatter_trace(chosen, name='selected individuals from first k-1 fronts', color_ix=0,
                                       markersize=5, marker_symbol='circle'))
    traces.append(add_3d_scatter_trace(selection, name='non-selected individuals from last front (nitching)',
                                       color_ix=3, markersize=5, marker_symbol='circle'))
    traces.append(add_3d_scatter_trace(selected, name='selected individuals from last front (nitching)',
                                       color_ix=1, markersize=5, marker_symbol='circle'))
    # Reference Points
    traces.append(add_3d_scatter_trace(ref_points, name='reference points', color_ix=4, markersize=3,
                                       marker_symbol='circle'))

    pair_traces = [add_3d_scatter_trace(pair, name=None, color_ix=7, mode='lines', legend=False) for pair in pairs]
    traces += pair_traces

    fig = go.Figure(data=traces)
    fig.update_layout(scene=dict(
        xaxis_title='f1(x)',
        yaxis_title='f2(x)',
        zaxis_title='f3(x)'))
    fig.update_layout(title=title, legend_itemsizing='constant',
                      legend=dict(font=dict(size=18 * label_scale)))
    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.show()

    if file_path is not None and save is True:
        time.sleep(1.5)
        plotly_save(fig, file_path, size, save_png)

