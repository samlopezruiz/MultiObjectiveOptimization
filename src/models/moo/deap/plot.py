import numpy as np
import plotly.io as pio
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from src.models.moo.deap.nsgaiii import find_ideal_point, find_extreme_points, construct_hyperplane, \
    normalize_objectives, generate_reference_points, associate, ind_to_fitness_array, plane_from_intercepts

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
]


def calc_geo_points(pop, calc_rps):
    ideal_point = find_ideal_point(pop)
    extremes = find_extreme_points(pop)
    intercepts = construct_hyperplane(pop, extremes)
    _ = normalize_objectives(pop, intercepts, ideal_point)
    rps = None
    if calc_rps:
        rps = generate_reference_points(3)
        associate(pop, rps)
    return ideal_point, extremes, intercepts, rps


def add_3d_scatter_trace(data, name, ix=0, markersize=5, marker_symbol='circle',
                         mode='markers', legend=True, opacity=1):
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
                    color=colors[ix])
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


def plot_objective_space(pop, label_scale=1, plot_norm=True):
    if len(pop[0].fitness.values) <= 0:
        print('Cannot plot: individuals have no evaluation')
        return

    ideal_point, extremes, intercepts, rps = calc_geo_points(pop, calc_rps=False)

    fig = make_subplots(rows=1, cols=1)
    traces = []
    # Population
    pop_objective_space = ind_to_fitness_array(pop)
    traces.append(add_3d_scatter_trace(pop_objective_space, name='population', ix=0, markersize=5))
    traces.append(
        add_3d_scatter_trace(np.zeros((1, 3)), name='reference', ix=5, markersize=10, marker_symbol='cross'))

    # Ideal point
    traces.append(
        add_3d_scatter_trace(ideal_point.reshape(1, -1), name='ideal_point', ix=2, markersize=5, marker_symbol='x'))

    # Hyperplane
    extremes_obj_space = ind_to_fitness_array(extremes)
    traces.append(add_3d_scatter_trace(extremes_obj_space, name='extremes', ix=3, markersize=5))
    verts = np.array([(intercepts[0], 0, 0), (0, intercepts[1], 0), (0, 0, intercepts[2])])
    traces.append(add_3d_scatter_trace(verts, name='intercepts', ix=1, markersize=4, marker_symbol='x'))
    xrange, yrange, z = plane_from_intercepts(intercepts, mesh_size=0.01)
    traces.append(add_3d_surface_trace(xrange, yrange, z))

    # Normalized Objectives
    if plot_norm:
        norm_obj_space = ind_to_fitness_array(pop, 'normalized_values')
        traces.append(add_3d_scatter_trace(norm_obj_space, name='normalized', ix=0, markersize=3,
                                           marker_symbol='circle-open'))

    fig = go.Figure(data=traces)
    fig.update_layout(title='Objective Space', legend_itemsizing='constant',
                      legend=dict(font=dict(size=18 * label_scale)))
    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.show()


def plot_norm_objective_space(pop, label_scale=1):
    if len(pop[0].fitness.values) <= 0:
        print('Cannot plot: individuals have no evaluation')
        return

    ideal_point, extremes, intercepts, rps = calc_geo_points(pop, calc_rps=True)

    fig = make_subplots(rows=1, cols=1)
    traces = []

    # Ideal point
    traces.append(
        add_3d_scatter_trace(ideal_point.reshape(1, -1), name='ideal_point', ix=2, markersize=5, marker_symbol='x'))

    # Normalized Objectives
    norm_obj_space = ind_to_fitness_array(pop, 'normalized_values')
    traces.append(add_3d_scatter_trace(norm_obj_space, name='normalized', ix=1, markersize=5,
                                       marker_symbol='circle-open'))

    # Reference Points
    ref_points = np.array([[rp[0], rp[1], rp[2]] for rp in rps])
    traces.append(add_3d_scatter_trace(ref_points, name='reference points', ix=4, markersize=3,
                                       marker_symbol='circle'))

    pairs = []
    for ind in pop:
        pairs.append(np.array([(ind.fitness.normalized_values, list(ind.reference_point))]).squeeze(axis=0))

    pair_traces = [add_3d_scatter_trace(pair, name=None, ix=0, mode='lines', legend=False) for pair in pairs]
    traces += pair_traces

    fig = go.Figure(data=traces)
    fig.update_layout(scene=dict(
        xaxis_title='f1(x)',
        yaxis_title='f2(x)',
        zaxis_title='f3(x)'))
    fig.update_layout(title='Objective Space', legend_itemsizing='constant',
                      legend=dict(font=dict(size=18 * label_scale)))
    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.show()


def plot_pop_obj_space(pop, label_scale=1):

    fig = make_subplots(rows=1, cols=1)
    traces = []
    # Population
    traces.append(add_3d_scatter_trace(pop, name='population', ix=0, markersize=5))
    traces.append(
        add_3d_scatter_trace(np.zeros((1, 3)), name='reference', ix=5, markersize=10, marker_symbol='cross'))

    ideal_point = np.min(pop, axis=0)
    # Ideal point
    traces.append(
        add_3d_scatter_trace(ideal_point.reshape(1, -1), name='ideal_point', ix=2, markersize=5, marker_symbol='x'))

    fig = go.Figure(data=traces)
    fig.update_layout(scene=dict(
        xaxis_title='f1(x)',
        yaxis_title='f2(x)',
        zaxis_title='f3(x)'))
    fig.update_layout(title='Objective Space', legend_itemsizing='constant',
                      legend=dict(font=dict(size=18 * label_scale)))
    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.show()


def plot_pop_obj_hist(pop_hist, label_scale=1):
    fig = make_subplots(rows=1, cols=1)
    traces = []

    ideal_point = np.min(pop_hist[-1], axis=0)
    # Ideal point
    traces.append(
        add_3d_scatter_trace(ideal_point.reshape(1, -1), name='ideal_point', ix=2, markersize=5, marker_symbol='x'))
    traces.append(
        add_3d_scatter_trace(np.zeros((1, 3)), name='reference', ix=5, markersize=10, marker_symbol='cross'))
    # Population
    plot_gens = 10
    step_size = len(pop_hist) // plot_gens
    for i in range(0, len(pop_hist) + step_size, step_size):
        ix = min(i, len(pop_hist) - 1)
        color_ix = 1 if i == min(i, len(pop_hist) - 1) else 0
        traces.append(add_3d_scatter_trace(pop_hist[ix], name='gen ' + str(ix),
                                           ix=color_ix, markersize=4, opacity=ix / len(pop_hist)))

    fig = go.Figure(data=traces)
    fig.update_layout(scene=dict(
        xaxis_title='f1(x)',
        yaxis_title='f2(x)',
        zaxis_title='f3(x)'))
    fig.update_layout(title='Objective Space', legend_itemsizing='constant',
                      legend=dict(font=dict(size=18 * label_scale)))
    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.show()