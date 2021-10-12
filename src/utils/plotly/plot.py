import time
import plotly.io as pio
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from src.models.moo.utils.plot import plot_multiple_pop
from src.utils.plot.plot import plot_hist_hv
from src.utils.plotly.utils import plotly_params_check, plotly_save

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

gray_colors = [
    'lightgray', 'gray', 'darkgray', 'lightslategray',
]


def plotly_time_series(df, title=None, save=False, legend=True, file_path=None, size=(1980, 1080), color_col=None,
                       markers='lines+markers', xaxis_title="time", markersize=5, plot_title=True, label_scale=1,
                       adjust_height=(False, 0.6), plot_ytitles=False, **kwargs):
    params_ok, params = plotly_params_check(df, **kwargs)
    features, rows, cols, type_plot, alphas = params
    n_rows = len(set(rows))
    n_cols = 1  # len(set(cols))
    if not params_ok:
        return

    f = len(features)
    if adjust_height[0]:
        heights = [(1 - adjust_height[1]) / (n_rows - 1) for _ in range(n_rows - 1)]
        heights.insert(0, adjust_height[1])
    else:
        heights = [1 / n_rows for _ in range(n_rows)]

    fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=True, row_heights=heights)
    n_colors = len(df[color_col].unique()) if color_col is not None else 2
    for i in range(f):
        df_ss = df[features[i]].dropna()
        fig.append_trace(
            go.Bar(
                x=df_ss.index,
                y=df_ss,
                orientation="v",
                visible=True,
                showlegend=legend,
                name=features[i],
                opacity=alphas[i]
            ) if type_plot[i] == 'bar' else
            go.Scatter(
                x=df_ss.index,
                y=df_ss,
                visible=True,
                showlegend=legend,
                name=features[i],
                mode=markers,
                line=dict(color='lightgray' if color_col is not None else None),
                marker=dict(size=markersize,
                            color=None if color_col is None else df[color_col].values,
                            colorscale=colors[:n_colors]),
                opacity=alphas[i]
            ),
            row=rows[i] + 1,
            col=cols[i] + 1
        )
        if i == n_rows - 1:
            fig['layout']['xaxis' + str(i + 1)]['title'] = xaxis_title

        if plot_ytitles:
            for i in range(n_rows):
                fig['layout']['yaxis' + str(i + 1)]['title'] = features[i]

    fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False,
                      title=title if plot_title else None, legend=dict(font=dict(size=18 * label_scale)))

    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))

    # plotly(fig)
    fig.show()
    time.sleep(1.5)

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size)


def plot_results_moo(res, file_path=None, title=None, save_plots=False, use_date=False):
    # Plotting hypervolume
    plot_hist_hv(res, save=save_plots)

    # Plot Optimum Solutions
    pop_hist = [gen.pop.get('F') for gen in res.history]
    if res.opt.get('F').shape[1] == 3:
        plot_multiple_pop([pop_hist[-1], res.opt.get('F')],
                          labels=['population', 'optimum'],
                          save=save_plots,
                          opacities=[.2, 1],
                          plot_border=True,
                          file_path=file_path,
                          title=title,
                          use_date=use_date)
    else:
        print('Objective space cannot be plotted. Shape={}'.format(res.opt.get('F').shape[1]))
