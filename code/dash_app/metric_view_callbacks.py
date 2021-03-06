from dash.dependencies import Input, Output, State
from server import app
from data_filter import get_metrics_df, get_constraint_scores_df
from dash.exceptions import PreventUpdate

import cufflinks as cf

# import plotly.plotly as py
# import pandas as pd
# import numpy as np


cf.go_offline()


def _create_figure_from_df(df, perp, view_perp_scale="log", subplot_height=115):
    """From user param, obtain `df`, a DataFrame of constraint scores or metric scores.
    Then using `cufflinks` to generate line charts according to the column names in `df`
    The xaxis is perplexity values in `view_perp_scale` in ['log', 'linear'] scale.
    An additional vertical line is to indicate the position of the current `perp`.
    """
    n_metrics = len(df.columns)
    # Available cufflinks thems: ['pearl', 'white', 'ggplot', 'solar', 'space']
    figure = df.iplot(
        kind="scatter",
        asFigure=True,
        theme="white",
        vspan={"x0": perp, "x1": perp, "color": "rgba(30,30,30,0.3)", "dash": "dash"},
        subplots=True,
        shape=(n_metrics, 1),
        shared_xaxes=True,
        subplot_titles=True,
        legend=False,
        fill=False,  # True to fill the area under the line
        vertical_spacing=0.1,
    )
    limit_yaxes = {
        f"yaxis{i+1}": {"range": [df[df.columns[i]].min(), df[df.columns[i]].max()]}
        for i in range(n_metrics)
    }
    figure["layout"].update(
        dict(
            margin=dict(b=60, l=30, t=40, r=10),
            height=n_metrics * subplot_height,
            xaxis=(
                dict(title="Log Perplexity", tickprefix="", type="log")
                if view_perp_scale == "log"
                else dict(title="Perplexity", tickprefix="", type="linear")
            ),
        )
    ).update(limit_yaxes)
    return figure


@app.callback(
    [Output("metric-view-chain", "figure"), Output("metric-view-normal", "figure")],
    [
        Input("select-dataset", "value"),
        Input("select-perp-val", "value"),
        Input("select-base-perp-val", "value"),
        Input("select-perp-scale", "value"),
        Input("select-earlystop-val", "value"),
    ],
    [],
)
def update_metric_view(dataset_name, perp, base_perp, view_perp_scale, earlystop):
    """Render the metric scores line charts corresponding to the dataset and the given
    params. The current perplexity is indicated in the line chart, so the charts will be
    updated whenever the perplexity is changed.
    """
    if None in [dataset_name, perp, view_perp_scale, earlystop]:
        raise PreventUpdate

    df_chain = get_metrics_df(dataset_name, earlystop, base_perp=base_perp)
    df_normal = get_metrics_df(dataset_name, earlystop, base_perp=None)

    return (
        _create_figure_from_df(df_chain, perp, view_perp_scale, subplot_height=135),
        _create_figure_from_df(df_normal, perp, view_perp_scale, subplot_height=135),
    )


@app.callback(
    [
        Output("select-perp-val", "value"),
        Output("constraint-score-view-chain", "figure"),
        Output("constraint-score-view-normal", "figure"),
        Output("links-memory-debug", "data"),
    ],
    [
        Input("btn-submit", "n_clicks"),
        Input("select-base-perp-val", "value"),
        Input("select-perp-scale", "value"),
        Input("select-earlystop-val", "value"),
    ],
    [State("select-dataset", "value"), State("links-memory", "data")],
)
def update_constraint_score_view(
    btn_submit, base_perp, view_perp_scale, earlystop, dataset_name, user_links
):
    """Render the constraint score line chart after finding the best perplexity
    according to the user constraints stored in `links-memory`
    """
    if None in [btn_submit, dataset_name, user_links]:
        raise PreventUpdate

    def _gen_fig_score(base_perp=None, detail=False):
        df, detail_links = get_constraint_scores_df(
            dataset_name, user_links, earlystop, base_perp=base_perp, debug=detail
        )
        best_perp = df["score_all_links"].idxmax()
        return (
            best_perp,
            detail_links,
            _create_figure_from_df(df, best_perp, view_perp_scale, subplot_height=135),
        )

    best_perp_chain, detail_links_chain, fig_scores_chain = _gen_fig_score(
        base_perp=base_perp, detail=True
    )
    best_perp_normal, _, fig_scores_normal = _gen_fig_score(base_perp=None)
    return best_perp_chain, fig_scores_chain, fig_scores_normal, detail_links_chain
