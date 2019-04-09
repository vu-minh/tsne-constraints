from dash.dependencies import Input, Output, State
from server import app
from data_filter import get_metrics_df, get_constraint_scores_df
from dash.exceptions import PreventUpdate

import cufflinks as cf

# import plotly.plotly as py
# import pandas as pd
# import numpy as np


cf.go_offline()


def _create_figure_from_df(df, perp, subplot_height=115, xscale="linear"):
    n_metrics = len(df.columns)
    # Available cufflinks thems: ['pearl', 'white', 'ggplot', 'solar', 'space']
    figure = df.iplot(
        kind="scatter",
        asFigure=True,
        theme="white",
        # vline=[perp],
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
                if xscale == "log"
                else dict(title="Perplexity", tickprefix="", type="linear")
            ),
        )
    ).update(limit_yaxes)
    return figure


@app.callback(
    Output("metric-view", "figure"),
    [Input("select-dataset", "value"), Input("select-perp-val", "value")],
)
def update_metric_view(dataset_name, perp):
    if None in [dataset_name, perp]:
        raise PreventUpdate

    try:
        df = get_metrics_df(dataset_name)
    except Exception:
        raise PreventUpdate
    return _create_figure_from_df(df, perp, subplot_height=115)


@app.callback(
    [Output("select-perp-val", "value"), Output("constraint-score-view", "figure")],
    [Input("btn-submit", "n_clicks")],
    [State("select-dataset", "value"), State("links-memory", "data")],
)
def update_constraint_score_view(btn_submit, dataset_name, user_links):
    if None in [btn_submit, dataset_name, user_links]:
        raise PreventUpdate

    df = get_constraint_scores_df(dataset_name, user_links)
    best_perp = df["score_all_links"].idxmax()
    fig_scores = _create_figure_from_df(df, best_perp, subplot_height=125)
    return best_perp, fig_scores
