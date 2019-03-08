from dash.dependencies import Input, Output, State
from server import app
from data_filter import get_embedding, get_metrics_df, get_constraint_scores_df
from dash.exceptions import PreventUpdate

import plotly.plotly as py
import cufflinks as cf
import pandas as pd
import numpy as np


cf.go_offline()


def _create_figure_from_df(df, perp):
    n_metrics = len(df.columns)
    #Available cufflinks thems: ['pearl', 'white', 'ggplot', 'solar', 'space']
    figure = df.iplot(kind='scatter', asFigure=True, theme='white',
                      vline=[perp],
                      subplots=True, shape=(n_metrics, 1), shared_xaxes=True,
                      subplot_titles=True, legend=False, fill=True,
                      vertical_spacing=0.06)
    limit_yaxes = {
        f"yaxis{i+1}":{'range':[df[df.columns[i]].min(),
                                df[df.columns[i]].max()]}
        for i in range(n_metrics)
    }
    figure['layout'].update(dict(
        margin=dict(b=50, l=30, t=40, r=10),
        height=n_metrics * 115,
        xaxis=dict(title='Log Perplexity', tickprefix='', type='log'),
    )).update(limit_yaxes)
    return figure


@app.callback(
    Output('metric-view', 'figure'),
    [Input('select-dataset', 'value'),
     Input('select-perp-val', 'value')]
)
def update_metric_view(dataset_name, perp):
    if None in [dataset_name, perp]:
        raise PreventUpdate

    df = get_metrics_df(dataset_name)
    return _create_figure_from_df(df, perp)


@app.callback(
    [Output('select-perp-val', 'value'),
     Output('constraint-score-view', 'figure')],
    [Input('btn-submit', 'n_clicks')],
    [State('select-dataset', 'value'),
     State('links-memory', 'data')]
)
def update_constraint_score_view(btn_submit, dataset_name, user_links):
    if None in [btn_submit, dataset_name, user_links]:
        raise PreventUpdate

    df = get_constraint_scores_df(dataset_name, user_links)
    idx_max = df['score_all_links'].idxmax()
    best_perp = df.index.values[idx_max]
    fig_scores = _create_figure_from_df(df, best_perp)
    return best_perp, fig_scores
