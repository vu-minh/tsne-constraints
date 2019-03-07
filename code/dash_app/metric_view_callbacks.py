from dash.dependencies import Input, Output, State
from server import app
from data_filter import get_embedding, get_metrics_df
from dash.exceptions import PreventUpdate

import plotly.plotly as py
import cufflinks as cf
import pandas as pd
import numpy as np


#Available cufflinks thems: ['pearl', 'white', 'ggplot', 'solar', 'space']
cf.set_config_file(theme='white')


def _create_figure_from_df(df, perp):
    n_metrics = len(df.columns)
    figure = df.iplot(kind='scatter', asFigure=True,
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
    Output('constraint-score-view', 'figure'),
    [Input('select-dataset', 'value'),
     Input('select-perp-val', 'value')],
    []
)
def update_constraint_score_view(dataset_name, perp):
    if None in [dataset_name, perp]:
        return {'data': []}

    df = cf.datagen.lines(3, columns=['test1','test2','test4'], dateIndex=False)
    return _create_figure_from_df(df, perp)
