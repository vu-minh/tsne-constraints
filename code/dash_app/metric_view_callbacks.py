from dash.dependencies import Input, Output, State
from server import app
from data_filter import get_embedding
from dash.exceptions import PreventUpdate

import plotly.plotly as py
import cufflinks as cf
import pandas as pd
import numpy as np


#Available cufflinks thems: ['pearl', 'white', 'ggplot', 'solar', 'space']
cf.set_config_file(theme='white')


@app.callback(
    Output('metric-view', 'figure'),
    [Input('btn-metrics', 'n_clicks')]
)
def update_metric_view(btn_metrics):
    print('Update metric view: ', btn_metrics)

    if not btn_metrics:
        raise PreventUpdate

    # create test figure with cufflinks
    df = cf.datagen.lines(3,columns=['a','b','c'])

    figure = df.iplot(kind='scatter', asFigure=True,
                      subplots=True, shape=(3,1), shared_xaxes=True, fill=True,)

    figure['layout']['xaxis'].update({'title': 'Perplexity', 'tickprefix': '$'})
    # for i, trace in enumerate(figure['data']):
    #     trace['name'] = 'Trace {}'.format(i)

    return figure
