"""Link buttons: render 2 buttons Must-link and Cannot-link
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State


def layout():
    return html.Div([
        html.Div([
            html.Button(id='btn-ml', children='Must-link'),
            html.Div(id='list-ml', children='List Must-links'),
        ]),

        html.Div([
            html.Button(id='btn-cl', children='Cannot-link'),
            html.Div(id='list-cl', children='List Cannot-links'),
        ]),
    ])

def callback(app):
    @app.callback(
        Output('list-ml', 'children'),
        [Input('btn-ml', 'n_clicks')],
        [State('scatter', 'selectedData')]
    )
    def create_ml(_, selected_data):
        selected_idn = list(map(lambda p: p['pointIndex'], selected_data['points'])) if selected_data else []
        return ', '.join(map(str, selected_idn))


    @app.callback(
        Output('list-cl', 'children'),
        [Input('btn-cl', 'n_clicks')],
        [State('scatter', 'selectedData')]
    )
    def create_cl(_, selected_data):
        selected_idn = list(map(lambda p: p['pointIndex'], selected_data['points'])) if selected_data else []
        return ', '.join(map(str, selected_idn))
