"""Main Scatter
"""
import joblib
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State


def layout():
    return html.Div([
        html.Pre(id='lbl_perp', children='Choosen perp'),

        html.Div([
            dcc.Graph(id='scatter')
        ],style=dict(width=1000, height=1200, display='inline-block')),
    ])


def callback(app, embedding_dir, labels):

    @app.callback(
        Output('scatter', 'figure'),
        [Input('perp_val', 'value')]
    )
    def update_scatter(perp):
        in_name = f"{embedding_dir}/DIGITS_perp={perp}.z"
        Z = joblib.load(in_name)
        
        trace0 = go.Scatter(
            x = Z[:,0],
            y = Z[:,1],
            mode = 'markers',
            marker = dict(
                size = 10,
                color = labels, # 'rgba(255, 182, 193, .3)',
                colorscale='Rainbow', # https://plot.ly/r/reference/#heatmap-colorscale
                line = dict(width = 0,)
            )
        )
        data = [trace0]

        layout = dict(title = 'Scatter',
                    clickmode='event+select', # https://plot.ly/python/reference/#layout-clickmode
                    yaxis = dict(zeroline = False),
                    xaxis = dict(zeroline = False),
                    autosize=False,
                    width=800,
                    height=800,
                    )

        fig = dict(data=data, layout=layout)
        return fig


    @app.callback(
        Output('lbl_perp', 'children'),
        [Input('scatter', 'clickData'),
        Input('scatter', 'selectedData')]
    )
    def on_select(click_data, selected_data):
        click_idn = map(lambda p: str(p['pointIndex']), click_data['points']) if click_data else []
        selected_idn = map(lambda p: str(p['pointIndex']), selected_data['points']) if selected_data else []
        return f"CLICK: {', '.join(click_idn)},\nSELECT: {', '.join(selected_idn)}"
