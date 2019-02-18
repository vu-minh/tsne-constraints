import numpy as np

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State

import os
from flask import send_from_directory

import pandas as pd
import ssl


# avoid certificate-verify SSL error in urllib
ssl._create_default_https_context = ssl._create_unverified_context

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__) #, external_stylesheets=external_stylesheets)

# serving css locally
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

# for running app with `gunicorn`, should define `server` instance:
server = app.server


@app.server.route('/static/<path:path>')
def static_file(path):
    static_folder = os.path.join(os.getcwd(), 'static')
    return send_from_directory(static_folder, path)


# load external data file
df_arg = pd.read_csv(
    'https://gist.githubusercontent.com/chriddyp/'
    'c78bf172206ce24f77d6363a2d754b59/raw/'
    'c353e8ef842413cae56ae3920b8fd78468aa4cb2/'
    'usa-agricultural-exports-2011.csv')

df_country = pd.read_csv(
    'https://gist.githubusercontent.com/chriddyp/'
    'cb5392c35661370d95f300086accea51/raw/'
    '8e0768211f6b747c0db42a9ce9a0937dafcbd8b2/'
    'indicators.csv')
available_indicators = df_country['Indicator Name'].unique()


def generate_table(df, max_rows=10):
    '''Utill function to generate table
    '''
    return html.Table(
        # header
        [html.Tr([html.Th(col) for col in df.columns])] +

        # body
        [ html.Tr([
            html.Td(df.iloc[i][col]) for col in df.columns
        ]) for i in range(min(len(df), max_rows)) ]
    )


def generate_bar_chart():
    '''Util function to generate simple bar chart, using dash-core-component
    '''
    return dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
            ],
            'layout': {
                'title': 'Dash Data Visualization'
            }
        }
    )

# app layout
app.layout = html.Div(children=[
    html.Link(
        rel='stylesheet',
        href='/static/dash-app.css'
    ),

    # Scatter plot indicators selection
    html.Div([
        html.Div([dcc.Dropdown(
            id='xaxis-column',
            options=[{'label': i, 'value': i} for i in available_indicators],
            value='Fertility rate, total (births per woman)'
        )], style={'width': '40%', 'display': 'inline-block'}),
        html.Div([dcc.Dropdown(
            id='yaxis-column',
            options=[{'label': i, 'value': i} for i in available_indicators],
            value='Life expectancy at birth, total (years)'
        )], style={'width': '40%', 'display': 'inline-block'}),
        html.Button(id='btn-update-chart', children='Update Chart'),
    ], style={'display': 'inline-block', 'width': '100%'}),

    # Scatter plot
    html.Div([
        dcc.Graph(
            id='scatter',
            hoverData={'points': [{'customdata': 'Japan'}]}
        )
    ], style={'display': 'block', 'width': '100%', 'align': 'center'}),


    # Interactive Input elements
    html.Div([
        html.H2('Component with callback'),
        dcc.Input(id='user-input', value='Enter some values', type='text'),
        dcc.Input(id='passive-input', value='Input without firing callback', type='text'),
        html.Button(id='btn-send', n_clicks=0, children='Submit'),
        html.Div(id='debug-text')
    ]),


    # Show table from agriculture datasets
    generate_table(df_arg, max_rows=5),


    # show bar chart from hard-coded data
    generate_bar_chart()
])


# connect the gui component by `callback`
@app.callback(
    Output(component_id='debug-text', component_property='children'),
    [Input('btn-send', 'n_clicks'),
     Input(component_id='user-input', component_property='value')],
    [State('passive-input', 'value')]
)
def update_user_debug_text(btn_n_clicks, active_input, passive_input):
    return '''
        Button clicked {} times.
        Active input = {}.
        Passive input = {}
    '''.format(btn_n_clicks, active_input, passive_input)


# update scatter plot
@app.callback(
    Output('scatter', 'figure'),
    [Input('btn-update-chart', 'n_clicks')],
    [State('xaxis-column', 'value'),
     State('yaxis-column', 'value')]
)
def update_graph(a,b,c):
    N = 500

    trace0 = go.Scatter(
        x = np.random.randn(N),
        y = np.random.randn(N)+2,
        name = 'Above',
        mode = 'markers',
        marker = dict(
            size = 10,
            color = 'rgba(152, 0, 0, .8)',
            line = dict(
                width = 2,
                color = 'rgb(0, 0, 0)'
            )
        )
    )

    trace1 = go.Scatter(
        x = np.random.randn(N),
        y = np.random.randn(N)-2,
        name = 'Below',
        mode = 'markers',
        marker = dict(
            size = 10,
            color = 'rgba(255, 182, 193, .9)',
            line = dict(
                width = 2,
            )
        )
    )

    data = [trace0, trace1]

    layout = dict(title = 'Styled Scatter',
                  yaxis = dict(zeroline = False),
                  xaxis = dict(zeroline = False)
                 )

    fig = dict(data=data, layout=layout)
    return fig


def update_graph2(btn_n_clicks, xaxis_column_name, yaxis_column_name):
    print(xaxis_column_name, yaxis_column_name)
    dff = df_country[df_country['Year']==2007]

    return {
        'data': [go.Scatter(
            x=dff[dff['Indicator Name']==xaxis_column_name]['Value'],
            y=dff[dff['Indicator Name']==yaxis_column_name]['Value'],
            text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
            customdata=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
            mode='markers',
            marker={
                'size': 15,
                'opacity': 0.5,
                'line': {'width': 0.5, 'color': 'white'}
            }
        )],
        'layout': go.Layout(
            xaxis={
                'title': xaxis_column_name,
                'type': 'linear'
            },
            yaxis={
                'title': yaxis_column_name,
                'type': 'log'
            },
            height=480,
            hovermode='closest'
        )
    }


if __name__ == '__main__':
    app.run_server(debug=True)
