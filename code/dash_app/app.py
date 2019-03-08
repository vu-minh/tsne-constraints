# https://community.plot.ly/t/splitting-callback-definitions-in-multiple-files/10583/2
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto

from server import app, server
import cytoplot_callbacks
import local_storage_callbacks
import metric_view_callbacks


list_datasets = [
   #  'DIGITS',
   'FASHION200',
   #  'FASHION100', 'FASHION500',
   #  'QUICKDRAW100', 'QUICKDRAW200',
   #  'COIL20_100', 'COIL20_200'
]

###############################################################################
# cytoscape stylesheet
# ref for cytospace js style: http://js.cytoscape.org/
default_cyto_node_style = dict(
    selector='.img-node',
    style={
        'shape': 'rectangle',  # 'ellipse'
        'border-color': 'white',
        'overlay-opacity': 0,
        'background-color': 'white',
        'background-fit': 'contain',  # 'cover',
        'background-image': 'data(url)',
    }
)

default_cyto_selected_node_style = dict(
    # auto supported selector: http://js.cytoscape.org/#selectors/state
    selector='node:selected',
    style={
        'shape': 'ellipse',
        'border-width': 0.5,
        'border-color': 'blue',
    }
)

default_cyto_selected_edge_style = dict(
    selector='edge:selected',
    style={
        'width': 0.4,
        'line-style': 'dotted'
    }
)

default_cyto_edge_style = dict(
    selector='edge',
    style={
        'width': 0.2,
        'z-index': 100,  # does not work, edges are always rendered under nodes
    }
)

default_cyto_sim_link_style = dict(
    selector='.sim-link',
    style={
        'line-color': 'green',
        'line-style': 'solid',
    }
)

default_cyto_dis_link_style = dict(
    selector='.dis-link',
    style={
        'line-color': 'red',
        'line-style': 'solid',
    }
)

additional_cyto_css = []


###############################################################################
# layout components

control_app_layout = html.Div([
    dcc.Dropdown(id='select-dataset', value='FASHION200',
                 options=[{'label': name, 'value': name}
                          for name in list_datasets]),
    dcc.Dropdown(id='select-perp-val', value=30,
                 options=[{'label': perp, 'value': perp}
                          for perp in range(1, 500)]),
])

control_cyto_layout = html.Div([
    dbc.Button(id='btn-sim', children='Similar', n_clicks_timestamp=0,
               outline=True, color='success', className='mr-2'),
    dbc.Button(id='btn-dis', children='Dissimilar', n_clicks_timestamp=0,
               outline=True, color='danger', className='mr-2'),
    dbc.Button(id='btn-del-link', children='Delete link', n_clicks_timestamp=0,
               outline=True, color='primary', className='mr-2'),
    dbc.Button(id='btn-submit', children='Find best viz',
               outline=True, className='mr-2'),
    dbc.Button(id='btn-auto-constraint', children='Generate Constraints',
               outline=True, className='mr-2')
])

cytoplot_option_layout = html.Div([
    dcc.RadioItems(
        id='select-cmap', value='gray_r',
        options=[{'label': label, 'value': value}
                 for label, value in [
                     ('Gray', 'gray'),
                     ('Gray invert', 'gray_r'),
                     ('Color', 'color')]],
        labelStyle={'display': 'inline-block'}
    ),
    dcc.Slider(
        id='slider-img-size', min=0.0, max=3.0, step=0.2,
        value=1.0, included=False,
        marks={i * 0.1: '{:.1f}'.format(i * 0.1)
               for i in list(range(1, 12)) + list(range(12, 32, 2))},
    )
], style={'display': 'inline'})

cytoplot_layout = cyto.Cytoscape(
    id='cytoplot',
    layout={'name': 'preset', 'animate': True, 'fit': True},
    style={'width': '100%', 'height': '95vh'},
    stylesheet=[
        default_cyto_node_style,
        default_cyto_edge_style,
        default_cyto_sim_link_style,
        default_cyto_dis_link_style,
        default_cyto_selected_node_style,
        default_cyto_selected_edge_style
    ] + additional_cyto_css,
    elements=[],
    autoungrabify=True,  # can not move nodes
    autounselectify=False,  # can select nodes
)

links_view_layout = dbc.ListGroup(id='list-links-view', children=[])

debug_layout = html.Pre(id='txt-debug', children='Debug',
                        style={'display': 'inline', 'overflow': 'scroll',
                               'border': '1px solid #ccc'})

metric_view_layout = dcc.Graph(id='metric-view')
constraint_score_view_layout = dcc.Graph(id='constraint-score-view')

###############################################################################
# local storage for storing links
links_storage_memory = dcc.Store(id='links-memory', storage_type='memory')
# best_perp_storage_memory = dcc.Store(id='best-perp-memory',
#                                      storage_type='memory')

###############################################################################
# app layout

left_layout = html.Div([
    control_app_layout,
    control_cyto_layout,
    links_view_layout,
    # debug_layout
], style=dict(height='90vh'))

right1_layout = html.Div([
    constraint_score_view_layout
], style=dict(height='90vh'))

right2_layout = html.Div([
    metric_view_layout
], style=dict(height='90vh'))

center_layout = html.Div([
    cytoplot_option_layout,
    cytoplot_layout,
], style=dict(height='90vh'))

bottom_layout = html.Div([
    debug_layout
    # metric_view_layout
], style=dict(height='10vh'))

app.layout = dbc.Container([
    dbc.Row([
        links_storage_memory,
        # best_perp_storage_memory
    ]),
    dbc.Row([
        dbc.Col([left_layout], md=2),
        dbc.Col([center_layout], md=6),
        dbc.Col([right1_layout], md=2),
        dbc.Col([right2_layout], md=2),
    ]),
    dbc.Row([
        dbc.Col([bottom_layout])
    ])
], fluid=True)


if __name__ == '__main__':
    app.run_server(debug=True, threaded=True, processes=1)
