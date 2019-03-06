# https://community.plot.ly/t/splitting-callback-definitions-in-multiple-files/10583/2
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto

from server import app, server
import cytoplot_callbacks


list_datasets = [
    'DIGITS',
    'FASHION100', 'FASHION200',
    'QUICKDRAW100', 'QUICKDRAW200',
    'COIL20_100', 'COIL20_200'
]


list_perps = [5, 10, 20, 50, 100, 200, 500]


# ref for cytospace js style: http://js.cytoscape.org/
default_cyto_node_style = dict(
    selector='.img_node',
    style={
        # 'label': 'data(label)',
        # 'width': 6,
        # 'height': 6,
        'shape': 'rectangle',  # 'ellipse'
        'background-color': 'white',
        'background-fit': 'contain',  # 'cover',
        'background-image': 'data(url)'
    }
)


default_cyto_selected_node_style = dict(
    # auto supported selector: http://js.cytoscape.org/#selectors/state
    selector='node:selected',
    style={
        'width': 12,
        'height': 12,
        'shape': 'ellipse',
        'border-width': 1,
        'border-color': 'blue',
    }
)


default_cyto_selected_edge_style = dict(
    selector='edge:selected',
    style={
        'width': 1.5,
        'line-style': 'solid'
    }
)


default_cyto_edge_style = dict(
    selector='edge',
    style={
        'z-index': 100,  # does not work, edges are always rendered under nodes
        'width': 1,
    }
)


default_cyto_sim_link_style = dict(
    selector='.sim-link',
    style={
        'line-color': 'green',
        'line-style': 'dotted',
    }
)


default_cyto_dissim_link_style = dict(
    selector='.dissim-link',
    style={
        'line-color': 'red',
        'line-style': 'dotted',
    }
)


control_app_layout = html.Div([
    dcc.Dropdown(id='select_dataset', value='DIGITS',
                 options=[{'label': name, 'value': name}
                          for name in list_datasets]),
    dcc.Dropdown(id='perp_val', value=5,
                 options=[{'label': perp, 'value': perp}
                          for perp in list_perps]),
    dbc.Button(id='btn-sim', children='Similar', n_clicks_timestamp=0,
               outline=True, color='success', className='mr-2'),
    dbc.Button(id='btn-dissim', children='Dissimilar', n_clicks_timestamp=0,
               outline=True, color='danger', className='mr-2'),
    dbc.Button(id='btn-del-link', children='Delete link', n_clicks_timestamp=0,
               outline=True, color='primary', className='mr-2'),
])


control_cyto_layout = html.Div([
    dcc.RadioItems(
        id='select_cmap', value='gray_r',
        options=[{'label': label, 'value': value}
                 for label, value in [
                     ('Gray scale', 'gray'),
                     ('Gray scale invert', 'gray_r'),
                     ('Color', 'color')]],
        labelStyle={'display': 'inline-block'}
    ),
    dcc.Slider(
        id='slider_img_size', min=0.0, max=8.0, step=0.5,
        value=3.0, included=False,
        marks={i * 0.1: '' if i < 10 else i * 0.1
               for i in list(range(1, 6)) + list(range(10, 85, 5))}
    ),
])


cytoplot_layout = cyto.Cytoscape(
    id='cytoplot',
    layout={'name': 'preset', 'animate': True},
    style={'width': '100%', 'height': '85vh'},
    stylesheet=[
        default_cyto_node_style,
        default_cyto_edge_style,
        default_cyto_sim_link_style,
        default_cyto_dissim_link_style,
        default_cyto_selected_node_style,
        default_cyto_selected_edge_style
    ],
    elements=[],
    autoungrabify=True,  # can not move nodes
    autounselectify=False,  # can select nodes
)


app.layout = html.Div(style=dict(height='90vh'), children=[
    control_app_layout,
    control_cyto_layout,
    cytoplot_layout
])


if __name__ == '__main__':
    app.run_server(debug=True, threaded=True, processes=1)
