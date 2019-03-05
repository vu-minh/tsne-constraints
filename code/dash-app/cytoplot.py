"""Using Cytospace to make scatter with images
"""
import json
import joblib
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State


list_datasets = [
    'DIGITS',
    'FASHION100', 'FASHION200',
    'QUICKDRAW100', 'QUICKDRAW200',
    'COIL20_100', 'COIL20_200'
]

default_cyto_node_stylesheet = dict(
    selector='.img_node',
    style={
        'width':6, 'height':6,
        'background-color':'white',
        'background-fit':'cover',
        'background-image':'data(url)'
    }
)

def create_imgsize_slider():
    return dcc.Slider(
        id='slider_img_size',
        min=2, max=18, value=6,
        marks={i:i for i in range(2,19,2)},
    )



def layout():
    return html.Div(style=dict(height='90vh'), children=[
        html.Div(id='list-selected-nodes', style={'display': 'none'}),
        html.Div([
            dcc.Dropdown(id='select_dataset', value='DIGITS',
                         options=[{'label': name, 'value': name}
                                  for name in list_datasets]),
            dbc.Button(id='btn-sim', children='Similar pair',
                       outline=True, color='success', className='mr-2'),
            dbc.Button(id='btn-dissim', children='Dissimilar pair',
                       outline=True, color='danger', className='mr-2'),
            dcc.RadioItems(id='select_cmap', value='gray_r',
                           options=[
                               {'label': 'Gray scale', 'value': 'gray'},
                               {'label': 'Gray scale invert', 'value': 'gray_r'},
                               {'label': 'Color', 'value': 'color'}
                           ],
                           labelStyle={'display': 'inline-block'}
            ),
            dcc.Slider(id='slider_img_size', min=0.0, max=8.0, step=0.5,
                       value=3.0, included=False,
                       marks={i * 0.1 : '' if i < 10 else i * 0.1
                              for i in list(range(1,6)) + list(range(10,85,5))
                       }),
        ]),
        cyto.Cytoscape(
            id='cytoplot',
            layout={'name': 'preset', 'animate': True},
            style={'width': '100%', 'height': '85vh'},
            stylesheet=[default_cyto_node_stylesheet],
            elements=[],
            autoungrabify=True, # can not move nodes
            autounselectify=False, # can select nodes
        ),
        dcc.Markdown(id='msg-debug-graph', children='Debug graph'),
    ])


def callback(app, embedding_dir, labels):
    @app.callback(
        Output('cytoplot', 'stylesheet'),
        [Input('slider_img_size', 'value')],
        []
    )
    def change_cyto_style(img_size):
        node_style = default_cyto_node_stylesheet
        if img_size:
            node_style['style']['width'] = img_size
            node_style['style']['height'] = img_size
        return [node_style]

    def _build_cyto_nodes(dataset_name, perp, cmap_type):
        in_name = f"{embedding_dir}/{dataset_name}_perp={perp}.z"
        Z = joblib.load(in_name)[:200,:]
        return [dict(
            classes='img_node', group='nodes',
            data=dict(id=str(idx), xname=f"Node_{idx}",
                      url=f"/static/svg/COIL20_200_{cmap_type}.svg#{idx}"),
            position=dict(x=x, y=y),
        )for idx, [x, y] in enumerate(Z)]

    def _create_new_cyto_edges(selected_nodes, edge_type):
        print(selected_nodes)
        return [dict(classes=edge_type, group='edges',
                     data={'source': selected_nodes[0]['id'],
                           'target': selected_nodes[1]['id']})]

    @app.callback(
        Output('cytoplot', 'elements'),
        [Input('btn-sim', 'n_clicks'),
         Input('btn-dissim', 'n_clicks'),
         Input('select_dataset', 'value'),
         Input('perp_val', 'value'),
         Input('select_cmap', 'value'),
        ],
        [State('cytoplot', 'selectedNodeData'),
         State('cytoplot', 'elements')]
    )
    def update_cytoplot(sim_pair_clicked, dissim_pair_clicked,
                        dataset_name, perp, cmap_type,
                        selected_nodes, current_elems):
        if None in [perp, dataset_name, cmap_type]:
            return

        # always update new nodes (to update img_size, update position or cmap)
        nodes = _build_cyto_nodes(dataset_name, perp, cmap_type)
        # filter the current edges
        edges = [elem for elem in current_elems if elem['group']=='edges']

        new_edge = []
        if sim_pair_clicked and selected_nodes:
            new_edge = _create_new_cyto_edges(selected_nodes, 'sim')
        if dissim_pair_clicked and selected_nodes:
            new_edge = _create_new_cyto_edges(selected_nodes, 'dissim')

        return nodes + edges + new_edge
