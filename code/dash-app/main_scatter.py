"""Main Scatter
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


def layout():
    stylesheet = [dict(
        selector='.img_node',
        style={
            'width':6, 'height':6,
            'background-fit':'cover',
            'background-image':'data(url)'
        }
    ),]

    return html.Div(style=dict(height='90vh'), children=[
        html.Div(id='list-selected-nodes', style={'display': 'none'}),
        #dbc.Button(id='btn-ml', children='Must-link'),
        cyto.Cytoscape(
            id='cytoplot',
            layout={'name': 'preset'},
            style={'width': '100%', 'height': '85vh'},
            stylesheet=stylesheet,
            elements=[],
            autoungrabify=True, # can not move nodes
            autounselectify=False, # can select nodes
        ),
        dcc.Markdown(id='msg-debug-graph', children='Debug graph'),
        dcc.Graph(id='scatter', style=dict(height='85vh')),
        # NOTE: dcc.Graph(stype={'height':'inherit'}) to make graph have
        # the same height as its parent.
    ])


def callback(app, embedding_dir, labels):
    @app.callback(
        Output('list-selected-nodes', 'children'),
        [Input('cytoplot', 'selectedNodeData')]
    )
    def on_multi_select(list_points):
        if not list_points:
            return ""
        list_ids = list(map(lambda p: p['id'], list_points))
        return json.dumps(list_ids)


    @app.callback(
        Output('cytoplot', 'elements'),
        [Input('btn-ml', 'n_clicks'),
         Input('perp_val', 'value')],
        [State('list-selected-nodes', 'children')]
    )
    def update_cytoplot(user_set_ml, perp, list_ids_str):
        if perp is None:
            return

        in_name = f"{embedding_dir}/DIGITS_perp={perp}.z"
        Z = joblib.load(in_name)
        elems = [dict(
            classes='img_node',
            data=dict(id=str(idx), xname=f"Node_{idx}",
                      url=f"/static/svg/DIGITS_color.svg#{idx}"),
            position=dict(x=x, y=y),
            #grabbable=False,
            #selectable=True,
        )for idx, [x, y] in enumerate(Z)]

        must_links = []
        if list_ids_str and list_ids_str != "":
            list_ids = json.loads(list_ids_str)
            must_links = [dict(
                data={'source': list_ids[0], 'target': list_ids[1]}
            )]

        return elems + must_links


    @app.callback(
        Output('scatter', 'figure'),
        [Input('perp_val', 'value')]
    )
    def update_scatter(perp):
        if perp is None:
            return

        in_name = f"{embedding_dir}/DIGITS_perp={perp}.z"
        Z = joblib.load(in_name)

        marker_src = [f"/static/MNIST.svg#{img_id}"
                      for img_id in range(len(Z))]

        trace0 = go.Scatter(
            x = Z[:,0],
            y = Z[:,1],
            customdata = marker_src,
            mode = 'markers',
            marker = dict(
                size = 4,
                opacity = 0.1,
                color = labels, # 'rgba(255, 182, 193, .3)',
                colorscale='Rainbow', # https://plot.ly/r/reference/#heatmap-colorscale
                line = dict(width = 0,),
            ),
            selected = dict (marker = dict ( opacity = 1.0) ),
        )

        layout = dict(
            title = f"Perplexity = {perp}",
            clickmode='event+select', # https://plot.ly/python/reference/#layout-clickmode
            yaxis = dict(zeroline = False),
            xaxis = dict(zeroline = False),
        )

        fig = dict(data=[trace0], layout=layout)
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
