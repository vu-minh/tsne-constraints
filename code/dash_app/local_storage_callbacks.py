import json
import joblib
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from server import app


@app.callback(
    Output('links-memory', 'data'),
    [Input('cytoplot', 'elements')]
)
def store_links(elems):
    return [(e['data']['source'], e['data']['target'], e['classes'])
            for e in (elems or []) if e['group'] == 'edges'][::-1]


@app.callback(
    [Output('txt-debug', 'children'),
     Output('list-links-view', 'children')],
    [Input('links-memory', 'modified_timestamp')],
    [State('links-memory', 'data')]
)
def on_links_data(ts, links):
    if ts is None:
        raise PreventUpdate
    txt_debug = json.dumps(links or [], indent=2)
    list_links_view = [dbc.ListGroupItem(id=str(link), children=str(link),
                                         n_clicks_timestamp=0, action=True)
                       for link in links]

    return txt_debug, list_links_view


@app.callback(
    Output('select-perp-val', 'value'),
    [Input('btn-submit', 'n_clicks')],
    [State('links-memory', 'data')]
)
def update_selected_perp(is_submit, links):
    print('User constraints: ', links)
    if not is_submit:
        return 20  # default perp value
    return 50
