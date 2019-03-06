import json
import joblib
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from server import app


@app.callback(
    Output('links_memory', 'data'),
    [Input('cytoplot', 'elements')]
)
def store_links(elems):
    return [(e['data']['source'], e['data']['target'], e['classes'])
            for e in (elems or []) if e['group'] == 'edges'][::-1]


@app.callback(
    Output('txt_debug', 'children'),
    [Input('links_memory', 'modified_timestamp')],
    [State('links_memory', 'data')]
)
def on_links_data(ts, links):
    if ts is None:
        raise PreventUpdate
    return json.dumps(links or [], indent=2)
