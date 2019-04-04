import json
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_html_components as html
import dash_bootstrap_components as dbc
from server import app
from tcommon import colorize_link


@app.callback(Output("links-memory", "data"), [Input("cytoplot", "elements")])
def store_links(elems):
    """When the elements (the links are selected) in the cytoplot changed,
    Store them in the browser's memory with key name "links-memory".
    """
    # TODO cannot distinguish which is the source of current event
    return [
        (e["data"]["source"], e["data"]["target"], e["classes"])
        for e in (elems or [])
        if e["group"] == "edges"
    ][::-1]


@app.callback(
    [Output("txt-debug", "children"), Output("list-links-view", "children")],
    [Input("links-memory", "modified_timestamp")],
    [
        State("links-memory", "data"),
        State("select-dataset", "value"),
        State("select-cmap", "value"),
    ],
)
def on_links_data(links_timestamp, links, dataset_name, cmap_type):
    """When the local storage "links-memory" changes,
    Update the list view "list-links-view", that is a list of all selected links.
    """
    if links_timestamp is None:
        raise PreventUpdate

    list_links_view = [
        dbc.ListGroupItem(
            id=str(link),
            children=_create_link_item(link, dataset_name, cmap_type),
            n_clicks_timestamp=0,
            action=True,
        )
        for link in links
    ]

    txt_debug = json.dumps(links or [], indent=2)
    return txt_debug, list_links_view


def _create_link_item(link, dataset_name, cmap_type):
    """Create item in ListGroupItem, which has 3 elements:
        + a 1st image for link[0] (id of first datapoint)
        + a line connect two images with color determined by link[2] ('sim' or 'dissim')
        + a 2nd image for link[1] (id of second datapoint)

    :link: list of string
    :dataset_name: name of dataset to get image
    :cmap_type: color map type: colored/gray/invert_gray

    :returns: a Div of 3 items
    """
    item1, item2, link_type = link
    img1 = html.Img(
        src=f"/static/svg/{dataset_name}_{cmap_type}.svg#{item1}", width="22%"
    )
    img2 = html.Img(
        src=f"/static/svg/{dataset_name}_{cmap_type}.svg#{item2}", width="22%"
    )
    line = html.Hr(
        style={
            "border": "1.5px solid " + colorize_link(link_type),
            "display": "inline-block",
            "width": "50%",
        }
    )
    item = html.Div([img1, line, img2])
    return item
