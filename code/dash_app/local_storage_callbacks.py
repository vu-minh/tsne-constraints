import json
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_html_components as html
import dash_bootstrap_components as dbc
from server import app
from tcommon import colorize_link, get_image_elem


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
    ]  # [::-1] # will reverse list when showing it


@app.callback(
    [Output("txt-debug", "children"), Output("list-links-view", "children")],
    [
        Input("links-memory", "modified_timestamp"),
        Input("links-memory-debug", "modified_timestamp"),
    ],
    [
        State("links-memory", "data"),
        State("links-memory-debug", "data"),
        State("select-dataset", "value"),
        State("select-cmap", "value"),
    ],
)
def on_links_data(
    links_timestamp,
    links_debug_timestamp,
    links,
    links_with_score,
    dataset_name,
    cmap_type,
):
    """When the local storage "links-memory" changes,
    Update the list view "list-links-view", that is a list of all selected links.
    For debug, the `links_with_score` will store additional values `q_ij` and `log q_ij`
    `links-memory-debug` local storage will be updated when we find the best perp.
    """
    if links_timestamp is None:
        raise PreventUpdate

    merged_links = _merge_links(links, (links_with_score or []))
    list_links_view = [
        dbc.ListGroupItem(
            id=str(link),
            children=_create_link_item(idx + 1, link, dataset_name, cmap_type),
            n_clicks_timestamp=0,
            action=True,
        )
        for idx, link in reversed(list(enumerate(merged_links)))
    ]

    txt_debug = json.dumps(merged_links, indent=2)
    return txt_debug, list_links_view


def _merge_links(links1, links2=None):
    """Merge `links-memory` (with item of length 3) with
    `links-memory-debug` (with item of length 5)
    """
    merged_links = []
    for item1, item2, link_type in links1:
        q_ij, score_ij = None, None
        for i1, i2, t, q, log_q in links2:
            if (item1, item2) == (i1, i2) or (item1, item2) == (i2, i1):
                q_ij, score_ij = q, log_q
                break
        merged_links.append([item1, item2, link_type, q_ij, score_ij])
    return merged_links


def _create_link_item(idx, link, dataset_name, cmap_type):
    """Create item in ListGroupItem, which has 3 elements:
        + a 1st image for link[0] (id of first datapoint)
        + a line connect two images with color determined by link[2] ('sim' or 'dissim')
        + a 2nd image for link[1] (id of second datapoint)
        + additional debug info `q_ij` and `log q_ij` in string format, default to None

    :link: list of string
    :dataset_name: name of dataset to get image
    :cmap_type: color map type: colored/gray/invert_gray

    :returns: a Div of 3 items
    """
    item1, item2, link_type, q_ij, score_ij = link

    img1 = get_image_elem(dataset_name, item1, cmap_type, width="20%")
    img2 = get_image_elem(dataset_name, item2, cmap_type, width="20%")
    line = html.Div(
        (
            html.Div(
                [f"q={q_ij}", html.Br(), f"log={score_ij}"],
                style={
                    "text-align": "center",
                    "font-size": "12px",
                    "color": colorize_link(link_type),
                },
            )
            if q_ij is not None
            else html.Hr(style={"border": "1.5px solid " + colorize_link(link_type)})
        ),
        style={"display": "inline-block", "width": "50%"},
    )

    id_text = html.Div(
        idx, style={"display": "inline-block", "text-align": "center", "padding": "4px"}
    )
    item = html.Div([id_text, img1, line, img2])
    return item
