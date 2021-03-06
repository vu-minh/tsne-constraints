import json
from functools import partial
from itertools import combinations

import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

from server import app
from data_filter import get_embedding
from tcommon import get_image_url, get_auto_generated_constraints


def _add_or_edit_style_for_tap_node(tap_id, styles):
    """[NOT USED] use css selector to add style for node by id"""
    found = False
    target_node_selector = 'node[id="{}"]'.format(tap_id)
    selected_node_style = {"border-width": 1, "border-color": "blue"}

    for style in styles:
        if style["selector"].startswith("node[id="):
            style["selector"] = target_node_selector
            found = True
            break
    if not found:
        new_style = {"selector": target_node_selector, "style": selected_node_style}
        styles.append(new_style)
    return styles


@app.callback(
    Output("cytoplot", "stylesheet"),
    [Input("slider-img-size", "value")],
    [State("cytoplot", "stylesheet")],
)
def change_cyto_style(img_size, current_styles):
    style_list = current_styles
    if img_size:
        scale_factor = {
            ".img-node": 1.0,
            "node:selected": 1.4,
            "edge": 0.1,
            "edge:selected": 0.3,
        }
        for style in style_list:
            selector = style["selector"]
            if selector in scale_factor.keys():
                scaled_size = img_size * scale_factor[selector]
                style["style"]["width"] = scaled_size
                style["style"]["height"] = scaled_size
                style["style"]["border-width"] = 0.1 * scaled_size
    return style_list


def _build_cyto_nodes(dataset_name, perp, earlystop, base_perp, cmap_type):
    # TODO cache
    Z = get_embedding(dataset_name, perp, earlystop, base_perp)
    return [
        dict(
            group="nodes",
            classes="img-node",
            data=dict(
                id=str(idx),
                label=f"node_{idx}",
                url=get_image_url(dataset_name, idx, cmap_type),
            ),
            position=dict(x=x, y=y),
        )
        for idx, [x, y] in enumerate(Z)
    ]


def _create_or_edit_edges(old_edges, selected_nodes, edge_type):
    if (not edge_type) or (not selected_nodes) or len(selected_nodes) < 2:
        return old_edges

    pairs = combinations([n["id"] for n in selected_nodes], 2)
    new_edges = []
    for pair in pairs:
        existed = False
        for e in old_edges:
            if set((e["data"]["source"], e["data"]["target"])) == set(pair):
                e["data"]["source"], e["data"]["target"] = pair
                e["classes"] = edge_type
                existed = True

        if not existed:
            new_edges.append(
                {
                    "classes": edge_type,
                    "group": "edges",
                    "data": {"source": pair[0], "target": pair[1]},
                }
            )
    return old_edges + new_edges


def _delete_edges(old_edges, edges_to_del):
    new_edges = old_edges
    if edges_to_del:
        del_pairs = [set([e["source"], e["target"]]) for e in edges_to_del]
        new_edges = [
            e
            for e in old_edges
            if set([e["data"]["source"], e["data"]["target"]]) not in del_pairs
        ]
    return new_edges


def _load_auto_generated_constraints(dataset_name, n_sim=50, n_dis=50):
    links = get_auto_generated_constraints(dataset_name, n_sim, n_dis)
    print(links)
    return [
        {
            "classes": "sim-link" if t == 1 else "dis-link",
            "group": "edges",
            "data": {"source": str(p1), "target": str(p2)},
        }
        for p1, p2, t in links
    ]


@app.callback(
    Output("cytoplot", "elements"),
    [
        Input("btn-sim", "n_clicks_timestamp"),
        Input("btn-dis", "n_clicks_timestamp"),
        Input("btn-del-link", "n_clicks_timestamp"),
        Input("btn-auto", "n_clicks_timestamp"),
        Input("select-dataset", "value"),
        Input("select-perp-val", "value"),
        Input("select-base-perp-val", "value"),
        Input("select-cmap", "value"),
        Input("select-earlystop-val", "value"),
    ],
    [
        State("cytoplot", "selectedNodeData"),
        State("cytoplot", "selectedEdgeData"),
        State("cytoplot", "elements"),
    ],
)
def update_cytoplot(
    btn_sim,
    btn_dis,
    btn_del,
    btn_auto,
    dataset_name,
    perp,
    base_perp,
    cmap_type,
    earlystop,
    selected_nodes,
    selected_edges,
    current_elems,
):
    # update new function of dash 0.38: callback_context
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    if None in [perp, dataset_name, cmap_type, earlystop]:
        return []

    # always update new nodes (to update img_size, update position or cmap)
    nodes = _build_cyto_nodes(dataset_name, perp, earlystop, base_perp, cmap_type)
    # filter the current edges
    old_edges = [e for e in current_elems if e["group"] == "edges"]

    # get lastest triggered event (note, in the doc, it takes the first elem)
    # https://dash.plot.ly/faqs
    # we can also access to `ctx.stages` and `ctx.inputs`
    last_event = ctx.triggered[-1]
    last_btn = last_event["prop_id"].split(".")[0]
    print("Last button: ", last_btn)

    actions_on_edges = {
        "btn-sim": partial(
            _create_or_edit_edges, old_edges, selected_nodes, "sim-link"
        ),
        "btn-dis": partial(
            _create_or_edit_edges, old_edges, selected_nodes, "dis-link"
        ),
        "btn-del-link": partial(_delete_edges, old_edges, selected_edges),
        "btn-auto": partial(_load_auto_generated_constraints, dataset_name),
    }

    if last_btn in actions_on_edges:
        edges = actions_on_edges[last_btn]()
    else:  # no button clicked
        edges = old_edges

    return nodes + edges
