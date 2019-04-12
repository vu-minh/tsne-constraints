# https://community.plot.ly/t/splitting-callback-definitions-in-multiple-files/10583/2
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto

from server import app, server
import cytoplot_callbacks
import local_storage_callbacks
import metric_view_callbacks


list_datasets = ["FASHION200", "COIL20", "DIGITS", "QUICKDRAW500", "COUNTRY2014"]

list_base_perps = [10, 30, None]

###############################################################################
# cytoscape stylesheet
# ref for cytospace js style: http://js.cytoscape.org/
default_cyto_node_style = dict(
    selector=".img-node",
    style={
        "shape": "rectangle",  # 'ellipse'
        "border-color": "white",
        "overlay-opacity": 0,
        "background-color": "white",
        "background-fit": "contain",  # 'cover',
        "background-image": "data(url)",
    },
)

default_cyto_selected_node_style = dict(
    # auto supported selector: http://js.cytoscape.org/#selectors/state
    selector="node:selected",
    style={"shape": "ellipse", "border-width": 0.5, "border-color": "blue"},
)

default_cyto_selected_edge_style = dict(
    selector="edge:selected", style={"width": 0.4, "line-style": "dotted"}
)

default_cyto_edge_style = dict(
    selector="edge",
    style={
        "width": 0.2,
        "z-index": 100,  # does not work, edges are always rendered under nodes
    },
)

default_cyto_sim_link_style = dict(
    selector=".sim-link", style={"line-color": "green", "line-style": "solid"}
)

default_cyto_dis_link_style = dict(
    selector=".dis-link", style={"line-color": "red", "line-style": "solid"}
)

additional_cyto_css = []


###############################################################################
# layout components

INLINE = {"display": "inline-block"}

buttons_layout = {
    "btn-sim": ("Similar", "success"),
    "btn-dis": ("Dissimilar", "danger"),
    "btn-submit": ("Find best viz", "secondary"),
    "btn-del-link": ("Delete link", "primary"),
    "btn-auto": ("Generate constraints", "secondary"),
}

control_buttons = html.Div(
    [
        dbc.Button(
            id=btn_id,
            children=label,
            n_clicks_timestamp=0,
            outline=True,
            color=color,
            style={"padding": "4px", "margin": "4px"},
        )
        for btn_id, (label, color) in buttons_layout.items()
    ],
    className="mr-3",
)

cytoplot_layout = cyto.Cytoscape(
    id="cytoplot",
    layout={"name": "preset", "animate": True, "fit": True},
    style={"width": "100%", "height": "90vh"},
    stylesheet=[
        default_cyto_node_style,
        default_cyto_edge_style,
        default_cyto_sim_link_style,
        default_cyto_dis_link_style,
        default_cyto_selected_node_style,
        default_cyto_selected_edge_style,
    ]
    + additional_cyto_css,
    elements=[],
    autoungrabify=True,  # can not move nodes
    autounselectify=False,  # can select nodes
)

links_view_layout = dbc.ListGroup(
    id="list-links-view",
    children=[],
    style={
        "display": "inline-block",
        "overflow-y": "scroll",
        "width": "100%",
        "height": "80vh",
    },
)

debug_layout = html.Pre(
    id="txt-debug",
    children="Debug",
    style={"display": "inline", "overflow": "scroll", "border": "1px solid #ccc"},
)

non_floatting_bar = {"displayModeBar": False}

metric_view_layout = dcc.Graph(id="metric-view-chain", config=non_floatting_bar)
constraint_score_view_layout = dcc.Graph(
    id="constraint-score-view-chain", config=non_floatting_bar
)

metric_view_layout2 = dcc.Graph(id="metric-view-normal", config=non_floatting_bar)
constraint_score_view_layout2 = dcc.Graph(
    id="constraint-score-view-normal", config=non_floatting_bar
)


###############################################################################
# local storage for storing links
links_storage_memory = dcc.Store(id="links-memory", storage_type="memory")
links_storage_memory_debug = dcc.Store(id="links-memory-debug", storage_type="memory")
# best_perp_storage_memory = dcc.Store(id='best-perp-memory',
#                                      storage_type='memory')

###############################################################################
# metric anc score view on the right

right_layout_for_chain = html.Div(
    [
        html.H5("chain-tSNE"),
        dbc.Row(
            [
                dbc.Col([constraint_score_view_layout], md=6),
                dbc.Col([metric_view_layout], md=6),
            ]
        ),
    ],
    style=dict(height="40vh", paddingBottom="10vh"),
)

right_layout_for_normal = html.Div(
    [
        html.H5("tSNE normal"),
        dbc.Row(
            [
                dbc.Col([constraint_score_view_layout2], md=6),
                dbc.Col([metric_view_layout2], md=6),
            ]
        ),
    ],
    style=dict(height="40vh", paddingTop="5vh"),
)

##############################################################################
# control components in the navbar

select_dataset_name = dbc.FormGroup(
    [
        dbc.Label("Dataset name", html_for="select-dataset", className="mr-2"),
        dcc.Dropdown(
            id="select-dataset",
            value=None,
            options=[{"label": name, "value": name} for name in list_datasets],
            style={"width": "160px"},
        ),
    ],
    className="mr-3",
)

select_perplexity = dbc.FormGroup(
    [
        dbc.Label("Perplexity", html_for="select-perp-val", className="mr-2"),
        dcc.Dropdown(
            id="select-perp-val",
            value=30,
            options=[{"label": perp, "value": perp} for perp in range(1, 500)],
            style={"width": "80px"},
        ),
    ],
    className="mr-3",
)

select_base_perplexity = dbc.FormGroup(
    [
        dbc.Label("Base perlexity", html_for="select-base-perp-val", className="mr-2"),
        dcc.Dropdown(
            id="select-base-perp-val",
            value=30,
            options=[
                {"label": perp or "No chain", "value": perp} for perp in list_base_perps
            ],
            style={"width": "90px"},
        ),
    ],
    className="mr-3",
)

select_earlystop = dbc.FormGroup(
    [
        dbc.Label("Early-stop", html_for="select-earlystop-val", className="mr-2"),
        dcc.Dropdown(
            id="select-earlystop-val",
            value="",
            options=[
                {"label": "No", "value": ""},
                {"label": "Yes", "value": "_earlystop"},
            ],
            style={"width": "80px"},
            disabled=False,
        ),
    ],
    className="mr-3",
)

option_view_perp_scale = dbc.FormGroup(
    [
        dbc.Label("View perp", className="mr-2"),
        dcc.Dropdown(
            id="select-perp-scale",
            value="log",
            options=[
                {"label": "linear", "value": "linear"},
                {"label": "log", "value": "log"},
            ],
            style={"width": "120px"},
        ),
    ],
    className="mr-3",
)

option_select_scatter_color = dbc.FormGroup(
    [
        dbc.Label("Color item", className="mr-2"),
        dcc.Dropdown(
            id="select-cmap",
            value="gray_r",
            options=[
                {"label": label, "value": value}
                for label, value in [
                    ("Gray", "gray"),
                    ("Gray invert", "gray_r"),
                    ("Color", "color"),
                ]
            ],
            style={"width": "120px"},
        ),
    ],
    className="mr-3",
)

slider_select_scatter_zoom_factor = dbc.FormGroup(
    [
        dcc.Slider(
            id="slider-img-size",
            min=0.0,
            max=3.0,
            step=0.2,
            value=0.5,
            included=False,
            marks={
                i * 0.1: f"{i*0.1:.1f}"
                for i in list(range(1, 11, 2)) + list(range(12, 32, 2))
            },
        )
    ],
    className="mr-3",
)

data_control_form = dbc.Form(
    [select_dataset_name, select_base_perplexity], inline=True, style={"width": "28%"}
)

zoom_control_form = dbc.Form(
    [slider_select_scatter_zoom_factor], style={"width": "30%"}
)

display_control_form = dbc.Form(
    [
        select_perplexity,
        option_select_scatter_color,
        option_view_perp_scale,
        select_earlystop,
    ],
    inline=True,
    style={"width": "42%"},
)

navbar = dbc.Navbar(
    [data_control_form, zoom_control_form, display_control_form],
    style={"width": "100%"},
)

##############################################################################
# main app layout
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                navbar,
                links_storage_memory,
                links_storage_memory_debug,
                # best_perp_storage_memory
            ]
        ),
        dbc.Row(
            [
                dbc.Col([control_buttons, links_view_layout], md=2),
                dbc.Col([cytoplot_layout], md=6),
                dbc.Col([right_layout_for_chain, right_layout_for_normal], md=4),
            ]
        ),
        # dbc.Row([dbc.Col([bottom_layout])]),
    ],
    fluid=True,
)


if __name__ == "__main__":
    app.run_server(debug=True, threaded=True, host="0.0.0.0", processes=1)
