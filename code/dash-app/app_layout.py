import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

from link_buttons import layout as link_buttons_layout
from main_scatter import layout as main_scatter_layout


def _scatter_plot_with_control_buttons():
    return html.Div([
        link_buttons_layout(),
        main_scatter_layout(),
    ])


def _constraint_view():
    return html.Div([html.Pre(id='lbl_perp', children='Choosen perp')])


def _metrics_view():
    perps = [5, 10, 20, 50, 100]

    return html.Div([
        dcc.Dropdown(
            id='perp_val',
            options=[{'label': perp, 'value': perp} for perp in perps],
            value=perps[0],
        )
    ], id='metric_view')


def _body():
    return dbc.Container([
        dbc.Row([
            dbc.Col([_constraint_view()], md=3),
            dbc.Col([_scatter_plot_with_control_buttons()], md=6),
            dbc.Col([_metrics_view()], md=3)
        ]),
        dbc.Row([dbc.Col("Status bar")])
    ], fluid=True)


def _navbar():
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Link", href="#")),
            dbc.DropdownMenu(
                nav=True,
                in_navbar=True,
                label="Menu",
                children=[
                    dbc.DropdownMenuItem("Entry 1"),
                    dbc.DropdownMenuItem("Entry 2"),
                    dbc.DropdownMenuItem(divider=True),
                    dbc.DropdownMenuItem("Entry 3"),
                ],
            ),
        ],
        brand="t-SNE with constraints :D",
        brand_href="#",
        sticky="top",
    )


def _local_css():
    return html.Link(
        rel='stylesheet',
        href='/static/dash-app.css'
    )


def layout():
    return html.Div([
        # _local_css(),
        # _navbar(),
        _body(),
    ])
