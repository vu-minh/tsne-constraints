"""Main app
"""
import os
import joblib
import json
import numpy as np

from flask import send_from_directory
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State

import pandas as pd
import ssl

from app_layout import layout as main_layout
from link_buttons import callback as link_buttons_callback
from main_scatter import callback as main_scatter_callback


# avoid certificate-verify SSL error in urllib
ssl._create_default_https_context = ssl._create_unverified_context

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# for running app with `gunicorn`, should define `server` instance:
server = app.server

# serving css locally
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

@app.server.route('/static/<path:path>')
def static_file(path):
    static_folder = os.path.join(os.getcwd(), 'static')
    return send_from_directory(static_folder, path)

###############################################################################

embedding_dir = './embeddings'

from sklearn.datasets import load_digits
X, labels = load_digits(return_X_y=True)


###############################################################################
# app layout, combine layouts of all component


app.layout = main_layout

###############################################################################
# callback of all components

main_scatter_callback(app, embedding_dir, labels)
link_buttons_callback(app)


if __name__ == '__main__':
    app.run_server(debug=True)
