"""
Produce scatter plots
"""

import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from app import app
from . import utils
from .constants import *


def plot_scatter(data, x, y, filter_columns, union=True):
    # Initialize figure object
    fig = go.Figure()
    if len(filter_columns) > 0:
        # Apply filtering from user selections
        bright, dim = utils.filter_data(data, filter_columns, union=union)
        utils.add_scatter(fig, data=bright, x=x, y=y, markersize=MARKER_SIZE)
        utils.add_scatter(fig, data=dim, x=x, y=y, markersize=MARKER_SIZE, opacity=0.15)
    else:
        utils.add_scatter(fig, data=data, x=x, y=y, markersize=MARKER_SIZE)
    # Assign figure properties
    fig.update_layout(
        width=1200,
        height=800,
        xaxis_title=FEATURES[x],
        yaxis_title=FEATURES[y],
        **PLOT_KWARGS
    )
    fig.update_xaxes(**AXES_KWARGS)
    fig.update_yaxes(**AXES_KWARGS)
    return fig


df = pd.read_csv(DATA_PATH)
genres = [c for c in df.columns if 'genre_' in c]

layout = html.Div([
    html.Div(
        [
            utils.get_header(NUM_BANDS),
            utils.get_caption(NUM_BANDS, NUM_WORDS),
            utils.make_features_dropdown_div(
                label="X-axis",
                id="scatter-dropdown-x",
                features=FEATURES,
                value='words_per_song',
            ),
            utils.make_features_dropdown_div(
                label="Y-axis",
                id="scatter-dropdown-y",
                features=FEATURES,
                value='words_per_song_uniq',
            ),
            utils.make_genre_dropdown_div(
                id="scatter-dropdown-genre",
                genres={g: utils.get_genre_label(df, g) for g in genres},
            ),
            utils.make_radio_div("scatter-radio-genre"),
            dcc.Loading(
                id="scatter-plot-loading",
                type='default',
                children=dcc.Graph(id="scatter-graph"),
            ),
        ],
    )
], style={})


@app.callback(
    Output("scatter-graph", "figure"),
    [Input("scatter-dropdown-x", "value"),
     Input("scatter-dropdown-y", "value"),
     Input("scatter-dropdown-genre", "value"),
     Input("scatter-radio-genre", "value")])
def display_plot(x, y, z, selection):
    if z is None:
        z = []
    fig = plot_scatter(df, x, y, z, union=(selection == 'union'))
    return fig
