"""
Produce scatter plots
"""

import pandas as pd
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from app import app
from . import utils
from .config import *


def plot_scatter(data, x, y, filter_columns, filter_pattern, union=True):
    # Initialize figure object
    fig = go.Figure()
    if len(filter_columns) > 0 or filter_pattern is not None:
        # Apply filtering from user selections
        bright, dim = utils.filter_data(data, filter_columns, filter_pattern, union=union)
        utils.add_scatter(fig, data=bright, x=x, y=y, markersize=MARKER_SIZE)
        utils.add_scatter(fig, data=dim, x=x, y=y, markersize=MARKER_SIZE, opacity=0.15)
    else:
        utils.add_scatter(fig, data=data, x=x, y=y, markersize=MARKER_SIZE)
    # Assign figure properties
    fig.update_layout(
        width=1000,
        height=560,
        xaxis_title=FEATURES[x],
        yaxis_title=FEATURES[y],
        **PLOT_KWARGS
    )
    fig.update_xaxes(**AXES_KWARGS)
    fig.update_yaxes(**AXES_KWARGS)
    return fig


df = pd.read_csv(LYRICAL_COMPLEXITY_PATH)
tfidf = pd.read_csv(WORDCLOUD_PATH, index_col=0)
genres = [c for c in df.columns if 'genre_' in c]

controls_card = utils.make_controls_card([
    html.Div([
        dcc.Link('Swarm plots', href='/apps/swarm'),
        html.Label('|', style={'margin-left': 10, 'margin-right': 10}),
        dcc.Link('Scatter plots', href='/apps/scatter'),
    ]),
    utils.get_caption(NUM_BANDS, NUM_WORDS),
    utils.make_features_dropdown_div(
        label="X-axis:",
        id="scatter-dropdown-x",
        features=FEATURES,
        value='words_per_song',
    ),
    utils.make_features_dropdown_div(
        label="Y-axis:",
        id="scatter-dropdown-y",
        features=FEATURES,
        value='words_per_song_uniq',
    ),
    utils.make_genre_dropdown_div(
        id="scatter-dropdown-genre",
        genres={g: utils.get_genre_label(df, g) for g in genres},
    ),
    utils.make_radio_div("scatter-radio-genre"),
    utils.make_band_input_div(id="scatter-input-band"),
])

wordcloud_card = utils.make_wordcloud_card("scatter")

layout = html.Div([
    utils.get_header(NUM_BANDS),
    dbc.Card([
        dbc.CardBody(
            dcc.Loading(
                id="scatter-plot-loading",
                type='default',
                children=dcc.Graph(id="scatter-graph"),
            ),
            style={'height': 600, 'margin': '0 auto'}
        ),
    ]),
    dbc.Card([
        dbc.CardBody(
            dbc.Row([
                dbc.Col([controls_card], md=7),
                dbc.Col([wordcloud_card], md=5),
            ], style={'height': '100%'}),
        )
    ], style={'height': 440, 'margin-bottom': 5}),
])


@app.callback(
    Output("scatter-graph", "figure"),
    [Input("scatter-dropdown-x", "value"),
     Input("scatter-dropdown-y", "value"),
     Input("scatter-dropdown-genre", "value"),
     Input("scatter-radio-genre", "value"),
     Input("scatter-input-band", "value"),])
def display_plot(x, y, z, selection, search):
    if z is None:
        z = []
    fig = plot_scatter(df, x, y, z, search, union=(selection == 'union'))
    return fig

@app.callback(
    [Output("scatter-word-cloud-image", "src"),
     Output("scatter-word-cloud-header", "children")],
    [Input("scatter-graph", "clickData")])
def make_image(click):
    return utils.make_image(tfidf, click)
