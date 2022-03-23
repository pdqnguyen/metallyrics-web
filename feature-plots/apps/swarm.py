"""
Produce swarm plots
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from app import app
from . import utils
from .config import *


plt.switch_backend('Agg')


def get_swarm_data(series, figsize, markersize):
    """Produce a seaborn swarm plot and return the plot coordinates and properties.
    """
    fig = plt.figure(figsize=figsize)
    ax = sns.swarmplot(x=series, size=markersize)
    swarm_pts = ax.collections[0].get_offsets()
    swarm_data = pd.DataFrame(
        swarm_pts,
        index=series.index,
        columns=['x', 'y']
    )
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    swarm_props = {
        'xlim': ax.get_xlim(),
        'ylim': ax.get_ylim(),
        'axsize': (bbox.width * fig.dpi, bbox.height * fig.dpi),
        'markersize': markersize
    }
    plt.close()
    return swarm_data, swarm_props


def plot_scatter(data, filter_columns, filter_pattern, sns_props, union=True):
    """Initialize `plotly.graph_objects.Figure` object which will contain swarm plots.
    """
    xlim = sns_props['xlim']
    ylim = sns_props['ylim']
    axsize = sns_props['axsize']
    size = sns_props['markersize']
    # Initialize figure object
    fig = go.Figure()
    if len(filter_columns) > 0 or filter_pattern is not None:
        # Apply filtering from user selections
        bright, dim = utils.filter_data(data, filter_columns, filter_pattern, union=union)
        utils.add_scatter(fig, data=bright, swarm=True, markersize=size)
        utils.add_scatter(fig, data=dim, swarm=True, markersize=size, opacity=0.15)
    else:
        utils.add_scatter(fig, data=data, swarm=True, markersize=size)
    # Assign figure properties
    fig.update_layout(
        width=axsize[0],
        height=axsize[1],
        **PLOT_KWARGS
    )
    fig.update_xaxes(range=xlim, side='top', **AXES_KWARGS)
    fig.update_yaxes(range=ylim, tickvals=[0], ticktext=[''], **AXES_KWARGS)
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
        label="Plot feature:",
        id="swarm-dropdown-feature",
        features=FEATURES,
        value='unique_first_words',
    ),
    utils.make_genre_dropdown_div(
        id="swarm-dropdown-genre",
        genres={g: utils.get_genre_label(df, g) for g in genres},
    ),
    utils.make_radio_div("swarm-radio-genre"),
    utils.make_band_input_div(id="swarm-input-band"),
])

wordcloud_card = utils.make_wordcloud_card("swarm")

layout = html.Div([
    utils.get_header(NUM_BANDS),
    dbc.Card([
        dbc.CardBody(
            dcc.Loading(
                id="swarm-plot-loading",
                type='default',
                children=dcc.Graph(id="swarm-graph"),
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


def match_bands(pattern):
    band_names = df.name
    if pattern is not None:
        return band_names[band_names.str.match(pattern)].values
    else:
        return band_names


@app.callback(
    Output("swarm-graph", "figure"),
    [Input("swarm-dropdown-feature", "value"),
     Input("swarm-dropdown-genre", "value"),
     Input("swarm-radio-genre", "value"),
     Input("swarm-input-band", "value"),])
def display_plot(feature, cols, selection, search):
    df_sort = df.sort_values(feature)
    swarm, swarm_props = get_swarm_data(
        df_sort[feature],
        figsize=FIGURE_SIZE,
        markersize=MARKER_SIZE,
    )
    swarm_df = pd.concat((df_sort, swarm), axis=1)
    if cols is None:
        cols = []
    fig = plot_scatter(swarm_df, cols, search, swarm_props, union=(selection == 'union'))
    return fig

@app.callback(
    [Output("swarm-word-cloud-image", "src"),
     Output("swarm-word-cloud-header", "children")],
    [Input("swarm-graph", "clickData")])
def make_image(click):
    return utils.make_image(tfidf, click)
