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

TITLE = "Vocabulary of heavy metal artists"
FEATURES = {
    'reviews': 'Number of album reviews',
    'rating': 'Average album review score',
    'unique_first_words': f"Number of unique words in first {NUM_WORDS:,.0f} words",
    'word_count': 'Total number of words in discography',
    'words_per_song': 'Average words per song',
    'words_per_song_uniq': 'Average unique words per song',
    'seconds_per_song': 'Average song length in seconds',
    'word_rate': 'Average words per second',
    'word_rate_uniq': f"Average unique words per second",
    'types': 'Types (total unique words)',
    'TTR': 'Type-token ratio (TTR)',
    'logTTR': 'Log-corrected TTR',
    'MTLD': 'MTLD',
    'logMTLD': 'Log(MTLD)',
    'vocd-D': 'vocd-D',
    'logvocd-D': 'Log(vocd-D)',
}


def add_scatter(fig, data, x, y, opacity=1.0):
    if data[x].max() - data[x].min() > 10:
        x_template = '%{x:.0f}'
    else:
        x_template = '%{x:.3g}'
    if data[y].max() - data[y].min() > 10:
        y_template = '%{y:.0f}'
    else:
        y_template = '%{y:.3g}'
    hovertemplate = '<b>%{customdata[0]}</b><br><br>'\
                    'X: ' + x_template + '<br>'\
                    'Y: ' + y_template + '<br>'\
                    'Genre: %{customdata[1]}'
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=data[x],
            y=data[y],
            customdata=data[['name', 'genre']],
            opacity=opacity,
            marker=dict(
                size=10,
                color='#1f77b4',
                line=dict(width=2, color='DarkSlateGray')
            ),
            hovertemplate=hovertemplate,
            name='',
        )
    )


def plot_scatter(data, x, y, filter_columns, union=True):
    # Initialize figure object
    fig = go.Figure()
    if len(filter_columns) > 0:
        # Apply filtering from user selections
        if union:
            filt = (data[filter_columns] > 0).any(axis=1)
        else:
            filt = (data[filter_columns] > 0).all(axis=1)
        bright = data[filt]
        dim = data[~filt]
        add_scatter(fig, bright, x, y)
        add_scatter(fig, dim, x, y, opacity=0.15)
    else:
        add_scatter(fig, data, x, y)
    # Assign figure properties
    fig.update_layout(
        width=1200,
        height=800,
        showlegend=False,
        hoverlabel=dict(bgcolor='#730000', font_color='#EBEBEB', font_family='Monospace'),
        template='plotly_dark',
        xaxis_title=FEATURES[x],
        yaxis_title=FEATURES[y],
    )
    fig.update_xaxes(gridwidth=2, gridcolor='#444444')
    fig.update_yaxes(gridwidth=2, gridcolor='#444444')
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
        style={
            'width': '800px',
            'font-family': 'Helvetica',
        }
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
