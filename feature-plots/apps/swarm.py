"""
Produce swarm plots
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from app import app
from . import utils
from .constants import *


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


def add_scatter(fig, data, size, opacity=1.0):
    """Add a plotly scatter plot to visualize the swarm data.
    """
    # Custom genre string, wrapped at 25 characters per line
    customdata = pd.DataFrame({
        'name': data['name'],
        'genre_split': data['genre'].str.wrap(25).apply(lambda x: x.replace('\n', '<br>'))
    })
    if data['x'].max() - data['x'].min() > 10:
        # Whole number values if values span more than a factor of ten
        hovertemplate = '<b>%{customdata[0]}</b><br><br>'\
                        'Value: %{x:.0f}<br>'\
                        'Genre: %{customdata[1]}'
    else:
        # Three significant figures otherwise
        hovertemplate = '<b>%{customdata[0]}</b><br><br>'\
                        'Value: %{x:.3g}<br>'\
                        'Genre: %{customdata[1]}'
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=data['x'],
            y=data['y'],
            customdata=customdata,
            opacity=opacity,
            marker=dict(
                size=size,
                color='#1f77b4',
                line=dict(width=2, color='DarkSlateGray')
            ),
            hovertemplate=hovertemplate,
            name='',
        )
    )
    return


def plot_scatter(data, filter_columns, sns_props, union=True):
    """Initialize `plotly.graph_objects.Figure` object which will contain swarm plots.
    """
    xlim = sns_props['xlim']
    ylim = sns_props['ylim']
    axsize = sns_props['axsize']
    size = sns_props['markersize']
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
        add_scatter(fig, bright, size)
        add_scatter(fig, dim, size, opacity=0.15)
    else:
        add_scatter(fig, data, size)
    # Assign figure properties
    fig.update_layout(
        autosize=False,
        width=axsize[0],
        height=axsize[1],
        showlegend=False,
        hoverlabel=dict(bgcolor='#730000', font_color='#EBEBEB', font_family='Monospace'),
        template='plotly_dark',
    )
    fig.update_xaxes(range=xlim, gridwidth=2, gridcolor='#444444', side='top')
    fig.update_yaxes(range=ylim, gridwidth=2, gridcolor='#444444', tickvals=[0], ticktext=[''])
    return fig


df = pd.read_csv(DATA_PATH)
genres = [c for c in df.columns if 'genre_' in c]

layout = html.Div([
    html.Div(
        [
            utils.get_header(NUM_BANDS),
            utils.get_caption(NUM_BANDS, NUM_WORDS),
            utils.make_features_dropdown_div(
                label="Plot feature",
                id="swarm-dropdown-feature",
                features=FEATURES,
                value=list(FEATURES.keys())[0]
            ),
            utils.make_genre_dropdown_div(
                id="swarm-dropdown-genre",
                genres={g: utils.get_genre_label(df, g) for g in genres},
            ),
            utils.make_radio_div("swarm-radio-genre"),
            dcc.Loading(
                id="swarm-plot-loading",
                type='default',
                children=dcc.Graph(id="swarm-graph"),
            ),
        ],
        style={
            'width': '800px',
            'font-family': 'Helvetica',
        }
    )
], style={})


@app.callback(
    Output("swarm-graph", "figure"),
    [Input("swarm-dropdown-feature", "value"),
     Input("swarm-dropdown-genre", "value"),
     Input("swarm-radio-genre", "value")])
def display_plot(feature, cols, selection):
    df_sort = df.sort_values(feature)
    swarm, swarm_props = get_swarm_data(
        df_sort[feature],
        figsize=FIGURE_SIZE,
        markersize=MARKER_SIZE,
    )
    swarm_df = pd.concat((df_sort, swarm), axis=1)
    if cols is None:
        cols = []
    fig = plot_scatter(swarm_df, cols, swarm_props, union=(selection == 'union'))
    return fig
