"""
Produce scatter plots of lyrical complexity measures computed
in lyrical_complexity_pre.py
"""

import yaml
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from ..app import app


plt.switch_backend('Agg')

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath('../data').resolve().joinpath('data.csv')

NUM_WORDS = 10000
NUM_BANDS = 200
FIGURE_SIZE = (20, 10)
MARKER_SIZE = 18

TITLE = "Vocabulary of heavy metal artists"
FEATURES = {
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
CONFIG = 'config.yaml'


def get_config(filename, required=('input', 'output')):
    with open(filename, 'r') as f:
        cfg = yaml.safe_load(f)
    for key in required:
        if key not in cfg.keys():
            raise KeyError(f"missing field {key} in {filename}")
    return cfg


def get_genre_label(data, col):
    """Get column names that start with 'genre_'.
    """
    genre = col.replace('genre_', '')
    genre = genre[0].upper() + genre[1:]
    label = f"{genre} ({data[col].sum()} bands)"
    return label


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

dropdown_feature = dcc.Dropdown(
    id="dropdown_feature",
    options=[
        {'label': v, 'value': k}
        for k, v in FEATURES.items()
    ],
    value=list(FEATURES.keys())[0],
    clearable=False,
    style={'background-color': '#111111', 'verticalAlign': 'middle'}
)

dropdown_genre = dcc.Dropdown(
    id="dropdown_genre",
    options=[
        {'label': get_genre_label(df, g), 'value': g}
        for g in genres
    ],
    clearable=False,
    multi=True,
    style={'background-color': '#111111', 'verticalAlign': 'middle'}
)

radio_genre = dcc.RadioItems(
    id="radio_genre",
    options=[
        {'label': 'Match ANY selected genre', 'value': 'union'},
        {'label': 'Match ALL selected genres', 'value': 'inter'},
    ],
    value='union',
    labelStyle={'display': 'inline-block'}
)

layout = html.Div([
    html.Div(
        [
            html.H1(f"Lyrical complexity of the top {NUM_BANDS} heavy metal artists"),
            html.P([f"This interactive swarm plot shows various lexical properties of the {NUM_BANDS} "
                    f"most-reviewed heavy metal artists who have at least {NUM_WORDS:,.0f} words "
                    "in their full collection of lyrics. Review counts are based on reviews at ",
                    html.A("metal-archives", href='https://www.metal-archives.com/', target='_blank'),
                    ", and lyrics are sourced from ",
                    html.A("darklyrics", href='http://www.darklyrics.com/', target='_blank'),
                    "."]),
            html.Div(
                [
                    html.P(
                        "Plot feature",
                        style={'margin-right': '2em'}
                    ),
                    html.Div([dropdown_feature], className='dropdown', style={'width': '60%'})
                ],
                style={'display': 'flex'}
            ),
            html.Div(
                [
                    html.P(
                        "Filter by genre:",
                        style={'margin-right': '2em'}
                    ),
                    html.Div([dropdown_genre], className='dropdown', style={'width': '30%'})
                ],
                style={'display': 'flex'}
            ),
            html.Div(
                [
                    html.P(
                        "Filter mode:",
                        style={'margin-right': '2em', 'width': '20%'}
                    ),
                    html.Div(
                        [radio_genre],
                        className='radio',
                        style={'width': '80%'}
                    )
                ],
                style={'width': '500px', 'display': 'flex'}
            ),
            dcc.Loading(
                id='perf-plot-loading',
                type='default',
                children=dcc.Graph(id="graph"),
            ),
        ],
        style={
            'width': '800px',
            'font-family': 'Helvetica',
        }
    )
], style={})


@app.callback(
    Output("graph", "figure"),
    [Input("dropdown_feature", "value"),
     Input("dropdown_genre", "value"),
     Input("radio_genre", "value")])
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