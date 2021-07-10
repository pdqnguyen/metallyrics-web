"""
Produce scatter plots of lyrical complexity measures computed
in lyrical_complexity_pre.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

import utils


plt.switch_backend('Agg')


FEATURES = {
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


def uniq_first_words(x, num_words):
    """Of the first `num_words` in this text, how many are unique?
    """
    return len(set(x[:num_words]))

def get_band_stats(data):
    data['words_uniq'] = data['words'].apply(set)
    data['word_count'] = data['words'].apply(len)
    data['word_count_uniq'] = data['words_uniq'].apply(len)
    data['words_per_song'] = data['word_count'] / data['songs']
    data['words_per_song_uniq'] = data['word_count_uniq'] / data['songs']
    data['seconds_per_song'] = data['seconds'] / data['songs']
    data['word_rate'] = data['word_count'] / data['seconds']
    data['word_rate_uniq'] = data['word_count_uniq'] / data['seconds']
    data.loc[data['word_rate'] == np.inf, 'word_rate'] = 0
    data.loc[data['word_rate_uniq'] == np.inf, 'word_rate_uniq'] = 0
    return data


def get_band_words(data, num_bands=None, num_words=None):
    """Filter bands by word count and reviews, and count number of unique first words.
    """
    data_short = data[data['word_count'] > num_words].copy()
    top_reviewed_bands = data_short.sort_values('reviews')['name'][-num_bands:]
    data_short = data_short.loc[top_reviewed_bands.index]
    data_short['unique_first_words'] = data_short['words'].apply(uniq_first_words, args=(num_words,))
    data_short = data_short.sort_values('unique_first_words')
    return data_short


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


if __name__ == '__main__':
    cfg = utils.get_config(CONFIG, required=('input',))
    df = utils.load_bands(cfg['input'])
    df = get_band_stats(df)
    df = get_band_words(df, num_bands=cfg['num_bands'], num_words=cfg['num_words'])
    genres = [c for c in df.columns if 'genre_' in c]
    features = {k: v for k, v in FEATURES.items() if cfg['features'].get(k, False)}
    if cfg['features'].get('unique_first_words', False):
        key = 'unique_first_words'
        value = f"Number of unique words in first {cfg['num_words']:,.0f} words"
        features = dict([(key, value)] + list(features.items()))

    app = dash.Dash(__name__)

    dropdown_feature = dcc.Dropdown(
        id="dropdown_feature",
        options=[
            {'label': v, 'value': k}
            for k, v in features.items()
        ],
        value=list(features.keys())[0],
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

    app.layout = html.Div([
        html.Div(
            [
                html.H1(f"Lyrical Complexity of the Top {cfg['num_bands']} Artists"),
                html.P("This interactive swarm plot shows the most-reviewed artists who have at least "
                       f"{cfg['num_words']:,.0f} words in their collection of song lyrics."),
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
                dcc.Graph(id="graph"),
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
            figsize=(cfg['fig_width'], cfg['fig_height']),
            markersize=cfg['markersize'],
        )
        swarm_df = pd.concat((df_sort, swarm), axis=1)
        if cols is None:
            cols = []
        fig = plot_scatter(swarm_df, cols, swarm_props, union=(selection == 'union'))
        return fig

    app.run_server(debug=True)
