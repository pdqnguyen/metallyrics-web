from io import BytesIO
import base64
import pandas as pd
import dash_bootstrap_components as dbc
from dash import dcc, html
import plotly.graph_objects as go
from wordcloud import WordCloud


def get_genre_label(data, col):
    """Get column names that start with 'genre_'.
    """
    genre = col.replace('genre_', '')
    genre = genre[0].upper() + genre[1:]
    label = f"{genre} ({data[col].sum()} bands)"
    return label


def get_header(num_bands):
    return html.H1(f"Lyrical vocabulary of heavy metal artists")


def get_caption(num_bands, num_words):
    return html.P([
        f"These interactive plots show various lexical properties of the {num_bands} "
        f"most-reviewed heavy metal artists who have at least {num_words:,.0f} words "
        "in their full collection of lyrics. Review statistics are based on reviews at ",
        html.A("metal-archives", href='https://www.metal-archives.com/', target='_blank'),
        ", and lyrics are sourced from ",
        html.A("darklyrics", href='http://www.darklyrics.com/', target='_blank'),
        ". Click on any band in the graph to generate a word cloud of that band's ",
        "lyrics on the right."
    ])


def make_dropdown_div(label, dropdown):
    div = html.Div(
        [
            html.P(
                label,
                style={'margin-right': '2em'}
            ),
            html.Div([dropdown], className='dropdown', style={'width': '60%'})
        ],
        style={'margin-bottom': '1em', 'width': 700, 'display': 'flex'}
    )
    return div


def make_features_dropdown_div(label, id, features, value=None):
    dropdown = dcc.Dropdown(
        id=id,
        options=[
            {'label': v, 'value': k}
            for k, v in features.items()
        ],
        clearable=False,
        value=(value if value else list(features.keys())[0]),
        style={'background-color': '#111111', 'verticalAlign': 'middle'},
    )
    div = make_dropdown_div(label, dropdown)
    return div


def make_genre_dropdown_div(id, genres, value=None):
    dropdown = dcc.Dropdown(
        id=id,
        options=[
            {'label': v, 'value': k}
            for k, v in genres.items()
        ],
        multi=True,
        style={'background-color': '#111111', 'verticalAlign': 'middle'},
    )
    div = make_dropdown_div("Filter by genre:", dropdown)
    return div


def make_radio_div(id):
    radio = dcc.RadioItems(
        id=id,
        options=[
            {'label': ' Match ANY selected genre', 'value': 'union'},
            {'label': ' Match ALL selected genres', 'value': 'inter'},
        ],
        value='union',
        labelStyle={'display': 'inline-block', 'margin-right': '2em'}
    )
    div = html.Div(
        [
            html.P(
                "Filter mode:",
                style={'margin-right': '2em', 'width': '20%', 'display': 'inline-block'}
            ),
            html.Div(
                [radio],
                className='radio',
                style={'display': 'inline-block'}
            )
        ],
    )
    return div


def make_band_input_div(id):
    dropdown = dcc.Input(
        id=id,
        style={'background-color': '#111111', 'verticalAlign': 'middle', 'color': 'white'},
    )
    div = make_dropdown_div("Search by band name:", dropdown)
    return div


def filter_data(data, filter_columns, filter_pattern, union=True):
    filt = pd.Series(True, index=data.index)
    if len(filter_columns) > 0:
        if union:
            filt = (data[filter_columns] > 0).any(axis=1)
        else:
            filt = (data[filter_columns] > 0).all(axis=1)
    names = data.name
    if filter_pattern is not None:
        filt = filt & names.str.lower().str.contains(filter_pattern.lower())
    bright = data[filt]
    dim = data[~filt]
    return bright, dim


def add_scatter(fig, data, x='x', y='y', swarm=False, markersize=10, opacity=1.0):
    """Add a `plotly.graph_object.Scatter` trace to the figure.
    """
    # Custom genre string, wrapped at 25 characters per line
    customdata = pd.DataFrame({
        'name': data['name'],
        'genre_split': data['genre'].str.wrap(25).apply(lambda x: x.replace('\n', '<br>'))
    })
    if data[x].max() - data[x].min() > 10:
        # Whole number values if values span more than a factor of ten
        x_template = '%{x:.0f}'
    else:
        # Three significant figures otherwise
        x_template = '%{x:.3g}'
    if swarm:
        hovertemplate = '<b>%{customdata[0]}</b><br><br>'\
                        'Value: ' + x_template + '<br>'\
                        'Genre: %{customdata[1]}'
    else:
        if data[y].max() - data[y].min() > 10:
            # Whole number values if values span more than a factor of ten
            y_template = '%{y:.0f}'
        else:
            # Three significant figures otherwise
            y_template = '%{y:.3g}'
        hovertemplate = '<b>%{customdata[0]}</b><br><br>' \
                        'X: ' + x_template + '<br>' \
                        'Y: ' + y_template + '<br>' \
                        'Genre: %{customdata[1]}'
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=data[x],
            y=data[y],
            customdata=customdata,
            opacity=opacity,
            marker=dict(
                size=markersize,
                color='#1f77b4',
                line=dict(width=2, color='DarkSlateGray')
            ),
            hovertemplate=hovertemplate,
            name='',
        )
    )
    return


def make_controls_card(contents):
    card = dbc.Card(
        [dbc.CardBody(contents)],
        style={'height': '100%', 'background-color': '#333333'}
    )
    return card


def make_wordcloud_card(plot_type):
    card = dbc.Card([
        dbc.CardHeader("Word cloud:", id=f"{plot_type}-word-cloud-header"),
        dbc.CardBody([
            dcc.Loading(
                id=f"{plot_type}-word-cloud-loading",
                type='default',
                children=html.Img(
                    id=f"{plot_type}-word-cloud-image",
                    style={
                        'display': 'block',
                        'margin-left': 'auto',
                        'margin-right': 'auto',
                    }
                ),
            ),
        ]),
    ], style={'width': '100%', 'height': '100%', 'background-color': '#333333'})
    return card


def make_wordcloud(data, name):
    words = data.loc[name].to_dict()
    wc = WordCloud(background_color='black', width=480, height=300)
    wc.fit_words(words)
    return wc.to_image()


def make_image(data, click):
    if click is not None:
        name = click['points'][0]['customdata'][0]
        img = BytesIO()
        make_wordcloud(data, name).save(img, format='PNG')
        out = 'data:image/png;base64,{}'.format(
            base64.b64encode(img.getvalue()).decode())
        return out, "Word cloud: " + name
    else:
        return None, "Word cloud:"
