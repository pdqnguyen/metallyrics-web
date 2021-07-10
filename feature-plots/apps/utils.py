import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go

def get_genre_label(data, col):
    """Get column names that start with 'genre_'.
    """
    genre = col.replace('genre_', '')
    genre = genre[0].upper() + genre[1:]
    label = f"{genre} ({data[col].sum()} bands)"
    return label


def get_header(num_bands):
    return html.H1(f"Lyrical complexity of the top {num_bands} heavy metal artists")


def get_caption(num_bands, num_words):
    return html.P([
        f"This interactive swarm plot shows various lexical properties of the {num_bands} "
        f"most-reviewed heavy metal artists who have at least {num_words:,.0f} words "
        "in their full collection of lyrics. Review statistics are based on reviews at ",
        html.A("metal-archives", href='https://www.metal-archives.com/', target='_blank'),
        ", and lyrics are sourced from ",
        html.A("darklyrics", href='http://www.darklyrics.com/', target='_blank'),
        "."
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
        style={'display': 'flex'}
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
            {'label': 'Match ANY selected genre', 'value': 'union'},
            {'label': 'Match ALL selected genres', 'value': 'inter'},
        ],
        value='union',
        labelStyle={'display': 'inline-block'}
    )
    div = html.Div(
        [
            html.P(
                "Filter mode:",
                style={'margin-right': '2em', 'width': '20%'}
            ),
            html.Div(
                [radio],
                className='radio',
                style={'width': '80%'}
            )
        ],
        style={'width': '500px', 'display': 'flex'}
    )
    return div


def filter_data(data, filter_columns, union=True):
    if union:
        filt = (data[filter_columns] > 0).any(axis=1)
    else:
        filt = (data[filter_columns] > 0).all(axis=1)
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
