import pathlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc


DATA_PATH = pathlib.Path(__file__).parent.joinpath('./data').resolve()
DATA_PREFIX = {
    "World": "countries_",
    "United States": "states_",
}
DATA_NAMES = {
    "band_num": "band_num.csv",
    "album_num": "album_num.csv",
    "review_num": "review_num.csv",
    "review_avg": "review_avg.csv",
    "review_weighted": "review_weighted.csv",
}
REGIONS = {
    'World': 'world',
    'United States': 'usa',
}
FEATURES = {
    "band_num": "Peak number of active bands",
    "album_num": "Total number of albums",
    "review_num": "Total number of reviews",
    "review_avg": "Average review score",
    "review_weighted": "Weighted-average review score",
}
DATE_RANGE = (1970, 2022)
DATE_INTERVAL = 10


STATE_NAMES = {
    'AK': 'Alaska',
    'AL': 'Alabama',
    'AR': 'Arkansas',
    'AZ': 'Arizona',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Connecticut',
    'DE': 'Delaware',
    'FL': 'Florida',
    'GA': 'Georgia',
    'HI': 'Hawaii',
    'IA': 'Iowa',
    'ID': 'Idaho',
    'IL': 'Illinois',
    'IN': 'Indiana',
    'KS': 'Kansas',
    'KY': 'Kentucky',
    'LA': 'Louisiana',
    'MA': 'Massachusetts',
    'MD': 'Maryland',
    'ME': 'Maine',
    'MI': 'Michigan',
    'MN': 'Minnesota',
    'MO': 'Missouri',
    'MS': 'Mississippi',
    'MT': 'Montana',
    'NC': 'North Carolina',
    'ND': 'North Dakota',
    'NE': 'Nebraska',
    'NH': 'New Hampshire',
    'NJ': 'New Jersey',
    'NM': 'New Mexico',
    'NV': 'Nevada',
    'NY': 'New York',
    'OH': 'Ohio',
    'OK': 'Oklahoma',
    'OR': 'Oregon',
    'PA': 'Pennsylvania',
    'RI': 'Rhode Island',
    'SC': 'South Carolina',
    'SD': 'South Dakota',
    'TN': 'Tennesee',
    'TX': 'Texas',
    'UT': 'Utah',
    'VA': 'Virginia',
    'VT': 'Vermont',
    'WA': 'Washington',
    'WI': 'Wisconsin',
    'WV': 'West Virginia',
    'WY': 'Wyoming',
}


def create_figure(data, feature, show_states=False):
    if show_states:
        labels = [STATE_NAMES[c] for c in data.columns]
    else:
        labels = list(data.columns)
    if feature == 'band_num':
        values = data.max(axis=0).values
        z = np.log10(data.sum(axis=0))
        colorbar = dict(
            len=0.75,
            title=FEATURES[feature],
            tickvals=[0, 1, 2, 3, 4],
            ticktext=['1', '10', '100', '1000', '10000']
        )
    elif feature[-4:] == '_num':
        values = data.sum(axis=0).values
        z = np.log10(data.sum(axis=0))
        colorbar = dict(
            len=0.75,
            title=FEATURES[feature],
            tickvals=[0, 1, 2, 3, 4],
            ticktext=['1', '10', '100', '1000', '10000']
        )
    else:
        values = data.mean(axis=0).values
        z = data.mean(axis=0)
        colorbar = dict(len=0.75, title=FEATURES[feature])
    text = [f"<b>{label}:</b><br>{value:.0f}" for label, value in zip(labels, values)]
    if show_states:
        locationmode = 'USA-states'
    else:
        locationmode = 'country names'
    kwargs = dict(z=z, colorbar=colorbar, text=text, hoverinfo='text', locations=data.columns,
                  locationmode=locationmode, colorscale='viridis')
    fig = go.Figure(go.Choropleth(**kwargs))
    return fig


app = dash.Dash(
    __name__,
    title="Album reviews",
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

server = app.server

app.layout = html.Div([
    html.Div([
        "Links:",
        html.Label('|', style={'margin-left': 10, 'margin-right': 10}),
        html.A('Github', href='https://www.github.com/pdqnguyen/metallyrics-web'),
        html.Label('|', style={'margin-left': 10, 'margin-right': 10}),
        html.A('Dataset dashboard', href='https://metal-lyrics-feature-plots.herokuapp.com/'),
        html.Label('|', style={'margin-left': 10, 'margin-right': 10}),
        html.A('Network graph', href='https://metal-lyrics-network-graph.herokuapp.com/'),
        html.Label('|', style={'margin-left': 10, 'margin-right': 10}),
        html.A('Genre predictor', href='https://metal-lyrics-genre-classifier.herokuapp.com/'),
    ]),
    html.Div([
        html.P("Region:", style={'display': 'inline-block', 'margin-right': 20}),
        dbc.RadioItems(
            id='radio',
            options=[{'value': x, 'label': x} for x in DATA_PREFIX.keys()],
            value=list(DATA_PREFIX.keys())[0],
            inline=True,
            style={'display': 'inline-block', 'margin-right': 50},
        ),
        html.P("Value:", style={'display': 'inline-block', 'margin-right': 20}),
        html.Div([
            dcc.Dropdown(
                id='dropdown',
                options=[{'value': k, 'label': v} for k, v in FEATURES.items()],
                value=list(FEATURES.keys())[0],
                clearable=False,
                style={'width': 300, 'background-color': '#111111', 'verticalAlign': 'middle'},
            ),
        ], style={'display': 'inline-block', 'verticalAlign': 'middle', 'margin-right': 50}, className='dropdown'),
        html.P("Date range:", style={'display': 'inline-block', 'margin-right': 20}),
        html.Div([
            dcc.RangeSlider(
                id='slider',
                min=DATE_RANGE[0],
                max=DATE_RANGE[1],
                step=1,
                value=DATE_RANGE,
                marks={i: str(i) for i in range(DATE_RANGE[0], DATE_RANGE[1], DATE_INTERVAL)},
                tooltip={'always_visible': True}
            )
        ], style={'width': 400, 'display': 'inline-block', 'verticalAlign': 'bottom'}),
    ], style={'margin-top': 40}),
    html.Div(dcc.Graph(id="choropleth"), style={'horizontalAlign': 'left'}),
], style={'margin': 10})


@app.callback(
    Output("choropleth", "figure"),
    [Input("radio", "value"),
     Input("dropdown", "value"),
     Input("slider", "value")]
)
def display_choropleth(region, feature, date_range):
    basename = DATA_PREFIX[region] + DATA_NAMES[feature]
    df_path = DATA_PATH.joinpath(basename)
    df = pd.read_csv(df_path, index_col=0)
    start, end = date_range
    df = df.loc[start:end]
    scope = REGIONS[region]
    fig = create_figure(df, feature, show_states=(scope == 'usa'))
    fig.update_layout(
        height=800,
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig.update_geos(
        scope=scope,
        showland=True, landcolor="#111111",
        showocean=True, oceancolor="#111111",
        showlakes=True, lakecolor="#111111",
        showcountries=True, countrycolor="gray",
        showcoastlines=True, coastlinecolor="gray",
        framecolor="gray",
        bgcolor="#111111",
    )
    return fig


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-d', '--debug', default=False, action='store_true', help='run in debug mode')
    args = parser.parse_args()
    app.run_server(debug=args.debug)
