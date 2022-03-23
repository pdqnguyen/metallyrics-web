import pathlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc


DATA_PATH = pathlib.Path(__file__).parent.joinpath('./data').resolve()
REGIONS = {
    "world": {"label": "World", "path": "countries_"},
    "usa": {"label": "United States", "path": "states_"},
}
FEATURES = {
    "band_num": {
        "label": "Peak number of active bands",
        "path": "band_num.csv",
        "method": "max",
        "format": ",.0f",
        "log": True,
    },
    "band_dens": {
        "label": "Peak number of active bands per million people",
        "path": "band_dens.csv",
        "method": "max",
        "format": ",.2f",
        "log": True,
    },
    "album_num": {
        "label": "Total number of albums",
        "path": "album_num.csv",
        "method": "sum",
        "format": ",.0f",
        "log": True,
    },
    "album_dens": {
        "label": "Total number of albums per million people",
        "path": "album_dens.csv",
        "method": "sum",
        "format": ",.2f",
        "log": True,
    },
    "review_num": {
        "label": "Total number of reviews",
        "path": "review_num.csv",
        "method": "sum",
        "format": ",.0f",
        "log": True,
    },
    "review_dens": {
        "label": "Total number of reviews per million people",
        "path": "review_dens.csv",
        "method": "sum",
        "format": ",.2f",
        "log": True,
    },
    "review_avg": {
        "label": "Average review score",
        "path": "review_avg.csv",
        "method": "mean",
        "format": ",.2f",
        "log": False,
    },
    "review_weighted": {
        "label": "Weighted-average review score",
        "path": "review_weighted.csv",
        "method": "mean",
        "format": ",.2f",
        "log": False,
    },
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


def create_figure(data, feature_dict, show_states=False):
    colorbar_title = feature_dict['label']
    method = feature_dict['method']
    str_format = feature_dict['format']
    log_scale = feature_dict['log']
    if show_states:
        hover_labels = [STATE_NAMES[c] for c in data.columns]
    else:
        hover_labels = list(data.columns)
    values = eval(f"data.{method}(axis=0).values")
    if log_scale:
        z = np.log10(values)
        colorbar = dict(
            len=0.75,
            title=colorbar_title,
            tickvals=[0, 1, 2, 3, 4],
            ticktext=['1', '10', '100', '1000', '10000']
        )
    else:
        z = values
        colorbar = dict(len=0.75, title=colorbar_title)
    str_template = "<b>{label}:</b><br>{value:" + str_format + "}"
    text = [str_template.format(label=label, value=value) for label, value in zip(hover_labels, values)]
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
            options=[{'value': k, 'label': v['label']} for k, v in REGIONS.items()],
            value=list(REGIONS.keys())[0],
            inline=True,
            style={'display': 'inline-block', 'margin-right': 50},
        ),
        html.P("Value:", style={'display': 'inline-block', 'margin-right': 20}),
        html.Div([
            dcc.Dropdown(
                id='dropdown',
                options=[{'value': k, 'label': v['label']} for k, v in FEATURES.items()],
                value=list(FEATURES.keys())[0],
                clearable=False,
                style={'width': 500, 'background-color': '#111111', 'verticalAlign': 'middle'},
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
    region_dict = REGIONS[region]
    feature_dict = FEATURES[feature]
    basename = region_dict['path'] + feature_dict['path']
    df_path = DATA_PATH.joinpath(basename)
    df = pd.read_csv(df_path, index_col=0)
    start, end = date_range
    df = df.loc[start:end]
    if region == 'usa':
        df = df.loc[:, df.columns.isin(STATE_NAMES.keys())]
    fig = create_figure(df, feature_dict, show_states=(region == 'usa'))
    fig.update_layout(
        height=800,
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig.update_geos(
        scope=region,
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
