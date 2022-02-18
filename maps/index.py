import pathlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc


PATH = pathlib.Path(__file__).parent
DATA_NAMES = {
    "band_num": "band_num.csv",
    "album_num": "album_num.csv",
    "review_num": "review_num.csv",
    "review_avg": "review_avg.csv",
    "review_weighted": "review_weighted.csv",
}
FEATURES = {
    "band_num": "Number of bands",
    "album_num": "Number of albums",
    "review_num": "Number of reviews",
    "review_avg": "Average review score",
    "review_weighted": "Weighted-average review score",
}

app = dash.Dash(
    __name__,
    title="Album reviews",
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

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
        html.P("Show value:", style={'display': 'inline-block', 'margin-right': 20}),
        html.Div([
            dcc.Dropdown(
                id='dropdown',
                options=[{'value': k, 'label': v} for k, v in FEATURES.items()],
                value=list(FEATURES.keys())[0],
                clearable=False,
                style={'width': 300, 'background-color': '#111111', 'verticalAlign': 'middle'},
            ),
        ], style={'display': 'inline-block', 'verticalAlign': 'middle', 'margin-right': 40}, className='dropdown'),
        html.P("Date range:", style={'display': 'inline-block', 'margin-right': 20}),
        html.Div([
            dcc.RangeSlider(
                id='slider',
                min=1970,
                max=2022,
                step=1,
                value=[1970, 2022],
                marks={i: str(i) for i in range(1970, 2022, 10)},
                tooltip={'always_visible': True}
            )
        ], style={'width': 400, 'display': 'inline-block', 'verticalAlign': 'bottom'}),
    ], style={'margin-top': 40}),
    dcc.Graph(id="choropleth"),
], style={'margin': 10})


@app.callback(Output("choropleth", "figure"), [Input("dropdown", "value"), Input("slider", "value")])
def display_choropleth(feature, date_range):
    df_path = PATH.joinpath('./data').resolve().joinpath("countries_" + DATA_NAMES[feature])
    df = pd.read_csv(df_path, index_col=0)
    kwargs = dict(
        locations=df.columns,
        locationmode="country names",
        colorscale='viridis',
    )
    start, end = date_range
    df = df.loc[start:end]
    if feature[-4:] == '_num':
        fig = go.Figure(
            go.Choropleth(
                z=np.log10(df.sum(axis=0)),
                customdata=df.sum(axis=0),
                colorbar=dict(
                    len=0.75,
                    title=FEATURES[feature],
                    tickvals=[0, 1, 2, 3, 4],
                    ticktext=['1', '10', '100', '1000', '10000']
                ),
                hovertemplate='%{location}: %{customdata:.0f}',
                **kwargs,
            )
        )
    else:
        fig = go.Figure(
            go.Choropleth(
                z=df.mean(axis=0),
                colorbar=dict(
                    len=0.75,
                    title=FEATURES[feature],
                ),
                hovertemplate='%{location}: %{z:.1f}',
                **kwargs,
            )
        )
    fig.update_layout(
        height=800,
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig.update_geos(
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
