from dash import dcc, html
from dash.dependencies import Input, Output

from app import app
from app import server
from apps import swarm, scatter

app.layout = html.Div([
    dcc.Location(id='url', refresh=False, pathname='/apps/swarm'),
    html.Div([
        "Links:",
        html.Label('|', style={'margin-left': 10, 'margin-right': 10}),
        dcc.Link('Github', href='https://www.github.com/pdqnguyen/metallyrics-web'),
        html.Label('|', style={'margin-left': 10, 'margin-right': 10}),
        dcc.Link('Network graph', href='https://metal-lyrics-network-graph.herokuapp.com/'),
        html.Label('|', style={'margin-left': 10, 'margin-right': 10}),
        dcc.Link('Genre predictor', href='https://metal-lyrics-genre-classifier.herokuapp.com/'),
    ], style={'margin-bottom': 10}),
    html.Div([
        dcc.Link('Swarm plots', href='/apps/swarm'),
        html.Label('|', style={'margin-left': 10, 'margin-right': 10}),
        dcc.Link('Scatter plots', href='/apps/scatter'),
    ]),
    html.Div(id='page-content', children=[])
], style={'width': 1600, 'margin': 10})

@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/apps/swarm':
        return swarm.layout
    elif pathname == '/apps/scatter':
        return scatter.layout
    else:
        return "404 Page Error"


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-d', '--debug', default=False, action='store_true', help='run in debug mode')
    args = parser.parse_args()
    app.run_server(debug=args.debug)
