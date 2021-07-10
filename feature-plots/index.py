import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from app import server
from apps import swarm, scatter

app.layout = html.Div([
    dcc.Location(id='url', refresh=False, pathname='/apps/swarm'),
    html.Div([
        dcc.Link('Swarm plots', href='/apps/swarm'),
        html.Label(' | '),
        dcc.Link('Scatter plots', href='/apps/scatter'),
    ], className='row'),
    html.Div(id='page-content', children=[], style={'width': 800})
])

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
