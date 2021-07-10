import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from app import server
from apps import swarm, scatter

app.layout = html.Div([
    html.Div([
        dcc.Link('Swarm plots|', href='/apps/swarm'),
        dcc.Link('Scatter plots|', href='/apps/scatter'),
    ], className='row'),
    dcc.Location(id='url', refresh=False, pathname=''),
    html.Div(id='page-content', children=[])
])

@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/apps/swarm':
        return swarm.layout:
    elif pathname == '/apps/scatter':
        return scatter.layout
    else:
        return "404 Page Error"