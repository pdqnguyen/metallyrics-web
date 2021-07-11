import dash
import dash_bootstrap_components as dbc
from apps.config import TITLE

app = dash.Dash(
    __name__,
    title=TITLE,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
server = app.server
