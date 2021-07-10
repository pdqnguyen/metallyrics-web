import dash
from apps.constants import TITLE

app = dash.Dash(__name__, title=TITLE, suppress_callback_exceptions=True)
server = app.server
