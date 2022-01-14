"""
Predict genre labels
"""


import os
import pickle
import h5py
import numpy as np
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dash_table import DataTable, FormatTemplate
from dash.dependencies import Input, Output, State
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import load_model
from ml_utils import create_keras_model


# app metadata
TITLE = "Genre predictor"

# pipeline location
MODELS_PATH = os.path.join(os.path.dirname(__file__), 'models')
PIPELINES = {
    'keras': 'Neural Network',
    'lgbm': 'Gradient Boosted Decision Trees',
    'logreg': 'Logistic Regression',
    # 'bernoulli': 'Bernoulli NB',
}


def get_pipeline(name):
    with open(os.path.join(MODELS_PATH, name + '_pipeline.pkl'), 'rb') as f:
        out = pickle.load(f)
    if name == 'keras':
        # extra code for rebuilding a KerasClassifier wrapper object
        # this is necessary because KerasClassifiers are not well suited for I/O
        clf_filename = os.path.join(MODELS_PATH, name + '_model.h5')
        clf = KerasClassifier(create_keras_model)
        clf.model = load_model(clf_filename)
        with h5py.File(clf_filename, 'r') as clf_h5:
            clf.classes_ = clf_h5.attrs['classes']
        out.classifier = clf
    return out


threshold_dict = {key: None for key in PIPELINES.keys()}
with open(os.path.join(MODELS_PATH, "thresholds.csv"), 'r') as f:
    for line in f.readlines():
        row = line.split(",")
        if len(row) > 1:
            threshold_dict[row[0]] = np.array([float(x) for x in row[1:]])

app = dash.Dash(
    __name__,
    title=TITLE,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP, 'assets/stylesheet.css']
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
    ]),
    dbc.Row([
        dbc.Col(
            html.Div([
                dcc.Textarea(
                    id='textarea',
                    value='Enter lyrics here',
                    style={'background-color': '#fff', 'width': '100%', 'height': 800},
                ),
            ])
        ),
        dbc.Col(
            html.Div([
                html.Div("Choose a model:"),
                dbc.RadioItems(
                    id="model-radio",
                    labelClassName="model-group-labels",
                    labelCheckedClassName="model-group-labels-checked",
                    className="model-group-items",
                    options=[{"label": label, "value": name} for name, label in PIPELINES.items()],
                    value=list(PIPELINES.keys())[0],
                    style={'margin-top': 10},
                ),
                dbc.Button("Predict", id='textarea-button', style={'margin-top': 30}),
                html.Div(
                    dcc.Loading(html.Div(id='table-div', style={'width': 300, 'height': 200})),
                    id='table-loading-div',
                    style={'margin-top': 10, 'width': 300, 'height': 200}
                ),
            ], style={'width': 500})
        )
    ], style={'margin-top': 10}),
], style={'margin': 10})


@app.callback(
    # Output('textarea-output', 'children'),
    Output('table-div', 'children'),
    Input('textarea-button', 'n_clicks'),
    [State('textarea', 'value'),
     State('model-radio', 'value')],
)
def update_output(n_clicks, text, model_name):
    pipeline = get_pipeline(model_name)
    threshold = threshold_dict[model_name]
    if threshold is not None:
        pipeline.threshold = threshold
    print(f"{model_name} thresholds:", pipeline.threshold)
    if n_clicks is None:
        return
    results = pipeline.classify_text(text)
    output = "Classification:\n"
    if results[0][2] < 1:
        output += "NONE\n"
    else:
        output += ", ".join([res[0].upper() for res in results if res[2] > 0]) + "\n"
    output += "\nIndividual label probabilities:\n"
    rows = []
    for label, prob, pred in results:
        genre = label[0].upper() + label[1:] + " Metal"
        if pipeline.threshold is not None:
            idx = list(pipeline.labels).index(label)
            prob_rescaled = min(1.0, prob * 0.5 / pipeline.threshold[idx])
            output += "{:<10s}{:>10.3g}%{:>10.3g}%\n".format(label, 100 * prob, 100 * prob_rescaled)
            rows.append({'genre': genre, 'p': prob_rescaled})
        else:
            output += "{:<10s}{:>10.3g}%\n".format(label, 100 * prob)
            rows.append({'genre': genre, 'p': prob})
    percentage = FormatTemplate.percentage(2)
    columns = [
        dict(id='genre', name='Genre'),
        dict(id='p', name='Probability', type='numeric', format=percentage),
    ]
    return DataTable(
        id='table',
        data=rows,
        columns=columns,
        cell_selectable=False,
        style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
        style_cell={'backgroundColor': 'rgb(50, 50, 50)', 'color': '#ccc'},
        style_data_conditional=[
            {
                'if': {'filter_query': '{p} >= 0.5'},
                'backgroundColor': 'rgb(100, 100, 100)',
                'color': 'white'
            },
        ],
    )


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-d', '--debug', default=False, action='store_true', help='run in debug mode')
    args = parser.parse_args()
    app.run_server(debug=args.debug)
