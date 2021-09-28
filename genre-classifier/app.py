"""
Predict genre labels
"""


import glob
import os
import pickle
import h5py
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash_table import DataTable, Format
from dash.dependencies import Input, Output, State
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import load_model
from ml_utils import create_keras_model


# app metadata
TITLE = "Genre predictor"

# defaults
KERAS_MODEL = False

# pipeline location
MODELS_PATH = os.path.join(os.path.dirname(__file__), 'models')

PIPELINES = {
    'keras': 'Neural Network',
    'logreg': 'Logistic Regression',
}

app = dash.Dash(
    __name__,
    title=TITLE,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP, 'assets/stylesheet.css']
)
server = app.server


def get_pipeline(name):
    with open(os.path.join(MODELS_PATH, name + '_pipeline.pkl'), 'rb') as f:
        out = pickle.load(f)
    if name == 'keras':
        # extra code for rebuilding a KerasClassifier wrapper object
        # this is necessary because KerasClassifiers are not well suited for I/O
        # but scikit-multilearn handles KerasClassifiers better than native Keras models
        # look for and load all Keras models found in the same directory as the pipeline
        classifiers = []
        print(os.path.join(MODELS_PATH, 'keras*.h5'))
        clf_fnames = glob.glob(str(os.path.join(MODELS_PATH, 'keras*.h5')))
        for i, clf_fname in enumerate(clf_fnames):
            clf = KerasClassifier(create_keras_model)
            clf.model = load_model(clf_fname)
            # get the 'classes_' attribute, necessary for the wrapper to make predictions
            with h5py.File(clf_fname, 'r') as clf_h5:
                clf.classes_ = clf_h5.attrs['classes']
            classifiers.append(clf)
        # make sure number of Keras models found matches the number needed in the pipeline
        if len(classifiers) == len(out.classifier.classifiers_):
            out.classifier.classifiers_ = classifiers
        else:
            raise OSError(f"{len(classifiers)} Keras models found,"
                          f" {len(out.classifier.classifiers_)} needed")
    # import numpy as np
    # pipeline.set_threshold(0.5 * np.ones(len(pipeline.threshold)))
    print("Thresholds:", out.threshold)
    return out


app.layout = html.Div(
    dbc.Row([
        dbc.Col(
            html.Div([
                dcc.Textarea(
                    id='textarea',
                    value='Enter lyrics here',
                    style={'width': '100%', 'height': 800},
                ),
            ])
        ),
        dbc.Col(
            html.Div([
                dbc.Button("Predict", id='textarea-button', style={'margin-right': 30, 'display': 'inline-block'}),
                html.Div(
                    [
                        html.Div("Model selection:", style={'margin-right': 10, 'display': 'inline-block'}),
                        html.Div(
                            dbc.RadioItems(
                                id="model-radio",
                                labelClassName="model-group-labels",
                                labelCheckedClassName="model-group-labels-checked",
                                className="model-group-items",
                                options=[{"label": label, "value": name} for name, label in PIPELINES.items()],
                                value=list(PIPELINES.keys())[0],
                                inline=True,
                            ),
                            style={'display': 'inline-block'},
                        )
                    ],
                    style={'display': 'inline-block'},
                ),
                html.Div(id='table-div', style={'width': 500}),
                # dt.DataTable(id='table', columns=[{'name': i, 'id': i} for i in COLUMNS]),
                # html.Div(id='textarea-output', style={'margin-top': 20, 'whiteSpace': 'pre-line'}),
            ])
        )
    ]),
    style={'margin': 20}
)


@app.callback(
    # Output('textarea-output', 'children'),
    Output('table-div', 'children'),
    Input('textarea-button', 'n_clicks'),
    [State('textarea', 'value'),
     State('model-radio', 'value')],
)
def update_output(n_clicks, text, model_name):
    pipeline = get_pipeline(model_name)
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
    for res in results:
        idx = list(pipeline.labels).index(res[0])
        prob = min(1.0, res[1] * 0.5 / pipeline.threshold[idx])
        output += "{:<10s}{:>10.3g}%{:>10.3g}%\n".format(res[0], 100 * res[1], 100 * prob)
        rows.append({'genre': res[0], 'p': res[1], 'p_r': prob})
    percentage = {'specifier': '.3g'}
    columns = [
        dict(id='genre', name='Genre'),
        dict(id='p', name='Probability', type='numeric', format=percentage),
        dict(id='p_r', name='Reweighted probability', type='numeric', format=percentage),
    ]
    return DataTable(
        id='table',
        data=rows,
        columns=columns,
        style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
        style_cell={'backgroundColor': 'rgb(50, 50, 50)', 'color': '#ccc'},
        style_data_conditional=[
            {
                'if': {
                    'filter_query': '{p_r} >= 0.5',
                },
                'backgroundColor': 'rgb(100, 100, 100)',
                'color': 'white'
            },
        ]
    )


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-d', '--debug', default=False, action='store_true', help='run in debug mode')
    args = parser.parse_args()
    app.run_server(debug=args.debug)
