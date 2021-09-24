"""
Predict genre labels
"""


import glob
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pathlib
import pickle
import h5py
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import load_model
from ml_utils import create_keras_model

# App metadata
TITLE = "Genre predictor"


# Dataset location
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath('./data').resolve()

app = dash.Dash(
    __name__,
    title=TITLE,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
server = app.server

with open(DATA_PATH.joinpath('pipeline.pkl'), 'rb') as f:
    pipeline = pickle.load(f)
# extra code for rebuilding a KerasClassifier wrapper object
# this is necessary because KerasClassifiers are not well suited for I/O
# but scikit-multilearn handles KerasClassifiers better than native Keras models
# look for and load all Keras models found in the same directory as the pipeline
classifiers = []
print(DATA_PATH.joinpath('keras*.h5'))
clf_fnames = glob.glob(str(DATA_PATH.joinpath('keras*.h5')))
for i, clf_fname in enumerate(clf_fnames):
    clf = KerasClassifier(create_keras_model)
    clf.model = load_model(clf_fname)
    # get the 'classes_' attribute, necessary for the wrapper to make predictions
    with h5py.File(clf_fname, 'r') as clf_h5:
        clf.classes_ = clf_h5.attrs['classes']
    classifiers.append(clf)
# make sure number of Keras models found matches the number needed in the pipeline
if len(classifiers) == len(pipeline.classifier.classifiers_):
    pipeline.classifier.classifiers_ = classifiers
else:
    raise OSError(f"{len(classifiers)} Keras models found,"
                  f" {len(pipeline.classifier.classifiers_)} needed")
print("Thresholds:", pipeline.threshold)

app.layout = html.Div(
    dbc.Row([
        dbc.Col(
            html.Div([
                dcc.Textarea(
                    id='textarea',
                    value='Enter lyrics here',
                    style={'width': '100%', 'height': 300},
                ),
            ])
        ),
        dbc.Col(
            html.Div([
                dbc.Button("Predict", id='textarea-button', style={'margin-bottom': 10}),
                html.Div(id='textarea-output', style={'whiteSpace': 'pre-line'}),
            ])
        )
    ]),
    style={'margin': 20}
)


@app.callback(
    Output('textarea-output', 'children'),
    Input('textarea-button', 'n_clicks'),
    State('textarea', 'value'),
)
def update_output(n_clicks, text):
    if n_clicks is None:
        return
    results = pipeline.classify_text(text)
    output = "Classification:\n"
    if results[0][2] < 1:
        output += "NONE\n"
    else:
        output += ", ".join([res[0].upper() for res in results if res[2] > 0]) + "\n"
    output += "\nIndividual label probabilities:\n"
    for res in results:
        output += "{:<10s}{:>5.3g}%\n".format(res[0], 100 * res[1])
    return output


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-d', '--debug', default=False, action='store_true', help='run in debug mode')
    args = parser.parse_args()
    app.run_server(debug=args.debug)
