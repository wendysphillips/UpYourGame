import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc

import itertools
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils

SS = "https://stackpath.bootstrapcdn.com/bootswatch/4.5.0/sketchy/bootstrap.min.css"
app = dash.Dash(__name__, external_stylesheets=[SS])
app.scripts.config.serve_locally = False

#import log

config = tf.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)

config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6

session = tf.Session(config=config)

keras.backend.set_session(session)

from keras.models import model_from_json
with open("/Users/wendy/WendysPython/DashGameApp/static/keras_model101.json", "r") as json_file:
    json_string = json_file.read()
model = model_from_json(json_string)

global graph
graph = tf.get_default_graph()

gamedetails = pd.read_csv("/Users/wendy/WendysPython/DashGameApp/static/gameinfoForKeras.csv")
#encoder = LabelEncoder()
#text_labels = encoder.classes_ 
text_labels = np.load("/Users/wendy/WendysPython/DashGameApp/static/text_labels.npy", allow_pickle = True)


app.layout = html.Div([ 
	dcc.Textarea(
    placeholder='Enter a value...',
    value='UpYourGame!!',
    style={'width': '100%', 'height':200, 'fontSize':100, 'color':'blue'}),  
	dcc.Textarea(
        id='textarea-state-example',
        value='Tell me what you want in a game',
        style={'width': '100%', 'height':100},
    ),
    html.Button('Submit', id='textarea-state-example-button', n_clicks=0),
    html.Div(id='textarea-state-example-output', style={'whiteSpace': 'pre-line'})
])



@app.callback(
    Output('textarea-state-example-output', 'children'),
    [Input('textarea-state-example-button', 'n_clicks')],
    [State('textarea-state-example', 'value')]
)
def update_output(n_clicks, value):
	if n_clicks > 0:
		tokenize = text.Tokenizer(num_words=300, char_level=False)
		sentence = [value]
		tokenize.fit_on_texts(sentence)
		input = tokenize.texts_to_matrix(sentence)
		input2 = np.asarray(input)
		with graph.as_default():
			preds = model.predict([input2])
		prediction_label = text_labels[np.argsort(preds)]
		Germans = prediction_label[0][97:100]
		ans = gamedetails['name'][gamedetails['id'].isin(Germans)]
		j = pd.Series.tolist(ans)
		recom3 = ("")
		recom=pd.DataFrame(index=j,columns=['B'])
		for game in j:
			z=game
			y = gamedetails[(gamedetails['name']==game)]['official_url'].values[0]
			recom3 = (recom3 + z + "\n" + y + "\n\n")
		#for game in j:
			#recom.loc[game, 'B']= gamedetails[(gamedetails['name']==game)]['official_url'].values[0]
		
		valu = recom3
		
		return '\nI think you would like: \n\n{}'.format(valu)

if __name__ == '__main__':
        app.run_server(host='0.0.0.0', debug=True, port=50800)