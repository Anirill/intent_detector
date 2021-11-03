from flask import Flask, render_template, request

# import joblib
# import json
import torch
import numpy as np

from .model import Intent
from .utils import word_dict, label_num, reversed_labels, lines


device = torch.device("cpu")

MODEL_STATE_DICT = 'model_intent_best_self+norm+drop256x256.pt'

intent_model = Intent()
intent_model.load_state_dict(torch.load(MODEL_STATE_DICT, map_location=torch.device('cpu')))
intent_model.eval()
# intent_model.to(device)

app = Flask(__name__)


@app.route("/", methods=["POST", "GET"])
@app.route("/static")
def index_page(text="Привет", prediction_message=""):
    app.static_folder = 'static'
    if request.method == "POST":
        # text = "new text"
        if request.form['submit_button'] == "Predict intent":
            text = request.form["text"]
        elif request.form['submit_button'] == "Random sentence":
            text = np.random.choice(lines)
        if text == '':
            text = '-'
        predictions, scores = intent_model.detection(text)
        prediction_message = []
        for i in range(len(predictions)):
            temp = []
            temp.append(f'Intent #{i+1}: \"{reversed_labels[predictions[i]]}\" -- ')
            temp.append(f'Score: {scores[i]:.3f} ')
            prediction_message.append(''.join(temp))

    return render_template('index_page.html', text=text, prediction_message=prediction_message)
