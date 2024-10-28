from flask import Flask, request, jsonify

import pickle


model_file = 'model1.bin'
dv_file = 'dv.bin'

app = Flask("score")

@app.route("/score", methods=["POST"])
def predict():
    input = request.get_json()

    with open(model_file, 'rb') as file:
        model = pickle.load(file)

    with open(dv_file, 'rb') as file:
        dv = pickle.load(file)

    X = dv.transform([input])
    y_pred = model.predict_proba(X)[0, 1]

    return jsonify({"subscription_probability": float(y_pred)})