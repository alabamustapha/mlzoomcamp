import pickle
from flask import Flask, request, jsonify

C=1.0
input_file = f"model_C={C}.bin"


app = Flask("churn")

@app.route("/ping", methods=["GET"])
def pong():
    return "pong"

@app.route("/predict", methods=["POST"])
def predict():
   
    customer = request.get_json()
    
    with open(input_file, "rb") as f_in:
        dv, model = pickle.load(f_in)

    X = dv.transform(customer)

    prediction = model.predict_proba(X)[0,1]

  
    return jsonify({
        "churn_probability": float(prediction),
        "churn" : bool(prediction > 0.5)
    })

if __name__ == "__main__":
    print("Starting the API service...")
    app.run(host="0.0.0.0", port=9696, debug=True)
