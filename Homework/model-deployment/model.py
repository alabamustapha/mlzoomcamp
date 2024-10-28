import pickle

with open('model1.bin', 'rb') as file:
    model = pickle.load(file)

with open('dv.bin', 'rb') as file:
    dv = pickle.load(file)

input = {"job": "management", "duration": 400, "poutcome": "success"}

X = dv.transform([input])
y_pred = model.predict_proba(X)[0, 1]

print(y_pred)