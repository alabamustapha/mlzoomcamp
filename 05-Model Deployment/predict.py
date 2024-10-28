# ## reading a model
import pickle

C=1.0
input_file = f"model_C={C}.bin"

sample_data = [{'onlinesecurity': 'yes',
  'deviceprotection': 'yes',
  'gender': 'male',
  'partner': 'yes',
  'paymentmethod': 'credit_card_(automatic)',
  'paperlessbilling': 'yes',
  'seniorcitizen': 0,
  'techsupport': 'no',
  'phoneservice': 'yes',
  'dependents': 'no',
  'onlinebackup': 'yes',
  'contract': 'month-to-month',
  'multiplelines': 'no',
  'streamingmovies': 'no',
  'internetservice': 'fiber_optic',
  'streamingtv': 'no',
  'tenure': 43,
  'monthlycharges': 86.1,
  'totalcharges': 3551.65}]



with open(input_file, "rb") as f_in:
    dv, model = pickle.load(f_in)


X = dv.transform(sample_data)

predictions = model.predict_proba(X)[0,1]

print(f"Churn probability: {predictions:.3f}")

