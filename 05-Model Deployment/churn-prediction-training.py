import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import pickle

# parameters

C=1.0
n_splits=5
output_file = f"model_C={C}.bin"


# prepare the data

print("Loading the data...\n")
churn = pd.read_csv("customer-churn.csv")
churn.columns = churn.columns.str.lower().str.replace(" ", "_")

categorical_columns = churn.dtypes[churn.dtypes == "O"].index

print("Cleaning the data...\n")
for c in categorical_columns:
    churn[c] = churn[c].str.lower().str.replace(" ", "_")

churn.totalcharges = pd.to_numeric(churn.totalcharges, errors="coerce")
churn["totalcharges"] = churn.totalcharges.fillna(0)

churn.churn = churn.churn.apply(lambda churn: churn == "yes" ).astype(int)

print("Splitting the data...\n")

df_full_train, df_test =  train_test_split(churn, test_size=0.2, random_state=1)

df_train, df_val =  train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index()
df_test = df_test.reset_index()
df_val = df_val.reset_index()

y_train = df_train.churn
y_val = df_val.churn
y_test = df_test.churn


del df_train["churn"]
del df_val["churn"]
del df_test["churn"]


numerical = ["tenure", "monthlycharges", "totalcharges"]
categorical = [
    'onlinesecurity', 'deviceprotection', 'gender', 
    'partner', 'paymentmethod', 'paperlessbilling', 
    'seniorcitizen', 'techsupport', 'phoneservice', 
    'dependents', 'onlinebackup', 'contract', 
    'multiplelines', 'streamingmovies', 
    'internetservice', 'streamingtv'
]

# training the model

def train_model(df_train, y_train, c=0.001):
    print(f"Training the model with C={c}\n")
    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dicts)

    model = LogisticRegression(max_iter=1000, C=c, solver="liblinear")
    model.fit(X_train, y_train)
    
    return dv, model

# predict

def predict(df, dv, model):
    print("Predicting...\n")
    df_dicts = df[categorical + numerical].to_dict(orient="records")

    X = dv.fit_transform(df_dicts)

    y_pred = model.predict_proba(X)[:,1]

    return y_pred

# cross-validation

print("Cross-validation...\n")
scores = []

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

for fold, (train_idx, val_idx) in enumerate(kfold.split(df_full_train)):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]
    
    dv, model = train_model(df_train, df_train.churn, c=C)
    
    y_pred = predict(df_val, dv, model)

    score = roc_auc_score(df_val.churn, y_pred)
    
    print("*" * 20)
    print(f"AUC for fold {fold}: {score}")
    print("*" * 20)   

    scores.append(score)


print("\nCross-validation results:")
print(f"C={C}: {np.mean(scores):.2f} +/- {np.std(scores):.2f}")


# training the final model
print("\nTraining the final model...")
dv, model = train_model(df_full_train, df_full_train.churn, c=100)
y_pred = predict(df_test, dv, model)
score = roc_auc_score(y_test, y_pred)


# saving the model

print("\nSaving the model...")
with open(output_file, "wb") as f_out:
    pickle.dump((dv,model), f_out)