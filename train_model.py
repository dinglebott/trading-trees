import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
import numpy as np
from custom_modules import dataparser
from datetime import datetime
import json
import os

# PHASE 4: TRAIN AND EVALUATE FINAL MODEL
yearNow = 2026
instrument = "EUR_USD"
granularity = "H4"
candlesAhead = 7 # model predicts net return of the next n candles
deadzone = 0.0025 # defines half the width of the "flat" class (distance from midThreshold in either direction)
midThreshold = 0 # defines midpoint from which to split "up" and "down" classes

# LOAD AND SPLIT DATAFRAMES
df = dataparser.parseData(f"json_data/{instrument}_{granularity}_{yearNow - 16}-01-01_{yearNow}-01-01.json")
dfTrain = dataparser.splitByDate(df, datetime(yearNow - 16, 1, 1), datetime(yearNow - 2, 1, 1))
dfVal = dataparser.splitByDate(df, datetime(yearNow - 2, 1, 1), datetime(yearNow - 1, 1, 1))
dfTest = dataparser.splitByDate(df, datetime(yearNow - 1, 1, 1), datetime(yearNow, 1, 1))

# DEFINE FEATURES (use results from Phase 2)
directory = "results"
filename = "features.json"
filepath = os.path.join(directory, filename)
# deserialise json data
with open(filepath, "r") as file:
    rawFeatures = json.load(file) # rawFeatures is a Python dict
# extract top 11 features into list
bestFeatures = list(rawFeatures.keys())[:11]
print("Best features:", bestFeatures)

# DEFINE HYPERPARAMETERS (use results from Phase 3)
filename = "hyperparameters.json"
filepath = os.path.join(directory, filename)
# deserialise json data
with open(filepath, "r") as file:
    bestParams = json.load(file) # bestParams is a Python dict
# cast floats to ints where necessary
bestParams["max_depth"] = int(bestParams["max_depth"])
bestParams["min_child_weight"] = int(bestParams["min_child_weight"])
print("Best hyperparameters:", bestParams)

# TARGET VARIABLE: next n candles net return => positive (1) or negative (0)
for dataset in (dfTrain, dfVal, dfTest):
    dataset["forward_return"] = (dataset["close"].shift(-candlesAhead) / dataset["close"]) - 1
    conditions = [
        dataset["forward_return"] < midThreshold - deadzone, # downward move
        dataset["forward_return"] > midThreshold + deadzone # upward move
    ]
    choices = [0, 2]
    dataset["target"] = np.select(conditions, choices, default=1) # if not up or down, return flat (1)
    dataset.dropna(inplace=True)

# DEFINE DATASETS
X_train = dfTrain[bestFeatures]
y_train = dfTrain["target"]
X_val = dfVal[bestFeatures]
y_val = dfVal["target"]
X_test = dfTest[bestFeatures]
y_test = dfTest["target"]

# BUILD MODEL
model = xgb.XGBClassifier(**bestParams, eval_metric="mlogloss",
                          n_estimators=1000, # high ceiling
                          early_stopping_rounds=50, # stop after metric plateaus for 50 rounds
                          random_state=42,
                          device="cpu", # avoid transferring data to gpu
                          tree_method="hist")

# TRAIN MODEL
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# TEST MODEL
y_pred = model.predict(X_test)
y_predTrain = model.predict(X_train) # for overfitting evaluation
# returns 1D array of shape (n_samples)
# values 0 | 1 | 2
y_prob = model.predict_proba(X_test)
# returns 2D array of shape (n_samples, n_classes)
# chance of 0 | 1 | 2 for each datapoint

# EVALUATE MODEL
accuracy = accuracy_score(y_test, y_pred)*100
f1Score = f1_score(y_test, y_pred, average="macro", zero_division=0)
trainF1Score = f1_score(y_train, y_predTrain, average="macro", zero_division=0) # compare with f1Score to check overfitting
rocAucScore = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
# precision: accuracy of positive predictions for each class (up/down)
# recall: correctly identified positives / total true positives
# accuracy: correct predictions / total predictions
# F1 score: harmonic mean of precision and recall (0-1)
# ROC-AUC score: chance that a random positive is ranked higher than a random negative (0-1)

# CONFUSION MATRIX
cmatrix = confusion_matrix(y_test, y_pred)
# returns 2x2 numpy array breaking down true/false positives/negatives
cmatrixDf = pd.DataFrame(cmatrix, index=["Real -", "Real ~", "Real +"], columns=["Pred -", "Pred ~", "Pred +"])
print(f"Accuracy: {accuracy:.3f}%")
print(f"F1 score (macro-averaged): {f1Score:.5f}")
print(f"F1 score (train set): {trainF1Score:.5f}")
print(f"ROC-AUC score: {rocAucScore:.5f}")
print(f"Confusion matrix:\n{cmatrixDf}")

directory = "models"
if not os.path.exists(directory):
    os.makedirs(directory)
filename = f"XGBoost_{instrument}_{granularity}_{yearNow}.json"
filepath = os.path.join(directory, filename)
model.save_model(filepath)
print("\nModel saved to: " + filename)