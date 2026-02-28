import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
from custom_modules import dataparser
from datetime import datetime
import os

# PHASE 4: TRAIN AND EVALUATE FINAL MODEL
yearNow = 2026
instrument = "EUR_USD"
granularity = "H1"

# LOAD AND SPLIT DATAFRAMES
df = dataparser.parseData(f"json_data/{instrument}_{granularity}_{yearNow - 16}-01-01_{yearNow}-01-01.json")
dfTrain = dataparser.splitByDate(df, datetime(yearNow - 16, 1, 1), datetime(yearNow - 1, 1, 1))
dfTest = dataparser.splitByDate(df, datetime(yearNow - 1, 1, 1), datetime(yearNow, 1, 1))

# DEFINE FEATURES (use results from Phase 2)
features = [
    "return", "hl_spread", "oc_spread", "body_ratio",
    "normalised_ema15", "normalised_ema50",
    "rsi_14", "macd_hist",
    "atr_14", "bb_width",
    "vol_ratio", "bb_position",
    "return_lag1", "return_lag2", "return_lag3", "return_lag4", "return_lag5",
    "vol_ratio_lag1", "vol_ratio_lag2", "vol_ratio_lag3", "vol_ratio_lag4", "vol_ratio_lag5"
]

# DEFINE HYPERPARAMETERS (use results from Phase 3)
params = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3
}

# TARGET VARIABLE: next candle return => positive (1) or negative (0)
for df in (dfTrain, dfTest):
    df["target"] = (df["return"].shift(-1) > 0).astype(int) # boolean to integer
    df.dropna(inplace=True)

# DEFINE DATASETS
X_train = dfTrain[features]
y_train = dfTrain["target"]
X_test = dfTest[features]
y_test = dfTest["target"]

# BUILD MODEL
model = xgb.XGBClassifier(**params, eval_metric="logloss", random_state=42)

# TRAIN MODEL
model.fit(X_train, y_train)

# TEST MODEL
y_pred = model.predict(X_test)
# returns 1D array of shape (n_samples)
# values 0 | 1
y_prob = model.predict_proba(X_test)[:, 1]
# returns 2D array of shape (n_samples, 2)
# chance of 0 and 1 for each datapoint

# EVALUATE MODEL
accuracy = accuracy_score(y_test, y_pred)*100
f1Score = f1_score(y_test, y_pred, average="macro")
rocAucScore = roc_auc_score(y_test, y_prob)
# precision: accuracy of positive predictions for each class (up/down)
# recall: correctly identified positives / total true positives
# accuracy: correct predictions / total predictions
# F1 score: harmonic mean of precision and recall (0-1)
# ROC-AUC score: chance that a random positive is ranked higher than a random negative (0-1)

# CONFUSION MATRIX
cmatrix = confusion_matrix(y_test, y_pred)
# returns 2x2 numpy array breaking down true/false positives/negatives
cmatrixDf = pd.DataFrame(cmatrix, index=["Real -", "Real +"], columns=["Predict -", "Predict +"])
print(f"Accuracy: {accuracy:.3f}%")
print(f"F1 score (macro-averaged): {f1Score:.5f}")
print(f"ROC-AUC score: {rocAucScore:.5f}")
print(f"Confusion matrix: {cmatrixDf}")

directory = "models"
if not os.path.exists(directory):
    os.makedirs(directory)
filename = f"XGBoost_{instrument}_{granularity}_{yearNow}.json"
filepath = os.path.join(directory, filename)
model.save_model(filepath)
print("\nModel saved to: " + filename)