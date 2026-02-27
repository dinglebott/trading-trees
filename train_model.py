import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from custom_modules import dataparser

# CREATE MODEL
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
)

# LOAD DATAFRAMES
dfTrain = dataparser.parseData("json_data/EUR_USD/H1/fold_1/EUR_USD_H1_2010-01-01_2016-01-01_0227.json")
dfTest = dataparser.parseData("json_data/EUR_USD/H1/fold_1/EUR_USD_H1_2016-01-01_2017-01-01_0227.json")

# TARGET VARIABLE: next candle return => positive (1) or negative (0)
for df in (dfTrain, dfTest):
    df["target"] = (df["return"].shift(-1) > 0).astype(int) # boolean to integer
    df.dropna(inplace=True)

# DEFINE FEATURES
features = [
    "return", "hl_spread", "oc_spread", "body_ratio",
    "normalised_ema15", "normalised_ema50",
    "rsi_14", "macd_hist",
    "atr_14", "bb_width",
    "vol_ratio", "bb_position",
    "return_lag1", "return_lag2", "return_lag3", "return_lag4", "return_lag5",
    "vol_ratio_lag1", "vol_ratio_lag2", "vol_ratio_lag3", "vol_ratio_lag4", "vol_ratio_lag5"
]
X_train = dfTrain[features]
y_train = dfTrain["target"]
X_test = dfTest[features]
y_test = dfTest["target"]

# TRAIN MODEL
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)], # evaluate while training
    verbose=50 # print eval metric every 50 rounds
)

# TEST MODEL
y_pred = model.predict(X_test)
# returns 1D array of shape (n_samples)
# values 0 | 1
y_prob = model.predict_proba(X_test)[:, 1]
# returns 2D array of shape (n_samples, n_classes)
# col 0 = P(down), col 1 = P(up)

# EVALUATE MODEL
print(f"\nAccuracy: {(accuracy_score(y_test, y_pred)*100):.3f}%") # percentage accuracy to 3dp
print(f"F1 score: {f1_score(y_test, y_pred):.5f}")
print(f"ROC-AUC score: {roc_auc_score(y_test, y_pred)}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Down", "Up"]))
# precision: accuracy of positive predictions for each class (up/down)
# recall: correctly identified positives / total true positives
# accuracy: correct predictions / total predictions
# F1 score: harmonic mean of precision and recall (0-1)
# ROC-AUC score: chance that a random positive is ranked higher than a random negative (0-1)

# CONFUSION MATRIX
cmatrix = confusion_matrix(y_test, y_pred)
# returns 2x2 numpy array breaking down true/false positives/negatives
cmatrixDF = pd.DataFrame(cmatrix, index=["Real -", "Real +"], columns=["Pred -", "Pred +"])
print(f"\n{cmatrixDF}")

# FEATURE IMPORTANCE
importances = pd.Series(model.feature_importances_, index=features)
# model.feature_importances_ returns numpy array (indicates how important each feature was by weight)
importances.sort_values().plot(kind="barh", figsize=(8, 8), title="FEATURE IMPORTANCE")
plt.tight_layout()
plt.show()

# SAVE MODEL
model.save_model("xgb_model.json")
print("\nModel saved.")