import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from custom_modules import dataparser

# load DataFrames
dfTrain = dataparser.parseData("json_data/EUR_USD/H1/fold_1/EUR_USD_H1_2010-01-01_2016-01-01_0225-1009.json")
dfTest = dataparser.parseData("json_data/EUR_USD/H1/fold_1/EUR_USD_H1_2016-01-01_2017-01-01_0225-1009.json")

# target variable: predict next candle return => positive (1) or negative (0)
for df in (dfTrain, dfTest):
    df["target"] = (df["return"].shift(-1) > 0).astype(int)
    df.dropna(inplace=True)

# define features
features = [
    "return", "high_low_spread", "open_close_spread", "body_ratio",
    "ema_12", "ema_50",
    "rsi_14", "macd_hist",
    "atr_14", "bb_width",
    "volume_ratio", "bb_position",
    "lag_return_1", "lag_return_2", "lag_return_3", "lag_return_4", "lag_return_5",
    "lag_volume_1", "lag_volume_2", "lag_volume_3", "lag_volume_4", "lag_volume_5",
]

x_train = dfTrain[features]
y_train = dfTrain["target"]
x_test = dfTest[features]
y_test = dfTest["target"]

# ── 5. OPTIONAL: SCALE FEATURES ─────────────────────────────────────────────
# XGBoost is tree-based so scaling isn't strictly necessary,
# but it doesn't hurt and helps if you later compare with other models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)   # fit ONLY on train
X_test_scaled  = scaler.transform(x_test)        # transform test with same scaler

# ── 6. TRAIN BASELINE MODEL ─────────────────────────────────────────────────
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
)

model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    verbose=50,
)

# ── 7. EVALUATE ──────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ── 8. FEATURE IMPORTANCE ───────────────────────────────────────────────────
importances = pd.Series(model.feature_importances_, index=features)
importances.sort_values().plot(kind='barh', figsize=(8, 8), title='Feature Importance')
plt.tight_layout()
plt.show()

# ── 9. HYPERPARAMETER TUNING WITH TIME-SERIES CROSS VALIDATION ──────────────
tscv = TimeSeriesSplit(n_splits=5)    # respects temporal ordering across folds

param_grid = {
    'n_estimators':     [100, 200, 300],
    'max_depth':        [3, 4, 5, 6],
    'learning_rate':    [0.01, 0.05, 0.1],
    'subsample':        [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
}

search = RandomizedSearchCV(
    estimator=xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    param_distributions=param_grid,
    n_iter=40,              # number of random combinations to try
    scoring='accuracy',
    cv=tscv,
    verbose=1,
    n_jobs=-1,
    random_state=42,
)

search.fit(X_train_scaled, y_train)

print(f"\nBest params: {search.best_params_}")
print(f"Best CV accuracy: {search.best_score_:.4f}")

# ── 10. FINAL MODEL WITH BEST PARAMS ────────────────────────────────────────
best_model = search.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)

print("\n── Tuned Model Performance ──")
print(classification_report(y_test, y_pred_best, target_names=['Down', 'Up']))

# ── 11. SAVE MODEL + SCALER ─────────────────────────────────────────────────
best_model.save_model('xgb_model.json')
joblib.dump(scaler, 'scaler.pkl')
print("\nModel and scaler saved.")