# EXPORTS:
# evaluateFeatures() gets feature importances and returns as a Series
import xgboost as xgb
import pandas as pd
import numpy as np
from . import dataparser
from datetime import datetime
import shap

def evaluateFeatures(yearNow, instr, gran,
                 params={
                     "max_depth": 4,
                     "learning_rate": 0.07,
                     "subsample": 0.5,
                     "colsample_bytree": 0.5,
                     "min_child_weight": 140,
                     "reg_alpha": 14,
                     "reg_lambda": 25,
                     "device": "cuda", # use gpu
                     "tree_method": "hist"
                 }, n=5, deadzone=0.001, midThreshold=0):
    # DEFINE FEATURES
    features = [
        "hl_spread", "body_ratio",
        "normalised_ema15", "normalised_ema50",
        "rsi_14", "macd_hist",
        "atr_14", "bb_width",
        "vol_ratio", "bb_position",
        "return_lag1", "return_lag2", "return_lag3", "return_lag4",
        "vol_ratio_lag1", "vol_ratio_lag2", "vol_ratio_lag3", "vol_ratio_lag4",
        "volatility_momentum", "vol_trend", # v5+
        "trend_strength", "volatility_regime", # v5+
        "atr_adjusted_return", "vol_momentum", "dist_ema15" # v5.1+
    ]
    
    # LOAD DATAFRAME
    df = dataparser.parseData(f"json_data/{instr}_{gran}_{yearNow - 16}-01-01_{yearNow}-01-01.json")

    # INITIALISE CUMULATIVE SCORES
    allAvgShaps = pd.Series(0.0, index=features)

    # LOOP TEST THROUGH ALL FOLDS
    for fold in range(10):
        print(f"\nStarting fold {fold + 1}...")
        # split dataframes
        dfTrain = dataparser.splitByDate(df, datetime(yearNow - 16 + fold, 1, 1), datetime(yearNow - 11 + fold, 1, 1))
        dfVal = dataparser.splitByDate(df, datetime(yearNow - 11 + fold, 1, 1), datetime(yearNow - 10 + fold, 1, 1))
        dfTest = dataparser.splitByDate(df, datetime(yearNow - 10 + fold, 1, 1), datetime(yearNow - 9 + fold, 1, 1))
        dfTrain.drop(dfTrain.tail(n).index, inplace=True) # drop last n rows to prevent data leakage

        # target variable: next n candles net return => negative (0), flat (1), positive (2)
        for dataset in (dfTrain, dfVal, dfTest):
            dataset["forward_return"] = (dataset["close"].shift(-n) / dataset["close"]) - 1
            conditions = [
                dataset["forward_return"] < midThreshold - 0.5*dataset["atr_14"], # downward move
                dataset["forward_return"] > midThreshold + 0.5*dataset["atr_14"] # upward move
            ]
            choices = [0, 2]
            dataset["target"] = np.select(conditions, choices, default=1) # if not up or down, return flat (1)
            dataset.dropna(inplace=True)
        
        # define datasets
        X_train = dfTrain[features]
        y_train = dfTrain["target"]
        X_val = dfVal[features]
        y_val = dfVal["target"]
        X_test = dfTest[features]

        # train model
        model = xgb.XGBClassifier(**params, eval_metric="mlogloss",
                                  n_estimators=1000, # high ceiling
                                  early_stopping_rounds=50, # stop after metric plateaus for 50 rounds
                                  random_state=42)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # feature importance
        explainer = shap.TreeExplainer(model, X_train, feature_perturbation="interventional")
        shapValues = explainer(X_test, check_additivity=False) # shapValues is an Explanation object
        # shapValues.values returns a numpy array of shape (n_samples, n_features, n_classes)
        foldAvgShaps = np.mean(np.abs(shapValues.values), axis=(0, 2)) # foldAvgShaps is a numpy array, values averaged over samples and classes

        # add to cumulative score
        allAvgShaps += pd.Series(foldAvgShaps, index=features)

    # RETURN TEST DATA
    allAvgShaps = allAvgShaps / 10
    allAvgShaps.sort_values(ascending=False, inplace=True)
    return allAvgShaps