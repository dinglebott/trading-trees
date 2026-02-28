# EXPORTS:
# evaluateFeatures() gets feature importances and returns as a Series
import xgboost as xgb
import pandas as pd
from . import dataparser
from datetime import datetime

def evaluateFeatures(yearNow, instr, gran,
                 params={
                     "n_estimators": 200,
                     "max_depth": 4,
                     "learning_rate": 0.05,
                     "subsample": 0.8,
                     "colsample_bytree": 0.8,
                     "min_child_weight": 3
                 }):
    # CREATE MODEL
    model = xgb.XGBClassifier(**params, eval_metric="logloss", random_state=42)

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
    
    # LOAD DATAFRAME
    df = dataparser.parseData(f"json_data/{instr}_{gran}_{yearNow - 16}-01-01_{yearNow}-01-01.json")

    # INITIALISE CUMULATIVE SCORE
    avgImportances = pd.Series(0.0, index=features)

    # LOOP TEST THROUGH ALL FOLDS
    for fold in range(10):
        # split dataframes
        dfTrain = dataparser.splitByDate(df, datetime(yearNow - 16 + fold, 1, 1), datetime(yearNow - 9 + fold, 1, 1))

        # target variable: next candle return => positive (1) or negative (0)
        dfTrain["target"] = (dfTrain["return"].shift(-1) > 0).astype(int) # boolean to integer
        dfTrain.dropna(inplace=True)
        
        # define datasets
        X_train = dfTrain[features]
        y_train = dfTrain["target"]

        # train model
        model.fit(X_train, y_train)

        # feature importance
        importances = pd.Series(model.feature_importances_, index=features)
        # model.feature_importances_ returns numpy array (indicates how important each feature was by weight)

        # add to cumulative score
        avgImportances += importances

    # RETURN TEST DATA
    avgImportances /= 10
    avgImportances.sort_values(ascending=False, inplace=True)
    return avgImportances