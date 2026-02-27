# EXPORTS:
# evaluateFeatures()
import pandas as pd
import xgboost as xgb
import dataparser

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
    
    # CUMULATIVE SCORE
    avgImportances = pd.Series(0.0, index=features)

    # LOOP TEST THROUGH ALL FOLDS
    for index, year in enumerate(range(yearNow - 16, yearNow - 6), start=1):
        # LOAD DATAFRAMES
        dfTrain = dataparser.parseData(f"json_data/{instr}/{gran}/fold_{index}/{instr}_{gran}_{year}-01-01_{year + 6}-01-01.json")
        dfTest = dataparser.parseData(f"json_data/{instr}/{gran}/fold_{index}/{instr}_{gran}_{year + 6}-01-01_{year + 7}-01-01.json")

        # TARGET VARIABLE: next candle return => positive (1) or negative (0)
        for df in (dfTrain, dfTest):
            df["target"] = (df["return"].shift(-1) > 0).astype(int) # boolean to integer
            df.dropna(inplace=True)
        
        # DEFINE DATASETS
        X_train = dfTrain[features]
        y_train = dfTrain["target"]
        X_test = dfTest[features]
        y_test = dfTest["target"]

        # TRAIN MODEL
        model.fit(X_train, y_train)

        # FEATURE IMPORTANCE
        importances = pd.Series(model.feature_importances_, index=features)
        # model.feature_importances_ returns numpy array (indicates how important each feature was by weight)

        # ADD TO CUMULATIVE TEST SCORES
        avgImportances += importances

    # RETURN TEST DATA
    avgImportances /= 10
    return avgImportances