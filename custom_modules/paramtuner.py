# EXPORTS:
# tuneHyperparams() returns DataFrame of results from all folds, and DataFrame of final selected params
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import optuna
import pandas as pd
import numpy as np
from . import dataparser
from datetime import datetime

def tuneHyperparams(yearNow, instr, gran,
                    features=[
                            "return", "hl_spread", "oc_spread", "body_ratio",
                            "normalised_ema15", "normalised_ema50",
                            "rsi_14", "macd_hist",
                            "atr_14", "bb_width",
                            "vol_ratio", "bb_position",
                            "return_lag1", "return_lag2", "return_lag3", "return_lag4", "return_lag5",
                            "vol_ratio_lag1", "vol_ratio_lag2", "vol_ratio_lag3", "vol_ratio_lag4", "vol_ratio_lag5"
                            ], n=5, deadzone=0.001, midThreshold=0):
    # SUBFOLD SPLIT (respects time order)
    tscv = TimeSeriesSplit(n_splits=5)

    # DEFINE PARAMS
    param_grid = {
        "n_estimators":     [100, 200, 300],
        "max_depth":        [3, 4, 5, 6],
        "learning_rate":    [0.01, 0.05, 0.1],
        "subsample":        [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5]
    }
    
    # LOAD DATAFRAME
    df = dataparser.parseData(f"json_data/{instr}_{gran}_{yearNow - 16}-01-01_{yearNow}-01-01.json")

    # INITIALISE CUMULATIVE RESULTS
    allResults = pd.DataFrame(columns=list(param_grid.keys()))

    # LOOP TEST (builds a fresh model for each combination of hyperparameters)
    for fold in range(10):
        # split dataframes
        dfTrain = dataparser.splitByDate(df, datetime(yearNow - 16 + fold, 1, 1), datetime(yearNow - 9 + fold, 1, 1))

        # target variable: next n candles net return => negative (0), flat (1), positive (2)
        dfTrain["forward_return"] = (dfTrain["close"].shift(-n) / dfTrain["close"]) - 1
        conditions = [
            dfTrain["forward_return"] < midThreshold - deadzone, # downward move
            dfTrain["forward_return"] > midThreshold + deadzone # upward move
        ]
        choices = [0, 2]
        dfTrain["target"] = np.select(conditions, choices, default=1) # if not up or down, return flat (1)
        dfTrain.dropna(inplace=True)
        
        # define datasets
        X_train = dfTrain[features]
        y_train = dfTrain["target"]
        
        # create new tester object
        search = RandomizedSearchCV(
            estimator=xgb.XGBClassifier(eval_metric="logloss", random_state=42),
            param_distributions=param_grid,
            n_iter=50, # tries 50 random combinations from param_grid
            scoring="f1_macro", # metric to evaluate by
            cv=tscv, # cross-validation method (the time-series split defined above)
            verbose=1, # print progress as folds complete
            n_jobs=-1, # uses all CPU cores
            random_state=42 # seed for random combination sampling (for reproducibility)
        )

        # train and compare models
        search.fit(X_train, y_train)

        # append to aggregate results
        allResults.loc[len(allResults)] = search.best_params_
    
    # PARSE ALL DATA AND CONCLUDE
    finalParams = {
        "n_estimators":     int(allResults["n_estimators"].mode()[0]),
        "max_depth":        int(allResults["max_depth"].mode()[0]),
        "learning_rate":    allResults["learning_rate"].mean(),
        "subsample":        allResults["subsample"].mean(),
        "colsample_bytree": allResults["colsample_bytree"].mean(),
        "min_child_weight": int(allResults["min_child_weight"].mode()[0])
    }
    finalParams = pd.DataFrame.from_dict(finalParams, orient="index", columns=["Best value"])
    
    # RETURN RESULTS
    return allResults, finalParams