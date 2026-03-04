# EXPORTS:
# tuneHyperparams() returns DataFrame of results from all folds, and DataFrame of final selected params
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
import optuna
import pandas as pd
import numpy as np
from . import dataparser
from datetime import datetime

# suppress per-trial logs (set to INFO for normal logs, or DEBUG for more logs)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def tuneHyperparams(yearNow, instr, gran,
                    features=[
                            "hl_spread", "body_ratio",
                            "normalised_ema15", "normalised_ema50",
                            "rsi_14", "macd_hist",
                            "atr_14", "bb_width",
                            "vol_ratio", "bb_position",
                            "return_lag1", "return_lag2", "return_lag3", "return_lag4",
                            "vol_ratio_lag1", "vol_ratio_lag2", "vol_ratio_lag3", "vol_ratio_lag4",
                            "upper_wick", "lower_wick", "direction", "volatility_momentum", "vol_trend",
                            "trend_strength", "volatility_regime"
                            ], n=5, deadzone=0.001, midThreshold=0):
    # LOAD DATAFRAME
    df = dataparser.parseData(f"json_data/{instr}_{gran}_{yearNow - 16}-01-01_{yearNow}-01-01.json")

    # INITIALISE CUMULATIVE RESULTS (list of dicts)
    allResults = []

    # LOOP TEST (5 folds)
    for fold in range(5):
        print(f"Starting fold {fold + 1}...")
        # split dataframes
        foldDf = dataparser.splitByDate(df, datetime(yearNow - 15 + 2*fold, 1, 1), datetime(yearNow - 8 + 2*fold, 1, 1))

        # target variable: next n candles net return => negative (0), flat (1), positive (2)
        foldDf["forward_return"] = (foldDf["close"].shift(-n) / foldDf["close"]) - 1
        conditions = [
            foldDf["forward_return"] < midThreshold - deadzone, # downward move
            foldDf["forward_return"] > midThreshold + deadzone # upward move
        ]
        choices = [0, 2]
        foldDf["target"] = np.select(conditions, choices, default=1) # if not up or down, return flat (1)
        foldDf.dropna(inplace=True)
        
        # define datasets
        X_set = foldDf[features]
        y_set = foldDf["target"]
        
        # optuna magic (function is called repeatedly and optuna tries to maximise/minimise return value)
        def objective(trial):
            params = {
                "verbosity": 0,
                "objective": "multi:softprob", # function to minimise (multiclass probability)
                "num_class": 3, # no. of classes
                "max_depth": trial.suggest_int("max_depth", 3, 4),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "subsample": trial.suggest_float("subsample", 0.35, 0.65),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.35, 0.65),
                "min_child_weight": trial.suggest_int("min_child_weight", 60, 120),
                "reg_alpha": trial.suggest_float("reg_alpha", 1, 15, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 10, 30, log=True),
                "device": "cuda", # use gpu
                "tree_method": "hist"
            }

            # subfold split (respects time order)
            crossValSplit = TimeSeriesSplit(n_splits=4) # .split method returns tuples of indexes for training and validation
            foldScores = []

            # loop through subfolds
            for trainIndexes, valIndexes in crossValSplit.split(X_set):
                X_train = X_set.iloc[trainIndexes]
                X_val = X_set.iloc[valIndexes]
                y_train = y_set.iloc[trainIndexes]
                y_val = y_set.iloc[valIndexes]

                # create and train model
                model = xgb.XGBClassifier(**params, eval_metric="mlogloss", # multiclass log loss
                                          n_estimators=1000, # high ceiling
                                          early_stopping_rounds=50, # stop after metric plateaus for 50 rounds
                                          random_state=42)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

                # evaluate and append score
                y_pred = model.predict(X_val)
                score = f1_score(y_val, y_pred, average="macro", zero_division=0)
                foldScores.append(score)
            
            # pass score to study object
            return np.mean(foldScores)
        
        # study object (main optuna magic)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=60, show_progress_bar=True)

        # get results and append to cumulative results
        allResults.append(study.best_params)
    
    # PARSE ALL DATA AND CONCLUDE
    finalParams = pd.DataFrame(allResults).mean() # convert to Series by averaging best values of all parameters
    finalParams.loc["max_depth"] = round(finalParams.loc["max_depth"]) # round to nearest whole number (still a float)
    finalParams.loc["min_child_weight"] = round(finalParams.loc["min_child_weight"])
    
    # RETURN RESULTS
    return finalParams