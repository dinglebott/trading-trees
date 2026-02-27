import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import f1_score
import pandas as pd
from custom_modules import dataparser

# split into subfolds (respects time order)
tscv = TimeSeriesSplit(n_splits=5)

# define hyperparameters to test
param_grid = {
    "n_estimators":     [100, 200, 300],
    "max_depth":        [3, 4, 5, 6],
    "learning_rate":    [0.01, 0.05, 0.1],
    "subsample":        [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5]
}

# aggregate results from all folds
hyperparams = ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight"]
allResults = pd.DataFrame(columns=hyperparams)

# testing loop (builds a fresh model for each combination of hyperparameters)
for fold in range(1, 11):
    # load dataframes
    dfTrain = dataparser.parseData(f"json_data/EUR_USD/H1/fold_{fold}/EUR_USD_H1_{fold+2009}-01-01_{fold+2015}-01-01_0227.json")
    dfTest = dataparser.parseData(f"json_data/EUR_USD/H1/fold_{fold}/EUR_USD_H1_{fold+2015}-01-01_{fold+2016}-01-01_0227.json")

    # target variable: next candle return => positive (1) or negative (0)
    for df in (dfTrain, dfTest):
        df["target"] = (df["return"].shift(-1) > 0).astype(int) # boolean to integer
        df.dropna(inplace=True)

    # define features
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

    # create new tester object
    search = RandomizedSearchCV(
        estimator=xgb.XGBClassifier(eval_metric="logloss", random_state=42),
        param_distributions=param_grid,
        n_iter=40, # tries 40 random combinations from param_grid
        scoring="f1", # metric to evaluate by
        cv=tscv, # cross-validation method (the time-series split defined above)
        verbose=1, # print progress as folds complete
        n_jobs=-1, # uses all CPU cores
        random_state=42 # seed for random combination sampling (for reproducibility)
    )

    # test and compare
    search.fit(X_train, y_train)
    bestModel = search.best_estimator_ # save best model
    yPred = bestModel.predict(X_test)
    testF1 = f1_score(y_test, yPred) # get F1 score of best model on test set

    # print results
    print(f"\nFOLD {fold}:")
    print(f"Best params: {search.best_params_}")
    print(f"Best F1 score on test set: {(testF1*100):.3f}%")

    # append to aggregate results
    allResults.loc[len(allResults)] = search.best_params_

# print aggregate results dataframe
print("\nAggregate results:")
print(allResults)