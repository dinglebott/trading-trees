## RAW DATA COLLECTION
Instrument: EUR_USD with a granularity of 1hr\
Candle data: OHLCV (open, high, low, close, volume)\
Time period: 2010-01-01 to 2026-01-01\
Pulled from OANDA REST-v20 API, stored in JSON format

## DATASETS
**Feature selection and hyperparameter tuning:**\
10 folds of 6-year rolling windows\
Train: 2010-2015, Test: 2016\
Train: 2011-2016, Test: 2017\
...\
Train: 2019-2024, Test: 2025\
**Final model training:**\
Train: 2010-2024, Test: 2025

## INITIAL FEATURE ENGINEERING
Scoring metric for selection: SHAP values\
**Price:**\
Returns => Percentage change from previous close\
High-low spread (normalised) => (H - L) / C\
Open-close spread (normalised) => (C - O) / C\
Body ratio => OC spread / HL spread\
**Trend:**\
12-period EMA (normalised) => (C / EMA) - 1\
50-period EMA (normalised) => (C / EMA) - 1\
**Momentum:**\
14-period RSI\
12/26/9-period MACD histogram => (ema12 - ema26) - signal\
**Volatility:**\
14-period ATR\
Bollinger band width (normalised) => (upperband - lowerband) / midband\
**Volume:**\
Volume ratio => volume / volumesma30\
**Mean reversion:**\
Bollinger band position => (C - lowerband) / (upperband - lowerband)\
**Lagged features:**\
1/2/3/4/5-period lagged returns => Return values of previous 5 candles\
1/2/3/4/5-period lagged volume => Volume values of previous 5 candles

## HYPERPARAMETER TUNING
**Scoring metric:**\
Macro-adjusted F1 score (see Explanation of metrics below)\
**Variations tested:**\
No. of trees:      [100, 200, 300]\
Max tree depth:    [3, 4, 5, 6]\
Learning rate:     [0.01, 0.05, 0.1] => shrinks contribution of each tree\
Data subsample:    [0.7, 0.8, 1.0] => fraction of data sampled per tree\
Feature subsample: [0.7, 0.8, 1.0] => fraction of features sampled per tree\
Min child weight:  [1, 3, 5] => Higher values make model require more evidence to make a split


## MODEL EVALUATION
**Explanation of metrics:**\
Positive = 1 (upward move), Negative = 0 (downward move)\
Accuracy (0-100) => (Correct predictions / Total predictions) * 100%\
F1 score (0-1) => Harmonic mean of Precision and Recall\
F1 score (macro-averaged) => Unweighted mean of F1 score calculated for each class (1 and 0)\
ROC-AUC score (0-1) => Probability that a randomly chosen 1 is ranked higher than a randomly chosen 0 by the model\
Precision (0-1) => Correctly predicted 1's / All predicted 1's\
Recall (0-1) => Correctly predicted 1's / All real 1's

### Model 1
**Target variable:** Return of next candle\
**Train:** 2010 - 2024\
**Test:** 2025\
**Features:** ["return", "oc_spread", "body_ratio", "normalised_ema15", "normalised_ema50", "rsi_14", "vol_ratio", "bb_position", "vol_ratio_lag1", "vol_ratio_lag3", "vol_ratio_lag4"]\
*(features selected by default Gini importance metric, subsequent models use SHAP)*\
**Hyperparameters:** {\
                    "n_estimators": 300,\
                    "max_depth": 5,\
                    "learning_rate": 0.022,\
                    "subsample": 0.76,\
                    "colsample_bytree": 0.85,\
                    "min_child_weight": 5\
                }\
**Accuracy:** 51.834%\
**F1 score (macro-averaged):** 0.51715\
**ROC-AUC score:** 0.52330\
**Confusion matrix:**\
| &nbsp; | Pred - | Pred + |
| --- | --- | --- |
| Real - | 1765 | 1322 |
| Real + | 1672 | 1457 |

### Model 2
*Changes from v1: Target variable changed to net return of next 5 candles*\
**Features:** ["atr_14", "body_ratio", "normalised_ema50", "vol_ratio_lag5", "bb_width", "rsi_14", "macd_hist", "vol_ratio_lag4", "bb_position", "vol_ratio", "normalised_ema15"]\
**Hyperparameters:** {\
"n_estimators": 100,\
"max_depth": 4,\
"learning_rate": 0.07,\
"subsample": 0.78,\
"colsample_bytree": 0.84,\
"min_child_weight": 5\
}\
**Accuracy:** 52.220%\
**F1 score (macro-averaged):** 0.52151\
**ROC-AUC score:** 0.52614\
**Confusion matrix:**\
| &nbsp; | Pred - | Pred + |
| --- | --- | --- |
| Real - | 1505 | 1515 |
| Real + | 1455 | 1741 |

### Model 2.1
*Changes from v2: Added reg_alpha and reg_lambda hyperparameters*\
**Hyperparameters:** {\
"n_estimators": 100,\
"max_depth": 6,\
"learning_rate": 0.062,\
"subsample": 0.82,\
"colsample_bytree": 0.9,\
"min_child_weight": 5,\
"reg_alpha": 1.741,\
"reg_lambda": 12.61\
}\
**Accuracy:** 51.577%\
**F1 score (macro-averaged):** 0.51493\
**ROC-AUC score:** 0.52487\
**Confusion matrix:**\
| &nbsp; | Pred - | Pred + |
| --- | --- | --- |
| Real - | 1474 | 1546 |
| Real + | 1464 | 1732 |

### Model 2.2
*Changes from v2: Granularity H4 and prediction changed to next 7 candles*\
**Features:** ["normalised_ema50", "macd_hist", "atr_14", "bb_width", "bb_position", "rsi_14", "vol_ratio_lag2", "normalised_ema15", "vol_ratio_lag5", "vol_ratio_lag1", "return_lag3"]\
**Hyperparameters:** {\
"n_estimators": 200,\
"max_depth": 6,\
"learning_rate": 0.067,\
"subsample": 0.78,\
"colsample_bytree": 0.89,\
"min_child_weight": 5\
}\
**Accuracy:** 49.292%\
**F1 score (macro-averaged):** 0.49231\
**ROC-AUC score:** 0.50226\
**Confusion matrix:**\
| &nbsp; | Pred - | Pred + |
| --- | --- | --- |
| Real - | 356 | 412 |
| Real + | 376 | 410 |

### Model 3
*Changes from v2: Granularity H4, prediction changed to next 4 candles, added a third class "flat"*\
**Features:** ["atr_14", "vol_ratio_lag3", "normalised_ema50", "bb_width", "rsi_14", "macd_hist", "vol_ratio_lag4", "hl_spread", "vol_ratio_lag1", "vol_ratio", "bb_position"]\
**Hyperparameters:** {\
"n_estimators": 200,\
"max_depth": 4,\
"learning_rate": 0.085,\
"subsample": 0.88,\
"colsample_bytree": 0.89,\
"min_child_weight": 3\
}\
**Accuracy:** 39.097%\
**F1 score (macro-averaged):** 0.33958\
**ROC-AUC score:** 0.55234\
**Confusion matrix:**\
| &nbsp; | Pred - | Pred ~ | Pred + |
| --- | --- | --- | --- |
| Real - | 241 | 23 | 301 |
| Real ~ | 121 | 37 | 225 |
| Real + | 239 | 35 | 328 |