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

### Model 0
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
**Confusion matrix:** TN(1765), FP(1322), FN(1672), TP(1457)