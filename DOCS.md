## RAW DATA COLLECTION
Instrument: EUR_USD with a granularity of 1hr\
Candle data: OHLCV (open, high, low, close, volume)\
Time period: 2010-01-01 to 2026-01-01\
Pulled from OANDA REST-v20 API, stored in JSON format\
<br/>

## DATASETS
**Feature selection (SHAP):**\
10 folds of 7-year rolling windows\
Train: 2010-2015, Test: 2016\
Train: 2011-2016, Test: 2017\
...\
Train: 2019-2024, Test: 2025\
**Hyperparameter tuning (Optuna):**\
5 folds of 7-year rolling windows (advance by 2 years per fold)\
2011-2017, 2013-2019, ..., 2019-2025\
Cross-validation performed by dividing into subfolds\
**Final model training:**\
Train: 2010-2024, Test: 2025\
<br/>

## INITIAL FEATURE ENGINEERING
**Scoring metric:** SHAP values\
**Price:**\
Return => Percentage change from previous close *(Removed in v5+)*\
High-low spread (normalised) => (H - L) / C\
Open-close spread (normalised) => (C - O) / C *(Removed in v5+)*\
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
1/2/3/4/5-period lagged returns => Return values of previous 5 candles *(Removed lag5 in v5+)*\
1/2/3/4/5-period lagged volume => Volume values of previous 5 candles *(Removed lag5 in v5+)*\
**Hybrid features (added in v5+):**\
Volatility-adjusted momentum => atr_14 * rsi_14\
Volume-trend confluence => vol_ratio * ema50\
Trend strength => abs(ema15 - ema50)\
Volatility regime => atr_14 / atr_14 50-period mean\
Candle direction => Sign of close - open\
Upper wick => (High - candle top) / atr_14\
Lower wick => (Candle bottom - low) / atr_14\
**More features (added in v5.1+):**\
Volatility-adjusted return => return / atr_14\
Return acceleration => return_lag1 - return_lag2\
Volume momentum => vol_ratio - vol_ratio 5-period mean\
Distance from EMA (volatility-adjusted) => (C - ema15) / atr_14\
<br/>

## HYPERPARAMETER TUNING
**Scoring metric:** Macro-adjusted F1 score (see Explanation of metrics below)\
**Hyperparameters tested:**\
n_estimators: No. of decision trees to build\
max_depth: Maximum number of levels each tree can grow its depth to - lower values can reduce overfitting by limiting complexity\
learning_rate: Lower value reduces the contribution of each tree and prevents overfitting\
subsample: Fraction of data sampled per tree\
colsample_bytree: Fraction of features sampled per tree\
min_child_weight: Higher values make model require more evidence to make a split\
<br/>

## MODEL EVALUATION
**Explanation of metrics:**\
Positive = 1 (upward move), Negative = 0 (downward move)\
Accuracy (0-100) => (Correct predictions / Total predictions) * 100%\
F1 score (0-1) => Harmonic mean of Precision and Recall\
F1 score (macro-averaged) => Unweighted mean of F1 score calculated for each class (1 and 0)\
ROC-AUC score (0-1) => Probability that a randomly chosen 1 is ranked higher than a randomly chosen 0 by the model\
Precision (0-1) => Correctly predicted 1's / All predicted 1's\
Recall (0-1) => Correctly predicted 1's / All real 1's\
<br/>

### Model 5.1
*Changes from v5: Added more features, include 15 instead of 11 features, calibrated min_child_weight*\
**Features:** ["atr_14", "volatility_momentum", "vol_ratio_lag3", "vol_ratio_lag4", "volatility_regime", "hl_spread", "vol_ratio_lag1", "normalised_ema50", "trend_strength", "vol_momentum", "bb_width", "vol_ratio", "macd_hist", "upper_wick", "vol_trend"]\
**Hyperparameters:** {\
"max_depth": 4 *(3, 4)*,\
"learning_rate": 0.07579 *(0.005, 0.1)*,\
"subsample": 0.40476 *(0.35, 0.65)*,\
"colsample_bytree": 0.49986 *(0.35, 0.65)*,\
"min_child_weight": 91 *(60, 120)*,\
"reg_alpha": 4.82835 *(1, 15)*,\
"reg_lambda": 16.83605 *(10, 30)*\
}\
*(Search spaces in italicised brackets)*\
**Accuracy:** 41.677%\
**F1 score (macro-averaged):** 0.40622\
**F1 score (train set):** 0.44044\
**ROC-AUC score:** 0.59140\
**Confusion matrix:**
| &nbsp; | Pred - | Pred ~ | Pred + |
| --- | --- | --- | --- |
| Real - | 161 | 181 | 130 |
| Real ~ | 117 | 317 | 115 |
| Real + | 165 | 196 | 168 |
<br/>

### Model 5
*Changes from v4.3: Added new features*\
**Features:** ["atr_14", "volatility_momentum", "vol_ratio_lag3", "vol_ratio_lag4", "volatility_regime", "hl_spread", "vol_ratio_lag1", "vol_ratio", "normalised_ema50", "trend_strength", "bb_width"]\
**Hyperparameters:** {\
"max_depth": 3 *(3, 4)*,\
"learning_rate": 0.05885 *(0.005, 0.1)*,\
"subsample": 0.52596 *(0.35, 0.65)*,\
"colsample_bytree": 0.51763 *(0.35, 0.65)*,\
"min_child_weight": 84 *(40, 100)*,\
"reg_alpha": 4.38079 *(1, 15)*,\
"reg_lambda": 22.18329 *(10, 30)*\
}\
*(Search spaces in italicised brackets)*\
**Accuracy:** 41.419%\
**F1 score (macro-averaged):** 0.40125\
**F1 score (train set):** 0.43508\
**ROC-AUC score:** 0.59432\
**Confusion matrix:**
| &nbsp; | Pred - | Pred ~ | Pred + |
| --- | --- | --- | --- |
| Real - | 145 | 194 | 133 |
| Real ~ | 113 | 325 | 111 |
| Real + | 161 | 196 | 172 |
<br/>

### Model 4.3
*Changes from v4.2: Implemented early stopping, utilise CUDA for training, tightened Optuna search space further*\
**Features:** ["atr_14", "vol_ratio_lag3", "vol_ratio_lag4", "normalised_ema50", "vol_ratio", "hl_spread", "vol_ratio_lag1", "bb_width", "macd_hist", "rsi_14", "bb_position"]\
**Hyperparameters:** {\
"max_depth": 4 *(3, 4)*,\
"learning_rate": 0.05196 *(0.005, 0.1)*,\
"subsample": 0.49914 *(0.35, 0.65)*,\
"colsample_bytree": 0.43479 *(0.35, 0.65)*,\
"min_child_weight": 60 *(40, 100)*,\
"reg_alpha": 8.42810 *(1, 15)*,\
"reg_lambda": 21.18867 *(10, 30)*\
}\
*(Search spaces in italicised brackets)*\
**Accuracy:** 40.323%\
**F1 score (macro-averaged):** 0.38989\
**F1 score (train set):** 0.43716\
**ROC-AUC score:** 0.57782\
**Confusion matrix:**
| &nbsp; | Pred - | Pred ~ | Pred + |
| --- | --- | --- | --- |
| Real - | 128 | 188 | 156 |
| Real ~ | 104 | 313 | 132 |
| Real + | 138 | 207 | 184 |
<br/>

### Model 4.2
*Changes from v4.1: Deadzone = 0.0015, tightened Optuna search space further to reduce overfitting*\
**Features:** ["atr_14", "vol_ratio_lag3", "bb_width", "normalised_ema50", "hl_spread", "vol_ratio_lag4", "vol_ratio_lag1", "rsi_14", "macd_hist", "vol_ratio", "bb_position"]\
**Hyperparameters:** {\
"n_estimators": 440 *(100, 700)*,\
"max_depth": 4 *(3, 4)*,\
"learning_rate": 0.07122 *(0.005, 0.1)*,\
"subsample": 0.54548 *(0.4, 0.65)*,\
"colsample_bytree": 0.46456 *(0.4, 0.65)*,\
"min_child_weight": 56 *(1, 100)*,\
"reg_alpha": 5.43709 *(1, 10)*,\
"reg_lambda": 16.82634 *(10, 30)*\
}\
*(Search spaces in italicised brackets)*\
**Accuracy:** 39.871%\
**F1 score (macro-averaged):** 0.39014\
**F1 score (train set):** 0.51425\
**ROC-AUC score:** 0.56332\
**Confusion matrix:**
| &nbsp; | Pred - | Pred ~ | Pred + |
| --- | --- | --- | --- |
| Real - | 151 | 181 | 140 |
| Real ~ | 118 | 296 | 135 |
| Real + | 181 | 177 | 171 |
<br/>

### Model 4.1
*Changes from v4: Deadzone = 0.002, tightened Optuna search space to reduce overfitting*\
**Features:** ["atr_14", "vol_ratio_lag3", "bb_width", "normalised_ema50", "hl_spread", "vol_ratio_lag4", "vol_ratio_lag1", "rsi_14", "macd_hist", "vol_ratio", "bb_position"]\
**Hyperparameters:** {\
"n_estimators": 600 *(100, 700)*,\
"max_depth": 5 *(3, 5)*,\
"learning_rate": 0.08105 *(0.005, 0.1)*,\
"subsample": 0.58484 *(0.4, 0.8)*,\
"colsample_bytree": 0.62415 *(0.4, 0.8)*,\
"min_child_weight": 33 *(1, 100)*,\
"reg_alpha": 0.49837 *(0.01, 5)*,\
"reg_lambda": 10.54544 *(5, 20)*\
}\
*(Search spaces in italicised brackets)*\
**Accuracy:** 44.581%\
**F1 score (macro-averaged):** 0.35252\
**F1 score (train set):** 0.60808\
**ROC-AUC score:** 0.56264\
**Confusion matrix:**
| &nbsp; | Pred - | Pred ~ | Pred + |
| --- | --- | --- | --- |
| Real - | 66 | 259 | 51 |
| Real ~ | 92 | 543 | 87 |
| Real + | 92 | 278 | 82 |
<br/>

### Model 4
*Changes from v3: Switched to Optuna for hyperparameter tuning instead of RandomizedSearchCV*\
**Features:** ["atr_14", "vol_ratio_lag3", "normalised_ema50", "bb_width", "vol_ratio_lag4", "rsi_14", "vol_ratio_lag1", "macd_hist", "hl_spread", "vol_ratio", "return_lag4"]\
**Hyperparameters:** {\
"n_estimators": 570 *(100, 700)*,\
"max_depth": 5 *(3, 6)*,\
"learning_rate": 0.05155 *(0.005, 0.1)*,\
"subsample": 0.71238 *(0.5, 1.0)*,\
"colsample_bytree": 0.86145 *(0.5, 1.0)*,\
"min_child_weight": 37 *(1, 100)*,\
"reg_alpha": 0.87683 *(0.01, 5)*,\
"reg_lambda": 4.14204 *(1, 20)*\
}\
*(Search spaces in italicised brackets)*\
**Accuracy:** 38.323%\
**F1 score (macro-averaged):** 0.33282\
**F1 score (train set):** 0.59334\
**ROC-AUC score:** 0.53686\
**Confusion matrix:**
| &nbsp; | Pred - | Pred ~ | Pred + |
| --- | --- | --- | --- |
| Real - | 230 | 35 | 300 |
| Real ~ | 143 | 37 | 203 |
| Real + | 232 | 43 | 327 |
<br/>

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
**Confusion matrix:**
| &nbsp; | Pred - | Pred ~ | Pred + |
| --- | --- | --- | --- |
| Real - | 241 | 23 | 301 |
| Real ~ | 121 | 37 | 225 |
| Real + | 239 | 35 | 328 |
<br/>

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
**Confusion matrix:**
| &nbsp; | Pred - | Pred + |
| --- | --- | --- |
| Real - | 356 | 412 |
| Real + | 376 | 410 |
<br/>

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
**Confusion matrix:**
| &nbsp; | Pred - | Pred + |
| --- | --- | --- |
| Real - | 1474 | 1546 |
| Real + | 1464 | 1732 |
<br/>

### Model 2
*Changes from v1: Target variable changed to net return of next 5 candles, feature selection metric changed to SHAP values*\
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
**Confusion matrix:**
| &nbsp; | Pred - | Pred + |
| --- | --- | --- |
| Real - | 1505 | 1515 |
| Real + | 1455 | 1741 |
<br/>

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
**Confusion matrix:**
| &nbsp; | Pred - | Pred + |
| --- | --- | --- |
| Real - | 1765 | 1322 |
| Real + | 1672 | 1457 |