# RAW DATA COLLECTION\
Instrument: EUR_USD with a granularity of 1hr\
Candle data: OHLCV (open, high, low, close, volume)\
Time period: 2010 to 2025\
Pulled from OANDA REST-v20 API, stored in JSON format\
\
# DATASETS\
10 folds of 6-year rolling windows\
Train: 2010-2015, Test: 2016\
Train: 2011-2016, Test: 2017\
...\
Train: 2019-2024, Test: 2025\
\
# INITIAL FEATURE ENGINEERING\
## Price:\
Returns => Percentage change from previous close\
High-low-spread (normalised) => (H - L) / C\
Open-close-spread (normalised) => |O - C| / C\
## Trend:\
12-period EMA (normalised) => (C / EMA) - 1\
50-period SMA (normalised) => (C / SMA) - 1\
## Momentum:\
## Volatility:\
## Volume:\