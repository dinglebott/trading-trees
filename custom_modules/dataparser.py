# EXPORTS:
# parseData()
import json
import numpy as np
import pandas as pd

def parseData(jsonPath):
    # deserialise json data
    with open(jsonPath, "r") as file:
        rawData = json.load(file) # rawData is a Python dict
    
    # unpack dict into DataFrame
    records = []
    for c in rawData["candles"]:
        if c["complete"]:
            records.append({
                "time": c["time"],
                "open": float(c["mid"]["o"]), # convert from string
                "high": float(c["mid"]["h"]),
                "low": float(c["mid"]["l"]),
                "close": float(c["mid"]["c"]),
                "volume": c["volume"]
            })
    df = pd.DataFrame(records)

    # ADD FEATURES
    # helper
    def getEma(period):
        return df["close"].ewm(span=period, adjust=False).mean()
    # returns
    df["return"] = df["close"].pct_change()
    # spreads
    df["hl_spread"] = (df["high"] - df["low"]) / df["close"]
    df["oc_spread"] = (df["close"] - df["open"]) / df["close"]
    df["body_ratio"] = (df["oc_spread"] / df["hl_spread"]).clip(-1, 1) # prevent infinity values
    # EMAs
    for period in (15, 50):
        rawEma = getEma(period)
        df[f"normalised_ema{period}"] = (df["close"] / rawEma) - 1
    # RSI
    def rsi(series, n=14):
        delta = series.diff()
        avgGain = delta.clip(lower=0).rolling(n).mean()
        avgLoss = (-delta.clip(upper=0)).rolling(n).mean()
        relativeStrength = avgGain / avgLoss
        return 100 - (100 / (1 + relativeStrength))
    df["rsi_14"] = rsi(df["close"])
    # MACD histogram
    macd = getEma(12) - getEma(26)
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd - macd_signal
    # ATR
    trueRange = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1) # greatest of 3 values
    df["atr"] = trueRange.rolling(14).mean() / df["close"]
    # Bollinger bands
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["bb_width"] = (bb_upper - bb_lower) / bb_mid
    df["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower)
    # volume ratio
    vol_sma30 = df["volume"].rolling(30).mean()
    df["vol_ratio"] = df["volume"] / vol_sma30
    # lagged returns and volumes
    for lag in range(1, 6):
        df[f"return_lag{lag}"] = df["return"].shift(lag)
        df[f"vol_ratio_lag{lag}"] = df["vol_ratio"].shift(lag)
    
    # drop empty rows and return
    df.dropna()
    return df