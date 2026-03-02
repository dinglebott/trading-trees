import xgboost as xgb
from custom_modules import datafetcher, dataparser
import os
import json

# GLOBAL VARIABLES
yearNow = 2026
instrument = "EUR_USD"
granularity = "H4"
version = 4

# DEFINE FEATURES (copy-paste from the model training features exactly)
directory = "results"
filename = f"features_v{version}.json"
filepath = os.path.join(directory, filename)
# deserialise json data
with open(filepath, "r") as file:
    rawFeatures = json.load(file) # rawFeatures is a Python dict
# extract top 11 features into list
features = list(rawFeatures.keys())[:11]

# LOAD MODEL
model = xgb.XGBClassifier()
try:
    directory = "models"
    filename = f"XGBoost_{instrument}_{granularity}_{yearNow}_v{version}.json"
    filepath = os.path.join(directory, filename)
    model.load_model(filepath)
except (xgb.core.XGBoostError, FileNotFoundError):
    print("Error loading model. Model may not exist.")

# FETCH AND PARSE CURRENT DATA
jsonPath = datafetcher.getData(instrument, granularity, 100, "live")
df = dataparser.parseData(jsonPath)

# GET PREDICTION
latestCandle = df[features].iloc[[-1]] # slice out last row (last candle)
prediction = model.predict(latestCandle)[0] # gets the only element of the 1D numpy array [n_samples]
probabilities = model.predict_proba(latestCandle)[0] # gets the only row of the 2D numpy array [n_samples, n_classes]

# DISPLAY RESULTS
match prediction:
    case 0:
        predictionLabel = "DOWN"
    case 1:
        predictionLabel = "FLAT"
    case 2:
        predictionLabel = "UP"
print(f"Prediction: {predictionLabel}")
print(f"Confidence: {probabilities.max()*100:.2f}%")