import xgboost as xgb
from custom_modules import datafetcher, dataparser
import os

# GLOBAL VARIABLES
yearNow = 2026
instrument = "EUR_USD"
granularity = "H1"
version = 0

# DEFINE FEATURES (copy-paste from the model training features exactly)
features = [

]

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
prediction = model.predict(latestCandle)[0] # gets the only element of the 1D numpy array
probabilities = model.predict_proba(latestCandle)[0] # gets the only row of the 2D numpy array [P(0), P(1)]

# DISPLAY RESULTS
print("Prediction:", "UP" if prediction == 1 else "DOWN")
print(f"Confidence: {max(probabilities)*100:.2f}%")