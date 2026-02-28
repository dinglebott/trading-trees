import xgboost as xgb
from custom_modules import datafetcher, dataparser

# GLOBAL VARIABLES
yearNow = 2026
instrument = "EUR_USD"
granularity = "H1"

# LOAD MODEL
model = xgb.XGBClassifier()
try:
    model.load_model(f"XGBoost_{instrument}_{granularity}_{yearNow}.json")
except (xgb.core.XGBoostError, FileNotFoundError):
    print("Error loading model. Model may not exist.")

# FETCH AND PARSE CURRENT DATA
jsonPath = datafetcher.getData(instrument, granularity, 100, "live")
df = dataparser.parseData(jsonPath)

# DEFINE FEATURES (copy-paste from the model training features exactly)
features = [

]

# GET PREDICTION
latestCandle = df[features].iloc[[-1]] # slice out last row (last candle)
prediction = model.predict(latestCandle)[0] # gets the only element of the 1D numpy array
probabilities = model.predict_proba(latestCandle)[0] # gets the only row of the 2D numpy array [P(0), P(1)]

# DISPLAY RESULTS
print("Prediction:", "UP" if prediction == 1 else "DOWN")
print(f"Confidence: {max(probabilities)*100:.2f}%")