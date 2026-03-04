from custom_modules import paramtuner
import os
import json

# PHASE 3: TUNE HYPERPARAMETERS
yearNow = 2026
instrument = "EUR_USD"
granularity = "H4"
candlesAhead = 7 # model predicts net return of the next n candles
deadzone = 0.0025 # defines half the width of the "flat" class (distance from midThreshold in either direction)
midThreshold = 0 # defines midpoint from which to split "up" and "down" classes

# select features from Phase 2
directory = "results"
filename = "features.json"
filepath = os.path.join(directory, filename)
# deserialise json data
with open(filepath, "r") as file:
    rawFeatures = json.load(file) # rawFeatures is a Python dict
# extract top 11 features into list
bestFeatures = list(rawFeatures.keys())[:11]
print("Best features:", bestFeatures)

# pass to hyperparameter tuner
finalParams = paramtuner.tuneHyperparams(yearNow, instrument, granularity, bestFeatures, n=candlesAhead, deadzone=deadzone, midThreshold=midThreshold)
print("\nFinal hyperparameters:")
print(finalParams) # pd Series

# save results to json
directory = "results"
if not os.path.exists(directory):
    os.makedirs(directory)
filename = "hyperparameters.json"
filepath = os.path.join(directory, filename)
finalParams.to_json(filepath, indent=4)