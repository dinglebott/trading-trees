from custom_modules import paramtuner
import os
import json

# PHASE 3: TUNE HYPERPARAMETERS
yearNow = 2026
instrument = "EUR_USD"
granularity = "H4"
candlesAhead = 4
threshold = 0.001

# select features from Phase 2
directory = "results"
filename = "feature_selection.json"
filepath = os.path.join(directory, filename)
# deserialise json data
with open(filepath, "r") as file:
    rawFeatures = json.load(file) # rawFeatures is a Python dict
# extract top 11 features into list
bestFeatures = list(rawFeatures.keys())[:11]
print("Best features:", bestFeatures)

# pass to hyperparameter tuner
allResults, finalParams = paramtuner.tuneHyperparams(yearNow, instrument, granularity, bestFeatures, n=candlesAhead, threshold=threshold)
print("\nAll results:")
print(allResults)
print("\nFinal hyperparameters:")
print(finalParams)

# save results to json
directory = "results"
if not os.path.exists(directory):
    os.makedirs(directory)
filename = "hyperparameter_tuning.json"
filepath = os.path.join(directory, filename)
finalParams.to_json(filepath, indent=4)