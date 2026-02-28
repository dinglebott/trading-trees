from custom_modules import featurepicker
import pandas as pd
import os

# PHASE 2: SELECT FEATURES
yearNow = 2026
instrument = "EUR_USD"
granularity = "H1"

# uses all features and default hyperparameters
importances = featurepicker.evaluateFeatures(yearNow, instrument, granularity)
print(importances)

# save results to json
directory = "results"
if not os.path.exists(directory):
    os.makedirs(directory)
filename = "feature_selection.json"
filepath = os.path.join(directory, filename)
importances.to_json(filepath, indent=4)