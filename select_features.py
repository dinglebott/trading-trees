from custom_modules import featurepicker
import os

# PHASE 2: SELECT FEATURES
yearNow = 2026
instrument = "EUR_USD"
granularity = "H4"
candlesAhead = 4 # model predicts net return of the next n candles
deadzone = 0.0015 # defines half the width of the "flat" class (distance from midThreshold in either direction)
midThreshold = 0 # defines midpoint from which to split "up" and "down" classes

# uses all features and default hyperparameters
shaps = featurepicker.evaluateFeatures(yearNow, instrument, granularity, n=candlesAhead, deadzone=deadzone, midThreshold=midThreshold)
print(f"\n{shaps}")

# save results to json
directory = "results"
if not os.path.exists(directory):
    os.makedirs(directory)
filename = "features.json"
filepath = os.path.join(directory, filename)
shaps.to_json(filepath, indent=4)