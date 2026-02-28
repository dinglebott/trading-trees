from custom_modules import featurepicker

# PHASE 2: SELECT FEATURES
yearNow = 2026
instrument = "EUR_USD"
granularity = "H1"

# uses all features and default hyperparameters
importances = featurepicker.evaluateFeatures(yearNow, instrument, granularity)
print(importances)