from custom_modules import featurepicker

# PHASE 2: SELECT FEATURES (all features, default hyperparameters)
yearNow = 2026

importances = featurepicker.evaluateFeatures(yearNow, "EUR_USD", "H1")
print(importances)