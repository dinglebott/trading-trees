from custom_modules import paramtuner

# PHASE 3: TUNE HYPERPARAMETERS
yearNow = 2026

# use pruned features as found in Phase 2
allResults, finalParams = paramtuner.tuneHyperparams(yearNow, "EUR_USD", "H1", prunedFeatures=[])
print("All results:")
print(allResults)
print("\nFinal hyperparameters:")
print(finalParams)