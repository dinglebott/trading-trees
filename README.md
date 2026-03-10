## About project
Goal: Predict the direction of future price movements in forex markets by training a machine-learning model\
Model type: Tree-based model which uses gradient boosting (XGBoost)\
See DOCS.md for details on data collection, tuning, training, and testing methodology\
Fully-trained models located in models folder, tuned with results obtained by me\
*See forex-RNN for part 2 of the project*

## Outline of methodology
Phase 1: Fetch historical data in `fetch_data.py`\
Phase 2: Feature engineering and selection in `select_features.py`\
Phase 3: Tune hyperparameters (using features from Phase 2) in `tune_params.py`\
Phase 4: Train final model and evaluate (using features and hyperparameters from Phases 2 and 3) in `train_model.py`

## Project structure
The `custom_modules` folder contains most of the actual code as reusable functions. The modules are then called by the top-level scripts to execute the code

## How to build a model
The top-level scripts contain 6 global variables (`yearNow`, `instrument`, `granularity`, `candlesAhead`, `deadzone`, `midThreshold`).\
*See comments in the .py files for explanations of each variable*\
Change these accordingly if you want to build your own model using this framework.\
In Phase 4, double-check the final features and hyperparameters in their respective JSON files (`features.json` and `hyperparameters.json`).\
*Rename them with their version names once the model is trained to prevent future models overwriting them*\
Output for all phases is printed to the terminal, and the final model is automatically saved as a JSON file.
#### IMPORTANT:
You need an OANDA API key to pull historical data (or you can use the data I pulled already).\
If you have a key, set it as an environment variable `API_KEY` in a local `.env` file.\
(The code fetches from the api-fxtrade.oanda.com live server, so if your key is from a demo account, change this to api-fxpractice.oanda.com)
#### IMPORTANT:
The code tunes and trains the model using a CUDA-enabled GPU. If your device doesn't have this, change the "device" parameter from "cuda" to "cpu" in the relevant parts of `featurepicker.py` and `paramtuner.py`

## How to use a model
Run `use_model.py`\
*Don't use models before v3 as they are binary classifiers and the code doesn't work with them*\
Prediction and confidence are printed to the terminal.\
Remember to set the correct global variables, and also set the version number of the model you want (in the version variable).\
Obviously make sure you have the correct model trained for your use case first.

## Why???
ongod why did i put so much effort into this