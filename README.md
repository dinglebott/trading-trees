## About project
Goal: Predict the direction of future price movements in forex markets by training a machine-learning model\
See DOCS.md for details on data collection, tuning, training, and testing methodology\
Fully-trained models located in models folder, tuned with results obtained by me

## Outline of methodology
Phase 1: Fetch historical data => fetch_data.py\
Phase 2: Feature engineering and selection => select_features.py\
Phase 3: Tune hyperparameters (using features from Phase 2) => tune_params.py\
Phase 4: Train final model and evaluate (using features and hyperparameters from Phases 2 and 3) => train_model.py

## Project structure
The custom_modules folder contains most of the actual code as reusable functions\
The modules are then called by the top-level scripts to execute the code

## How to replicate
Each of the top-level scripts (fetch_data, select_features, tune_params, train_model) contain 4 global variables (yearNow, instrument, granularity, candlesAhead)\
*candlesAhead determines the number of candles ahead for which the model predicts the net movement*\
Change these accordingly if you want to build your own model using this framework\
Note the features you want to keep after Phase 2, and pass them in the features argument for Phase 3\
In Phase 4, set the final features and hyperparameters in the features and params variables respectively\
Output for all phases is printed to the terminal, and the final model is automatically saved as a JSON file
#### IMPORTANT:
You need an OANDA API key to pull historical data (or you can use the data I pulled already)\
If you have a key, set it as an environment variable API_KEY in a local .env file\
The code fetches from the api-fxtrade.oanda.com live server, so if your key is from a demo account, change this to api-fxpractice.oanda.com

## How to use
Open use_model.py and copy-paste the exact same features used in train_model.py into the features variable\
Prediction and confidence is printed to the terminal\
Remember to set the correct global variables (yearNow, instrument, granularity, candlesAhead)\
Also set the version number of the model you want (in the version variable)\
Obviously make sure you have the correct model trained for your use case first

## Why???
ongod why did i put so much effort into this