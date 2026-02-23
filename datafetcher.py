import os
from dotenv import load_dotenv
import requests
import json
from datetime import datetime
import pandas as pd

# get oanda api key (from .env file)
load_dotenv()
apiKey = os.getenv("API_KEY")

# FETCHING DATA (also saves locally as JSON file)
def getData(instr="EUR_USD", gran="H1", count=500):
    # granularity values: S5 S10 S15 S30, M1 M2 M4 M5 M10 M15 M30, H1 H2 H3 H4 H6 H8 H12, D W M
    # max count: 5000

    # get response
    headers = {"Authorization": f"Bearer {apiKey}"}
    params = {
        "granularity": gran,
        "count": count,
        "price": "M"
    }
    baseUrl = "https://api-fxtrade.oanda.com" # access token generated from live account (don't use fxpractice)
    endpoint = f"/v3/instruments/{instr}/candles"
    response = requests.get(baseUrl + endpoint, headers=headers, params=params)
    
    # inspect response
    if response.status_code != 200:
        raise Exception(response.text)
    else:
        print("Successful fetch")
    
    # save response
    data = response.json()
    directory = f"json_data/{instr}/{gran}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    timestamp = datetime.now().strftime("%m%d_%H%M") # extract month date hour minute from datetime
    filename = f"{instr}_{gran}_{count}_{timestamp}.json"
    filepath = os.path.join(directory, filename)
    with open(filepath, "w") as file:
        json.dump(data, file, indent=4)
    
    # expose file path
    return filepath

# PARSING DATA
def parseData(jsonPath):
    # deserialise json data
    with open(jsonPath, "r") as file:
        rawData = json.load(file) # rawData is a Python dict
    
    # unpack dict
    records = []
    for c in rawData["candles"]:
        if c["complete"]:
            records.append({
                "time": c["time"],
                "open": float(c["mid"]["o"]), # convert from string
                "high": float(c["mid"]["h"]),
                "low": float(c["mid"]["l"]),
                "close": float(c["mid"]["c"]),
                "volume": c["volume"]
            })
    df = pd.DataFrame(records)