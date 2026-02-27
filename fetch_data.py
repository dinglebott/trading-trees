from custom_modules import datafetcher
from datetime import datetime

# PHASE 1: FETCH HISTORICAL DATA
yearNow = 2026

# 10 folds
for index, year in enumerate(range(yearNow - 16, yearNow - 6), start=1):
    datafetcher.getDataLoop(datetime(year, 1, 1), datetime(year + 6, 1, 1), "EUR_USD", "H1", f"fold_{index}") # train (6yrs)
    datafetcher.getDataLoop(datetime(year + 6, 1, 1), datetime(year + 7, 1, 1), "EUR_USD", "H1", f"fold_{index}") # test (1yr)
# fetch total data
datafetcher.getDataLoop(datetime(yearNow - 16, 1, 1), datetime(yearNow - 1, 1, 1), "EUR_USD", "H1", "all")
datafetcher.getDataLoop(datetime(yearNow - 1, 1, 1), datetime(yearNow, 1, 1), "EUR_USD", "H1", "all")