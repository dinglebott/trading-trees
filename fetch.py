from custom_modules import datafetcher
from datetime import datetime

# 10 folds
for x in range(2010, 2020):
    datafetcher.getDataLoop(datetime(x, 1, 1), datetime(x + 6, 1, 1), "EUR_USD", "H1", f"fold_{x - 2009}") # train (6yrs)
    datafetcher.getDataLoop(datetime(x + 6, 1, 1), datetime(x + 7, 1, 1), "EUR_USD", "H1", f"fold_{x - 2009}") # test (1yr)