from custom_modules import datafetcher
from datetime import datetime

datafetcher.getDataLoop(datetime(2005, 1, 1), datetime(2020, 1, 1), "EUR_USD", "H4") # 15yrs
datafetcher.getDataLoop(datetime(2020, 1, 1), datetime(2023, 1, 1), "EUR_USD", "H4") # 3yrs
datafetcher.getDataLoop(datetime(2023, 1, 1), datetime(2026, 1, 1), "EUR_USD", "H4") # 3yrs

datafetcher.getDataLoop(datetime(2005, 1, 1), datetime(2020, 1, 1), "EUR_USD", "H1")
datafetcher.getDataLoop(datetime(2020, 1, 1), datetime(2023, 1, 1), "EUR_USD", "H1")
datafetcher.getDataLoop(datetime(2023, 1, 1), datetime(2026, 1, 1), "EUR_USD", "H1")