import datafetcher
from datetime import datetime
import parser

datafetcher.getDataLoop(datetime(2005, 1, 1), datetime(2019, 12, 31), "EUR_USD", "H4") # 15yrs
datafetcher.getDataLoop(datetime(2020, 1, 1), datetime(2022, 12, 31), "EUR_USD", "H4") # 3yrs
datafetcher.getDataLoop(datetime(2023, 1, 1), datetime(2025, 12, 31), "EUR_USD", "H4") # 3yrs

datafetcher.getDataLoop(datetime(2005, 1, 1), datetime(2019, 12, 31), "EUR_USD", "H1")
datafetcher.getDataLoop(datetime(2020, 1, 1), datetime(2022, 12, 31), "EUR_USD", "H1")
datafetcher.getDataLoop(datetime(2023, 1, 1), datetime(2025, 12, 31), "EUR_USD", "H1")