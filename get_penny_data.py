from data_handling.data_handler import *
from data.tickers import *

TICKERS = TICKERS_PENNY
START = "2018-11-01"
END = "2025-10-31"

cache_file = "data/penny20_2015-11-01_2020-10-31.csv"
# prices = loadOrDownloadPrices(TICKERS, start=START, end=END,
#                                 min_days=500, cache_path=cache_file)
prices = loadOrDownloadPrices_debug(TICKERS, start=START, end=END,
                                min_days=500, cache_path=cache_file)

relativePrices = (prices / prices.shift(1)).dropna().to_numpy()
dates = prices.index[1:]
