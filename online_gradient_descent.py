# Online Gradient Descent (Algorithm 8 in Introduction to Online Convex Optimization - Elad Hazan)
#  for portfolio selection
# Data source: Stooq via pandas_datareader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr

def fetchStooqClose(ticker, start=None, end=None):
    # Get Close prices for a single ticker from Stooq
    df = pdr.DataReader(ticker, "stooq", start=pd.to_datetime(start) if start else None,
                        end=pd.to_datetime(end) if end else None)
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return None
    df = df.sort_index()
    if "Close" not in df.columns:
        return None
    s = df["Close"].rename(ticker)

    return s if s.notna().sum() > 0 else None

def downloadPricesStooq(tickers, start=None, end=None, min_days=500):
    series = []
    for t in tickers:
        s = fetchStooqClose(t, start=start, end=end)
        if s is not None:
            series.append(s)

    if not series:
        raise RuntimeError("All Stooq downloads failed")

    # Make price table, remove tickers with small number of entries, and rows with NaNs
    prices = pd.concat(series, axis=1)
    prices = prices.dropna(axis=1, thresh=min_days)
    prices = prices.dropna()

    if prices.shape[1] < 2 or len(prices) < 2:
        raise RuntimeError("Not enough columns or rows in price data after cleaning")
    
    return prices

def main():
    # Use ETF data from Stooq
    TICKERS = ["SPY", "QQQ", "DIA", "IWM", "EFA", "EEM"]
    START = "2020-01-01"
    END = None  # Until current date

    prices = downloadPricesStooq(TICKERS, start=START, end=END, min_days=500)
    print(prices)

if __name__ == "__main__":
    main()