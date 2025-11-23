import pandas as pd
from pandas_datareader import data as pdr
import os

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

def fetchStooqCloseErrorHandling(ticker, start=None, end=None):
    try:
        print(f"Downloading {ticker} from Stooq...")
        df = pdr.DataReader(
            ticker,
            "stooq",
            start=pd.to_datetime(start) if start else None,
            end=pd.to_datetime(end) if end else None,
        )
        print(f"  -> got shape {df.shape} for {ticker}")
    except Exception as e:
        print(f"ERROR downloading {ticker}: {e}")
        return None

    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        print(f"  -> empty DataFrame for {ticker}")
        return None

    if "Close" not in df.columns:
        print(f"  -> 'Close' column missing for {ticker}; columns={df.columns}")
        return None

    s = df["Close"].rename(ticker)
    if s.notna().sum() == 0:
        print(f"  -> 'Close' is all NaN for {ticker}")
        return None

    return s


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


def loadOrDownloadPrices(tickers, start=None, end=None, min_days=500, cache_path=None):
    """Load prices from a local cache if available; otherwise download from Stooq
    and save to cache_path."""
    if cache_path is None:
        tickers_str = "_".join(tickers)
        cache_path = f"data/stooq_{tickers_str}_{start}_{end}.csv"

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path):
        print(f"Loading cached prices from {cache_path}")
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        print("Cache not found, downloading from Stooq...")
        prices = downloadPricesStooq(tickers, start=start, end=end, min_days=min_days)
        prices.to_csv(cache_path)
        print(f"Saved prices to {cache_path}")

    return prices
