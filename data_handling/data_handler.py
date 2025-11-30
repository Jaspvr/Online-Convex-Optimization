import pandas as pd
from pandas_datareader import data as pdr
import os

# def fetchStooqClose(ticker, start=None, end=None):
#     # Get Close prices for a single ticker from Stooq
#     df = pdr.DataReader(ticker, "stooq", start=pd.to_datetime(start) if start else None,
#                         end=pd.to_datetime(end) if end else None)
#     if not isinstance(df, pd.DataFrame) or len(df) == 0:
#         return None
#     df = df.sort_index()
#     if "Close" not in df.columns:
#         return None
#     s = df["Close"].rename(ticker)

#     return s if s.notna().sum() > 0 else None

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


def fetchStooqClose(ticker, start=None, end=None):
    # Get Close prices for a single ticker from Stooq
    df = pdr.DataReader(
        ticker, "stooq",
        start=pd.to_datetime(start) if start else None,
        end=pd.to_datetime(end) if end else None,
    )
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        print(f"[fetchStooqClose] No DataFrame or empty for {ticker}")
        return None

    df = df.sort_index()
    if "Close" not in df.columns:
        print(f"[fetchStooqClose] No 'Close' column for {ticker}")
        return None

    s = df["Close"].rename(ticker)
    if s.notna().sum() == 0:
        print(f"[fetchStooqClose] All NaNs for {ticker}")
        return None

    return s

def downloadPricesStooq_debug(tickers, start=None, end=None, min_days=500):
    """
    Debug version of downloadPricesStooq:
    - Same signature and return type.
    - Prints what gets dropped at each cleaning step.
    - Additionally: keeps ONLY columns whose non-NaN count == 1759.
    """
    series = []
    print("=== Fetching individual series ===")
    for t in tickers:
        s = fetchStooqClose(t, start=start, end=end)
        if s is None:
            print(f"Ticker {t}: FAILED (no usable data)")
            continue

        non_na = s.notna().sum()
        print(
            f"Ticker {t}: non-NaN count = {non_na}, "
            f"range = [{s.index.min().date()} .. {s.index.max().date()}]"
        )
        series.append(s)

    if not series:
        raise RuntimeError("All Stooq downloads failed")

    # Step 1: raw outer-joined table (no dropping yet)
    raw = pd.concat(series, axis=1, join="outer").sort_index()
    print("\n=== After outer concat (raw) ===")
    print(f"Shape: {raw.shape}")
    print(f"Date range: [{raw.index.min().date()} .. {raw.index.max().date()}]")
    print("NaN counts per column (raw):")
    print(raw.isna().sum())

    # Step 2: keep ONLY columns with exactly 1759 non-NaN entries
    non_na_counts = raw.notna().sum()
    keep_cols = non_na_counts[non_na_counts == 1759].index.tolist()
    dropped_cols = sorted(set(raw.columns) - set(keep_cols))

    prices_cols_filtered = raw[keep_cols]

    print("\n=== After filtering for non-NaN count == 1759 ===")
    print(f"Shape: {prices_cols_filtered.shape}")
    if dropped_cols:
        print("Dropped tickers (non-NaN count != 1759):", dropped_cols)
    else:
        print("No tickers dropped at 1759-count filter.")
    print("Non-NaN counts of kept columns:")
    print(prices_cols_filtered.notna().sum())

    # Step 3: row dropna (this is where early years often get chopped)
    prices_final = prices_cols_filtered.dropna()
    print("\n=== After row dropna() (final) ===")
    print(f"Shape: {prices_final.shape}")
    print(f"Date range: [{prices_final.index.min().date()} .. {prices_final.index.max().date()}]")

    rows_before = len(prices_cols_filtered)
    rows_after = len(prices_final)
    print(f"Rows removed by dropna(): {rows_before - rows_after}")

    print("\nFirst 10 dates (after 1759-count filter, before row dropna):")
    print(prices_cols_filtered.head(10))

    print("\nFirst 10 dates (after row dropna):")
    print(prices_final.head(10))

    if prices_final.shape[1] < 2 or len(prices_final) < 2:
        raise RuntimeError("Not enough columns or rows in price data after cleaning")

    return prices_final


def loadOrDownloadPrices_debug(tickers, start=None, end=None, min_days=500, cache_path=None):
    """
    Debug version of loadOrDownloadPrices:
    - Same parameters and cache behavior.
    - Uses downloadPricesStooq_debug when cache is missing so you can
      see exactly why data is dropped / start date shifts.
    """
    if cache_path is None:
        tickers_str = "_".join(tickers)
        cache_path = f"data/stooq_{tickers_str}_{start}_{end}.csv"

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path):
        print(f"Loading cached prices from {cache_path}")
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        print("Cache not found, downloading from Stooq (DEBUG mode)...")
        prices = downloadPricesStooq_debug(tickers, start=start, end=end, min_days=min_days)
        prices.to_csv(cache_path)
        print(f"Saved prices to {cache_path}")

    return prices