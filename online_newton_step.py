# Online Newton Step (Algorithm 12 in Introduction to Online Convex Optimization - Elad Hazan)
#  for portfolio selection
# Data source: Stooq via pandas_datareader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr

# Data handling functions
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


class OnlinePortfolio:
    def __init__(self, data):
        self.data = data
        self.T, self.n = data.shape
        
        # Initialize weights as 1/n for each (x1 in K)
        # Using an n-dimensional simplex K as the convex set
        self.weights = np.ones(self.n) / self.n

    def loss(self, xt, t):
        ''' Log loss function '''
        pt = self.data[t] # price relatives that actually happened

        # Take the negative log of the growth factor (weights * outcome)
        # Summing losses gives -log(XT/X0), therefore minimizing this loss is maximizing wealth
        # (The price relatives ratio XT/X0 indicates how much we have increased our wealth)
        xpMul = max(float(xt @ pt), 1e-10)
        return -np.log(xpMul)

    def gradient(self, xt, t):
        return []

    def projectToK(self, y):
        return []

    def ons(self, gamma, epsilon):
        ''' Online Newton Step function '''
        xt = self.weights.copy()

        X = np.zeros((self.T, self.n)) # weights
        L = np.zeros((self.T)) # losses
        G = np.zeros((self.T, self.n)) # gradients

        for t in range(self.T):
            # "Play" xt and observe cost (line 3 in Algorithm 12)
            X[t] = xt
            L[t] = self.loss(xt, t)
            G[t] = self.gradient(xt, t)

            # Rank-1 update (line 4 in Algorithm 12)
            A  = []

            # Newton step
            # np.linalg.solve(a, b) computes x = a^(-1)b from ax = b
            invAg = np.linalg.solve(A, G[t])
            yt = xt - (1.0 / gamma) * invAg

            # Generalized projection
            xt = self.projectToK(yt) # xt updated for next iteration

        return [], [], []
        


def main():
    # Use ETF data from Stooq
    TICKERS = ["SPY", "QQQ", "DIA", "IWM", "EFA", "EEM"]
    START = "2020-01-01"
    END = None  # Until current date

    prices = downloadPricesStooq(TICKERS, start=START, end=END, min_days=500)
    print(prices)

    # Form the table into T x n time series data
    # Get the ratios of the prices compared to the previous day: xt = Pt / Pt-1
    relativePrices = (prices / prices.shift(1)).dropna().to_numpy()
    dates = prices.index[1:] # Shift dates to match the relative price data (will use for plotting)
    T, n = relativePrices.shape # T is trading days, n is number of assets

    gamma = 0.5
    epsilon = 1e-2
    portfolio = OnlinePortfolio(relativePrices)
    X, wealth, loss = portfolio.ons(gamma, epsilon)

    
if __name__ == "__main__":
    main()