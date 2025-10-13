# Online Gradient Descent (Algorithm 8 in Introduction to Online Convex Optimization - Elad Hazan)
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
        pt = self.data[t]
        # We are taking the gradient of the loss with respect to the "decision" of the 
        # weights for round t
        # (dloss/dweight) = (d/dweight)(-log(weight @ outcome))
        # Using chain rule, we get the result gradient: -weightt / (weightt @ outcomet)
        xpMul = max(float(xt @ pt), 1e-10)
        return -xt / xpMul
    
    def projectToK(self, y):
        return []

    def odg(self, eta):
        xt = self.weights.copy() # Initial weight spread is uniform distribution

        X = np.zeros((self.T, self.n)) # Decisions (weight spreads)
        G = np.zeros((self.T, self.n)) # Gradients
        L = np.zeros(self.T) # Loss vector (1 entry per round)

        # Go through each tiem step and update the weight distribution according to OGD
        for t in range(self.T): 
            # "Play" xt (observe loss - ft(xt) in textbook. Line 3 of Algorithm 8)
            X[t] = xt
            L[t] = self.loss(xt, t)

            # Line 4 in algorithm 8
            G[t] = self.gradient(xt, t)
            yNext = X[t] - eta * G[t]
            xt = self.projectToK(yNext)
        
        return X, G, L


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

    eta = 0.25
    portfolio = OnlinePortfolio(relativePrices)
    result = portfolio.odg(eta)
    print(result)



if __name__ == "__main__":
    main()