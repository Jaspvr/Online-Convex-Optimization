# Online Newton Step (Algorithm 12 in Introduction to Online Convex Optimization - Elad Hazan)
#  for portfolio selection
# Data source: Stooq via pandas_datareader

import numpy as np
import matplotlib.pyplot as plt

from data_handling.data_handler import *
from best_stock import bestInHindsight
from data.tickers import *
from optimal_crp import optimalCrpWeightsCvx
from uniform_crp import uniformCRP

class OnlinePortfolio:
    def __init__(self, data):
        self.data = data
        self.T, self.n = data.shape # number of trading periods and number of stocks
        self.c = 1 # smallest price relative
        self.C = 1 # largest price relative
        
        # Initialize weights as 1/n for each (x1 in K)
        # Using an n-dimensional simplex K as the convex set
        self.weights = np.ones(self.n) / self.n

    def computeEta(self, xt):
        # eta = (c / C) * root(8 * ln(num stocks) / (num trading days)) via PLG
        self.c = min(xt)
        self.C = max(xt)
        return (self.c / self.C) * np.sqrt(8 * np.log(self.n) / self.T)

    def loss(self, xt, t):
        ''' Log loss function '''
        rt = self.data[t] # price relatives that actually happened

        # Take the negative log of the growth factor (weights * outcome)
        # Summing losses gives -log(XT/X0), therefore minimizing this loss is maximizing wealth
        # (The price relatives ratio XT/X0 indicates how much we have increased our wealth)
        xpMul = max(float(xt @ rt), 1e-10)
        return -np.log(xpMul)

    def gradient(self, xt, t):
        ''' Function to take the gradient of the loss with respect to the "decision" 
        of the weights for round t '''
        rt = self.data[t]

        # (dloss/dweight) = (d/dweight)(-log(weight @ outcome))
        xpMul = max(float(xt @ rt), 1e-10)
        return -rt / xpMul

    def eg(self):
        ''' Exponentiated Gradient algorithm '''
        xt = self.weights.copy()
        
        X = np.zeros((self.T, self.n)) # weights
        L = np.zeros((self.T)) # losses

        for t in range(self.T):
            X[t] = xt
            L[t] = self.loss(xt, t)
            gradt = self.gradient(xt, t)

            eta = self.computeEta(self.data[t])

            # Use the previous time step's xt (xt not updated yet here)
            yt = xt * np.exp(eta * gradt)

            # Normalize back to simplex by dividing by sum from PLG
            xt = yt / sum(yt)

        # Multiply decisions (X) by the actual price relative outcomes to get the 
        # growth of the portfolio in each stock ticker based on the decision made.
        # "growth" is a vector of length T that holds how much the portfolio grew each day.
        growth = (X * self.data).sum(axis=1)
        wealth = growth.cumprod()
        return X, wealth, L

def main():
    # Use ETF data from Stooq
    TICKERS = TICKERS_SP20
    START = "2020-01-01"
    END = "2025-11-01"  # Until current date

    # prices = downloadPricesStooq(TICKERS, start=START, end=END, min_days=500)
    # cache_file = "data/sp20Group_2015-11-01_2025-11-01.csv"
    # cache_file = "data/sp20_2015-11-01_2025-11-01.csv" # GS instead of NVIDIA
    cache_file = "data/sp20new_2015-11-01_2025-11-01.csv" # with nvidia
    prices = loadOrDownloadPrices(TICKERS, start=START, end=END,
                                 min_days=500, cache_path=cache_file)
    # print(prices)

    # Form the table into T x n time series data
    # Get the ratios of the prices compared to the previous day: xt = Pt / Pt-1
    relativePrices = (prices / prices.shift(1)).dropna().to_numpy()
    dates = prices.index[1:] # Shift dates to match the relative price data (will use for plotting)

    portfolio = OnlinePortfolio(relativePrices)
    X, wealth, loss = portfolio.eg()

    # For comparison
    wealthUniformCRP = uniformCRP(relativePrices)
    _, wealthOptimalCRP = optimalCrpWeightsCvx(relativePrices)
    wealthBestStock = bestInHindsight(relativePrices)

    print("Weight distributions: ", X)
    print("Losses: ", loss)
    print("Final wealth (EG): ", wealth[-1])

    # Plot the log wealth growth over time. Use log wealth since it matches with the loss
    plt.figure()
    plt.plot(dates, np.log(wealth), label="EG (log-wealth)")
    plt.plot(dates, np.log(wealthBestStock),
             label=f"Best single stock")
    plt.title("Exponentiated Gradient - Portfolio Log Wealth")
    plt.xlabel("date")
    plt.ylabel("log wealth")
    plt.legend()
    plt.tight_layout()
    plt.show()

    
if __name__ == "__main__":
    main()