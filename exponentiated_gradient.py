# Online Newton Step (Algorithm 12 in Introduction to Online Convex Optimization - Elad Hazan)
#  for portfolio selection
# Data source: Stooq via pandas_datareader

import numpy as np
import matplotlib.pyplot as plt

from data_handler import downloadPricesStooq
from cvxpy_projection import cvxpyProjectToK

class OnlinePortfolio:
    def __init__(self, data):
        self.data = data
        self.T, self.n = data.shape
        
        # Initialize weights as 1/n for each (x1 in K)
        # Using an n-dimensional simplex K as the convex set
        self.weights = np.ones(self.n) / self.n

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

        return []
        
        


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

    portfolio = OnlinePortfolio(relativePrices)
    X, wealth, loss = portfolio.eg()

    # Best stock in hindsight for comparison
    cumulativeWs = np.cumprod(relativePrices, axis=0)
    finalW = cumulativeWs[-1, :]
    bestIdx = int(np.argmax(finalW))
    wealthBestStock = cumulativeWs[:, bestIdx]

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