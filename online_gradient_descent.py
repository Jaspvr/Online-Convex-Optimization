# Online Gradient Descent (Algorithm 8 in Introduction to Online Convex Optimization - Elad Hazan)
#  for portfolio selection
# Data source: Stooq via pandas_datareader

import numpy as np
import matplotlib.pyplot as plt

from data_handler import downloadPricesStooq

class OnlinePortfolio:
    def __init__(self, priceRelatives):
        self.priceRelatives = priceRelatives
        self.T, self.n = priceRelatives.shape
        
        # Initialize weights as 1/n for each (x1 in K)
        # Using an n-dimensional simplex K as the convex set
        self.weights = np.ones(self.n) / self.n

        self.eta  = np.zeros(self.T) # one eta per round
        self.G = 0 # factor used to calculate eta
        self.D = 2**(0.5) # factor used to calculate eta

    def computeEta(self, grad, t):
        gradMag = np.linalg.norm(grad)
        if gradMag > self.G:
            self.G = gradMag

        self.eta[t] = self.D / (self.G * ((t+1)**0.5)) # t+1 to make rounds 1-indexed

    def loss(self, xt, t):
        ''' Log loss function '''
        rt = self.priceRelatives[t] # price relatives that actually happened

        # Take the negative log of the growth factor (weights * outcome)
        # Summing losses gives -log(XT/X0), therefore minimizing this loss is maximizing wealth
        # (The price relatives ratio XT/X0 indicates how much we have increased our wealth)
        xpMul = max(float(xt @ rt), 1e-10)
        return -np.log(xpMul)

    def gradient(self, xt, t):
        ''' Function to take the gradient of the loss with respect to the "decision" 
        of the weights for round t '''
        rt = self.priceRelatives[t]

        # (dloss/dweight) = (d/dweight)(-log(weight @ outcome))
        # Using chain rule, we get the result gradient: -outcomet / (weightt @ outcomet)
        xpMul = max(float(xt @ rt), 1e-10)
        return -rt / xpMul

    def projectToK(self, y):
        ''' K is defined as an n-dimensional simplex with the rule of x >= 0 and sum of x = 1. 
        This function takes a weight decision vector and projects it to land in K if it does not already.
        A Euclidean projection can be used to ensure the new point is as close to the original as possible
        while still being in K. '''

        # xtNew = max(xt - lambda)
        u = np.sort(y)[::-1] # Sort with largest first
        cumsum = np.cumsum(u) # cumulative sum array

        # top k entries are positive and get shifted by a constant, rest become 0.
        positivity = u * np.arange(1, len(u)+1)
        rho = positivity > (cumsum - 1.0)
        positiveIndices = np.nonzero(rho)[0]
        lastPositiveIdx = positiveIndices[-1]

        shift = (cumsum[lastPositiveIdx] - 1.0) / (lastPositiveIdx + 1.0)
        x = np.maximum(y - shift, 0.0)

        # Ensure sum of weights is exactly 1
        s = x.sum()
        if s > 0:
            x *= 1.0 / s 

        return x

    def odg(self):
        xt = self.weights.copy() # Initial weight spread is uniform distribution

        X = np.zeros((self.T, self.n)) # Decisions (weight spreads)
        Grad = np.zeros((self.T, self.n)) # Gradients
        L = np.zeros(self.T) # Loss vector (1 entry per round)

        # Go through each time step and update the weight distribution according to OGD
        for t in range(self.T): 
            # "Play" xt (observe loss - ft(xt) in textbook. Line 3 of Algorithm 8)
            X[t] = xt
            L[t] = self.loss(xt, t)

            # Line 4 in algorithm 8
            Grad[t] = self.gradient(xt, t)
            self.computeEta(Grad[t], t)
            yNext = X[t] - self.eta[t] * Grad[t]
            xt = self.projectToK(yNext)

        # Multiply decisions (X) by the actual price relative outcomes to get the 
        # growth of the portfolio in each stock ticker based on the decision made.
        # "growth" is a vector of length T that holds how much the portfolio grew each day.
        growth = (X * self.priceRelatives).sum(axis=1)
        wealth = growth.cumprod()
        return X, wealth, L


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

    portfolio = OnlinePortfolio(relativePrices)
    X, wealth, loss = portfolio.odg()

    # Best stock in hindsight for comparison
    cumulativeWs = np.cumprod(relativePrices, axis=0)
    finalW = cumulativeWs[-1, :]
    bestIdx = int(np.argmax(finalW))
    wealthBestStock = cumulativeWs[:, bestIdx]

    print("Weight distributions: ", X) # possibly add simplyfied visualization
    print("Losses: ", loss)
    print("Final wealth (OGD): ", wealth[-1])

    # Plot the log wealth growth over time. Use log wealth since it matches with the loss
    plt.figure()
    plt.plot(dates, np.log(wealth), label="OGD (log-wealth)")
    plt.plot(dates, np.log(wealthBestStock),
             label=f"Best single stock")
    plt.title("Online Gradient Descent - Portfolio Log Wealth")
    plt.xlabel("date")
    plt.ylabel("log wealth")
    plt.legend()
    plt.tight_layout()
    plt.show()

    
if __name__ == "__main__":
    main()