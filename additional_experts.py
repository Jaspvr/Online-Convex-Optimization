import numpy as np
import matplotlib.pyplot as plt

from data_handling.data_handler import downloadPricesStooq
from best_stock import bestInHindsight
from projections import projectToK, cvxpyOgdProjectToK
from data.tickers import *
from additional_experts_helpers import *

class OnlinePortfolioOGD:
    def __init__(self, priceRelatives, numBundles, groups):
        self.priceRelatives = priceRelatives
        self.T, self.n = priceRelatives.shape
        self.numBundles = numBundles
        self.groups = groups
        
        # Initialize weights as 1/n for each (x1 in K)
        # Using an n-dimensional simplex K as the convex set
        self.weights = np.ones(self.n) / self.n
        self.weightsBundles = np.ones(self.n + self.numBundles) / (self.n + self.numBundles)

        self.eta  = np.zeros(self.T+1) # one eta per round

        self.G = 0 # factor used to calculate eta
        self.D = 2**(0.5) # factor used to calculate eta

    def computeEta(self, grad, t):
        gradMag = np.linalg.norm(grad)
        if gradMag > self.G:
            self.G = gradMag

        self.eta[t] = self.D / (self.G * ((t+1)**0.5)) # t+1 to make rounds 1-indexed
        print(self.eta[t])

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

    def odg(self):
        xt = self.weights.copy() # Initial weight spread is uniform distribution
        bundleXt = self.weightsBundles.copy()

        X = np.zeros((self.T, self.n)) # Decisions (weight spreads)
        bundlesX = np.zeros((self.T, self.n + self.numBundles))
        Grad = np.zeros((self.T, self.n)) # Gradients
        L = np.zeros(self.T) # Loss vector (1 entry per round)

        # Go through each time step and update the weight distribution according to OGD
        for t in range(self.T): 
            # "Play" xt (observe loss - ft(xt) in textbook. Line 3 of Algorithm 8)
            X[t] = xt
            bundlesX[t] = bundleXt
            L[t] = self.loss(xt, t)

            # Line 4 in algorithm 8
            Grad[t] = self.gradient(bundleXt, t)
            self.computeEta(Grad[t], t)

            # Get xt for next round
            yNext = bundlesX[t] - self.eta[t] * Grad[t]
            bundleXt = cvxpyOgdProjectToK(yNext)

            xt = eliminateBundles(bundleXt, self.groups, self.n)

        # Multiply decisions (X) by the actual price relative outcomes to get the 
        # growth of the portfolio in each stock ticker based on the decision made.
        # "growth" is a vector of length T that holds how much the portfolio grew each day.
        growth = (X * self.priceRelatives).sum(axis=1)
        wealth = growth.cumprod()
        return X, wealth, L


def main():
    # Ticker order: Tech (4), Health Care (3), Financials (4), Consumer Discretionary (3),
    # Industrials (3), Energy (3)
    TICKERS = TICKERS_GROUP_SP
    START = "2015-11-01"
    END = "2025-11-01"
    groups = [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9, 10], [11, 12, 13], [14, 15, 16], [17, 18, 19]]

    prices = downloadPricesStooq(TICKERS, start=START, end=END, min_days=500)
    print(prices)
    prices = bundles(prices, groups)

    relativePrices = (prices / prices.shift(1)).dropna().to_numpy()
    dates = prices.index[1:]

    numBundles = 6
    portfolio = OnlinePortfolioOGD(relativePrices, numBundles, groups)
    X, wealth, loss = portfolio.odg()

    # For comparison
    wealthBestStock = bestInHindsight(relativePrices)

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