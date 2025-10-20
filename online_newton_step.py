# Online Newton Step (Algorithm 12 in Introduction to Online Convex Optimization - Elad Hazan)
#  for portfolio selection
# Data source: Stooq via pandas_datareader

import numpy as np
import matplotlib.pyplot as plt

from data_handler import downloadPricesStooq

class OnlinePortfolio:
    def __init__(self, data):
        self.data = data
        self.T, self.n = data.shape
        
        # Initialize weights as 1/n for each (x1 in K)
        # Using an n-dimensional simplex K as the convex set
        self.weights = np.ones(self.n) / self.n

        self.G = 0 # factor used to calculate gamma
        self.D = 2**(0.5) # factor used to calculate epsilon
        self.gamma = 0
        self.epsilon = 0

    def computeGammaEpsilon(self, grad, alpha):
        ''' Function to compute gamma and epsilon tuning parameters '''
        gradMag = np.linalg.norm(grad)
        if gradMag > self.G:
            self.G = gradMag

        self.gamma = 0.5 * min(1/(self.G * self.D), alpha)
        self.epsilon = 1 / ((self.G**2)*(self.D**2))

    def loss(self, xt, t):
        ''' Log loss function '''
        pt = self.data[t] # price relatives that actually happened

        # Take the negative log of the growth factor (weights * outcome)
        # Summing losses gives -log(XT/X0), therefore minimizing this loss is maximizing wealth
        # (The price relatives ratio XT/X0 indicates how much we have increased our wealth)
        xpMul = max(float(xt @ pt), 1e-10)
        return -np.log(xpMul)

    def gradient(self, xt, t):
        ''' Function to take the gradient of the loss with respect to the "decision" 
        of the weights for round t '''
        pt = self.data[t]

        # (dloss/dweight) = (d/dweight)(-log(weight @ outcome))
        xpMul = max(float(xt @ pt), 1e-10)
        return -pt / xpMul

    def projectEuclid(self, y):
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
    
    def projectToK(self, yt, At):
        ''' This function projects onto the simplex (sum of weights = 1, each weight >= 0).
        ONS accomplishes this by finding the closest feasible portfolio to y in the At induced
        norm ('At' takes into account 2nd order information). Essentially we project back 
        using curvature-aware distance.
        '''

        # To get this projection, we need to find x on the simplex that minimizes
        # transpose(x-y)*A*(x-y). The gradient of this wrt x is A(x-y). What we can
        # do is gradient descent on this minimization by using repeated euclidean
        # projections while reducing x by its gradient each time to approach the solution

        alpha = 1e-2 # This should be improved upon/researched further
        xt = self.projectEuclid(yt)
        for _ in range(50):
            gt = At @ (xt - yt)
            xt = self.projectEuclid(xt - alpha * gt)

        return xt

    def ons(self, alpha):
        ''' Online Newton Step function '''
        xt = self.weights.copy()
        
        X = np.zeros((self.T, self.n)) # weights
        L = np.zeros((self.T)) # losses
        Grad = np.zeros((self.T, self.n)) # gradients

        for t in range(self.T):
            # "Play" xt and observe cost (line 3 in Algorithm 12)
            X[t] = xt
            L[t] = self.loss(xt, t)
            gradt = self.gradient(xt, t)
            Grad[t] = gradt

            self.computeGammaEpsilon(gradt, alpha)

            # Rank-1 update (line 4 in Algorithm 12)
            if t == 0:
                # 'A' starts as an epsilon scaled Identity matrix
                At = self.epsilon * np.eye(self.n)
            At  = At + np.outer(gradt, gradt) # does gt @ gtTranspose

            # Newton step
            # np.linalg.solve(a, b) computes x = a^(-1)b from ax = b
            invAg = np.linalg.solve(At, gradt)
            yt = xt - (1.0 / self.gamma) * invAg

            # Generalized projection
            xt = self.projectToK(yt, At) # xt updated for next iteration

        # Multiply decisions (X) by the actual price relative outcomes to get the 
        # growth of the portfolio in each stock ticker based on the decision made.
        # "growth" is a vector of length T that holds how much the portfolio grew each day.
        growth = (X * self.data).sum(axis=1)
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

    alpha = 1e-3
    portfolio = OnlinePortfolio(relativePrices)
    X, wealth, loss = portfolio.ons(alpha)

    # Best stock in hindsight for comparison
    cumulativeWs = np.cumprod(relativePrices, axis=0)
    finalW = cumulativeWs[-1, :]
    bestIdx = int(np.argmax(finalW))
    wealthBestStock = cumulativeWs[:, bestIdx]

    print("Weight distributions: ", X)
    print("Losses: ", loss)
    print("Final wealth (ONS): ", wealth[-1])

    # Plot the log wealth growth over time. Use log wealth since it matches with the loss
    plt.figure()
    plt.plot(dates, np.log(wealth), label="ONS (log-wealth)")
    plt.plot(dates, np.log(wealthBestStock),
             label=f"Best single stock")
    plt.title("Online Newton Step - Portfolio Log Wealth")
    plt.xlabel("date")
    plt.ylabel("log wealth")
    plt.legend()
    plt.tight_layout()
    plt.show()

    
if __name__ == "__main__":
    main()