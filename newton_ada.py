import numpy as np
import matplotlib.pyplot as plt

from data_handling.data_handler import *
from projections import *
from best_stock import bestInHindsight
from data.tickers import *
from optimal_crp import optimalCrpWeightsCvx
from uniform_crp import uniformCRP
from computeAtG import commputeAtG

class OnlinePortfolio:
    def __init__(self, data):
        self.data = data
        self.T, self.n = data.shape

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
        pt = self.data[t]
        xpMul = max(float(xt @ pt), 1e-10)
        return -np.log(xpMul)

    def gradient(self, xt, t):
        ''' Function to take the gradient of the loss with respect to the "decision" 
        of the weights for round t '''
        pt = self.data[t]
        xpMul = max(float(xt @ pt), 1e-10)
        return -pt / xpMul

    def ons(self, alpha, p):
        ''' Online Newton Step function '''
        xt = self.weights.copy()

        At = 0
        
        X = np.zeros((self.T, self.n))
        L = np.zeros((self.T))
        Grad = np.zeros((self.T, self.n))

        for t in range(self.T):
            X[t] = xt
            L[t] = self.loss(xt, t)
            gradt = self.gradient(xt, t)
            Grad[t] = gradt

            self.computeGammaEpsilon(gradt, alpha)

            if t == 0:
                At = self.epsilon * np.eye(self.n)
            At  = At + np.outer(gradt, gradt)

            # Newton step / Adagrad step
            atG = commputeAtG
            yt = xt - (1.0 / self.gamma) * atG

            xt  = cvxpyOnsProjectToK(yt, At)

        growth = (X * self.data).sum(axis=1)
        wealth = growth.cumprod()
        return X, wealth, L
        


def main():
    # Use ETF data from Stooq
    TICKERS = TICKERS_SP20
    START = "2005-11-01"
    END = "2015-10-31"  # Until current date

    # prices = downloadPricesStooq(TICKERS, start=START, end=END, min_days=500)
    # cache_file = "data/sp20Group_2015-11-01_2025-11-01.csv"
    # cache_file = "data/sp20_2015-11-01_2025-11-01.csv" # GS instead of NVIDIA
    # cache_file = "data/sp20new_2015-11-01_2025-11-01.csv" # with nvidia
    cache_file = "data/sp20new_2005-11-01_2015-11-01.csv" # with nvidia 2005-2015
    prices = loadOrDownloadPrices(TICKERS, start=START, end=END,
                                 min_days=500, cache_path=cache_file)
    print(prices)

    relativePrices = (prices / prices.shift(1)).dropna().to_numpy()
    dates = prices.index[1:]

    alpha = 1
    portfolio = OnlinePortfolio(relativePrices)
    X, wealth, loss = portfolio.ons(alpha)

    # Comparison
    wealthUniformCRP = uniformCRP(relativePrices)
    _, wealthOptimalCRP = optimalCrpWeightsCvx(relativePrices)
    wealthBestStock = bestInHindsight(relativePrices)

    print("Weight distributions: ", X)
    print("Losses: ", loss)
    print("Final wealth (ONS): ", wealth[-1])
    print("Final log wealth (ONS): ", np.log(wealth[-1]))

    plt.figure()
    plt.plot(dates, np.log(wealth), label="ONS")
    plt.plot(dates, np.log(wealthUniformCRP),
             label=f"Uniform CRP")
    plt.plot(dates, np.log(wealthOptimalCRP),
             label=f"Optimal CRP")
    plt.title("Online Newton Step vs Baseline Strategies")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Log Wealth")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Plots/ons_vs_baselines_sp20_scaledEta.pdf")  # vector graphic
    plt.show()

    
if __name__ == "__main__":
    main()