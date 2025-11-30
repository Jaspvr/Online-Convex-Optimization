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
        self.T, self.n = data.shape
        self.c = 1
        self.C = 1
        self.learnScalar = 1
        self.weights = np.ones(self.n) / self.n

    def computeEta(self, xt):
        self.c = min(xt)
        self.C = max(xt)
        return (self.c / self.C) * np.sqrt(8 * np.log(self.n) / self.T)

    def loss(self, xt, t):
        ''' Log loss function '''
        rt = self.data[t]
        xpMul = max(float(xt @ rt), 1e-10)
        return -np.log(xpMul)

    def gradient(self, xt, t):
        ''' Function to take the gradient of the loss with respect to the "decision" 
        of the weights for round t '''
        rt = self.data[t]
        xpMul = max(float(xt @ rt), 1e-10)
        return -rt / xpMul

    def eg(self):
        ''' Exponentiated Gradient algorithm '''
        xt = self.weights.copy()
        
        X = np.zeros((self.T, self.n))
        L = np.zeros((self.T))

        for t in range(self.T):
            X[t] = xt
            L[t] = self.loss(xt, t)
            gradt = self.gradient(xt, t)

            eta = self.learnScalar * self.computeEta(self.data[t])

            yt = xt * np.exp(-eta * gradt)

            xt = yt / sum(yt)

        growth = (X * self.data).sum(axis=1)
        wealth = growth.cumprod()
        return X, wealth, L

def main():
    TICKERS = TICKERS_SP20
    START = "2015-11-01"
    END = "2020-10-31"

    cache_file = "data/penny20_2015-11-01_2020-10-31.csv"
    prices = loadOrDownloadPrices(TICKERS, start=START, end=END,
                                 min_days=500, cache_path=cache_file)

    relativePrices = (prices / prices.shift(1)).dropna().to_numpy()
    dates = prices.index[1:]

    portfolio = OnlinePortfolio(relativePrices)
    X, wealth, loss = portfolio.eg()

    wealthUniformCRP = uniformCRP(relativePrices)
    _, wealthOptimalCRP = optimalCrpWeightsCvx(relativePrices)
    wealthBestStock = bestInHindsight(relativePrices)

    print("Final wealth (EG): ", wealth[-1])


    plt.figure()
    plt.plot(dates, np.log(wealth), label="EG")
    # plt.plot(dates, np.log(wealthBestStock),
    #          label=f"Best single stock")
    plt.plot(dates, np.log(wealthUniformCRP),
             label=f"Uniform CRP")
    plt.plot(dates, np.log(wealthOptimalCRP),
             label=f"Optimal CRP")
    plt.title("Exponentiated Gradient vs Baseline Strategies")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Log Wealth")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Plots/eg_vs_baselines_sp20.pdf")  #
    plt.show()

    
if __name__ == "__main__":
    main()