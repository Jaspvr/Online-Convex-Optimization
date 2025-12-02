import numpy as np
import matplotlib.pyplot as plt

from data_handling.data_handler import downloadPricesStooq, loadOrDownloadPrices
from best_stock import bestInHindsight
from worst_stock import worstInHindsight
from projections import *
from data.tickers import *
from additional_experts_helpers import *
from online_newton_step import OnlinePortfolio
from uniform_crp import uniformCRP
from optimal_crp import optimalCrpWeightsCvx
from best_groups import *


class OnlinePortfolioBundlesONS:
    def __init__(self, priceRelatives, numStocks, numBundles, groups):
        self.data = priceRelatives
        self.T, _ = priceRelatives.shape
        self.numStocks = numStocks
        self.numBundles = numBundles
        self.groups = groups

        self.weights = np.ones(self.numStocks) / self.numStocks
        self.weightsBundles = np.ones(self.numStocks + self.numBundles) / (self.numStocks + self.numBundles)

        self.G = 0
        self.D = 2**(0.5)
        self.gamma = 0
        self.epsilon = 0

    def computeGammaEpsilon(self, grad, alpha):
        gradMag = np.linalg.norm(grad)
        if gradMag > self.G:
            self.G = gradMag
        self.gamma = 0.5 * min(1/(self.G * self.D), alpha)
        self.epsilon = 1 / ((self.G**2)*(self.D**2))

    def loss(self, xt, t):
        pt = self.data[t]
        xpMul = max(float(xt @ pt), 1e-10)
        return -np.log(xpMul)

    def gradient(self, xt, t):
        pt = self.data[t]
        xpMul = max(float(xt @ pt), 1e-10)
        return -pt / xpMul
    
    def onsBasicBundling(self, alpha):
        xt = self.weights.copy()
        bundleXt = self.weightsBundles.copy()

        At = 0
        X = np.zeros((self.T, self.numStocks))
        bundlesX = np.zeros((self.T, self.numStocks + self.numBundles))
        L = np.zeros((self.T))
        Grad = np.zeros((self.T, self.numStocks + self.numBundles))
        for t in range(self.T):
            X[t] = xt
            bundlesX[t] = bundleXt

            L[t] = self.loss(bundleXt, t)

            gradt = self.gradient(bundleXt, t)
            Grad[t] = gradt

            self.computeGammaEpsilon(gradt, alpha)
            if t == 0:
                At = self.epsilon * np.eye(self.numBundles + self.numStocks)
            At  = At + np.outer(gradt, gradt)

            invAg = np.linalg.solve(At, gradt)
            yt = xt - (1.0 / self.gamma) * invAg

            bundleXt  = cvxpyOnsProjectToK(yt, At)

            xt = eliminateBundles(bundleXt, self.groups, self.numStocks)

        growth = (X * self.data).sum(axis=1)
        wealth = growth.cumprod()
        return X, wealth, L
   
    def onsGiveToBest(self):
        xt = self.weights.copy()
        bundleXt = self.weightsBundles.copy()

        At = 0
        X = np.zeros((self.T, self.numStocks))
        bundlesX = np.zeros((self.T, self.numStocks + self.numBundles))
        L = np.zeros((self.T))
        Grad = np.zeros((self.T, self.numStocks + self.numBundles))
        for t in range(self.T):
            X[t] = xt
            bundlesX[t] = bundleXt

            L[t] = self.loss(bundleXt, t)

            gradt = self.gradient(bundleXt, t)
            Grad[t] = gradt

            self.computeGammaEpsilon(gradt, alpha)
            if t == 0:
                At = self.epsilon * np.eye(self.numBundles + self.numStocks)
            At  = At + np.outer(gradt, gradt)

            invAg = np.linalg.solve(At, gradt)
            yt = xt - (1.0 / self.gamma) * invAg

            bundleXt  = cvxpyOnsProjectToK(yt, At)

            xt = eliminateBundles_toBest(bundleXt, self.groups, self.numStocks)

        growth = (X * self.data).sum(axis=1)
        wealth = growth.cumprod()
        return X, wealth, L
       
    
        


def main():
    # Ticker order: Tech (4), Health Care (3), Financials (4), Consumer Discretionary (3),
    # Industrials (3), Energy (3)
    # TICKERS = TICKERS_GROUP_SP20
    TICKERS = TICKERS_GROUP_SP40
    START = "2015-11-01"
    END = "2025-11-01"
    # groups = bestGroup20
    groups = groups40

    # cache_file = "data/sp20Group_2015-11-01_2025-11-01.csv"
    cache_file = "data/sp40Group_2015-11-01_2025-11-01.csv" 
    prices = loadOrDownloadPrices(TICKERS, start=START, end=END,
                                 min_days=500, cache_path=cache_file)
    
    relativePrices = (prices / prices.shift(1)).dropna()
    print("relp shape: ", relativePrices.shape)
    dates = prices.index[1:]
    numStocks = prices.shape[1]

    relativePricesBundles = bundles(relativePrices, groups)
    print("bundles shape: ", relativePricesBundles.shape)

    numBundles = len(groups)
    portfolio = OnlinePortfolioBundlesONS(relativePricesBundles, numStocks, numBundles, groups)
    # XBundles, wealthBundles, _ = portfolio.ogdGiveToBest()
    XBundles, wealthBundles, _ = portfolio.onsBasicBundling()

    # For comparison
    relativePrices = (prices / prices.shift(1)).dropna().to_numpy()
    wealthUniformCRP = uniformCRP(relativePrices)
    _, wealthOptimalCRP = optimalCrpWeightsCvx(relativePrices)
    portfolio = OnlinePortfolio(relativePrices)
    XRegular, wealthRegular, _ = portfolio.ons()

    print("Weight distributions regular: ", XRegular)
    print("Weight distributions bundles: ", XBundles)
    print("Final wealth (OGD regular): ", wealthRegular[-1])
    print("Final wealth (OGD bundles): ", wealthBundles[-1])
    print("Final log wealth (OGD regular): ", np.log(wealthRegular[-1]))
    print("Final log wealth (OGD bundles): ", np.log(wealthBundles[-1]))

    # Plot the log wealth growth over time. Use log wealth since it matches with the loss
    plt.figure()
    plt.plot(dates, np.log(wealthRegular), label="OGD Regular (log-wealth)")
    plt.plot(dates, np.log(wealthBundles), label="OGD Bundles (log-wealth)")
    plt.plot(dates, np.log(wealthOptimalCRP), label="Optimal CRP")
    plt.plot(dates, np.log(wealthUniformCRP), label="Uniform CRP")
    plt.title(labels[tuple(TICKERS)])
    plt.xlabel("Date")
    plt.ylabel("Portfolio Log Wealth")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Plots/ogd_addexp_sp40Groups_basic_baselines.pdf")  # vector graphic
    plt.show()

if __name__ == "__main__":
    main()