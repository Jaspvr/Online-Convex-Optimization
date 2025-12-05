import numpy as np
import matplotlib.pyplot as plt

from data_handling.data_handler import downloadPricesStooq, loadOrDownloadPrices
from best_stock import bestInHindsight
from worst_stock import worstInHindsight
from projections import projectToK, cvxpyOgdProjectToK
from data.tickers import *
from additional_experts_helpers import *
from online_gradient_descent import OnlinePortfolioOGD
from uniform_crp import uniformCRP
from optimal_crp import optimalCrpWeightsCvx
from best_groups import *


class OnlinePortfolioBundlesOGD:
    def __init__(self, priceRelatives, numStocks, numBundles, groups):
        self.priceRelatives = priceRelatives
        self.T, _ = priceRelatives.shape
        self.numStocks = numStocks
        self.numBundles = numBundles
        self.groups = groups

        self.etaScalar = 20
        
        self.weights = np.ones(self.numStocks) / self.numStocks
        self.weightsBundles = np.ones(self.numStocks + self.numBundles) / (self.numStocks + self.numBundles)

        self.eta  = np.zeros(self.T+1) # one eta per round

        self.G = 0 # factor used to calculate eta
        self.D = 2**(0.5) # factor used to calculate eta

    def computeEta(self, grad, t):
        gradMag = np.linalg.norm(grad)
        if gradMag > self.G:
            self.G = gradMag

        self.eta[t] = self.D / (self.G * ((t+1)**0.5))

    def loss(self, xt, t):
        ''' Log loss function '''
        rt = self.priceRelatives[t]

        xpMul = max(float(xt @ rt), 1e-10)
        return -np.log(xpMul)

    def gradient(self, xt, t):
        ''' Function to take the gradient of the loss with respect to the "decision" 
        of the weights for round t '''
        rt = self.priceRelatives[t]

        xpMul = max(float(xt @ rt), 1e-10)
        return -rt / xpMul

    def ogdBasicBundling(self):
        xt = self.weights.copy() # Initial weight spread is uniform distribution
        bundleXt = self.weightsBundles.copy()

        X = np.zeros((self.T, self.numStocks)) # Decisions (weight spreads)
        bundlesX = np.zeros((self.T, self.numStocks + self.numBundles))
        Grad = np.zeros((self.T, self.numStocks + self.numBundles)) # Gradients
        L = np.zeros(self.T) # Loss vector (1 entry per round)

        for t in range(self.T): 
            X[t] = xt
            bundlesX[t] = bundleXt
            L[t] = self.loss(bundleXt, t)

            Grad[t] = self.gradient(bundleXt, t)
            self.computeEta(Grad[t], t)

            yNext = bundlesX[t] - self.etaScalar * self.eta[t] * Grad[t]
            bundleXt = cvxpyOgdProjectToK(yNext)

            xt = eliminateBundles(bundleXt, self.groups, self.numStocks)
            # xt = eliminateBundles_toBest(bundleXt, self.groups, self.priceRelatives[t], self.numStocks)

       
        priceRelativesNoBundles = self.priceRelatives[:, :self.numStocks]
        growth = (X * priceRelativesNoBundles).sum(axis=1)
        wealth = growth.cumprod()
        return X, wealth, L
    
    def ogdGiveToBest(self):
        xt = self.weights.copy() # Initial weight spread is uniform distribution
        bundleXt = self.weightsBundles.copy()

        X = np.zeros((self.T, self.numStocks)) # Decisions (weight spreads)
        bundlesX = np.zeros((self.T, self.numStocks + self.numBundles))
        Grad = np.zeros((self.T, self.numStocks + self.numBundles)) # Gradients
        L = np.zeros(self.T) # Loss vector (1 entry per round)

        for t in range(self.T): 
            X[t] = xt
            bundlesX[t] = bundleXt
            L[t] = self.loss(bundleXt, t)

            Grad[t] = self.gradient(bundleXt, t)
            self.computeEta(Grad[t], t)

            yNext = bundlesX[t] - self.etaScalar * self.eta[t] * Grad[t]
            bundleXt = cvxpyOgdProjectToK(yNext)

            xt = eliminateBundles_toBest(bundleXt, self.groups, self.priceRelatives[t], self.numStocks)

       
        priceRelativesNoBundles = self.priceRelatives[:, :self.numStocks]
        growth = (X * priceRelativesNoBundles).sum(axis=1)
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
    portfolio = OnlinePortfolioBundlesOGD(relativePricesBundles, numStocks, numBundles, groups)
    XBundles, wealthBundles, _ = portfolio.ogdGiveToBest()
    # XBundles, wealthBundles, _ = portfolio.ogdBasicBundling()

    # For comparison
    relativePrices = (prices / prices.shift(1)).dropna().to_numpy()
    wealthBestStock = bestInHindsight(relativePrices)
    wealthWorstStock = worstInHindsight(relativePrices)
    wealthUniformCRP = uniformCRP(relativePrices)
    _, wealthOptimalCRP = optimalCrpWeightsCvx(relativePrices)
    portfolio = OnlinePortfolioOGD(relativePrices)
    XRegular, wealthRegular, _ = portfolio.odg()

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
    plt.savefig("Plots/ogd_addexp_sp40Groups_salloc.pdf")  # vector graphic
    plt.show()

if __name__ == "__main__":
    main()