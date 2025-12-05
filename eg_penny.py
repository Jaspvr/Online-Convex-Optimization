import numpy as np
import matplotlib.pyplot as plt

from data_handling.data_handler import *
from best_stock import bestInHindsight
from data.tickers import *
from optimal_crp import optimalCrpWeightsCvx
from uniform_crp import uniformCRP
from exponentiated_gradient import OnlinePortfolio as OnlinePortfolioEG
from online_gradient_descent import OnlinePortfolioOGD
from online_newton_step import OnlinePortfolio as OnlinePortfolioONS

class OnlinePortfolioEGPenny:
    def __init__(self, data, lam):
        self.data = data
        self.T, self.n = data.shape
        self.c = 1
        self.C = 1
        self.learnScalar = 100
        self.weights = np.ones(self.n) / self.n

        # Volatility tracking for each stock:
        self.volatility = np.ones(self.n)

        self.lam = lam

    def updateVolatility(self, rt):
        dailyReturn = rt - 1.0 

        self.volatility = np.sqrt(
            (1.0 - self.lam) * (self.volatility ** 2) +
            self.lam * (dailyReturn ** 2) +
            1e-10
        )

    def computeEta(self, rt):
        if min(rt) < self.c:
            self.c = min(rt)
        
        if max(rt) > self.C:
            self.C = max(rt)

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
            rt = self.data[t]

            eta = self.learnScalar * self.computeEta(rt)

            self.updateVolatility(rt)
            volScaledGradient = gradt / (self.volatility + 1e-10)

            yt = xt * np.exp(-eta * volScaledGradient)

            xt = yt / sum(yt)

        growth = (X * self.data).sum(axis=1)
        wealth = growth.cumprod()
        return X, wealth, L

def main():
    TICKERS = TICKERS_PENNY30
    START = "2020-11-01"
    END = "2025-10-31"

    cache_file = "data/penny20_2020-11-01_2025-10-31.csv"
    prices = loadOrDownloadPrices(TICKERS, start=START, end=END,
                                 min_days=500, cache_path=cache_file)

    relativePrices = (prices / prices.shift(1)).dropna().to_numpy()
    dates = prices.index[1:]

    # lams = [0.01, 0.05, 0.1, 0.2]
    # lamWealths = []
    # for l in lams:
    #     portfolio = OnlinePortfolioEGPenny(relativePrices, l)
    #     _, wealth, _ = portfolio.eg()
    #     lamWealths.append(wealth)


    wealthUniformCRP = uniformCRP(relativePrices)
    _, wealthOptimalCRP = optimalCrpWeightsCvx(relativePrices)
    wealthBestStock = bestInHindsight(relativePrices)
    peg = OnlinePortfolioEG(relativePrices)
    _, wealthNormalEG, _ = peg.eg()

    pons = OnlinePortfolioONS(relativePrices)
    _, wealthNormalONS, _ = pons.ons(1)

    pogd = OnlinePortfolioOGD(relativePrices)
    _, wealthNormalOGD, _ = pogd.odg()



    plt.figure()
    plt.plot(dates, np.log(wealthNormalOGD), label="OGD")
    plt.plot(dates, np.log(wealthNormalONS), label="ONS")
    plt.plot(dates, np.log(wealthNormalEG), label="EG")
    # plt.plot(dates, np.log(lamWealths[0]), label="Volatility-Scaled EG, Lambda=0.01")
    # plt.plot(dates, np.log(lamWealths[1]), label="Volatility-Scaled EG, Lambda=0.05")
    # plt.plot(dates, np.log(lamWealths[2]), label="Volatility-Scaled EG, Lambda=0.1")
    # plt.plot(dates, np.log(lamWealths[2]), label="Volatility-Scaled EG, Lambda=0.2")
    plt.title("OGD, ONS, EG on PENNY")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Log Wealth")
    plt.legend()
    plt.tight_layout()
    # plt.savefig("Plots/eg_vs_baselines_sp20.pdf")  #
    plt.savefig("Plots/penny_eg_ogd_ons.pdf")  #
    plt.show()

    
if __name__ == "__main__":
    main()