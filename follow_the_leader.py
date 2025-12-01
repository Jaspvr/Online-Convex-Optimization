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
        self.weights = np.ones(self.n) / self.n

    def followLeader(self):
        xt = self.weights.copy()
        
        X = np.zeros((self.T, self.n))
        L = np.zeros((self.T))

        cummulativeRet = np.zeros(self.n)

        for t in range(self.T):
            X[t] = xt
            rt = self.data[t]

            cummulativeRet += np.log(rt)
            bestIdx = np.argmax(cummulativeRet)

            xt = np.zeros_like(xt)
            xt[bestIdx] = 1.0

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

    portfolio = OnlinePortfolio(relativePrices)
    _, wealth, _ = portfolio.followLeader()

    wealthUniformCRP = uniformCRP(relativePrices)
    _, wealthOptimalCRP = optimalCrpWeightsCvx(relativePrices)
    wealthBestStock = bestInHindsight(relativePrices)

    print("Final wealth follow leader: ", wealth[-1])


    plt.figure()
    plt.plot(dates, np.log(wealth), label="Follow-the-leader")
    plt.title("Follow The Leader Algorithm")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Log Wealth")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Plots/follow_leader.pdf")  #
    plt.show()

    
if __name__ == "__main__":
    main()