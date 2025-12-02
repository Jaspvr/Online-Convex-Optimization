import numpy as np
import matplotlib.pyplot as plt

from data_handling.data_handler import *
from best_stock import bestInHindsight
from data.tickers import *
from optimal_crp import optimalCrpWeightsCvx
from uniform_crp import uniformCRP

class OnlinePortfolioFTLV:
    def __init__(self, data):
        self.data = data
        self.T, self.n = data.shape
        self.weights = np.ones(self.n) / self.n

    def followLeader2Positives(self):
        xt = self.weights.copy()
        
        X = np.zeros((self.T, self.n))
        L = np.zeros((self.T))

        prevUpIndices = []
        for t in range(self.T):
            X[t] = xt
            rt = self.data[t]

            upIndices = []
            for i, rti in enumerate(rt):
                if rti > 1:
                    upIndices.append(i)
            
            inBoth = []
            for i in upIndices:
                if i in prevUpIndices:
                    inBoth.append(i)

            if inBoth != []:
                allocation = 1.0 / len(inBoth)
                xt = np.zeros_like(xt)
                for i in inBoth:
                    xt[i] = allocation
            
            prevUpIndices = upIndices
            
            # Otherwise stick with weight choice of previous round
        
        growth = (X * self.data).sum(axis=1)
        wealth = growth.cumprod()
        return X, wealth, L

    def followLeaderPositives(self):
        xt = self.weights.copy()
        
        X = np.zeros((self.T, self.n))
        L = np.zeros((self.T))

        for t in range(self.T):
            X[t] = xt
            rt = self.data[t]

            upIndices = []
            for i, rti in enumerate(rt):
                if rti > 1:
                    upIndices.append(i)
            
            if upIndices != []:
                allocation = 1.0 / len(upIndices)
                xt = np.zeros_like(xt)
                for i in upIndices:
                    xt[i] = allocation
            
            # Otherwise stick with weight choice of previous round
        
        growth = (X * self.data).sum(axis=1)
        wealth = growth.cumprod()
        return X, wealth, L

    def followLeader3Day(self):
        xt = self.weights.copy()
        
        X = np.zeros((self.T, self.n))
        L = np.zeros((self.T))

        rets = [[0, 0, 0] for _ in range(self.n)]

        for t in range(self.T):
            X[t] = xt
            rt = self.data[t]

            sumArr = []
            for i in range(self.n):
                rets[i][t%3] = rt[i]
                sumArr.append(sum(rets[i]))

            bestIdx = np.argmax(sumArr)

            xt = np.zeros_like(xt)
            xt[bestIdx] = 1.0

        growth = (X * self.data).sum(axis=1)
        wealth = growth.cumprod()
        return X, wealth, L

    def followLeader1Day(self):
        xt = self.weights.copy()
        
        X = np.zeros((self.T, self.n))
        L = np.zeros((self.T))

        rets = [[0, 0, 0] for _ in range(self.n)]

        for t in range(self.T):
            X[t] = xt
            rt = self.data[t]

            bestIdx = np.argmax(rt)

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

    relative_df = (prices / prices.shift(1)).dropna()
    relative_df.to_csv("relative_prices_with_dates.csv")

    portfolio = OnlinePortfolioFTLV(relativePrices)
    _, wealth, _ = portfolio.followLeader3Day()

    wealthUniformCRP = uniformCRP(relativePrices)
    _, wealthOptimalCRP = optimalCrpWeightsCvx(relativePrices)
    wealthBestStock = bestInHindsight(relativePrices)

    print("Final wealth follow leader: ", wealth[-1])


    plt.figure()
    plt.plot(dates, np.log(wealth), label="FTL Modified")
    plt.title("Modified Follow The Leader Algorithm")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Log Wealth")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Plots/follow_leader_mod.pdf")  #
    plt.show()

    
if __name__ == "__main__":
    main()