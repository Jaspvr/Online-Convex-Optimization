from additional_experts import *
from random_groups import randomGroups

def main():
    TICKERS = TICKERS_GROUP_SP20
    START = "2005-11-01"
    END = "2015-11-01"


    prices = downloadPricesStooq(TICKERS, start=START, end=END, min_days=500)
    relativePrices = (prices / prices.shift(1)).dropna()
    dates = prices.index[1:]
    numStocks = prices.shape[1]

    bestGroup = []
    maxWealth = None

    for i in range(20):
        print("\niteration: ", i)
        groups = randomGroups(20)

        relativePricesBundles = bundles(relativePrices, groups)

        numBundles = len(groups)
        portfolio = OnlinePortfolioBundlesOGD(relativePricesBundles, numStocks, numBundles, groups)
        _, wealthBundles, _ = portfolio.odg()

        if not maxWealth or wealthBundles > maxWealth:
            bestGroup = groups
    
    print(bestGroup)


if __name__ == "__main__":
    main()