from additional_experts import *
from random_groups import randomGroups

def main():
    TICKERS = TICKERS_GROUP_SP40
    START = "2005-11-01"
    END = "2015-11-01"


    cache_file = "data/sp40Group_2005-11-01_2015-11-01.csv"
    # cache_file = "data/sp40Group_2015-11-01_2025-11-01.csv"

    # cache_file = "data/sp20Group_2015-11-01_2025-11-01.csv"
    # cache_file = "data/sp20Group_2005-11-01_2015-11-01.csv" 

    # cache_file = "data/sp20new_2005-11-01_2015-11-01.csv" # with nvidia
    prices = loadOrDownloadPrices(TICKERS, start=START, end=END,
                                 min_days=500, cache_path=cache_file)
    relativePrices = (prices / prices.shift(1)).dropna()
    dates = prices.index[1:]
    numStocks = prices.shape[1]
    print(numStocks)

    bestGroup = []
    maxWealth = None

    for i in range(20):
        print("\niteration: ", i)
        groups = randomGroups(40)

        relativePricesBundles = bundles(relativePrices, groups)

        numBundles = len(groups)
        portfolio = OnlinePortfolioBundlesOGD(relativePricesBundles, numStocks, numBundles, groups)
        _, wealthBundles, _ = portfolio.ogdBasicBundling()

        if not maxWealth or wealthBundles[-1] > maxWealth:
            bestGroup = groups
            maxWealth = wealthBundles[-1]
    
    print(bestGroup)
    print(wealthBundles[-1])


if __name__ == "__main__":
    main()