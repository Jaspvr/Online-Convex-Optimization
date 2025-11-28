from newton_ada import *

def main():
    # Use ETF data from Stooq
    TICKERS = TICKERS_SP20
    # START = "2005-11-01"
    # END = "2015-10-31" 

    START = "2015-11-01"
    END = "2025-11-01"  

    # prices = downloadPricesStooq(TICKERS, start=START, end=END, min_days=500)
    # cache_file = "data/sp20Group_2015-11-01_2025-11-01.csv"
    # cache_file = "data/sp20_2015-11-01_2025-11-01.csv" # GS instead of NVIDIA
    cache_file = "data/sp20new_2015-11-01_2025-11-01.csv" # with nvidia
    # cache_file = "data/sp20new_2005-11-01_2015-11-01.csv" # with nvidia 2005-2015
    prices = loadOrDownloadPrices(TICKERS, start=START, end=END,
                                 min_days=500, cache_path=cache_file)

    relativePrices = (prices / prices.shift(1)).dropna().to_numpy()

    alpha = 1
    
    maxW = 0
    bestP = -2
    ps = np.linspace(-0.5, -1.0, 11)
    for p in ps:
        portfolio = OnlinePortfolio(relativePrices)
        _, wealth, _ = portfolio.ons(alpha, p)
        if maxW < wealth[-1]:
            bestP = p
            maxW = wealth[-1]
        print("p choice: ", p)
        print("wealth: ", wealth[-1])

    print("Best p: ", bestP)
    print("Max wealth: ", maxW)
    return bestP, maxW

if __name__ == "__main__":
    main()
