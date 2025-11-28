import numpy as np

from data_handling.data_handler import *
from projections import *
from best_stock import bestInHindsight
from data.tickers import *
from optimal_crp import optimalCrpWeightsCvx
from uniform_crp import uniformCRP
from newton_ada import OnlinePortfolio

def commputeAtG(A, grad, p):
    """ONS updates with A^{-1}, AdaGrad updates with A^{-1/2}
    here we parametrize the exponent, giving A^{p}@grad"""

    # Use A^p = Q Λ^p Q^T
    eigenValues, eigenVectors = np.linalg.eigh(A)
    eigenValues = np.clip(eigenValues, 1e-10, None)
    eigenValuesP = eigenValues ** p

    qtgrad = eigenVectors.T @ grad
    qtGradVals = eigenValuesP * qtgrad

    return eigenVectors @ qtGradVals


def getBestP():
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
    
    maxW = 0
    bestP = -2
    ps = np.linspace(-0.5, -1.0, 11)
    for p in ps:
        portfolio = OnlinePortfolio(relativePrices)
        _, wealth, _ = portfolio.ons(alpha, p)
        if maxW < wealth[-1]:
            bestP = p

    return bestP, maxW


def main():
    return getBestP()

if __name__ == "__main__":
    main()
