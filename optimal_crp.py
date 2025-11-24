import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from data_handling.data_handler import downloadPricesStooq
from best_stock import bestInHindsight
from projections import projectToK
from data.tickers import *


def optimalCrpWeightsCvx(rt):
    """
    Solve for optimal CRP using cvxpy
    """
    T, n = rt.shape

    xt = cp.Variable(n)
    constraints = [
        xt >= 0,
        cp.sum(xt) == 1
    ]

    # Objective: maximize final wealth which is sumt log(xt^T rt)
    eps = 1e-12
    obj = cp.Maximize(cp.sum(cp.log(rt @ xt + eps)))
    prob = cp.Problem(obj, constraints)
    prob.solve()

    # Weights and cumulative wealth for optimal CRP
    return xt.value, np.cumprod(rt @ xt.value)


def optimalCrpWeights(rt, maxIter=5000, stepSize=0.01, tol=1e-8):
    """
    Compute the optimal Constant Rebalanced Portfolio (CRP) in hindsight.

    rt: T x n array, each row is the vector of price relatives at time t
    Returns: xStar (n,), and the wealth trajectory using xStar
    """
    T, n = rt.shape

    # Start from uniform portfolio weights x0
    xt = np.ones(n) / n

    for _ in range(maxIter):
        # gradient of sumt log(xt^T rt) w.r.t. xt:
        # gradient = sumt rt / (xt^T rt)
        portfolioReturns = rt @ xt
        portfolioReturns = np.clip(portfolioReturns, 1e-12, None)

        # For each t, row t is rt / (xt^T rt)
        grad = (rt / portfolioReturns[:, None]).sum(axis=0)  # shape (n,)
        xtNew = xt + stepSize * grad

        # project back to simplex (valid portfolio)
        xtNew = projectToK(xtNew)

        # check convergence
        if np.linalg.norm(xtNew - xt, ord=1) < tol:
            xt = xtNew
            break

        xt = xtNew

    # Wealth trajectory of the CRP using fixed xt
    wealth_crp = np.cumprod(rt @ xt)

    print("optimal crp weights:", xt)
    return xt, wealth_crp


def main():
    TICKERS = TICKERS_SP10
    START = "2020-01-01"
    END = None  # Until current date

    prices = downloadPricesStooq(TICKERS, start=START, end=END, min_days=500)
    relativePrices = (prices / prices.shift(1)).dropna().to_numpy()
    dates = prices.index[1:]

    wealthBestStock = best_in_hindsight(relativePrices)

    xStar, wealthCRP = optimal_crp_weights_cvx(relativePrices)
    print("Optimal CRP weights (hindsight):", xStar)
    print("Final wealth (CRP):", wealthCRP[-1])

    # Plot the log wealth growth over time.
    plt.figure()
    plt.plot(dates, np.log(wealthBestStock),
             label=f"Best single stock")
    plt.plot(dates, np.log(wealthCRP),
             label="Optimal CRP (hindsight)")
    plt.title("Online Gradient Descent vs Optimal CRP - Portfolio Log Wealth")
    plt.xlabel("date")
    plt.ylabel("log wealth")
    plt.legend()
    plt.tight_layout()
    plt.show()

    
if __name__ == "__main__":
    main()