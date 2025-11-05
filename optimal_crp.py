import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from data_handling.data_handler import downloadPricesStooq
from data.tickers import *


def project_to_simplex(v):
    """
    Euclidean projection of v onto the probability simplex:
        { w : w_i >= 0, sum_i w_i = 1 }
    """
    v = np.asarray(v, dtype=float)
    n = v.size
    # Sort v in descending order
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # Find rho
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    return w


def optimal_crp_weights(rt, maxIter=5000, stepSize=0.01, tol=1e-8):
    """
    Compute the optimal Constant Rebalanced Portfolio (CRP) in hindsight.

    rt: T x n array, each row is the vector of price relatives at time t
    Returns: xStar (n,), and the wealth trajectory using xStar
    """
    # rt is T x n
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
        xtNew = project_to_simplex(xtNew)

        # check convergence
        if np.linalg.norm(xtNew - xt, ord=1) < tol:
            xt = xtNew
            break

        xt = xtNew

    # Wealth trajectory of the CRP using fixed xt
    wealth_crp = np.cumprod(rt @ xt)

    return xt, wealth_crp


def optimal_crp_weights_cvx(rt):
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

    xStar = xt.value
    xStar = project_to_simplex(xStar)

    wealthCrp = np.cumprod(rt @ xStar)

    return xStar, wealthCrp


def main():
    # Use ETF data from Stooq
    TICKERS = TICKERS_SP10
    START = "2020-01-01"
    END = None  # Until current date

    prices = downloadPricesStooq(TICKERS, start=START, end=END, min_days=500)
    relativePrices = (prices / prices.shift(1)).dropna().to_numpy()
    dates = prices.index[1:]

    # Best stock in hindsight for comparison
    cumulativeWs = np.cumprod(relativePrices, axis=0)
    finalW = cumulativeWs[-1, :]
    bestIdx = int(np.argmax(finalW))
    wealthBestStock = cumulativeWs[:, bestIdx]

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