import numpy as np
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


def optimal_crp_weights(relativePrices, max_iter=5000, step_size=0.01, tol=1e-8):
    """
    Compute the optimal Constant Rebalanced Portfolio (CRP) in hindsight.

    relativePrices: T x n array, each row xt are price relatives at time t.
    Returns: w_star (n,), and the wealth trajectory using w_star.
    """
    X = relativePrices  # T x n
    T, n = X.shape

    # Start from uniform weights
    w = np.ones(n) / n

    for it in range(max_iter):
        # gradient of sum_t log(w^T x_t) wrt w:
        # ∇ = sum_t x_t / (w^T x_t)
        wx = X @ w                 # shape (T,)
        # avoid division by very small numbers
        wx = np.clip(wx, 1e-12, None)
        grad = (X / wx[:, None]).sum(axis=0)  # shape (n,)

        # gradient ascent step
        w_new = w + step_size * grad

        # project back to simplex
        w_new = project_to_simplex(w_new)

        # check convergence
        if np.linalg.norm(w_new - w, ord=1) < tol:
            w = w_new
            break
        w = w_new

    # Wealth trajectory of the CRP:
    wealth_crp = np.cumprod(X @ w)

    return w, wealth_crp


def main():
    # Use ETF data from Stooq
    TICKERS = TICKERS_SP
    START = "2020-01-01"
    END = None  # Until current date

    prices = downloadPricesStooq(TICKERS, start=START, end=END, min_days=500)
    print(prices)

    # Form the table into T x n time series data
    # Get the ratios of the prices compared to the previous day: xt = Pt / Pt-1
    relativePrices = (prices / prices.shift(1)).dropna().to_numpy()
    dates = prices.index[1:] # Shift dates to match the relative price data (will use for plotting)

    # portfolio = OnlinePortfolio(relativePrices)
    # X, wealth, loss = portfolio.odg()

    # Best stock in hindsight for comparison
    cumulativeWs = np.cumprod(relativePrices, axis=0)
    finalW = cumulativeWs[-1, :]
    bestIdx = int(np.argmax(finalW))
    wealthBestStock = cumulativeWs[:, bestIdx]

    w_star, wealthCRP = optimal_crp_weights(relativePrices)
    print("Optimal CRP weights (hindsight):", w_star)
    print("Final wealth (CRP):", wealthCRP[-1])

    # print("Weight distributions (OGD): ", X)
    # print("Losses (OGD): ", loss)
    # print("Final wealth (OGD): ", wealth[-1])

    # Plot the log wealth growth over time.
    plt.figure()
    # plt.plot(dates, np.log(wealth), label="OGD (log-wealth)")
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