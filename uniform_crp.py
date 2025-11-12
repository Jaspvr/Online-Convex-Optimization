import numpy as np

def uniformCRP(rt):
    n = rt.shape[1]
    uniform_weights = np.ones(n) / n  # 1/n allocation to each stock
    daily_growth = rt @ uniform_weights
    cumulative_wealth = np.cumprod(daily_growth)

    return cumulative_wealth