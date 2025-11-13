import numpy as np

def bundles(relativePrices, groups):
    # prices is all of the relative closing stock prices for the studied time period
    # groups is an array that describes how to create bundles
    # ex. groups = [[0, 1], [2,3]]: two groups, one with 0, 1, the other with 2, 3.
    df = relativePrices.copy()

    for group in groups:
        # Average of price relatives, inserted to the right side of the df
        df[df.shape[1]] = df.iloc[:, group].mean(axis=1)

    return df.to_numpy()


def distributeBundles(weightsBundle, groups, n):
    xt = np.zeros(n)

    for k, group in enumerate(groups):
        if len(group) == 0:
            continue
        share = weightsBundle[k] / len(group)
        for stock_idx in group:
            xt[stock_idx] += share

    return xt


def eliminateBundles(weights, groups, n):
    for k, group in enumerate(groups):
        # get the weight associated with the group per element of group
        w = weights[-(len(groups)-k)] / len(groups[k])
        for col in group:
            weights[col] += w
    
    # Remove group columns
    return weights[:n]