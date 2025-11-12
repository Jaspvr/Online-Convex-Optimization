def bundles(prices, groups):
    # prices is all of the closing stock prices for the studied time period
    # groups is an array that describes how to create bundles
    # ex. groups = [[0, 1], [2,3]]: two groups, one with 0, 1, the other with 2, 3.
    df = prices.copy()

    for group in groups:
        df[df.shape[1]] = df.iloc[:, group].sum(axis=1)  # insert on the right side

    return df


def eliminateBundles(weights, groups, n):
    for k, group in enumerate(groups):
        # get the weight associated with the group per element of group
        w = weights[-(len(groups)-k)] / len(groups[k])
        for col in group:
            weights[col] += w
    
    # Remove group columns
    return weights[:n]