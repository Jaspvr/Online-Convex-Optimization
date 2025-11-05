import numpy as np

def best_in_hindsight(rt):
    """
    Calculate wealth of portfolio if full portfolio is allocated to the
    best performing stock in hindsight.
    rt: Price relatives for all stocks at all time steps
    """
    cumulativeWs = np.cumprod(rt, axis=0)
    finalW = cumulativeWs[-1, :]
    bestIdx = int(np.argmax(finalW))
    
    # Total wealth after each trading day
    return cumulativeWs[:, bestIdx]