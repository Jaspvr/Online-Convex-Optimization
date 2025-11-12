import numpy as np

def worstInHindsight(rt):
    cumulativeWs = np.cumprod(rt, axis=0)
    finalW = cumulativeWs[-1, :]
    worstIdx = int(np.argmin(finalW))
    
    # Total wealth after each trading day
    return cumulativeWs[:, worstIdx]