import numpy as np

def get_pnl(series, entry, exit):
    """
    The series should be a numpy array and entry and exit are just floats.
    It is "long-only" right now as in you only go long on the portfolio.
    However, since the portfolio should be made of long and short positions, it is not actually long only.
    """
    pos = np.zeros(len(series))
    start_ind = 0
    cur = 0
    nz = np.diff(((series >= exit) + ((series <= entry) * -1)))
    inds = np.nonzero(nz)[0]
    dif = nz[inds]
    for i in range(len(dif)):
        if cur == 0: # no position
            if (dif[i] == -1 and i == 0) or (dif[i] == -1 and dif[i-1] == -1): # buy
                pos[start_ind:inds[i]-1] = cur
                cur = 1
                start_ind = inds[i]
        else: # long
            if dif[i] == 1 and dif[i-1] == 1: # sell
                pos[start_ind:inds[i] - 1] = cur
                cur = 0
                start_ind = inds[i]
    pos[start_ind:] = cur
    pnl = np.zeros(len(series))
    pnl[1:] = np.diff(series)
    equity = (pnl * pos).cumsum()
    return equity