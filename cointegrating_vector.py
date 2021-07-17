import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../..')
from utils.backtest_utils import get_pnl
from statsmodels.tsa.tsatools import lagmat

def get_vector_resid(arr):
    # check if any columns are zeros
    if len(arr) < 2:
            return np.zeros(arr.shape[1])
    
    zero_inds = np.where(np.sum(np.abs(arr), axis=0) == 0)[0]
    
    if len(zero_inds) > 0:
        arr = np.delete(arr, zero_inds, axis=1)
        
        if len(arr) < 2:
            return np.zeros(arr.shape[1])
        y = arr[:,0]
        X = np.hstack([arr[:,1:], np.ones((arr.shape[0],1))])
        hedges = np.dot(np.linalg.pinv(X), y)
        resid = y - np.dot(X,hedges)
        for i in zero_inds:
            hedges = np.insert(hedges,i,0)
        return hedges, resid
    else:
        y = arr[:,0]
        X = np.hstack([arr[:,1:], np.ones((arr.shape[0],1))])
        hedges = np.dot(np.linalg.pinv(X), y)
        resid = y - np.dot(X,hedges)
        return hedges, resid

def main():
    train_dataset = np.load('data/train_1.npy')
    test_dataset = np.load('data/test_1.npy')

    test_1 = test_dataset[10,:,:]
    
    hedges, resid = get_vector_resid(test_1)

    pnl = get_pnl(resid, resid.std()*1.5, -resid.std()*1.5)
    print(f'Final PnL: {pnl[-1]}')

if __name__=="__main__":
    main()