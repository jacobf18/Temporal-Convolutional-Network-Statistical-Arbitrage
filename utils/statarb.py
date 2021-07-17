import numpy as np
import torch

def get_vector_resid_tensor(ts: torch.Tensor):
    if len(ts) < 2:
        return torch.zeros(ts.shape[1], dtype=torch.float64), torch.zeros(ts.shape[0], dtype=torch.float64)
    temp = torch.sum(torch.abs(ts), dim=0)
    num_zero_inds = len(torch.where(temp == 0))
    keep_inds = torch.where(temp > 0)[0]
    zero_inds = torch.where(temp == 0)[0]

    ts_filtered = ts[:,keep_inds.tolist()]

    if ts_filtered.nelement() == 0:
        return torch.zeros(ts.shape[1], dtype=torch.float64), torch.zeros(ts.shape[0], dtype=torch.float64)
    
    # assuming the 2d tensor has no zero columns
    nrows, ncols = ts.shape
    y = ts_filtered[:,0]
    X = torch.hstack([ts_filtered[:,1:], torch.ones((nrows, 1))])
    hedges = torch.mv(torch.pinverse(X), y)
    resid = y - torch.mv(X,hedges)

    if num_zero_inds > 0:
        for i in zero_inds.tolist():
            hedges = torch.cat([hedges[:i], torch.Tensor([0]), hedges[i:]],0)

    return hedges.type(torch.float64), resid.type(torch.float64)

def get_vector_resid(arr):
    # check if any columns are zeros
    if len(arr) < 2:
            return np.zeros(arr.shape[1]), np.zeros(arr.shape[0])
    
    zero_inds = np.where(np.sum(np.abs(arr), axis=0) == 0)[0]
    
    if len(zero_inds) > 0:
        arr = np.delete(arr, zero_inds, axis=1)
        
        if len(arr) < 2:
            return np.zeros(arr.shape[1]), np.zeros(arr.shape[0])
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
    
def getPolyVal(x,coeffs):
    curVal=0
    for curValIndex in range(len(coeffs)-1):
        curVal=(curVal+coeffs[curValIndex])*x
    return curVal+coeffs[len(coeffs)-1]

tau_star_nc = [-1.04, -1.53, -2.68, -3.09, -3.07, -3.77]
tau_min_nc = [-19.04,-19.62,-21.21,-23.25,-21.63,-25.74]
tau_max_nc = [np.inf,1.51,0.86,0.88,1.05,1.24]

small_scaling = torch.Tensor([1,1,1e-2])
tau_nc_smallp = [ [0.6344,1.2378,3.2496],
                  [1.9129,1.3857,3.5322],
                  [2.7648,1.4502,3.4186],
                  [3.4336,1.4835,3.19],
                  [4.0999,1.5533,3.59],
                  [4.5388,1.5344,2.9807]]
tau_nc_smallp = torch.Tensor(tau_nc_smallp)*small_scaling

large_scaling = torch.Tensor([1,1e-1,1e-1,1e-2])
tau_nc_largep = [ [0.4797,9.3557,-0.6999,3.3066],
                  [1.5578,8.558,-2.083,-3.3549],
                  [2.2268,6.8093,-3.2362,-5.4448],
                  [2.7654,6.4502,-3.0811,-4.4946],
                  [3.2684,6.8051,-2.6778,-3.4972],
                  [3.7268,7.167,-2.3648,-2.8288]]
tau_nc_largep = torch.Tensor(tau_nc_largep)*large_scaling

def mackinnonp(teststat):
    maxstat = tau_max_nc
    minstat = tau_min_nc
    starstat = tau_star_nc
    if teststat > maxstat[0]:
        return 1.0
    elif teststat < minstat[0]:
        return 0.0
    if teststat <= starstat[0]:
        tau_coef = tau_nc_smallp[0]
    else:
        tau_coef = tau_nc_largep[0]
    normal_dist = torch.distributions.normal.Normal(0,1)
    p = getPolyVal(teststat, torch.flip(tau_coef,[0]))
    return normal_dist.cdf(getPolyVal(teststat, torch.flip(tau_coef,[0])))

def lagmat(x, maxlag):
    nobs, nvar = x.shape
    lm = torch.zeros((nobs + maxlag, nvar * (maxlag + 1)), dtype=torch.float32)
    for k in range(0, int(maxlag + 1)):
        lm[maxlag - k:nobs + maxlag - k, nvar * (maxlag - k):nvar * (maxlag - k + 1)] = x

    startobs, stopobs = maxlag, nobs

    lags = lm[startobs:stopobs,:]
    return lags

def adfuller(ts):
    nobs = ts.shape[0]
    ntrend = 0
    # from Greene referencing Schwert 1989
    maxlag = int(np.ceil(12.0 * np.power(nobs / 100.0, 1 / 4.0)))
    maxlag = min(nobs // 2 - ntrend - 1, maxlag)
    
    maxlag = 1
    xdiff = torch.diff(ts)
    xdall = lagmat(xdiff[:, None], maxlag)
    
    nobs = xdall.shape[0]
    xdall[:, 0] = ts[-nobs - 1 : -1]
    
    xdshort = xdiff[-nobs:]
    
    X = xdall[:, : maxlag + 1]
    params = torch.matmul(torch.linalg.pinv(X), xdshort)
    resid = xdshort - torch.matmul(X,params)
    
    variance = torch.dot(resid,resid) * torch.linalg.inv(torch.matmul(torch.transpose(X,0,1), X)) / (nobs - len(params))
    t_value = params[0] / (variance[0,0]**0.5)
    
    pvalue = mackinnonp(t_value)
    
    return pvalue

def hurst(arr: torch.Tensor, max_lag: int):
    lags = range(2, max_lag)
    
    log_lags = torch.log(torch.Tensor(lags).reshape([-1,1]))
    
    tau = torch.zeros(len(log_lags),1)
    
    for i, lag in enumerate(lags):
        tau[i] = torch.std(arr[lag:] - arr[:-lag])
    
    X = torch.hstack([log_lags, torch.ones(len(log_lags),1)])
    b = torch.matmul(torch.pinverse(X), torch.log(tau))
    
    return b[0]