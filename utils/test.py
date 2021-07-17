import numpy as np

import torch

from data import get_returns_prices
from statarb import adfuller_ts
import adfuller

train_prices_npy, test_prices_npy = get_returns_prices('../data/')

train_prices = torch.from_numpy(np.transpose(train_prices_npy, [0, 2, 1])).float()
test_prices = torch.from_numpy(np.transpose(test_prices_npy, [0, 2, 1])).float()

log_prices = torch.log(train_prices)
nsamples = log_prices.shape[2]
means = torch.unsqueeze(torch.mean(log_prices, dim = 2), dim=2).repeat((1,1,nsamples))
stds = torch.unsqueeze(torch.std(log_prices, dim = 2), dim=2).repeat((1,1,nsamples))
X = (log_prices - means) / stds

hedges = torch.Tensor([-0.1824, -0.0074,  0.2239,  0.1035,  0.0879])

p = test_prices[0]

portfolio = torch.matmul(hedges, p)
portfolio.requires_grad_()
portfolio_npy = portfolio.detach().numpy()

print(adfuller_ts(portfolio), adfuller.adfuller(portfolio_npy)[1])