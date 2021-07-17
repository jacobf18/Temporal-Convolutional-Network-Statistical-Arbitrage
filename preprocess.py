import numpy as np
import pandas as pd
import sys
sys.path.append('../..')
from utils.data_utils import load_data, filter_data, get_train_test

def remove_days(df):
    """
    Returns a list of days that can be removed (weekends, holidays, etc.)
    """
    days = list(dict.fromkeys(df.index.date))
    out_days = []
    for d in days:
        test = df.loc[d:d+pd.DateOffset(1)]
        if len(test.drop_duplicates(keep=False)) == 0:
            out_days.append(d)
    return out_days

def main():
    prices = load_data('../../selection_scores/data/', 'Close', pd.Timestamp('2019-01-01'), pd.Timestamp('2020-01-01'))

    # remove after hours and pretrading data
    filter_data(prices, 'ffill')

    # reindex to include every minute to make the data regular
    indx = pd.date_range(start=prices.index[0], end=prices.index[-1], freq='1T')
    prices = prices.reindex(indx,fill_value=0.0)

    filter_data(prices, 'ffill')

    # remove holidays, weekends, etc.
    out_days = remove_days(prices)
    prices.drop(prices.loc[np.isin(prices.index.date,out_days)].index, inplace=True)

    # get returns of data
    returns = prices.pct_change().fillna(value=0.0)
    
    # for 1 year, should have about 252 trading days, each with the same number of data points

    trains = []
    tests = []

    for train, test in get_train_test(returns, 5):
        trains.append(train)
        tests.append(prices.loc[test.index])
    
    train_dataset = np.rollaxis(np.dstack(trains),-1)
    test_dataset = np.rollaxis(np.dstack(tests),-1)

    # write train and test datasets to disk
    with open('data/train_1.npy', 'wb') as f:
        np.save(f,train_dataset)
    with open('data/test_1.npy', 'wb') as f:
        np.save(f,test_dataset)

if __name__=="__main__":
    main()