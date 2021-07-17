import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import datetime

def load_data(root: str, column: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Loads data from the root directory from the start_date to the end_date

    Args:
        root: root directory to search for files
        column: Open, High, Low, Close, or Volume to return
        start_date: starting date
        end_date: ending date

    Returns:
        Pandas dataframe with timestamp index and tickers as columns

    Raises:
        ValueError: raised if column is not valid
    """
    files = glob(f'{root}/*.parquet')

    # Check column is valid
    if column not in ['Open', 'High', 'Low', 'Close', 'Volume']:
        raise ValueError(f'column {column} is not a valid column name.')

    # Load in data into dictionary to concatenate using an outer join
    dfs = {}
    print('Loading Files...')
    for f in tqdm(files):
        df = pd.read_parquet(f)
        # Get the ticker to use as a column name
        ticker = f[f.rfind('/')+1:f.rfind('.parquet')]

        # Check if we have any data
        if df.shape[0] == 0:
            continue

        dfs[ticker] = df[column].loc[start_date:end_date]
    if len(dfs) == 0:
        raise ValueError(f'No data in range {start_date}:{end_date}')
    return pd.concat(dfs, join='outer', axis=1)


def filter_data(prices: pd.DataFrame, filter_type: str) -> None:
    """
    Filters the dataset to include only trading hours.

    Args:
        prices: pandas dataframe of prices for each security

    Returns:
        None: modifies the prices dataframe in place
    """
    prices.drop(prices[prices.index.time <
                datetime.time(9, 30)].index, inplace=True)
    prices.drop(prices[prices.index.time >
                datetime.time(15, 59)].index, inplace=True)
    if filter_type == 'drop':
        prices.dropna(inplace=True)
    elif filter_type == 'ffill':
        prices.fillna(method='ffill',inplace=True)

def get_train_test(prices, train_period: int) -> tuple:
    """
    Generator function that yields a train and a test data based on the training period length.

    Args:
        train_period: number of days to create select a mean-reverting portfolio

    Returns:
        tuple: (train,test) are both pandas dataframes with the same structure as prices
    """
    days = list(dict.fromkeys(prices.index.date))
    train_day = 0
    test_day = train_period
    last = len(days) - 1

    while test_day <= last:
        train = prices.loc[days[train_day]:days[test_day]]
        test = prices.loc[days[test_day]:days[test_day]+pd.DateOffset(1)]

        yield (train, test)

        train_day += 1
        test_day += 1
        
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

def get_returns_prices(root_dir):
    prices = load_data(root_dir, 'Close', pd.Timestamp('2019-01-01'), pd.Timestamp('2020-01-01'))

    # remove after hours and pretrading data
    filter_data(prices, 'ffill')

    # reindex to include every minute to make the data regular
    indx = pd.date_range(start=prices.index[0], end=prices.index[-1], freq='1T')
#     prices = prices.reindex(indx,fill_value=0.0)
    prices = prices.reindex(indx, method='ffill')

    filter_data(prices, 'ffill')

    # remove holidays, weekends, etc.
    out_days = remove_days(prices)
    prices.drop(prices.loc[np.isin(prices.index.date,out_days)].index, inplace=True)

    # get returns of data
#     returns = prices.pct_change().fillna(value=0.0)
    
    # for 1 year, should have about 252 trading days, each with the same number of data points

    trains = []
    tests = []

    for train, test in get_train_test(prices, 5):
        trains.append(train)
        tests.append(prices.loc[test.index])
    
    train_dataset = np.rollaxis(np.dstack(trains),-1)
    test_dataset = np.rollaxis(np.dstack(tests),-1)
    
    return train_dataset, test_dataset

if __name__=="__main__":
    main()