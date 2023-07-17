import yaml, os
import os
import fnmatch
import pandas as pd
import numpy as np

with open('conf/create_clean_data.yaml', 'r') as file:
    conf = yaml.safe_load(file)


#  Downloading {#64c,26}
def process_data(file_path, dataset_extension):

    df = pd.read_csv(file_path, header=None, names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Open Interest'])
    df['Date'] = pd.to_datetime(df['Date'])  
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    ticker = os.path.basename(file_path).split('.')[0][:-len(dataset_extension)]
    df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
    return df

def load_data(dataset_location, dataset_assets, dataset_extension = ''):
    dfs = []
    for filename in os.listdir(dataset_location):
        if filename in [str(x)+str(dataset_extension)+'.CSV' for x in dataset_assets]:
            file_path = os.path.join(dataset_location, filename)
            df = process_data(file_path, dataset_extension)
            dfs.append(df)
    df = pd.concat(dfs, axis=1)
    return df

def adjust_and_swap_index(df):
    
    df.columns = df.columns.swaplevel(0, 1)
    df = df.sort_index(axis=1)

    return df

#  Prepocessing {#16c,14}
def filter_zeros(df, max_zero_days_per_future, max_zero_futures_per_day_ratio):
    zero_days_per_future = (df == 0).sum(axis=0)
    zero_futures_per_day = (df == 0).sum(axis=1)
    zero_days_per_future_ratio = zero_days_per_future.div(df.shape[0])
    zero_futures_per_day_ratio = zero_futures_per_day.div(df.shape[1])
    df_filtered = df.loc[:, zero_days_per_future_ratio < max_zero_days_per_future / df.shape[0]]
    df_filtered = df_filtered.loc[zero_futures_per_day_ratio < max_zero_futures_per_day_ratio, :]
    return df_filtered

def fill_zeros(df):
    df.replace(0, np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    return df



if __name__ == "__main__":

    if conf['dataset_type'] == 'pinnacle':        
        #  Downloading {#64c,3}
        prices = load_data(dataset_location = conf['pinnacle_location'], 
                           dataset_assets = conf['pinnacle_assets']   , 
                           dataset_extension = conf['pinnacle_extension'])
    else:
        print('Other data (US stocks) not prepared yet. Use pinnacle dataset')
    
#  Downloading {#64c,1}
    prices = adjust_and_swap_index(prices)
    #  Prepocessing {#16c,5}
    prices = filter_zeros(df = prices, 
                          max_zero_days_per_future = conf['max_zero_days_per_future'], 
                          max_zero_futures_per_day_ratio = conf['max_zero_futures_per_day_ratio'])
                          
    prices = fill_zeros(df = prices)

    print(prices.head())
    prices.to_pickle(f'cleaned/{conf["dataset_type"]}.pkl')
    print(f'\nfile saved: cleaned/{conf["dataset_type"]}.pkl\n')

