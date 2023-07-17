import yaml, os
import os
import fnmatch
import pandas as pd
import numpy as np

with open('conf/create_dataset.yaml', 'r') as file:
    conf = yaml.safe_load(file)


def process_data(file_path):

    df = pd.read_csv(file_path, header=None, names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Open Interest'])
    df['Date'] = pd.to_datetime(df['Date'])  
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    ticker = os.path.basename(file_path).split('.')[0]
    df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
    return df

def load_data(dataset_location, dataset_assets, dataset_extension=None):

    dfs = []
    for filename in os.listdir(dataset_location):
        if filename[:2] in assets:
            file_path = os.path.join(directory, filename)
            df = process_data(file_path)
            dfs.append(df)
    df = pd.concat(dfs, axis=1)
    return df

if __name__ == "__main__":
     dataset_type = conf['dataset_type']
     print(dataset_type)

    # if dataset_type == 'pinnacle':        
    #     files_location = conf['pinnacle_location']
    #     assets = conf['pinnacle_assets']   
    #     files_extension = conf['pinnacle_extension']
    #     df = load_data(files_location, assets, files_extension)
    #     print(df)
    # else:
    #     print('hi')
