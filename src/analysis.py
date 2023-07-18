import pandas as pd 
import yaml

with open('conf/analysis.yaml', 'r') as file:
    conf = yaml.safe_load(file)

if __name__ == "__main__":

    
    prices = pd.read_pickle(conf['pinnacle_location'])

    print()
    print(prices.head())
    print()

    # assets_prices = []
    # excluded_tickers = []
    # for i in assets:
    #     if i != 'SPY':
    #         prices_ = prices.drop(columns='Open Interest').loc[:, (slice(None), str(i)+'_RAD')].dropna()
    #         prices_.columns = prices_.columns.droplevel(1)     
    #         # print(i, len(prices_.columns))

    #         if len(prices_.columns) == 5:
    #             prices_.columns = ['Open', 'High', 'Low', 'Close', 'Volume']        
    #             assets_prices.append(bt.feeds.PandasData(dataname=prices_, plot=False))
    #         else:
    #             excluded_tickers.append(i)

    # display(prices_.head())