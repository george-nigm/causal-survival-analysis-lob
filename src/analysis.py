import yaml
import pandas as pd 
import numpy as np
import backtrader as bt
import riskfolio as rp
import quantstats
from portfolio_opt_methods import equal_weights, random_weights, mv_portfolio, grangers_causation_matrix_portfolio

with open('conf/analysis.yaml', 'r') as file:
    conf = yaml.safe_load(file)

def get_portfolio_weights(returns, index_, full_conf):

    weights = pd.DataFrame([])

    for num, i in enumerate(index_):
        Y = returns.iloc[i-full_conf['portfolio_method_config']['window_size']:i,:] # taking last window_size days

        if full_conf['portfolio_method_config']['method'] == 'random_weights':
            w = random_weights(Y)

        if full_conf['portfolio_method_config']['method'] == 'equal_weights':
            w = equal_weights(Y)

        if full_conf['portfolio_method_config']['method'] == 'MV':            
            w = mv_portfolio(Y, full_conf['mv_config'])
        
        if full_conf['portfolio_method_config']['method'] == 'grangers_causation_matrix':            
            w = grangers_causation_matrix_portfolio(Y, full_conf['grangers_causation_matrix_config'])
            
        if w is None:
            w = weights.tail(1).T
        weights = pd.concat([weights, w.T], axis = 0)

    return weights


class AssetAllocation(bt.Strategy):
    params = (
        ('assets', None),
        ('weights', None),
    )

    def __init__(self):         
        j = 0
        for i in self.params.assets:
            setattr(self, i, self.datas[j])
            j += 1
        
        self.counter = 0
        
    def next(self):
        if self.counter in self.params.weights.index.tolist():
            
            # print("Before rebalancing positions:")
            # for i, d in enumerate(self.datas):
            #     pos = self.getposition(d).size
            #     if pos != 0:
            #         print(f'{self.params.weights.columns[i]}: Position size: {pos}')

            reb_weights = []
            for i in self.params.assets:
                w = self.params.weights.loc[self.counter, i]
                reb_weights.append(round(w,3))
                self.order_target_percent(getattr(self, i), target=w)
            print('\n',len(reb_weights), reb_weights)

        self.counter += 1


def backtest(datas, strategy, assets, weights, start, end, backtrader_config, plot=False, **kwargs):
    cerebro = bt.Cerebro()
    print('\nok\n')

    # Here we add transaction costs and other broker costs
    cerebro.broker.setcash(backtrader_config['cash'])
    cerebro.broker.setcommission(commission=backtrader_config['commission']) # Commission 0.5%
    cerebro.broker.set_slippage_perc(backtrader_config['slippage_perc'], # Slippage 0.5%
                                     slip_open=True,
                                     slip_limit=True,
                                     slip_match=True,
                                     slip_out=False)

    # Here we add the indicators that we are going to store
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=backtrader_config['riskfreerate'])
    cerebro.addanalyzer(bt.analyzers.Returns)
    cerebro.addanalyzer(bt.analyzers.DrawDown)
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='PyFolio')
    print(weights)
    cerebro.addstrategy(strategy, assets=assets, weights=weights, **kwargs)
    cerebro.addobserver(bt.observers.Value)
    cerebro.addobserver(bt.observers.DrawDown)
    
    for data in datas:
        cerebro.adddata(data)
    
    print('Start Portfolio Value: %.2f' % cerebro.broker.getvalue())
    results = cerebro.run(stdstats=False)
    print('\nFinal Portfolio Value: %.2f' % cerebro.broker.getvalue())


    if plot:
        # cerebro.plot(iplot=False, start=start, end=end)
        cerebro.plot(iplot=False)
    return (results,
            results[0].analyzers.drawdown.get_analysis()['max']['drawdown'],
            results[0].analyzers.returns.get_analysis()['rnorm100'],
            results[0].analyzers.sharperatio.get_analysis()['sharperatio'])

if __name__ == "__main__":
    print() 

    method_config = conf['portfolio_method_config']
    backtrader_config = conf['backtrader_config'] 
    
    if conf['dataset_type'] == 'pinnacle':
        dataset_config = conf['pinnacle_config']
        prices = pd.read_pickle(dataset_config['location'])
        assets = dataset_config['assets']

    prices = prices[prices.index >= conf['start_data']]
    prices = prices[prices.index <= conf['end_data']]
    
    assets_prices = []
    excluded_tickers = []

    for i in assets:
        prices_ = prices.drop(columns='Open Interest').loc[:, (slice(None), i)].dropna()
        prices_.columns = prices_.columns.droplevel(1)     

        if len(prices_.columns) == 5: # check fullfilness for backtesting
            prices_.columns = ['Open', 'High', 'Low', 'Close', 'Volume']        
            assets_prices.append(bt.feeds.PandasData(dataname=prices_.iloc[method_config['window_size']:], plot=False))
            # print('prices_.index[0]: ', prices_.index[0])
            # print('prices_.iloc[method_config["window_size:"]:].index[0]: ', prices_.iloc[method_config['window_size']:].index[0])
            # break
        else:
            excluded_tickers.append(i)

    assets = [x for x in assets if x not in excluded_tickers]
    prices = prices[[col for col in prices.columns if col[1] in assets]]
    print('\n', prices)
        
    
    data = prices.loc[:, ('Close', slice(None))]
    data.columns = data.columns.droplevel(0)
    data.columns = assets
    returns = data.pct_change().dropna()
    print('\n', returns) 
    

    if isinstance(method_config['rebalancing'], int):
        rebalance_days = returns.iloc[method_config['window_size']::method_config['rebalancing']].index

    # if method_config['rebalancing'] == 'last_available_day_of_month':
    #     rebalance_days = returns.groupby([returns.index.year, returns.index.month]).tail(1).index

    # if method_config['rebalancing'] == 'quarterly_last_available_day_of_month':
    #     rebalance_days = returns.groupby([returns.index.year, returns.index.month]).tail(1).index
    #     rebalance_days = [x for x in rebalance_days if float(x.month) % 3.0 == 0 ] 

    returns_dates = returns.index
    print('\n', returns_dates)
    rebalance_rows = [returns_dates.get_loc(x) for x in rebalance_days]
    # rebalance_rows = [returns_dates.get_loc(x) for x in rebalance_days if returns_dates.get_loc(x) > method_config['window_size']]
    print('\n', rebalance_rows)


    # Weights calculation {#25f,4}
    weights = get_portfolio_weights(returns = returns, 
                                    index_ = rebalance_rows,
                                    full_conf = conf
                                    )


    rebalance_rows = [x - method_config['window_size'] for x in rebalance_rows]
    
    weights.index = rebalance_rows
    print('\n', weights)



    #  Backtrader run{#05a,10}
    results, dd, cagr, sharpe = backtest(assets_prices,
                                AssetAllocation,
                                assets, 
                                weights,
                                start=conf['start_data'],
                                end=conf['end_data'],
                                backtrader_config = backtrader_config)

    print(f'\ndd: {dd}, cagr: {cagr}, sharpe: {sharpe}')

    strat = results[0]
    portfolio_stats = strat.analyzers.getbyname('PyFolio')
    returns, positions, transactions, gross_lev = portfolio_stats.get_pf_items()
    returns.index = returns.index.tz_convert(None)
    if conf['save_report'] == True:
        quantstats.reports.html(returns, output=f'output-scratch/{conf["dataset_type"]}-{conf["start_data"][:4]}-{conf["end_data"][:4]}-{method_config["method"]}.html', title=f'{conf["dataset_type"]}-{conf["start_data"][:4]}-{conf["end_data"][:4]}-{method_config["method"]}')
        print(f'file saved: output-scratch/{conf["dataset_type"]}-{conf["start_data"][:4]}-{conf["end_data"][:4]}-{method_config["method"]}.html')