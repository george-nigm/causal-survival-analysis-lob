import yaml
import pandas as pd 
import numpy as np
import backtrader as bt
import riskfolio as rp
import quantstats
import pickle 
from portfolio_opt_methods import equal_weights, random_weights, mv_portfolio, grangers_causation_matrix_portfolio, PCMCI_wrapped
import os
import re
import datetime
import wandb
from matplotlib import pyplot as plt

with open('conf/analysis.yaml', 'r') as file:
    conf = yaml.safe_load(file)

with open('conf/analysis_wandb_opt.yaml', 'r') as file:
    conf_wandb = yaml.safe_load(file)

def create_exp_folder(output_folder, special_label=''):
    os.makedirs(output_folder, exist_ok=True)
    dirs = [d for d in os.listdir(output_folder) if os.path.isdir(os.path.join(output_folder, d))]
    exp_nums = [int(re.match(r'(\d+)', d).group(1)) for d in dirs if re.match(r'(\d+)', d)]
    if exp_nums:
        experiment_num = max(exp_nums) + 1
    else:
        experiment_num = 1
    now = datetime.datetime.now()
    date_time = now.strftime("%d.%m | %H-%M |")
    directory = f"{experiment_num}. {date_time} {special_label}"
    full_path = os.path.join(output_folder, directory)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    return full_path

def get_portfolio_weights(returns, index_, full_conf):

    weights = pd.DataFrame([])
    port_cov = {}

    for num, i in enumerate(index_):
        Y = returns.iloc[i-full_conf['portfolio_method_config']['window_size']:i,:] # taking last window_size days

        if full_conf['portfolio_method_config']['method'] == 'random_weights':
            w = random_weights(Y)
            port_cov[f'{Y.index[0].date()}-{Y.index[-1].date()}'] = None

        if full_conf['portfolio_method_config']['method'] == 'equal_weights':
            w = equal_weights(Y)
            port_cov[f'{Y.index[0].date()}-{Y.index[-1].date()}'] = None

        if full_conf['portfolio_method_config']['method'] == 'MV':            
            w , port_cov_window = mv_portfolio(Y, full_conf['mv_config'])
            port_cov[f'{Y.index[0].date()}-{Y.index[-1].date()}'] = port_cov_window
        
        if full_conf['portfolio_method_config']['method'] == 'grangers_causation_matrix':            
            w, port_cov_window = grangers_causation_matrix_portfolio(Y, full_conf['mv_config'], full_conf['grangers_causation_matrix_config'])
            port_cov[f'{Y.index[0].date()}-{Y.index[-1].date()}'] = port_cov_window

        if full_conf['portfolio_method_config']['method'] == 'PCMCI_wrapped':            
            w, port_cov_window = PCMCI_wrapped(Y, full_conf['mv_config'], full_conf['PCMCI_wrapped_causation_matrix_config'])
            port_cov[f'{Y.index[0].date()}-{Y.index[-1].date()}'] = port_cov_window
            
        if w is None:
            w = weights.tail(1).T
        weights = pd.concat([weights, w.T], axis = 0)
        

    return weights, port_cov


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
            
            reb_weights = []
            for i in self.params.assets:
                w = self.params.weights.loc[self.counter, i]
                reb_weights.append(round(w,3))
                self.order_target_percent(getattr(self, i), target=w)
            
            cash = self.broker.getcash()
            value = self.broker.getvalue()
            invested = value - cash

            print(f'{self.counter} day.\nPortfolio Value: %.2f' % value)

        self.counter += 1
    
    # def notify_order(self, order):
    #     if order.status in [order.Submitted, order.Accepted]:
    #         # Order has been submitted/accepted - no action required
    #         return

    #     # Check if an order has been completed
    #     if order.status in [order.Completed]:
    #         if order.isbuy():
    #             self.log('BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f, Stock: %s' %
    #                     (order.executed.price,
    #                     order.executed.value,
    #                     order.executed.comm,
    #                     order.data._name))

    #         else:  # Sell
    #             self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f, Stock: %s' %
    #                     (order.executed.price,
    #                     order.executed.value,
    #                     order.executed.comm,
    #                     order.data._name))

    #     elif order.status in [order.Canceled, order.Margin, order.Rejected]:
    #         self.log('Order Canceled/Margin/Rejected for Stock: %s' % order.data._name)

    # def log(self, txt, dt=None):
    #     dt = dt or self.datas[0].datetime.date(0)
    #     print('%s, %s' % (dt.isoformat(), txt))


class CloseOnly(bt.feeds.PandasData):
    lines = ('close',)
    params = (
        ('open', 'close'),
        ('high', 'close'),
        ('low', 'close'),
        ('volume', -1),
        ('openinterest', -1),
    )


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

def wandb_train():    
    print() 

    # Specify the hyperparameter to be tuned along with an initial value
    configs = {'rebalancing': 22,
               'window_size': 252}

    run = wandb.init(project="geo-george-hyperparameter-sweeps-partI", config=configs)
    config = run.config
    
    conf['portfolio_method_config']['rebalancing'] = config.rebalancing
    conf['portfolio_method_config']['window_size'] = config.window_size

    

    method_config = conf['portfolio_method_config']
    backtrader_config = conf['backtrader_config'] 
    
    if conf['dataset_type'] == 'pinnacle':
        dataset_config = conf['pinnacle_config']
        prices = pd.read_pickle(dataset_config['location'])
        assets = dataset_config['assets']

    if conf['dataset_type'] == 'CPT':
        dataset_config = conf['CPT_config']
        prices = pd.read_pickle(dataset_config['location'])
        assets = dataset_config['assets']

    prices = prices[prices.index >= conf['start_data']]
    prices = prices[prices.index <= conf['end_data']]

    assets_prices = []
    excluded_tickers = []


    if conf['dataset_type'] == 'pinnacle':
        for i in assets:
            prices_ = prices.drop(columns='Open Interest').loc[:, (slice(None), i)].dropna()
            prices_.columns = prices_.columns.droplevel(1)     
            if len(prices_.columns) == 5: # check fullfilness for backtesting
                prices_.columns = ['Open', 'High', 'Low', 'Close', 'Volume']      
                assets_prices.append(bt.feeds.PandasData(dataname=prices_.iloc[method_config['window_size']:], name=i, plot=False))
            else:
                excluded_tickers.append(i)

    if conf['dataset_type'] == 'CPT':
        for i in assets:
            prices_ = prices.loc[:, i].dropna()
            intermediate = prices_.iloc[method_config['window_size']:]
            prices_ = pd.DataFrame(intermediate.values, columns=['Close'])
            prices_.index = intermediate.index
            assets_prices.append(CloseOnly(dataname=prices_,  name=i))
            

    assets = [x for x in assets if x not in excluded_tickers]
    if conf['dataset_type'] == 'pinnacle':
        prices = prices[[col for col in prices.columns if col[1] in assets]]
        data = prices.loc[:, ('Close', slice(None))]
        data.columns = data.columns.droplevel(0)
        data.columns = assets
    
    if conf['dataset_type'] == 'CPT':
        prices = prices[[col for col in prices.columns if col in assets]]
        data = prices

    returns = data.pct_change().dropna()
    
    if isinstance(method_config['rebalancing'], int):
        rebalance_days = returns.iloc[method_config['window_size']::method_config['rebalancing']].index

    # if method_config['rebalancing'] == 'last_available_day_of_month':
    #     rebalance_days = returns.groupby([returns.index.year, returns.index.month]).tail(1).index

    # if method_config['rebalancing'] == 'quarterly_last_available_day_of_month':
    #     rebalance_days = returns.groupby([returns.index.year, returns.index.month]).tail(1).index
    #     rebalance_days = [x for x in rebalance_days if float(x.month) % 3.0 == 0 ] 

    returns_dates = returns.index
    rebalance_rows = [returns_dates.get_loc(x) for x in rebalance_days]
    # rebalance_rows = [returns_dates.get_loc(x) for x in rebalance_days if returns_dates.get_loc(x) > method_config['window_size']]
    print('\n', len(rebalance_rows), rebalance_rows)

    # Weights calculation {#25f,4}
    weights, port_cov = get_portfolio_weights(returns = returns, 
                                    index_ = rebalance_rows,
                                    full_conf = conf
                                    )

    rebalance_rows = [x - method_config['window_size'] for x in rebalance_rows]
    weights.index = rebalance_rows

    #  Backtrader run{#05a,10}
    results, dd, cagr, sharpe = backtest(assets_prices,
                                AssetAllocation,
                                assets, 
                                weights,
                                start=conf['start_data'],
                                end=conf['end_data'],
                                backtrader_config = backtrader_config)
    
    # fixing sharp
    strat = results[0]
    portfolio_stats = strat.analyzers.getbyname('PyFolio')
    returns_bt, positions, transactions, gross_lev = portfolio_stats.get_pf_items()
    returns_bt.index = returns_bt.index.tz_convert(None)
    returns_bt = returns_bt[1:]
    risk_free_rate = 0.0
    sharpe_ratio = np.sqrt(252) * (returns_bt.mean() - risk_free_rate) / returns_bt.std()
    sharpe = sharpe_ratio

    print(f'\ndd: {dd}, cagr: {cagr}, sharpe: {sharpe}')

    strat = results[0]
    portfolio_stats = strat.analyzers.getbyname('PyFolio')
    returns_bt, positions, transactions, gross_lev = portfolio_stats.get_pf_items()
    returns_bt.index = returns_bt.index.tz_convert(None)


    # log metrics to wandb
    wandb.log({"rebalancing": config.rebalancing,
               "window_size": config.window_size,
               "dd": dd, "cagr": cagr, "sharpe": sharpe
               })
    arr = [i for i in range(config.window_size, len(returns_bt.values)+config.window_size)]
    # print(arr)
    data = [[x, y] for (x, y) in zip(arr, returns_bt.values)]
    table = wandb.Table(data=data, columns = ["returns_bt_index", "returns_bt_values"])
    wandb.log({"my_lineplot_id" : wandb.plot.line(table, "returns_bt_index", 
            "returns_bt_values", stroke=None, title="Average Precision")})


    # if conf['save_report'] == True:

    #     formatted_string = "{}-{}-{}-{}".format(conf["dataset_type"], conf["start_data"][:4], conf["end_data"][:4], method_config["method"])
    #     exp_folder = create_exp_folder(output_folder = "output-scratch", special_label=formatted_string)
        
    #     quantstats.reports.html(returns_bt, output=f'{exp_folder}/report.html', title=f'{formatted_string}')        

    #     returns.to_csv(f'{exp_folder}/returns.csv')

    #     with open(f'{exp_folder}/port_cov.pkl', 'wb') as f:
    #         pickle.dump(port_cov, f)

    #     print(f'files saved: {exp_folder}/report.html')


if __name__ == "__main__":
    import wandb
    import random
    # wandb.login()

    sweep_id = wandb.sweep(conf_wandb) 
    wandb.agent(sweep_id, function=wandb_train)

    
    # wandb.init(project="my-awesome-project",
    #            config={
    #            "learning_rate": 0.02,
    #            "architecture": "CNN",
    #            "dataset": "CIFAR-100",
    #            "epochs": 10})

    # # simulate training
    # epochs = 10
    # offset = random.random() / 5
    # for epoch in range(2, epochs):
    #     acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    #     loss = 2 ** -epoch + random.random() / epoch + offset
        
    #     # log metrics to wandb
    #     wandb.log({"acc": acc, "loss": loss})
        
    # # [optional] finish the wandb run, necessary in notebooks
    # wandb.finish()

