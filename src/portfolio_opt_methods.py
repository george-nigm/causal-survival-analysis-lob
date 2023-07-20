import numpy as np
import pandas as pd
import riskfolio as rp

def equal_weights(returns):
    n_assets = returns.shape[1]
    w = np.full(n_assets, 1.0 / n_assets)       
    w = pd.DataFrame(w, index=returns.columns, columns = ['weights'])
    return w
            
def random_weights(returns):
    n_assets = returns.shape[1]
    w = np.random.random(n_assets)
    w /= np.sum(w)
    w = pd.DataFrame(w, index=returns.columns, columns = ['weights'])
    return w


def mv_portfolio(returns, mv_conf, constraints = [None, None]):
    port = rp.Portfolio(returns=returns)
                
    # # Add portfolio constraints
    # port.ainequality = constraints[0] # A
    # port.binequality = constraints[1] # B

    method_mu = mv_conf['method_mu']
    method_cov = mv_conf['method_cov']

    port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=mv_conf['d_lib_const'])

    # -------------------------------------------------------------- #
    # DPhil
    # port.cov = grangers_causation_matrix(Y, variables = Y.columns)   
    # port.cov.reset_index(drop=True, inplace=True)
    # -------------------------------------------------------------- #
        
    # port.solvers = ['MOSEK']
    # port.alpha = mv_conf['alpha']
    model= mv_conf['model'] # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
    rm = mv_conf['rm'] # Risk measure used, this time will be variance
    obj = mv_conf['obj'] # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
    hist = mv_conf['hist'] # Use historical scenarios for risk measures that depend on scenarios
    rf = mv_conf['rf'] # Risk free rate
    l = mv_conf['l'] # Risk aversion factor, only useful when obj is 'Utility'
    try:
        w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
        print(f'\nweights successfully calculated')
    except:               
        w = None
        print(f'\nweights index unsuccessful. weights.tail(1).T used.\n')
    return w
        