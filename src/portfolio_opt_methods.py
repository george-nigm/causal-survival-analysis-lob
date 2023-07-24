import numpy as np
import pandas as pd
import riskfolio as rp
from statsmodels.tsa.stattools import grangercausalitytests


def make_positive_definite(cov_matrix, epsilon=1e-2):
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)

    if np.any(eigvals < epsilon):
        eigvals[eigvals < epsilon] = epsilon        
        positive_definite_cov = np.dot(eigvecs, np.dot(np.diag(eigvals), eigvecs.T))
    else:
        positive_definite_cov = cov_matrix

    return positive_definite_cov

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

    print(f'\n{returns.index[0].date()}')
    print(f'{returns.index[-1].date()}')
                
    # # Add portfolio constraints
    # port.ainequality = constraints[0] # A
    # port.binequality = constraints[1] # B

    method_mu = mv_conf['method_mu']
    method_cov = mv_conf['method_cov']

    port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=mv_conf['d_lib_const'])

    # port.cov = make_positive_definite(port.cov)

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
        print(f'weights successfully calculated for {returns.index[0].date()}-{returns.index[-1].date()}')
    except:               
        w = None
        print(f'weights index unsuccessful for {returns.index[0].date()}-{returns.index[-1].date()}. weights.tail(1).T used.\n')
    return w, port.cov
        
# -------------------------------------------------------------- #
# My DPhil Functions
# -------------------------------------------------------------- #

def grangers_causation_matrix(data, variables, maxlag, test='ssr_chi2test', verbose=False):    
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

def grangers_causation_matrix_portfolio(returns, grangers_causation_matrix_config, constraints = [None, None]):

    port = rp.Portfolio(returns=returns)

    print(f'\n{returns.index[0].date()}')
    print(f'{returns.index[-1].date()}')
                
    # # Add portfolio constraints
    # port.ainequality = constraints[0] # A
    # port.binequality = constraints[1] # B

    method_mu = grangers_causation_matrix_config['method_mu']
    method_cov = grangers_causation_matrix_config['method_cov']

    port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=grangers_causation_matrix_config['d_lib_const'])

    # -------------------------------------------------------------- #
    # DPhil
    port.cov = grangers_causation_matrix(returns, variables = returns.columns, maxlag = grangers_causation_matrix_config['maxlag'], test = grangers_causation_matrix_config['test'])   
    port.cov.reset_index(drop=True, inplace=True)
    # -------------------------------------------------------------- #
        
    # port.solvers = ['MOSEK']
    # port.alpha = grangers_causation_matrix_config['alpha']
    model= grangers_causation_matrix_config['model'] # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
    rm = grangers_causation_matrix_config['rm'] # Risk measure used, this time will be variance
    obj = grangers_causation_matrix_config['obj'] # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
    hist = grangers_causation_matrix_config['hist'] # Use historical scenarios for risk measures that depend on scenarios
    rf = grangers_causation_matrix_config['rf'] # Risk free rate
    l = grangers_causation_matrix_config['l'] # Risk aversion factor, only useful when obj is 'Utility'
    try:
        w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
        print(f'\nweights successfully calculated')
    except:               
        w = None
        print(f'\nweights index unsuccessful. weights.tail(1).T used.\n')
    return w, port.cov