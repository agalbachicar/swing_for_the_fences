import logging
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats as scipy_stats

import quantstats as qs

import pickle

from load_data import (
    load_btc_data, load_btc_csv, load_glassnode_json, load_glassnode_csv,
    load_alternative_me_csv, load_gtrends_csv
)
from btc_strategy import (
    FeatureBuilder, TradeStrategy, FundamentalStrategyPipelineOptimizer,
    MlModelStrategyPipelineOptimizer, StrategyPipelineFixedModelRF,
    StrategyPipelineFixedModelBagging, StrategyPipelinePurgedKFoldRFModel,
    StrategyPipelinePurgedKFoldBaggingModel, StrategyPipelineFixedModelBoosting,
    StrategyPipelinePurgedKFoldBoostingModel,
    run_feature_engineering_for_glassnode_features
)
from bet_sizing import getSignal
from sharpe_ratio_stats import (
    estimated_sharpe_ratio, ann_estimated_sharpe_ratio,
    estimated_sharpe_ratio_stdev, probabilistic_sharpe_ratio,
    min_track_record_length, num_independent_trials,
    expected_maximum_sr, deflated_sharpe_ratio
)

CPUS = 7

MODEL_NAME = 'Bagging_Optim_Feat_Imp'

PLOT_RETURN_HIST = True

#-------------------------------------------------------------------------------

def configure_logger(log_path=None):
    log_formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(funcName)s:%(lineno)d][%(levelname)-5.5s]  %(message)s")
    logger = logging.getLogger()

    if log_path:
        file_handler = logging.FileHandler("{}.log".format(log_path))
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    logger.setLevel(logging.DEBUG)
#-------------------------------------------------------------------------------

def plot_returns(returns, strategy_name):
    plt.hist(returns, 100, facecolor='blue', alpha=0.7, log=True)
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.title('Histogram of returns for {}'.format(strategy_name))
    plt.grid(True)
    plt.show()

#-------------------------------------------------------------------------------
def plot_psr(base_returns, st_returns):
    psr_base = np.asarray([probabilistic_sharpe_ratio(returns=base_returns, sr_benchmark=float(i)/100.) for i in range(0, 101)])
    psr_st = np.asarray([probabilistic_sharpe_ratio(returns=st_returns, sr_benchmark=float(i)/100.) for i in range(0, 101)])

    psr_base_odds = np.divide(psr_base, 1. - psr_base)
    psr_st_odds = np.divide(psr_st, 1. - psr_st)
    psr_base_odds_log = np.log10(psr_base_odds)
    psr_st_odds_log = np.log10(psr_st_odds)

    x = np.asarray([i / 100. for i in range(0, 101)])

    plt.plot(x, psr_base_odds_log, color='blue', label='Odds ratio PSR buy and hold')
    plt.plot(x, psr_st_odds_log, color='red', label='Odds ratio PSR strategy under test')
    plt.legend()
    plt.xlabel('Target Sharpe Ratio')
    plt.ylabel('Odds ratio of Probabilistic Sharpe Ratio')
    plt.title('Odds ratio of Probabilistic Sharpe Ratio')
    plt.grid()
    plt.show()


#-------------------------------------------------------------------------------
def plot_value(df):
    df.plot(y='Value', color='blue')
    plt.xlabel('Timestampt')
    plt.ylabel('Portfolio value [USD]')
    plt.title('Portfolio valuation')
    plt.grid()
    plt.show()

#-------------------------------------------------------------------------------

def strategy_report(st_returns, st_name, underlying_asset_returns=pd.Series()):
    print('-------------------------------------------------------------------')
    print('-------------------------------------------------------------------')
    print('Strategy: {}'.format(st_name))
    print('Sharpe: {}'.format(qs.stats.sharpe(st_returns, periods=365, annualize=True, trading_year_days=365)))
    print('Sortino: {}'.format(qs.stats.sortino(st_returns)))
    print('Adjusted Sortino: {}'.format(qs.stats.adjusted_sortino(st_returns)))
    print('Win loss ratio: {}'.format(qs.stats.win_loss_ratio(st_returns)))
    print('Win rate: {}'.format(qs.stats.win_rate(st_returns)))
    print('Avg loss: {}'.format(qs.stats.avg_loss(st_returns)))
    print('Avg win: {}'.format(qs.stats.avg_win(st_returns)))
    print('Avg return: {}'.format(qs.stats.avg_return(st_returns)))
    print('Volatility: {}'.format(qs.stats.volatility(st_returns, periods=st_returns.shape[0], annualize=True, trading_year_days=365)))
    print('Value at risk: {}'.format(qs.stats.value_at_risk(st_returns, sigma=1, confidence=0.95)))
    if not underlying_asset_returns.empty:
        df = pd.merge(st_returns, underlying_asset_returns, how='inner', left_index=True, right_index=True)
        print('Correlation to underlying: {}'.format(df.corr()))
    print('-------------------------------------------------------------------')
    print('Sharpe: {}'.format(estimated_sharpe_ratio(st_returns)))
    print('Annualized Sharpe: {}'.format(ann_estimated_sharpe_ratio(st_returns, periods=365)))
    print('STDDEV Sharpe: {}'.format(estimated_sharpe_ratio_stdev(returns=st_returns)))
    psrs = [probabilistic_sharpe_ratio(returns=st_returns, sr_benchmark=float(i)/100.) for i in range(0, 101)]
    print('PSR: {}'.format(psrs))
    print('-------------------------------------------------------------------')
    print('Mean return: {}'.format(qs.stats.avg_return(st_returns)))
    print('Variance of returns: {}'.format(qs.stats.volatility(st_returns, annualize=False) ** 2))
    print('Skewness of returns: {}'.format(scipy_stats.skew(st_returns, nan_policy='omit')))
    print('Kurtosis of returns: {}'.format(scipy_stats.kurtosis(st_returns, nan_policy='omit')))
    print('-------------------------------------------------------------------')

#-------------------------------------------------------------------------------

def daily_return(prices):
    df0 = prices.index.searchsorted(prices.index-pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(prices.index[df0-1],
                    index=prices.index[prices.shape[0]-df0.shape[0]:])
    df0 = prices.loc[df0.index] / prices.loc[df0.values].values-1  # Daily returns
    return df0
#-------------------------------------------------------------------------------

def run_strategy(df, params):
    i=0
    last_portfolio_value = 1
    tx_costs = 0.
    
    for index,row in df.iterrows():
        if row['bets_usd'] > 0:
            bet = row['bets_usd']
            initial_price = row['Close']
            price_i = row['Close']
            # Will track the date index
            j = 1
            
            # Pays to enter the position
            df.loc[df.index[i], 'Value'] = df.loc[df.index[i], 'Value'] - row['bets_usd'] * params['buy_fee']
            tx_costs = tx_costs + np.abs(row['bets_usd'] * params['buy_fee'])

            # After t1_length without touching any of the barrier, the position
            # is dismantled. If it if happens before t1_length, then we do it earlier.
            while price_i < (initial_price * (1 + params['pt'])) and price_i > (initial_price * (1 - params['sl'])) and (j < params['t1_length'] + 1): #Cuando se cumple la condición, salimos de la posición
                price_i = df.loc[df.index[i+j], 'Close']
                df.loc[df.index[i+j], 'Value'] = df.loc[df.index[i], 'Value'] + row['bets_usd'] * (price_i / initial_price - 1.)
                last_portfolio_value = df.loc[df.index[i+j], 'Value']
                j = j + 1       
            
            # Fee to leave the position.
            df.loc[df.index[i+j-1]:, 'Value'] = last_portfolio_value - row['bets_usd'] * (price_i / initial_price) * params['sell_fee']
            tx_costs = tx_costs + np.abs(row['bets_usd'] * (price_i / initial_price) * params['sell_fee'])
        
        if row['bets_usd'] < 0:
            bet = row['bets_usd']
            initial_price = row['Close']
            price_i = row['Close']
            j = 1 # Will track the date index
            
            # Shorting BTC. Two fees needs to be paids, the sell + the loan fee.
            df.loc[df.index[i], 'Value'] = df.loc[df.index[i], 'Value'] - np.abs(row['bets_usd']) * params['sell_fee'] - np.abs(row['bets_usd']) * params['short_fee']
            tx_costs = tx_costs + np.abs(row['bets_usd']) * params['sell_fee'] + np.abs(row['bets_usd']) * params['short_fee']

            # Same as before, waiting for any barrier to be touched.
            while price_i < (initial_price * (1 + params['pt'])) and price_i > (initial_price * (1 - params['sl'])) and (j < params['t1_length'] + 1): #Cuando se cumple la condición, salimos de la posición
                price_i = df.loc[df.index[i+j], 'Close']
                df.loc[df.index[i+j], 'Value'] = df.loc[df.index[i], 'Value'] + row['bets_usd'] * (price_i / initial_price - 1.)
                last_portfolio_value = df.loc[df.index[i+j], 'Value']
                j = j + 1   
            # We pay the commission to leave the position.
            df.loc[df.index[i+j-1]:, 'Value'] = last_portfolio_value - np.abs(row['bets_usd']) * (price_i / initial_price) * params['buy_fee']
            tx_costs = tx_costs + np.abs(row['bets_usd']) * (price_i / initial_price) * params['buy_fee']

        # If there is no money, we stop.
        if last_portfolio_value <= 0.:
            break
        i = i + 1 # Moves forward with the next row.

    return df, tx_costs

def build_portfolio_df(strategy, params):
    # Obtain all the probabilities and the predictions
    if not strategy.params['features']:
        y_pred_prob = strategy.model.predict_proba(strategy.Xy.copy()[strategy.Xy.columns.difference(['bin', 't1', 'Close'])])[:, 1]
        y_pred = strategy.model.predict(strategy.Xy.copy()[strategy.Xy.columns.difference(['bin', 't1', 'Close'])])
    else:
        y_pred_prob = strategy.model.predict_proba(strategy.Xy.copy().loc[:, strategy.params['features']])[:, 1]
        y_pred = strategy.model.predict(strategy.Xy.copy().loc[:, strategy.params['features']])
    y_prob_df = pd.Series(data=y_pred_prob, index=strategy.Xy.index)
    y_pred_df = pd.Series(data=y_pred, index=strategy.Xy.index)

    # Computes the bet sizes
    bets = getSignal(strategy.ts.tbe, 1. / params['steps'], y_prob_df, y_pred_df, params['num_classes'], params['cpus'])
    bets_df = bets.to_frame()
    bets_df.index.name = 'Timestamp'
    bets_df.rename(columns={0:'bets'}, inplace=True)

    # Generate a new df
    df = strategy.ts.df['Close'].copy()
    df = pd.merge(df, bets_df, how='left', left_index=True, right_index=True)
    df['bets_usd'] = df['bets'] * params['budget']
    df['Value'] = params['budget']
    df.fillna(0, inplace=True)
    

    df, tx_costs = run_strategy(df, params)
    df['rets'] = daily_return(df['Value'])

    return df, tx_costs

#-------------------------------------------------------------------------------
# configure_logger()


filename = '../datasets/{}.pickle'.format(MODEL_NAME)
optimizer = pickle.load(open(filename, 'rb'))

strategy = optimizer.strategies[0]

params = {
    'cpus': CPUS,
    'num_classes': 2,
    'steps': 5,
    'budget': 1000000.,
    'buy_fee': 0.001, # https://www.binance.com/en/fee/schedule
    'sell_fee': 0.001, # https://www.binance.com/en/fee/schedule
    'short_fee':  np.power(1. + 0.05, strategy.params['t1_length'] / 365.) - 1., # https://defirate.com/lend/
    'pt': strategy.params['pt'],
    'sl': strategy.params['sl'],
    't1_length': strategy.params['t1_length'],
    'btc_initial_price': strategy.Xy.loc[strategy.Xy.index[0], 'Close'],
}


(df, tx_costs) = build_portfolio_df(strategy, params)
base_returns = daily_return(strategy.ts.df['Close'])

strategy_report(base_returns, 'B&H BTC')

strategy_report(df['rets'].fillna(0.), 'Tesis!', underlying_asset_returns=base_returns)
return_over_tx_costs = (df.loc[df.index[-1], 'Value'] - params['budget']) / tx_costs
print('Return over tx costs: {}'.format((df.loc[df.index[-1], 'Value'] - params['budget']) / tx_costs))
print('Average uniqueness: {}'.format(strategy.ts.average_uniqueness))
print('Labels: {}'.format(strategy.ts.t_events.value_counts()))
print('Metalabels: {}'.format(strategy.ts.labels['bin'].value_counts()))

lbl_df = strategy.ts.t_events.copy(deep=True).to_frame()
lbl_df.rename(columns={0: 'label'}, inplace=True)
lbl_df['metalabel'] = strategy.ts.labels['bin']
print('Contention table: \n{}'.format(pd.crosstab(lbl_df.label, lbl_df.metalabel)))


plot_psr(base_returns, df['rets'])
plot_value(df)

if PLOT_RETURN_HIST:
    plot_returns(base_returns, 'buy and hold strategy')
    plot_returns(df['rets'], 'strategy under test')

# -------------------------------------------------------------------                                                                                   
# -------------------------------------------------------------------                                                                                   
# Strategy: Tesis! Bagging                                                                                                          
# Sharpe: 0.7901972898996218                                                                                                                            
# Sortino: 12.499238365984796                                                                                                                           
# Adjusted Sortino: 8.83829620825491                                                                                                                    
# Win loss ratio: 13.403411723196182                                                                                                                    
# Win rate: 0.5476190476190477                                                                                                                          
# Avg loss: -1004.3747517126245                                                                                                                         
# Avg win: 13462.048321587246                                                                                                                           
# Avg return: 6917.714074142066
# Volatility: 33935.59792899352
# Value at risk: -2833.2860307735104

# -------------------------------------------------------------------                                                                                   
# -------------------------------------------------------------------                                                                                   
# Strategy: Tesis! RF                                                                                                                                    
# Sharpe: 0.5331993542290487                                                                                                                            
# Sortino: 40.62166813835788                                                                                                                            
# Adjusted Sortino: 28.723857003742374                                                                                                                  
# Win loss ratio: 58.45766557584883                                                                                                                     
# Win rate: 0.5384615384615384                                                                                                                          
# Avg loss: -300.0                                                                                                                                      
# Avg win: 17537.29967275465                                                                                                                            
# Avg return: 9304.699823790965
# Volatility: 20938.00253814203
# Value at risk: -1765.8582955438108

# -------------------------------------------------------------------                                                                                   
# -------------------------------------------------------------------                                                                                   
# Strategy: Tesis! Boosting                                                                                                                                      
# Sharpe: 0.6199248250793591                                                                                                                            
# Sortino: 41.961986670248905                                                                                                                           
# Adjusted Sortino: 29.671605326592513                                                                                                                  
# Win loss ratio: 54.9517764968014                                                                                                                      
# Win rate: 0.5384615384615384                                                                                                                          
# Avg loss: -466.6666666666667                                                                                                                          
# Avg win: 25644.162365173986                                                                                                                           
# Avg return: 13593.010504324455
# Volatility: 26308.683734492122
# Value at risk: -2211.284855388429

# -------------------------------------------------------------------
# -------------------------------------------------------------------
# Strategy: B&H BTC
# Sharpe: 0.6689333503235667
# Sortino: 0.822262647432692
# Adjusted Sortino: 0.5814274939160597
# Win loss ratio: 0.9869764955295283
# Win rate: 0.546458141674333
# Avg loss: -103.37643002028393
# Avg win: 102.03010662177336
# Avg return: 8.869644280895432
# Volatility: 4802.854810620638
# Value at risk: -404.7023768361658
