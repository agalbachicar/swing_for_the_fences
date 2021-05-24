import logging
import sys

import numpy as np
import pandas as pd

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
    run_feature_engineering_for_glassnode_features,
    MlModelOptimStrategyFeatImportance
)

# File to load
DEFAULT_BTC_PERIODICITY_DATA = 'daily'

# File paths to multiple datasets
FEATURE_FILE_PATH = '../datasets/processed/btc_features_{}.csv'
T1_FILE_PATH = '../datasets/processed/t1_{}.csv'
T_EVENTS_FILE_PATH = '../datasets/processed/t_events_{}.csv'
TARGET_FILE_PATH = '../datasets/processed/target_{}.csv'
TRIPLE_BARRIER_EVENTS_FILE_PATH = '../datasets/processed/triple_barrier_events_{}.csv'
LABELS_FILE_PATH = '../datasets/processed/labels_{}.csv'
NUM_CO_EVENTS_FILE_PATH = '../datasets/processed/numCoEvents_{}.csv'
CEW_FILE_PATH = '../datasets/processed/cew_{}.csv'
GLASSNODE_CSV_FILE_PATH = '../datasets/glassnode/csv/dataset.csv'

# Fundamental features parameters.
RSI_WINDOW_LENGTHS = [5, 10, 15, 30]            # RSI
AUTOCORR_LAGS = [1, 3, 5]                       # Auto correlation.
AUTOCORR_WINDOW_LENGTH = 45
VOLATILITY_WINDOW_LENGTHS = [5, 10, 15, 30]     # Volatility
MIN_SL_SADF = 50                                # SADF
MAX_SL_SADF = 300
CONSTANTS_SADF = ['nt', 'ct', 'ctt']
LAGS_SADF = [1, 2, 3]

# Fundamental trading strategy parameters.
DEFAULT_FAST_WIN_LENGTH = 2
DEFAULT_SLOW_WIN_LENGTH = 5
DEFAULT_T1_LENGTH = 6
DEFAULT_PT_SL = [0.015, 0.005]
DEFAULT_VOLATILITY_LENGTH = 45
DEFAULT_MIN_RET = 0.01
DEFAULT_CPUS = 7

# Number of samples to find the best trade strategy parameters
FUNDAMENTAL_STRATEGY_OPTIM_SAMPLES = 100
ML_STRATEGY_OPTIM_SAMPLES = 100

# Secondary model to use
MODEL_NAME = 'Bagging_Optim_Feat_Imp'

# Feature engineering task selection
PROCESS_GLASSNODE_FEATURES = False
PROCESS_FINANCIAL_FEATURES = False
# Primary and secondary model task selection
PROCESS_PRIMARY_MODEL_OPTIMIZATION = False
PRIMARY_MODEL_SELECTION_METRIC = 'neg_log_loss'
PROCESS_SECONDARY_MODEL_OPTIMIZATION = False
SECONDARY_MODEL_SELECTION_METRIC = 'neg_log_loss'

RUN_FEAT_IMPORTANCE=False

# Saves the optimized secondary model
SAVE_OPTIMIZED_SECONDARY_MODEL = True


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


def save_df_to_csv(df, file_path):
    df.to_csv(file_path, index_label='Timestamp', date_format='%s')


def process_glassnode_info():
    # Loads data from the json files and then moves it to a csv file with all
    # the data together.
    save_df_to_csv(load_glassnode_json(), GLASSNODE_CSV_FILE_PATH)
    logging.debug('Saved into {} Glassnode features.'.format(
        GLASSNODE_CSV_FILE_PATH))
    # Loads the CSV file into a data frame and applies transformations.
    run_feature_engineering_for_glassnode_features(
        load_glassnode_csv(), GLASSNODE_CSV_FILE_PATH)
    logging.debug('Processed Glassnode features and saved into {}.'.format(
        GLASSNODE_CSV_FILE_PATH))


def load_glassnode_features_based_on_args(df, glassnode_features):
    if glassnode_features == 'no' or glassnode_features not in {'no', 'only', 'combined'}:
        logging.debug('PASO POR NO')
        return df
    glassnode_df = load_glassnode_csv()
    if glassnode_features == 'only':
        logging.debug('PASO POR ONLY')
        df = df[['CloseFFD']]
    df = pd.merge_asof(df, glassnode_df, left_index=True,
                       right_index=True, tolerance=pd.Timedelta("11h"))
    return df


def load_sentiment_features_based_on_args(df, sentiment_features):
    if sentiment_features == 'no' or sentiment_features not in {'no', 'only', 'combined'}:
        return df
    if sentiment_features == 'only':
        df = df[['CloseFFD']]

    gtrends_df = load_gtrends_csv()
    df = pd.merge_asof(df, gtrends_df, left_index=True,
                       right_index=True, tolerance=pd.Timedelta("11h"))
    df[gtrends_df.columns] = df[gtrends_df.columns].interpolate()
    alternative_me_df = load_alternative_me_csv()
    df = pd.merge_asof(df, alternative_me_df, left_index=True,
                       right_index=True, tolerance=pd.Timedelta("11h"))
    return df


def process_features(periodicity):
    df = load_btc_data(periodicity)
    logging.debug('Loaded data with {} periodicity'.format(periodicity))

    feature_builder = FeatureBuilder(df)
    logging.debug('Adding RSI')
    feature_builder.add_rsi(RSI_WINDOW_LENGTHS)
    logging.debug('Adding log returns')
    feature_builder.add_log_ret()
    logging.debug('Adding autocorrelation')
    feature_builder.add_autocorr(AUTOCORR_LAGS, AUTOCORR_WINDOW_LENGTH)
    logging.debug('Adding volatility')
    feature_builder.add_volatility(VOLATILITY_WINDOW_LENGTHS)
    logging.debug('Adding sadf')
    feature_builder.add_sadf(MIN_SL_SADF, MAX_SL_SADF,
                             CONSTANTS_SADF, LAGS_SADF)
    logging.debug('Adding fractionally differentiated features')
    feature_builder.fractional_differentiation()
    logging.debug('Update the volume based features')
    feature_builder.volume_features()

    logging.debug('Populated features')

    filepath = FEATURE_FILE_PATH.format(periodicity)
    save_df_to_csv(feature_builder.df, filepath)
    logging.debug('Saved data to {}'.format(filepath))


def process_labels(periodicity):
    filepath = FEATURE_FILE_PATH.format(periodicity)
    df = load_btc_csv(filepath)
    df = load_glassnode_features_based_on_args(df)

    ts = TradeStrategy(df)
    logging.debug('Computing EWM events...')
    ts.compute_ewm_events(DEFAULT_FAST_WIN_LENGTH, DEFAULT_SLOW_WIN_LENGTH,
                          DEFAULT_T1_LENGTH)
    logging.debug('Computing triple barrier events...')
    ts.triple_barrier_events(DEFAULT_PT_SL, DEFAULT_VOLATILITY_LENGTH,
                             DEFAULT_MIN_RET, DEFAULT_CPUS)
    # Una pequeña tabla de contención que nos indica como se distribuyen
    # los labels.
    logging.debug('Label contention table')
    logging.debug(ts.labels['bin'].value_counts())

    ts.coevents(DEFAULT_CPUS)

    logging.debug('Saved datasets')
    save_df_to_csv(ts.t1, T1_FILE_PATH.format(periodicity))
    save_df_to_csv(ts.t_events, T_EVENTS_FILE_PATH.format(periodicity))
    save_df_to_csv(ts.target, T1_FILE_PATH.format(periodicity))
    save_df_to_csv(ts.tbe, TRIPLE_BARRIER_EVENTS_FILE_PATH.format(periodicity))
    save_df_to_csv(ts.labels, LABELS_FILE_PATH.format(periodicity))
    save_df_to_csv(ts.numCoEvents, NUM_CO_EVENTS_FILE_PATH.format(periodicity))
    save_df_to_csv(ts.cew, CEW_FILE_PATH.format(periodicity))


def print_metric_table(optimizer, name):
    logging.debug('Strategies by {}'.format(name))
    logging.debug('{}\t\tParameters'.format(name))
    result = optimizer.order_strategies_by(name)
    for m, st in result:
        logging.debug('{}\t\t{}'.format(m, st.params))
    logging.debug('Best strategy by {}: {}\t\t{}'.format(
        name, result[-1][0], result[-1][1].params))


def print_optimization_results(optimizer, model_name, metric_name):
    metrics = ['f1_score', 'accuracy', 'auc',
               'recall', 'precision', 'oob_score', 'neg_log_loss']
    if metric_name not in metrics:
        for name in metrics:
            if (model_name != 'Boosting' or model_name != 'Boosting_Optim') and name == 'oob_score':
                continue
            print_metric_table(optimizer, name)
    else:
        print_metric_table(optimizer, metric_name)


def optimize_strategy(periodicity, glassnode_features, sentiment_features,
                      model_name, num_samples, metric_name):
    filepath = FEATURE_FILE_PATH.format(periodicity)
    df = load_btc_csv(filepath)
    df = load_sentiment_features_based_on_args(df, sentiment_features)
    df = load_glassnode_features_based_on_args(df, glassnode_features)

    optimizer = FundamentalStrategyPipelineOptimizer(df, DEFAULT_CPUS, 123)
    if model_name == 'RF':
        logging.debug('Optimizing strategy with RF...')
        optimizer.optimize_strategies(
            num_samples, StrategyPipelineFixedModelRF)
    elif model_name == 'Bagging':
        logging.debug('Optimizing strategy with Bagging...')
        optimizer.optimize_strategies(
            num_samples, StrategyPipelineFixedModelBagging)
    elif model_name == 'Boosting':
        logging.debug('Optimizing strategy with Boosting...')
        optimizer.optimize_strategies(
            num_samples, StrategyPipelineFixedModelBoosting)

    print_optimization_results(optimizer, model_name, metric_name)


def optimize_model(periodicity, glassnode_features, sentiment_features,
                   model_name, num_samples, metric_name):
    filepath = FEATURE_FILE_PATH.format(periodicity)
    df = load_btc_csv(filepath)
    df = load_glassnode_features_based_on_args(df, glassnode_features)
    df = load_sentiment_features_based_on_args(df, sentiment_features)

    optimizer = MlModelStrategyPipelineOptimizer(
        df, DEFAULT_CPUS, 123, model_name)
    if model_name == 'RF':
        logging.debug('Optimizing strategy with RF...')
        optimizer.optimize_strategies(
            num_samples, StrategyPipelinePurgedKFoldRFModel)
    elif model_name == 'Bagging':
        logging.debug('Optimizing strategy with Bagging...')
        optimizer.optimize_strategies(
            num_samples, StrategyPipelinePurgedKFoldBaggingModel)
    elif model_name == 'Boosting':
        logging.debug('Optimizing strategy with Boosting...')
        optimizer.optimize_strategies(
            num_samples, StrategyPipelinePurgedKFoldBoostingModel)

    print_optimization_results(optimizer, model_name, metric_name)

def feature_importance_model(periodicity, glassnode_features, sentiment_features, model_name):
    filepath = FEATURE_FILE_PATH.format(periodicity)
    df = load_btc_csv(filepath)
    df = load_glassnode_features_based_on_args(df, glassnode_features)
    df = load_sentiment_features_based_on_args(df, sentiment_features)

    optimizer = MlModelOptimStrategyFeatImportance(
        df, DEFAULT_CPUS, 123, model_name)
    if model_name == 'RF_Optim':
        logging.debug('Optimizing strategy with RF...')
        optimizer.feature_importance(
            1, StrategyPipelinePurgedKFoldRFModel)
    elif model_name == 'Bagging_Optim':
        logging.debug('Optimizing strategy with Bagging...')
        optimizer.feature_importance(
            1, StrategyPipelinePurgedKFoldBaggingModel)
    elif model_name == 'Boosting_Optim':
        logging.debug('Optimizing strategy with Boosting...')
        optimizer.feature_importance(
            1, StrategyPipelinePurgedKFoldBoostingModel)
    filename = '../datasets/{}_importance.pickle'.format(model_name)
    pickle.dump(optimizer, open(filename, 'wb'))

    print('Mean decrease in F1-score for {}'.format(model_name))
    print(optimizer.importance)
    print('Features above the mean of Mean decrease in F1-score for {}'.format(model_name))
    print(optimizer.importance.loc[optimizer.importance['mean'] > optimizer.importance['mean'].mean(),])


def save_optimized_model(periodicity, glassnode_features, sentiment_features,
                         model_name):
    filepath = FEATURE_FILE_PATH.format(periodicity)
    df = load_btc_csv(filepath)
    df = load_glassnode_features_based_on_args(df, glassnode_features)
    df = load_sentiment_features_based_on_args(df, sentiment_features)

    optimizer = MlModelStrategyPipelineOptimizer(
        df, DEFAULT_CPUS, 123, model_name)

    if model_name == 'Bagging_Optim':
        logging.debug('Training strategy with optimum Bagging...')
        optimizer.optimize_strategies(
            1, StrategyPipelinePurgedKFoldBaggingModel)
    elif model_name == 'RF_Optim':
        logging.debug('Training strategy with optimum RF...')
        optimizer.optimize_strategies(
            1, StrategyPipelinePurgedKFoldRFModel)
    elif model_name == 'Boosting_Optim':
        logging.debug('Training strategy with optimum Boosting...')
        optimizer.optimize_strategies(
            1, StrategyPipelinePurgedKFoldBoostingModel)
    elif model_name == 'Bagging_Optim_Feat_Imp': 
        logging.debug('Training strategy with optimum Bagging...')
        optimizer.optimize_strategies(
            1, StrategyPipelinePurgedKFoldBaggingModel)

    logging.debug('Training strategy with optimum {}...'.format(model_name))
    print_optimization_results(optimizer, model_name, 'all')

    filename = '../datasets/{}.pickle'.format(model_name)
    pickle.dump(optimizer, open(filename, 'wb'))

configure_logger()

# Feature engineering
if PROCESS_GLASSNODE_FEATURES:
    logging.debug('Processing glassnode features...')
    process_glassnode_info()
else:
    logging.debug('Skipping to process glassnode info.')

if PROCESS_FINANCIAL_FEATURES:
    logging.debug('Loading features...')
    process_features(DEFAULT_BTC_PERIODICITY_DATA)
else:
    logging.debug('Skipping to load features.')

# Primary model optimization
if PROCESS_PRIMARY_MODEL_OPTIMIZATION:
    logging.debug('Processing primary model optimization...')
    optimize_strategy(DEFAULT_BTC_PERIODICITY_DATA,
                      'combined', # Use Glassnode features
                      'combined', # Use Social features
                      MODEL_NAME,
                      FUNDAMENTAL_STRATEGY_OPTIM_SAMPLES,
                      PRIMARY_MODEL_SELECTION_METRIC)
else:
    logging.debug('Skipping primary model optimization...')
# Close
# Bagging: 2021-05-16 13:02:20,292 [pipeline_tesis.py:print_metric_table:195][DEBUG]  Best strategy by f1_score: 0.631578947368421         {'ewm_lengths': (5, 20), 't1_length': 8, 'pt': 0.02, 'sl': 0.03, 'min_ret': 0.03, 'vol_len': 5, 'cpus': 7}
# Boosting:  Best strategy by f1_score: 0.7777777777777777                {'ewm_lengths': (10, 50), 't1_length': 8, 'pt': 0.01, 'sl': 0.07, 'min_ret': 0.02, 'vol_len': 10, 'cpus': 7, 'n_estimators': 571, 'max_features': 0.556184303727964, 'max_depth': 2, 'cv_splits': 6, 'embargo': 0.01, 'learning_rate': 0.006954236541921682}


# CloseFFD
# Bagging: Best strategy by f1_score: 0.5833333333333334                {'ewm_lengths': (10, 100), 't1_length': 8, 'pt': 0.02, 'sl': 0.03, 'min_ret': 0.008, 'vol_len': 25, 'cpus': 7}
# Boosting: Best strategy by f1_score: 0.8571428571428571                {'ewm_lengths': (20, 200), 't1_length': 21, 'pt': 0.05, 'sl': 0.05, 'min_ret': 0.007, 'vol_len': 10, 'cpus': 7}
# RF: Best strategy by f1_score: 0.75              {'ewm_lengths': (20, 200), 't1_length': 21, 'pt': 0.06, 'sl': 0.09, 'min_ret': 0.001, 'vol_len': 5, 'cpus': 7}

# Secondary model optimization
if PROCESS_SECONDARY_MODEL_OPTIMIZATION:
    logging.debug('Processing secondary model optimization...')
    optimize_model(DEFAULT_BTC_PERIODICITY_DATA,
                   'combined', # Use Glassnode features
                   'combined', # Use Social features
                   MODEL_NAME,
                   ML_STRATEGY_OPTIM_SAMPLES,
                   SECONDARY_MODEL_SELECTION_METRIC)
else:
    logging.debug('Skipping secondary model optimization.')
# CloseFFD
# Bagging: Best strategy by f1_score: 0.38888888888888884               {'ewm_lengths': (10, 100), 't1_length': 8, 'pt': 0.02, 'sl': 0.03, 'min_ret': 0.008, 'vol_len': 25, 'cpus': 7, 'n_estimators': 1083, 'max_features': 0.8922007990260129, 'max_depth': 2, 'cv_splits': 6, 'embargo': 0.01, 'learning_rate': 0.0050251905757574085}
# Boosting: Best strategy by f1_score: 0.611111111111111         {'ewm_lengths': (20, 200), 't1_length': 21, 'pt': 0.05, 'sl': 0.05, 'min_ret': 0.07, 'vol_len': 10, 'cpus': 7, 'n_estimators': 974, 'max_features': 0.5537046328904053, 'max_depth': 2, 'cv_splits': 6, 'embargo': 0.01, 'learning_rate': 0.007325458838987043}
# RF: Best strategy by f1_score: 0.8333333333333334                {'ewm_lengths': (20, 200), 't1_length': 21, 'pt': 0.06, 'sl': 0.09, 'min_ret': 0.001, 'vol_len': 5, 'cpus': 7, 'n_estimators': 903, 'max_features': 0.5546179069545076, 'max_depth': 2, 'cv_splits': 6, 'embargo': 0.01, 'learning_rate': 0.005689782451735344}

# All in:
# Log loss:
# Bagging: Best strategy by neg_log_loss: -0.3899522644006023           {'ewm_lengths': (5, 30), 't1_length': 3, 'pt': 0.06, 'sl': 0.02, 'min_ret': 0.005, 'vol_len': 50, 'cpus': 7, 'n_estimators': 720, 'max_features': 0.29363265635308283, 'max_depth': 2, 'cv_splits': 6, 'embargo': 0.01, 'learning_rate': 0.006165097685267023}
# RF: Best strategy by neg_log_loss: -0.48767384994052104          {'ewm_lengths': (5, 20), 't1_length': 3, 'pt': 0.07, 'sl': 0.06, 'min_ret': 0.008, 'vol_len': 5, 'cpus': 7, 'n_estimators': 1519, 'max_features': 0.11658662380619006, 'max_depth': 2, 'cv_splits': 6, 'embargo': 0.01, 'learning_rate': 0.0017207862726338471}
# Boosting: Best strategy by neg_log_loss: -0.48581458298971725          {'ewm_lengths': (5, 20), 't1_length': 3, 'pt': 0.07, 'sl': 0.06, 'min_ret': 0.008, 'vol_len': 5, 'cpus': 7, 'n_estimators': 1009, 'max_features': 0.11658662380619006, 'max_depth': 2, 'cv_splits': 6, 'embargo': 0.01, 'learning_rate': 0.0017207862726338471}

if RUN_FEAT_IMPORTANCE:
    logging.debug('Feature importance MDA.')
    feature_importance_model(DEFAULT_BTC_PERIODICITY_DATA,
                             'combined', # Use Glassnode features
                             'combined', # Use Social features
                             MODEL_NAME)
else:
    logging.debug('Skipping feature importance.')


# All in - Log loss:
# Bagging:
#                                  mean       std                                                                                                       
# CloseFFD                     0.039714  0.026393                                                                                                       
# HighFFD                      0.018739  0.019605                                                                                                       
# LowFFD                       0.002177  0.001215                                                                                                       
# OpenFFD                      0.006869  0.003897                                                                                                       
# SentimentIndex              -0.003622  0.002974                                                                                                       
# Volume_(BTC)-log             0.032430  0.015899                                                                                                       
# Volume_(Currency)-log        0.020350  0.008696                                                                                                       
# active-addresses-FFD        -0.006143  0.004894                                                                                                       
# autocorr_1                  -0.005062  0.004809                                                                                                       
# autocorr_3                  -0.010104  0.004387                                                                                                       
# block-size-total_log        -0.005010  0.006844                                                                                                       
# bsadf_ct_1                   0.002128  0.002060                                                                                                       
# bsadf_ctt_1                  0.003910  0.003278                                                                                                       
# bsadf_ctt_2                  0.004907  0.004139                                                                                                       
# bsadf_ctt_3                  0.003264  0.004178
# bsadf_nt_2                   0.003447  0.004077                                                                                              [67/1931]
# daysTillHalving              0.003387  0.003194
# fees-mean-log                0.004142  0.002743
# log_ret                      0.013930  0.006140
# market-cap-log               0.002412  0.001501
# rsi_10                      -0.005796  0.003979
# rsi_5                       -0.011244  0.013815
# sopr                        -0.006307  0.004879
# transfer-volume-mean-log    -0.010212  0.007704
# transfer-volume-median-log   0.006073  0.006431
# transfer-volume-total-log   -0.007392  0.004249
# trgt                         0.003003  0.007776
# utx-os-created-log          -0.017422  0.018345
# utxo-value-created-mean-log -0.005338  0.003951
# utxo-value-created-total    -0.003077  0.004157
# utxo-value-spent-mean-log   -0.002586  0.004585
# utxo-value-spent-total      -0.005101  0.001508
# vol_10                      -0.002654  0.006811
# vol_15                       0.008678  0.003120
# vol_5                        0.020516  0.015831

# RF:
#                                  mean       std                                                                                              [83/1908]
# CloseFFD                     0.008665  0.007385                                                                                                       
# HighFFD                      0.011431  0.010204                                                                                                       
# LowFFD                       0.003861  0.003425                                                                                                       
# OpenFFD                      0.004214  0.004956                                                                                                       
# Volume_(BTC)-log             0.007523  0.012080                                                                                                       
# Volume_(Currency)-log        0.012031  0.004804                                                                                                       
# blocks-mined                 0.001984  0.003196                                                                                                       
# bsadf_ct_1                   0.002377  0.002542                                                                                                       
# bsadf_ct_3                   0.003108  0.002764                                                                                                       
# bsadf_ctt_1                  0.004133  0.002447                                                                                                       
# bsadf_ctt_2                  0.003344  0.002819                                                                                                       
# bsadf_ctt_3                  0.004858  0.003455                                                                                                       
# circulating-supply-log-diff  0.009187  0.005753                                                                                                       
# fees-mean-log               -0.001771  0.001304                                                                                                       
# fees-total-log              -0.002155  0.001364 
# log_ret                      0.003414  0.002555                                                                                              [67/1908]
# ratio-log                    0.001735  0.001687
# rsi_10                      -0.011193  0.006028
# rsi_15                      -0.004123  0.002613
# rsi_30                       0.002040  0.001457
# rsi_5                        0.006248  0.012782
# side                        -0.004614  0.003350
# sopr                        -0.001784  0.001777
# transaction-rate            -0.002794  0.003339
# transfer-volume-mean-log    -0.002778  0.002633
# transfer-volume-median-log  -0.004706  0.004579
# transfer-volume-total-log   -0.002363  0.002790
# trgt                         0.024192  0.011238
# utxo-value-created-mean-log -0.006001  0.004078
# utxo-value-created-total    -0.002207  0.002609
# utxo-value-spent-mean-log   -0.005968  0.002441
# utxo-value-spent-total      -0.002879  0.003847
# vol_10                       0.015258  0.008501
# vol_15                       0.010937  0.006960
# vol_5                        0.004613  0.008096

# Boosting:
#                                  mean       std                                                                                                       
# CloseFFD                     0.036499  0.019866                                                                                                       
# HighFFD                      0.009038  0.011843                                                                                                       
# LowFFD                       0.002198  0.002309                                                                                                       
# OpenFFD                      0.002291  0.002814                                                                                                       
# Volume_(BTC)-log             0.019667  0.005771                                                                                                       
# Volume_(Currency)-log        0.011959  0.003881                                                                                                       
# active-addresses-FFD        -0.003439  0.003194                                                                                                       
# autocorr_3                   0.003395  0.001363                                                                                                       
# block-size-total_log        -0.003024  0.006063                                                                                                       
# bsadf_ct_2                   0.005366  0.003432                                                                                                       
# bsadf_ctt_3                  0.007280  0.005685                                                                                                       
# circulating-supply-log-diff  0.008752  0.007484                                                                                                       
# log_ret                      0.010494  0.016700                                                                                                       
# rsi_10                      -0.004397  0.004965
# rsi_5                        0.019879  0.020595
# side                        -0.002534  0.001602
# transaction-size-mean       -0.002307  0.001792
# transfer-volume-mean-log    -0.003906  0.003429
# trgt                         0.013330  0.018414
# utx-os-created-log          -0.010869  0.007732
# utxo-value-created-mean-log -0.002342  0.001846
# utxo-value-created-total    -0.004653  0.005475
# utxo-value-spent-mean-log   -0.004022  0.002646
# utxo-value-spent-total      -0.003874  0.002865
# vol_5                        0.013671  0.017678

if SAVE_OPTIMIZED_SECONDARY_MODEL:
    logging.debug('Save optimized secondary model.')
    save_optimized_model(DEFAULT_BTC_PERIODICITY_DATA,
                         'combined', # Use Glassnode features
                         'combined', # Use Social features
                         MODEL_NAME)
else:
    logging.debug('Skipping to train and save secondary model.')


