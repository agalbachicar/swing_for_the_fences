import logging
import random

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection._split import _BaseKFold
from sklearn.metrics import roc_curve, classification_report, log_loss, accuracy_score, recall_score, auc, average_precision_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from cv import crossValidationTrain, PurgedKFold
from events import getEwmEvents
from features import *
from feature_importance import featImpMDA2
from labelling import getVerticalBarrier, getDailyVol, getEvents, getBins
from frac_diff import compute_multiple_ffd, get_d_optim, fracDiff_FFD
from mpfin import mpPandasObj
from sample_weights import mpNumCoEvents, mpSampleTW, mpSampleW
from structural_breaks import compute_bsadf

# -------------------------------------------------------------------------------


def run_feature_engineering_for_glassnode_features(df, file_path):
    """
    """
    # Apply logarithmic transformation to total-addresses and drop total-addresses.
    df['total-addresses-log'] = np.log(df['total-addresses'])
    # Drop new-addresses, total-addresses, sending-addresses, receiving-addresses
    # FFD to active-addresses with 0.1 and apply log.
    df.drop(['new-addresses', 'total-addresses', 'sending-addresses',
             'receiving-addresses'], axis=1, inplace=True)
    df['active-addresses-FFD'] = fracDiff_FFD(
        df['active-addresses'].to_frame(), d=0.1, thres=1e-3)
    df.drop(['active-addresses'], axis=1, inplace=True)
    # Drop block-interval-mean, block-interval-median, block-size-total, block-size-mean and keep blocks-mined
    df['block-size-total_log'] = np.log(df['block-size-total'])
    df.drop(['block-interval-mean', 'block-interval-median',
             'block-size-total', 'block-size-mean'], axis=1, inplace=True)
    # Fees require a log transform and then dropped.
    df['fees-mean-log'] = np.log(df['fees-mean'] + 1)
    df['fees-total-log'] = np.log(df['fees-total'] + 1)
    df.drop(['fees-mean', 'fees-total'], axis=1, inplace=True)
    # sopr, ratio, daysTillHalving, price-drawdown-from-ath, market-cap and circulating-supply
    df['market-cap-log'] = np.log(df['market-cap'])
    df['circulating-supply-log'] = np.log(df['circulating-supply'])
    df['circulating-supply-log-diff'] = df['circulating-supply-log'].diff()
    df['ratio-log'] = np.log(df['ratio'])
    df.drop(['market-cap', 'circulating-supply', 'ratio'], axis=1, inplace=True)
    # transaction-size-total, transaction-rate, transaction-count
    df['transaction-rate-d'] = pd.qcut(df['transaction-rate'],
                                       q=10, labels=False)
    df.drop(['transaction-size-total', 'transaction-count'],
            axis=1, inplace=True)
    # transaction-size-mean dont touch
    df['transfer-volume-mean-log'] = np.log(df['transfer-volume-mean'])
    df['transfer-volume-median-log'] = np.log(df['transfer-volume-median'])
    df['transfer-volume-total-log'] = np.log(df['transfer-volume-total'])
    df.drop(['transfer-volume-mean', 'transfer-volume-median',
             'transfer-volume-total'], axis=1, inplace=True)
    # UTXO
    df['utx-os-created-log'] = np.log(df['utx-os-created'])
    df['utxo-value-spent-mean-log'] = np.log(df['utxo-value-spent-mean'])
    df['utxo-value-spent-median-log'] = np.log(df['utxo-value-spent-median'])
    df['utxo-value-created-mean-log'] = np.log(df['utxo-value-created-mean'])
    df.drop(['utx-os-created', 'utx-os-spent', 'utxo-value-spent-mean',
             'utxo-value-spent-median', 'utxo-value-created-mean'], axis=1, inplace=True)

    df.to_csv(file_path, index_label='Timestamp', date_format='%s')


class FeatureBuilder:
    """
    Eases the process of adding features to a financial data frame.
    """

    def __init__(self, df):
        """
        Constructs a FeatureBuilder.

        @param df The financial pandas DataFrame.
        """
        self.df = df

    def add_rsi(self, window_lengths):
        for window_length in window_lengths:
            self.df['rsi_{}'.format(window_length)] = rsi(
                self.df['Close'], window_length)

    def add_log_ret(self):
        self.df['log_ret'] = log_ret(self.df['Close'])

    def add_autocorr(self, lags, window_length):
        for lag in lags:
            self.df['autocorr_{}'.format(lag)] = autocorr(
                self.df['Close'], window_length, lag)

    def add_volatility(self, window_lengths):
        for window_length in window_lengths:
            self.df['vol_{}'.format(window_length)] = volatility(
                self.df['Close'], window_length)

    def add_sadf(self, min_sl, max_sl, constants, lags):
        # Computo el logaritmo de precios.
        log_p = self.df['Close'].apply(lambda x: np.log(x)).dropna()

        # Genero un data frame con las columnas que voy a computar.
        bsadf = pd.DataFrame(index=log_p.index, columns=[
                             'bsadf_{}_{}'.format(c, l) for c in constants for l in lags])

        # Computo SADF
        for constant in constants:
            for lag in lags:
                col_name = 'bsadf_{}_{}'.format(constant, lag)
                logging.debug('Computing {}'.format(col_name))
                df0 = compute_bsadf(log_p.to_frame(), min_sl,
                                    max_sl, constant, lag)
                bsadf.loc[:, col_name] = df0['bsadf']
        # Uno los dataframes.
        self.df = pd.merge(self.df, bsadf, how='inner',
                           left_index=True, right_index=True)

    def volume_features(self):
        self.df['Volume_(BTC)-log'] = np.log(self.df['Volume_(BTC)'])
        self.df['Volume_(Currency)-log'] = np.log(self.df['Volume_(Currency)'])
        self.df.drop(['Volume_(BTC)', 'Volume_(Currency)'], axis=1, inplace=True)

    def fractional_differentiation(self):
        self.df['OpenFFD'] = fracDiff_FFD(
            self.df['Open'].to_frame(), d=0.4, thres=1e-3)
        self.df['HighFFD'] = fracDiff_FFD(
            self.df['High'].to_frame(), d=0.4, thres=1e-3)
        self.df['LowFFD'] = fracDiff_FFD(
            self.df['Low'].to_frame(), d=0.2, thres=1e-3)
        self.df['CloseFFD'] = fracDiff_FFD(
            self.df['Close'].to_frame(), d=0.4, thres=1e-3)

        self.df.drop(['Open', 'High', 'Low'], axis=1, inplace=True)
# -------------------------------------------------------------------------------


class TradeStrategy:
    def __init__(self, df):
        self.df = df
        self.t1 = None
        self.t_events = None
        self.target = None
        self.tbe = None
        self.labels = None
        self.numCoEvents = None
        self.cew = None
        self.average_uniqueness = None

    def compute_ewm_events(self, fast_window_length, slow_window_length, t1_length):
        self.t_events = getEwmEvents(
            self.df['CloseFFD'], fast_window_length, slow_window_length)
        # Computamos las marcas temporales de la ventana. Para cada evento
        # en tEvents (inicio de la ventana), obtenemos el final de la ventana.
        # Nota: a diferencia de la notebook con labelling unicamente, tEvents
        #       aquí es una serie con el side de la apuesta por lo que debemos
        #       pasar el indice a getVerticalBarrier() para reutilizar la funcion.
        self.t1 = getVerticalBarrier(
            self.t_events.index, self.df['CloseFFD'], numDays=t1_length)

    def triple_barrier_events(self, ptSl, volatility_length, min_ret, cpus):
        # Computamos la volatilidad diaria, suavizada con una media
        # movil pesada.
        self.target = getDailyVol(
            close=self.df['CloseFFD'], span0=volatility_length)

        # Generamos los eventos de la triple frontera. En esta funcion obtenemos
        # un dataframe cuyo indice es cuando ocurre el evento y tiene 2 columnasÑ
        # - t1: momento en el que sucede el evento.
        # - trgt: retorno obtenido en ese momento.
        self.tbe = getEvents(self.df['CloseFFD'], self.t_events.index, ptSl,
                             self.target, min_ret, cpus, t1=self.t1,
                             side=self.t_events)

        # Obtenemos los labels! Los labels nos dan la siguiente informacion:
        # - Indice: momento en el que ocurre el evento segun nuestra estrategia.
        # - Columna ret: el retorno que vamos a obtener.
        # - Columna bin: lo que sucede con la señal de la apuesta:
        #   - 1: se toma la apuesta.
        #   - 0: no se toma la apuesta.
        self.labels = getBins(self.tbe, self.df['CloseFFD'])

    def coevents(self, cpus):
        # Obtengo los coeficientes de concurrencia para cada evento.
        self.numCoEvents = mpPandasObj(mpNumCoEvents, ('molecule', self.tbe.index), cpus,
                                       closeIdx=self.df.index, t1=self.tbe['t1'])
        self.numCoEvents = self.numCoEvents.loc[~self.numCoEvents.index.duplicated(
            keep='last')]
        self.numCoEvents = self.numCoEvents.reindex(
            self.df['CloseFFD'].index).fillna(0)

        # Genero un data frame que contenga en una columna los pesos por concurrencia
        # y en otra los pesos por concurrencia + retornos.
        self.cew = pd.DataFrame()
        self.cew['tW'] = mpPandasObj(mpSampleTW, ('molecule', self.tbe.index), cpus,
                                     t1=self.tbe['t1'], numCoEvents=self.numCoEvents)
        self.cew['w'] = mpPandasObj(mpSampleW, ('molecule', self.tbe.index), cpus,
                                    t1=self.tbe['t1'], numCoEvents=self.numCoEvents,
                                    close=self.df['Close'])
        self.cew['w'] *= self.cew.shape[0]/self.cew['w'].sum()
        self.average_uniqueness = self.cew.mean()[0]
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------


class StrategyPipeline:
    METRICS = {'auc', 'accuracy', 'oob_score',
               'precision', 'recall', 'f1_score', 'neg_log_loss'}

    def __init__(self, df):
        self.df = df

    def compute_bets(self):
        pass

    def build_bet_sizing_model(self):
        pass

    def prepare_features(self):
        pass

    def train_bet_sizing_model(self):
        pass

    def eval_bet_sizing_model(self):
        pass

# -------------------------------------------------------------------------------


class StrategyPipelineFixedModel(StrategyPipeline):

    def __init__(self, df, params):
        StrategyPipeline.__init__(self, df)
        self.params = params
        self.ts = TradeStrategy(self.df)
        self.metrics = {
            'oob_score': None,
            'auc': None,
            'class_report': None,
            'accuracy': None,
            'recall': None,
            'precision': None,
            'f1_score': None,
            'neg_log_loss': None,
        }
        self.model = None

    def compute_bets(self):
        self.ts.compute_ewm_events(
            self.params['ewm_lengths'][0], self.params['ewm_lengths'][1], self.params['t1_length'])
        self.ts.triple_barrier_events([self.params['pt'], self.params['sl']],
                                      self.params['vol_len'], self.params['min_ret'], self.params['cpus'])
        self.ts.coevents(self.params['cpus'])

    def prepare_features(self):
        # Preparamos la informacion para introducirla en un modelo.
        Xy = (pd.merge_asof(self.ts.tbe,
                            self.df.loc[self.ts.tbe.index],
                            left_index=True,
                            right_index=True,
                            direction='forward').dropna())
        Xy = (pd.merge_asof(Xy,
                            self.ts.labels[['bin']],
                            left_index=True,
                            right_index=True,
                            direction='forward').dropna())
        sample_weights_df = self.ts.cew.loc[Xy.index, ['w']]
        tt1 = self.ts.t1.loc[Xy.index]
        X = Xy.copy()[Xy.columns.difference(['bin', 't1', 'Close'])].values
        y = Xy.copy().bin.values
        sample_weights = sample_weights_df.copy().w.values
        # Realizamos un train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.8, shuffle=False)
        self.sample_weights_train, self.sample_weights_test = sample_weights[
            :self.X_train.shape[0]], sample_weights[self.X_train.shape[0]:]

    def train_bet_sizing_model(self):
        try:
            self.model.fit(self.X_train, self.y_train,
                           self.sample_weights_train)
        except ValueError:
            pass

    def eval_bet_sizing_model(self):
        # Predecimos con el test set y contrastamos.
        try:
            aux = self.model.predict_proba(self.X_test)
        except:
            logging.debug(
                'Discarding model because of lack of samples to evaluate for either true or false.')
            return
        if aux.shape[1] == 1:
            logging.debug(
                'Discarding model because of lack of samples to evaluate for either true or false.')
            self.fpr, self.tpr, self.class_report = None, None, None
            return

        y_pred_prob = aux[:, 1]
        y_pred = self.model.predict(self.X_test)

        # Computamos todas las metricas
        if hasattr(self.model, 'oob_score_'):
            self.metrics['oob_score'] = self.model.oob_score_
        else:
            self.metrics['oob_score'] = 0.
        self.metrics['class_report'] = classification_report(
            self.y_test, y_pred)
        self.metrics['accuracy'] = accuracy_score(self.y_test, y_pred)
        self.metrics['precision'] = average_precision_score(
            self.y_test, y_pred_prob)
        self.metrics['recall'] = recall_score(self.y_test, y_pred)
        self.metrics['f1_score'] = f1_score(self.y_test, y_pred)
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_prob)
        self.metrics['auc'] = auc(fpr, tpr)
        self.metrics['neg_log_loss'] = -log_loss(self.y_test, y_pred_prob)


class StrategyPipelineFixedModelRF(StrategyPipelineFixedModel):
    def __init__(self, df, params):
        StrategyPipelineFixedModel.__init__(self, df, params)

    def build_bet_sizing_model(self):
        # Hiperpametros
        n_estimator = 1000     # Numero de arboles.
        # Porcion del data set a samplear para cada arbol.
        max_samples = self.ts.average_uniqueness
        criterion = 'entropy'  # Es un clasificador, necesitamos definir la metrica del modelo
        # que va a optimizar. Podria ser gini tambien.
        # Numero de niveles que va a tener cada arbol. Al elegir un split binario
        max_depth = 2
        # permitimos que el entrenamiento sea mas lento.
        # Para mas argumentos, revisar https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        self.model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimator,
                                            criterion=criterion, n_jobs=self.params['cpus'], random_state=123,
                                            oob_score=True)


class StrategyPipelineFixedModelBagging(StrategyPipelineFixedModel):
    def __init__(self, df, params):
        StrategyPipelineFixedModel.__init__(self, df, params)

    def build_bet_sizing_model(self):
        # Hiperpametros
        n_estimator = 1000    # Numero de arboles.
        # Porcion del data set a samplear para cada arbol.
        max_samples = self.ts.average_uniqueness
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html
        self.model = BaggingClassifier(max_samples=max_samples, n_estimators=n_estimator,
                                       n_jobs=self.params['cpus'], random_state=123, oob_score=True)


class StrategyPipelineFixedModelBoosting(StrategyPipelineFixedModel):
    def __init__(self, df, params):
        StrategyPipelineFixedModel.__init__(self, df, params)

    def build_bet_sizing_model(self):
        # Hiperpametros
        loss = 'deviance'                           # Logistic regression
        learning_rate = 0.01                        # Learning rate
        n_estimators = 1000                         # Learning stages
        # Porcion del data set a samplear para cada arbol. Might increase bias.
        subsample = self.ts.average_uniqueness
        criterion = 'friedman_mse'
        # The maximum depth of the individual regression estimators
        max_depth = 2
        # Maximum features to be used for each tree => sqrt(num_features)
        max_features = 'sqrt'
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
        self.model = GradientBoostingClassifier(loss=loss,
                                                learning_rate=learning_rate,
                                                n_estimators=n_estimators,
                                                subsample=subsample,
                                                criterion=criterion,
                                                max_depth=max_depth,
                                                init=None,
                                                random_state=123,
                                                max_features=max_features)

# -------------------------------------------------------------------------------


class StrategyPipelinePurgedKFoldModel(StrategyPipeline):
    def __init__(self, df, params):
        StrategyPipeline.__init__(self, df)
        self.params = params
        self.ts = TradeStrategy(self.df)
        self.metrics = {
            'oob_score': None,
            'auc': None,
            'class_report': None,
            'accuracy': None,
            'recall': None,
            'precision': None,
        }
        self.model = None

    def compute_bets(self):
        self.ts.compute_ewm_events(
            self.params['ewm_lengths'][0], self.params['ewm_lengths'][1], self.params['t1_length'])
        self.ts.triple_barrier_events([self.params['pt'], self.params['sl']],
                                      self.params['vol_len'], self.params['min_ret'], self.params['cpus'])
        self.ts.coevents(self.params['cpus'])

    def prepare_features(self):
        # Preparamos la informacion para introducirla en un modelo.
        self.Xy = (pd.merge_asof(self.ts.tbe,
                                 self.df.loc[self.ts.tbe.index],
                                 left_index=True,
                                 right_index=True,
                                 direction='forward').dropna())
        self.Xy = (pd.merge_asof(self.Xy,
                                 self.ts.labels[['bin']],
                                 left_index=True,
                                 right_index=True,
                                 direction='forward').dropna())
        self.sample_weights_df = self.ts.cew.loc[self.Xy.index, ['w']]
        self.tt1 = self.ts.t1.loc[self.Xy.index]

    def train_bet_sizing_model(self):
        self.pkf = PurgedKFold(
            n_splits=self.params['cv_splits'], t1=self.tt1, pctEmbargo=self.params['embargo'])

    def eval_bet_sizing_model(self):
        # Predecimos con el test set y contrastamos.
        if not self.params['features']:
            aucs, accuracies, oob_scores, recalls, precisions, class_reports, f1_scores, log_loss_scores = crossValidationTrain(
                self.pkf, self.model, self.Xy.copy()[self.Xy.columns.difference(['bin', 't1', 'Close'])], self.Xy.copy().bin, self.sample_weights_df)
        else:
            aucs, accuracies, oob_scores, recalls, precisions, class_reports, f1_scores, log_loss_scores = crossValidationTrain(
                self.pkf, self.model, self.Xy.copy().loc[:, self.params['features']], self.Xy.copy().bin, self.sample_weights_df)

        # Computamos todas las metricas
        if oob_scores:
            self.metrics['oob_score'] = np.mean(oob_scores)
        self.metrics['class_report'] = class_reports
        self.metrics['accuracy'] = np.mean(accuracies)
        self.metrics['precision'] = np.mean(precisions)
        self.metrics['recall'] = np.mean(recalls)
        self.metrics['auc'] = np.mean(aucs)
        self.metrics['f1_score'] = np.mean(f1_scores)
        self.metrics['neg_log_loss'] = np.median(log_loss_scores)


class StrategyPipelinePurgedKFoldRFModel(StrategyPipelinePurgedKFoldModel):
    def __init__(self, df, params):
        StrategyPipelinePurgedKFoldModel.__init__(self, df, params)

    def build_bet_sizing_model(self):
        # Hiperpametros
        criterion = 'entropy'  # Es un clasificador, necesitamos definir la metrica del modelo
        # que va a optimizar. Podria ser gini tambien.

        # Para mas argumentos, revisar https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        self.model = RandomForestClassifier(max_depth=self.params["max_depth"],
                                            n_estimators=self.params["n_estimators"],
                                            max_features=self.params["max_features"],
                                            criterion=criterion,
                                            n_jobs=self.params['cpus'],
                                            random_state=123,
                                            oob_score=True)


class StrategyPipelinePurgedKFoldBaggingModel(StrategyPipelinePurgedKFoldModel):
    def __init__(self, df, params):
        StrategyPipelinePurgedKFoldModel.__init__(self, df, params)

    def build_bet_sizing_model(self):
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html
        self.model = BaggingClassifier(max_samples=self.ts.average_uniqueness,
                                       n_estimators=self.params["n_estimators"],
                                       max_features=self.params["max_features"],
                                       n_jobs=self.params['cpus'],
                                       random_state=123,
                                       oob_score=True)


class StrategyPipelinePurgedKFoldBoostingModel(StrategyPipelinePurgedKFoldModel):
    def __init__(self, df, params):
        StrategyPipelinePurgedKFoldModel.__init__(self, df, params)

    def build_bet_sizing_model(self):
        # Hiperpametros
        loss = 'deviance'                           # Logistic regression
        criterion = 'friedman_mse'
        # Maximum features to be used for each tree => sqrt(num_features)
        max_features = 'sqrt'
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
        self.model = GradientBoostingClassifier(loss=loss,
                                                learning_rate=self.params["learning_rate"],
                                                n_estimators=self.params["n_estimators"],
                                                subsample=self.ts.average_uniqueness,
                                                criterion=criterion,
                                                max_depth=self.params["max_depth"],
                                                init=None,
                                                random_state=123,
                                                max_features=max_features)
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------


class StrategyPipelineOptimizer:
    def __init__(self, seed):
        self.strategies = []
        random.seed(seed)

    def sample_strategy_parameters(self):
        pass

    def order_strategies_by(self, metric_name):
        '''
        @param metric_name One in {'auc', 'accuracy', 'oob_score', 'precision', 'recall', 'f1_score', 'neg_log_loss'}
        '''
        if metric_name not in StrategyPipeline.METRICS:
            return []
        result = []
        for st in self.strategies:
            if metric_name in st.metrics and st.metrics[metric_name]:
                result.append((st.metrics[metric_name], st))
        return sorted(result, key=lambda x: x[0])

    def optimize_strategies(self, n_samples, StrategyPipelineType):
        for i in range(n_samples):
            try:
                logging.debug(
                    'Building model and strategy {} out of {}'.format(i+1, n_samples))
                # Gets the parameters for the strategy
                strategy_params = self.sample_strategy_parameters()
                # Builds the strategy
                strategy_pipeline = StrategyPipelineType(
                    self.df.copy(deep=True), strategy_params)
                # Run it!
                strategy_pipeline.compute_bets()
                strategy_pipeline.build_bet_sizing_model()
                strategy_pipeline.prepare_features()
                strategy_pipeline.train_bet_sizing_model()
                strategy_pipeline.eval_bet_sizing_model()
                # Store the pipeline
                self.strategies.append(strategy_pipeline)
            except Exception as e:
               logging.debug('Skipping model {} due to an error in the pipeline.'.format(i+1))
               logging.debug('Error: {}'.format(e))

# -------------------------------------------------------------------------------


class FundamentalStrategyPipelineOptimizer(StrategyPipelineOptimizer):
    def __init__(self, df, cpus, seed):
        StrategyPipelineOptimizer.__init__(self, seed)

        self.df = df
        self.ewm_lengths = [(5, 20), (5, 30), (10, 30), (10, 50),
                            (10, 100), (10, 200), (20, 200), (50, 200)]
        self.t1_lengths = [2, 3, 5, 8, 13, 21]
        self.pts = [x / 100 for x in range(1, 11)]
        self.sls = [x / 100 for x in range(1, 11)]
        self.min_rets = [x / 1000 for x in range(1, 11)]
        self.volatility_lengths = [5, 10, 25, 50]
        self.cpus = [cpus]

    def sample_strategy_parameters(self):
        return {
            'ewm_lengths': random.choice(self.ewm_lengths),
            't1_length': random.choice(self.t1_lengths),
            'pt': random.choice(self.pts),
            'sl': random.choice(self.sls),
            'min_ret': random.choice(self.min_rets),
            'vol_len': random.choice(self.volatility_lengths),
            'cpus': random.choice(self.cpus),
        }

# -------------------------------------------------------------------------------

class MlModelStrategyPipelineOptimizer(StrategyPipelineOptimizer):
    HYPER_PARAMS = {
        "RF": {
            "ewm_lengths": [(20, 200)],
            "t1_lengths": [21],
            "pts": [0.06],
            "sls": [0.09],
            "min_rets": [0.001],
            "volatility_lengths": [5],
            "n_estimators": [500, 2000],
            "max_features": [0.1, 1.0],
            "max_depth": [2],
            "cv_splits": [6],
            "embargo": [0.01],
            "learning_rate": [0.0001, 0.01],
        },
        "Bagging": {
            "ewm_lengths": [(10, 100)],
            "t1_lengths": [8],
            "pts": [0.02],
            "sls": [0.03],
            "min_rets": [0.008],
            "volatility_lengths": [25],
            "n_estimators": [500, 2000],
            "max_features": [0.1, 1.0],
            "max_depth": [2],
            "cv_splits": [6],
            "embargo": [0.01],
            "learning_rate": [0.0001, 0.01],
        },
        "Boosting": {
            "ewm_lengths": [(20, 200)],
            "t1_lengths": [21],
            "pts": [0.05],
            "sls": [0.05],
            "min_rets": [0.07],
            "volatility_lengths": [10],
            "n_estimators": [500, 1500],
            "max_features": [0.1, 1.0],
            "max_depth": [2],
            "cv_splits": [6],
            "embargo": [0.01],
            "learning_rate": [0.0001, 0.01],
        },
        "Bagging_Optim": {
            "ewm_lengths": [(5, 30)],
            "t1_lengths": [3],
            "pts": [0.06],
            "sls": [0.02],
            "min_rets": [0.005],
            "volatility_lengths": [50],
            "n_estimators": [720, 720],
            "max_features": [0.29363265635308283, 0.29363265635308283],
            "max_depth": [2],
            "cv_splits": [6],
            "embargo": [0.01],
            "learning_rate": [0.0005299875819337446, 0.0005299875819337446],
        },
        "Boosting_Optim": {
            "ewm_lengths": [(5, 20)],
            "t1_lengths": [5],
            "pts": [0.07],
            "sls": [0.06],
            "min_rets": [0.008],
            "volatility_lengths": [5],
            "n_estimators": [1009, 1009],
            "max_features": [0.11658662380619006, 0.11658662380619006],
            "max_depth": [2],
            "cv_splits": [6],
            "embargo": [0.01],
            "learning_rate": [0.0017207862726338471, 0.0017207862726338471],
        },
        "RF_Optim": {
            "ewm_lengths": [(5, 20)],
            "t1_lengths": [3],
            "pts": [0.07],
            "sls": [0.06],
            "min_rets": [0.008],
            "volatility_lengths": [5],
            "n_estimators": [1519, 1519],
            "max_features": [0.11658662380619006, 0.11658662380619006],
            "max_depth": [2],
            "cv_splits": [6],
            "embargo": [0.01],
            "learning_rate": [0.0001, 0.01],
        },
        "Bagging_Optim_Feat_Imp": {
            "ewm_lengths": [(5, 30)],
            "t1_lengths": [3],
            "pts": [0.06],
            "sls": [0.02],
            "min_rets": [0.005],
            "volatility_lengths": [50],
            "n_estimators": [720, 720],
            "max_features": [0.29363265635308283, 0.29363265635308283],
            "max_depth": [2],
            "cv_splits": [6],
            "embargo": [0.01],
            "learning_rate": [0.0005299875819337446, 0.0005299875819337446],
            "features": ['CloseFFD', 'HighFFD', 'LowFFD', 'OpenFFD', 'Volume_(BTC)-log', 'Volume_(Currency)-log', 'bsadf_ct_1', 'bsadf_ctt_1', 'bsadf_ctt_2', 'bsadf_ctt_3', 'bsadf_nt_2', 'daysTillHalving', 'fees-mean-log', 'log_ret', 'market-cap-log', 'transfer-volume-median-log', 'trgt', 'vol_15', 'vol_5'],
        },
    }

    def __init__(self, df, cpus, seed, model_name):
        StrategyPipelineOptimizer.__init__(self, seed)

        self.df = df

        # self.ewm_lengths = MlModelStrategyPipelineOptimizer.HYPER_PARAMS[
        #     model_name]["ewm_lengths"]
        # self.t1_lengths = MlModelStrategyPipelineOptimizer.HYPER_PARAMS[model_name]["t1_lengths"]
        # self.pts = MlModelStrategyPipelineOptimizer.HYPER_PARAMS[model_name]["pts"]
        # self.sls = MlModelStrategyPipelineOptimizer.HYPER_PARAMS[model_name]["sls"]
        # self.min_rets = MlModelStrategyPipelineOptimizer.HYPER_PARAMS[model_name]["min_rets"]
        # self.volatility_lengths = MlModelStrategyPipelineOptimizer.HYPER_PARAMS[
        #     model_name]["volatility_lengths"]
        # self.cpus = [cpus]

        self.ewm_lengths = [(5, 20), (5, 30), (10, 30), (10, 50),
                            (10, 100), (10, 200), (20, 200), (50, 200)]
        self.t1_lengths = [2, 3, 5, 8, 13, 21]
        self.pts = [x / 100 for x in range(1, 11)]
        self.sls = [x / 100 for x in range(1, 11)]
        self.min_rets = [x / 1000 for x in range(1, 11)]
        self.volatility_lengths = [5, 10, 25, 50]
        self.cpus = [cpus]

        self.n_estimators = MlModelStrategyPipelineOptimizer.HYPER_PARAMS[
            model_name]["n_estimators"]
        self.max_depth = MlModelStrategyPipelineOptimizer.HYPER_PARAMS[model_name]["max_depth"]
        self.max_features = MlModelStrategyPipelineOptimizer.HYPER_PARAMS[model_name]["max_features"]

        self.cv_splits = MlModelStrategyPipelineOptimizer.HYPER_PARAMS[model_name]["cv_splits"]
        self.embargo = MlModelStrategyPipelineOptimizer.HYPER_PARAMS[model_name]["embargo"]

        # For boosting only
        self.learning_rate = MlModelStrategyPipelineOptimizer.HYPER_PARAMS[
            model_name]["learning_rate"]

        # For the bagging optimized with feature importance model only
        if 'features' in MlModelStrategyPipelineOptimizer.HYPER_PARAMS[model_name]:
            self.features = MlModelStrategyPipelineOptimizer.HYPER_PARAMS[model_name]["features"]
            logging.debug('Subset of features: {}'.format(self.features))
        else:
            self.features = []


    def sample_strategy_parameters(self):
        return {
            'ewm_lengths': random.choice(self.ewm_lengths),
            't1_length': random.choice(self.t1_lengths),
            'pt': random.choice(self.pts),
            'sl': random.choice(self.sls),
            'min_ret': random.choice(self.min_rets),
            'vol_len': random.choice(self.volatility_lengths),
            'cpus': random.choice(self.cpus),
            'n_estimators': random.randint(self.n_estimators[0], self.n_estimators[1]),
            'max_features': random.uniform(self.max_features[0], self.max_features[1]),
            'max_depth': random.choice(self.max_depth),
            'cv_splits': random.choice(self.cv_splits),
            'embargo': random.choice(self.embargo),
            'learning_rate': random.uniform(self.learning_rate[0], self.learning_rate[1]),
            'features': self.features,
        }
#-------------------------------------------------------------------------------

class MlModelOptimStrategyFeatImportance(MlModelStrategyPipelineOptimizer):
    def __init__(self, df, cpus, seed, model_name):
        MlModelStrategyPipelineOptimizer.__init__(self, df, cpus, seed, model_name)
        self.importance = None
        self.mean_f1_score = None

    def feature_importance(self, n_samples, StrategyPipelineType):
        for i in range(n_samples):
            logging.debug(
                'Building model and strategy {} out of {}'.format(i+1, n_samples))
            # Gets the parameters for the strategy
            strategy_params = self.sample_strategy_parameters()
            # Builds the strategy
            strategy_pipeline = StrategyPipelineType(
                self.df.copy(deep=True), strategy_params)
            # Run it!
            strategy_pipeline.compute_bets()
            strategy_pipeline.build_bet_sizing_model()
            strategy_pipeline.prepare_features()

            self.importance, self.mean_f1_score = featImpMDA2(strategy_pipeline.model,
                                                              strategy_pipeline.Xy.copy()[strategy_pipeline.Xy.columns.difference(['bin', 't1', 'Close'])],
                                                              strategy_pipeline.Xy.copy().bin,
                                                              strategy_pipeline.params['cv_splits'],
                                                              strategy_pipeline.sample_weights_df['w'],
                                                              strategy_pipeline.Xy.copy()['t1'],
                                                              strategy_pipeline.params['embargo'],
                                                              scoring='neg_log_loss')
