from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier
from scipy.stats import rv_continuous, kstest


def clfHyperFit(feat, lbl, t1, pipe_clf, param_grid, cv=3,
                bagging=[0, None, 1.0], rndSearchIter=0, n_jobs=-1, pctEmbargo=0,
                **fit_params):
    '''
    See Advances in Financial Analytics, snippet 9.3 page 131.
    '''
    if set(lbl.values) == {0, 1}:
        scoring = 'f1'  # F1 for meta-labeling
    else:
        scoring = 'neg_log_loss'  # Symmetric towards all classes

    # 1) Hyperparameter searching, on train data
    inner_cv = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)
    if rndSearchIter == 0:
        gs = GridSearchCV(estimator=pipe_clf, param_grid=param_grid,
                          scoring=scoring, cv=inner_cv, n_jobs=n_jobs, iid=False)
    else:
        gs = RandomizedSearchCV(estimator=pipe_clf,
                                param_distributions=param_grid, scoring=scoring, cv=inner_cv,
                                n_jobs=n_jobs, iid=False, n_iter=rndSearchIter)
    gs = gs.fit(feat, lbl, **fit_params).best_estimator_
    # 2) Fit validated model on the entirety of the data
    if bagging[1] > 0:
        gs = BaggingClassifier(bare_estimator=TheNewPipe(gs.steps),
                               n_estimators=int(bagging[0]), max_samples=float(bagging[1]),
                               max_features=float(bagging[2]), n_jobs=n_jobs)
        gs = gs.fit(
            feat, lbl, sample_weight=fit_params[gs.base_estimator.steps[-1][0] + '__sample_weight'])
        gs = Pipeline([('bag', gs)])
    return gs


class FAPipeline(Pipeline):
    '''
    See Advances in Financial Analytics, snippet 9.2 page 131.
    '''

    def fit(self, X, y, sample_weight=None, **fit_params):
        if sample_weight is not None:
            fit_params[self.steps[-1][0]+'__sample_weight'] = sample_weight
        return super(FAPipeline, self).fit(X, y, **fit_params)


class LogUniformGen(rv_continuous):
    # random numbers log-uniformly distributed between 1 and e
    def _cdf(self, x):
        return np.log(x / self.a) / np.log(self.b / self.a)


def logUniform(a=1, b=np.exp(1)):
    '''
    See Advances in Financial Analytics, snippet 9.4 page 133.
    '''
    return LogUniformGen(a=a, b=b, name='logUniform')
