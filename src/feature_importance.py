import numpy as np
import pandas as pd

from cv import PurgedKFold
from sklearn.metrics import log_loss, accuracy_score, f1_score

def featImpMDI(fit, featNames):
    '''
    Computes the Mean Decrease Impurity feature importance
    for a tree based classifier.

    See Advances in Financial Analytics, snippet 8.2, page 115.

    @note Assumes that fit was created with max_features=1, i.e. one
          variable per decision node.
    @param fit A tree based classifier.
    @param featNames A list with feature names.
    @return A data frame with the mean and std importance per feature.
    '''
    df0 = {i: tree.feature_importances_ for i,
           tree in enumerate(fit.estimators_)}
    df0 = pd.DataFrame.from_dict(df0, orient='index')
    df0.columns = featNames
    df0 = df0.replace(0, np.nan)  # Because max_features = 1
    imp = pd.concat({'mean': df0.mean(), 'std': df0.std()
                     * df0.shape[0]**-0.5}, axis=1)
    imp /= imp['mean'].sum()
    return imp


def featImpMDA(clf, X, y, cv, sample_weight, t1, pctEmbargo, scoring='neg_log_loss'):
    '''
    See Advances in Financial Analytics, snippet 8.3, page 116.
    '''
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise Exception('Unsupported scoring method <{}>.'.format(scoring))
    cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)  # purged cv
    src0, src1 = pd.Series(), pd.DataFrame(columns=X.columns)
    for i, (train, test) in enumerate(cvGen.split(X=X)):
        X0, y0, w0 = X.iloc[train, :], y.iloc[train], sample_weight.iloc[train]
        X1, y1, w1 = X.iloc[test, :], y.iloc[test], sample_weight.iloc[test]
        fit = clf.fit(X=X0, y=y0, sample_weight=w0.values)
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X1)
            src0.loc[i] = - log_loss(y1, prob,
                                     sample_weight=w1.values, labels=clf.classes_)
        else:
            pred = fit.predict(X1)
            src0.loc[i] = accuracy_score(y1, pred, sample_weight=w1.values)
        for j in X.columns:
            X1_ = X1.copy(deep=True)
            np.random.shuffle(X1_[j].values)  # permutation of a single column
            if scoring == 'neg_log_loss':
                prob = fit.predict_proba(X1_)
                src1.loc[i, j] = - \
                    log_loss(y1, prob, sample_weight=w1.values,
                             labels=clf.classes_)
            else:
                pred = fit.predict(X1_)
                src1.loc[i, j] = accuracy_score(
                    y1, pred, sample_weight=w1.values)
    imp = (-src1).add(src0, axis=0)
    if scoring == 'neg_log_loss':
        imp = imp / -src1
    else:
        imp = imp / (1. - src1)
    imp = pd.concat({'mean': imp.mean(), 'std': imp.std()
                     * imp.shape[0]**-0.5}, axis=1)
    return imp, src0.mean()

def featImpMDA2(clf, X, y, cv, sample_weight, t1, pctEmbargo, scoring='neg_log_loss'):
    '''
    See Advances in Financial Analytics, snippet 8.3, page 116.
    '''
    if scoring not in ['neg_log_loss', 'accuracy', 'f1_score']:
        raise Exception('Unsupported scoring method <{}>.'.format(scoring))
    cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)  # purged cv
    src0, src1 = pd.Series(), pd.DataFrame(columns=X.columns)
    for i, (train, test) in enumerate(cvGen.split(X=X)):
        X0, y0, w0 = X.iloc[train, :], y.iloc[train], sample_weight.iloc[train]
        X1, y1, w1 = X.iloc[test, :], y.iloc[test], sample_weight.iloc[test]
        fit = clf.fit(X=X0, y=y0, sample_weight=w0.values)
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X1)
            src0.loc[i] = - log_loss(y1, prob,
                                     sample_weight=w1.values, labels=clf.classes_)
        elif scoring == 'accuracy':
            pred = fit.predict(X1)
            src0.loc[i] = accuracy_score(y1, pred, sample_weight=w1.values)
        else:
            pred = fit.predict(X1)
            src0.loc[i] = f1_score(y1, pred)

        for j in X.columns:
            X1_ = X1.copy(deep=True)
            np.random.shuffle(X1_[j].values)  # permutation of a single column
            if scoring == 'neg_log_loss':
                prob = fit.predict_proba(X1_)
                src1.loc[i, j] = - \
                    log_loss(y1, prob, sample_weight=w1.values,
                             labels=clf.classes_)
            elif scoring == 'accuracy':
                pred = fit.predict(X1_)
                src1.loc[i, j] = accuracy_score(
                    y1, pred, sample_weight=w1.values)
            else:
                pred = fit.predict(X1_)
                src1.loc[i, j] = f1_score(
                    y1, pred, sample_weight=w1.values)
    imp = (-src1).add(src0, axis=0)
    if scoring == 'neg_log_loss':
        imp = imp / -src1
    elif scoring == 'accuracy':
        imp = imp / (1. - src1)
    else:
        imp = imp / (1. - src1)
    imp = pd.concat({'mean': imp.mean(), 'std': imp.std()
                     * imp.shape[0]**-0.5}, axis=1)
    return imp, src0.mean()

def featImpSFI(featNames, clf, trnsX, cont, scoring, cvGen):
    '''
    See Advances in Financial Analytics, snippet 8.4, page 118.
    '''
    imp = pd.DataFrame(columns=['mean', 'std'])
    for featName in featNames:
        df0 = cvScore(clf, X=trnsX[[featName]], y=cont['bin'],
                      sample_weight=cont['w'], scoring=scoring, cvGen=cvGen)
        imp.loc[featName, 'mean'] = df0.mean()
        imp.loc[featName, 'std'] = df0.std() * df0.shape[0]**-0.5
    return imp

# def get_eVec(dot, varThres):
#     '''
#     See Advances in Financial Analytics, snippet 8.5, page 119
#     '''
#     # Compute eVec from dot product matrix, reduce dimension
#     eVal, eVec = np.linal.eigh(dot)
#     idx = eval.argsort()[::-1] # arguments for sorting eVal desc
#     eVal, eVec = eVal[idx], eVec[:, idx]
#     #2) Only positive eVals
#     eVal = pd.Series(eval, index=[['PC_' + str(i+1)] for i in range(eVal.shape[0])])
#     eVec = pd.DataFrame(eVec, index=dot.index, columns=eVal.index)
#     eVec = eVec.loc[:, eVal.index]
#     # 3) Reduce dimension, from PCs
#     cumVar = eVal.cumsum() / eVal.sum()
#     dim = cumVar.values.searchsorted(varThres)
#     eVal, eVec = eVal.iloc[:dim+1], eVec.iloc[:, :dim+1]
#     return eVal, eVec


def orthoFeats(dfX, varThres=0.95):
    '''
    See Advances in Financial Analytics, snippet 8.5, page 119
    '''
    # Given a dataframe dfX of features, compute orthofeatures dfP
    dfZ = dfX.sub(dfX.mean(), axis=1).div(dfX.std(), axis=1)  # standardize
    dot = pd.DataFrame(np.dot(dfZ.T, dfZ),
                       index=dfX.columns, columns=dfX.columns)
    eVal, eVec = get_eVec(dot, varThres)
    dfP = np.dot(dfZ, eVec)
    return dfP


def featImportance(trnsX, cont, n_estimators=1000, cv=10, max_samples=1.,
                   numThreads=24, pctEmbargo=0, scoring='accuracy', method='SFI',
                   minWLeaf=0., **kargs):
    '''
    See Advances in Financial Analytics, snippet 8.8, page 123
    '''
    n_jobs = (-1 if numThreads > 1 else 1)
    # 1) Prepare classifier, cv, max_features=1, to prevent masking
    clf = DecisionTreeClassifier(criterion='entropy', max_features=1,
                                 class_weight='balanced', min_weight_fraction_leaf=minWLeaf)
    clf = BaggingClassifier(base_estimator=clf, n_estimators=n_estimators,
                            max_features=1., max_samples=max_samples, oob_score=True,
                            n_jobs=n_jobs)
    fit = clf.fit(X=trnsX, y=cont['bin'], sample_weight=cont['w'].values)
    oob = fit.oob_score_
    if method == 'MDI':
        imp = featImpMDI(fit, trnsX.columns)
        oos = cvScore(clf, X=trnsX, y=cont['bin'], cv=cv,
                      sample_weight=cont['w'], t1=cont['t1'], pctEmbargo=pctEmbargo,
                      scoring=scoring).mean()
    elif method == 'MDA':
        imp, oos = featImpMDA(clf, X=trnsX, y=cont['bin'], cv=cv,
                              sample_weight=cont['w'], t1=cont['t1'], pctEmbargo=pctEmbargo,
                              scoring=scoring)
    elif method == 'SFI':
        cvGen = PurgedKFold(n_splits=cv, t1=cont['t1'], pctEmbargo=pctEmbargo)
        oos = cvScore(clf, X=trnsX, y=cont['bin'], cv=cv,
                      sample_weight=cont['w'], t1=cont['t1'], pctEmbargo=pctEmbargo,
                      scoring=scoring).mean()
        clf.n_jobs = 1  # parallelize featImpSFI rather than clf
        imp = mpPandasObj(featImpSFI, ('featNames', trnsX.columns), numThreads,
                          clf=clf, trnsX=trnsX, cont=cont, scoring=scoring, cvGen=cvGen)
    return imp, oob, oos


def plotFeatImp(pathOut, imp, oob, oos, method, tag=0, simNum=9, **kwargs):
    '''
    See Advances in Financial Analytics, snippet 8.10, page 124
    '''
    plt.figure(figsize=(15, imp.shape[0]/5.))
    imp = imp.sort_values('mean', ascending=True)
    ax = imp['mean'].plot(kind='barh', color='b', alpha=0.25, xerr=imp['std'],
                          error_kw={'ecolor': 'r'})
    if method == 'MDI':
        plt.xlim([0, imp.sum(axis=1).max()])
        plt.axvline(1./imp.shape[0], lineWidth=1,
                    color='r', linestyle='dotted')
        ax.get_yaxis().set_visible(False)
    for i, j in zip(ax.patches, imp.index):
        ax.text(i.get_width()/2, i.get_y()+i.get_height()/2, j, ha='center',
                va='center', color='black')
    plt.title('tag={} | simNum={} | oob={} | oos={}'.format(
        tag, simNum, round(oob, 4), round(oos, 4)))
    if pathOut:
        plt.savefig(pathOut+'featImportance_'+str(simNum)+'.png', dpi=100)
    plt.plot()
