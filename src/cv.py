import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection._split import _BaseKFold
from sklearn.metrics import roc_curve, classification_report, log_loss, accuracy_score, recall_score, auc, average_precision_score, f1_score


def getTrainTimes(t1, testTimes):
    '''
    Given testTimes, find the times of the training observations.

    There are three conditions that would make a sample to be dropped. Let i be
    the index of a train sample and j the index of a test sample. Let 0,1 be the
    start and end of a sample, then:

    - t_{j,0} <= t_{i,0} <= t_{j,1}            --> train starts between test
    - t_{j,0} <= t_{i,1} <= t_{j,1}            --> train ends between test
    - t_{i,0} <= t_{j,0} <= t_{j,1} <= t_{i,1} --> test is contained in train

    See Advances in Financial Analytics, snippet 7.1, page 106.

    @param t1 A pandas Series where the index tells when the observation started
            and the value when it ended.
    @param testTimes Times of testing observations.
    @return A purged t1.
    '''
    trn = t1.copy(deep=True)
    for i, j in testTimes.iteritems():
        # Train stars with index
        df0 = trn[(i <= trn.index) & (trn.index <= j)].index
        # Train ends within test
        df1 = trn[(i <= trn) & (trn <= j)].index
        # Train envelops test
        df2 = trn[(trn.index <= i) & (j <= trn)].index
        # Removes the union of the previous three data frames.
        trn = trn.drop(df0.union(df1).union(df2))
    return trn


def getEmbargoTimes(times, pctEmbargo):
    '''
    Drops 2 * pctEmbargo percentage of samples at the beginning and end of times
    to further prevent leakage.

    See Advances in Financial Analytics, snippet 7.2, page 108.

    @param times A data series of times to drop labels from.
    @param pctEmbargo The percentage of times's size to drop.
    @return A copy of times but with dropped items at the beginning and end
        because of pctEmbargo.
    '''
    step = int(times.shape[0] * pctEmbargo)
    if step == 0:
        mbrg = pd.Series(times, index=times)
    else:
        mbrg = pd.Series(times[step:], index=times[:-step])
        mbrg = mbrg.append(pd.Series(times[-1], index=times[-step:]))
    return mbrg


class PurgedKFold(_BaseKFold):
    '''
    Extend KFold class to work with labels that span intervals
    The train is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle=False), w/o training samples in
    between

    See Advances in Financial Analytics, snippet 7.3, page 109.
    '''

    def __init__(self, n_splits=3, t1=None, pctEmbargo=0.):
        if not isinstance(t1, pd.Series):
            raise ValueError('Label Through Dates must be a pd.Series')
        super(PurgedKFold, self).__init__(
            n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pctEmbargo = pctEmbargo

    def split(self, X, y=None, groups=None):
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError('X and ThruDateValues must have the same index')

        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0]*self.pctEmbargo)
        test_starts = [(i[0], i[-1]+1)
                       for i in np.array_split(np.arange(X.shape[0]), self.n_splits)]
        for i, j in test_starts:
            t0 = self.t1.index[i]  # start of test set
            test_indices = indices[i:j]
            maxT1Idx = self.t1.index.searchsorted(
                self.t1.index[test_indices].max())
            train_indices = self.t1.index.searchsorted(
                self.t1[self.t1 <= t0].index)
            if maxT1Idx < X.shape[0]:  # right train (with embargo)
                train_indices = np.concatenate(
                    (train_indices, indices[maxT1Idx+mbrg:]))
            yield train_indices, test_indices


def cvScore(clf, X, y, sample_weight, scoring='neg_log_loss',
            t1=None, cv=None, cvGen=None, pctEmbargo=None):
    '''
    Scores a purged k fold cross validation training using either neg_log_loss
    or accuracy_score.

    See Advances in Financial Analytics, snippet 7.4, page 110.

    @param clf Classification model to fit.
    @param X Model parameters.
    @param y Classification values for X
    @param sample_weight Uniqueness weights of X.
    @param t1 Triple barrier times.
    @param cv Number of cross validation splits.
    @param cvGen A _BaseKFold class. When None, PurgedKFold is used instead.
    @param pctEmbargo The percentage of embargo on samples to use.
    @return An array with the score result per cross validation split.
    '''
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise Exception('wrong scoring method.')
    idx = pd.IndexSlice
    if cvGen is None:
        cvGen = PurgedKFold(n_splits=cv, t1=t1,
                            pctEmbargo=pctEmbargo)  # purged
    score = []
    for train, test in cvGen.split(X=X):
        fit = clf.fit(X=X.iloc[idx[train], :], y=y.iloc[idx[train]],
                      sample_weight=sample_weight.iloc[idx[train]].values)
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X.iloc[idx[test], :])
            score_ = -log_loss(y.iloc[idx[test]], prob,
                               sample_weight=sample_weight.iloc[idx[test]].values,
                               labels=clf.classes_)
        else:
            pred = fit.predict(X.iloc[idx[test], :])
            score_ = accuracy_score(y.iloc[idx[test]], pred,
                                    sample_weight=sample_weight.iloc[idx[test]].values)
        score.append(score_)
    return np.array(score)


def crossValPlot(skf, classifier, X_, y_):
    '''
    Splits X_ and y_ with skf and fits the classifier at the same time that
    plots the ROC result. It leads to a ROC plot with multiple curves (one per
    CV split) and provides a mean result for the final train result.

    Use this method without PurgedKFold

    See https://github.com/BlackArbsCEO/Adv_Fin_ML_Exercises/blob/master/notebooks/07.%20Cross%20Validation%20in%20Finance.ipynb

    @param skf A _BaseKFold instance. 
    @param classifier A classifier to be trained with skf.
    @param X_ The parameters of the classifier.
    @param y_ The outputs of the parameters.
    '''
    X = np.asarray(X_)
    y = np.asarray(y_)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    f, ax = plt.subplots(figsize=(10, 7))
    i = 0
    for train, test in skf.split(X, y):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3,
                label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    ax.legend(bbox_to_anchor=(1, 1))
    ax.grid()


def crossValPlot2(skf, classifier, X, y, sample_weights=None):
    '''
    Splits X_ and y_ with skf and fits the classifier at the same time that
    plots the ROC result. It leads to a ROC plot with multiple curves (one per
    CV split) and provides a mean result for the final train result.

    Use this method with PurgedKFold

    See https://github.com/BlackArbsCEO/Adv_Fin_ML_Exercises/blob/master/notebooks/07.%20Cross%20Validation%20in%20Finance.ipynb

    @param skf A PurgedKFold instance. 
    @param classifier A classifier to be trained with skf.
    @param X The parameters of the classifier.
    @param y The outputs of the parameters.
    @param sample_weights The weights to use for each sample.
    '''
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    idx = pd.IndexSlice
    f, ax = plt.subplots(figsize=(10, 7))
    i = 0
    for train, test in skf.split(X, y):
        if sample_weights is not None:
            probas_ = (classifier.fit(X.iloc[idx[train]], y.iloc[idx[train]], sample_weights.iloc[idx[train]].w.values)
                       .predict_proba(X.iloc[idx[test]]))
        else:
            probas_ = (classifier.fit(X.iloc[idx[train]], y.iloc[idx[train]])
                       .predict_proba(X.iloc[idx[test]]))
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y.iloc[idx[test]], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3,
                label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    ax.legend(bbox_to_anchor=(1, 1))
    ax.grid()


def crossValidationTrain(skf, classifier, X, y, sample_weights=None):
    '''
    Splits X_ and y_ with skf and fits the classifier at the same time that
    plots the ROC result. It leads to a ROC plot with multiple curves (one per
    CV split) and provides a mean result for the final train result.

    Use this method with PurgedKFold

    See https://github.com/BlackArbsCEO/Adv_Fin_ML_Exercises/blob/master/notebooks/07.%20Cross%20Validation%20in%20Finance.ipynb

    @param skf A PurgedKFold instance. 
    @param classifier A classifier to be trained with skf.
    @param X The parameters of the classifier.
    @param y The outputs of the parameters.
    @param sample_weights The weights to use for each sample.
    '''
    aucs = []
    accuracies = []
    oob_scores = []
    recalls = []
    precisions = []
    class_reports = []
    f1_scores = []
    log_loss_scores = []
    idx = pd.IndexSlice
    for train, test in skf.split(X, y):
        if sample_weights is not None:
            try:
                probas_ = (classifier.fit(X.iloc[idx[train]], y.iloc[idx[train]], sample_weights.iloc[idx[train]].w.values)
                           .predict_proba(X.iloc[idx[test]]))
            except ValueError:
                continue
        else:
            try:
                probas_ = (classifier.fit(X.iloc[idx[train]], y.iloc[idx[train]])
                           .predict_proba(X.iloc[idx[test]]))
            except ValueError:
                continue
        if probas_.shape[1] == 1:
            logging.debug(
                'Discarding model because of lack of samples to evaluate for either true or false.')
            continue
        y_pred_prob = probas_[:, 1]
        y_pred = classifier.predict(X.iloc[idx[test]])

        if hasattr(classifier, 'oob_score_'):
            oob_scores.append(classifier.oob_score_)
        class_reports.append(classification_report(y.iloc[idx[test]], y_pred))
        accuracies.append(accuracy_score(y.iloc[idx[test]], y_pred))
        recalls.append(recall_score(y.iloc[idx[test]], y_pred))
        f1_scores.append(f1_score(y.iloc[idx[test]], y_pred))
        log_loss_scores.append(-log_loss(y.iloc[idx[test]], y_pred_prob))
        precisions.append(average_precision_score(
            y.iloc[idx[test]], y_pred_prob))
        # Compute ROC curve and area the curve
        fpr, tpr, _ = roc_curve(y.iloc[idx[test]], y_pred_prob)
        aucs.append(auc(fpr, tpr))
    if hasattr(classifier, 'oob_score_'):
        return aucs, accuracies, oob_scores, recalls, precisions, class_reports, f1_scores, log_loss_scores
    else:
        return aucs, accuracies, None, recalls, precisions, class_reports, f1_scores, log_loss_scores
