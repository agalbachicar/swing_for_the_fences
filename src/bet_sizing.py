import numpy as np
import pandas as pd

from scipy import interp
from scipy.stats import norm

from mpfin import mpPandasObj

def getSignal(events, stepSize, prob, pred, numClasses, numThreads, **kwargs):
    '''
    Computes a bet size out of the probability of the predictions that follows
    a discretized sigmoid function of 1 / stepSize steps.

    The discretization step in the pipeline will run in numThreads threads.

    See Advances in Financial Analytics, snippet 10.1, page 143.

    @param events Determines whether the probability comes from a meta labeling
        or not. When events has 'side', the return value takes the sign of the
        side.
    @param stepSize A number in [0, 1] that is the inverse of the number of
        steps.
    @param prob The probabilities of the predictions pred.
    @param pred The predictions. It could be [-1, 1] or [-1, 0, 1] or other set.
    @param numClasses The number of possible values in the predictions.
    @param numThreads The number of threads that will be open to run the
        discretization.
    @return A series of bet sizes.
    '''
    # Get signals from predictions
    if prob.shape[0] == 0:
        return pd.Series()
    # 1) Generate signals from multinomial classification (one vs rest, OvR)
    # t-value of OvR
    # The book has the following line which is incorrect. Denominator should be:
    # sqrt(1/numClasses * (1 - numClasses)) because it is a proportion test
    # statistic.
    # signal0 = (prob - 1. / numClasses) / (prob * (1. - prob))**0.5
    d = (1./numClasses * (1. - 1./numClasses))**0.5
    signal0 = (prob - 1. / numClasses) / d
    # signal = side * size
    signal0 = pred * (2 * norm._cdf(signal0) - 1)
    # meta-labeling
    if 'side' in events:
        signal0 *= events.loc[signal0.index, 'side']
    # 2) Compute average signal among those concurrently open
    df0 = signal0.to_frame('signal').join(events[['t1']], how='left')
    df0 = avgActiveSignals(df0, numThreads)
    signal1 = discreteSignal(signal0=df0, stepSize=stepSize)
    return signal1


def avgActiveSignals(signals, numThreads):
    '''
    Multiple bets can run in parallel (long and short bets) so it would be wise
    to have a smooth and consistent position at each time as a consequence of
    all the running bets.

    See Advances in Financial Analytics, snippet 10.2, page 144.

    @param signals A data frame with the bet size at each [t0-t1] for each bet.
    @param numThreads The number of threads in which averaging will happen.
    @return A series with the bet sizes in signal with an average applied.
    '''
    # Compute the average signal among those active
    # 1) time points where signals change (either starts or one ends)
    tPnts = set(signals['t1'].dropna().values)
    tPnts = tPnts.union(signals.index.values)
    tPnts = list(tPnts)
    tPnts.sort()
    out = mpPandasObj(mpAvgActiveSignals, ('molecule', tPnts), numThreads,
                      signals=signals)
    return out


def mpAvgActiveSignals(signals, molecule):
    '''
    At time loc, average signal among those still active.
    Signal is active if:
        a) issued before or at loc AND
        b) loc before signal's endtime, or endtime is still unknown (NaT)

    See Advances in Financial Analytics, snippet 10.2, page 144.
    '''
    out = pd.Series()
    for loc in molecule:
        df0 = (signals.index.values <= loc) & (
            (loc < signals['t1']) | pd.isnull(signals['t1']))
        act = signals[df0].index
        if len(act) > 0:
            out[loc] = signals.loc[act, 'signal'].mean()
        else:
            out[loc] = 0  # no signals active at this time
    return out


def discreteSignal(signal0, stepSize):
    '''
    Applies the discretization.

    See Advances in Financial Analytics, snippet 10.3, page 145.

    @param signal0 The bet size to discretize
    @param stepSize The steps size (or inverse of number of steps) to have.
    @return A discretized step size.
    '''
    signal1 = (signal0 / stepSize).round() * stepSize  # discretize
    signal1[signal1 > 1] = 1  # cap
    signal1[signal1 < -1] = -1  # floor
    return signal1

# The following set of functions are used with custom calibration of the
# positions with sigmoid functions that account for maximum dollar positions
# per bet.


def betSize(w, x):
    '''
    See Advances in Financial Analytics, snippet 10.4, page 144.
    '''
    return x * (w + x**2)**-0.5


def getTPos(f, w, mP, maxPos):
    '''
    See Advances in Financial Analytics, snippet 10.4, page 144.
    '''
    return int(betSize(w, f - mP) * maxPos)


def invPrince(f, w, m):
    '''
    See Advances in Financial Analytics, snippet 10.4, page 144.
    '''
    return f - m * (w / (1 - m**2))**0.5


def limitPrice(tPos, pos, f, w, maxPos):
    '''
    See Advances in Financial Analytics, snippet 10.4, page 144.
    '''
    sgn = (1 if tPos >= pos else -1)
    lP = 0
    for j in range(abs(pos + sgn), abs(tPos + 1)):
        lP += invPrince(f, w, j / float(maxPos))
    lP /= tPos - pos
    return lP


def getW(x, m):
    '''
    See Advances in Financial Analytics, snippet 10.4, page 144.
    '''
    # 0 < alpha < 1
    return x**2 * (m**-2 - 1)
