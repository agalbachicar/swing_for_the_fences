import numpy as np
import pandas as pd


def mpNumCoEvents(closeIdx, t1, molecule):
    '''
    Compute the number of concurrent events per bar.

    Any event that starts before t1[modelcule].max() impacts the count.

    See Advances in Financial Analytics, snippet 4.1, page 60.

    @param closeIdx An index series of close prices. It must be related with `t1`.
    @param t1 An index series of vertical barrier events.
    @param molecule Event' series index.
           molecule[0] is the date of the first event on which the weight will be computed.
           molecule[-1] is the date of the last event on which the weight will be computed.
    @return A series with overlaps counts whose index is the same as t1.
    '''
    # 1) find events that span the period [molecule[0],molecule[-1]]
    # unclosed events still must impact other weights
    t1 = t1.fillna(closeIdx[-1])
    t1 = t1[t1 >= molecule[0]]  # events that end at or after molecule[0]
    # events that start at or before t1[molecule].max()
    t1 = t1.loc[:t1[molecule].max()]
    # 2) count events spanning a bar
    iloc = closeIdx.searchsorted(np.array([t1.index[0], t1.max()]))
    count = pd.Series(0, index=closeIdx[iloc[0]:iloc[1]+1])
    for tIn, tOut in t1.iteritems():
        count.loc[tIn:tOut] += 1.
    return count.loc[molecule[0]:t1[molecule].max()]


def mpSampleTW(t1, numCoEvents, molecule):
    '''
    Computes the sample weight of an event as the reciprocal of the hsrmonic average
    of c_t over the event's lifespan.

    See Advances in Financial Analytics, snippet 4.2, page 62.

    @param t1 An index series of vertical barrier events.
    @param numCoEvents The output of mpNumCoEvents()
    @param molecule Event' series index.
    @return A series of weights for each event in t1.
    '''
    # Derive avg. uniqueness over the events lifespan
    wght = pd.Series(index=molecule)
    for tIn, tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn] = (1./numCoEvents.loc[tIn:tOut]).mean()
    return wght


def mpSampleW(t1, numCoEvents, close, molecule):
    '''
    Derive sample weight by return attribution

    See Advances in Financial Analytics, snippet 4.2, page 62.
    '''
    ret = np.log(close).diff()  # log-returns, so that they are additive
    wght = pd.Series(index=molecule)
    for tIn, tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn] = (ret.loc[tIn:tOut]/numCoEvents.loc[tIn:tOut]).sum()
    return wght.abs()


def getTimeDecay(tW, clfLastW=1.):
    '''
    Apply piecewise-linear decay to observed uniqueness (tW)
    Newest observation gets weight=1, oldest observation gets weight=clfLastW
    '''
    clfW = tW.sort_index().cumsum()
    if clfLastW >= 0:
        slope = (1.-clfLastW)/clfW.iloc[-1]
    else:
        slope = 1./((clfLastW+1)*clfW.iloc[-1])
    const = 1.-slope*clfW.iloc[-1]
    clfW = const+slope*clfW
    clfW[clfW < 0] = 0
    # print(const,slope)
    return clfW


def getIndMatrix(barIx, t1):
    '''
    Get Indicator matrix

    See Advances in Financial Analytics, snippet 4.3, page 64.
    '''
    indM = (pd.DataFrame(0, index=barIx, columns=range(t1.shape[0])))
    for i, (t0, t1) in enumerate(t1.iteritems()):
        indM.loc[t0:t1, i] = 1.
    return indM


def getAvgUniqueness(indM):
    '''
    Average uniqueness from indicator matrix

    See Advances in Financial Analytics, snippet 4.4, page 65.
    '''
    c = indM.sum(axis=1)  # concurrency
    u = indM.div(c, axis=0)  # uniqueness
    avgU = u[u > 0].mean()  # avg. uniqueness
    return avgU


def seqBootstrap(indM, sLength=None):
    '''
    Generate a sample via sequential bootstrap

    See Advances in Financial Analytics, snippet 4.5, page XX.
    '''
    if sLength is None:
        sLength = indM.shape[1]
    phi = []
    while len(phi) < sLength:
        avgU = pd.Series()
        for i in indM:
            indM_ = indM[phi+[i]]  # reduce indM
            avgU.loc[i] = getAvgUniqueness(indM_).iloc[-1]
        prob = avgU/avgU.sum()  # draw prob
        phi += [np.random.choice(indM.columns, p=prob)]
    return phi
