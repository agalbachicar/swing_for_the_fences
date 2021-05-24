import numpy as np
import pandas as pd


def compute_bsadf(log_p, minSL, maxSL, constant, lags):
    df0 = pd.DataFrame(index=log_p.index, columns=['bsadf'])
    for j in range(minSL+1, log_p.shape[0]):
        res = get_bsadf(log_p[:j], minSL, maxSL, constant, lags)
        df0.loc[res['Time']] = res['bsadf']
    return df0


def get_bsadf(logP, minSL, maxSL, constant, lags):
    '''
    See Advances in Financial Analytics, snippet 17.1, page 258.

    @param logP A series containing log-prices.
    @param minSL the minimum sample length (tau), used by the final regression.
    @param constant The regression's time trend component. When:
        - 'nc': no time trend, only a constant.
        - 'ct': a constant plus a linear polynomial time trend.
        - 'ctt': a constant plus a second-degree polynomial time trend.
    @param lags The number of lags used in the ADF specification.
    '''
    y, x = getYX(logP, constant=constant, lags=lags)
    # Wraps the range in such a way that enough information is taken into
    # account and the problem keeps tractable.
    max_range = y.shape[0]+lags-minSL+1
    min_range = max(0, max_range - maxSL)
    startPoints, bsadf, allADF = range(min_range, max_range), None, []
    for start in startPoints:
        y_, x_ = y[start:], x[start:]
        bMean_, bStd_ = getBetas(y_, x_)
        bMean_, bStd_ = bMean_[0, 0], bStd_[0, 0]**0.5
        if np.isnan(bMean_):
            raise ValueError(
                'bMean_ is nan. y_: {} | x_: {} | bMean_: {}'.format(y_, x_, bMean_))
        if np.isnan(bStd_):
            raise ValueError(
                'bStd_ is nan. y_: {} | x_: {} | bStd_: {}'.format(y_, x_, bStd_))
        allADF.append(bMean_ / bStd_)
        # if not bsadf or allADF[-1] > bsadf: bsadf = allADF[-1]
        # if not bsadf: bsadf = allADF[-1]
        # elif allADF[-1] > bsadf: bsadf = allADF[-1]
    bsadf = max(allADF)
    out = {'Time': logP.index[-1], 'bsadf': bsadf}
    return out


def getYX(series, constant, lags):
    '''
    See Advances in Financial Analytics, snippet 17.2, page 258.
    '''
    series_ = series.diff().dropna()
    x = lagDF(series_, lags).dropna()
    x.iloc[:, 0] = series.values[-x.shape[0]-1:-1, 0]  # Lagged level
    y = series_.iloc[-x.shape[0]:].values
    if constant != 'nc':
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
        if constant == 'ct':
            trend = np.arange(x.shape[0]).reshape(-1, 1)
            x = np.append(x, trend, axis=1)
        if constant == 'ctt':
            trend = np.arange(x.shape[0]).reshape(-1, 1)
            x = np.append(x, trend**2, axis=1)
    return y, x


def lagDF(df0, lags):
    '''
    See Advances in Financial Analytics, snippet 17.3, page 259.
    '''
    df1 = pd.DataFrame()
    if isinstance(lags, int):
        lags = range(lags + 1)
    else:
        lags = [int(lag) for lag in lags]
    for lag in lags:
        df_ = df0.shift(lag).copy(deep=True)
        df_.columns = [str(i)+'_'+str(lag) for i in df_.columns]
        df1 = df1.join(df_, how='outer')
    return df1


def getBetas(y, x):
    '''
    See Advances in Financial Analytics, snippet 17.4, page 259.
    '''
    xy = np.dot(x.T, y)
    xx = np.dot(x.T, x)
    xxinv = np.linalg.inv(xx)
    bMean = np.dot(xxinv, xy)
    err = y-np.dot(x, bMean)
    bVar = np.dot(err.T, err) / (x.shape[0] - x.shape[1]) * xxinv
    return bMean, bVar
