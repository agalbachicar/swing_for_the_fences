import numpy as np


def rsi(close, window_length=14):
    """
    Computes the RSI index over a price series sampled at constant time step.

    @param close A series that contains the close price values.
    @param window_length The number of events to apply an average.
    @return A series that contains the RSI index of the close price series.
    """
    # Get the difference in price from previous step
    delta = close.diff()
    # Get rid of the first row, which is NaN since it did not have a previous
    # row to calculate the differences
    delta = delta[1:]
    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    # Calculate the SMA
    roll_up = up.rolling(window_length).mean()
    roll_down = down.abs().rolling(window_length).mean()
    # Calculate the RSI based on SMA
    relative_strength = roll_up / roll_down
    return 100.0 - (100.0 / (1.0 + relative_strength))


def log_ret(close):
    """
    Computes the log of the returns inter sample.

    @param close A series that contains the close price values.
    @return A series that contains the log of the inter sample returns
    """
    return np.log(close).diff()


def autocorr(close, window_length=50, lag=1):
    """
    Computes the auto correlation of the price series.

    @param close A series that contains the close price values.
    @param window_length The window size.
    @param lag The lag positions to consider when computing the auto correlation.
    @return A series that contains the auto correlation.
    """
    log_ret_series = log_ret(close)
    return \
        log_ret_series.rolling(window=window_length,
                               min_periods=window_length,
                               center=False).apply(lambda x: x.autocorr(lag=1), raw=False)


def volatility(close, window_length=50):
    """
    Computes the rolling volatility of prices for a given window length.

    @param close A series that contains the close price values.
    @param window_length The window size.
    @return A series of price volatility.
    """
    log_ret_series = log_ret(close)
    return \
        log_ret_series.rolling(window=window_length,
                               min_periods=window_length,
                               center=False).std()
