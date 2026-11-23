import sys
from collections.abc import Sized
from typing import cast

import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from scipy import signal
from tsa_lth.modelling import estimateARMA

type TimeSeries = ArrayLike


def plot_timeseries(title, x_label="x", y_label="y"):
    plt.figure()
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.plot()


def plot_acf(ts: TimeSeries, max_lag = None):
    ts_len = len(cast(Sized, ts))
    max_recommended_lags = ts_len // 4

    if max_lag is None:
        max_lag = max_recommended_lags 
    elif max_lag > max_recommended_lags:
        print("Warning: ACF will be calculated for lags larger than N/4; this is not recommended, see p. 47", file=sys.stderr)


def my_filter(A, C, y):
    """Obtain the residual of ARMA filtering a process.

    Takes in a dataset as well as a specification of the AR and MA model 
    components. The provided ARMA model is then used or calculated and is then
    used to filter the data in reverse, in order to compute the error residual 
    and then discarding initial samples to get rid of initialisation effects.

    Args:
        A: Either the coefficients of the AR polynomial to use, or the model
            order to estimate parameters for.
        C: Either the coefficients of the MA polynomial to use, or the model
            order to estimate parameters for.
        y: The data for the model.

    Returns:
        The error residual with initial samples removed.
    """
    model = estimateARMA(y, A, C, plot=False)
    assert model is not None
    e_hat = signal.lfilter(model.A, model.C, y)
    na = len(model.A) - 1
    assert na > 0
    return e_hat[na:]

def plot_pacf(): ...
