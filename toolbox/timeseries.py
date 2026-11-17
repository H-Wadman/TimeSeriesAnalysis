import sys
from collections.abc import Sized
from typing import cast

import matplotlib.pyplot as plt
from numpy.typing import ArrayLike

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



def plot_pacf(): ...
