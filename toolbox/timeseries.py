import sys
from collections.abc import Sized
from enum import Enum
from typing import cast
from dataclasses import dataclass
from collections.abc import Callable

import matplotlib.pyplot as plt
import tsa_lth
from numpy.typing import ArrayLike
from scipy import signal
from tsa_lth.analysis import acf, pacf
from tsa_lth.modelling import estimateARMA
from tsa_lth.tests import check_if_normal, monti_test, normaltest

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


def my_filter(A: int | list[int], C: int | list[int], y):
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
    model = estimateARMA(y, A, C, plot=False) #type: ignore
    assert model is not None
    e_hat = signal.lfilter(model.A, model.C, y)
    na = len(model.A) - 1
    assert na >= 0
    return e_hat[na:]

class WhitenessTest(Enum):
    @dataclass
    class WhitenessInfo:
        name: str
        # test: Callable[[TimeSeries, int, float], tuple[bool, float, float]]
        test: Callable
    LJUNG_BOX = WhitenessInfo("Ljung-Box-Pierce", tsa_lth.tests.lbp_test)
    MCLEOD_LI = ("McLeod-Li", tsa_lth.tests.ml_test)
    MONTI = ("Monti", tsa_lth.tests.monti_test)
    SIGN_CHANGE = ("Sign change", sign_change_test)

def sign_change_test():
    ...


class NormalityTest(Enum):
    DAGOSTINO_PEARSON = "D'agostino-Pearson"

def verify_white_and_normal(
        model_residual: TimeSeries,
        whiteness_test=WhitenessTest.MONTI,
        normal_test=NormalityTest.DAGOSTINO_PEARSON
    ):
    tsa_lth.tests.whiteness_test(model_residual)
    acfEst = cast(ArrayLike, acf(model_residual, "auto", includeZeroLag=False))
    if is_AP_normal(acfEst):
        print("The ACF is deemed normal according to D'Agostino-Pearson")
    else:
        print("The ACF is *NOT* deemed normal according to D'Agostino-Pearson")
    pacfEst = pacf(model_residual, "auto", includeZeroLag=False)

    if is_AP_normal(pacfEst):
        print("The PACF is deemed normal according to D'Agostino-Pearson")
    else:
        print("The PACF is *NOT* deemed normal according to D'Agostino-Pearson")


def is_AP_normal(ts: TimeSeries, alpha=0.05):
    """Indicates normality according to the d'Agostino-Pearson test

    Takes a time series and returns whether it is normal-distributed according
    to the d'Agostino-Pearson test with specified alpha.

    Args:
        ts: The data to test for normality
        alpha: The tolerance for error, i.e. 1 - alpha is our confidence level
    """
    _, p = normaltest(ts)
    # We use a 95% confidence
    deemed_normal = p > 0.05
    return deemed_normal

def plot_pacf(): ...
