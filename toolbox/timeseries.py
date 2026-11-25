import sys
from collections.abc import Callable, Sized
from enum import Enum
from typing import NamedTuple, cast

import matplotlib.pyplot as plt
import numpy as np
import tsa_lth
from matplotlib.axes import Axes
from scipy import signal
from tsa_lth.analysis import acf, pacf
from tsa_lth.modelling import estimateARMA
from tsa_lth.tests import normaltest

type TimeSeries = np.ndarray


def plot_timeseries(title, x_label="x", y_label="y"):
    plt.figure()
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.plot()


def plot_confidence_interval(n: int, max_lag: int, ax: Axes | None = None):
    """Plot approximate 95% correlation bounds over lags -max_lag to max_lag.

    Args:
        n: Number of observations used to calculate the correlation.
        max_lag: Largest absolute lag shown.
        ax: Axes to draw on; defaults to the current pyplot axes.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")

    plot = plt if ax is None else ax
    bound = 2 / np.sqrt(n)
    plot.plot([-max_lag, max_lag], [bound, bound], "r--")
    plot.plot([-max_lag, max_lag], [-bound, -bound], "r--")


def plot_acf(ts: TimeSeries, max_lag=None):
    ts_len = len(cast(Sized, ts))
    max_recommended_lags = ts_len // 4

    if max_lag is None:
        max_lag = max_recommended_lags
    elif max_lag > max_recommended_lags:
        print(
            "Warning: ACF will be calculated for lags larger than N/4; this is not recommended, see p. 47",
            file=sys.stderr,
        )


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
    model = estimateARMA(y, A, C, plot=False)  # type: ignore
    assert model is not None
    e_hat = signal.lfilter(model.A, model.C, y)
    na = len(model.A) - 1
    assert na >= 0
    return e_hat[na:]


def sign_change_test(
    data: TimeSeries, k: int, alpha: float
) -> tuple[bool, float, float]:
    # Credit for code snippet below to Filipp Lernbo, taken from tsa_lth.tests
    n_ratio, confInt = tsa_lth.tests.count_sign_changes(data, 1 - alpha)
    is_white = confInt[0] < n_ratio < confInt[1]
    print(
        f"  Sign change test:      {is_white} (white if {n_ratio:.2f} in [{confInt[0]:.2f},{confInt[1]:.2f}])"
    )

    return (is_white, n_ratio, confInt)


class WhitenessTest(Enum):
    type TestFunction = Callable[
        [TimeSeries], tuple[bool, float, float | tuple[float, float]]
    ]

    class WhitenessInfo(NamedTuple):
        name: str
        test: WhitenessTest.TestFunction

    LJUNG_BOX = WhitenessInfo(
        "Ljung-Box-Pierce", cast(TestFunction, tsa_lth.tests.lbp_test)
    )
    MCLEOD_LI = WhitenessInfo("McLeod-Li", cast(TestFunction, tsa_lth.tests.ml_test))
    MONTI = WhitenessInfo("Monti", cast(TestFunction, tsa_lth.tests.monti_test))
    SIGN_CHANGE = WhitenessInfo("Sign change", cast(TestFunction, sign_change_test))


class NormalityTest(Enum):
    DAGOSTINO_PEARSON = "D'agostino-Pearson"


def verify_white_and_normal(
    model_residual: TimeSeries,
    whiteness_test=WhitenessTest.MONTI,
    normal_test=NormalityTest.DAGOSTINO_PEARSON,
):
    white_name, white_test = whiteness_test.value
    is_white, *_ = white_test(model_residual)

    if is_white:
        print(f"The model is white according to the {white_name} test")
    else:
        print(f"The model is *NOT* white according to the {white_name} test")

    acfEst = np.array(acf(model_residual, "auto", includeZeroLag=False))
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
