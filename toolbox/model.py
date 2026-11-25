from abc import ABC, abstractmethod
from enum import Enum
from typing import cast, override

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from numpy.typing import ArrayLike
from scipy import signal
from timeseries import TimeSeries, plot_timeseries
from tsa_lth.analysis import box_cox, pzmap
from tsa_lth.modelling import estimateARMA, polydiv


class ModelStep(ABC):
    @abstractmethod
    def add_structure(self, ts: TimeSeries) -> TimeSeries: ...

    @abstractmethod
    def remove_structure(self, ts: TimeSeries) -> TimeSeries: ...

    def __or__(self, step: ModelStep) -> CompoundStep:
        return CompoundStep([self, step])


class CompoundStep(ModelStep):
    def __init__(self, steps: list[ModelStep]):
        self._steps = steps

    @override
    def remove_structure(self, ts: TimeSeries) -> TimeSeries:
        for step in self._steps:
            ts = step.remove_structure(ts)

        return ts

    @override
    def add_structure(self, ts: TimeSeries) -> TimeSeries:
        for step in self._steps:
            ts = step.add_structure(ts)

        return ts


# Class meant primarily for nominal typing / documentation
class Transform(ModelStep):
    @abstractmethod
    def transform_data(self, ts: TimeSeries) -> TimeSeries:
        ...

    @abstractmethod
    def undo_transform(self, ts:TimeSeries) -> TimeSeries:
        ...

    @override
    def remove_structure(self, ts: TimeSeries) -> TimeSeries:
        return self.transform_data(ts)

    @override
    def add_structure(self, ts: TimeSeries) -> TimeSeries:
        return self.undo_transform(ts)


class Log(Transform):
    @override
    def transform_data(self, ts: TimeSeries) -> TimeSeries:
        return np.log(ts)

    @override
    def undo_transform(self, ts: TimeSeries) -> TimeSeries:
        return np.exp(ts)


class Sqrt(Transform):
    @override
    def transform_data(self, ts: TimeSeries) -> TimeSeries:
        return np.sqrt(ts)

    @override
    def undo_transform(self, ts: TimeSeries) -> TimeSeries:
        return ts**2


class Transforms(Enum):
    LOG = Log
    SQRT = Sqrt


class Trend(ModelStep):
    @abstractmethod
    def detrend(self, ts: TimeSeries) -> TimeSeries:
        ...

    @abstractmethod
    def add_trend(self, ts: TimeSeries) -> TimeSeries:
        ...

    @override
    def remove_structure(self, ts: TimeSeries) -> TimeSeries:
        return self.detrend(ts)

    @override
    def add_structure(self, ts: TimeSeries) -> TimeSeries:
        return self.add_trend(ts)


class LeastSquares(Trend):
    def __init__(self):
        self._trend: TimeSeries | None = None

    @override
    def detrend(self, ts: TimeSeries) -> TimeSeries:
        N = len(ts)
        X = np.column_stack([np.ones(N), np.arange(1, N+1)])
        ls_linear_trend = np.linalg.inv(X.T @ X) @ X.T @ ts
        self._trend = ls_linear_trend
        return ts - ls_linear_trend

    @override
    def add_trend(self, ts:TimeSeries) -> TimeSeries:
        return ts + self._trend


class Season(ModelStep):
    @abstractmethod
    def remove_season(self, ts: TimeSeries) -> TimeSeries:
        ...

    @abstractmethod
    def add_season_back(self, ts: TimeSeries) -> TimeSeries:
        ...

    @override
    def remove_structure(self, ts: TimeSeries) -> TimeSeries:
        return self.remove_season(ts)

    @override
    def add_structure(self, ts: TimeSeries) -> TimeSeries:
        return self.add_season_back(ts)

class Nabla(Season):
    def __init__(self, s: int):
        assert s >= 1
        self.s = s

    @override
    def remove_season(self, ts: TimeSeries) -> TimeSeries:
        A = np.array([1] + [0] * (self.s - 1) + [-1])
        ts_s = np.array(scipy.signal.lfilter(A, [1], ts))[self.s:]

        return ts_s

    @override
    def add_season_back(self, ts:TimeSeries) -> TimeSeries:
        ...

class SumOfSine(ModelStep):
    def __init__(self):
        raise NotImplementedError()


class ARMAX(ModelStep):
    def __init__(self):
        raise NotImplementedError()


class BoxJenkins(ModelStep):
    def __init__(self):
        raise NotImplementedError()


type StationaryModel = ARMA | SumOfSine | ARMAX | BoxJenkins

# class SimpleModel(ModelStep):
#     def __init__(self):
#         raise NotImplementedError()


class Model:
    """DOCSTRING TBD - But only undo a transformation if it was the last
    transformation done; some of the transformations are stateful.
    """
    UNKNOWN = sentinel("UNKNOWN")

    type Unknown = UNKNOWN.__class__

    # def __init__(self, model: TimeSeries, validation: TimeSeries, test: TimeSeries | None):
    def __init__(self):
        self.outliers_handled = False
        self.transform: Transform | Model.Unknown | None = Model.UNKNOWN
        self.trend: Trend | Model.Unknown | None = Model.UNKNOWN
        self.season: Season | Model.Unknown | None = Model.UNKNOWN
        self.stationary_model: StationaryModel | Model.Unknown | None = Model.UNKNOWN

    def study(self, ts: TimeSeries) -> TimeSeries | None:
        print(
            "Remember to plot against simulated data at the end, to make sure"
            "everything is working as expected!"
        )
        if not self.outliers_handled:
            print(
                "Make sure outliers are handled: Look for outliers in the "
                "plotted dataset and remove them in an appropriate manner. "
                "Then set outliers_handled to True."
            )
            plot_timeseries(ts)

        if self.transform is not None:
            res = self.inv_transform_or_suggest(ts)
            if res == Model.UNKNOWN:
                return None
            else:
                ts = res

        if self.trend is not None:
            res = self.detrend_or_suggest(ts)

    def predict(self, steps: int, ts: TimeSeries) -> TimeSeries: ...

    def _any_is_unknown(self): ...

    def test(self, ts: TimeSeries): ...

    def inv_transform_or_suggest(self, ts: TimeSeries) -> TimeSeries | Model.Unknown:
        assert self.transform is not None
        if self.transform == Model.Unknown:
            # TODO: Plot Box-Cox plot and also print maximum value found, but
            # remind user to not pick the maximum value if it is extreme (Assume
            # it should be somewhere in [-2, 2] and almost certainly in [-0.5, 0.5]
            box_cox(ts)
            print(
                "In general we like log, but inspect the box-cox plot and see"
                " what makes the data the most normal. In general one should"
                " consider values only in [-2, 2], and preferrably in "
                "[-0.5, 0.5]"
            )
            return Model.UNKNOWN
        else:
            assert isinstance(self.transform, Transform)
            return self.transform.add_structure(ts)

    def detrend_or_suggest(self, ts: TimeSeries)-> TimeSeries | Model.Unknown:
        assert self.trend is not None
        if self.trend == Model.UNKNOWN:
            # TODO: Plot model with linear trend found by LSE and list other
            # possibilities (PEM, ...,)
            N = len(ts)
            X = np.column_stack([np.ones(N), np.arange(1, N+1)])
            ls_linear_trend = np.linalg.inv(X.T @ X) @ X.T @ ts

            plt.figure()
            plt.title("Detrend study (LSE plotted)")
            plt.grid(True, alpha=0.3)
            plt.plot(ts)
            plt.plot(ls_linear_trend)
            return Model.UNKNOWN

        else:
            assert isinstance(self.trend, Trend)


def my_estimate_arma(ts: TimeSeries, A: int, C: int) -> ARMA:
    model = estimateARMA(ts, A, C)
    return ARMA(model.A, model.C)  # type: ignore


class ARMA(ModelStep):
    def __init__(self, A: ArrayLike, C: ArrayLike, seed=None):
        self._A = np.array(A)
        self.C = np.array(C)
        self.generator = np.random.default_rng(seed)

    @property
    def A(self):
        return np.copy(self._A)

    @A.setter
    def A(self, poly: ArrayLike | int):
        roots = np.roots(poly)
        if any(abs(root) >= 1 for root in roots):
            raise ValueError("AR polynomial cannot be set to unstable polynomial")

        raise NotImplementedError(
            "Have not implemented proper checks for AR polynomial stability"
        )
        self._A = np.array(poly)

    @property
    def ord_A(self) -> int:
        return len(self._A)

    @property
    def ord_C(self) -> int:
        return len(self.C)

    def set_seed(self, seed: int):
        self.generator = np.random.default_rng(seed)

    def simulate(
        self, sigma2: float, N: int, buffer: int = 100, seed: int | None = None
    ) -> TimeSeries:
        """Creates a realisation of the ARMA process by driving white noise.

        Generates gaussian white noise, using its internal generator
        (recreating it with a new seed in case one is supplied), and feeds it
        through the ARMA filter.

        Args:
            sigma2: The variance of the driving white noise.
            N: The number of samples to generate.
            buffer: The number of initial samples to discard in order to avoid
                initialisation effects.
            seed: An optional seed for the internal random number generator to
                use.

        Returns:
            A simulated time series of length (N - buffer)

        """
        if seed is not None:
            self.generator = np.random.default_rng(seed)
        e = self.generator.normal(scale=np.sqrt(sigma2), size=N)
        y = cast(TimeSeries, signal.lfilter(self.C, self._A, e))
        return y[buffer:]

    def plot_poles_and_zeros(self):
        pzmap(self.C, self._A)

    def residual(self, ts: TimeSeries) -> TimeSeries:
        """Calculates the error residual of the model.

        Takes in a timerseries and filter it in revers using the model, then
        returns it. In case of a good model the residual should resemble a
        white noise.

        Args:
            ts: The time series to predict

        Returns:
            The residual error as a time series
        """
        e_hat = cast(TimeSeries, signal.lfilter(self._A, self.C, ts))
        order_a = len(self._A) - 1
        assert order_a >= 0
        return e_hat[order_a:]

    def predict(self, steps: int, ts: TimeSeries):
        _F, G = polydiv(self.C, self._A, steps)
        prediction = cast(TimeSeries, scipy.signal.lfilter(G, self.C, ts))
        error_residual = cast(TimeSeries, ts[steps:] - prediction[steps:])  # type: ignore
        return (prediction, error_residual)
        # Return (prediction, error_residual)
