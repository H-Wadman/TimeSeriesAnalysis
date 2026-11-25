from abc import ABC, abstractmethod
from enum import Enum
from typing import NamedTuple, cast, override

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from numpy.typing import ArrayLike
from scipy import signal
from tsa_lth.analysis import box_cox, pzmap
from tsa_lth.modelling import estimateARMA, polydiv

if __package__:
    from .timeseries import TimeSeries, plot_timeseries
else:
    from timeseries import TimeSeries, plot_timeseries


class _Unknown:
    pass


def _series(ts: ArrayLike, *, min_length: int = 1) -> TimeSeries:
    data = np.asarray(ts, dtype=float)
    if data.ndim != 1 or len(data) < min_length:
        raise ValueError(f"Expected a one-dimensional series of at least {min_length} samples")
    if not np.all(np.isfinite(data)):
        raise ValueError("Time series must contain only finite values")
    return data


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


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
        for step in reversed(self._steps):
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
        if np.any(np.asarray(ts) <= 0):
            raise ValueError("Log transform requires positive values")
        return np.log(ts)

    @override
    def undo_transform(self, ts: TimeSeries) -> TimeSeries:
        return np.exp(ts)


class Sqrt(Transform):
    @override
    def transform_data(self, ts: TimeSeries) -> TimeSeries:
        if np.any(np.asarray(ts) < 0):
            raise ValueError("Square-root transform requires non-negative values")
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
        self._coefficients: TimeSeries | None = None

    @override
    def detrend(self, ts: TimeSeries) -> TimeSeries:
        ts = _series(ts, min_length=2)
        x = np.arange(len(ts), dtype=float)
        self._coefficients = np.linalg.lstsq(
            np.column_stack((np.ones(len(ts)), x)), ts, rcond=None
        )[0]
        return ts - self.values(len(ts))

    def values(self, length: int, start: int = 0) -> TimeSeries:
        if self._coefficients is None:
            raise RuntimeError("Trend has not been fitted")
        return self._coefficients[0] + self._coefficients[1] * np.arange(start, start + length)

    @override
    def add_trend(self, ts:TimeSeries) -> TimeSeries:
        ts = _series(ts)
        return ts + self.values(len(ts))


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
        _positive_integer(s, "s")
        self.s = s
        self._anchors: TimeSeries | None = None

    @override
    def remove_season(self, ts: TimeSeries) -> TimeSeries:
        _positive_integer(self.s, "s")
        ts = _series(ts, min_length=self.s + 1)
        self._anchors = ts[:self.s].copy()
        return ts[self.s:] - ts[:-self.s]

    @override
    def add_season_back(self, ts:TimeSeries) -> TimeSeries:
        ts = _series(ts)
        if self._anchors is None or len(self._anchors) != self.s:
            raise RuntimeError("Seasonal differencing must be fitted before inversion")
        return _restore_seasonal_levels(ts, self._anchors)


def _restore_seasonal_levels(differences: TimeSeries, anchors: TimeSeries) -> TimeSeries:
    """Reconstruct levels from the preceding season and successive differences."""
    lag = len(anchors)
    levels = np.concatenate((anchors, np.empty(len(differences))))
    for i, difference in enumerate(differences):
        levels[i + lag] = difference + levels[i]
    return levels


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
    """Guided time-series modelling followed by sequential rolling forecasts.

    Configure each stage with a fitted step or ``None`` to skip it. Leave a
    stage as ``Model.UNKNOWN`` to receive its suggestion and stop the study.
    For predictions, ``validation`` must immediately follow the training
    series. Whiteness testing uses one-step innovations only. Predictions are
    plug-in inverse transforms and are not bias-adjusted.
    """
    UNKNOWN = _Unknown()
    Unknown = _Unknown

    # def __init__(self, model: TimeSeries, validation: TimeSeries, test: TimeSeries | None):
    def __init__(self):
        self.outliers_handled = False
        self.transform: Transform | Model.Unknown | None = Model.UNKNOWN
        self.trend: Trend | Model.Unknown | None = Model.UNKNOWN
        self.season: Season | Model.Unknown | None = Model.UNKNOWN
        self.stationary_model: StationaryModel | Model.Unknown | None = Model.UNKNOWN
        self._training_data: TimeSeries | None = None
        self._stationary_history: TimeSeries | None = None
        self._innovations: TimeSeries | None = None
        self._fitted_stages: tuple[object, ...] | None = None

    def study(self, ts: TimeSeries) -> TimeSeries | None:
        ts = _series(ts, min_length=2)
        source = ts.copy()
        self._training_data = None
        self._stationary_history = None
        self._innovations = None
        self._fitted_stages = None

        print(
            "After fitting, compare the model with simulated data to check "
            "that the model is appropriate."
        )
        if not self.outliers_handled:
            print(
                "Make sure outliers are handled: Look for outliers in the "
                "plotted dataset and remove them in an appropriate manner. "
                "Then set outliers_handled to True."
            )
            plot_timeseries("Training data")
            plt.plot(ts)
            return None

        if self.transform is not None:
            res = self.inv_transform_or_suggest(ts)
            if res is Model.UNKNOWN:
                return None
            ts = res

        if self.trend is not None:
            res = self.detrend_or_suggest(ts)
            if res is Model.UNKNOWN:
                return None
            ts = res

        if self.season is Model.UNKNOWN:
            print(
                "Choose a seasonal differencing period (for example, "
                "Nabla(period)) or set season to None to skip this stage."
            )
            return None
        if self.season is not None:
            ts = self.season.remove_structure(ts)

        if self.stationary_model is Model.UNKNOWN:
            print(
                "Choose a stationary model (for example, ARMA(A, C)) or set "
                "stationary_model to None only if no stationary model is needed."
            )
            return None
        if self.stationary_model is None:
            innovations = ts.copy()
        elif not isinstance(self.stationary_model, ARMA):
            raise NotImplementedError(
                "The guided Model workflow currently supports ARMA stationary models"
            )
        else:
            innovations = self.stationary_model.residual(ts)
        self._training_data = source
        self._stationary_history = ts.copy()
        self._innovations = innovations.copy()
        self._fitted_stages = self._stage_signature()
        return innovations

    def _stage_signature(self) -> tuple[object, ...]:
        stages = (self.transform, self.trend, self.season, self.stationary_model)
        signature: list[object] = []
        for stage in stages:
            if stage is None:
                signature.append(None)
            elif isinstance(stage, LeastSquares):
                coefficients = stage._coefficients
                signature.append(
                    (id(stage), None if coefficients is None else tuple(coefficients))
                )
            elif isinstance(stage, Nabla):
                anchors = stage._anchors
                signature.append(
                    (id(stage), stage.s, None if anchors is None else anchors.tobytes())
                )
            elif isinstance(stage, ARMA):
                signature.append((id(stage), tuple(stage.A), tuple(stage.C)))
            else:
                signature.append(id(stage))
        return tuple(signature)

    def _any_is_unknown(self) -> bool:
        return any(
            stage is Model.UNKNOWN
            for stage in (self.transform, self.trend, self.season, self.stationary_model)
        )

    def test(self, ts: TimeSeries | None = None):
        """Run a Ljung-Box-Pierce whiteness test on one-step innovations."""
        from tsa_lth.tests import lbp_test

        residuals = self._innovations if ts is None else _series(ts, min_length=3)
        if residuals is None:
            raise RuntimeError("Study a complete model before testing its innovations")
        if len(residuals) < 3:
            raise ValueError("At least three one-step innovations are required")
        return lbp_test(residuals, K=min(25, len(residuals) - 1))

    def predict(self, steps: int, ts: TimeSeries) -> "PredictionResult":
        """Return rolling forecasts and errors at validation-local target indices.

        Each target forecast uses observations only through ``target - steps``,
        even when that origin is inside the training series. Targets lacking
        enough history for the selected seasonal and
        ARMA filters are omitted; the conditional ARMA filters are initialized
        at the beginning of the available training history. An entirely
        unsupported validation set raises ``ValueError``. The original-scale
        error is actual minus predicted.
        """
        _positive_integer(steps, "steps")
        steps = int(steps)
        validation = _series(ts)
        training, arma = self._prediction_setup()
        seasonal_lag = self.season.s if isinstance(self.season, Nabla) else 0
        warmup = max(arma.ord_A, arma.ord_C) - 1

        # The origin needs at least warmup + 1 observations after differencing.
        first_target = max(len(training), steps + seasonal_lag + warmup)
        observations = np.concatenate((training, validation))
        if first_target >= len(observations):
            raise ValueError(
                "Validation has no targets with sufficient training history "
                "for the requested horizon and model"
            )

        forecasts = []
        for target in range(first_target, len(observations)):
            observed = observations[:target - steps + 1].copy()
            forecasts.append(self._forecast_target(observed, steps, arma, seasonal_lag))

        indices = np.arange(first_target - len(training), len(validation), dtype=int)
        predictions = np.asarray(forecasts)
        return PredictionResult(indices, predictions, validation[indices] - predictions)

    def _prediction_setup(self) -> tuple[TimeSeries, ARMA]:
        if self._training_data is None or self._stationary_history is None:
            raise RuntimeError("Study a complete model before predicting")
        if self._fitted_stages != self._stage_signature():
            raise RuntimeError("Model stages changed after study; study the model again")
        if not isinstance(self.stationary_model, ARMA):
            raise NotImplementedError("Rolling prediction currently requires an ARMA model")
        if self.trend is not None and not isinstance(self.trend, LeastSquares):
            raise NotImplementedError("Rolling prediction requires a LeastSquares trend")
        if self.season is not None and not isinstance(self.season, Nabla):
            raise NotImplementedError("Rolling prediction requires Nabla seasonal differencing")
        return self._training_data, self.stationary_model

    def _forecast_target(
        self, observed: TimeSeries, steps: int, arma: ARMA, seasonal_lag: int
    ) -> float:
        levels = self._to_detrended_scale(observed)
        stationary = (
            levels[seasonal_lag:] - levels[:-seasonal_lag] if seasonal_lag else levels
        )
        forecast = arma.forecast(stationary, steps)
        if seasonal_lag:
            forecast = _restore_seasonal_levels(forecast, levels[-seasonal_lag:])

        target = len(observed) + steps - 1
        return self._to_original_scale(forecast[-1:].copy(), target)

    def _to_original_scale(self, prediction: TimeSeries, target: int) -> float:
        if isinstance(self.trend, LeastSquares):
            prediction = prediction + self.trend.values(1, start=target)
        if isinstance(self.transform, Transform):
            prediction = self.transform.add_structure(prediction)
        return float(prediction[0])

    def _to_detrended_scale(self, ts: TimeSeries) -> TimeSeries:
        values = (
            self.transform.remove_structure(ts)
            if isinstance(self.transform, Transform)
            else np.asarray(ts, dtype=float)
        )
        if isinstance(self.trend, LeastSquares):
            values = values - self.trend.values(len(values))
        return values

    def inv_transform_or_suggest(self, ts: TimeSeries) -> TimeSeries | Model.Unknown:
        assert self.transform is not None
        if self.transform is Model.UNKNOWN:
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
            return self.transform.remove_structure(ts)

    def detrend_or_suggest(self, ts: TimeSeries)-> TimeSeries | Model.Unknown:
        assert self.trend is not None
        if self.trend is Model.UNKNOWN:
            # TODO: Plot model with linear trend found by LSE and list other
            # possibilities (PEM, ...,)
            trend = LeastSquares()
            trend.detrend(ts)
            ls_linear_trend = trend.values(len(ts))

            plt.figure()
            plt.title("Detrend study (LSE plotted)")
            plt.grid(True, alpha=0.3)
            plt.plot(ts)
            plt.plot(ls_linear_trend)
            return Model.UNKNOWN

        else:
            assert isinstance(self.trend, Trend)
            return self.trend.remove_structure(ts)


class PredictionResult(NamedTuple):
    """Aligned forecasts; indices are zero-based offsets into validation."""

    indices: np.ndarray
    predictions: TimeSeries
    errors: TimeSeries


def my_estimate_arma(ts: TimeSeries, A: int, C: int) -> ARMA:
    model = estimateARMA(ts, A, C)
    return ARMA(model.A, model.C)  # type: ignore


class ARMA(ModelStep):
    def __init__(self, A: ArrayLike, C: ArrayLike, seed=None):
        self._A = _series(A)
        self.C = _series(C)
        if self._A[0] == 0 or self.C[0] == 0:
            raise ValueError("AR and MA polynomials must have non-zero leading coefficients")
        self._A = self._A / self._A[0]
        self.C = self.C / self.C[0]
        self.generator = np.random.default_rng(seed)

    @override
    def remove_structure(self, ts: TimeSeries) -> TimeSeries:
        return self.residual(ts)

    @override
    def add_structure(self, ts: TimeSeries) -> TimeSeries:
        ts = _series(ts)
        return cast(TimeSeries, signal.lfilter(self.C, self._A, ts))

    @property
    def A(self):
        return np.copy(self._A)

    @A.setter
    def A(self, poly: ArrayLike | int):
        coefficients = _series(poly)
        if coefficients[0] == 0:
            raise ValueError("AR polynomial must have a non-zero leading coefficient")
        coefficients = coefficients / coefficients[0]
        roots = np.roots(coefficients)
        if any(abs(root) >= 1 for root in roots):
            raise ValueError("AR polynomial cannot be set to unstable polynomial")
        self._A = coefficients

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
        ts = _series(ts)
        e_hat = cast(TimeSeries, signal.lfilter(self._A, self.C, ts))
        startup = max(len(self._A), len(self.C)) - 1
        return e_hat[startup:]

    def forecast(self, history: TimeSeries, steps: int) -> TimeSeries:
        """Forecast a stationary ARMA series from its observed history."""
        _positive_integer(steps, "steps")
        history = _series(history, min_length=1)
        errors = cast(TimeSeries, signal.lfilter(self._A, self.C, history))
        ar_order = len(self._A) - 1
        ma_order = len(self.C) - 1
        observed = list(history)
        forecasts = np.empty(steps, dtype=float)
        for horizon in range(steps):
            value = 0.0
            for lag in range(1, ar_order + 1):
                if lag <= len(observed):
                    value -= self._A[lag] * observed[-lag]
            for lag in range(1, ma_order + 1):
                if lag <= len(errors):
                    value += self.C[lag] * errors[-lag]
            forecasts[horizon] = value
            observed.append(value)
        return forecasts

    def predict(self, steps: int, ts: TimeSeries):
        _positive_integer(steps, "steps")
        ts = _series(ts)
        _F, G = polydiv(self.C, self._A, steps)
        prediction = cast(TimeSeries, scipy.signal.lfilter(G, self.C, ts))
        error_residual = cast(TimeSeries, ts[steps:] - prediction[steps:])  # type: ignore
        return (prediction, error_residual)
        # Return (prediction, error_residual)
