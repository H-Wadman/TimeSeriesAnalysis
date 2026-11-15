from abc import ABC, abstractmethod
from typing import override

import matplotlib.pyplot as plt

from timeseries import TimeSeries, plot_timeseries
from tsa_lth.analysis import box_cox


class ModelStep(ABC):
    @abstractmethod
    def inverse(self, ts: TimeSeries) -> TimeSeries: ...

    @abstractmethod
    def predict(self, ts:TimeSeries) -> TimeSeries: ...

    def __or__(self, step: ModelStep) -> CompoundStep:
        return CompoundStep([self, step])

class CompoundStep(ModelStep):
    def __init__(self, steps: list[ModelStep]):
        self._steps = steps

    @override
    def predict(self, ts: TimeSeries) -> TimeSeries:
        for step in self._steps:
            ts = step.predict(ts)

        return ts

    @override
    def inverse(self, ts: TimeSeries) -> TimeSeries:
        for step in self._steps:
            ts = step.inverse(ts)

        return ts

    


class Transform(ModelStep):
    def __init__(self):
        raise NotImplementedError()

class Trend(ModelStep):
    def __init__(self):
        raise NotImplementedError()

class Season(ModelStep):
    def __init__(self):
        raise NotImplementedError()

class ARMA(ModelStep):
    def __init__(self):
        raise NotImplementedError()

class SumOfSine(ModelStep):
    def __init__(self):
        raise NotImplementedError()

type SimpleModel = ARMA | SumOfSine;

# class SimpleModel(ModelStep):
#     def __init__(self):
#         raise NotImplementedError()


class Model:
    UNKNOWN = sentinel("UNKNOWN")

    type Unknown = UNKNOWN.__class__

    # def __init__(self, model: TimeSeries, validation: TimeSeries, test: TimeSeries | None):
    def __init__(self):
        self.outliers_handled = False
        self.transform: Transform | Model.Unknown | None = Model.UNKNOWN
        self.trend: Trend | Model.Unknown | None = Model.UNKNOWN
        self.season: Season | Model.Unknown | None = Model.UNKNOWN
        self.simple_model: SimpleModel | Model.Unknown | None  = Model.UNKNOWN


    def study(self, ts: TimeSeries) -> TimeSeries | None:
        print("Remember to plot against simulated data at the end, to make sure"
              "everything is working as expected!")
        if not self.outliers_handled:
            print("Make sure outliers are handled: Look for outliers in the "
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



    def predict(self, ts: TimeSeries) -> TimeSeries:
        ...

    def _any_is_unknown(self):
        ...

    def test(self, ts: TimeSeries): ...

    def inv_transform_or_suggest(self, ts: TimeSeries):
        assert self.transform is not None
        if self.transform == Model.Unknown:
            #TODO: Plot Box-Cox plot and also print maximum value found, but
            # remind user to not pick the maximum value if it is extreme (Assume
            # it should be somewhere in [-2, 2] and almost certainly in [-0.5, 0.5]
            box_cox(ts)
            print("In general we like log, but inspect the box-cox plot and")
            return Model.UNKNOWN
        else:
            assert isinstance(self.transform, Transform)
            return self.transform.inverse(ts)

    def detrend_or_suggest(self, ts: TimeSeries):
        assert self.trend is not None
        if self.trend == Model.UNKNOWN:
            #TODO: Plot model with linear trend found by LSE and list other
            # possibilities (PEM, ...,)
            plt.figure()
            plt.title("Detrend study (LSE plotted)")
            plt.grid(True, alpha=0.3)
            plt.plot()

            
            raise NotImplementedError()
        else:
            assert isinstance(self.trend, Trend)

