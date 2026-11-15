import numpy as np
from numpy.typing import ArrayLike
from scipy import signal

from toolbox.timeseries import TimeSeries


class ARMA:
    def __init__(self, A: ArrayLike, C: ArrayLike, seed=None):
        self.A = np.array(A)
        self.C = np.array(C)
        self.generator = np.random.default_rng(seed)

    @property
    def A(self):
        return np.copy(self.A)

    @A.setter
    def A(self, poly: ArrayLike):
        roots = np.roots(poly)
        if any(abs(root) >= 1 for root in roots):
            raise ValueError("AR polynomial cannot be set to unstable polynomial")

        raise NotImplementedError(
            "Have not implemented proper checks for AR polynomial stability"
        )
        self.A = np.array(poly)

    def simulate(self, sigma2, N, buffer=100, seed=None) -> TimeSeries:
        if seed is not None:
            self.generator = np.random.default_rng(seed)
        e = self.generator.normal(scale=np.sqrt(sigma2), size=N)
        y = signal.lfilter(self.C, self.A, e)
        return y[buffer:]
