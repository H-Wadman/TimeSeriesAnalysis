import unittest

import numpy as np
from numpy.testing import assert_allclose

from toolbox.model import LeastSquares, Log, Nabla


class PreprocessingTests(unittest.TestCase):
    def test_seasonal_round_trip(self):
        data = np.array([4.0, 7.0, 8.0, 11.0, 14.0, 16.0])
        for lag in (1, 2, 4):
            with self.subTest(lag=lag):
                step = Nabla(lag)
                assert_allclose(step.add_structure(step.remove_structure(data)), data)

    def test_seasonal_invalid_inputs_and_missing_anchors(self):
        for lag in (0, -1, 1.5, True):
            with self.subTest(lag=lag), self.assertRaises(ValueError):
                Nabla(lag)
        step = Nabla(2)
        with self.assertRaises(RuntimeError):
            step.add_structure(np.array([1.0]))
        with self.assertRaises(ValueError):
            step.remove_structure(np.array([1.0, 2.0]))

    def test_compound_inverse_runs_in_reverse_order(self):
        data = np.array([2.0, 4.0, 3.0, 8.0, 6.0])
        step = Log() | Nabla(2)
        assert_allclose(step.add_structure(step.remove_structure(data)), data)

    def test_linear_trend_round_trip_and_extrapolation(self):
        data = np.array([3.0, 5.0, 7.0, 9.0])
        trend = LeastSquares()
        residual = trend.remove_structure(data)
        assert_allclose(residual, np.zeros(4), atol=1e-12)
        assert_allclose(trend.add_structure(residual), data)
        assert_allclose(trend.values(2, start=4), [11.0, 13.0])

    def test_unfitted_trend_and_invalid_transform(self):
        with self.assertRaises(RuntimeError):
            LeastSquares().add_structure(np.array([1.0]))
        with self.assertRaises(ValueError):
            Log().remove_structure(np.array([0.0]))


if __name__ == "__main__":
    unittest.main()
