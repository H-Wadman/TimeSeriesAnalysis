import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from toolbox.model import ARMA, LeastSquares, Log, Model, Nabla


def configured_model(training, *, season=None, arma=None):
    model = Model()
    model.outliers_handled = True
    model.transform = None
    model.trend = None
    model.season = season
    model.stationary_model = arma
    model.study(training)
    return model


class ModelWorkflowTests(unittest.TestCase):
    def test_unknown_stage_stops_study_until_selected(self):
        model = Model()
        model.outliers_handled = True
        model.transform = None
        model.trend = None
        model.season = None
        model.stationary_model = Model.UNKNOWN
        self.assertIsNone(model.study(np.arange(8.0)))
        with self.assertRaises(RuntimeError):
            model.predict(1, np.array([8.0]))

    def test_ar1_forecast_and_validation_alignment(self):
        training = np.array([1.0, 0.5, 0.25, 0.125])
        model = configured_model(training, arma=ARMA([1.0, -0.5], [1.0]))
        result = model.predict(1, np.array([0.0625, 0.03125]))
        assert_array_equal(result.indices, [0, 1])
        assert_allclose(result.predictions, [0.0625, 0.03125])
        assert_allclose(result.errors, [0.0, 0.0])

    def test_forecasts_within_training_stop_at_the_origin(self):
        training = np.array([1.0, 2.0, 4.0, 8.0])
        model = configured_model(training, arma=ARMA([1.0, -0.5], [1.0]))

        result = model.predict(2, np.array([10.0, 20.0, 30.0]))

        assert_array_equal(result.indices, [0, 1, 2])
        assert_allclose(result.predictions, [1.0, 2.0, 2.5])
        assert_allclose(result.errors, [9.0, 18.0, 27.5])

    def test_transform_and_fitted_trend_are_inverted_at_future_indices(self):
        training = np.exp(2.0 + 0.1 * np.arange(10))
        validation = np.exp(2.0 + 0.1 * np.arange(10, 13))
        model = Model()
        model.outliers_handled = True
        model.transform = Log()
        model.trend = LeastSquares()
        model.season = None
        model.stationary_model = ARMA([1.0], [1.0])
        innovations = model.study(training)
        assert_allclose(innovations, 0.0, atol=1e-12)
        result = model.predict(1, validation)
        assert_array_equal(result.indices, [0, 1, 2])
        assert_allclose(result.predictions, validation, rtol=1e-12)

    def test_seasonal_forecasts_reconstruct_intermediate_horizons(self):
        training = np.tile([1.0, 2.0], 6)
        validation = np.tile([1.0, 2.0], 3)
        model = configured_model(
            training, season=Nabla(2), arma=ARMA([1.0], [1.0])
        )
        for horizon in (1, 2, 3):
            with self.subTest(horizon=horizon):
                result = model.predict(horizon, validation)
                assert_array_equal(result.indices, np.arange(len(validation)))
                assert_allclose(result.predictions, validation, atol=1e-12)
                assert_allclose(result.errors, 0.0, atol=1e-12)

    def test_rolling_predictions_do_not_use_later_validation_values(self):
        training = np.tile([1.0, 2.0], 6)
        validation = np.tile([1.0, 2.0], 4)
        model = configured_model(
            training, season=Nabla(2), arma=ARMA([1.0], [1.0])
        )
        original = model.predict(2, validation)
        changed = validation.copy()
        changed[6:] = [100.0, -50.0]
        alternative = model.predict(2, changed)
        prefix = original.indices <= 5
        assert_array_equal(original.indices[prefix], alternative.indices[prefix])
        assert_allclose(original.predictions[prefix], alternative.predictions[prefix])

    def test_insufficient_horizon_history_is_explicit(self):
        training = np.tile([1.0, 2.0], 4)
        model = configured_model(training, arma=ARMA([1.0], [1.0]))
        with self.assertRaises(ValueError):
            model.predict(20, np.array([1.0, 2.0]))

    def test_changing_fitted_model_configuration_requires_a_new_study(self):
        model = configured_model(
            np.array([1.0, 0.5, 0.25, 0.125]),
            arma=ARMA([1.0, -0.5], [1.0]),
        )
        model.stationary_model.C = np.array([1.0, 0.1])
        with self.assertRaises(RuntimeError):
            model.predict(1, np.array([0.0625]))

    def test_ar_polynomial_setter_accepts_stable_and_rejects_unstable(self):
        model = ARMA([1.0], [1.0])
        model.A = [1.0, -0.5]
        assert_allclose(model.A, [1.0, -0.5])
        with self.assertRaises(ValueError):
            model.A = [1.0, -1.2]


if __name__ == "__main__":
    unittest.main()
