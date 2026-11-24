"""Regression-test the accuracy metrics benchmark contract."""

import unittest

import numpy as np

from utils.metrics import compute_metrics
from utils import tsfm_metrics


class AccuracyMetricTests(unittest.TestCase):
    """Define the AccuracyMetricTests type and its associated runtime state or contract.
    
    Bases:
        unittest.TestCase
    """
    def test_observed_mask_excludes_imputed_targets(self):
        """Regression-check that observed mask excludes imputed targets.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        actual = np.asarray([[1.0, 2.0, 1000.0]])
        predicted = np.asarray([[1.0, 4.0, -1000.0]])
        observed = np.asarray([[True, True, False]])

        metrics = compute_metrics(actual, predicted, observed_mask=observed)

        self.assertEqual(metrics["Observed points"], 2)
        self.assertAlmostEqual(metrics["MAE"], 1.0)

    def test_scaled_errors_use_context_scales(self):
        """Regression-check that scaled errors use context scales.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        actual = np.asarray([[2.0, 4.0], [6.0, 8.0]])
        predicted = actual + 2.0
        mae_scale = np.asarray([2.0, 2.0])
        mse_scale = np.asarray([4.0, 4.0])

        metrics = compute_metrics(
            actual,
            predicted,
            mase_scale=mae_scale,
            rmsse_scale=mse_scale,
        )

        self.assertAlmostEqual(metrics["MASE"], 1.0)
        self.assertAlmostEqual(metrics["RMSSE"], 1.0)

    def test_shared_metrics_match_tsfm_reference(self):
        """Regression-check the recovered TFB/TSFM-Bench metric contract.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        actual = np.asarray([[1.0, 2.0, 4.0], [2.0, 5.0, 10.0]])
        predicted = np.asarray([[2.0, 1.0, 5.0], [1.0, 8.0, 8.0]])

        metrics = compute_metrics(actual, predicted)

        self.assertAlmostEqual(metrics["MAE"], tsfm_metrics.mae(actual, predicted))
        self.assertAlmostEqual(metrics["MSE"], tsfm_metrics.mse(actual, predicted))
        self.assertAlmostEqual(metrics["RMSE"], tsfm_metrics.rmse(actual, predicted))
        self.assertAlmostEqual(metrics["NRMSE"], tsfm_metrics.nrmse(actual, predicted))
        self.assertAlmostEqual(metrics["MAPE (%)"], tsfm_metrics.mape(actual, predicted))
        self.assertAlmostEqual(metrics["sMAPE (%)"], tsfm_metrics.smape(actual, predicted))
        self.assertAlmostEqual(metrics["WAPE"], tsfm_metrics.wape(actual, predicted))
        self.assertAlmostEqual(metrics["MSMAPE (%)"], tsfm_metrics.msmape(actual, predicted))

    def test_tsfm_zero_denominator_contract_is_preserved(self):
        """Regression-check legacy zero-denominator behavior.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        actual = np.zeros((1, 3))
        predicted = np.ones((1, 3))

        metrics = compute_metrics(actual, predicted)

        self.assertTrue(np.isinf(metrics["MAPE (%)"]))
        self.assertAlmostEqual(metrics["NRMSE"], 1.0e8)
        self.assertTrue(np.isinf(metrics["WAPE"]))
        self.assertTrue(np.isnan(metrics["MASE"]))
        self.assertTrue(np.isnan(metrics["RMSSE"]))

    def test_scaled_errors_average_window_level_ratios(self):
        """Ensure unequal seasonal scales do not collapse into a ratio of averages.

        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        actual = np.asarray([[1.0], [1.0]])
        predicted = np.asarray([[2.0], [5.0]])
        mae_scale = np.asarray([1.0, 4.0])
        mse_scale = np.asarray([1.0, 16.0])

        metrics = compute_metrics(
            actual,
            predicted,
            mase_scale=mae_scale,
            rmsse_scale=mse_scale,
        )

        self.assertAlmostEqual(metrics["MASE"], 1.0)
        self.assertAlmostEqual(metrics["RMSSE"], 1.0)


if __name__ == "__main__":
    unittest.main()
