"""Regression-test the reproducibility benchmark contract."""

import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from benchmark.config import BenchmarkConfig, DeviceCategory
from benchmark.runner import seed_benchmark_rng


class ReproducibilityTests(unittest.TestCase):
    """Define the ReproducibilityTests type and its associated runtime state or contract.
    
    Bases:
        unittest.TestCase
    """
    def test_config_retains_explicit_root_seed(self):
        """Regression-check that config retains explicit root seed.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        config = BenchmarkConfig(
            device_category=DeviceCategory.CPU,
            models=["units"],
            context_lens=[512],
            horizon_lens=[96],
            batch_sizes=[512],
            benchmark_seed=24680,
        )

        self.assertEqual(config.benchmark_seed, 24680)

    def test_root_seed_reproduces_python_numpy_and_torch_streams(self):
        """Regression-check that root seed reproduces python numpy and torch streams.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        with patch.dict("os.environ", {"BENCHMARK_EXECUTION_DEVICE": "cpu"}):
            seed_benchmark_rng(12345)
            first = (
                random.random(),
                np.random.random(4),
                torch.rand(4),
            )

            seed_benchmark_rng(12345)
            second = (
                random.random(),
                np.random.random(4),
                torch.rand(4),
            )

        self.assertEqual(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)

    def test_different_root_seed_changes_rng_streams(self):
        """Regression-check that different root seed changes rng streams.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        with patch.dict("os.environ", {"BENCHMARK_EXECUTION_DEVICE": "cpu"}):
            seed_benchmark_rng(12345)
            first = torch.rand(4)
            seed_benchmark_rng(54321)
            second = torch.rand(4)

        self.assertFalse(torch.equal(first, second))

    def test_units_unseen_prompt_uses_forecasting_token_centroid(self):
        """Regression-check that units unseen prompt uses forecasting token centroid.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        try:
            from models.units import UniTSModel
        except ModuleNotFoundError as exc:
            if exc.name == "timm":
                self.skipTest("Units model dependency is isolated in .venv_units")
            raise

        state = {
            "prompt_tokens.ECL": torch.full((1, 2, 3, 4), 2.0),
            "mask_tokens.ECL": torch.full((1, 2, 1, 4), 4.0),
            "cls_tokens.LTF_ECL_p96": torch.zeros(1),
            "prompt_tokens.ETTh1": torch.full((1, 1, 3, 4), 6.0),
            "mask_tokens.ETTh1": torch.full((1, 1, 1, 4), 8.0),
            "cls_tokens.LTF_ETTh1_p96": torch.zeros(1),
            "prompt_tokens.ECG5000": torch.full((1, 1, 3, 4), 100.0),
            "mask_tokens.ECG5000": torch.full((1, 1, 1, 4), 100.0),
            "cls_tokens.CLS_ECG5000": torch.zeros(1),
        }

        sources = UniTSModel._initialize_unseen_forecast_tokens(state)

        self.assertEqual(sources, ("ECL", "ETTh1"))
        torch.testing.assert_close(state["prompt_tokens.custom"], torch.full((1, 1, 3, 4), 4.0))
        torch.testing.assert_close(state["mask_tokens.custom"], torch.full((1, 1, 1, 4), 6.0))


if __name__ == "__main__":
    unittest.main()
