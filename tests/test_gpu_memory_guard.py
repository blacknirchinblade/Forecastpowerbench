"""Regression-test the gpu memory guard benchmark contract."""

import unittest
from unittest.mock import patch

from benchmark.runner import BenchmarkRunner


class GPUMemoryGuardTests(unittest.TestCase):
    """Define the GPUMemoryGuardTests type and its associated runtime state or contract.
    
    Bases:
        unittest.TestCase
    """
    @staticmethod
    def _runner(observations, limit_gb=42.0):
        """Construct a minimal runner with controlled GPU-memory observations.
        
        Args:
            observations (Any): Configuration or runtime value named observations.
            limit_gb (Any): Configuration or runtime value named limit_gb. Defaults to 42.0.
        
        Returns:
            Runtime value with no declared return annotation; see the implementation and callers.
        """
        runner = BenchmarkRunner.__new__(BenchmarkRunner)
        runner.gpu_batch_memory_peaks = {
            ("model", 512, 96): list(observations)
        }
        runner._execution_device = lambda: "cuda"
        runner._gpu_memory_limit_gb = lambda: limit_gb
        return runner

    def test_linear_growth_skips_unsafe_next_batch(self):
        """Regression-check that linear growth skips unsafe next batch.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        runner = self._runner([(512, 11.52), (1024, 22.57)])
        with patch("benchmark.runner.torch.cuda.is_available", return_value=True):
            should_skip, estimate, limit = runner._should_skip_gpu_batch_for_memory(
                "model", 512, 96, 2048
            )
        self.assertTrue(should_skip)
        self.assertGreater(estimate, limit)

    def test_flat_memory_growth_allows_next_batch(self):
        """Regression-check that flat memory growth allows next batch.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        runner = self._runner([(512, 4.0), (1024, 4.5)])
        with patch("benchmark.runner.torch.cuda.is_available", return_value=True):
            should_skip, estimate, limit = runner._should_skip_gpu_batch_for_memory(
                "model", 512, 96, 2048
            )
        self.assertFalse(should_skip)
        self.assertLess(estimate, limit)

    def test_observations_are_model_and_configuration_specific(self):
        """Regression-check that observations are model and configuration specific.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        runner = self._runner([(512, 11.52), (1024, 22.57)])
        with patch("benchmark.runner.torch.cuda.is_available", return_value=True):
            should_skip, estimate, _ = runner._should_skip_gpu_batch_for_memory(
                "other-model", 512, 96, 2048
            )
        self.assertFalse(should_skip)
        self.assertIsNone(estimate)


if __name__ == "__main__":
    unittest.main()
