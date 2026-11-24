"""Regression-test the gpu preflight benchmark contract."""

import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from benchmark.config import DeviceCategory
from benchmark.runner import BenchmarkRunner


class _Utilization:
    """Define the _Utilization type and its associated runtime state or contract.
    
    Attributes:
        gpu: Gpu state retained by each instance.
    """
    def __init__(self, gpu):
        """Initialize _Utilization and capture its validated configuration or runtime state.
        
        Args:
            gpu (Any): Configuration or runtime value named gpu.
        
        Returns:
            None; initializes the instance in place.
        """
        self.gpu = gpu


class GPUPreflightTests(unittest.TestCase):
    """Define the GPUPreflightTests type and its associated runtime state or contract.
    
    Bases:
        unittest.TestCase
    """
    @staticmethod
    def _runner(limit=10.0):
        """Construct a runner with mocked GPU-utilization samples for preflight tests.
        
        Args:
            limit (Any): Configuration or runtime value named limit. Defaults to 10.0.
        
        Returns:
            Runtime value with no declared return annotation; see the implementation and callers.
        """
        runner = BenchmarkRunner.__new__(BenchmarkRunner)
        runner.config = SimpleNamespace(
            device_category=DeviceCategory.RTX5090,
            max_idle_gpu_util_percent=limit,
        )
        return runner

    def test_transient_busy_gpu_waits_then_accepts_idle_window(self):
        """Regression-check that transient busy gpu waits then accepts idle window.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        values = iter([12.0] * 5 + [0.0] * 5)
        fake_nvml = types.SimpleNamespace(
            nvmlDeviceGetHandleByIndex=lambda _index: object(),
            nvmlDeviceGetUtilizationRates=lambda _handle: _Utilization(next(values)),
        )
        monotonic_values = iter([0.0, 0.5, 1.0])
        with patch.dict(
            sys.modules, {"pynvml": fake_nvml}
        ), patch.dict(
            os.environ,
            {"GPU_PREFLIGHT_WAIT_SEC": "30", "GPU_PREFLIGHT_POLL_SEC": "1"},
        ), patch(
            "benchmark.runner.time.sleep"
        ), patch(
            "benchmark.runner.time.monotonic", side_effect=lambda: next(monotonic_values)
        ):
            self._runner()._preflight_device_exclusivity()

    def test_persistently_busy_gpu_is_rejected(self):
        """Regression-check that persistently busy gpu is rejected.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        fake_nvml = types.SimpleNamespace(
            nvmlDeviceGetHandleByIndex=lambda _index: object(),
            nvmlDeviceGetUtilizationRates=lambda _handle: _Utilization(25.0),
        )
        with patch.dict(
            sys.modules, {"pynvml": fake_nvml}
        ), patch.dict(
            os.environ,
            {"GPU_PREFLIGHT_WAIT_SEC": "0", "GPU_PREFLIGHT_POLL_SEC": "1"},
        ), patch(
            "benchmark.runner.time.sleep"
        ), patch(
            "benchmark.runner.time.monotonic", side_effect=[0.0, 0.0]
        ):
            with self.assertRaisesRegex(RuntimeError, "remained busy"):
                self._runner()._preflight_device_exclusivity()


if __name__ == "__main__":
    unittest.main()
