"""Regression-test the config timeout benchmark contract."""

import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from benchmark.config import BenchmarkConfig, DeviceCategory
from benchmark.data import SourceSeries
from benchmark.runner import BenchmarkRunner, ConfigDurationExceeded


class _Monitor:
    """Define the _Monitor type and its associated runtime state or contract."""
    def start(self):
        """Start _Monitor collection or lifecycle state.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        pass

    def stop(self):
        """Stop _Monitor collection or lifecycle state and return its final result when applicable.
        
        Returns:
            Runtime value with no declared return annotation; see the implementation and callers.
        """
        return {}


class _SlowModel:
    """Define the _SlowModel type and its associated runtime state or contract."""
    def forecast(self, inputs, horizon):
        """Run _SlowModel point forecasting for one batched context input.
        
        Args:
            inputs (Any): Batched model context with shape (batch, context).
            horizon (Any): Configuration or runtime value named horizon.
        
        Returns:
            A pair containing normalized point forecasts and adapter-reported model-call duration.
        """
        time.sleep(1.0)
        return np.zeros((len(inputs), horizon), dtype=np.float32), 1.0


class ConfigTimeoutTests(unittest.TestCase):
    """Define the ConfigTimeoutTests type and its associated runtime state or contract.
    
    Bases:
        unittest.TestCase
    """
    def test_slow_cpu_configuration_is_interrupted(self):
        """Regression-check that slow cpu configuration is interrupted.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BenchmarkConfig(
                device_category=DeviceCategory.CPU,
                models=["units"],
                context_lens=[4],
                horizon_lens=[2],
                batch_sizes=[2],
                repeats=1,
                warmups=0,
                benchmark_batches=1,
                max_config_duration_sec=0.05,
                output_dir=temp_dir,
            )
            source = SourceSeries(
                source_id="source",
                source_file="source.csv",
                source_relative_path="source.csv",
                dataset_group="dataset",
                source_path="source.csv",
                target_column="PM2.5",
                time_column="Timestamp",
                values=np.arange(10, dtype=np.float32),
                observed_mask=None,
                timestamps_ns=None,
                regular_start_timestamp_ns=0,
                segments=np.asarray([[0, 10]], dtype=np.int64),
                expected_interval_ns=1,
                preprocessing={},
            )
            runner = BenchmarkRunner(config, object(), _Monitor())
            runner._measure_idle_baseline = lambda: 0.0

            started = time.monotonic()
            with patch("benchmark.runner.get_model", return_value=_SlowModel()), patch.dict(
                "os.environ", {"MODEL_STABILIZATION_SEC": "0", "TQDM_DISABLE": "1"}
            ):
                with self.assertRaises(ConfigDurationExceeded):
                    runner._run_source_batch_size(
                        model_name="units",
                        ctx=4,
                        hor=2,
                        batch_size=2,
                        sources=[source],
                        dataset_name="dataset",
                    )

            self.assertLess(time.monotonic() - started, 0.75)


if __name__ == "__main__":
    unittest.main()
