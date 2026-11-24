"""Regression-test the result validation benchmark contract."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.validate_results import validate_predictions, validate_summary


class ResultValidationTests(unittest.TestCase):
    """Define the ResultValidationTests type and its associated runtime state or contract.
    
    Bases:
        unittest.TestCase
    """
    def _write_prediction(self, root: Path, policy: str = "per_source_global_median") -> None:
        """Write prediction for ResultValidationTests.
        
        Args:
            root (Path): Configuration or runtime value named root.
            policy (str): Named validation or preprocessing policy recorded in metadata. Defaults to
                          'per_source_global_median'.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        path = root / "benchmark_predictions" / "units" / "source.npz"
        path.parent.mkdir(parents=True)
        np.savez(
            path,
            y_pred=np.ones((2, 3), dtype=np.float32),
            y_true=np.ones((2, 3), dtype=np.float32),
            y_true_observed=np.ones((2, 3), dtype=np.bool_),
            benchmark_seed=np.asarray(12345),
            reference_batch_size=np.asarray(512),
            context_mae_scale=np.ones(2, dtype=np.float32),
            context_mse_scale=np.ones(2, dtype=np.float32),
            accuracy_seasonality=np.asarray(24),
            imputation_policy=np.asarray(policy),
        )

    def test_prediction_validation_accepts_release_schema(self):
        """Regression-check that prediction validation accepts release schema.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_prediction(root)

            errors = validate_predictions(root, ["units"], 12345, 512)

            self.assertEqual(errors, [])

    def test_prediction_validation_rejects_unknown_imputation(self):
        """Regression-check that prediction validation rejects unknown imputation.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_prediction(root, policy="unknown")

            errors = validate_predictions(root, ["units"], 12345, 512)

            self.assertTrue(any("imputation policy" in error for error in errors))

    def test_zero_gpu_utilization_is_valid_when_telemetry_is_finite(self):
        """Regression-check that zero gpu utilization is valid when telemetry is finite.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "benchmark_results_batch_summary.csv"
            row = {
                "model_name": "units",
                "context_len": "512",
                "horizon_len": "96",
                "batch_size": "4",
                "status_values": "SUCCESS",
                "is_complete_success": "true",
                "measurement_duration_sec_mean": "1",
                "throughput_windows_per_sec_mean": "40",
                "throughput_points_per_sec_mean": "3840",
                "peak_cpu_ram_gb_mean": "1",
                "avg_power_w_mean": "50",
                "total_energy_j_mean": "50",
                "power_sample_count_mean": "20",
                "power_scope": "gpu_board",
                "peak_gpu_allocated_gb_mean": "1",
                "peak_gpu_memory_used_gb_mean": "2",
                "gpu_memory_total_gb_mean": "32",
                "avg_gpu_compute_util_percent_mean": "0",
                "avg_gpu_temp_c_mean": "40",
            }
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)

            errors = validate_summary(root, ["units"], "5090")

            self.assertEqual(errors, [])

    def test_hopper_devices_require_discrete_gpu_telemetry(self):
        """Reject Hopper success rows when dedicated-GPU metrics are absent.

        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "benchmark_results_batch_summary.csv"
            row = {
                "model_name": "units",
                "context_len": "512",
                "horizon_len": "96",
                "batch_size": "64",
                "status_values": "SUCCESS",
                "is_complete_success": "true",
                "measurement_duration_sec_mean": "1",
                "throughput_windows_per_sec_mean": "64",
                "throughput_points_per_sec_mean": "6144",
                "peak_cpu_ram_gb_mean": "1",
                "avg_power_w_mean": "300",
                "total_energy_j_mean": "300",
                "power_sample_count_mean": "20",
                "power_scope": "gpu_board",
            }
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)

            for device in ("h100", "h200"):
                with self.subTest(device=device):
                    errors = validate_summary(root, ["units"], device)
                    self.assertTrue(
                        any(
                            "peak_gpu_memory_used_gb_mean" in error
                            for error in errors
                        )
                    )

    def test_jetson_hardware_throttle_status_is_rejected(self):
        """Regression-check that jetson hardware throttle status is rejected.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "benchmark_results_batch_summary.csv"
            row = {
                "model_name": "units",
                "context_len": "512",
                "horizon_len": "96",
                "batch_size": "64",
                "status_values": "HARDWARE_THROTTLED",
                "is_complete_success": "false",
                "power_scope": "sum_vdd_gpu_soc_vdd_cpu_cv_vin_sys_5v0",
            }
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)

            errors = validate_summary(root, ["units"], "jetson")

            self.assertTrue(any("HARDWARE_THROTTLED" in error for error in errors))

    def test_cpu_timeout_is_a_valid_terminal_configuration(self):
        """Regression-check that cpu timeout is a valid terminal configuration.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "benchmark_results_batch_summary.csv"
            row = {
                "model_name": "units",
                "context_len": "512",
                "horizon_len": "96",
                "batch_size": "65536",
                "status_values": "TIMEOUT",
                "is_complete_success": "false",
                "power_scope": "cpu_package",
            }
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)

            errors = validate_summary(root, ["units"], "cpu")

            self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
