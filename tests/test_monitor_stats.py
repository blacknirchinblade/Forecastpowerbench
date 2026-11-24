"""Regression-test the monitor stats benchmark contract."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from benchmark.monitors.stats import summarize_power_interval
from benchmark.runner import BenchmarkRunner, t_critical_95
from benchmark.monitors.jetson import JetsonPowerMonitor
from benchmark.monitors.nvml import NVMLPowerMonitor
from benchmark.monitors.sysfs import SysfsPowerMonitor


class PowerIntervalTests(unittest.TestCase):
    """Define the PowerIntervalTests type and its associated runtime state or contract.
    
    Bases:
        unittest.TestCase
    """
    def test_nvml_monitor_retains_selected_physical_gpu_index(self):
        """Retain the physical GPU index shared with CUDA and device control.

        Returns:
            None; the callable operates through assertions.
        """
        self.assertEqual(NVMLPowerMonitor(gpu_index=2).gpu_index, 2)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            NVMLPowerMonitor(gpu_index=-1)

    def test_small_sample_confidence_interval_uses_student_t(self):
        """Regression-check that small sample confidence interval uses student t.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        self.assertAlmostEqual(t_critical_95(5), 2.776)

    def test_samples_outside_measurement_window_are_not_counted(self):
        """Regression-check that samples outside measurement window are not counted.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        metrics = summarize_power_interval(
            powers=[10.0, 20.0, 30.0, 40.0],
            timestamps_ns=[0, 1_000_000_000, 2_000_000_000, 3_000_000_000],
            start_ns=1_000_000_000,
            end_ns=2_000_000_000,
        )
        self.assertEqual(metrics["power_sample_count"], 2)
        self.assertAlmostEqual(metrics["telemetry_duration_sec"], 1.0)
        self.assertAlmostEqual(metrics["avg_power_w"], 25.0)
        self.assertAlmostEqual(metrics["total_energy_j"], 25.0)

    def test_interval_boundaries_are_interpolated(self):
        """Regression-check that interval boundaries are interpolated.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        metrics = summarize_power_interval(
            powers=[10.0, 20.0, 30.0, 40.0],
            timestamps_ns=[0, 1_000_000_000, 2_000_000_000, 3_000_000_000],
            start_ns=500_000_000,
            end_ns=2_500_000_000,
        )
        self.assertEqual(metrics["power_sample_count"], 2)
        self.assertAlmostEqual(metrics["avg_power_w"], 25.0)
        self.assertAlmostEqual(metrics["total_energy_j"], 50.0)

    def test_gpu_memory_is_aggregated_without_retaining_raw_samples(self):
        """Regression-check that gpu memory is aggregated without retaining raw samples.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        metrics = BenchmarkRunner._align_monitor_metrics(
            {
                "gpu_memory_scope": "dedicated_vram",
                "telemetry_samples": [
                    {
                        "timestamp_ns": 1_000_000_000,
                        "power_w": 10.0,
                        "gpu_memory_used_gb": 2.0,
                        "gpu_memory_free_gb": 8.0,
                        "gpu_memory_total_gb": 10.0,
                    },
                    {
                        "timestamp_ns": 2_000_000_000,
                        "power_w": 20.0,
                        "gpu_memory_used_gb": 3.0,
                        "gpu_memory_free_gb": 7.0,
                        "gpu_memory_total_gb": 10.0,
                    },
                    {
                        "timestamp_ns": 3_000_000_000,
                        "power_w": 30.0,
                        "gpu_memory_used_gb": 5.0,
                        "gpu_memory_free_gb": 5.0,
                        "gpu_memory_total_gb": 10.0,
                    },
                ],
            },
            start_ns=2_000_000_000,
            end_ns=3_000_000_000,
        )

        self.assertAlmostEqual(metrics["gpu_memory_total_gb"], 10.0)
        self.assertAlmostEqual(metrics["avg_gpu_memory_used_gb"], 4.0)
        self.assertAlmostEqual(metrics["peak_gpu_memory_used_gb"], 5.0)
        self.assertAlmostEqual(metrics["min_gpu_memory_free_gb"], 5.0)
        self.assertEqual(metrics["gpu_memory_scope"], "dedicated_vram")
        self.assertNotIn("telemetry_samples", metrics)

    def test_jetson_shared_memory_parser_reports_gib(self):
        """Regression-check that jetson shared memory parser reports gib.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        parsed = JetsonPowerMonitor._parse_shared_ram(
            "RAM 11599/62878MB (lfb 906x4MB) GR3D_FREQ 99%"
        )
        self.assertIsNotNone(parsed)
        used_gb, total_gb = parsed
        self.assertAlmostEqual(used_gb, 11599 / 1024.0)
        self.assertAlmostEqual(total_gb, 62878 / 1024.0)

    def test_jetson_temperature_parser_accepts_both_formats(self):
        """Regression-check that jetson temperature parser accepts both formats.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        monitor = JetsonPowerMonitor()
        temperatures = monitor._parse_temperatures("cpu@41.2C gpu@43.218C/100.0C tj@45.0C")

        self.assertEqual(temperatures["cpu"], 41.2)
        self.assertEqual(temperatures["gpu"], 43.218)
        self.assertEqual(temperatures["tj"], 45.0)

    def test_jetson_oc_counter_reader_uses_channel_names(self):
        """Regression-check that jetson oc counter reader uses channel names.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            hwmon = root / "hwmon9"
            hwmon.mkdir()
            (hwmon / "oc1_event_cnt").write_text("2\n")
            (hwmon / "oc3_event_cnt").write_text("17\n")
            with patch("benchmark.monitors.jetson.Path", return_value=root):
                counts = JetsonPowerMonitor._read_oc_event_counts()
        self.assertEqual(counts, {"oc1": 2, "oc3": 17})

    def test_rpi_throttle_parser_accepts_vcgencmd_hex(self):
        """Regression-check that rpi throttle parser accepts vcgencmd hex.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        self.assertEqual(SysfsPowerMonitor._parse_throttled_output("throttled=0xe0008\n"), 0xE0008)
        self.assertIsNone(SysfsPowerMonitor._parse_throttled_output("not available"))

    def test_rpi_current_throttle_invalidates_interval(self):
        """Regression-check that rpi current throttle invalidates interval.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        self.assertTrue(SysfsPowerMonitor._rpi_throttle_event(0xE0008, 0xE0008))

    def test_rpi_new_historical_throttle_invalidates_interval(self):
        """Regression-check that rpi new historical throttle invalidates interval.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        self.assertTrue(SysfsPowerMonitor._rpi_throttle_event(0, 0x80000))

    def test_rpi_old_historical_flags_do_not_invalidate_clean_interval(self):
        """Regression-check that rpi old historical flags do not invalidate clean interval.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        self.assertFalse(SysfsPowerMonitor._rpi_throttle_event(0xE0000, 0xE0000))

    def test_rpi_monitor_samples_temperature_without_power_sensor(self):
        """Regression-check that rpi monitor samples temperature without power sensor.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        monitor = SysfsPowerMonitor(interval_ms=1)
        monitor.sensor_path = None
        with (
            patch.object(monitor, "_read_rpi_throttled", side_effect=[0, 0]),
            patch.object(monitor, "_read_cpu_temp", return_value=47.5),
            patch("benchmark.monitors.sysfs.time.sleep", side_effect=lambda _: None),
        ):
            monitor.start()
            while not monitor.sample_rows:
                pass
            metrics = monitor.stop()

        self.assertEqual(metrics["power_scope"], "unavailable")
        self.assertEqual(metrics["power_sample_count"], 0)
        self.assertEqual(metrics["avg_cpu_temp_c"], 47.5)
        self.assertTrue(metrics["rpi_throttle_status_available"])
        self.assertFalse(metrics["hardware_throttled"])
        self.assertGreaterEqual(len(metrics["telemetry_samples"]), 1)


if __name__ == "__main__":
    unittest.main()
