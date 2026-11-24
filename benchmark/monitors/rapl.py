"""Collect and summarize rapl telemetry around measured inference intervals."""

import threading
import time
import logging
import numpy as np
import glob
import os
from typing import Dict
from benchmark.monitors.base import PowerMonitor
from benchmark.monitors.stats import summarize_power_interval

logger = logging.getLogger(__name__)

NAN_METRICS = {
    "power_scope": "cpu_package",
    "power_sample_count": 0,
    "avg_power_w": float("nan"),
    "power_std_w": float("nan"),
    "power_p50_w": float("nan"),
    "power_p95_w": float("nan"),
    "peak_power_w": float("nan"),
    "total_energy_j": float("nan"),
    "avg_gpu_compute_util_percent": float("nan"),
    "avg_gpu_memory_util_percent": float("nan"),
    "gpu_compute_util_std_percent": float("nan"),
    "gpu_memory_util_std_percent": float("nan"),
    "avg_gpu_temp_c": float("nan"),
    "max_gpu_temp_c": float("nan"),
    "avg_cpu_temp_c": float("nan"),
    "telemetry_samples": [],
}


class RAPLPowerMonitor(PowerMonitor):
    """Reads CPU package power from Intel RAPL sysfs interface.
    
    Bases:
        PowerMonitor
    
    Attributes:
        cpu_temp_paths: Cpu temp paths state retained by each instance.
        cumulative_energy_j: Cumulative energy j state retained by each instance.
        disabled_reason: Disabled reason state retained by each instance.
        end_ns: Inclusive measurement end timestamp in nanoseconds.
        end_time: End time state retained by each instance.
        interval_ms: Telemetry sampling interval in milliseconds.
        max_energy_range_uj: Max energy range uj state retained by each instance.
        max_power_w: Max power w state retained by each instance.
        power_samples: Power samples state retained by each instance.
        rapl_paths: Rapl paths state retained by each instance.
        running: Running state retained by each instance.
        sample_rows: Sample rows state retained by each instance.
        start_ns: Inclusive measurement start timestamp in nanoseconds.
        start_time: Start time state retained by each instance.
        thread: Thread state retained by each instance.
        timestamps: Timestamps state retained by each instance.
    """

    def __init__(self, interval_ms: int = 50):
        """Initialize RAPLPowerMonitor and capture its validated configuration or runtime state.
        
        Args:
            interval_ms (int): Telemetry sampling interval in milliseconds. Defaults to 50.
        
        Returns:
            None; initializes the instance in place.
        """
        self.interval_ms = interval_ms
        self.thread = None
        self.running = False

        self.power_samples = []
        self.timestamps = []
        self.rapl_paths = self._readable_rapl_paths()
        self.sample_rows = []
        self.disabled_reason = ""
        self.max_power_w = float(os.environ.get("RAPL_MAX_POWER_W", "1000"))
        self.max_energy_range_uj = self._read_max_energy_ranges()
        self.cpu_temp_paths = self._find_cpu_temp_paths()
        self.cumulative_energy_j = 0.0
        self.start_time = None
        self.end_time = None
        self.start_ns = None
        self.end_ns = None
        if not self.rapl_paths:
            self.disabled_reason = "No readable Intel RAPL energy_uj files found"
            logger.warning("[RAPL] %s. CPU power telemetry will be NaN.", self.disabled_reason)
        if not self.cpu_temp_paths:
            logger.warning(
                "[RAPL] No CPU package temperature sensor found; avg_cpu_temp_c will be NaN."
            )

    def _readable_rapl_paths(self):
        """Return RAPL energy-counter paths that exist and are readable by the benchmark process.
        
        Returns:
            Runtime value with no declared return annotation; see the implementation and callers.
        """
        readable_paths = []
        for path in glob.glob("/sys/class/powercap/intel-rapl:*/energy_uj"):
            # Read package domains only (intel-rapl:N). Subdomains such as
            # intel-rapl:N:M are already included in package energy.
            if os.path.basename(os.path.dirname(path)).count(":") != 1:
                continue
            try:
                with open(path, "r") as f:
                    value = f.read().strip()
                if value.isdigit():
                    readable_paths.append(path)
            except PermissionError:
                logger.warning("[RAPL] Permission denied for %s; skipping this domain.", path)
            except OSError as exc:
                logger.warning("[RAPL] Could not read %s: %s", path, exc)
        return readable_paths

    def _read_max_energy_ranges(self):
        """Read max energy ranges for RAPLPowerMonitor.
        
        Returns:
            Runtime value with no declared return annotation; see the implementation and callers.
        """
        ranges = {}
        for path in self.rapl_paths:
            range_path = os.path.join(os.path.dirname(path), "max_energy_range_uj")
            try:
                with open(range_path, "r") as f:
                    ranges[path] = int(f.read().strip())
            except (OSError, ValueError):
                continue
        return ranges

    @staticmethod
    def _find_cpu_temp_paths():
        """Find cpu temp paths for RAPLPowerMonitor.
        
        Returns:
            Runtime value with no declared return annotation; see the implementation and callers.
        """
        paths = []
        for hwmon_dir in glob.glob("/sys/class/hwmon/hwmon*"):
            try:
                with open(os.path.join(hwmon_dir, "name"), "r") as f:
                    name = f.read().strip().lower()
            except OSError:
                continue
            if name not in {"coretemp", "k10temp", "zenpower"}:
                continue
            package_paths = []
            for label_path in glob.glob(os.path.join(hwmon_dir, "temp*_label")):
                try:
                    with open(label_path, "r") as f:
                        label = f.read().strip().lower()
                except OSError:
                    continue
                if "package" in label or label in {"tctl", "tdie"}:
                    input_path = label_path.replace("_label", "_input")
                    if os.path.isfile(input_path):
                        package_paths.append(input_path)
            paths.extend(package_paths or glob.glob(os.path.join(hwmon_dir, "temp1_input")))
        return sorted(set(paths))

    def _read_cpu_temp_c(self):
        """Read cpu temp c for RAPLPowerMonitor.
        
        Returns:
            Runtime value with no declared return annotation; see the implementation and callers.
        """
        temperatures = []
        for path in self.cpu_temp_paths:
            try:
                with open(path, "r") as f:
                    value = float(f.read().strip())
                temperatures.append(value / 1000.0 if value > 1000 else value)
            except (OSError, ValueError):
                continue
        return max(temperatures) if temperatures else float("nan")

    def _read_energy_values(self):
        """Read energy values for RAPLPowerMonitor.
        
        Returns:
            Runtime value with no declared return annotation; see the implementation and callers.
        """
        values = {}
        for path in self.rapl_paths:
            try:
                with open(path, "r") as f:
                    values[path] = int(f.read().strip())
            except (OSError, ValueError) as exc:
                logger.warning(
                    "[RAPL] Lost access to %s: %s. Disabling CPU power telemetry.", path, exc
                )
                self.rapl_paths = []
                self.disabled_reason = str(exc)
                return None
        return values

    def _monitor_loop(self):
        """Run the background collection loop owned by RAPLPowerMonitor until it is stopped.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        prev_energy = self._read_energy_values()
        if not prev_energy:
            self.running = False
            return
        prev_time = time.perf_counter()
        next_time = prev_time
        self.start_time = prev_time
        self.end_time = prev_time

        while self.running:
            next_time += self.interval_ms / 1000.0
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.perf_counter()
            if not self.running:
                break

            curr_time = time.perf_counter()
            curr_energy = self._read_energy_values()
            if not curr_energy:
                self.running = False
                return
            self.end_time = curr_time

            delta_uj = 0
            for path, curr_value in curr_energy.items():
                prev_value = prev_energy.get(path, curr_value)
                path_delta = curr_value - prev_value
                if path_delta < 0:
                    max_range = self.max_energy_range_uj.get(path)
                    if max_range and prev_value <= max_range:
                        path_delta = (max_range - prev_value) + curr_value
                if path_delta >= 0:
                    delta_uj += path_delta
            delta_t = curr_time - prev_time

            if delta_t > 0 and delta_uj >= 0:
                timestamp_ns = time.time_ns()
                power_w = (delta_uj / 1e6) / delta_t
                if power_w > self.max_power_w:
                    logger.debug("[RAPL] Dropping implausible power sample %.2f W", power_w)
                    prev_energy, prev_time = curr_energy, curr_time
                    continue
                self.cumulative_energy_j += delta_uj / 1e6
                cpu_temp_c = self._read_cpu_temp_c()
                self.power_samples.append(power_w)
                self.timestamps.append(timestamp_ns)
                self.sample_rows.append(
                    {
                        "timestamp_ns": int(timestamp_ns),
                        "power_w": float(power_w),
                        "gpu_compute_util_percent": float("nan"),
                        "gpu_memory_util_percent": float("nan"),
                        "gpu_temp_c": float("nan"),
                        "cpu_temp_c": cpu_temp_c,
                        "raw_line": "",
                    }
                )

            prev_energy, prev_time = curr_energy, curr_time

    def start(self) -> None:
        """Start RAPLPowerMonitor collection or lifecycle state.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        if not self.rapl_paths:
            return
        self.power_samples = []
        self.timestamps = []
        self.sample_rows = []
        self.cumulative_energy_j = 0.0
        self.start_time = None
        self.end_time = None
        self.start_ns = time.time_ns()
        self.end_ns = None
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self) -> Dict[str, float]:
        """Stop RAPLPowerMonitor collection or lifecycle state and return its final result when applicable.
        
        Returns:
            Dict[str, float]: Named values produced by this operation.
        """
        self.end_ns = time.time_ns()
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

        if self.power_samples:
            powers = np.array(self.power_samples)
            result = summarize_power_interval(powers, self.timestamps, self.start_ns, self.end_ns)
            result["power_scope"] = "cpu_package"
            result.update(
                {
                    "avg_gpu_compute_util_percent": float("nan"),
                    "avg_gpu_memory_util_percent": float("nan"),
                    "gpu_compute_util_std_percent": float("nan"),
                    "gpu_memory_util_std_percent": float("nan"),
                    "avg_gpu_temp_c": float("nan"),
                    "max_gpu_temp_c": float("nan"),
                    "avg_cpu_temp_c": float(
                        np.nanmean([sample["cpu_temp_c"] for sample in self.sample_rows])
                    )
                    if any(np.isfinite(sample["cpu_temp_c"]) for sample in self.sample_rows)
                    else float("nan"),
                    "telemetry_samples": [
                        {
                            "sample_index": int(i),
                            "timestamp_ns": int(sample["timestamp_ns"]),
                            "elapsed_ms": float(
                                (sample["timestamp_ns"] - self.sample_rows[0]["timestamp_ns"]) / 1e6
                            ),
                            "power_w": sample["power_w"],
                            "gpu_compute_util_percent": sample["gpu_compute_util_percent"],
                            "gpu_memory_util_percent": sample["gpu_memory_util_percent"],
                            "gpu_temp_c": sample["gpu_temp_c"],
                            "cpu_temp_c": sample["cpu_temp_c"],
                            "raw_line": sample["raw_line"],
                        }
                        for i, sample in enumerate(self.sample_rows)
                    ]
                    if self.sample_rows
                    else [],
                }
            )
            return result
        metrics = dict(NAN_METRICS)
        metrics["power_sample_count"] = int(len(self.power_samples))
        metrics["telemetry_duration_sec"] = (
            max(0.0, (self.end_ns - self.start_ns) / 1e9)
            if self.start_ns is not None and self.end_ns is not None
            else float("nan")
        )
        metrics["avg_cpu_temp_c"] = self._read_cpu_temp_c()
        metrics["telemetry_samples"] = (
            [
                {
                    "sample_index": int(i),
                    "timestamp_ns": int(sample["timestamp_ns"]),
                    "elapsed_ms": float(
                        (sample["timestamp_ns"] - self.sample_rows[0]["timestamp_ns"]) / 1e6
                    ),
                    "power_w": sample["power_w"],
                    "gpu_compute_util_percent": sample["gpu_compute_util_percent"],
                    "gpu_memory_util_percent": sample["gpu_memory_util_percent"],
                    "gpu_temp_c": sample["gpu_temp_c"],
                    "cpu_temp_c": sample["cpu_temp_c"],
                    "raw_line": sample["raw_line"],
                }
                for i, sample in enumerate(self.sample_rows)
            ]
            if self.sample_rows
            else []
        )
        return metrics

    def get_raw_line(self) -> str:
        """Return a concise representation of the most recent RAPL sample.
        
        Returns:
            str: Result produced by this operation.
        """
        return ""
