"""Collect and summarize sysfs telemetry around measured inference intervals."""

import glob
import logging
import os
import shutil
import subprocess
import threading
import time
from typing import Dict, Optional

import numpy as np

from benchmark.monitors.base import PowerMonitor
from benchmark.monitors.stats import summarize_power_interval

logger = logging.getLogger(__name__)


class SysfsPowerMonitor(PowerMonitor):
    """Sample an ARM board or external sensor exposed through Linux sysfs.
    
    Bases:
        PowerMonitor
    
    Attributes:
        end_ns: Inclusive measurement end timestamp in nanoseconds.
        interval_ms: Telemetry sampling interval in milliseconds.
        running: Running state retained by each instance.
        sample_rows: Sample rows state retained by each instance.
        samples: Samples state retained by each instance.
        scale: Scale state retained by each instance.
        sensor_path: Sensor path state retained by each instance.
        start_ns: Inclusive measurement start timestamp in nanoseconds.
        thread: Thread state retained by each instance.
        throttled_end: Throttled end state retained by each instance.
        throttled_start: Throttled start state retained by each instance.
    """

    def __init__(self, interval_ms: int = 50):
        """Initialize SysfsPowerMonitor and capture its validated configuration or runtime state.
        
        Args:
            interval_ms (int): Telemetry sampling interval in milliseconds. Defaults to 50.
        
        Returns:
            None; initializes the instance in place.
        """
        self.interval_ms = interval_ms
        self.sensor_path = self._find_power_sensor()
        self.scale = float(os.environ.get("POWER_SENSOR_SCALE", "1e-6"))
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.samples = []
        self.sample_rows = []
        self.start_ns = None
        self.end_ns = None
        self.throttled_start: Optional[int] = None
        self.throttled_end: Optional[int] = None
        if self.sensor_path:
            logger.info(
                "[SYSFS] Board power sensor: %s (scale=%g W/unit)",
                self.sensor_path,
                self.scale,
            )
        else:
            logger.warning(
                "[SYSFS] No board power sensor found. Power fields will be NaN. "
                "Expose an INA2xx/USB sensor through hwmon or set POWER_SENSOR_PATH."
            )

    @staticmethod
    def _find_power_sensor() -> Optional[str]:
        """Find power sensor for SysfsPowerMonitor.
        
        Returns:
            Optional[str]: Result produced by this operation.
        """
        configured = os.environ.get("POWER_SENSOR_PATH", "").strip()
        if configured:
            if os.path.isfile(configured) and os.access(configured, os.R_OK):
                return configured
            logger.warning("[SYSFS] POWER_SENSOR_PATH is not readable: %s", configured)
            return None

        candidates = sorted(glob.glob("/sys/class/hwmon/hwmon*/power*_input"))
        readable = [
            path for path in candidates if os.path.isfile(path) and os.access(path, os.R_OK)
        ]
        if len(readable) > 1:
            logger.warning(
                "[SYSFS] Multiple power sensors found; using %s. Set POWER_SENSOR_PATH to select explicitly.",
                readable[0],
            )
        return readable[0] if readable else None

    @staticmethod
    def _read_cpu_temp() -> float:
        """Read cpu temp for SysfsPowerMonitor.
        
        Returns:
            float: Computed scalar described by this callable.
        """
        for path in sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp")):
            try:
                with open(path, encoding="ascii") as handle:
                    value = float(handle.read().strip())
                return value / 1000.0 if value > 1000 else value
            except (OSError, ValueError):
                continue
        return float("nan")

    def _read_power_w(self) -> Optional[float]:
        """Read power w for SysfsPowerMonitor.
        
        Returns:
            Optional[float]: Result produced by this operation.
        """
        if not self.sensor_path:
            return None
        try:
            with open(self.sensor_path, encoding="ascii") as handle:
                value = float(handle.read().strip())
        except (OSError, ValueError) as exc:
            logger.warning("[SYSFS] Lost access to power sensor %s: %s", self.sensor_path, exc)
            self.sensor_path = None
            return None
        power_w = value * self.scale
        return power_w if np.isfinite(power_w) and power_w >= 0 else None

    @staticmethod
    def _parse_throttled_output(output: str) -> Optional[int]:
        """Parse throttled output for SysfsPowerMonitor.
        
        Args:
            output (str): Configuration or runtime value named output.
        
        Returns:
            Optional[int]: Result produced by this operation.
        """
        value = output.strip().split("=", maxsplit=1)[-1]
        try:
            return int(value, 0)
        except ValueError:
            return None

    @classmethod
    def _read_rpi_throttled(cls) -> Optional[int]:
        """Read rpi throttled for SysfsPowerMonitor.
        
        Returns:
            Optional[int]: Result produced by this operation.
        """
        command = shutil.which("vcgencmd")
        if not command:
            return None
        try:
            result = subprocess.run(
                [command, "get_throttled"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return cls._parse_throttled_output(result.stdout)

    @staticmethod
    def _rpi_throttle_event(start: Optional[int], end: Optional[int]) -> bool:
        """Return whether Raspberry Pi throttle flags were active or changed during the interval.
        
        Args:
            start (Optional[int]): Configuration or runtime value named start.
            end (Optional[int]): Configuration or runtime value named end.
        
        Returns:
            bool: Whether the condition described by this callable holds.
        """
        if start is None or end is None:
            return False
        current_mask = 0xF
        historical_mask = 0xF0000
        new_historical_flags = (end & historical_mask) & ~(start & historical_mask)
        return bool((start & current_mask) or (end & current_mask) or new_historical_flags)

    def _monitor_loop(self) -> None:
        """Run the background collection loop owned by SysfsPowerMonitor until it is stopped.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        start_ns = time.time_ns()
        next_sample = time.perf_counter()
        while self.running:
            timestamp_ns = time.time_ns()
            power_w = self._read_power_w()
            if power_w is not None:
                self.samples.append((timestamp_ns, power_w))
            cpu_temp_c = self._read_cpu_temp()
            power_text = f"{power_w:.6f}W" if power_w is not None else "unavailable"
            self.sample_rows.append(
                {
                    "sample_index": len(self.sample_rows),
                    "timestamp_ns": timestamp_ns,
                    "elapsed_ms": (timestamp_ns - start_ns) / 1e6,
                    "power_w": power_w if power_w is not None else float("nan"),
                    "gpu_compute_util_percent": float("nan"),
                    "gpu_memory_util_percent": float("nan"),
                    "gpu_temp_c": float("nan"),
                    "cpu_temp_c": cpu_temp_c,
                    "raw_line": f"power={power_text} cpu_temp={cpu_temp_c:.1f}C",
                }
            )
            next_sample += self.interval_ms / 1000.0
            delay = next_sample - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_sample = time.perf_counter()

    def start(self) -> None:
        """Start SysfsPowerMonitor collection or lifecycle state.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        self.samples = []
        self.sample_rows = []
        self.throttled_start = self._read_rpi_throttled()
        self.throttled_end = None
        self.start_ns = time.time_ns()
        self.end_ns = None
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self) -> Dict[str, float]:
        """Stop SysfsPowerMonitor collection or lifecycle state and return its final result when applicable.
        
        Returns:
            Dict[str, float]: Named values produced by this operation.
        """
        self.end_ns = time.time_ns()
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self.thread = None
        self.throttled_end = self._read_rpi_throttled()

        cpu_temps = np.asarray(
            [row["cpu_temp_c"] for row in self.sample_rows], dtype=np.float64
        )
        finite_cpu_temps = cpu_temps[np.isfinite(cpu_temps)]
        throttle_available = self.throttled_start is not None and self.throttled_end is not None
        throttle_event = self._rpi_throttle_event(self.throttled_start, self.throttled_end)
        common_metrics = {
            "avg_gpu_compute_util_percent": float("nan"),
            "avg_gpu_memory_util_percent": float("nan"),
            "gpu_compute_util_std_percent": float("nan"),
            "gpu_memory_util_std_percent": float("nan"),
            "avg_gpu_temp_c": float("nan"),
            "max_gpu_temp_c": float("nan"),
            "avg_cpu_temp_c": float(np.mean(finite_cpu_temps))
            if finite_cpu_temps.size
            else self._read_cpu_temp(),
            "rpi_throttle_status_available": throttle_available,
            "rpi_throttled_start": self.throttled_start,
            "rpi_throttled_end": self.throttled_end,
            "hardware_throttled": throttle_event,
            "telemetry_samples": list(self.sample_rows),
        }

        powers = np.asarray([value for _, value in self.samples], dtype=np.float64)
        if powers.size:
            timestamps = np.asarray([timestamp for timestamp, _ in self.samples], dtype=np.int64)
            result = summarize_power_interval(powers, timestamps, self.start_ns, self.end_ns)
            result["power_scope"] = "configured_board_sensor"
            result.update(common_metrics)
            return result

        result = {
            "power_scope": "unavailable",
            "power_sample_count": 0,
            "avg_power_w": float("nan"),
            "power_std_w": float("nan"),
            "power_p50_w": float("nan"),
            "power_p95_w": float("nan"),
            "peak_power_w": float("nan"),
            "total_energy_j": float("nan"),
            "telemetry_duration_sec": (
                max(0.0, (self.end_ns - self.start_ns) / 1e9)
                if self.start_ns is not None and self.end_ns is not None
                else float("nan")
            ),
        }
        result.update(common_metrics)
        return result

    def get_raw_line(self) -> str:
        """Return a concise representation of the most recent sysfs sample.
        
        Returns:
            str: Result produced by this operation.
        """
        return self.sample_rows[-1]["raw_line"] if self.sample_rows else ""
