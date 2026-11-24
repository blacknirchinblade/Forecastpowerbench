"""Collect and summarize jetson telemetry around measured inference intervals."""

import subprocess
import threading
import time
import re
import logging
import select
import numpy as np
from typing import Dict
from pathlib import Path
from benchmark.monitors.base import PowerMonitor
from benchmark.monitors.stats import summarize_power_interval

logger = logging.getLogger(__name__)

try:
    import pynvml

    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False


class JetsonPowerMonitor(PowerMonitor):
    """High-precision monitor for Jetson devices. Uses tegrastats as the master clock to synchronize SoC power rail readings with pynvml GPU utilization polls.
    
    Parses per-rail power (VDD_GPU_SOC, VDD_CPU_CV, VIN_SYS_5V0) and
    temperature (gpu, cpu, soc0-2) from tegrastats output for granular
    energy attribution and thermal throttling detection.
    
    Bases:
        PowerMonitor
    
    Attributes:
        cpu_cv_power_samples: Cpu cv power samples state retained by each instance.
        cpu_temp_samples: Cpu temp samples state retained by each instance.
        end_ns: Inclusive measurement end timestamp in nanoseconds.
        gpu_mem_free_gb_samples: Gpu mem free gb samples state retained by each instance.
        gpu_mem_used_gb_samples: Gpu mem used gb samples state retained by each instance.
        gpu_mem_util_samples: Gpu mem util samples state retained by each instance.
        gpu_memory_total_gb: Gpu memory total gb state retained by each instance.
        gpu_soc_power_samples: Gpu soc power samples state retained by each instance.
        gpu_temp_samples: Gpu temp samples state retained by each instance.
        gpu_util_samples: Gpu util samples state retained by each instance.
        interval_ms: Telemetry sampling interval in milliseconds.
        last_raw_line: Last raw line state retained by each instance.
        oc_event_counts_start: Oc event counts start state retained by each instance.
        power_samples: Power samples state retained by each instance.
        proc: Proc state retained by each instance.
        running: Running state retained by each instance.
        sample_rows: Sample rows state retained by each instance.
        start_ns: Inclusive measurement start timestamp in nanoseconds.
        thread: Thread state retained by each instance.
        timestamps: Timestamps state retained by each instance.
        vin_sys_power_samples: Vin sys power samples state retained by each instance.
    """

    def __init__(self, interval_ms: int = 50):
        """Initialize JetsonPowerMonitor and capture its validated configuration or runtime state.
        
        Args:
            interval_ms (int): Telemetry sampling interval in milliseconds. Defaults to 50.
        
        Returns:
            None; initializes the instance in place.
        """
        self.interval_ms = interval_ms

        self.proc = None
        self.thread = None
        self.running = False

        self.power_samples = []
        self.timestamps = []
        self.gpu_util_samples = []
        self.gpu_mem_util_samples = []
        self.gpu_mem_used_gb_samples = []
        self.gpu_mem_free_gb_samples = []
        self.gpu_memory_total_gb = float("nan")

        self.gpu_temp_samples = []
        self.cpu_temp_samples = []

        self.gpu_soc_power_samples = []
        self.cpu_cv_power_samples = []
        self.vin_sys_power_samples = []

        self.last_raw_line = ""
        self.sample_rows = []
        self.start_ns = None
        self.end_ns = None
        self.oc_event_counts_start = {}

    @staticmethod
    def _read_oc_event_counts() -> Dict[str, int]:
        """Read cumulative Orin hardware over-current counters by channel.
        
        Returns:
            Dict[str, int]: Named values produced by this operation.
        """
        counts: Dict[str, int] = {}
        root = Path("/sys/devices/platform/soctherm-oc-event/hwmon")
        for path in root.glob("hwmon*/oc*_event_cnt"):
            channel = path.name.removesuffix("_event_cnt")
            try:
                counts[channel] = int(path.read_text().strip())
            except (OSError, ValueError):
                continue
        return counts

    def _parse_power_rails(self, line: str) -> Dict[str, float]:
        """Parse individual power rails from tegrastats output.
        
        Args:
            line (str): Configuration or runtime value named line.
        
        Returns:
            Dict[str, float]: Named values produced by this operation.
        """
        rails = {}
        pattern = re.compile(r"(VDD_GPU_SOC|VDD_CPU_CV|VIN_SYS_5V0)\s+([\d.]+)(mW|W)")
        for m in pattern.finditer(line):
            name = m.group(1)
            value = float(m.group(2))
            unit = m.group(3)
            rails[name] = value / 1000.0 if unit == "mW" else value
        return rails

    def _parse_temperatures(self, line: str) -> Dict[str, float]:
        """Parse tegrastats temperatures with or without a critical value.
        
        Args:
            line (str): Configuration or runtime value named line.
        
        Returns:
            Dict[str, float]: Named values produced by this operation.
        """
        temps = {}
        pattern = re.compile(r"(gpu|cpu|soc0|soc1|soc2|tj)@([\d.]+)C(?:/[\d.]+C)?")
        for m in pattern.finditer(line):
            sensor = m.group(1)
            current_temp = float(m.group(2))
            temps[sensor] = current_temp
        return temps

    @staticmethod
    def _parse_gpu_util(line: str) -> float:
        """Parse gpu util for JetsonPowerMonitor.
        
        Args:
            line (str): Configuration or runtime value named line.
        
        Returns:
            float: Computed scalar described by this callable.
        """
        match = re.search(r"GR3D_FREQ\s+(\d+(?:\.\d+)?)%", line)
        return float(match.group(1)) if match else float("nan")

    @staticmethod
    def _parse_shared_ram(line: str):
        """Return tegrastats shared unified-memory used/total values in GiB.
        
        Args:
            line (str): Configuration or runtime value named line.
        
        Returns:
            Runtime value with no declared return annotation; see the implementation and callers.
        """
        match = re.search(r"\bRAM\s+(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)MB\b", line)
        if not match:
            return None
        used_gb = float(match.group(1)) / 1024.0
        total_gb = float(match.group(2)) / 1024.0
        return used_gb, total_gb

    def _parse_total_power(self, line: str) -> float:
        """Sum all identified SoC power rails for total system power.
        
        Args:
            line (str): Configuration or runtime value named line.
        
        Returns:
            float: Computed scalar described by this callable.
        """
        total_power_w = 0.0
        rails = self._parse_power_rails(line)
        for rail_w in rails.values():
            total_power_w += rail_w
        return total_power_w

    def _monitor_loop(self):
        """Run the background collection loop owned by JetsonPowerMonitor until it is stopped.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        handle = None
        if NVML_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                pass

        while self.running:
            try:
                line = self.proc.stdout.readline()
                if not line:
                    break
                line = line.strip()
                t_now = time.time_ns()
                sample = {
                    "timestamp_ns": int(t_now),
                    "power_w": float("nan"),
                    "gpu_soc_power_w": float("nan"),
                    "cpu_cv_power_w": float("nan"),
                    "system_5v_power_w": float("nan"),
                    "gpu_compute_util_percent": float("nan"),
                    "gpu_memory_util_percent": float("nan"),
                    "gpu_memory_used_gb": float("nan"),
                    "gpu_memory_free_gb": float("nan"),
                    "gpu_memory_total_gb": float("nan"),
                    "gpu_temp_c": float("nan"),
                    "cpu_temp_c": float("nan"),
                }

                self.last_raw_line = line

                rails = self._parse_power_rails(line)
                if rails:
                    power = float(sum(rails.values()))
                    self.power_samples.append(power)
                    self.timestamps.append(t_now)
                    sample["power_w"] = power
                if "VDD_GPU_SOC" in rails:
                    self.gpu_soc_power_samples.append(rails["VDD_GPU_SOC"])
                    sample["gpu_soc_power_w"] = float(rails["VDD_GPU_SOC"])
                if "VDD_CPU_CV" in rails:
                    self.cpu_cv_power_samples.append(rails["VDD_CPU_CV"])
                    sample["cpu_cv_power_w"] = float(rails["VDD_CPU_CV"])
                if "VIN_SYS_5V0" in rails:
                    self.vin_sys_power_samples.append(rails["VIN_SYS_5V0"])
                    sample["system_5v_power_w"] = float(rails["VIN_SYS_5V0"])

                temps = self._parse_temperatures(line)
                if "gpu" in temps:
                    self.gpu_temp_samples.append(temps["gpu"])
                    sample["gpu_temp_c"] = float(temps["gpu"])
                if "cpu" in temps:
                    self.cpu_temp_samples.append(temps["cpu"])
                    sample["cpu_temp_c"] = float(temps["cpu"])

                if handle:
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        self.gpu_util_samples.append(util.gpu)
                        sample["gpu_compute_util_percent"] = float(util.gpu)
                    except Exception:
                        pass
                shared_ram = self._parse_shared_ram(line)
                if shared_ram is not None:
                    used_gb, total_gb = shared_ram
                    free_gb = max(0.0, total_gb - used_gb)
                    mem_util = used_gb / total_gb * 100.0 if total_gb > 0 else float("nan")
                    self.gpu_mem_util_samples.append(mem_util)
                    self.gpu_mem_used_gb_samples.append(used_gb)
                    self.gpu_mem_free_gb_samples.append(free_gb)
                    self.gpu_memory_total_gb = float(total_gb)
                    sample["gpu_memory_util_percent"] = float(mem_util)
                    sample["gpu_memory_used_gb"] = float(used_gb)
                    sample["gpu_memory_free_gb"] = float(free_gb)
                    sample["gpu_memory_total_gb"] = float(total_gb)
                if not np.isfinite(sample["gpu_compute_util_percent"]):
                    gr3d_util = self._parse_gpu_util(line)
                    if np.isfinite(gr3d_util):
                        self.gpu_util_samples.append(gr3d_util)
                        sample["gpu_compute_util_percent"] = gr3d_util

                self.sample_rows.append(sample)
            except Exception:
                break

    def start(self) -> None:
        """Start JetsonPowerMonitor collection or lifecycle state.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        logger.debug(f"[JetsonMonitor] Starting telemetry at {self.interval_ms}ms interval...")
        self.power_samples = []
        self.timestamps = []
        self.gpu_util_samples = []
        self.gpu_mem_util_samples = []
        self.gpu_mem_used_gb_samples = []
        self.gpu_mem_free_gb_samples = []
        self.gpu_memory_total_gb = float("nan")
        self.gpu_temp_samples = []
        self.cpu_temp_samples = []
        self.gpu_soc_power_samples = []
        self.cpu_cv_power_samples = []
        self.vin_sys_power_samples = []
        self.sample_rows = []
        self.start_ns = None
        self.end_ns = None
        self.oc_event_counts_start = self._read_oc_event_counts()
        self.running = True

        self.proc = subprocess.Popen(
            ["sudo", "-n", "tegrastats", "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # tegrastats process startup can exceed one sampling interval. Prime and
        # discard its first line before the benchmark timer starts so short runs
        # still receive samples on the requested interval instead of all-NaN data.
        if self.proc.stdout is not None:
            timeout_sec = max(1.0, (self.interval_ms / 1000.0) * 10)
            ready, _, _ = select.select([self.proc.stdout], [], [], timeout_sec)
            if ready:
                self.proc.stdout.readline()
            else:
                logger.warning(
                    "[JetsonMonitor] tegrastats did not produce a priming sample within %.1fs",
                    timeout_sec,
                )

        self.start_ns = time.time_ns()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self) -> Dict[str, float]:
        """Stop JetsonPowerMonitor collection or lifecycle state and return its final result when applicable.
        
        Returns:
            Dict[str, float]: Named values produced by this operation.
        """
        self.end_ns = time.time_ns()
        logger.debug("[JetsonMonitor] Stopping telemetry collection...")
        self.running = False

        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()

        if self.thread:
            self.thread.join(timeout=2.0)

        metrics = {"power_scope": "sum_vdd_gpu_soc_vdd_cpu_cv_vin_sys_5v0"}

        oc_event_counts_end = self._read_oc_event_counts()
        metrics["jetson_oc_counter_available"] = bool(
            self.oc_event_counts_start and oc_event_counts_end
        )
        total_oc_delta = 0
        for channel in ("oc1", "oc2", "oc3"):
            start_count = self.oc_event_counts_start.get(channel)
            end_count = oc_event_counts_end.get(channel)
            delta = (
                max(0, end_count - start_count)
                if start_count is not None and end_count is not None
                else 0
            )
            metrics[f"jetson_{channel}_event_delta"] = int(delta)
            total_oc_delta += delta
        metrics["jetson_oc_event_delta"] = int(total_oc_delta)

        metrics.update(
            summarize_power_interval(
                self.power_samples, self.timestamps, self.start_ns, self.end_ns
            )
        )

        metrics["avg_gpu_compute_util_percent"] = (
            float(np.mean(self.gpu_util_samples)) if self.gpu_util_samples else float("nan")
        )
        metrics["avg_gpu_memory_util_percent"] = (
            float(np.mean(self.gpu_mem_util_samples)) if self.gpu_mem_util_samples else float("nan")
        )
        metrics["gpu_compute_util_std_percent"] = (
            float(np.std(self.gpu_util_samples)) if self.gpu_util_samples else float("nan")
        )
        metrics["gpu_memory_util_std_percent"] = (
            float(np.std(self.gpu_mem_util_samples)) if self.gpu_mem_util_samples else float("nan")
        )
        metrics["gpu_memory_scope"] = "shared_unified_system_memory"
        metrics["gpu_memory_total_gb"] = self.gpu_memory_total_gb
        metrics["avg_gpu_memory_used_gb"] = (
            float(np.mean(self.gpu_mem_used_gb_samples))
            if self.gpu_mem_used_gb_samples
            else float("nan")
        )
        metrics["peak_gpu_memory_used_gb"] = (
            float(np.max(self.gpu_mem_used_gb_samples))
            if self.gpu_mem_used_gb_samples
            else float("nan")
        )
        metrics["min_gpu_memory_free_gb"] = (
            float(np.min(self.gpu_mem_free_gb_samples))
            if self.gpu_mem_free_gb_samples
            else float("nan")
        )

        metrics["avg_gpu_temp_c"] = (
            float(np.mean(self.gpu_temp_samples)) if self.gpu_temp_samples else float("nan")
        )
        metrics["max_gpu_temp_c"] = (
            float(np.max(self.gpu_temp_samples)) if self.gpu_temp_samples else float("nan")
        )
        metrics["avg_cpu_temp_c"] = (
            float(np.mean(self.cpu_temp_samples)) if self.cpu_temp_samples else float("nan")
        )

        metrics["avg_gpu_soc_power_w"] = (
            float(np.mean(self.gpu_soc_power_samples))
            if self.gpu_soc_power_samples
            else float("nan")
        )
        metrics["peak_gpu_soc_power_w"] = (
            float(np.max(self.gpu_soc_power_samples))
            if self.gpu_soc_power_samples
            else float("nan")
        )
        metrics["avg_cpu_cv_power_w"] = (
            float(np.mean(self.cpu_cv_power_samples)) if self.cpu_cv_power_samples else float("nan")
        )
        metrics["peak_cpu_cv_power_w"] = (
            float(np.max(self.cpu_cv_power_samples)) if self.cpu_cv_power_samples else float("nan")
        )
        metrics["avg_system_5v_power_w"] = (
            float(np.mean(self.vin_sys_power_samples))
            if self.vin_sys_power_samples
            else float("nan")
        )
        metrics["peak_system_5v_power_w"] = (
            float(np.max(self.vin_sys_power_samples))
            if self.vin_sys_power_samples
            else float("nan")
        )
        if self.sample_rows:
            t0 = self.sample_rows[0]["timestamp_ns"]
            metrics["telemetry_samples"] = [
                {
                    "sample_index": int(i),
                    "timestamp_ns": int(sample["timestamp_ns"]),
                    "elapsed_ms": float((sample["timestamp_ns"] - t0) / 1e6),
                    "power_w": sample["power_w"],
                    "gpu_soc_power_w": sample["gpu_soc_power_w"],
                    "cpu_cv_power_w": sample["cpu_cv_power_w"],
                    "system_5v_power_w": sample["system_5v_power_w"],
                    "gpu_compute_util_percent": sample["gpu_compute_util_percent"],
                    "gpu_memory_util_percent": sample["gpu_memory_util_percent"],
                    "gpu_memory_used_gb": sample["gpu_memory_used_gb"],
                    "gpu_memory_free_gb": sample["gpu_memory_free_gb"],
                    "gpu_memory_total_gb": sample["gpu_memory_total_gb"],
                    "gpu_temp_c": sample["gpu_temp_c"],
                    "cpu_temp_c": sample["cpu_temp_c"],
                }
                for i, sample in enumerate(self.sample_rows)
            ]
        else:
            metrics["telemetry_samples"] = []

        return metrics

    def get_raw_line(self) -> str:
        """Return the most recent raw tegrastats line for diagnostics.
        
        Returns:
            str: Result produced by this operation.
        """
        return self.last_raw_line
