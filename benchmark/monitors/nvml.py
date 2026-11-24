"""Collect and summarize nvml telemetry around measured inference intervals."""

import threading
import time
import logging
import numpy as np
from typing import Dict
from benchmark.monitors.base import PowerMonitor
from benchmark.monitors.stats import summarize_power_interval

logger = logging.getLogger(__name__)

try:
    import pynvml

    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False


class NVMLPowerMonitor(PowerMonitor):
    """High-precision monitor for discrete NVIDIA GPUs.

    The physical NVML index must match the GPU selected by the launcher before
    CUDA remaps that device to logical index zero.
    
    Bases:
        PowerMonitor
    
    Attributes:
        end_ns: Inclusive measurement end timestamp in nanoseconds.
        gpu_mem_free_gb_samples: Gpu mem free gb samples state retained by each instance.
        gpu_mem_used_gb_samples: Gpu mem used gb samples state retained by each instance.
        gpu_mem_util_samples: Gpu mem util samples state retained by each instance.
        gpu_memory_total_gb: Gpu memory total gb state retained by each instance.
        gpu_index: Physical NVML device index selected for this benchmark process.
        gpu_temp_samples: Gpu temp samples state retained by each instance.
        gpu_util_samples: Gpu util samples state retained by each instance.
        interval_ms: Telemetry sampling interval in milliseconds.
        power_samples: Power samples state retained by each instance.
        running: Running state retained by each instance.
        sample_rows: Sample rows state retained by each instance.
        start_ns: Inclusive measurement start timestamp in nanoseconds.
        thread: Thread state retained by each instance.
        timestamps: Timestamps state retained by each instance.
    """

    def __init__(self, interval_ms: int = 50, gpu_index: int = 0):
        """Initialize NVMLPowerMonitor and capture its validated configuration or runtime state.
        
        Args:
            interval_ms (int): Telemetry sampling interval in milliseconds. Defaults to 50.
            gpu_index (int): Physical NVML device index monitored by this process. Defaults to 0.
        
        Returns:
            None; initializes the instance in place.
        """
        self.interval_ms = interval_ms
        if gpu_index < 0:
            raise ValueError("gpu_index must be non-negative")
        self.gpu_index = gpu_index
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
        self.sample_rows = []
        self.start_ns = None
        self.end_ns = None

    def _monitor_loop(self):
        """Run the background collection loop owned by NVMLPowerMonitor until it is stopped.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        next_time = time.perf_counter()

        while self.running:
            t_now = time.time_ns()
            sample = {
                "timestamp_ns": int(t_now),
                "power_w": float("nan"),
                "gpu_compute_util_percent": float("nan"),
                "gpu_memory_util_percent": float("nan"),
                "gpu_memory_used_gb": float("nan"),
                "gpu_memory_free_gb": float("nan"),
                "gpu_memory_total_gb": float("nan"),
                "gpu_temp_c": float("nan"),
                "cpu_temp_c": float("nan"),
                "raw_line": "",
            }

            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_w = power_mw / 1000.0
                self.power_samples.append(power_w)
                self.timestamps.append(t_now)
                sample["power_w"] = float(power_w)

                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.gpu_util_samples.append(util.gpu)
                sample["gpu_compute_util_percent"] = float(util.gpu)

                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_util = (mem_info.used / mem_info.total) * 100.0
                used_gb = mem_info.used / (1024**3)
                free_gb = mem_info.free / (1024**3)
                total_gb = mem_info.total / (1024**3)
                self.gpu_mem_util_samples.append(mem_util)
                self.gpu_mem_used_gb_samples.append(used_gb)
                self.gpu_mem_free_gb_samples.append(free_gb)
                self.gpu_memory_total_gb = float(total_gb)
                sample["gpu_memory_util_percent"] = float(mem_util)
                sample["gpu_memory_used_gb"] = float(used_gb)
                sample["gpu_memory_free_gb"] = float(free_gb)
                sample["gpu_memory_total_gb"] = float(total_gb)

                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    self.gpu_temp_samples.append(temp)
                    sample["gpu_temp_c"] = float(temp)
                except Exception:
                    pass
            except Exception as e:
                logger.debug(f"NVML poll failed: {e}")

            self.sample_rows.append(sample)

            next_time += self.interval_ms / 1000.0
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.perf_counter()

    def start(self) -> None:
        """Start NVMLPowerMonitor collection or lifecycle state.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        if not NVML_AVAILABLE:
            logger.error("[NVMLMonitor] pynvml is not available.")
            return

        logger.debug(f"[NVMLMonitor] Starting telemetry at {self.interval_ms}ms interval...")
        self.power_samples = []
        self.timestamps = []
        self.gpu_util_samples = []
        self.gpu_mem_util_samples = []
        self.gpu_mem_used_gb_samples = []
        self.gpu_mem_free_gb_samples = []
        self.gpu_memory_total_gb = float("nan")
        self.gpu_temp_samples = []
        self.sample_rows = []
        self.start_ns = time.time_ns()
        self.end_ns = None
        self.running = True

        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self) -> Dict[str, float]:
        """Stop NVMLPowerMonitor collection or lifecycle state and return its final result when applicable.
        
        Returns:
            Dict[str, float]: Named values produced by this operation.
        """
        self.end_ns = time.time_ns()
        logger.debug("[NVMLMonitor] Stopping telemetry collection...")
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

        metrics = {"power_scope": "gpu_board"}

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
        metrics["gpu_memory_scope"] = "dedicated_vram"
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
        metrics["avg_cpu_temp_c"] = float("nan")
        if self.sample_rows:
            t0 = self.sample_rows[0]["timestamp_ns"]
            metrics["telemetry_samples"] = [
                {
                    "sample_index": int(i),
                    "timestamp_ns": int(sample["timestamp_ns"]),
                    "elapsed_ms": float((sample["timestamp_ns"] - t0) / 1e6),
                    "power_w": sample["power_w"],
                    "gpu_compute_util_percent": sample["gpu_compute_util_percent"],
                    "gpu_memory_util_percent": sample["gpu_memory_util_percent"],
                    "gpu_memory_used_gb": sample["gpu_memory_used_gb"],
                    "gpu_memory_free_gb": sample["gpu_memory_free_gb"],
                    "gpu_memory_total_gb": sample["gpu_memory_total_gb"],
                    "gpu_temp_c": sample["gpu_temp_c"],
                    "cpu_temp_c": sample["cpu_temp_c"],
                    "raw_line": sample["raw_line"],
                }
                for i, sample in enumerate(self.sample_rows)
            ]
        else:
            metrics["telemetry_samples"] = []

        return metrics

    def get_raw_line(self) -> str:
        """Return a concise representation of the most recent NVML telemetry sample.
        
        Returns:
            str: Result produced by this operation.
        """
        return ""
