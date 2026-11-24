"""Collect and summarize null telemetry around measured inference intervals."""

from typing import Dict
from benchmark.monitors.base import PowerMonitor


class NullPowerMonitor(PowerMonitor):
    """Explicit fallback for unsupported devices. Returns NaN, never zero.
    
    Bases:
        PowerMonitor
    """

    def start(self) -> None:
        """Start NullPowerMonitor collection or lifecycle state.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        pass

    def stop(self) -> Dict[str, float]:
        """Stop NullPowerMonitor collection or lifecycle state and return its final result when applicable.
        
        Returns:
            Dict[str, float]: Named values produced by this operation.
        """
        metrics = {
            k: float("nan")
            for k in [
                "avg_power_w",
                "peak_power_w",
                "total_energy_j",
                "avg_gpu_compute_util_percent",
                "avg_gpu_memory_util_percent",
                "avg_gpu_temp_c",
                "max_gpu_temp_c",
                "avg_cpu_temp_c",
            ]
        }
        metrics["power_scope"] = "unavailable"
        return metrics

    def get_raw_line(self) -> str:
        """Return an empty diagnostic line because this backend collects no telemetry.
        
        Returns:
            str: Result produced by this operation.
        """
        return ""
