"""Collect and summarize base telemetry around measured inference intervals."""

import time
import logging
from abc import ABC, abstractmethod
from typing import Dict

logger = logging.getLogger(__name__)


class PowerMonitor(ABC):
    """Abstract strategy interface for hardware telemetry collection. Implementations must handle high-frequency sampling and energy integration.
    
    Bases:
        ABC
    """

    @abstractmethod
    def start(self) -> None:
        """Begin telemetry collection. Must be called immediately before the inference forward pass.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        pass

    @abstractmethod
    def stop(self) -> Dict[str, float]:
        """Stop PowerMonitor collection or lifecycle state and return its final result when applicable.
        
        Returns:
            Dict[str, float]: Named values produced by this operation.
        """
        pass

    @abstractmethod
    def get_raw_line(self) -> str:
        """Returns the most recent raw telemetry string (e.g., raw tegrastats line). Returns an empty string if not applicable.
        
        Returns:
            str: Result produced by this operation.
        """
        pass

    def measure_idle_baseline(self, duration_sec: float = 15.0) -> Dict[str, float]:
        """Measure idle power consumption with no inference workload running. Returns avg_power_w (used as baseline for subtraction) and temperature readings.
        
        This is CRITICAL for research accuracy: on edge devices like Jetson,
        idle draw can be 30-50% of peak power. Without subtraction, energy-per-window
        numbers are systematically inflated.
        
        Args:
            duration_sec (float): Requested duration in seconds. Defaults to 15.0.
        
        Returns:
            Dict[str, float]: Named values produced by this operation.
        """
        logger.debug(f"[PowerMonitor] Measuring idle baseline for {duration_sec}s (no workload)...")
        self.start()
        time.sleep(duration_sec)
        metrics = self.stop()
        baseline_w = metrics.get("avg_power_w", float("nan"))
        logger.debug(f"[PowerMonitor] Idle baseline: {baseline_w:.2f}W")
        return metrics
