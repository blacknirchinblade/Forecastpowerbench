"""Implement the cpu hardware profile and its setup, verification, and restoration lifecycle.
"""

import glob
import logging
import os
from benchmark.devices.base import DeviceProfile
from benchmark.config import DeviceCategory

logger = logging.getLogger(__name__)


class CPUProfile(DeviceProfile):
    """Locks CPU frequency governor to 'performance' to prevent dynamic scaling.
    
    Bases:
        DeviceProfile
    
    Attributes:
        original_governors: Original governors state retained by each instance.
    """

    def __init__(self, category: DeviceCategory = DeviceCategory.CPU):
        """Initialize CPUProfile and capture its validated configuration or runtime state.
        
        Args:
            category (DeviceCategory): Hardware category represented by this device profile. Defaults to
                                       DeviceCategory.CPU.
        
        Returns:
            None; initializes the instance in place.
        """
        super().__init__(category)
        self.original_governors = {}

    def setup(self) -> None:
        """Prepare CPUProfile hardware or runtime state before benchmark execution.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        logger.info("[CPUProfile] Setting CPU governor to 'performance'...")
        cpu_paths = glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")
        failed_paths = []
        for path in cpu_paths:
            try:
                with open(path, "r") as f:
                    original_governor = f.read().strip()
                with open(path, "w") as f:
                    f.write("performance")
                self.original_governors[path] = original_governor
            except Exception as e:
                failed_paths.append((path, e))
        if failed_paths:
            logger.warning(
                "[CPUProfile] Could not set governor for %d/%d CPU(s). "
                "Continuing with the current governor; run with sudo if locked CPU frequency is required. First error: %s: %s",
                len(failed_paths),
                len(cpu_paths),
                failed_paths[0][0],
                failed_paths[0][1],
            )
        if self.original_governors and len(failed_paths) < len(cpu_paths):
            logger.info("[CPUProfile] CPU governor set to 'performance' where permitted.")
        else:
            logger.info("[CPUProfile] CPU governor unchanged.")
        self._log_effective_configuration()

    @staticmethod
    def _read_values(pattern):
        """Read values for CPUProfile.
        
        Args:
            pattern (Any): Recursive source-file glob pattern.
        
        Returns:
            Runtime value with no declared return annotation; see the implementation and callers.
        """
        values = []
        for path in glob.glob(pattern):
            try:
                with open(path, "r") as f:
                    values.append(f.read().strip())
            except OSError:
                continue
        return values

    def _log_effective_configuration(self) -> None:
        """Log effective configuration for CPUProfile.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        governors = sorted(
            set(self._read_values("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"))
        )
        min_freqs = [
            int(value)
            for value in self._read_values("/sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq")
            if value.isdigit()
        ]
        max_freqs = [
            int(value)
            for value in self._read_values("/sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq")
            if value.isdigit()
        ]
        current_freqs = [
            int(value)
            for value in self._read_values("/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq")
            if value.isdigit()
        ]
        logger.info(
            "[CPUProfile] Effective configuration: logical_cpus=%d governors=%s min_mhz=%s max_mhz=%s current_mhz_range=%s",
            os.cpu_count() or 0,
            ",".join(governors) or "unknown",
            f"{min(min_freqs) / 1000:.0f}" if min_freqs else "unknown",
            f"{max(max_freqs) / 1000:.0f}" if max_freqs else "unknown",
            (
                f"{min(current_freqs) / 1000:.0f}-{max(current_freqs) / 1000:.0f}"
                if current_freqs
                else "unknown"
            ),
        )

    def apply_power_mode(self, mode_id: str) -> None:
        """Apply power mode for CPUProfile.
        
        Args:
            mode_id (str): Device-specific power-mode identifier to apply or verify.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        logger.info(f"[CPUProfile] Power mode {mode_id} not applicable for CPU profile.")

    def supports_power_mode_switching(self) -> bool:
        """Return whether supports power mode switching for CPUProfile.
        
        Returns:
            bool: Whether the condition described by this callable holds.
        """
        return False

    def teardown(self) -> None:
        """Restore host state changed by CPUProfile after benchmark execution.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        logger.info("[CPUProfile] Restoring original CPU governors...")
        failed_paths = []
        for path, gov in self.original_governors.items():
            try:
                with open(path, "w") as f:
                    f.write(gov)
            except Exception as e:
                failed_paths.append((path, e))
        if failed_paths:
            logger.warning(
                "[CPUProfile] Could not restore governor for %d CPU(s). First error: %s: %s",
                len(failed_paths),
                failed_paths[0][0],
                failed_paths[0][1],
            )
