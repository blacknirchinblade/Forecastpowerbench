"""Implement the nvml hardware profile and its setup, verification, and restoration lifecycle.
"""

import logging
import os
import shutil
import subprocess
from benchmark.devices.base import DeviceProfile
from benchmark.config import DeviceCategory

logger = logging.getLogger(__name__)


class NVMLProfile(DeviceProfile):
    """Manage persistence, clocks, and power for one discrete NVIDIA GPU.
    
    Bases:
        DeviceProfile
    
    Attributes:
        auto_lock_max: Auto lock max state retained by each instance.
        clocks_locked_by_us: Clocks locked by us state retained by each instance.
        clock_policy: Effective clock-control policy recorded for reproducibility.
        gpu_index: Physical NVIDIA device index controlled by this profile.
        max_clock: Max clock state retained by each instance.
        min_clock: Min clock state retained by each instance.
        original_persistence_enabled: Original persistence enabled state retained by each instance.
        original_power_limit_w: Original power limit w state retained by each instance.
        persistence_changed: Persistence changed state retained by each instance.
        power_limit_changed: Power limit changed state retained by each instance.
        selected_gpu_memory_mib: Reported memory capacity of the selected GPU.
        selected_gpu_mig_mode: Reported MIG mode of the selected GPU.
        selected_gpu_name: Reported product name of the selected GPU.
        selected_gpu_uuid: Stable UUID of the selected GPU.
    """

    def __init__(
        self, category: DeviceCategory, min_clock_mhz: int = None, max_clock_mhz: int = None
    ):
        """Initialize NVMLProfile and capture its validated configuration or runtime state.
        
        Args:
            category (DeviceCategory): Hardware category represented by this device profile.
            min_clock_mhz (int): Configuration or runtime value named min_clock_mhz. Defaults to None.
            max_clock_mhz (int): Configuration or runtime value named max_clock_mhz. Defaults to None.
        
        Returns:
            None; initializes the instance in place.
        """
        super().__init__(category)
        self.gpu_index = int(os.environ.get("BENCHMARK_GPU_INDEX", "0"))
        if self.gpu_index < 0:
            raise ValueError("BENCHMARK_GPU_INDEX must be non-negative")
        self.min_clock = min_clock_mhz
        self.max_clock = max_clock_mhz
        self.auto_lock_max = os.environ.get("GPU_LOCK_CLOCKS", "true").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.original_persistence_enabled = False
        self.original_power_limit_w = None
        self.persistence_changed = False
        self.power_limit_changed = False
        self.clocks_locked_by_us = False
        self.clock_policy = "uninitialized"
        self.selected_gpu_name = ""
        self.selected_gpu_memory_mib = None
        self.selected_gpu_mig_mode = "unavailable"
        self.selected_gpu_uuid = ""

    def _query_state(self):
        """Query state for NVMLProfile.
        
        Returns:
            Runtime value with no declared return annotation; see the implementation and callers.
        """
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(self.gpu_index),
                "--query-gpu=persistence_mode,power.limit,clocks.max.graphics",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return False, None, None
        fields = [field.strip() for field in result.stdout.splitlines()[0].split(",")]
        try:
            power_limit = float(fields[1])
        except (IndexError, ValueError):
            power_limit = None
        try:
            max_clock = int(float(fields[2]))
        except (IndexError, ValueError):
            max_clock = None
        return fields[0].lower() == "enabled", power_limit, max_clock

    def _query_identity(self):
        """Return the selected GPU identity, memory size, and MIG mode.

        Returns:
            tuple[str, float | None, str, str]: GPU name, memory in MiB,
            normalized MIG mode, and UUID.
        """
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(self.gpu_index),
                "--query-gpu=name,memory.total,mig.mode.current,uuid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return "", None, "unavailable", ""
        fields = [field.strip() for field in result.stdout.splitlines()[0].split(",")]
        try:
            memory_mib = float(fields[1])
        except (IndexError, ValueError):
            memory_mib = None
        return (
            fields[0] if fields else "",
            memory_mib,
            fields[2].lower() if len(fields) > 2 else "unavailable",
            fields[3] if len(fields) > 3 else "",
        )

    def _sudo_nvidia_smi(
        self, *args: str, check: bool = True, capture_output: bool = True
    ) -> subprocess.CompletedProcess:
        """Run selected-GPU controls directly as root or through noninteractive sudo.
        
        Args:
            args (str): Additional positional arguments forwarded by the callable. Defaults to empty.
            check (bool): Configuration or runtime value named check. Defaults to True.
            capture_output (bool): Whether subprocess standard output and error are captured. Defaults
                                   to True.
        
        Returns:
            subprocess.CompletedProcess: Result produced by this operation.
        """
        privilege_prefix = []
        if os.geteuid() != 0 and shutil.which("sudo") is not None:
            privilege_prefix = ["sudo", "-n"]
        return subprocess.run(
            [
                *privilege_prefix,
                "nvidia-smi",
                "-i",
                str(self.gpu_index),
                *args,
            ],
            check=check,
            capture_output=capture_output,
            text=True,
        )

    def setup(self) -> None:
        """Prepare NVMLProfile hardware or runtime state before benchmark execution.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        gpu_name, memory_mib, mig_mode, gpu_uuid = self._query_identity()
        self.selected_gpu_name = gpu_name
        self.selected_gpu_memory_mib = memory_mib
        self.selected_gpu_mig_mode = mig_mode
        self.selected_gpu_uuid = gpu_uuid
        logger.info(
            "[NVMLProfile] Selected physical GPU index %d: name=%s uuid=%s "
            "memory=%s MiB mig=%s",
            self.gpu_index,
            gpu_name or "unavailable",
            gpu_uuid or "unavailable",
            f"{memory_mib:.0f}" if memory_mib is not None else "unavailable",
            mig_mode,
        )
        hopper_requirements = {
            DeviceCategory.H100: ("H100", 70),
            DeviceCategory.H200: ("H200", 120),
        }
        if self.category in hopper_requirements:
            expected_name, minimum_memory_gib = hopper_requirements[self.category]
            if expected_name not in gpu_name.upper():
                raise RuntimeError(
                    f"Device category {self.category.value} requires an NVIDIA "
                    f"{expected_name}, found {gpu_name!r} "
                    f"at physical index {self.gpu_index}."
                )
            if memory_mib is None or memory_mib < minimum_memory_gib * 1024:
                raise RuntimeError(
                    f"{expected_name} benchmarking requires a full GPU with at least "
                    f"{minimum_memory_gib} GiB "
                    f"visible memory; index {self.gpu_index} reports {memory_mib!r} MiB. "
                    "Do not use a MIG slice for cross-device results."
                )
            if mig_mode == "enabled":
                raise RuntimeError(
                    f"{expected_name} MIG mode is enabled. Use a full non-MIG "
                    f"{expected_name} before benchmarking."
                )

        (
            self.original_persistence_enabled,
            self.original_power_limit_w,
            detected_max_clock,
        ) = self._query_state()
        logger.info(f"[NVMLProfile] Enabling persistence mode for {self.category.value}...")
        if not self.original_persistence_enabled:
            try:
                self._sudo_nvidia_smi("-pm", "1")
                self.persistence_changed = True
            except subprocess.CalledProcessError as e:
                logger.warning(
                    "[NVMLProfile] Could not enable persistence mode; run the launcher with sudo keepalive or configure passwordless sudo. Continuing with default GPU state: %s",
                    (e.stderr or str(e)).strip(),
                )

        if self.auto_lock_max and not (self.min_clock and self.max_clock) and detected_max_clock:
            self.min_clock = detected_max_clock
            self.max_clock = detected_max_clock
        if self.min_clock and self.max_clock:
            logger.info(
                f"[NVMLProfile] Locking GPU clocks to {self.min_clock}-{self.max_clock} MHz..."
            )
            try:
                self._sudo_nvidia_smi("-lgc", f"{self.min_clock},{self.max_clock}")
                self.clocks_locked_by_us = True
                self.clock_policy = (
                    f"locked_graphics_{self.min_clock}_{self.max_clock}_mhz"
                )
            except subprocess.CalledProcessError as e:
                self.clock_policy = "dynamic_control_denied"
                logger.warning(
                    "[NVMLProfile] GPU clock control was denied by the host. "
                    "Continuing with dynamic clocks; record this policy in the "
                    "experiment: %s",
                    (e.stderr or str(e)).strip(),
                )
        elif not self.auto_lock_max:
            self.clock_policy = "dynamic_requested"
            logger.info(
                "[NVMLProfile] GPU_LOCK_CLOCKS=false; using provider-managed dynamic clocks."
            )
        else:
            self.clock_policy = "dynamic_max_clock_unavailable"
            logger.warning(
                "[NVMLProfile] No clock limits provided. GPU clocks will dynamically scale. Variance expected!"
            )

    def apply_power_mode(self, mode_id: str) -> None:
        """Apply power mode for NVMLProfile.
        
        Args:
            mode_id (str): Device-specific power-mode identifier to apply or verify.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        if mode_id == "0" or mode_id == "default":
            if self.original_power_limit_w is None:
                logger.info(
                    "[NVMLProfile] Default power limit could not be queried; leaving it unchanged."
                )
                return
            _persistence, effective_limit_w, _max_clock = self._query_state()
            if (
                effective_limit_w is not None
                and abs(effective_limit_w - self.original_power_limit_w) <= 0.5
            ):
                logger.info(
                    "[NVMLProfile] Baseline power limit already active at %.1fW; "
                    "no control write required.",
                    effective_limit_w,
                )
                return
            logger.info(
                "[NVMLProfile] Restoring baseline power limit %.1fW for mode %s.",
                self.original_power_limit_w,
                mode_id,
            )
            try:
                self._sudo_nvidia_smi("-pl", f"{self.original_power_limit_w:.1f}")
            except subprocess.CalledProcessError as e:
                logger.warning("[NVMLProfile] Failed to restore baseline power limit: %s", e)
            return
        logger.info(f"[NVMLProfile] Setting power limit to {mode_id}W...")
        try:
            self._sudo_nvidia_smi("-pl", mode_id, capture_output=False)
            self.power_limit_changed = True
        except subprocess.CalledProcessError as e:
            logger.warning(f"[NVMLProfile] Failed to set power limit to {mode_id}W: {e}")

    def verify_power_mode(self, mode_id: str) -> bool:
        """Verify power mode for NVMLProfile.
        
        Args:
            mode_id (str): Device-specific power-mode identifier to apply or verify.
        
        Returns:
            bool: Whether the condition described by this callable holds.
        """
        _persistence, effective_limit_w, _max_clock = self._query_state()
        expected_limit_w = self.original_power_limit_w if mode_id in {"0", "default"} else None
        if expected_limit_w is None:
            try:
                expected_limit_w = float(mode_id)
            except ValueError:
                logger.error("[NVMLProfile] Cannot verify non-numeric power mode %r", mode_id)
                return False
        verified = (
            effective_limit_w is not None and abs(effective_limit_w - expected_limit_w) <= 0.5
        )
        logger.info(
            "[POWER MODE VERIFY] requested=%s expected=%.1fW effective=%s result=%s",
            mode_id,
            expected_limit_w,
            f"{effective_limit_w:.1f}W" if effective_limit_w is not None else "unavailable",
            "PASS" if verified else "FAIL",
        )
        return verified

    def teardown(self) -> None:
        """Restore host state changed by NVMLProfile after benchmark execution.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        logger.info("[NVMLProfile] Restoring pre-benchmark GPU state...")
        if self.power_limit_changed and self.original_power_limit_w is not None:
            self._sudo_nvidia_smi(
                "-pl", f"{self.original_power_limit_w:.1f}", check=False, capture_output=True
            )
        if self.clocks_locked_by_us:
            self._sudo_nvidia_smi("-rgc", check=False, capture_output=True)
        if self.persistence_changed:
            self._sudo_nvidia_smi("-pm", "0", check=False, capture_output=True)
