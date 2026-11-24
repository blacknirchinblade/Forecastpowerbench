"""Implement the jetson hardware profile and its setup, verification, and restoration lifecycle.
"""

import subprocess
import logging
import os
from benchmark.devices.base import DeviceProfile
from benchmark.config import DeviceCategory

logger = logging.getLogger(__name__)


class JetsonProfile(DeviceProfile):
    """Manages power modes and clock locking for NVIDIA Jetson devices. Requires passwordless sudo for nvpmodel and jetson_clocks.
    
    Bases:
        DeviceProfile
    
    Attributes:
        clock_state_path: Clock state path state retained by each instance.
        clock_state_stored: Clock state stored state retained by each instance.
        original_power_mode: Original power mode state retained by each instance.
        power_mode_id: Power mode id state retained by each instance.
    """

    def __init__(self, power_mode_id: str = "0"):
        """Initialize JetsonProfile and capture its validated configuration or runtime state.
        
        Args:
            power_mode_id (str): Configuration or runtime value named power_mode_id. Defaults to '0'.
        
        Returns:
            None; initializes the instance in place.
        """
        super().__init__(DeviceCategory.JETSON)
        self.power_mode_id = power_mode_id
        self.original_power_mode = None
        self.clock_state_path = f"/tmp/powerbench_jetson_clocks_{os.getpid()}.conf"
        self.clock_state_stored = False

    @staticmethod
    def _capture(command):
        """Capture capture for JetsonProfile.
        
        Args:
            command (Any): Command and arguments executed without a shell.
        
        Returns:
            Runtime value with no declared return annotation; see the implementation and callers.
        """
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        return result.returncode, (result.stdout + result.stderr).strip()

    def _query_power_mode(self):
        """Query power mode for JetsonProfile.
        
        Returns:
            Runtime value with no declared return annotation; see the implementation and callers.
        """
        _returncode, output = self._capture(["sudo", "-n", "nvpmodel", "-q"])
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        mode_name = lines[0].removeprefix("NV Power Mode:").strip() if lines else "unknown"
        mode_id = lines[1] if len(lines) > 1 and lines[1].isdigit() else None
        return mode_name, mode_id

    def _log_effective_configuration(self, clocks_locked: bool) -> None:
        """Log effective configuration for JetsonProfile.
        
        Args:
            clocks_locked (bool): Whether the device clocks were successfully locked.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        mode_name, mode_id = self._query_power_mode()
        logger.info(
            "[JetsonProfile] Effective nvpmodel mode: %s (id=%s)", mode_name, mode_id or "unknown"
        )
        if mode_name.upper() == "MAXN":
            if clocks_locked:
                logger.info(
                    "[JetsonProfile] MAXN selected; nvpmodel exposes the platform's maximum mode limits, "
                    "and jetson_clocks has pinned CPU, GPU, and EMC clocks for deterministic inference."
                )
            else:
                logger.info(
                    "[JetsonProfile] MAXN selected with dynamic CPU/GPU clock scaling restored."
                )

        _returncode, output = self._capture(["sudo", "-n", "jetson_clocks", "--show"])
        prefixes = (
            "SOC family:",
            "Online CPUs:",
            "cpu0:",
            "GPU MinFreq=",
            "Active GPU TPCs:",
            "EMC MinFreq=",
            "DLA0_CORE:",
            "DLA1_CORE:",
            "PVA0_VPS0:",
            "FAN ",
            "NV Power Mode:",
        )
        for line in output.splitlines():
            line = line.strip()
            if line.startswith(prefixes):
                logger.info("[JetsonProfile] %s", line)

    def _apply_and_lock(self, mode_id: str) -> None:
        """Apply and lock for JetsonProfile.
        
        Args:
            mode_id (str): Device-specific power-mode identifier to apply or verify.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        _mode_name, current_mode_id = self._query_power_mode()
        if current_mode_id == str(mode_id):
            logger.info("[JetsonProfile] Requested nvpmodel mode %s is already active.", mode_id)
        else:
            logger.info("[JetsonProfile] Applying nvpmodel mode %s...", mode_id)
            result = subprocess.run(
                ["sudo", "-n", "nvpmodel", "-m", str(mode_id)],
                input="no\n",
                capture_output=True,
                text=True,
                check=False,
            )
            output = (result.stdout + result.stderr).strip()
            if result.returncode != 0:
                if "reboot required" in output.lower():
                    raise RuntimeError(
                        f"Jetson mode {mode_id} requires a reboot. Run "
                        f"'sudo nvpmodel --force -m {mode_id}', allow the board to reboot, "
                        f"then launch this benchmark with POWER_MODES={mode_id}."
                    )
                raise RuntimeError(
                    f"Failed to apply Jetson power mode {mode_id} "
                    f"(exit {result.returncode}): {output}"
                )
        # nvpmodel can reset clock constraints, so lock clocks after every mode change.
        subprocess.run(["sudo", "-n", "jetson_clocks"], check=True)
        # Keep thermals deterministic and avoid hot-surface throttling during
        # sustained inference. Some L4T releases print a write warning for a
        # secondary fan node even though the primary PWM is set correctly, so
        # capture the command and verify the effective state instead.
        fan_result = subprocess.run(
            ["sudo", "-n", "jetson_clocks", "--fan"],
            capture_output=True,
            text=True,
            check=False,
        )
        show_returncode, show_output = self._capture(["sudo", "-n", "jetson_clocks", "--show"])
        fan_is_max = any(
            line.startswith("FAN ") and "pwm1=255" in line for line in show_output.splitlines()
        )
        if fan_result.returncode != 0 or show_returncode != 0 or not fan_is_max:
            output = (fan_result.stdout + fan_result.stderr).strip()
            raise RuntimeError(
                "Failed to verify maximum Jetson fan speed after jetson_clocks --fan: "
                f"{output or show_output or 'no diagnostic output'}"
            )
        self.power_mode_id = mode_id
        self._log_effective_configuration(clocks_locked=True)

    def setup(self) -> None:
        """Prepare JetsonProfile hardware or runtime state before benchmark execution.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        mode_name, self.original_power_mode = self._query_power_mode()
        stored = subprocess.run(
            ["sudo", "-n", "jetson_clocks", "--store", self.clock_state_path],
            capture_output=True,
            text=True,
            check=False,
        )
        self.clock_state_stored = stored.returncode == 0
        if not self.clock_state_stored:
            logger.warning(
                "[JetsonProfile] Could not snapshot the original clock state: %s",
                (stored.stdout + stored.stderr).strip(),
            )
        logger.info(
            "[JetsonProfile] Original power mode: %s (id=%s)",
            mode_name,
            self.original_power_mode or "unknown",
        )
        logger.info(
            "[JetsonProfile] Original state captured; each benchmark power mode will be applied and verified before sampling."
        )

    def apply_power_mode(self, mode_id: str) -> None:
        """Apply power mode for JetsonProfile.
        
        Args:
            mode_id (str): Device-specific power-mode identifier to apply or verify.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        self._apply_and_lock(mode_id)

    def verify_power_mode(self, mode_id: str) -> bool:
        """Verify power mode for JetsonProfile.
        
        Args:
            mode_id (str): Device-specific power-mode identifier to apply or verify.
        
        Returns:
            bool: Whether the condition described by this callable holds.
        """
        mode_name, effective_mode_id = self._query_power_mode()
        verified = effective_mode_id == str(mode_id)
        logger.info(
            "[POWER MODE VERIFY] requested=%s effective_name=%s effective_id=%s result=%s",
            mode_id,
            mode_name,
            effective_mode_id or "unavailable",
            "PASS" if verified else "FAIL",
        )
        return verified

    def teardown(self) -> None:
        """Restore host state changed by JetsonProfile after benchmark execution.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        logger.info("[JetsonProfile] Restoring pre-benchmark power mode and clock scaling...")
        if self.original_power_mode is not None:
            subprocess.run(
                ["sudo", "-n", "nvpmodel", "-m", self.original_power_mode],
                check=False,
            )
        clocks_restored = False
        if self.clock_state_stored:
            restored = subprocess.run(
                ["sudo", "-n", "jetson_clocks", "--restore", self.clock_state_path],
                capture_output=True,
                text=True,
                check=False,
            )
            clocks_restored = restored.returncode == 0
            if not clocks_restored:
                logger.warning(
                    "[JetsonProfile] Could not restore the original clock state: %s",
                    (restored.stdout + restored.stderr).strip(),
                )
        removed = subprocess.run(
            ["sudo", "-n", "rm", "-f", self.clock_state_path],
            capture_output=True,
            text=True,
            check=False,
        )
        if removed.returncode != 0:
            logger.warning(
                "[JetsonProfile] Could not remove clock-state file: %s",
                (removed.stdout + removed.stderr).strip(),
            )
        self._log_effective_configuration(clocks_locked=not clocks_restored)
