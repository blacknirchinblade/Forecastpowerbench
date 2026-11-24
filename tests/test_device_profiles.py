"""Regression-test the device profiles benchmark contract."""

import subprocess
import unittest
from unittest.mock import patch

from benchmark.config import DeviceCategory
from benchmark.devices.jetson import JetsonProfile
from benchmark.devices.nvml import NVMLProfile
from experiments.forecasting import auto_detect_device


class JetsonProfileTests(unittest.TestCase):
    """Define the JetsonProfileTests type and its associated runtime state or contract.
    
    Bases:
        unittest.TestCase
    """
    @patch("benchmark.devices.jetson.subprocess.run")
    @patch.object(JetsonProfile, "_query_power_mode", return_value=("MODE_50W", "3"))
    @patch.object(JetsonProfile, "_log_effective_configuration")
    def test_power_mode_locks_clocks_and_maximizes_fan(self, _log, _query, run):
        """Regression-check that power mode locks clocks and maximizes fan.
        
        Args:
            _log (Any): Configuration or runtime value named _log.
            _query (Any): Configuration or runtime value named _query.
            run (Any): Configuration or runtime value named run.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        run.return_value = subprocess.CompletedProcess(
            [], 0, stdout="FAN Dynamic Speed Control=disabled hwmon2_pwm1=255\n", stderr=""
        )
        profile = JetsonProfile()

        profile.apply_power_mode("3")

        commands = [call.args[0] for call in run.call_args_list]
        self.assertIn(["sudo", "-n", "jetson_clocks"], commands)
        self.assertIn(["sudo", "-n", "jetson_clocks", "--fan"], commands)

    @patch("benchmark.devices.jetson.subprocess.run")
    def test_teardown_removes_root_owned_clock_snapshot_with_sudo(self, run):
        """Regression-check that teardown removes root owned clock snapshot with sudo.
        
        Args:
            run (Any): Configuration or runtime value named run.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        run.return_value = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        profile = JetsonProfile()
        profile.original_power_mode = "0"
        profile.clock_state_stored = True

        profile.teardown()

        commands = [call.args[0] for call in run.call_args_list]
        self.assertIn(
            ["sudo", "-n", "rm", "-f", profile.clock_state_path],
            commands,
        )


class NVMLProfileTests(unittest.TestCase):
    """Verify discrete-GPU identity, selection, and Hopper partition guards.

    Bases:
        unittest.TestCase
    """

    @patch("benchmark.devices.nvml.shutil.which", return_value="/usr/bin/sudo")
    @patch("benchmark.devices.nvml.os.geteuid", return_value=1000)
    @patch("benchmark.devices.nvml.subprocess.run")
    def test_selected_gpu_index_targets_query_and_control_commands(
        self, run, _geteuid, _which
    ):
        """Ensure NVML state queries and controls address the selected card.

        Args:
            run (Any): Mocked subprocess runner used to inspect commands.
            _geteuid (Any): Mocked non-root effective user identifier.
            _which (Any): Mocked sudo executable discovery.

        Returns:
            None; the callable operates through assertions.
        """
        run.return_value = subprocess.CompletedProcess(
            [], 0, stdout="Enabled, 700, 1980\n", stderr=""
        )
        with patch.dict("os.environ", {"BENCHMARK_GPU_INDEX": "3"}):
            profile = NVMLProfile(DeviceCategory.H200)

        profile._query_state()
        profile._sudo_nvidia_smi("-pm", "1", check=False)

        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(commands[0][1:3], ["-i", "3"])
        self.assertEqual(commands[1][3:5], ["-i", "3"])

    @patch("benchmark.devices.nvml.os.geteuid", return_value=0)
    @patch("benchmark.devices.nvml.subprocess.run")
    def test_root_controls_gpu_without_sudo(self, run, _geteuid):
        """Invoke nvidia-smi directly in minimal root cloud containers.

        Args:
            run (Any): Mocked subprocess runner used to inspect commands.
            _geteuid (Any): Mocked root effective user identifier.

        Returns:
            None; the callable operates through assertions.
        """
        run.return_value = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        profile = NVMLProfile(DeviceCategory.H100)

        profile._sudo_nvidia_smi("-pm", "1", check=False)

        self.assertEqual(
            run.call_args.args[0],
            ["nvidia-smi", "-i", "0", "-pm", "1"],
        )

    @patch.object(NVMLProfile, "_sudo_nvidia_smi")
    @patch.object(
        NVMLProfile,
        "_query_state",
        return_value=(True, 700.0, 1980),
    )
    def test_default_mode_skips_unchanged_power_limit(self, _state, control):
        """Avoid a privileged write when baseline power is already effective.

        Args:
            _state (Any): Mocked effective persistence, power, and clock state.
            control (Any): Mocked privileged NVIDIA control command.

        Returns:
            None; the callable operates through assertions.
        """
        profile = NVMLProfile(DeviceCategory.H100)
        profile.original_power_limit_w = 700.0

        profile.apply_power_mode("0")

        control.assert_not_called()
        self.assertTrue(profile.verify_power_mode("0"))

    @patch.object(
        NVMLProfile,
        "_query_identity",
        return_value=("NVIDIA H100 80GB HBM3", 81559.0, "disabled", "GPU-test"),
    )
    @patch.object(
        NVMLProfile,
        "_query_state",
        return_value=(True, 700.0, 1980),
    )
    def test_explicit_dynamic_clock_policy_is_recorded(self, _state, _identity):
        """Record provider-managed clocks without attempting privileged control.

        Args:
            _state (Any): Mocked effective persistence, power, and clock state.
            _identity (Any): Mocked selected-GPU identity query.

        Returns:
            None; the callable operates through assertions.
        """
        with patch.dict("os.environ", {"GPU_LOCK_CLOCKS": "false"}):
            profile = NVMLProfile(DeviceCategory.H100)

        profile.setup()

        self.assertEqual(profile.clock_policy, "dynamic_requested")
        self.assertFalse(profile.clocks_locked_by_us)

    @patch.object(
        NVMLProfile,
        "_query_identity",
        return_value=("NVIDIA H200", 143000.0, "enabled", "GPU-test"),
    )
    def test_h200_rejects_mig_partition(self, _identity):
        """Reject a partitioned H200 before changing hardware state.

        Args:
            _identity (Any): Mocked selected-GPU identity query.

        Returns:
            None; the callable operates through assertions.
        """
        profile = NVMLProfile(DeviceCategory.H200)

        with self.assertRaisesRegex(RuntimeError, "MIG mode is enabled"):
            profile.setup()

    @patch.object(
        NVMLProfile,
        "_query_identity",
        return_value=("NVIDIA H100 80GB HBM3", 81559.0, "enabled", "GPU-test"),
    )
    def test_h100_rejects_mig_partition(self, _identity):
        """Reject a partitioned H100 before changing hardware state.

        Args:
            _identity (Any): Mocked selected-GPU identity query.

        Returns:
            None; the callable operates through assertions.
        """
        profile = NVMLProfile(DeviceCategory.H100)

        with self.assertRaisesRegex(RuntimeError, "MIG mode is enabled"):
            profile.setup()

    @patch("experiments.forecasting.torch.cuda.get_device_name", return_value="NVIDIA H200")
    @patch("experiments.forecasting.torch.cuda.is_available", return_value=True)
    @patch("builtins.open", side_effect=OSError)
    @patch("experiments.forecasting.os.path.exists", return_value=False)
    def test_auto_detection_recognizes_h200(
        self, _exists, _open, _available, _device_name
    ):
        """Recognize an H200 CUDA device as its dedicated category.

        Args:
            _exists (Any): Mocked filesystem signature check.
            _open (Any): Mocked device-tree file access.
            _available (Any): Mocked CUDA availability query.
            _device_name (Any): Mocked CUDA device-name query.

        Returns:
            None; the callable operates through assertions.
        """
        self.assertEqual(auto_detect_device(), DeviceCategory.H200)

    @patch(
        "experiments.forecasting.torch.cuda.get_device_name",
        return_value="NVIDIA H100 80GB HBM3",
    )
    @patch("experiments.forecasting.torch.cuda.is_available", return_value=True)
    @patch("builtins.open", side_effect=OSError)
    @patch("experiments.forecasting.os.path.exists", return_value=False)
    def test_auto_detection_recognizes_h100(
        self, _exists, _open, _available, _device_name
    ):
        """Recognize an H100 CUDA device as its dedicated category.

        Args:
            _exists (Any): Mocked filesystem signature check.
            _open (Any): Mocked device-tree file access.
            _available (Any): Mocked CUDA availability query.
            _device_name (Any): Mocked CUDA device-name query.

        Returns:
            None; the callable operates through assertions.
        """
        self.assertEqual(auto_detect_device(), DeviceCategory.H100)


if __name__ == "__main__":
    unittest.main()
