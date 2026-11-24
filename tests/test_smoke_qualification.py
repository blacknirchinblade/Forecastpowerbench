"""Integration-test the all-model smoke qualification shell contract."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class SmokeQualificationTests(unittest.TestCase):
    """Verify the release-gate wrapper independently of expensive model inference.

    Bases:
        unittest.TestCase
    """

    def test_all_model_two_phase_contract_and_artifact_gate(self):
        """Exercise smoke defaults, launcher invocation, and final artifact checks.

        Returns:
            None; the callable operates through assertions.
        """
        source_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            shutil.copy2(source_root / "smoke_test.sh", root / "smoke_test.sh")
            launcher = root / "run_unified_benchmark.sh"
            launcher.write_text(
                """#!/bin/bash
set -euo pipefail
test "$1" = "--phase"
test "$2" = "all"
mkdir -p "$OUTPUT_DIR/benchmark_metadata" "$OUTPUT_DIR/model_logs"
printf '%s\\n' \
  "models=$MODEL_LIST" \
  "batches=$BATCH_SIZES" \
  "warmups=$WARMUPS" \
  "repeats=$REPEATS" \
  "power_switching=$POWER_MODE_SWITCHING" \
  "gpu_lock=$GPU_LOCK_CLOCKS" \
  "auto_resume=$AUTO_RESUME" > "$OUTPUT_DIR/captured.env"
printf 'summary\\nrow\\n' > "$OUTPUT_DIR/benchmark_results_batch_summary.csv"
printf 'source\\nrow\\n' > "$OUTPUT_DIR/dataset_preprocessing_report.csv"
printf 'hash  source\\n' > "$OUTPUT_DIR/benchmark_metadata/source_files.sha256"
for model in $MODEL_LIST; do
  mkdir -p "$OUTPUT_DIR/benchmark_predictions/$model/source"
  printf 'npz' > "$OUTPUT_DIR/benchmark_predictions/$model/source/prediction.npz"
  printf 'prediction log\\n' > "$OUTPUT_DIR/model_logs/${model}_prediction.log"
  printf 'benchmark log\\n' > "$OUTPUT_DIR/model_logs/${model}_benchmark.log"
done
""",
                encoding="utf-8",
            )
            launcher.chmod(0o755)
            output_dir = root / "qualification"
            environment = os.environ.copy()
            environment.update(
                {
                    "OUTPUT_DIR": str(output_dir),
                    "DATA_DIR": str(root / "dataset"),
                    "DEVICE_CONFIG_FILE": str(root / "missing-device.env"),
                }
            )

            completed = subprocess.run(
                ["bash", str(root / "smoke_test.sh")],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            captured = (output_dir / "captured.env").read_text(encoding="utf-8")
            models = next(
                line.removeprefix("models=")
                for line in captured.splitlines()
                if line.startswith("models=")
            ).split()
            self.assertEqual(len(models), 13)
            self.assertIn("batches=4,16", captured)
            self.assertIn("warmups=1", captured)
            self.assertIn("repeats=2", captured)
            self.assertIn("power_switching=true", captured)
            self.assertIn("gpu_lock=true", captured)
            self.assertIn("auto_resume=false", captured)
            marker = (output_dir / "QUALIFICATION_PASSED.txt").read_text(
                encoding="utf-8"
            )
            self.assertIn("research_result=false", marker)
            self.assertIn("[SMOKE] PASS", completed.stdout)

    def test_hopper_installer_defaults_to_cuda_12_wheels(self):
        """Select cu129 for Hopper while preserving cu130 for newer desktop GPUs.

        Returns:
            None; the callable operates through assertions.
        """
        helper = Path(__file__).resolve().parents[1] / "scripts" / "torch_device.sh"
        command = """
source "$1"
uv() { printf '%s\\n' "$*"; }
install_torch_stack torch
"""
        observed = {}
        for device in ("h100", "h200", "5090"):
            environment = os.environ.copy()
            environment["DEVICE"] = device
            environment.pop("BENCHMARK_DEVICE", None)
            environment.pop("GPU_TORCH_INDEX_URL", None)
            completed = subprocess.run(
                ["bash", "-c", command, "bash", str(helper)],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            observed[device] = completed.stdout

        self.assertIn("https://download.pytorch.org/whl/cu129", observed["h100"])
        self.assertIn("https://download.pytorch.org/whl/cu129", observed["h200"])
        self.assertIn("https://download.pytorch.org/whl/cu130", observed["5090"])


if __name__ == "__main__":
    unittest.main()
