"""Verify package metadata and checkout-independent command behavior."""

from __future__ import annotations

import re
import subprocess
import tempfile
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIRST_PARTY_PATHS = (
    ROOT / "powerbench",
    ROOT / "analysis",
    ROOT / "benchmark",
    ROOT / "experiments",
    ROOT / "models",
    ROOT / "scripts",
)


class PackagePortabilityTests(unittest.TestCase):
    """Protect the anonymous artifact from host-specific path assumptions.

    Bases:
        unittest.TestCase
    """

    def test_console_entry_points_are_declared(self) -> None:
        """Require the public benchmark commands in package metadata.

        Returns:
            None; missing entry points fail the assertion.
        """
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
        scripts = metadata["project"]["scripts"]
        self.assertEqual(scripts["powerbench"], "powerbench.cli:main")
        self.assertEqual(scripts["powerbench-forecast"], "experiments.forecasting:main")

    def test_launcher_help_is_independent_of_current_directory(self) -> None:
        """Run launcher help outside the checkout without creating local files.

        Returns:
            None; command failure or leaked files fails the assertion.
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            working_directory = Path(temporary_directory)
            completed = subprocess.run(
                ["bash", str(ROOT / "run_unified_benchmark.sh"), "--help"],
                cwd=working_directory,
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(list(working_directory.iterdir()), [])

    def test_first_party_sources_have_no_user_home_literals(self) -> None:
        """Reject embedded Linux home directories and private IPv4 addresses.

        Returns:
            None; every discovered host-specific literal is reported.
        """
        home_pattern = re.compile(
            r"/(?:root/|(?:home|Users)/[A-Za-z0-9_.-]+/)"
        )
        private_ip_pattern = re.compile(
            r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
            r"|192\.168\.\d{1,3}\.\d{1,3}"
            r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"
        )
        failures: list[str] = []
        for directory in FIRST_PARTY_PATHS:
            for path in directory.rglob("*"):
                if not path.is_file() or "__pycache__" in path.parts:
                    continue
                if path.suffix not in {".py", ".sh", ".md", ".yaml", ".yml", ".json"}:
                    continue
                text = path.read_text(errors="replace")
                if home_pattern.search(text) or private_ip_pattern.search(text):
                    failures.append(str(path.relative_to(ROOT)))
        self.assertFalse(failures, f"Host-specific paths found: {failures}")


if __name__ == "__main__":
    unittest.main()
