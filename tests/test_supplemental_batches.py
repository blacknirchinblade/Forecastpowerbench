"""Regression-test the supplemental batches benchmark contract."""

import csv
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from scripts.supplemental_batches import find_pending_models, prepare_supplement


FIELDNAMES = [
    "benchmark_seed",
    "model_name",
    "device_category",
    "dataset_name",
    "context_len",
    "horizon_len",
    "batch_size",
    "power_mode",
    "expected_repeats",
    "full_batches_per_iteration",
    "status_values",
    "is_complete_success",
]


class SupplementalBatchTests(unittest.TestCase):
    """Define the SupplementalBatchTests type and its associated runtime state or contract.
    
    Bases:
        unittest.TestCase
    """
    def setUp(self):
        """Create isolated fixtures required by SupplementalBatchTests tests.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        self.args = Namespace(
            output_dir=self.output_dir,
            models="units,ttm",
            device="5090",
            seed=12345,
            dataset="cnemc_imputed",
            context_lens="512",
            horizon_lens="96",
            batch_sizes="16,32",
            power_modes="0",
            repeats=5,
            benchmark_batches=300,
        )

    def tearDown(self):
        """Release fixtures and temporary artifacts created by SupplementalBatchTests tests.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        self.temp_dir.cleanup()

    @staticmethod
    def _row(model, batch_size, status="SUCCESS"):
        """Construct one synthetic result row for supplemental-sweep tests.
        
        Args:
            model (Any): Configuration or runtime value named model.
            batch_size (Any): Number of source-local windows passed to one model call.
            status (Any): Configuration or runtime value named status. Defaults to 'SUCCESS'.
        
        Returns:
            Runtime value with no declared return annotation; see the implementation and callers.
        """
        return {
            "benchmark_seed": "12345",
            "model_name": model,
            "device_category": "5090",
            "dataset_name": "cnemc_imputed",
            "context_len": "512",
            "horizon_len": "96",
            "batch_size": str(batch_size),
            "power_mode": "0",
            "expected_repeats": "5",
            "full_batches_per_iteration": "300",
            "status_values": status,
            "is_complete_success": "True" if status == "SUCCESS" else "False",
        }

    def _write_summary(self, rows):
        """Write summary for SupplementalBatchTests.
        
        Args:
            rows (Any): Per-repeat or summary records to process.
        
        Returns:
            Runtime value with no declared return annotation; see the implementation and callers.
        """
        path = self.output_dir / "benchmark_results_batch_summary.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_prepare_removes_only_requested_partial_rows(self):
        """Regression-check that prepare removes only requested partial rows.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        rows = [self._row(model, batch) for model in ("units", "ttm") for batch in (64, 128)]
        rows.append(self._row("units", 16))
        summary = self._write_summary(rows)

        pending = find_pending_models(self.args, rows)
        self.assertEqual(pending, ["units", "ttm"])
        prepare_supplement(self.args, pending)

        with summary.open(newline="") as handle:
            retained = list(csv.DictReader(handle))
        self.assertEqual(
            {(row["model_name"], int(row["batch_size"])) for row in retained},
            {("units", 64), ("units", 128), ("ttm", 64), ("ttm", 128)},
        )
        backups = list(
            (self.output_dir / "benchmark_metadata" / "supplemental_backups").glob(
                "*/benchmark_results_batch_summary.csv"
            )
        )
        self.assertEqual(len(backups), 1)

    def test_completed_models_are_skipped_and_oom_is_a_valid_ceiling(self):
        """Regression-check that completed models are skipped and oom is a valid ceiling.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        rows = [self._row(model, batch) for model in ("units", "ttm") for batch in (64, 128)]
        rows.extend([self._row("units", 16), self._row("units", 32)])
        rows.append(self._row("ttm", 16, "OOM"))

        self.assertEqual(find_pending_models(self.args, rows), [])

    def test_missing_supplemental_rows_leave_base_csv_byte_identical(self):
        """Regression-check that missing supplemental rows leave base csv byte identical.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        rows = [self._row(model, batch) for model in ("units", "ttm") for batch in (64, 128)]
        summary = self._write_summary(rows)
        original = summary.read_bytes()

        pending = find_pending_models(self.args, rows)
        prepare_supplement(self.args, pending)

        self.assertEqual(summary.read_bytes(), original)

    def test_protocol_mismatch_refuses_merge(self):
        """Regression-check that protocol mismatch refuses merge.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        rows = [self._row(model, 64) for model in ("units", "ttm")]
        rows[1]["expected_repeats"] = "3"

        with self.assertRaisesRegex(ValueError, "existing repeats"):
            find_pending_models(self.args, rows)


if __name__ == "__main__":
    unittest.main()
