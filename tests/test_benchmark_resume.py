"""Regression-test the benchmark resume benchmark contract."""

import csv
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from scripts.benchmark_resume import find_resume_index, prepare_resume


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


class BenchmarkResumeTests(unittest.TestCase):
    """Define the BenchmarkResumeTests type and its associated runtime state or contract.
    
    Bases:
        unittest.TestCase
    """
    def setUp(self):
        """Create isolated fixtures required by BenchmarkResumeTests tests.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        (self.output_dir / "model_logs").mkdir()
        self.args = Namespace(
            output_dir=self.output_dir,
            models="units,chronos2,moirai2",
            device="5090",
            seed=12345,
            dataset="cnemc_imputed",
            context_lens="512",
            horizon_lens="96",
            batch_sizes="64,128,256",
            power_modes="0",
            repeats=5,
            benchmark_batches=300,
        )

    def tearDown(self):
        """Release fixtures and temporary artifacts created by BenchmarkResumeTests tests.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        self.temp_dir.cleanup()

    @staticmethod
    def _row(model, batch_size, status="SUCCESS"):
        """Construct one synthetic Phase-2 result row for resume tests.
        
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

    def _complete_log(self, model):
        """Write a synthetic model log containing the benchmark completion marker.
        
        Args:
            model (Any): Configuration or runtime value named model.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        path = self.output_dir / "model_logs" / f"{model}_benchmark.log"
        path.write_text("[COMPLETE] Finished source-directory benchmark: dataset\n")

    def _write_summary(self, rows):
        """Write summary for BenchmarkResumeTests.
        
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

    def test_resume_restarts_first_partial_model(self):
        """Regression-check that resume restarts first partial model.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        rows = [self._row("units", batch) for batch in (64, 128, 256)]
        rows.extend(self._row("chronos2", batch) for batch in (64, 128))
        self._complete_log("units")

        self.assertEqual(find_resume_index(self.args, rows), 1)

    def test_oom_is_a_valid_completed_ceiling(self):
        """Regression-check that oom is a valid completed ceiling.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        rows = [self._row("units", 64), self._row("units", 128, "OOM")]
        self._complete_log("units")

        self.assertEqual(find_resume_index(self.args, rows), 1)

    def test_supplemental_rows_do_not_invalidate_base_sweep(self):
        """Regression-check that supplemental rows do not invalidate base sweep.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        rows = [self._row("units", batch) for batch in (16, 32, 64, 128, 256)]
        self._complete_log("units")

        self.assertEqual(find_resume_index(self.args, rows), 1)

    def test_prepare_preserves_completed_rows_and_backs_up_partial_rows(self):
        """Regression-check that prepare preserves completed rows and backs up partial rows.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        rows = [self._row("units", batch) for batch in (64, 128, 256)]
        rows.extend(self._row("chronos2", batch) for batch in (64, 128))
        rows.append(self._row("moirai2", 64))
        summary_path = self._write_summary(rows)
        self._complete_log("units")
        self._complete_log("chronos2")

        resume_index = find_resume_index(self.args, rows)
        prepare_resume(self.args, resume_index)

        with summary_path.open(newline="") as handle:
            retained = list(csv.DictReader(handle))
        self.assertEqual({row["model_name"] for row in retained}, {"units"})
        backups = list(
            (self.output_dir / "benchmark_metadata" / "resume_backups").glob(
                "*/benchmark_results_batch_summary.csv"
            )
        )
        self.assertEqual(len(backups), 1)
        self.assertFalse(
            (self.output_dir / "model_logs" / "chronos2_benchmark.log").exists()
        )


if __name__ == "__main__":
    unittest.main()
