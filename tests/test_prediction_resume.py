"""Regression-test the prediction resume benchmark contract."""

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from scripts.prediction_resume import find_resume_index, prepare_resume


class PredictionResumeTests(unittest.TestCase):
    """Define the PredictionResumeTests type and its associated runtime state or contract.
    
    Bases:
        unittest.TestCase
    """
    def setUp(self):
        """Create isolated fixtures required by PredictionResumeTests tests.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        (self.output_dir / "benchmark_metadata").mkdir()
        (self.output_dir / "model_logs").mkdir()
        self.args = Namespace(
            output_dir=self.output_dir,
            models="units,chronos2,moirai2",
            device="cpu",
            seed=12345,
            dataset="cnemc_imputed",
            context_lens="512",
            horizon_lens="96",
            prediction_batch_size=512,
        )

    def tearDown(self):
        """Release fixtures and temporary artifacts created by PredictionResumeTests tests.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        self.temp_dir.cleanup()

    def _write_metadata(self, model):
        """Write metadata for PredictionResumeTests.
        
        Args:
            model (Any): Configuration or runtime value named model.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        metadata = {
            "benchmark_seed": 12345,
            "config": {
                "models": [model],
                "device_category": "cpu",
                "dataset_folder": "cnemc_imputed",
                "context_lens": [512],
                "horizon_lens": [96],
                "prediction_batch_size": 512,
            },
            "source_manifest": {
                "dataset_sha256": "dataset-sha",
                "source_count": 2,
            },
        }
        path = self.output_dir / "benchmark_metadata" / f"{model}_prediction.json"
        path.write_text(json.dumps(metadata))

    def _write_prediction(self, model, source):
        """Write prediction for PredictionResumeTests.
        
        Args:
            model (Any): Configuration or runtime value named model.
            source (Any): One independently validated source series.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        path = (
            self.output_dir
            / "benchmark_predictions"
            / model
            / "cnemc_imputed"
            / source
            / "ctx_512_hor_96.npz"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            y_pred=np.ones((2, 96), dtype=np.float32),
            y_true=np.ones((2, 96), dtype=np.float32),
            y_true_observed=np.ones((2, 96), dtype=np.bool_),
            benchmark_seed=np.asarray(12345),
            reference_batch_size=np.asarray(512),
            context_mae_scale=np.ones(2, dtype=np.float32),
            context_mse_scale=np.ones(2, dtype=np.float32),
            accuracy_seasonality=np.asarray(24),
            imputation_policy=np.asarray("per_source_global_median"),
        )

    def test_resume_restarts_first_incomplete_prediction_model(self):
        """Regression-check that resume restarts first incomplete prediction model.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        for model in ("units", "chronos2"):
            self._write_metadata(model)
        for source in ("source_a", "source_b"):
            self._write_prediction("units", source)
        self._write_prediction("chronos2", "source_a")

        self.assertEqual(find_resume_index(self.args), 1)

    def test_prepare_preserves_complete_models_and_moves_partial_state(self):
        """Regression-check that prepare preserves complete models and moves partial state.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        for model in ("units", "chronos2"):
            self._write_metadata(model)
        for source in ("source_a", "source_b"):
            self._write_prediction("units", source)
        self._write_prediction("chronos2", "source_a")
        (self.output_dir / "model_logs" / "chronos2_prediction.log").write_text(
            "interrupted"
        )

        resume_index = find_resume_index(self.args)
        prepare_resume(self.args, resume_index)

        self.assertTrue((self.output_dir / "benchmark_predictions" / "units").is_dir())
        self.assertFalse((self.output_dir / "benchmark_predictions" / "chronos2").exists())
        backups = list(
            (self.output_dir / "benchmark_metadata" / "prediction_resume_backups").glob(
                "*/benchmark_predictions/chronos2"
            )
        )
        self.assertEqual(len(backups), 1)


if __name__ == "__main__":
    unittest.main()
