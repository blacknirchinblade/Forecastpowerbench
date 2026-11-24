"""Regression-test the kairos adapter benchmark contract."""

import importlib
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch


class _FakeKairos:
    """Define the _FakeKairos type and its associated runtime state or contract.
    
    Attributes:
        calls: Calls state retained by each instance.
    """
    def __init__(self):
        """Initialize _FakeKairos and capture its validated configuration or runtime state.
        
        Returns:
            None; initializes the instance in place.
        """
        self.calls = []

    def __call__(self, **kwargs):
        """Execute the callable inference behavior implemented by _FakeKairos.
        
        Args:
            kwargs (Any): Configuration or runtime value named kwargs. Defaults to empty.
        
        Returns:
            Runtime value with no declared return annotation; see the implementation and callers.
        """
        self.calls.append(kwargs)
        batch_size = len(kwargs["past_target"])
        horizon = kwargs["prediction_length"]
        quantiles = torch.stack(
            [
                torch.full((batch_size, horizon), 1.0),
                torch.full((batch_size, horizon), 2.0),
                torch.full((batch_size, horizon), 3.0),
            ],
            dim=1,
        )
        return {"prediction_outputs": quantiles}


class KairosAdapterTests(unittest.TestCase):
    """Define the KairosAdapterTests type and its associated runtime state or contract.
    
    Bases:
        unittest.TestCase
    """
    def test_forecast_preserves_prior_paper_direct_point_method(self):
        """Regression-check that forecast preserves prior paper direct point method.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        fake_kairos_module = types.ModuleType("utils.kairos.model.kairos")
        fake_kairos_module.AutoModel = object
        with patch.dict(sys.modules, {"utils.kairos.model.kairos": fake_kairos_module}):
            sys.modules.pop("models.kairos", None)
            module = importlib.import_module("models.kairos")

        model = module.KairosModel.__new__(module.KairosModel)
        model.device = "cpu"
        model.context_len = 4
        model.batch_size = 2
        model.model_context_len = 4
        model.model = _FakeKairos()

        inputs = np.arange(12, dtype=np.float32).reshape(2, 6)
        predictions, _ = model.forecast(inputs, 3)

        self.assertEqual(predictions.shape, (2, 3))
        selected = inputs[:, -4:]
        expected_mean = selected.mean(axis=1, keepdims=True)
        expected_std = selected.std(axis=1, keepdims=True) + 1e-6
        expected_point = np.repeat(2.0 * expected_std + expected_mean, 3, axis=1)
        np.testing.assert_allclose(predictions, expected_point)
        self.assertEqual(len(model.model.calls), 1)
        call = model.model.calls[0]
        self.assertFalse(call["generation"])
        self.assertNotIn("infer_is_positive", call)
        self.assertNotIn("force_flip_invariance", call)
        self.assertEqual(tuple(call["past_target"].shape), (2, 4))
        np.testing.assert_allclose(
            call["past_target"].numpy(),
            (selected - expected_mean) / expected_std,
        )


if __name__ == "__main__":
    unittest.main()
