"""Regression-test the chronos2 adapter benchmark contract."""

import importlib
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch


class _FakePipeline:
    """Define the _FakePipeline type and its associated runtime state or contract.
    
    Attributes:
        received_inputs: Received inputs state retained by each instance.
    """
    def __init__(self):
        """Initialize _FakePipeline and capture its validated configuration or runtime state.
        
        Returns:
            None; initializes the instance in place.
        """
        self.received_inputs = None

    def predict_quantiles(self, inputs, prediction_length, **_kwargs):
        """Produce deterministic fake or adapter predictions through _FakePipeline.
        
        Args:
            inputs (Any): Batched model context with shape (batch, context).
            prediction_length (Any): Configuration or runtime value named prediction_length.
            _kwargs (Any): Configuration or runtime value named _kwargs. Defaults to empty.
        
        Returns:
            Runtime value with no declared return annotation; see the implementation and callers.
        """
        self.received_inputs = inputs
        points = [torch.zeros((1, prediction_length)) for _ in range(len(inputs))]
        return [], points


class Chronos2AdapterTests(unittest.TestCase):
    """Define the Chronos2AdapterTests type and its associated runtime state or contract.
    
    Bases:
        unittest.TestCase
    """
    def test_identity_collate_only_applies_to_prebatched_chronos_loader(self):
        """Regression-check that identity collate only applies to prebatched chronos loader.
        
        Returns:
            Runtime value with no declared return annotation; see the implementation and callers.
        """
        calls = []

        def fake_loader(*args, **kwargs):
            """Capture Chronos DataLoader arguments and return a lightweight fake loader.
            
            Args:
                args (Any): Additional positional arguments forwarded by the callable. Defaults to empty.
                kwargs (Any): Configuration or runtime value named kwargs. Defaults to empty.
            
            Returns:
                Runtime value with no declared return annotation; see the implementation and callers.
            """
            calls.append((args, kwargs))
            return object()

        fake_pipeline_module = types.ModuleType("chronos.chronos2.pipeline")
        fake_pipeline_module.DataLoader = fake_loader

        class FakeChronos2Pipeline:
            """Define the Chronos2AdapterTests.test_identity_collate_only_applies_to_prebatched_chronos_loader.FakeChronos2Pipeline type and its associated runtime state or contract.
            """
            pass

        FakeChronos2Pipeline.__module__ = fake_pipeline_module.__name__
        fake_chronos = types.ModuleType("chronos")
        fake_chronos.Chronos2Pipeline = FakeChronos2Pipeline
        with patch.dict(
            sys.modules,
            {
                "chronos": fake_chronos,
                fake_pipeline_module.__name__: fake_pipeline_module,
            },
        ):
            sys.modules.pop("models.chronos2", None)
            module = importlib.import_module("models.chronos2")
            self.assertTrue(module._install_identity_collate())

            prebatched = object()
            fake_pipeline_module.DataLoader(prebatched, batch_size=None, pin_memory=True)
            normal = object()
            fake_pipeline_module.DataLoader(normal, batch_size=4)

        prebatched_kwargs = calls[0][1]
        self.assertIs(prebatched_kwargs["collate_fn"](prebatched), prebatched)
        self.assertNotIn("collate_fn", calls[1][1])

    def test_forecast_passes_tensor_to_chronos_pipeline(self):
        """Regression-check that forecast passes tensor to chronos pipeline.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        fake_chronos = types.ModuleType("chronos")
        fake_chronos.Chronos2Pipeline = object
        with patch.dict(sys.modules, {"chronos": fake_chronos}):
            sys.modules.pop("models.chronos2", None)
            module = importlib.import_module("models.chronos2")

        model = module.Chronos2Model.__new__(module.Chronos2Model)
        model.context_len = 4
        model.batch_size = 2
        model.pipeline = _FakePipeline()
        inputs = np.arange(8, dtype=np.float64).reshape(2, 4)
        original_inputs = inputs.copy()

        predictions, _ = model.forecast(inputs, 3)

        self.assertTrue(torch.is_tensor(model.pipeline.received_inputs))
        self.assertEqual(model.pipeline.received_inputs.dtype, torch.float32)
        self.assertEqual(tuple(model.pipeline.received_inputs.shape), (2, 1, 4))
        self.assertEqual(predictions.shape, (2, 3))
        np.testing.assert_array_equal(inputs, original_inputs)


if __name__ == "__main__":
    unittest.main()
