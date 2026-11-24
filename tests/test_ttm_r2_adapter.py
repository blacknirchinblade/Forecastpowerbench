"""Regression-test the ttm r2 adapter benchmark contract."""

import importlib
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch


class _FakeTTM:
    """Define the _FakeTTM type and its associated runtime state or contract.
    
    Attributes:
        device: Requested execution device or device selector.
    """
    def __init__(self):
        """Initialize _FakeTTM and capture its validated configuration or runtime state.
        
        Returns:
            None; initializes the instance in place.
        """
        self.device = None

    def to(self, device):
        """Mirror the framework to lifecycle method for the _FakeTTM test or compatibility object.
        
        Args:
            device (Any): Requested execution device or device selector.
        
        Returns:
            Runtime value with no declared return annotation; see the implementation and callers.
        """
        self.device = device
        return self

    def eval(self):
        """Mirror the framework eval lifecycle method for the _FakeTTM test or compatibility object.
        
        Returns:
            Runtime value with no declared return annotation; see the implementation and callers.
        """
        return self

    def __call__(self, values):
        """Execute the callable inference behavior implemented by _FakeTTM.
        
        Args:
            values (Any): Values to normalize, summarize, or serialize.
        
        Returns:
            Runtime value with no declared return annotation; see the implementation and callers.
        """
        batch_size = values.shape[0]
        output = torch.arange(192, dtype=torch.float32).repeat(batch_size, 1)
        return types.SimpleNamespace(prediction_outputs=output.unsqueeze(-1))


class TTMR2AdapterTests(unittest.TestCase):
    """Define the TTMR2AdapterTests type and its associated runtime state or contract.
    
    Bases:
        unittest.TestCase
    """
    def _load_adapter(self):
        """Load adapter for TTMR2AdapterTests.
        
        Returns:
            Runtime value with no declared return annotation; see the implementation and callers.
        """
        calls = []

        def fake_get_model(path, **kwargs):
            """Return the fake TTM checkpoint object used to isolate adapter behavior.
            
            Args:
                path (Any): Filesystem path consumed by the operation.
                kwargs (Any): Configuration or runtime value named kwargs. Defaults to empty.
            
            Returns:
                Runtime value with no declared return annotation; see the implementation and callers.
            """
            calls.append((path, kwargs))
            return _FakeTTM()

        get_model_module = types.ModuleType("tsfm_public.toolkit.get_model")
        get_model_module.get_model = fake_get_model
        toolkit_module = types.ModuleType("tsfm_public.toolkit")
        toolkit_module.get_model = get_model_module
        package_module = types.ModuleType("tsfm_public")
        package_module.toolkit = toolkit_module
        modules = {
            "tsfm_public": package_module,
            "tsfm_public.toolkit": toolkit_module,
            "tsfm_public.toolkit.get_model": get_model_module,
        }
        sys.modules.pop("models.ttm_r2", None)
        with patch.dict(sys.modules, modules):
            adapter = importlib.import_module("models.ttm_r2")
        return adapter, calls

    def test_loads_exact_512_192_r2_checkpoint(self):
        """Regression-check that loads exact 512 192 r2 checkpoint.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        adapter, calls = self._load_adapter()
        model = adapter.TTM2Model(device="cpu", context_len=512, horizon_len=96)

        self.assertEqual(len(calls), 1)
        path, kwargs = calls[0]
        self.assertEqual(path, "ibm-granite/granite-timeseries-ttm-r2")
        self.assertEqual(kwargs["model_revision"], "512-192-r2")
        self.assertEqual(kwargs["context_length"], 512)
        self.assertEqual(kwargs["prediction_length"], 192)
        self.assertEqual(model.model.device, "cpu")

    def test_returns_requested_point_forecast_horizon(self):
        """Regression-check that returns requested point forecast horizon.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        adapter, _ = self._load_adapter()
        model = adapter.TTM2Model(device="cpu", context_len=512, horizon_len=96)
        predictions, elapsed = model.forecast(
            np.ones((3, 512), dtype=np.float32), horizon_len=96
        )

        self.assertEqual(predictions.shape, (3, 96))
        np.testing.assert_array_equal(predictions[0], np.arange(96, dtype=np.float32))
        self.assertGreaterEqual(elapsed, 0.0)

    def test_rejects_context_larger_than_native_checkpoint(self):
        """Regression-check that rejects context larger than native checkpoint.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        adapter, _ = self._load_adapter()
        with self.assertRaisesRegex(ValueError, "exceeds TTM r2 model context"):
            adapter.TTM2Model(device="cpu", context_len=1024, horizon_len=96)


if __name__ == "__main__":
    unittest.main()
