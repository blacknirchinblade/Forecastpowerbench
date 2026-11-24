"""Regression-test the moirai2 adapter benchmark contract."""

import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch


class _FakeModule:
    """Define the _FakeModule type and its associated runtime state or contract."""
    quantile_levels = [0.1, 0.5, 0.9]

    @classmethod
    def from_pretrained(cls, _model_id):
        """Construct _FakeModule through its pretrained-model compatibility entry point.
        
        Args:
            _model_id (Any): Configuration or runtime value named _model_id.
        
        Returns:
            Runtime value with no declared return annotation; see the implementation and callers.
        """
        return cls()


class _FakeForecast:
    """Define the _FakeForecast type and its associated runtime state or contract.
    
    Attributes:
        call_kwargs: Call kwargs state retained by each instance.
        context_length: Context length state retained by each instance.
        module: Module state retained by each instance.
        prediction_length: Prediction length state retained by each instance.
    """
    def __init__(self, module, prediction_length, context_length, **_kwargs):
        """Initialize _FakeForecast and capture its validated configuration or runtime state.
        
        Args:
            module (Any): Configuration or runtime value named module.
            prediction_length (Any): Configuration or runtime value named prediction_length.
            context_length (Any): Configuration or runtime value named context_length.
            _kwargs (Any): Configuration or runtime value named _kwargs. Defaults to empty.
        
        Returns:
            None; initializes the instance in place.
        """
        self.module = module
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.call_kwargs = None

    def to(self, _device):
        """Mirror the framework to lifecycle method for the _FakeForecast test or compatibility object.
        
        Args:
            _device (Any): Configuration or runtime value named _device.
        
        Returns:
            Runtime value with no declared return annotation; see the implementation and callers.
        """
        return self

    def eval(self):
        """Mirror the framework eval lifecycle method for the _FakeForecast test or compatibility object.
        
        Returns:
            Runtime value with no declared return annotation; see the implementation and callers.
        """
        return self

    def __call__(self, **kwargs):
        """Execute the callable inference behavior implemented by _FakeForecast.
        
        Args:
            kwargs (Any): Configuration or runtime value named kwargs. Defaults to empty.
        
        Returns:
            Runtime value with no declared return annotation; see the implementation and callers.
        """
        self.call_kwargs = kwargs
        batch = kwargs["past_target"].shape[0]
        return torch.arange(
            batch * 3 * self.prediction_length,
            dtype=torch.float32,
            device=kwargs["past_target"].device,
        ).reshape(batch, 3, self.prediction_length)

    def predict(self, _inputs):
        """Produce deterministic fake or adapter predictions through _FakeForecast.
        
        Args:
            _inputs (Any): Configuration or runtime value named _inputs.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        raise AssertionError("The redundant list-based predict wrapper must not run")


class Moirai2AdapterTests(unittest.TestCase):
    """Define the Moirai2AdapterTests type and its associated runtime state or contract.
    
    Bases:
        unittest.TestCase
    """
    def test_dense_benchmark_input_uses_equivalent_direct_tensor_transport(self):
        """Regression-check that dense benchmark input uses equivalent direct tensor transport.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        forecast_module = types.ModuleType(
            "utils.uni2ts.src.uni2ts.model.moirai2.forecast"
        )
        forecast_module.Moirai2Forecast = _FakeForecast
        module_module = types.ModuleType("utils.uni2ts.src.uni2ts.model.moirai2.module")
        module_module.Moirai2Module = _FakeModule

        with patch.dict(
            sys.modules,
            {
                forecast_module.__name__: forecast_module,
                module_module.__name__: module_module,
            },
        ):
            sys.modules.pop("models.moirai_2p0", None)
            from models.moirai_2p0 import Moirai2Model

            model = Moirai2Model(
                device="cpu", context_len=4, horizon_len=2, batch_size=2
            )
            predictions, _ = model.forecast(
                np.arange(8, dtype=np.float32).reshape(2, 4), 2
            )

        kwargs = model.model.call_kwargs
        self.assertEqual(tuple(kwargs["past_target"].shape), (2, 4, 1))
        self.assertTrue(torch.all(kwargs["past_observed_target"]))
        self.assertFalse(torch.any(kwargs["past_is_pad"]))
        expected = np.arange(12, dtype=np.float32).reshape(2, 3, 2)[:, 1]
        np.testing.assert_array_equal(predictions, expected)

    def test_non_finite_inputs_are_rejected_before_model_execution(self):
        """Regression-check that non finite inputs are rejected before model execution.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        forecast_module = types.ModuleType(
            "utils.uni2ts.src.uni2ts.model.moirai2.forecast"
        )
        forecast_module.Moirai2Forecast = _FakeForecast
        module_module = types.ModuleType("utils.uni2ts.src.uni2ts.model.moirai2.module")
        module_module.Moirai2Module = _FakeModule

        with patch.dict(
            sys.modules,
            {
                forecast_module.__name__: forecast_module,
                module_module.__name__: module_module,
            },
        ):
            sys.modules.pop("models.moirai_2p0", None)
            from models.moirai_2p0 import Moirai2Model

            model = Moirai2Model(
                device="cpu", context_len=4, horizon_len=2, batch_size=1
            )
            with self.assertRaisesRegex(ValueError, "non-finite"):
                model.forecast(
                    np.array([[1.0, 2.0, np.nan, 4.0]], dtype=np.float32), 2
                )


if __name__ == "__main__":
    unittest.main()
