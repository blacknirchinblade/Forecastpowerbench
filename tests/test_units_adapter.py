"""Regression tests for UniTS patch-aligned horizon adaptation."""

import unittest
from unittest.mock import patch

import numpy as np
import torch


class _FakeUniTSBackbone:
    """Expose the adapter/backbone shape contract without loading a checkpoint."""

    def __init__(self, args, configs_list, pretrain):
        """Capture the configured internal forecast horizon.

        Args:
            args: UniTS architecture configuration.
            configs_list: Per-task model configuration.
            pretrain: Whether to construct the pretraining head.

        Returns:
            None; initializes the fake backbone.
        """
        del args, pretrain
        self.pred_len = configs_list[0][1]["pred_len"]

    def to(self, device):
        """Accept the adapter's device placement operation.

        Args:
            device: Requested execution device.

        Returns:
            This fake backbone.
        """
        del device
        return self

    def eval(self):
        """Accept the adapter's evaluation-mode operation.

        Returns:
            This fake backbone.
        """
        return self

    def __call__(self, x_enc, **kwargs):
        """Return one-based internal forecast steps for slicing assertions.

        Args:
            x_enc: Batched context tensor.
            kwargs: Unused UniTS call arguments.

        Returns:
            Tensor shaped like the real UniTS forecast output.
        """
        del kwargs
        values = torch.arange(1, self.pred_len + 1, dtype=x_enc.dtype, device=x_enc.device)
        return values.view(1, self.pred_len, 1).repeat(x_enc.shape[0], 1, 1)


class UniTSHorizonTests(unittest.TestCase):
    """Verify that the UniTS adapter preserves requested horizon semantics."""

    def test_requested_horizons_round_up_to_complete_native_patches(self):
        """Map each requested horizon to the smallest enclosing native patch.

        Returns:
            None; assertions validate the patch-alignment contract.
        """
        try:
            from models.units import UniTSModel
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"UniTS dependencies are isolated in .venv_units: {exc}")

        expected = {
            1: 16,
            6: 16,
            12: 16,
            16: 16,
            24: 32,
            32: 32,
            48: 48,
            96: 96,
        }
        actual = {
            horizon: UniTSModel._align_horizon_to_patch(horizon)
            for horizon in expected
        }
        self.assertEqual(actual, expected)

    def test_non_positive_horizons_are_rejected(self):
        """Reject horizons that cannot represent a forecast.

        Returns:
            None; assertions validate input rejection.
        """
        try:
            from models.units import UniTSModel
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"UniTS dependencies are isolated in .venv_units: {exc}")

        for horizon in (None, 0, -1):
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                UniTSModel._align_horizon_to_patch(horizon)

    def test_adapter_generates_complete_patch_then_slices_requested_horizon(self):
        """Configure complete patches while returning only requested timesteps.

        Returns:
            None; assertions validate model configuration and forecast slicing.
        """
        try:
            import models.units as units
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"UniTS dependencies are isolated in .venv_units: {exc}")

        grad_was_enabled = torch.is_grad_enabled()
        try:
            with (
                patch.object(units, "UniTSBackbone", _FakeUniTSBackbone),
                patch.object(units.UniTSModel, "_load_checkpoint"),
            ):
                model = units.UniTSModel(
                    device="cpu",
                    context_len=512,
                    horizon_len=24,
                    batch_size=2,
                    num_workers=0,
                )
                predictions, _ = model.forecast(
                    np.ones((2, 512), dtype=np.float32),
                    horizon_len=24,
                )
        finally:
            torch.set_grad_enabled(grad_was_enabled)

        self.assertEqual(model.model_horizon_len, 32)
        self.assertEqual(model.configs_list[0][1]["pred_len"], 32)
        self.assertEqual(predictions.shape, (2, 24))
        np.testing.assert_array_equal(predictions[:, -1], np.array([24.0, 24.0]))


if __name__ == "__main__":
    unittest.main()
