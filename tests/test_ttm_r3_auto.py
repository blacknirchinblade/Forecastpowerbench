from types import SimpleNamespace

import numpy as np
import torch

import models.ttm_r3_auto as adapter
from benchmark.config import ResultRow
from benchmark.runner import SUMMARY_FIELDNAMES


class FakeTTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(context_length=52, prediction_length=16)
        self.seen_shape = None

    def forward(self, values):
        self.seen_shape = tuple(values.shape)
        batch_size = values.shape[0]
        predictions = torch.ones(
            (batch_size, 6, 1),
            dtype=values.dtype,
            device=values.device,
        )
        return SimpleNamespace(prediction_outputs=predictions)


def test_adaptive_ttm_fallback_pins_official_router_result(monkeypatch):
    calls = []
    fake_model = FakeTTM()

    def fake_get_model(model_path, **kwargs):
        calls.append((model_path, kwargs))
        if kwargs.get("return_model_key"):
            return "52-16-dec-52-r3"
        return fake_model

    monkeypatch.setattr(adapter, "get_ttm_model", fake_get_model)
    monkeypatch.setattr(adapter, "_resolve_exact_context_revision", lambda *_: None)
    model = adapter.TTMR3AutoModel(device="cpu", context_len=32, horizon_len=6)
    predictions, _ = model.forecast(np.ones((2, 32), dtype=np.float32), 6)

    assert model.model_variant == "52-16-dec-52-r3"
    assert predictions.shape == (2, 6)
    assert fake_model.seen_shape == (2, 32, 1)
    assert len(calls) == 2
    assert calls[0][0] == calls[1][0] == "ibm-research/ttm-r3"
    resolution_kwargs = dict(calls[0][1])
    resolution_kwargs.pop("return_model_key")
    load_kwargs = dict(calls[1][1])
    assert load_kwargs.pop("model_revision") == "52-16-dec-52-r3"
    assert resolution_kwargs == load_kwargs
    assert resolution_kwargs == {
        "context_length": 32,
        "prediction_length": 6,
        "freq_prefix_tuning": False,
        "freq": None,
        "prefer_l1_loss": False,
        "prefer_longer_context": False,
        "force_return": "zeropad",
        "prefer_known_mappings": False,
    }


def test_exact_context_prefers_smallest_sufficient_official_horizon(monkeypatch):
    revisions = (
        ("2048-96-r3", 2048, 96, False),
        ("2048-720-r3", 2048, 720, False),
        ("512-336-dec-512-r3", 512, 336, True),
    )
    monkeypatch.setattr(adapter, "_official_r3_revisions", lambda: revisions)

    assert adapter._resolve_exact_context_revision(2048, 96) == "2048-96-r3"
    assert adapter._resolve_exact_context_revision(2048, 192) == "2048-720-r3"
    assert adapter._resolve_exact_context_revision(2048, 384) == "2048-720-r3"
    assert adapter._resolve_exact_context_revision(1024, 192) is None


def test_results_have_auditable_model_variant_columns():
    assert "model_artifact" in ResultRow.model_fields
    assert "model_variant" in ResultRow.model_fields
    assert SUMMARY_FIELDNAMES[1:4] == [
        "model_name",
        "model_artifact",
        "model_variant",
    ]


def test_horizon_one_uses_native_head_then_slices(monkeypatch):
    calls = []
    fake_model = FakeTTM()

    def fake_get_model(model_path, **kwargs):
        calls.append((model_path, kwargs))
        if kwargs.get("return_model_key"):
            return "156-16-dec-52-r3"
        return fake_model

    monkeypatch.setattr(adapter, "get_ttm_model", fake_get_model)
    monkeypatch.setattr(adapter, "_resolve_exact_context_revision", lambda *_: None)
    model = adapter.TTMR3AutoModel(device="cpu", context_len=32, horizon_len=1)
    predictions, _ = model.forecast(np.ones((2, 32), dtype=np.float32), 1)

    assert predictions.shape == (2, 1)
    assert calls[1][1]["model_revision"] == "156-16-dec-52-r3"
    assert calls[1][1]["prediction_length"] == 16
