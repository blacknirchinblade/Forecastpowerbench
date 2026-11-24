"""Deterministic multi-objective scores used by publication analyses."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


CRITERIA = {
    "error": None,
    "energy": "energy_j_per_forecast",
    "latency": "inference_ms_per_forecast",
    "memory": "memory_gb",
}


def _minmax(values: pd.Series) -> tuple[pd.Series, tuple[float, float]]:
    numeric = pd.to_numeric(values, errors="coerce")
    lower = float(numeric.min())
    upper = float(numeric.max())
    if not np.isfinite(lower) or not np.isfinite(upper):
        return pd.Series(np.nan, index=values.index), (lower, upper)
    if upper == lower:
        return pd.Series(0.0, index=values.index), (lower, upper)
    return (numeric - lower) / (upper - lower), (lower, upper)


def _positive_log(values: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    invalid = numeric.notna() & numeric.le(0)
    if invalid.any():
        raise ValueError(f"{name} must contain positive values")
    return np.log10(numeric)


def normalize_score_components(
    rows: pd.DataFrame,
    *,
    error_metric: str = "mase",
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Normalize EAS error linearly and energy per forecast in log space."""
    required = {error_metric, "energy_j_per_forecast"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Missing EAS columns: {sorted(missing)}")

    normalized = rows.copy()
    normalized["error_norm"], error_bounds = _minmax(normalized[error_metric])
    energy_log = _positive_log(
        normalized["energy_j_per_forecast"], "energy_j_per_forecast"
    )
    normalized["energy_norm"], energy_bounds = _minmax(energy_log)
    return normalized, {"error": error_bounds, "energy_log10": energy_bounds}


def alpha_score(rows: pd.DataFrame, alpha: float) -> pd.Series:
    """Return the higher-is-better alpha-conditioned energy-accuracy score."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    required = {"error_norm", "energy_norm"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Missing normalized EAS columns: {sorted(missing)}")
    return 1.0 - (
        alpha * rows["error_norm"] + (1.0 - alpha) * rows["energy_norm"]
    )


def normalize_multiobjective_alpha_components(
    rows: pd.DataFrame,
    *,
    error_metric: str = "mase",
) -> tuple[pd.DataFrame, dict[str, dict[str, tuple[float, float]]]]:
    """Normalize lower-is-better deployment criteria within each hardware."""
    columns = {
        "error": error_metric,
        "energy": "energy_j_per_forecast",
        "latency": "inference_ms_per_forecast",
        "memory": "memory_gb",
    }
    required = {"hardware", *columns.values()}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Missing MOAS columns: {sorted(missing)}")

    normalized = rows.copy()
    bounds: dict[str, dict[str, tuple[float, float]]] = {}
    for hardware, index in normalized.groupby("hardware", sort=True).groups.items():
        bounds[str(hardware)] = {}
        for criterion, column in columns.items():
            values = normalized.loc[index, column]
            transformed = values if criterion == "error" else _positive_log(values, column)
            scaled, criterion_bounds = _minmax(transformed)
            normalized.loc[index, f"{criterion}_deployment_norm"] = scaled
            bounds[str(hardware)][criterion] = criterion_bounds
    return normalized, bounds


def multiobjective_alpha_score(
    rows: pd.DataFrame,
    priorities: Mapping[str, float],
) -> pd.Series:
    """Return MOAS for a declared non-negative criterion-priority vector."""
    names = tuple(CRITERIA)
    if set(priorities) != set(names):
        raise ValueError(f"priorities must contain exactly {list(names)}")
    weights = np.asarray([priorities[name] for name in names], dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("priorities must be finite, non-negative, and not all zero")
    weights /= weights.sum()
    columns = [f"{name}_deployment_norm" for name in names]
    missing = set(columns).difference(rows.columns)
    if missing:
        raise ValueError(f"Missing normalized MOAS columns: {sorted(missing)}")
    return 1.0 - rows[columns].mul(weights, axis=1).sum(axis=1)


def _pareto_mask(values: np.ndarray) -> np.ndarray:
    efficient = np.ones(len(values), dtype=bool)
    for index, candidate in enumerate(values):
        dominated = np.all(values <= candidate, axis=1) & np.any(
            values < candidate, axis=1
        )
        dominated[index] = False
        efficient[index] = not dominated.any()
    return efficient


def robust_rank_acceptability(
    rows: pd.DataFrame,
    *,
    error_metric: str = "mase",
    n_draws: int = 20_000,
    seed: int = 12_345,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Estimate preference-robust ranks using symmetric Dirichlet weights."""
    if n_draws <= 0:
        raise ValueError("n_draws must be positive")
    columns = {
        "error": error_metric,
        "energy": "energy_j_per_forecast",
        "latency": "inference_ms_per_forecast",
        "memory": "memory_gb",
    }
    required = {"hardware", "model_name", *columns.values()}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Missing PRRA columns: {sorted(missing)}")

    working = rows.copy()
    for column in columns.values():
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.replace([np.inf, -np.inf], np.nan)

    hardware = sorted(working["hardware"].dropna().astype(str).unique())
    if not hardware:
        raise ValueError("No hardware rows are available")
    complete_models = set(working["model_name"].dropna().astype(str))
    for device in hardware:
        device_rows = working[working["hardware"].astype(str).eq(device)]
        finite = device_rows[list(columns.values())].notna().all(axis=1)
        complete_models &= set(device_rows.loc[finite, "model_name"].astype(str))
    if not complete_models:
        raise ValueError("No model has complete criteria on every hardware")

    eligible = working[
        working["model_name"].astype(str).isin(complete_models)
    ].copy()
    duplicate = eligible.duplicated(["hardware", "model_name"], keep=False)
    if duplicate.any():
        raise ValueError("PRRA requires one operating point per hardware/model")

    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(len(columns)), size=n_draws)
    per_hardware_records: list[dict[str, object]] = []
    cost_records: list[dict[str, object]] = []

    for device in hardware:
        device_rows = eligible[
            eligible["hardware"].astype(str).eq(device)
        ].sort_values("model_name")
        models = device_rows["model_name"].astype(str).to_numpy()
        model_count = len(models)
        rank_costs = np.empty((model_count, len(columns)), dtype=float)
        raw_values = device_rows[list(columns.values())].to_numpy(dtype=float)

        for criterion_index, (criterion, column) in enumerate(columns.items()):
            ranks = device_rows[column].rank(method="average", ascending=True).to_numpy()
            rank_costs[:, criterion_index] = (
                (ranks - 1.0) / (model_count - 1.0) if model_count > 1 else 0.0
            )
            for model, cost in zip(models, rank_costs[:, criterion_index]):
                cost_records.append(
                    {
                        "hardware": device,
                        "model_name": model,
                        "criterion": criterion,
                        "rank_cost": float(cost),
                    }
                )

        sampled_costs = weights @ rank_costs.T
        sampled_costs += rng.random(sampled_costs.shape) * np.finfo(float).eps
        order = np.argsort(sampled_costs, axis=1, kind="stable")
        sampled_ranks = np.empty_like(order)
        sampled_ranks[
            np.arange(n_draws)[:, None], order
        ] = np.arange(1, model_count + 1)
        pareto = _pareto_mask(raw_values)

        for model_index, model in enumerate(models):
            ranks = sampled_ranks[:, model_index]
            expected_rank = float(ranks.mean())
            acceptability = (
                1.0
                if model_count == 1
                else 1.0 - (expected_rank - 1.0) / (model_count - 1.0)
            )
            per_hardware_records.append(
                {
                    "hardware": device,
                    "model_name": model,
                    "rank_acceptability": acceptability,
                    "expected_rank": expected_rank,
                    "top1_probability": float(np.mean(ranks <= 1)),
                    "top3_probability": float(np.mean(ranks <= min(3, model_count))),
                    "pareto_efficient": bool(pareto[model_index]),
                }
            )

    per_hardware = pd.DataFrame(per_hardware_records).sort_values(
        ["hardware", "model_name"]
    ).reset_index(drop=True)
    aggregate = (
        per_hardware.groupby("model_name", as_index=False)
        .agg(
            rank_acceptability=("rank_acceptability", "mean"),
            worst_hardware_acceptability=("rank_acceptability", "min"),
            top1_probability=("top1_probability", "mean"),
            top3_probability=("top3_probability", "mean"),
            pareto_hardware_fraction=("pareto_efficient", "mean"),
        )
        .sort_values(
            ["rank_acceptability", "worst_hardware_acceptability", "model_name"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )
    costs = pd.DataFrame(cost_records).sort_values(
        ["hardware", "model_name", "criterion"]
    ).reset_index(drop=True)
    return per_hardware, aggregate, costs
