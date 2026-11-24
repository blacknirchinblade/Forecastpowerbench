"""Collect and summarize stats telemetry around measured inference intervals."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Integrate timestamped samples using NumPy's available trapezoidal-rule implementation.
    
    Args:
        y (np.ndarray): Configuration or runtime value named y.
        x (np.ndarray): Configuration or runtime value named x.
    
    Returns:
        float: Computed scalar described by this callable.
    """
    try:
        return float(np.trapezoid(y, x))
    except AttributeError:
        return float(np.trapz(y, x))


def summarize_power_interval(
    powers: Sequence[float],
    timestamps_ns: Sequence[int],
    start_ns: int | None,
    end_ns: int | None,
) -> Dict[str, float]:
    """Summarize samples over the full monitored interval using held endpoints.
    
    Args:
        powers (Sequence[float]): Configuration or runtime value named powers.
        timestamps_ns (Sequence[int]): Configuration or runtime value named timestamps_ns.
        start_ns (int | None): Inclusive measurement start timestamp in nanoseconds.
        end_ns (int | None): Inclusive measurement end timestamp in nanoseconds.
    
    Returns:
        Dict[str, float]: Named values produced by this operation.
    """
    values = np.asarray(powers, dtype=np.float64)
    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    finite = np.isfinite(values)
    values = values[finite]
    timestamps = timestamps[finite]
    duration_sec = (
        max(0.0, (int(end_ns) - int(start_ns)) / 1e9)
        if start_ns is not None and end_ns is not None
        else float("nan")
    )
    result = {
        "telemetry_duration_sec": float(duration_sec),
        "power_sample_count": 0,
        "avg_power_w": float("nan"),
        "power_std_w": float("nan"),
        "power_p50_w": float("nan"),
        "power_p95_w": float("nan"),
        "peak_power_w": float("nan"),
        "total_energy_j": float("nan"),
    }
    if values.size == 0:
        return result

    if not np.isfinite(duration_sec) or duration_sec <= 0:
        result["power_sample_count"] = int(values.size)
        result.update(
            power_std_w=float(np.std(values)),
            power_p50_w=float(np.percentile(values, 50)),
            power_p95_w=float(np.percentile(values, 95)),
            peak_power_w=float(np.max(values)),
        )
        result["avg_power_w"] = float(np.mean(values))
        return result

    offsets = (timestamps - int(start_ns)) / 1e9
    order = np.argsort(offsets, kind="stable")
    offsets = offsets[order]
    values = values[order]

    inside = (offsets >= 0.0) & (offsets <= duration_sec)
    inside_offsets = offsets[inside]
    inside_values = values[inside]
    result["power_sample_count"] = int(inside_values.size)

    boundary_offsets = np.asarray([0.0, duration_sec], dtype=np.float64)
    boundary_values = np.interp(
        boundary_offsets,
        offsets,
        values,
        left=float(values[0]),
        right=float(values[-1]),
    )
    integration_offsets = np.concatenate(
        (
            [0.0],
            inside_offsets[(inside_offsets > 0.0) & (inside_offsets < duration_sec)],
            [duration_sec],
        )
    )
    integration_values = np.concatenate(
        (
            [boundary_values[0]],
            inside_values[(inside_offsets > 0.0) & (inside_offsets < duration_sec)],
            [boundary_values[1]],
        )
    )
    statistic_values = inside_values if inside_values.size else boundary_values
    result.update(
        power_std_w=float(np.std(statistic_values)),
        power_p50_w=float(np.percentile(statistic_values, 50)),
        power_p95_w=float(np.percentile(statistic_values, 95)),
        peak_power_w=float(np.max(statistic_values)),
    )
    energy_j = _trapezoid(integration_values, integration_offsets)
    result["total_energy_j"] = float(energy_j)
    result["avg_power_w"] = float(energy_j / duration_sec)
    return result
