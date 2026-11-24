#!/usr/bin/env python3
"""Fail-closed validation for a completed PowerBench result directory."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


def _finite_positive(row: dict[str, str], field: str) -> bool:
    """Return whether a value parses to a finite scalar greater than zero.
    
    Args:
        row (dict[str, str]): Configuration or runtime value named row.
        field (str): Configuration or runtime value named field.
    
    Returns:
        bool: Whether the condition described by this callable holds.
    """
    try:
        value = float(row.get(field, "nan"))
    except ValueError:
        return False
    return math.isfinite(value) and value > 0


def _finite(row: dict[str, str], field: str) -> bool:
    """Return whether finite for the module.
    
    Args:
        row (dict[str, str]): Configuration or runtime value named row.
        field (str): Configuration or runtime value named field.
    
    Returns:
        bool: Whether the condition described by this callable holds.
    """
    try:
        value = float(row.get(field, "nan"))
    except ValueError:
        return False
    return math.isfinite(value)


def _is_true(value: str) -> bool:
    """Return whether is true for the module.
    
    Args:
        value (str): Configuration or runtime value named value.
    
    Returns:
        bool: Whether the condition described by this callable holds.
    """
    return value.strip().lower() in {"1", "true", "yes"}


def validate_predictions(
    output_dir: Path,
    models: list[str],
    seed: int,
    prediction_batch_size: int,
) -> list[str]:
    """Validate predictions for the module.
    
    Args:
        output_dir (Path): Directory containing run artifacts.
        models (list[str]): Configuration or runtime value named models.
        seed (int): Root random seed used for deterministic execution.
        prediction_batch_size (int): Fixed batch size used only for deterministic Phase-1 prediction
                                     generation.
    
    Returns:
        list[str]: Result produced by this operation.
    """
    errors: list[str] = []
    root = output_dir / "benchmark_predictions"
    for model in models:
        files = sorted((root / model).rglob("*.npz"))
        if not files:
            errors.append(f"{model}: no prediction NPZ files")
            continue
        for path in files:
            try:
                with np.load(path, allow_pickle=False) as data:
                    required = {
                        "y_pred",
                        "y_true",
                        "y_true_observed",
                        "benchmark_seed",
                        "reference_batch_size",
                        "context_mae_scale",
                        "context_mse_scale",
                        "accuracy_seasonality",
                        "imputation_policy",
                    }
                    missing = required.difference(data.files)
                    if missing:
                        errors.append(f"{path}: missing fields {sorted(missing)}")
                        continue
                    prediction = data["y_pred"]
                    target = data["y_true"]
                    observed = data["y_true_observed"]
                    if prediction.shape != target.shape or target.shape != observed.shape:
                        errors.append(
                            f"{path}: shape mismatch pred={prediction.shape} true={target.shape} mask={observed.shape}"
                        )
                    if prediction.ndim != 2 or not np.isfinite(prediction).all():
                        errors.append(f"{path}: predictions must be a finite 2-D array")
                    if int(data["benchmark_seed"]) != seed:
                        errors.append(f"{path}: benchmark seed differs from {seed}")
                    if int(data["reference_batch_size"]) != prediction_batch_size:
                        errors.append(
                            f"{path}: prediction batch size differs from {prediction_batch_size}"
                        )
                    if str(data["imputation_policy"]) != "per_source_global_median":
                        errors.append(f"{path}: unexpected imputation policy")
                    window_count = prediction.shape[0] if prediction.ndim == 2 else -1
                    if data["context_mae_scale"].shape != (window_count,):
                        errors.append(f"{path}: invalid context_mae_scale shape")
                    if data["context_mse_scale"].shape != (window_count,):
                        errors.append(f"{path}: invalid context_mse_scale shape")
            except (OSError, ValueError, KeyError) as exc:
                errors.append(f"{path}: unreadable NPZ ({exc})")
    return errors


def validate_summary(output_dir: Path, models: list[str], device: str) -> list[str]:
    """Validate summary for the module.
    
    Args:
        output_dir (Path): Directory containing run artifacts.
        models (list[str]): Configuration or runtime value named models.
        device (str): Requested execution device or device selector.
    
    Returns:
        list[str]: Result produced by this operation.
    """
    errors: list[str] = []
    path = output_dir / "benchmark_results_batch_summary.csv"
    if not path.is_file():
        return [f"missing batch summary: {path}"]
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = set(reader.fieldnames or [])
    if not rows:
        return [f"empty batch summary: {path}"]
    forbidden_accuracy = {"nrmse", "smape", "mae", "mse", "rmse", "mape"}
    leaked = sorted(field for field in fields if field.lower() in forbidden_accuracy)
    if leaked:
        errors.append(f"performance summary contains accuracy fields: {leaked}")

    identity_fields = (
        "model_name",
        "device_category",
        "dataset_name",
        "pollutant",
        "context_len",
        "horizon_len",
        "batch_size",
        "power_mode",
    )
    seen_identities: set[tuple[str, ...]] = set()
    for row in rows:
        identity = tuple(row.get(field, "") for field in identity_fields)
        if identity in seen_identities:
            errors.append(
                "duplicate benchmark configuration: "
                + ", ".join(
                    f"{field}={row.get(field, '')}" for field in identity_fields
                )
            )
        seen_identities.add(identity)

    for model in models:
        model_rows = [row for row in rows if row.get("model_name") == model]
        if not model_rows:
            errors.append(f"{model}: no benchmark summary rows")
            continue
        for row in model_rows:
            label = (
                f"{model} ctx={row.get('context_len')} hor={row.get('horizon_len')} "
                f"bs={row.get('batch_size')}"
            )
            statuses = set(filter(None, row.get("status_values", "").split("|")))
            unknown_statuses = statuses.difference(
                {"SUCCESS", "OOM", "TIMEOUT", "ERROR", "SWAP_CONTAMINATED", "HARDWARE_THROTTLED"}
            )
            if unknown_statuses:
                errors.append(f"{label}: unknown status {sorted(unknown_statuses)}")
            if statuses.intersection({"ERROR", "SWAP_CONTAMINATED", "HARDWARE_THROTTLED"}):
                errors.append(f"{label}: invalid status {sorted(statuses)}")
            if "SUCCESS" not in statuses:
                continue
            if not _is_true(row.get("is_complete_success", "")):
                errors.append(f"{label}: incomplete successful repeats")
            for field in (
                "measurement_duration_sec_mean",
                "throughput_windows_per_sec_mean",
                "throughput_points_per_sec_mean",
                "peak_cpu_ram_gb_mean",
            ):
                if not _finite_positive(row, field):
                    errors.append(f"{label}: missing/non-positive {field}")

            if device in {"5090", "a6000", "h100", "h200", "jetson", "cpu"}:
                for field in (
                    "avg_power_w_mean",
                    "total_energy_j_mean",
                    "power_sample_count_mean",
                ):
                    if not _finite_positive(row, field):
                        errors.append(f"{label}: missing/non-positive {field}")
            if not row.get("power_scope", "").strip() or row.get("power_scope") == "unavailable":
                errors.append(f"{label}: missing power_scope")
            if device in {"5090", "a6000", "h100", "h200", "jetson"}:
                for field in (
                    "peak_gpu_allocated_gb_mean",
                    "peak_gpu_memory_used_gb_mean",
                    "gpu_memory_total_gb_mean",
                    "avg_gpu_temp_c_mean",
                ):
                    if not _finite_positive(row, field):
                        errors.append(f"{label}: missing/non-positive {field}")
                if not _finite(row, "avg_gpu_compute_util_percent_mean"):
                    errors.append(f"{label}: missing avg_gpu_compute_util_percent_mean")
            if device == "jetson":
                if not _is_true(row.get("jetson_oc_counter_available", "")):
                    errors.append(f"{label}: Jetson OC counter unavailable")
                for field in (
                    "avg_gpu_soc_power_w_mean",
                    "avg_cpu_cv_power_w_mean",
                    "avg_system_5v_power_w_mean",
                    "avg_cpu_temp_c_mean",
                ):
                    if not _finite_positive(row, field):
                        errors.append(f"{label}: missing/non-positive {field}")
                if not _finite(row, "jetson_oc_event_delta_mean"):
                    errors.append(f"{label}: missing jetson_oc_event_delta_mean")
                elif float(row["jetson_oc_event_delta_mean"]) != 0:
                    errors.append(f"{label}: hardware over-current events detected")
            if device == "cpu" and not _finite_positive(row, "avg_cpu_temp_c_mean"):
                errors.append(f"{label}: missing/non-positive avg_cpu_temp_c_mean")
    return errors


def main() -> int:
    """Parse command-line arguments and execute the the module workflow.
    
    Returns:
        int: Computed scalar described by this callable.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", required=True, help="Comma-separated model names")
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--prediction-batch-size", type=int, required=True)
    parser.add_argument(
        "--scope",
        choices=("all", "predictions", "benchmark"),
        default="all",
        help="Validate both phases, saved predictions only, or benchmark summary only.",
    )
    args = parser.parse_args()

    models = [model for model in args.models.split(",") if model]
    errors: list[str] = []
    if args.scope in {"all", "predictions"}:
        errors.extend(
            validate_predictions(args.output_dir, models, args.seed, args.prediction_batch_size)
        )
    if args.scope in {"all", "benchmark"}:
        errors.extend(validate_summary(args.output_dir, models, args.device.lower()))
    if errors:
        print(f"[VALIDATION] FAIL: {len(errors)} issue(s)")
        for error in errors[:100]:
            print(f"  - {error}")
        if len(errors) > 100:
            print(f"  - ... {len(errors) - 100} additional issue(s)")
        return 1
    print(
        f"[VALIDATION] PASS: {args.scope} outputs are complete for {len(models)} model(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
