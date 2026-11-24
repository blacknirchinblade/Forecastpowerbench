#!/usr/bin/env python3
"""Plan and prepare a non-destructive supplemental Phase-2 batch sweep."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


TERMINAL_CEILING_STATUSES = {"OOM", "TIMEOUT"}


def _split_csv(value: str) -> list[str]:
    """Split csv for the module.
    
    Args:
        value (str): Configuration or runtime value named value.
    
    Returns:
        list[str]: Result produced by this operation.
    """
    return [item.strip() for item in value.split(",") if item.strip()]


def _is_true(value: str) -> bool:
    """Return whether is true for the module.
    
    Args:
        value (str): Configuration or runtime value named value.
    
    Returns:
        bool: Whether the condition described by this callable holds.
    """
    return value.strip().lower() in {"1", "true", "yes"}


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read rows for the module.
    
    Args:
        path (Path): Filesystem path consumed by the operation.
    
    Returns:
        tuple[list[str], list[dict[str, str]]]: Result produced by this operation.
    """
    if not path.is_file():
        return [], []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _parse_int(row: dict[str, str], field: str) -> int:
    """Parse int for the module.
    
    Args:
        row (dict[str, str]): Configuration or runtime value named row.
        field (str): Configuration or runtime value named field.
    
    Returns:
        int: Computed scalar described by this callable.
    """
    return int(float(row.get(field, "")))


def _matches_requested_config(
    row: dict[str, str],
    *,
    models: set[str],
    device: str,
    dataset: str,
    contexts: set[int],
    horizons: set[int],
    batch_sizes: set[int],
    power_modes: set[str],
) -> bool:
    """Return whether a result row belongs to the requested supplemental sweep.
    
    Args:
        row (dict[str, str]): Configuration or runtime value named row.
        models (set[str]): Configuration or runtime value named models.
        device (str): Requested execution device or device selector.
        dataset (str): Window dataset used to construct deterministic batches.
        contexts (set[int]): Configuration or runtime value named contexts.
        horizons (set[int]): Configuration or runtime value named horizons.
        batch_sizes (set[int]): Configuration or runtime value named batch_sizes.
        power_modes (set[str]): Configuration or runtime value named power_modes.
    
    Returns:
        bool: Whether the condition described by this callable holds.
    """
    try:
        return (
            row.get("model_name") in models
            and row.get("device_category", "").lower() == device.lower()
            and row.get("dataset_name", "") == dataset
            and _parse_int(row, "context_len") in contexts
            and _parse_int(row, "horizon_len") in horizons
            and _parse_int(row, "batch_size") in batch_sizes
            and row.get("power_mode") in power_modes
        )
    except (TypeError, ValueError):
        return False


def _validate_protocol(
    args: argparse.Namespace,
    rows: list[dict[str, str]],
    *,
    models: list[str],
    contexts: list[int],
    horizons: list[int],
    batch_sizes: list[int],
    power_modes: list[str],
) -> None:
    """Validate protocol for the module.
    
    Args:
        args (argparse.Namespace): Configuration or runtime value named args.
        rows (list[dict[str, str]]): Per-repeat or summary records to process.
        models (list[str]): Configuration or runtime value named models.
        contexts (list[int]): Configuration or runtime value named contexts.
        horizons (list[int]): Configuration or runtime value named horizons.
        batch_sizes (list[int]): Configuration or runtime value named batch_sizes.
        power_modes (list[str]): Configuration or runtime value named power_modes.
    
    Returns:
        None; the operation updates state, validates input, or persists output in place.
    """
    if not rows:
        raise ValueError(
            "supplemental mode requires an existing non-empty benchmark summary"
        )
    requested_batches = set(batch_sizes)
    for model in models:
        model_rows = [row for row in rows if row.get("model_name") == model]
        if not model_rows:
            raise ValueError(f"{model}: no existing Phase-2 rows to supplement")

        for power_mode in power_modes:
            for context in contexts:
                for horizon in horizons:
                    config_rows = [
                        row
                        for row in model_rows
                        if row.get("power_mode") == power_mode
                        and row.get("context_len") == str(context)
                        and row.get("horizon_len") == str(horizon)
                    ]
                    if not config_rows:
                        raise ValueError(
                            f"{model}: no existing base sweep for power={power_mode}, "
                            f"ctx={context}, hor={horizon}"
                        )

                    seen_nonrequested: set[int] = set()
                    for row in config_rows:
                        try:
                            seed = _parse_int(row, "benchmark_seed")
                            repeats = _parse_int(row, "expected_repeats")
                            full_batches = _parse_int(row, "full_batches_per_iteration")
                            batch_size = _parse_int(row, "batch_size")
                        except (TypeError, ValueError) as exc:
                            raise ValueError(
                                f"{model}: malformed existing protocol row"
                            ) from exc
                        if seed != args.seed:
                            raise ValueError(
                                f"{model}: existing seed {seed} differs from {args.seed}"
                            )
                        if repeats != args.repeats:
                            raise ValueError(
                                f"{model}: existing repeats {repeats} differs from {args.repeats}"
                            )
                        if full_batches != args.benchmark_batches:
                            raise ValueError(
                                f"{model}: existing benchmark batch count {full_batches} "
                                f"differs from {args.benchmark_batches}"
                            )
                        if row.get("device_category", "").lower() != args.device.lower():
                            raise ValueError(
                                f"{model}: existing device differs from {args.device}"
                            )
                        if row.get("dataset_name", "") != args.dataset:
                            raise ValueError(
                                f"{model}: existing dataset differs from {args.dataset}"
                            )
                        if batch_size not in requested_batches:
                            if batch_size in seen_nonrequested:
                                raise ValueError(
                                    f"{model}: duplicate existing base row for batch {batch_size}"
                                )
                            seen_nonrequested.add(batch_size)


def _model_requested_rows_complete(
    rows: list[dict[str, str]],
    *,
    model: str,
    contexts: list[int],
    horizons: list[int],
    batch_sizes: list[int],
    power_modes: list[str],
) -> bool:
    """Return whether model requested rows complete for the module.
    
    Args:
        rows (list[dict[str, str]]): Per-repeat or summary records to process.
        model (str): Configuration or runtime value named model.
        contexts (list[int]): Configuration or runtime value named contexts.
        horizons (list[int]): Configuration or runtime value named horizons.
        batch_sizes (list[int]): Configuration or runtime value named batch_sizes.
        power_modes (list[str]): Configuration or runtime value named power_modes.
    
    Returns:
        bool: Whether the condition described by this callable holds.
    """
    model_rows = [row for row in rows if row.get("model_name") == model]
    for power_mode in power_modes:
        for context in contexts:
            for horizon in horizons:
                config_rows = [
                    row
                    for row in model_rows
                    if row.get("power_mode") == power_mode
                    and row.get("context_len") == str(context)
                    and row.get("horizon_len") == str(horizon)
                ]
                by_batch: dict[int, list[dict[str, str]]] = {}
                for row in config_rows:
                    try:
                        batch_size = _parse_int(row, "batch_size")
                    except (TypeError, ValueError):
                        return False
                    if batch_size in batch_sizes:
                        by_batch.setdefault(batch_size, []).append(row)

                reached_ceiling = False
                for batch_size in batch_sizes:
                    matches = by_batch.get(batch_size, [])
                    if reached_ceiling:
                        if matches:
                            return False
                        continue
                    if len(matches) != 1:
                        return False
                    row = matches[0]
                    statuses = set(filter(None, row.get("status_values", "").split("|")))
                    if statuses == {"SUCCESS"}:
                        if not _is_true(row.get("is_complete_success", "")):
                            return False
                    elif len(statuses) == 1 and statuses.issubset(
                        TERMINAL_CEILING_STATUSES
                    ):
                        reached_ceiling = True
                    else:
                        return False
    return True


def find_pending_models(
    args: argparse.Namespace, rows: list[dict[str, str]]
) -> list[str]:
    """Find pending models for the module.
    
    Args:
        args (argparse.Namespace): Configuration or runtime value named args.
        rows (list[dict[str, str]]): Per-repeat or summary records to process.
    
    Returns:
        list[str]: Result produced by this operation.
    """
    models = _split_csv(args.models)
    contexts = [int(value) for value in _split_csv(args.context_lens)]
    horizons = [int(value) for value in _split_csv(args.horizon_lens)]
    batch_sizes = [int(value) for value in _split_csv(args.batch_sizes)]
    power_modes = _split_csv(args.power_modes)
    if not batch_sizes or len(batch_sizes) != len(set(batch_sizes)):
        raise ValueError("supplemental batch sizes must be a non-empty unique list")
    if batch_sizes != sorted(batch_sizes) or any(value <= 0 for value in batch_sizes):
        raise ValueError("supplemental batch sizes must be positive and ascending")

    _validate_protocol(
        args,
        rows,
        models=models,
        contexts=contexts,
        horizons=horizons,
        batch_sizes=batch_sizes,
        power_modes=power_modes,
    )
    pending = []
    for model in models:
        complete = _model_requested_rows_complete(
            rows,
            model=model,
            contexts=contexts,
            horizons=horizons,
            batch_sizes=batch_sizes,
            power_modes=power_modes,
        )
        print(
            f"[SUPPLEMENTAL] {model}: {'already complete' if complete else 'run requested batches'}",
            file=sys.stderr,
        )
        if not complete:
            pending.append(model)
    return pending


def _atomic_filter_csv(
    path: Path,
    *,
    models: set[str],
    device: str,
    dataset: str,
    contexts: set[int],
    horizons: set[int],
    batch_sizes: set[int],
    power_modes: set[str],
) -> int:
    """Atomically update filter csv for the module.
    
    Args:
        path (Path): Filesystem path consumed by the operation.
        models (set[str]): Configuration or runtime value named models.
        device (str): Requested execution device or device selector.
        dataset (str): Window dataset used to construct deterministic batches.
        contexts (set[int]): Configuration or runtime value named contexts.
        horizons (set[int]): Configuration or runtime value named horizons.
        batch_sizes (set[int]): Configuration or runtime value named batch_sizes.
        power_modes (set[str]): Configuration or runtime value named power_modes.
    
    Returns:
        int: Computed scalar described by this callable.
    """
    fieldnames, rows = _read_rows(path)
    if not fieldnames:
        return 0
    retained = []
    removed = 0
    for row in rows:
        if _matches_requested_config(
            row,
            models=models,
            device=device,
            dataset=dataset,
            contexts=contexts,
            horizons=horizons,
            batch_sizes=batch_sizes,
            power_modes=power_modes,
        ):
            removed += 1
        else:
            retained.append(row)

    if removed == 0:
        return 0

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(retained)
        handle.flush()
    temporary.replace(path)
    return removed


def prepare_supplement(args: argparse.Namespace, pending: list[str]) -> None:
    """Prepare supplement for the module.
    
    Args:
        args (argparse.Namespace): Configuration or runtime value named args.
        pending (list[str]): Configuration or runtime value named pending.
    
    Returns:
        None; the operation updates state, validates input, or persists output in place.
    """
    if not pending:
        return
    contexts = {int(value) for value in _split_csv(args.context_lens)}
    horizons = {int(value) for value in _split_csv(args.horizon_lens)}
    batch_sizes = {int(value) for value in _split_csv(args.batch_sizes)}
    power_modes = set(_split_csv(args.power_modes))
    models = set(pending)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup_dir = args.output_dir / "benchmark_metadata" / "supplemental_backups" / stamp
    backup_dir.mkdir(parents=True, exist_ok=False)
    removed_counts: dict[str, int] = {}
    for filename in ("benchmark_results_batch_summary.csv", "benchmark_results_raw.csv"):
        path = args.output_dir / filename
        if not path.is_file():
            continue
        shutil.copy2(path, backup_dir / filename)
        removed_counts[filename] = _atomic_filter_csv(
            path,
            models=models,
            device=args.device,
            dataset=args.dataset,
            contexts=contexts,
            horizons=horizons,
            batch_sizes=batch_sizes,
            power_modes=power_modes,
        )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pending_models": pending,
        "context_lens": sorted(contexts),
        "horizon_lens": sorted(horizons),
        "batch_sizes": sorted(batch_sizes),
        "power_modes": sorted(power_modes),
        "removed_rows": removed_counts,
    }
    (backup_dir / "supplemental_prepare.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"[SUPPLEMENTAL] Backup created at {backup_dir}; only requested partial rows were cleared.",
        file=sys.stderr,
    )


def parse_args() -> argparse.Namespace:
    """Parse args for the module.
    
    Returns:
        argparse.Namespace: Result produced by this operation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--context-lens", required=True)
    parser.add_argument("--horizon-lens", required=True)
    parser.add_argument("--batch-sizes", required=True)
    parser.add_argument("--power-modes", required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--benchmark-batches", type=int, required=True)
    parser.add_argument("--prepare", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Parse command-line arguments and execute the the module workflow.
    
    Returns:
        int: Computed scalar described by this callable.
    """
    args = parse_args()
    _, rows = _read_rows(args.output_dir / "benchmark_results_batch_summary.csv")
    try:
        pending = find_pending_models(args, rows)
        if args.prepare:
            prepare_supplement(args, pending)
    except ValueError as exc:
        print(f"[SUPPLEMENTAL] Refusing unsafe merge: {exc}", file=sys.stderr)
        return 2
    print(",".join(pending))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
