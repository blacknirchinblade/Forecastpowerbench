#!/usr/bin/env python3
"""Plan and prepare a safe model-boundary resume of phase-2 results."""

from __future__ import annotations

import argparse
import csv
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


def _model_log_completed(output_dir: Path, model: str) -> bool:
    """Return whether a model log contains the explicit Phase-2 completion marker.
    
    Args:
        output_dir (Path): Directory containing run artifacts.
        model (str): Configuration or runtime value named model.
    
    Returns:
        bool: Whether the condition described by this callable holds.
    """
    path = output_dir / "model_logs" / f"{model}_benchmark.log"
    if not path.is_file():
        return False
    return "[COMPLETE] Finished source-directory benchmark" in path.read_text(
        errors="replace"
    )


def _model_rows_complete(
    rows: list[dict[str, str]],
    *,
    model: str,
    device: str,
    seed: int,
    dataset: str,
    contexts: list[int],
    horizons: list[int],
    batch_sizes: list[int],
    power_modes: list[str],
    repeats: int,
    benchmark_batches: int,
) -> bool:
    """Return whether model rows complete for the module.
    
    Args:
        rows (list[dict[str, str]]): Per-repeat or summary records to process.
        model (str): Configuration or runtime value named model.
        device (str): Requested execution device or device selector.
        seed (int): Root random seed used for deterministic execution.
        dataset (str): Window dataset used to construct deterministic batches.
        contexts (list[int]): Configuration or runtime value named contexts.
        horizons (list[int]): Configuration or runtime value named horizons.
        batch_sizes (list[int]): Configuration or runtime value named batch_sizes.
        power_modes (list[str]): Configuration or runtime value named power_modes.
        repeats (int): Configuration or runtime value named repeats.
        benchmark_batches (int): Configuration or runtime value named benchmark_batches.
    
    Returns:
        bool: Whether the condition described by this callable holds.
    """
    model_rows = [row for row in rows if row.get("model_name") == model]
    if not model_rows:
        return False

    for row in model_rows:
        try:
            if int(row.get("benchmark_seed", "-1")) != seed:
                return False
            if int(float(row.get("expected_repeats", "-1"))) != repeats:
                return False
            if int(float(row.get("full_batches_per_iteration", "-1"))) != benchmark_batches:
                return False
        except ValueError:
            return False
        if row.get("device_category", "").lower() != device.lower():
            return False
        if row.get("dataset_name", "") != dataset:
            return False

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
                        batch_size = int(row.get("batch_size", ""))
                    except ValueError:
                        return False
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
                        continue
                    if len(statuses) == 1 and statuses.issubset(TERMINAL_CEILING_STATUSES):
                        reached_ceiling = True
                        continue
                    return False

                # A publication run may later gain explicitly validated
                # supplemental batch sizes. They do not alter completion of the
                # configured base sweep.
    return True


def find_resume_index(args: argparse.Namespace, rows: list[dict[str, str]]) -> int:
    """Find resume index for the module.
    
    Args:
        args (argparse.Namespace): Configuration or runtime value named args.
        rows (list[dict[str, str]]): Per-repeat or summary records to process.
    
    Returns:
        int: Computed scalar described by this callable.
    """
    models = _split_csv(args.models)
    contexts = [int(value) for value in _split_csv(args.context_lens)]
    horizons = [int(value) for value in _split_csv(args.horizon_lens)]
    batch_sizes = [int(value) for value in _split_csv(args.batch_sizes)]
    power_modes = _split_csv(args.power_modes)

    for index, model in enumerate(models):
        complete = _model_log_completed(args.output_dir, model) and _model_rows_complete(
            rows,
            model=model,
            device=args.device,
            seed=args.seed,
            dataset=args.dataset,
            contexts=contexts,
            horizons=horizons,
            batch_sizes=batch_sizes,
            power_modes=power_modes,
            repeats=args.repeats,
            benchmark_batches=args.benchmark_batches,
        )
        state = "complete" if complete else "restart"
        print(f"[RESUME] {model}: {state}", file=sys.stderr)
        if not complete:
            return index
    return len(models)


def _atomic_filter_csv(path: Path, excluded_models: set[str]) -> None:
    """Atomically update filter csv for the module.
    
    Args:
        path (Path): Filesystem path consumed by the operation.
        excluded_models (set[str]): Configuration or runtime value named excluded_models.
    
    Returns:
        None; the operation updates state, validates input, or persists output in place.
    """
    fieldnames, rows = _read_rows(path)
    if not fieldnames:
        return
    retained = [row for row in rows if row.get("model_name") not in excluded_models]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(retained)
        handle.flush()
    temporary.replace(path)


def prepare_resume(args: argparse.Namespace, resume_index: int) -> None:
    """Prepare resume for the module.
    
    Args:
        args (argparse.Namespace): Configuration or runtime value named args.
        resume_index (int): Configuration or runtime value named resume_index.
    
    Returns:
        None; the operation updates state, validates input, or persists output in place.
    """
    models = _split_csv(args.models)
    if resume_index >= len(models):
        return

    affected_models = set(models[resume_index:])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = args.output_dir / "benchmark_metadata" / "resume_backups" / stamp
    backup_dir.mkdir(parents=True, exist_ok=False)

    for filename in ("benchmark_results_batch_summary.csv", "benchmark_results_raw.csv"):
        path = args.output_dir / filename
        if path.is_file():
            shutil.copy2(path, backup_dir / filename)
            _atomic_filter_csv(path, affected_models)

    log_backup_dir = backup_dir / "model_logs"
    for model in affected_models:
        log_path = args.output_dir / "model_logs" / f"{model}_benchmark.log"
        if log_path.is_file():
            log_backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(log_path, log_backup_dir / log_path.name)
            log_path.unlink()

    print(
        f"[RESUME] Backed up interrupted phase-2 state to {backup_dir}",
        file=sys.stderr,
    )
    print(
        f"[RESUME] Restarting at model {models[resume_index]}; completed earlier models are preserved.",
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
    resume_index = find_resume_index(args, rows)
    if args.prepare:
        prepare_resume(args, resume_index)
    print(resume_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
