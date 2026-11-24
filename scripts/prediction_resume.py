#!/usr/bin/env python3
"""Plan and prepare a safe model-boundary resume of prediction generation."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts.validate_results import validate_predictions
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repository root.
    from validate_results import validate_predictions


def _split_csv(value: str) -> list[str]:
    """Split csv for the module.
    
    Args:
        value (str): Configuration or runtime value named value.
    
    Returns:
        list[str]: Result produced by this operation.
    """
    return [item.strip() for item in value.split(",") if item.strip()]


def _prediction_files(output_dir: Path, model: str) -> list[Path]:
    """Return all prediction artifacts currently stored for one model.
    
    Args:
        output_dir (Path): Directory containing run artifacts.
        model (str): Configuration or runtime value named model.
    
    Returns:
        list[Path]: Result produced by this operation.
    """
    return sorted((output_dir / "benchmark_predictions" / model).rglob("*.npz"))


def _relative_prediction_files(output_dir: Path, model: str) -> set[str]:
    """Return model prediction artifact paths relative to the prediction root.
    
    Args:
        output_dir (Path): Directory containing run artifacts.
        model (str): Configuration or runtime value named model.
    
    Returns:
        set[str]: Result produced by this operation.
    """
    model_dir = output_dir / "benchmark_predictions" / model
    return {str(path.relative_to(model_dir)) for path in _prediction_files(output_dir, model)}


def _metadata_matches(
    output_dir: Path,
    model: str,
    *,
    device: str,
    seed: int,
    dataset: str,
    contexts: list[int],
    horizons: list[int],
    prediction_batch_size: int,
    dataset_sha256: str | None,
) -> tuple[bool, str | None, int]:
    """Return whether metadata matches for the module.
    
    Args:
        output_dir (Path): Directory containing run artifacts.
        model (str): Configuration or runtime value named model.
        device (str): Requested execution device or device selector.
        seed (int): Root random seed used for deterministic execution.
        dataset (str): Window dataset used to construct deterministic batches.
        contexts (list[int]): Configuration or runtime value named contexts.
        horizons (list[int]): Configuration or runtime value named horizons.
        prediction_batch_size (int): Fixed batch size used only for deterministic Phase-1 prediction
                                     generation.
        dataset_sha256 (str | None): Optional dataset fingerprint required for resume compatibility.
    
    Returns:
        tuple[bool, str | None, int]: Result produced by this operation.
    """
    path = output_dir / "benchmark_metadata" / f"{model}_prediction.json"
    try:
        metadata = json.loads(path.read_text())
        config = metadata["config"]
        source_manifest = metadata["source_manifest"]
        current_dataset_sha256 = str(source_manifest["dataset_sha256"])
        source_count = int(source_manifest["source_count"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False, dataset_sha256, 0

    expected = (
        int(metadata.get("benchmark_seed", -1)) == seed
        and config.get("models") == [model]
        and str(config.get("device_category", "")).lower() == device.lower()
        and config.get("dataset_folder") == dataset
        and [int(value) for value in config.get("context_lens", [])] == contexts
        and [int(value) for value in config.get("horizon_lens", [])] == horizons
        and int(config.get("prediction_batch_size", -1)) == prediction_batch_size
        and source_count > 0
        and bool(current_dataset_sha256)
    )
    if dataset_sha256 is not None and current_dataset_sha256 != dataset_sha256:
        expected = False
    return expected, current_dataset_sha256, source_count


def _completion_marker_present(output_dir: Path, model: str) -> bool:
    """Return whether completion marker present for the module.
    
    Args:
        output_dir (Path): Directory containing run artifacts.
        model (str): Configuration or runtime value named model.
    
    Returns:
        bool: Whether the condition described by this callable holds.
    """
    log_path = output_dir / "model_logs" / f"{model}_prediction.log"
    if not log_path.is_file():
        return False
    return "[COMPLETE] Finished prediction generation" in log_path.read_text(errors="replace")


def find_resume_index(args: argparse.Namespace) -> int:
    """Find resume index for the module.
    
    Args:
        args (argparse.Namespace): Configuration or runtime value named args.
    
    Returns:
        int: Computed scalar described by this callable.
    """
    models = _split_csv(args.models)
    contexts = [int(value) for value in _split_csv(args.context_lens)]
    horizons = [int(value) for value in _split_csv(args.horizon_lens)]
    reference_files: set[str] | None = None
    dataset_sha256: str | None = None

    for index, model in enumerate(models):
        metadata_ok, model_dataset_sha256, source_count = _metadata_matches(
            args.output_dir,
            model,
            device=args.device,
            seed=args.seed,
            dataset=args.dataset,
            contexts=contexts,
            horizons=horizons,
            prediction_batch_size=args.prediction_batch_size,
            dataset_sha256=dataset_sha256,
        )
        files = _relative_prediction_files(args.output_dir, model)
        marker_present = _completion_marker_present(args.output_dir, model)
        expected_legacy_count = source_count * len(contexts) * len(horizons)
        file_set_ok = bool(files) and (
            files == reference_files
            if reference_files is not None
            else marker_present or len(files) == expected_legacy_count
        )
        validation_errors = (
            validate_predictions(
                args.output_dir,
                [model],
                args.seed,
                args.prediction_batch_size,
            )
            if metadata_ok and file_set_ok
            else ["incomplete prediction state"]
        )
        complete = metadata_ok and file_set_ok and not validation_errors
        print(
            f"[PREDICTION RESUME] {model}: {'complete' if complete else 'restart'}",
            file=sys.stderr,
        )
        if not complete:
            return index
        if reference_files is None:
            reference_files = files
            dataset_sha256 = model_dataset_sha256
    return len(models)


def _move_if_present(path: Path, destination: Path) -> bool:
    """Move if present for the module.
    
    Args:
        path (Path): Filesystem path consumed by the operation.
        destination (Path): Configuration or runtime value named destination.
    
    Returns:
        bool: Whether the condition described by this callable holds.
    """
    if not path.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(destination))
    return True


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

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = (
        args.output_dir / "benchmark_metadata" / "prediction_resume_backups" / stamp
    )
    moved = False
    for model in models[resume_index:]:
        moved |= _move_if_present(
            args.output_dir / "benchmark_predictions" / model,
            backup_dir / "benchmark_predictions" / model,
        )
        moved |= _move_if_present(
            args.output_dir / "model_logs" / f"{model}_prediction.log",
            backup_dir / "model_logs" / f"{model}_prediction.log",
        )
        moved |= _move_if_present(
            args.output_dir / "benchmark_metadata" / f"{model}_prediction.json",
            backup_dir / "benchmark_metadata" / f"{model}_prediction.json",
        )
        work_dir = args.output_dir / ".prediction_work"
        for work_path in work_dir.glob(f"{model}_ctx_*.float32") if work_dir.is_dir() else ():
            moved |= _move_if_present(
                work_path,
                backup_dir / ".prediction_work" / work_path.name,
            )

    if moved:
        print(
            f"[PREDICTION RESUME] Backed up interrupted state to {backup_dir}",
            file=sys.stderr,
        )
    print(
        f"[PREDICTION RESUME] Restarting at model {models[resume_index]}; "
        "completed earlier predictions are preserved.",
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
    parser.add_argument("--prediction-batch-size", type=int, required=True)
    parser.add_argument("--prepare", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Parse command-line arguments and execute the the module workflow.
    
    Returns:
        int: Computed scalar described by this callable.
    """
    args = parse_args()
    resume_index = find_resume_index(args)
    if args.prepare:
        prepare_resume(args, resume_index)
    print(resume_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
