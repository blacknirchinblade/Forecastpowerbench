"""Load, validate, impute, cache, window, and prebuild independent time-series sources without cross-file leakage.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)

TIME_COLUMN_CANDIDATES = ("time", "timestamp", "datetime", "date")
CACHE_SCHEMA_VERSION = "2"


@dataclass(frozen=True)
class SourceSeries:
    """One validated source file represented by compact contiguous arrays.
    
    Attributes:
        dataset_group: Dataset group state retained by each instance.
        expected_interval_ns: Expected interval ns state retained by each instance.
        observed_mask: Observed mask state retained by each instance.
        preprocessing: Preprocessing state retained by each instance.
        regular_start_timestamp_ns: Regular start timestamp ns state retained by each instance.
        segments: Segments state retained by each instance.
        source_file: Source file state retained by each instance.
        source_id: Source id state retained by each instance.
        source_path: Path of the independent source CSV being processed.
        source_relative_path: Source relative path state retained by each instance.
        target_column: Optional explicit forecast-target column name.
        time_column: Optional explicit timestamp column name.
        timestamps_ns: Timestamps ns state retained by each instance.
        values: Values to normalize, summarize, or serialize.
    """

    source_id: str
    source_file: str
    source_relative_path: str
    dataset_group: str
    source_path: str
    target_column: str
    time_column: str
    values: np.ndarray
    observed_mask: Optional[np.ndarray]
    timestamps_ns: Optional[np.ndarray]
    regular_start_timestamp_ns: Optional[int]
    segments: np.ndarray
    expected_interval_ns: Optional[int]
    preprocessing: Dict[str, object]

    def observed_at(self, indices: np.ndarray) -> np.ndarray:
        """Return whether requested source positions were originally observed rather than imputed.
        
        Args:
            indices (np.ndarray): Integer positions requested from the deterministic window or source
                                  array.
        
        Returns:
            np.ndarray: NumPy data produced by this operation.
        """
        if self.observed_mask is None:
            return np.ones(indices.shape, dtype=np.bool_)
        return self.observed_mask[indices]

    def timestamps_at(self, indices: np.ndarray) -> np.ndarray:
        """Return timestamps at requested positions while preserving source-local alignment.
        
        Args:
            indices (np.ndarray): Integer positions requested from the deterministic window or source
                                  array.
        
        Returns:
            np.ndarray: NumPy data produced by this operation.
        """
        if self.timestamps_ns is not None:
            return self.timestamps_ns[indices]
        if self.regular_start_timestamp_ns is not None and self.expected_interval_ns is not None:
            return (
                self.regular_start_timestamp_ns
                + indices.astype(np.int64, copy=False) * self.expected_interval_ns
            )
        return np.full(indices.shape, -1, dtype=np.int64)


def discover_source_files(
    data_dir: str | Path,
    pattern: str = "**/*.csv",
    excluded_names: Sequence[str] = (),
    max_sources: Optional[int] = None,
) -> List[Path]:
    """Discover source files for the module.
    
    Args:
        data_dir (str | Path): Root directory searched recursively for source CSV files.
        pattern (str): Recursive source-file glob pattern. Defaults to '**/*.csv'.
        excluded_names (Sequence[str]): Source filenames excluded from recursive discovery. Defaults
                                        to ().
        max_sources (Optional[int]): Optional maximum number of source files retained after stable
                                     sorting. Defaults to None.
    
    Returns:
        List[Path]: Result produced by this operation.
    """
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    excluded = {name.strip() for name in excluded_names if name.strip()}
    paths = []
    for path in sorted(root.glob(pattern)):
        if not path.is_file() or path.name in excluded:
            continue
        relative = path.relative_to(root)
        if any(part.startswith(".") for part in relative.parts):
            continue
        paths.append(path)
    if max_sources is not None:
        paths = paths[:max_sources]
    if not paths:
        raise FileNotFoundError(
            f"No source files matched {pattern!r} in {root}; excluded={sorted(excluded)}"
        )
    return paths


def _resolve_column(
    columns: Iterable[object], requested: Optional[str], candidates: Sequence[str]
) -> Optional[str]:
    """Resolve column for the module.
    
    Args:
        columns (Iterable[object]): Available dataset column labels considered during resolution.
        requested (Optional[str]): Explicit column name requested by the operator, when provided.
        candidates (Sequence[str]): Ordered fallback column names accepted by the resolver.
    
    Returns:
        Optional[str]: Result produced by this operation.
    """
    normalized = {str(column).strip().lower(): str(column) for column in columns}
    if requested:
        resolved = normalized.get(requested.strip().lower())
        if resolved is None:
            raise ValueError(
                f"Requested column {requested!r} was not found. Available columns: {list(columns)}"
            )
        return resolved
    return next((normalized[name] for name in candidates if name in normalized), None)


def _resolve_target_column(
    frame: pd.DataFrame, requested: Optional[str], time_column: Optional[str]
) -> str:
    """Resolve target column for the module.
    
    Args:
        frame (pd.DataFrame): Configuration or runtime value named frame.
        requested (Optional[str]): Explicit column name requested by the operator, when provided.
        time_column (Optional[str]): Optional explicit timestamp column name.
    
    Returns:
        str: Result produced by this operation.
    """
    if requested:
        resolved = _resolve_column(frame.columns, requested, ())
        assert resolved is not None
        return resolved

    aggregate = _resolve_column(frame.columns, None, ("aggregate", "target", "y", "value"))
    if aggregate is not None:
        return aggregate

    ignored = {time_column} if time_column else set()
    candidates = [str(column) for column in frame.columns if str(column) not in ignored]
    numeric_candidates = [
        column
        for column in candidates
        if pd.to_numeric(frame[column], errors="coerce").notna().any()
    ]
    if len(numeric_candidates) != 1:
        raise ValueError(
            "Could not infer one target column. Pass --target-column explicitly; "
            f"numeric candidates={numeric_candidates}"
        )
    return numeric_candidates[0]


def _segment_bounds(
    timestamps_ns: Optional[np.ndarray], row_count: int
) -> Tuple[np.ndarray, Optional[int]]:
    """Return contiguous index ranges separated by timestamp-cadence discontinuities.
    
    Args:
        timestamps_ns (Optional[np.ndarray]): Configuration or runtime value named timestamps_ns.
        row_count (int): Configuration or runtime value named row_count.
    
    Returns:
        Tuple[np.ndarray, Optional[int]]: NumPy data produced by this operation.
    """
    if timestamps_ns is None or len(timestamps_ns) < 2:
        return np.asarray([[0, row_count]], dtype=np.int64), None

    deltas = np.diff(timestamps_ns)
    positive = deltas[deltas > 0]
    if positive.size == 0:
        return np.asarray([[0, len(timestamps_ns)]], dtype=np.int64), None

    expected = int(np.median(positive))
    boundaries = np.flatnonzero(deltas != expected) + 1
    starts = np.concatenate((np.asarray([0], dtype=np.int64), boundaries.astype(np.int64)))
    ends = np.concatenate(
        (boundaries.astype(np.int64), np.asarray([len(timestamps_ns)], dtype=np.int64))
    )
    return np.column_stack((starts, ends)).astype(np.int64, copy=False), expected


def load_source_file(
    path: str | Path,
    target_column: Optional[str] = None,
    time_column: Optional[str] = None,
    dataset_root: Optional[str | Path] = None,
) -> SourceSeries:
    """Load source file for the module.
    
    Args:
        path (str | Path): Filesystem path consumed by the operation.
        target_column (Optional[str]): Optional explicit forecast-target column name. Defaults to
                                       None.
        time_column (Optional[str]): Optional explicit timestamp column name. Defaults to None.
        dataset_root (Optional[str | Path]): Configuration or runtime value named dataset_root.
                                             Defaults to None.
    
    Returns:
        SourceSeries: Result produced by this operation.
    """
    source_path = Path(path).expanduser().resolve()
    root = Path(dataset_root).expanduser().resolve() if dataset_root else source_path.parent
    try:
        relative_path = source_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Source {source_path} is outside dataset root {root}") from exc
    source_relative_path = relative_path.as_posix()
    source_id = relative_path.with_suffix("").as_posix()
    dataset_group = relative_path.parts[0] if len(relative_path.parts) > 1 else "."
    if source_path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(source_path).reset_index()
    else:
        frame = pd.read_csv(source_path)
    original_rows = len(frame)
    resolved_time = _resolve_column(frame.columns, time_column, TIME_COLUMN_CANDIDATES)
    resolved_target = _resolve_target_column(frame, target_column, resolved_time)

    target = pd.to_numeric(frame[resolved_target], errors="coerce").astype(np.float64)
    target_values = target.to_numpy(copy=True)
    target_values[~np.isfinite(target_values)] = np.nan
    observed_mask = np.isfinite(target_values)

    timestamps_ns: Optional[np.ndarray]
    invalid_time_count = 0
    duplicate_time_count = 0
    reordered = False
    if resolved_time is not None:
        timestamps = pd.to_datetime(frame[resolved_time], errors="coerce", utc=True)
        valid_time = timestamps.notna().to_numpy()
        invalid_time_count = int((~valid_time).sum())
        timestamps = timestamps[valid_time]
        target_values = target_values[valid_time]
        observed_mask = observed_mask[valid_time]

        order = np.argsort(timestamps.astype("int64").to_numpy(), kind="stable")
        reordered = not np.array_equal(order, np.arange(len(order)))
        timestamps = timestamps.iloc[order]
        target_values = target_values[order]
        observed_mask = observed_mask[order]

        timestamp_values = timestamps.astype("int64").to_numpy()
        duplicate_time_count = int(len(timestamp_values) - len(np.unique(timestamp_values)))
        if duplicate_time_count:
            duplicate_frame = pd.DataFrame(
                {"timestamp_ns": timestamp_values, "target": target_values}
            )
            grouped = duplicate_frame.groupby("timestamp_ns", sort=True, as_index=False)[
                "target"
            ].median()
            timestamp_values = grouped["timestamp_ns"].to_numpy(dtype=np.int64, copy=False)
            target_values = grouped["target"].to_numpy(dtype=np.float64, copy=False)
            observed_mask = np.isfinite(target_values)
        timestamps_ns = np.ascontiguousarray(timestamp_values, dtype=np.int64)
    else:
        timestamps_ns = None

    invalid_target_count = int(np.isnan(target_values).sum())
    finite_target = np.isfinite(target_values)
    if not finite_target.any():
        raise ValueError(f"Source {source_path} has no finite values in target {resolved_target!r}")
    median = float(np.median(target_values[finite_target]))
    target_values[~finite_target] = median
    values = np.ascontiguousarray(target_values, dtype=np.float32)

    segments, expected_interval_ns = _segment_bounds(timestamps_ns, len(values))
    regular_start_timestamp_ns: Optional[int] = None
    if (
        timestamps_ns is not None
        and timestamps_ns.size
        and expected_interval_ns is not None
        and len(segments) == 1
    ):
        regular_start_timestamp_ns = int(timestamps_ns[0])
        timestamps_ns = None

    compact_observed_mask: Optional[np.ndarray]
    if bool(observed_mask.all()):
        compact_observed_mask = None
    else:
        compact_observed_mask = np.ascontiguousarray(observed_mask, dtype=np.bool_)
    gap_count = max(0, len(segments) - 1)
    report: Dict[str, object] = {
        "source_id": source_id,
        "source_file": source_path.name,
        "source_relative_path": source_relative_path,
        "dataset_group": dataset_group,
        "source_path": str(source_path),
        "target_column": resolved_target,
        "time_column": resolved_time or "",
        "original_rows": int(original_rows),
        "valid_rows": int(len(values)),
        "invalid_time_rows_dropped": invalid_time_count,
        "duplicate_time_rows_dropped": duplicate_time_count,
        "target_values_imputed": invalid_target_count,
        "target_median": median,
        "timestamps_reordered": reordered,
        "contiguous_segments": int(len(segments)),
        "timestamp_gap_count": gap_count,
        "expected_interval_ns": expected_interval_ns if expected_interval_ns is not None else "",
    }
    logger.debug(
        "[DATA] %s: rows=%d valid=%d imputed=%d dropped_time=%d duplicates=%d segments=%d target=%s",
        source_path.name,
        original_rows,
        len(values),
        invalid_target_count,
        invalid_time_count,
        duplicate_time_count,
        len(segments),
        resolved_target,
    )
    return SourceSeries(
        source_id=source_id,
        source_file=source_path.name,
        source_relative_path=source_relative_path,
        dataset_group=dataset_group,
        source_path=str(source_path),
        target_column=resolved_target,
        time_column=resolved_time or "",
        values=values,
        observed_mask=compact_observed_mask,
        timestamps_ns=timestamps_ns,
        regular_start_timestamp_ns=regular_start_timestamp_ns,
        segments=segments,
        expected_interval_ns=expected_interval_ns,
        preprocessing=report,
    )


def _source_cache_key(
    path: Path,
    dataset_root: Path,
    target_column: Optional[str],
    time_column: Optional[str],
) -> str:
    """Hash source identity and preprocessing policy into a cache-validation key.
    
    Args:
        path (Path): Filesystem path consumed by the operation.
        dataset_root (Path): Configuration or runtime value named dataset_root.
        target_column (Optional[str]): Optional explicit forecast-target column name.
        time_column (Optional[str]): Optional explicit timestamp column name.
    
    Returns:
        str: Result produced by this operation.
    """
    stat = path.stat()
    payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "relative_path": path.relative_to(dataset_root).as_posix(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "target_column": target_column or "",
        "time_column": time_column or "",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_path(cache_dir: Path, cache_key: str) -> Path:
    """Return the deterministic NPZ cache path for one source key.
    
    Args:
        cache_dir (Path): Optional directory for validated preprocessed source caches.
        cache_key (str): Content-derived identity used to validate cached source data.
    
    Returns:
        Path: Result produced by this operation.
    """
    return cache_dir / cache_key[:2] / f"{cache_key}.npz"


def _load_cached_source(cache_path: Path, cache_key: str, source_path: Path) -> SourceSeries:
    """Load cached source for the module.
    
    Args:
        cache_path (Path): Path of the prepared source cache artifact.
        cache_key (str): Content-derived identity used to validate cached source data.
        source_path (Path): Path of the independent source CSV being processed.
    
    Returns:
        SourceSeries: Result produced by this operation.
    """
    with np.load(cache_path, allow_pickle=False) as archive:
        stored_key = str(archive["cache_key"].item())
        if stored_key != cache_key:
            raise ValueError("cache key mismatch")
        observed = archive["observed_mask"]
        timestamps = archive["timestamps_ns"]
        regular_start = int(archive["regular_start_timestamp_ns"].item())
        expected_interval = int(archive["expected_interval_ns"].item())
        preprocessing = json.loads(str(archive["preprocessing_json"].item()))
        preprocessing["source_path"] = str(source_path)
        return SourceSeries(
            source_id=str(archive["source_id"].item()),
            source_file=str(archive["source_file"].item()),
            source_relative_path=str(archive["source_relative_path"].item()),
            dataset_group=str(archive["dataset_group"].item()),
            source_path=str(source_path),
            target_column=str(archive["target_column"].item()),
            time_column=str(archive["time_column"].item()),
            values=np.ascontiguousarray(archive["values"], dtype=np.float32),
            observed_mask=(
                np.ascontiguousarray(observed, dtype=np.bool_) if observed.size else None
            ),
            timestamps_ns=(
                np.ascontiguousarray(timestamps, dtype=np.int64) if timestamps.size else None
            ),
            regular_start_timestamp_ns=(regular_start if regular_start >= 0 else None),
            segments=np.ascontiguousarray(archive["segments"], dtype=np.int64),
            expected_interval_ns=(expected_interval if expected_interval > 0 else None),
            preprocessing=preprocessing,
        )


def _save_cached_source(path: Path, cache_key: str, source: SourceSeries) -> None:
    """Persist cached source for the module.
    
    Args:
        path (Path): Filesystem path consumed by the operation.
        cache_key (str): Content-derived identity used to validate cached source data.
        source (SourceSeries): One independently validated source series.
    
    Returns:
        None; the operation updates state, validates input, or persists output in place.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez(
        temp_path,
        cache_key=np.asarray(cache_key),
        source_id=np.asarray(source.source_id),
        source_file=np.asarray(source.source_file),
        source_relative_path=np.asarray(source.source_relative_path),
        dataset_group=np.asarray(source.dataset_group),
        source_path=np.asarray(source.source_path),
        target_column=np.asarray(source.target_column),
        time_column=np.asarray(source.time_column),
        values=source.values,
        observed_mask=(
            source.observed_mask
            if source.observed_mask is not None
            else np.empty(0, dtype=np.bool_)
        ),
        timestamps_ns=(
            source.timestamps_ns
            if source.timestamps_ns is not None
            else np.empty(0, dtype=np.int64)
        ),
        regular_start_timestamp_ns=np.asarray(
            source.regular_start_timestamp_ns
            if source.regular_start_timestamp_ns is not None
            else -1,
            dtype=np.int64,
        ),
        segments=source.segments,
        expected_interval_ns=np.asarray(
            source.expected_interval_ns if source.expected_interval_ns is not None else -1,
            dtype=np.int64,
        ),
        preprocessing_json=np.asarray(json.dumps(source.preprocessing, sort_keys=True)),
    )
    temp_path.replace(path)


def load_source_directory(
    data_dir: str | Path,
    pattern: str = "**/*.csv",
    excluded_names: Sequence[str] = (),
    max_sources: Optional[int] = None,
    target_column: Optional[str] = None,
    time_column: Optional[str] = None,
    cache_dir: Optional[str | Path] = None,
) -> List[SourceSeries]:
    """Load source directory for the module.
    
    Args:
        data_dir (str | Path): Root directory searched recursively for source CSV files.
        pattern (str): Recursive source-file glob pattern. Defaults to '**/*.csv'.
        excluded_names (Sequence[str]): Source filenames excluded from recursive discovery. Defaults
                                        to ().
        max_sources (Optional[int]): Optional maximum number of source files retained after stable
                                     sorting. Defaults to None.
        target_column (Optional[str]): Optional explicit forecast-target column name. Defaults to
                                       None.
        time_column (Optional[str]): Optional explicit timestamp column name. Defaults to None.
        cache_dir (Optional[str | Path]): Optional directory for validated preprocessed source
                                          caches. Defaults to None.
    
    Returns:
        List[SourceSeries]: Result produced by this operation.
    """
    dataset_root = Path(data_dir).expanduser().resolve()
    paths = discover_source_files(dataset_root, pattern, excluded_names, max_sources)
    resolved_cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None
    sources = []
    cache_hits = 0
    total = len(paths)
    for index, path in enumerate(paths, start=1):
        source: Optional[SourceSeries] = None
        if resolved_cache_dir is not None:
            cache_key = _source_cache_key(path, dataset_root, target_column, time_column)
            source_cache_path = _cache_path(resolved_cache_dir, cache_key)
            if source_cache_path.exists():
                try:
                    source = _load_cached_source(source_cache_path, cache_key, path)
                    cache_hits += 1
                except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                    logger.warning(
                        "[DATA CACHE] Rebuilding invalid cache %s: %s", source_cache_path, exc
                    )
                    source_cache_path.unlink(missing_ok=True)
            if source is None:
                source = load_source_file(path, target_column, time_column, dataset_root)
                _save_cached_source(source_cache_path, cache_key, source)
        else:
            source = load_source_file(path, target_column, time_column, dataset_root)
        sources.append(source)
        if index == 1 or index == total or index % 250 == 0:
            logger.debug("[DATA] Preprocessed %d/%d source CSV files", index, total)
    if resolved_cache_dir is not None:
        logger.info(
            "[DATA] Prepared %d source CSV files (cache hits=%d, rebuilt=%d)",
            total,
            cache_hits,
            total - cache_hits,
        )
    else:
        logger.info("[DATA] Prepared %d source CSV files", total)
    source_ids = [source.source_id for source in sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError(f"Source filename stems must be unique: {source_ids}")
    return sources


class MultiSourceWindowDataset(Dataset):
    """Index complete windows per source first, then expose one global index.
    
    Bases:
        Dataset
    
    Attributes:
        context_len: Number of historical points supplied for each forecast window.
        horizon_len: Number of future points requested from the model.
        include_targets: Whether each dataset item includes its future target window.
        source_ids: Source ids state retained by each instance.
        source_indices: Source indices state retained by each instance.
        source_offsets: Source offsets state retained by each instance.
        sources: Validated independent source series; sources are never concatenated before windowing.
        starts: Starts state retained by each instance.
        stride: Number of source points between consecutive window starts.
        window_ids: Window ids state retained by each instance.
        windows_by_source: Windows by source state retained by each instance.
    """

    def __init__(
        self,
        sources: Sequence[SourceSeries],
        context_len: int,
        horizon_len: int,
        stride: int,
        max_windows_per_source: Optional[int] = None,
        include_targets: bool = True,
    ) -> None:
        """Initialize MultiSourceWindowDataset and capture its validated configuration or runtime state.
        
        Args:
            sources (Sequence[SourceSeries]): Validated independent source series; sources are never
                                              concatenated before windowing.
            context_len (int): Number of historical points supplied for each forecast window.
            horizon_len (int): Number of future points requested from the model.
            stride (int): Number of source points between consecutive window starts.
            max_windows_per_source (Optional[int]): Optional deterministic cap applied independently to
                                                    each source. Defaults to None.
            include_targets (bool): Whether each dataset item includes its future target window.
                                    Defaults to True.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        if context_len <= 0 or horizon_len <= 0 or stride <= 0:
            raise ValueError("context_len, horizon_len, and stride must all be positive")
        self.sources = list(sources)
        self.context_len = int(context_len)
        self.horizon_len = int(horizon_len)
        self.stride = int(stride)
        self.include_targets = bool(include_targets)
        self.source_ids = [source.source_id for source in self.sources]
        self.windows_by_source: Dict[str, int] = {}
        self.source_offsets: Dict[str, Tuple[int, int]] = {}

        source_parts: List[np.ndarray] = []
        window_parts: List[np.ndarray] = []
        start_parts: List[np.ndarray] = []
        global_offset = 0
        required = self.context_len + self.horizon_len

        for source_index, source in enumerate(self.sources):
            starts_for_source: List[np.ndarray] = []
            remaining = max_windows_per_source
            for segment_start, segment_end in source.segments:
                last_start_exclusive = int(segment_end) - required + 1
                if last_start_exclusive <= int(segment_start):
                    continue
                starts = np.arange(
                    int(segment_start), last_start_exclusive, self.stride, dtype=np.int64
                )
                if remaining is not None:
                    starts = starts[:remaining]
                    remaining -= len(starts)
                if starts.size:
                    starts_for_source.append(starts)
                if remaining == 0:
                    break

            starts = (
                np.concatenate(starts_for_source)
                if starts_for_source
                else np.empty(0, dtype=np.int64)
            )
            count = int(len(starts))
            self.windows_by_source[source.source_id] = count
            self.source_offsets[source.source_id] = (global_offset, global_offset + count)
            global_offset += count
            if count == 0:
                continue
            source_parts.append(np.full(count, source_index, dtype=np.int32))
            window_parts.append(np.arange(count, dtype=np.int64))
            start_parts.append(starts)

        self.source_indices = (
            np.concatenate(source_parts) if source_parts else np.empty(0, dtype=np.int32)
        )
        self.window_ids = (
            np.concatenate(window_parts) if window_parts else np.empty(0, dtype=np.int64)
        )
        self.starts = np.concatenate(start_parts) if start_parts else np.empty(0, dtype=np.int64)

    def __len__(self) -> int:
        """Return the number of addressable items exposed by MultiSourceWindowDataset.
        
        Returns:
            int: Computed scalar described by this callable.
        """
        return int(len(self.starts))

    def __getitem__(self, index: int):
        """Build one complete source-local context/target window and its provenance metadata.
        
        Args:
            index (int): Configuration or runtime value named index.
        
        Returns:
            Runtime value with no declared return annotation; see the implementation and callers.
        """
        source_index = int(self.source_indices[index])
        start = int(self.starts[index])
        context_end = start + self.context_len
        horizon_end = context_end + self.horizon_len
        values = self.sources[source_index].values
        target = (
            torch.from_numpy(values[context_end:horizon_end])
            if self.include_targets
            else torch.empty(0, dtype=torch.float32)
        )
        return (
            torch.from_numpy(values[start:context_end]),
            target,
            source_index,
            int(self.window_ids[index]),
            start,
            int(index),
        )

    def __getitems__(self, indices):
        """Build one ordered batch with vectorized NumPy gathers.
        
        Args:
            indices (Any): Integer positions requested from the deterministic window or source array.
        
        Returns:
            Runtime value with no declared return annotation; see the implementation and callers.
        """
        global_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if global_indices.size == 0:
            return self._empty_batch()

        source_indices = self.source_indices[global_indices]
        starts = self.starts[global_indices]
        required = self.context_len + self.horizon_len
        context_batch = np.empty((len(global_indices), self.context_len), dtype=np.float32)
        target_batch = (
            np.empty((len(global_indices), self.horizon_len), dtype=np.float32)
            if self.include_targets
            else np.empty((len(global_indices), 0), dtype=np.float32)
        )

        for source_index in np.unique(source_indices):
            output_positions = np.flatnonzero(source_indices == source_index)
            source_starts = starts[output_positions]
            values = self.sources[int(source_index)].values
            source_windows = np.lib.stride_tricks.sliding_window_view(values, required)
            context_batch[output_positions] = source_windows[source_starts, : self.context_len]
            if self.include_targets:
                target_batch[output_positions] = source_windows[source_starts, self.context_len :]

        return (
            torch.from_numpy(context_batch),
            torch.from_numpy(target_batch),
            torch.from_numpy(source_indices.astype(np.int64, copy=False)),
            torch.from_numpy(self.window_ids[global_indices]),
            torch.from_numpy(starts),
            torch.from_numpy(global_indices),
        )

    def _empty_batch(self):
        """Return correctly typed empty tensors and metadata for a zero-length vectorized request.
        
        Returns:
            Runtime value with no declared return annotation; see the implementation and callers.
        """
        return (
            torch.empty((0, self.context_len), dtype=torch.float32),
            torch.empty(
                (0, self.horizon_len if self.include_targets else 0),
                dtype=torch.float32,
            ),
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, dtype=torch.int64),
        )


def identity_window_batch(batch):
    """DataLoader hook for datasets that already return a vectorized batch.
    
    Args:
        batch (Any): Configuration or runtime value named batch.
    
    Returns:
        Runtime value with no declared return annotation; see the implementation and callers.
    """
    return batch


class PreparedBenchmarkStream:
    """Preload a bounded deterministic window pool and expose zero-copy batches.
    
    Attributes:
        available_windows: Available windows state retained by each instance.
        batch_count: Batch count state retained by each instance.
        batch_size: Number of source-local windows passed to one model call.
        inputs: Batched model context with shape `(batch, context)`.
        pool_windows: Pool windows state retained by each instance.
    """

    def __init__(
        self,
        dataset: MultiSourceWindowDataset,
        batch_size: int,
        batch_count: int,
        max_pool_windows: int = 65_536,
    ) -> None:
        """Initialize PreparedBenchmarkStream and capture its validated configuration or runtime state.
        
        Args:
            dataset (MultiSourceWindowDataset): Window dataset used to construct deterministic batches.
            batch_size (int): Number of source-local windows passed to one model call.
            batch_count (int): Configuration or runtime value named batch_count.
            max_pool_windows (int): Maximum unique windows materialized in the prepared benchmark pool.
                                    Defaults to 65536.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        if len(dataset) == 0:
            raise ValueError("Cannot prepare benchmark batches from an empty dataset")
        if batch_size <= 0 or batch_count <= 0:
            raise ValueError("batch_size and batch_count must be positive")

        self.batch_size = int(batch_size)
        self.batch_count = int(batch_count)
        self.available_windows = len(dataset)
        self.pool_windows = min(
            self.available_windows,
            max(self.batch_size, int(max_pool_windows)),
        )

        # One repeated tail makes every cyclic batch a contiguous NumPy view,
        # including batches that wrap around the end of the source window pool.
        self.inputs = np.empty(
            (self.pool_windows + self.batch_size - 1, dataset.context_len),
            dtype=np.float32,
        )
        chunk_size = min(8192, self.pool_windows)
        for start in range(0, self.pool_windows, chunk_size):
            end = min(self.pool_windows, start + chunk_size)
            contexts = dataset.__getitems__(np.arange(start, end, dtype=np.int64))[0]
            self.inputs[start:end] = contexts.numpy()

        tail = self.batch_size - 1
        copied = 0
        while copied < tail:
            count = min(self.pool_windows, tail - copied)
            destination = self.pool_windows + copied
            self.inputs[destination : destination + count] = self.inputs[:count]
            copied += count

    @property
    def repeated(self) -> bool:
        """Return whether the fixed benchmark demand exceeds the unique preloaded window pool.
        
        Returns:
            bool: Whether the condition described by this callable holds.
        """
        return self.batch_size * self.batch_count > self.pool_windows

    def batch(self, batch_index: int) -> np.ndarray:
        """Return one contiguous full batch from the prebuilt pool, wrapping deterministically when repetition is required.
        
        Args:
            batch_index (int): Zero-based index of the prepared benchmark batch.
        
        Returns:
            np.ndarray: NumPy data produced by this operation.
        """
        if not 0 <= batch_index < self.batch_count:
            raise IndexError(batch_index)
        start = (int(batch_index) * self.batch_size) % self.pool_windows
        return self.inputs[start : start + self.batch_size]
