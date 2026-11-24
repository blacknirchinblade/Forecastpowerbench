"""Coordinate deterministic prediction generation and telemetry-aligned performance benchmarking.
"""

import os
import csv
import gc
import hashlib
import json
import platform
import subprocess
import sys
import time
import logging
import random
import signal
import shutil
import threading
import numpy as np
import pandas as pd
import torch
import psutil
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
from torch.utils.data import DataLoader
from pathlib import Path

from benchmark.config import (
    BenchmarkConfig,
    CPU_EXECUTION_CATEGORIES,
    DISCRETE_NVIDIA_CATEGORIES,
    ResultRow,
    ExecutionStatus,
)
from benchmark.devices.base import DeviceProfile
from benchmark.monitors.base import PowerMonitor
from benchmark.monitors.stats import summarize_power_interval
from benchmark.data import (
    MultiSourceWindowDataset,
    PreparedBenchmarkStream,
    SourceSeries,
    identity_window_batch,
    load_source_directory,
)
from models.factory import MODEL_ARTIFACTS, get_model, get_model_result_metadata

logger = logging.getLogger(__name__)
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception as exc:
    logger.warning("[DATALOADER] Could not set torch sharing strategy to file_system: %s", exc)


class ConfigDurationExceeded(TimeoutError):
    """Raised when one CPU batch-size configuration exceeds its time budget.
    
    Bases:
        TimeoutError
    """


def cuda_benchmark_enabled() -> bool:
    """Avoid probing CUDA when the selected benchmark backend is CPU.
    
    Returns:
        bool: Whether the condition described by this callable holds.
    """
    forced_device = os.environ.get("BENCHMARK_EXECUTION_DEVICE", "").strip().lower()
    if forced_device == "cpu":
        return False
    return torch.cuda.is_available()


def _is_batch_limit_runtime(exc: RuntimeError) -> bool:
    """Classify runtime failures that indicate an unsupported or memory-exhausting effective batch size.
    
    Args:
        exc (RuntimeError): Exception whose message and type are inspected.
    
    Returns:
        bool: Whether the condition described by this callable holds.
    """
    message = str(exc)
    return (
        "Efficient attention cannot produce valid seed and offset outputs" in message
        or "effective sampling batch exceeds efficient-attention limit" in message
        # Jetson does not expose desktop NVML.  The CUDA caching allocator can
        # therefore report this internal assertion instead of OutOfMemoryError
        # when a large allocation exhausts unified memory.
        or (
            "NVML_SUCCESS == r INTERNAL ASSERT FAILED" in message
            and "CUDACachingAllocator.cpp" in message
        )
    )


def seed_benchmark_rng(seed: int) -> None:
    """Seed every in-process RNG used by model construction and inference.
    
    Args:
        seed (int): Root random seed used for deterministic execution.
    
    Returns:
        None; the operation updates state, validates input, or persists output in place.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_benchmark_enabled():
        torch.cuda.manual_seed_all(seed)


def seed_data_worker(worker_id: int) -> None:
    """Derive deterministic Python and NumPy seeds for each loader worker.
    
    Args:
        worker_id (int): Zero-based DataLoader worker identifier assigned by PyTorch.
    
    Returns:
        None; the operation updates state, validates input, or persists output in place.
    """
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def configure_torch_cpu_threads() -> None:
    """Apply the benchmark's PyTorch intra-op and inter-op CPU thread settings once at startup.
    
    Returns:
        None; the operation updates state, validates input, or persists output in place.
    """
    thread_value = os.environ.get("TORCH_NUM_THREADS")
    if thread_value:
        try:
            threads = int(thread_value)
            if threads > 0:
                torch.set_num_threads(threads)
                logger.info("[CPU] PyTorch intra-op threads set to %d", threads)
        except (ValueError, RuntimeError) as exc:
            logger.warning("[CPU] Could not set TORCH_NUM_THREADS=%r: %s", thread_value, exc)

    interop_value = os.environ.get("TORCH_NUM_INTEROP_THREADS", "1")
    try:
        interop_threads = int(interop_value)
        if interop_threads > 0:
            torch.set_num_interop_threads(interop_threads)
            logger.info("[CPU] PyTorch inter-op threads set to %d", interop_threads)
    except (ValueError, RuntimeError) as exc:
        logger.warning("[CPU] Could not set TORCH_NUM_INTEROP_THREADS=%r: %s", interop_value, exc)


def process_tree_rss_gb(process: Optional[psutil.Process] = None) -> float:
    """Measure parent + child worker RSS so CPU RAM includes DataLoader workers.
    
    Args:
        process (Optional[psutil.Process]): Configuration or runtime value named process. Defaults
                                            to None.
    
    Returns:
        float: Computed scalar described by this callable.
    """
    process = process or psutil.Process()
    total = 0
    processes = [process]
    try:
        processes.extend(process.children(recursive=True))
    except psutil.Error:
        pass
    for proc in processes:
        try:
            total += proc.memory_info().rss
        except psutil.Error:
            continue
    return total / (1024**3)


class ProcessMemorySampler:
    """Background RSS sampler for transient CPU RAM peaks during inference.
    
    Attributes:
        interval_sec: Interval sec state retained by each instance.
        peak_gb: Peak gb state retained by each instance.
        process: Process state retained by each instance.
        running: Running state retained by each instance.
        thread: Thread state retained by each instance.
    """

    def __init__(self, interval_sec: float = 0.05):
        """Initialize ProcessMemorySampler and capture its validated configuration or runtime state.
        
        Args:
            interval_sec (float): Configuration or runtime value named interval_sec. Defaults to 0.05.
        
        Returns:
            None; initializes the instance in place.
        """
        self.interval_sec = interval_sec
        self.process = psutil.Process()
        self.peak_gb = 0.0
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start ProcessMemorySampler collection or lifecycle state.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        self.peak_gb = process_tree_rss_gb(self.process)
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        """Sample parent-and-worker RSS until collection is stopped, retaining the highest observation.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        while self.running:
            self.peak_gb = max(self.peak_gb, process_tree_rss_gb(self.process))
            time.sleep(self.interval_sec)

    def stop(self) -> float:
        """Stop ProcessMemorySampler collection or lifecycle state and return its final result when applicable.
        
        Returns:
            float: Computed scalar described by this callable.
        """
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.peak_gb = max(self.peak_gb, process_tree_rss_gb(self.process))
        return self.peak_gb


PROGRESS_DISABLED = (
    os.environ.get("TQDM_DISABLE", "").strip().lower() in {"1", "true", "yes", "on"}
    or not sys.stdout.isatty()
)


def live_batch_progress_enabled(is_warmup: bool) -> bool:
    """Never redraw the terminal while a measured iteration is active.
    
    Args:
        is_warmup (bool): Configuration or runtime value named is_warmup.
    
    Returns:
        bool: Whether the condition described by this callable holds.
    """
    return bool(is_warmup) and not PROGRESS_DISABLED


try:
    PROGRESS_MIN_INTERVAL = float(os.environ.get("TQDM_MININTERVAL", "2.0"))
except ValueError:
    PROGRESS_MIN_INTERVAL = 2.0
try:
    CPU_RAM_SAFETY_FRACTION = float(os.environ.get("CPU_RAM_SAFETY_FRACTION", "0.82"))
except ValueError:
    CPU_RAM_SAFETY_FRACTION = 0.82
try:
    CPU_RAM_ESTIMATE_HEADROOM = float(os.environ.get("CPU_RAM_ESTIMATE_HEADROOM", "1.08"))
except ValueError:
    CPU_RAM_ESTIMATE_HEADROOM = 1.08
try:
    GPU_MEMORY_SAFETY_FRACTION = float(
        os.environ.get("GPU_MEMORY_SAFETY_FRACTION", "0.90")
    )
except ValueError:
    GPU_MEMORY_SAFETY_FRACTION = 0.90
try:
    GPU_MEMORY_ESTIMATE_HEADROOM = float(
        os.environ.get("GPU_MEMORY_ESTIMATE_HEADROOM", "1.08")
    )
except ValueError:
    GPU_MEMORY_ESTIMATE_HEADROOM = 1.08
DEFAULT_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
SUMMARY_METRICS = [
    "measurement_duration_sec",
    "e2e_latency_ms",
    "latency_per_window_ms",
    "throughput_windows_per_sec",
    "throughput_points_per_sec",
    "model_call_duration_sec",
    "model_call_latency_per_batch_ms",
    "model_call_latency_per_window_ms",
    "model_call_throughput_windows_per_sec",
    "model_call_throughput_points_per_sec",
    "pipeline_overhead_sec",
    "pipeline_overhead_percent",
    "processed_windows",
    "processed_points",
    "measurement_cycles",
    "telemetry_duration_sec",
    "idle_baseline_w",
    "idle_gpu_compute_util_percent",
    "idle_cpu_util_percent",
    "avg_power_w",
    "power_sample_count",
    "power_std_w",
    "power_p50_w",
    "power_p95_w",
    "peak_power_w",
    "total_energy_j",
    "energy_per_window_j",
    "energy_per_1k_points_j",
    "edp_joule_seconds",
    "dynamic_avg_power_w",
    "dynamic_energy_j",
    "dynamic_energy_per_window_j",
    "dynamic_energy_per_1k_points_j",
    "dynamic_edp_joule_seconds",
    "avg_gpu_soc_power_w",
    "peak_gpu_soc_power_w",
    "avg_cpu_cv_power_w",
    "peak_cpu_cv_power_w",
    "avg_system_5v_power_w",
    "peak_system_5v_power_w",
    "jetson_oc_event_delta",
    "jetson_oc1_event_delta",
    "jetson_oc2_event_delta",
    "jetson_oc3_event_delta",
    "avg_gpu_temp_c",
    "max_gpu_temp_c",
    "avg_cpu_temp_c",
    "peak_gpu_allocated_gb",
    "peak_gpu_reserved_gb",
    "gpu_memory_total_gb",
    "avg_gpu_memory_used_gb",
    "peak_gpu_memory_used_gb",
    "min_gpu_memory_free_gb",
    "peak_cpu_ram_gb",
    "avg_gpu_compute_util_percent",
    "avg_gpu_memory_util_percent",
    "gpu_compute_util_std_percent",
    "gpu_memory_util_std_percent",
]
SUMMARY_FIELDNAMES = [
    "benchmark_seed",
    "model_name",
    "model_artifact",
    "model_variant",
    "device_category",
    "execution_device",
    "gpu_memory_scope",
    "power_scope",
    "jetson_oc_counter_available",
    "context_len",
    "horizon_len",
    "batch_size",
    "power_mode",
    "power_mode_verification",
    "power_mode_stabilization_sec",
    "dataset_name",
    "pollutant",
    "measurement_scope",
    "source_count",
    "total_windows",
    "window_stride",
    "num_workers",
    "warmup_scope",
    "warmup_batches_per_iteration",
    "warmup_windows_per_iteration",
    "full_batches_per_iteration",
    "status_values",
    "row_count_total",
    "success_count",
    "oom_count",
    "error_count",
    "swap_contaminated_count",
    "hardware_throttled_count",
    "repeat_ids_seen",
    "success_repeat_ids",
    "expected_repeats",
    "is_complete_success",
    "missing_success_repeats",
    "oom_ceiling_batch_size",
]
for _metric_name in SUMMARY_METRICS:
    SUMMARY_FIELDNAMES.extend(
        [
            f"{_metric_name}_mean",
            f"{_metric_name}_std",
            f"{_metric_name}_ci95",
        ]
    )


def progress(*args, **kwargs):
    """Construct a terminal-safe tqdm progress bar using the benchmark's common refresh and display policy.
    
    Args:
        args (Any): Additional positional arguments forwarded by the callable. Defaults to empty.
        kwargs (Any): Configuration or runtime value named kwargs. Defaults to empty.
    
    Returns:
        Runtime value with no declared return annotation; see the implementation and callers.
    """
    kwargs.setdefault("disable", PROGRESS_DISABLED)
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("bar_format", DEFAULT_BAR_FORMAT)
    kwargs.setdefault("mininterval", PROGRESS_MIN_INTERVAL)
    kwargs.setdefault("maxinterval", PROGRESS_MIN_INTERVAL)
    kwargs.setdefault("miniters", 1)
    kwargs.setdefault("file", sys.stdout)
    if "desc" in kwargs:
        terminal_columns = shutil.get_terminal_size(fallback=(120, 24)).columns
        description_limit = max(18, min(48, terminal_columns // 3))
        description = str(kwargs["desc"])
        if len(description) > description_limit:
            kwargs["desc"] = f"{description[: description_limit - 1]}…"
    if kwargs["disable"]:
        kwargs.pop("colour", None)
    return tqdm(*args, **kwargs)


def t_critical_95(sample_count: int) -> float:
    """Two-sided 95% Student-t critical value for small repeat counts.
    
    Args:
        sample_count (int): Configuration or runtime value named sample_count.
    
    Returns:
        float: Computed scalar described by this callable.
    """
    critical_by_df = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        25: 2.060,
        30: 2.042,
    }
    degrees_of_freedom = max(1, sample_count - 1)
    if degrees_of_freedom in critical_by_df:
        return critical_by_df[degrees_of_freedom]
    lower = max((df for df in critical_by_df if df < degrees_of_freedom), default=30)
    return critical_by_df[lower] if degrees_of_freedom <= 30 else 1.96


def clear_gpu_memory():
    """Release Python and CUDA caches between isolated configurations without entering a measured region.
    
    Returns:
        None; the callable operates through state changes, validation, or persistence.
    """
    if cuda_benchmark_enabled():
        # A failed Jetson allocation can leave one CUDA housekeeping call in an
        # error state. Cleanup must not turn a recoverable batch ceiling into a
        # fatal benchmark error.
        for cleanup in (
            torch.cuda.synchronize,
            torch.cuda.empty_cache,
            torch.cuda.reset_peak_memory_stats,
        ):
            try:
                cleanup()
            except RuntimeError as exc:
                logger.warning("[CUDA] Best-effort cleanup skipped: %s", exc)
    gc.collect()


class BenchmarkRunner:
    """Coordinate data preparation, prediction generation, device control, timed inference, telemetry, aggregation, and persistence.
    
    Attributes:
        batch_summary_csv_path: Batch summary csv path state retained by each instance.
        benchmark_seed: Benchmark seed state retained by each instance.
        config: Validated benchmark protocol and output configuration.
        cpu_batch_memory_peaks: Cpu batch memory peaks state retained by each instance.
        csv_path: Csv path state retained by each instance.
        current_power_mode: Current power mode state retained by each instance.
        current_power_mode_verification: Current power mode verification state retained by each instance.
        device_profile: Hardware profile responsible for setup, verification, and restoration.
        gpu_batch_memory_peaks: Gpu batch memory peaks state retained by each instance.
        idle_baseline_w: Idle baseline w state retained by each instance.
        idle_cpu_util_percent: Idle cpu util percent state retained by each instance.
        idle_gpu_util_percent: Idle gpu util percent state retained by each instance.
        output_dir: Directory containing run artifacts.
        power_monitor: Telemetry backend sampled around measured repeats.
        preprocessing_csv_path: Preprocessing csv path state retained by each instance.
        swap_baseline_bytes: Swap baseline bytes state retained by each instance.
    """
    def __init__(
        self, config: BenchmarkConfig, device_profile: DeviceProfile, power_monitor: PowerMonitor
    ):
        """Initialize BenchmarkRunner and capture its validated configuration or runtime state.
        
        Args:
            config (BenchmarkConfig): Validated benchmark protocol and output configuration.
            device_profile (DeviceProfile): Hardware profile responsible for setup, verification, and
                                            restoration.
            power_monitor (PowerMonitor): Telemetry backend sampled around measured repeats.
        
        Returns:
            None; initializes the instance in place.
        """
        configure_torch_cpu_threads()
        self.config = config
        self.device_profile = device_profile
        self.power_monitor = power_monitor
        self.current_power_mode = "N/A"
        self.current_power_mode_verification = "NOT_VERIFIED"
        self.benchmark_seed = int(config.benchmark_seed)
        os.environ["BENCHMARK_SEED"] = str(self.benchmark_seed)

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "benchmark_results_raw.csv"
        self.batch_summary_csv_path = self.output_dir / "benchmark_results_batch_summary.csv"
        self.preprocessing_csv_path = self.output_dir / "dataset_preprocessing_report.csv"

        if self.config.save_raw_performance_metrics:
            if not self.csv_path.exists():
                pd.DataFrame(columns=list(ResultRow.model_fields.keys())).to_csv(
                    self.csv_path, index=False
                )
            else:
                existing_df = pd.read_csv(self.csv_path, nrows=0)
                expected_columns = list(ResultRow.model_fields.keys())
                missing_columns = [
                    column for column in expected_columns if column not in existing_df.columns
                ]
                if missing_columns:
                    full_df = pd.read_csv(self.csv_path)
                    for column in missing_columns:
                        full_df[column] = ResultRow.model_fields[column].default
                    full_df = full_df.reindex(columns=expected_columns)
                    full_df.to_csv(self.csv_path, index=False)
                    logger.info(
                        "[RESULTS] Added missing result columns to %s: %s",
                        self.csv_path,
                        missing_columns,
                    )
        else:
            logger.info(
                "[RESULTS] Per-repeat raw performance CSV disabled; batch summaries remain enabled."
            )

        if self.config.phase == "benchmark":
            self._initialize_dict_csv(self.batch_summary_csv_path, SUMMARY_FIELDNAMES)

        self.idle_baseline_w = float("nan")
        self.idle_gpu_util_percent = float("nan")
        self.idle_cpu_util_percent = float("nan")
        self.swap_baseline_bytes = 0
        self.cpu_batch_memory_peaks: Dict[Tuple[str, int, int], List[Tuple[int, float]]] = {}
        self.gpu_batch_memory_peaks: Dict[Tuple[str, int, int], List[Tuple[int, float]]] = {}

    def _load_model_for_config(
        self,
        *,
        model_name: str,
        device: str,
        context_len: int,
        horizon_len: int,
        batch_size: int,
        num_workers: int,
    ):
        """Construct one deterministic model, retrying a known transient MOMENT loader failure.

        The retry runs before stabilization, idle-baseline sampling, warmups, and
        measured inference, so it cannot alter benchmark timing or telemetry.
        """
        retry_count = (
            max(0, int(os.environ.get("MOMENT_MODEL_LOAD_RETRIES", "2")))
            if model_name == "moment"
            else 0
        )
        retry_delay_sec = max(
            0.0,
            float(os.environ.get("MOMENT_MODEL_LOAD_RETRY_DELAY_SEC", "2")),
        )

        for attempt in range(retry_count + 1):
            seed_benchmark_rng(self.benchmark_seed)
            try:
                return get_model(
                    model_name,
                    device=device,
                    context_len=context_len,
                    horizon_len=horizon_len,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
            except TypeError as exc:
                is_known_transient = (
                    model_name == "moment"
                    and "unsupported operand type(s) for /: 'cell' and 'bool'"
                    in str(exc)
                )
                if not is_known_transient or attempt >= retry_count:
                    raise
                logger.warning(
                    "[MODEL LOAD RETRY] MOMENT initialization hit a transient "
                    "PyTorch runtime error for ctx=%d hor=%d bs=%d "
                    "(attempt %d/%d): %s",
                    context_len,
                    horizon_len,
                    batch_size,
                    attempt + 1,
                    retry_count + 1,
                    exc,
                )
                clear_gpu_memory()
                if retry_delay_sec:
                    time.sleep(retry_delay_sec)

        raise RuntimeError("Model construction retry loop ended unexpectedly.")

    @staticmethod
    def _initialize_dict_csv(path: Path, fieldnames: List[str]) -> None:
        """Create a schema-only CSV atomically when the requested result file does not yet exist.
        
        Args:
            path (Path): Filesystem path consumed by the operation.
            fieldnames (List[str]): Configuration or runtime value named fieldnames.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        if path.exists():
            existing_df = pd.read_csv(path, nrows=0)
            missing_columns = [column for column in fieldnames if column not in existing_df.columns]
            if missing_columns:
                full_df = pd.read_csv(path)
                for column in missing_columns:
                    full_df[column] = ""
                full_df = full_df.reindex(columns=fieldnames)
                full_df.to_csv(path, index=False)
                logger.info(
                    "[RESULTS] Added missing result columns to %s: %s", path, missing_columns
                )
            return
        with path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

    @staticmethod
    def _append_dict_rows(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
        """Append schema-aligned records to a CSV with one flush after a completed configuration.
        
        Args:
            path (Path): Filesystem path consumed by the operation.
            fieldnames (List[str]): Configuration or runtime value named fieldnames.
            rows (List[Dict[str, Any]]): Per-repeat or summary records to process.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        if not rows:
            return

        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fieldnames})

    def _cpu_ram_limit_gb(self) -> float:
        """Return the configured safe process-tree RAM ceiling for CPU execution.
        
        Returns:
            float: Computed scalar described by this callable.
        """
        configured = os.environ.get("CPU_RAM_HARD_LIMIT_GB")
        if configured:
            try:
                value = float(configured)
                if value > 0:
                    return value
            except ValueError:
                logger.warning(
                    "[CPU RAM] Invalid CPU_RAM_HARD_LIMIT_GB=%r; using system RAM fraction.",
                    configured,
                )
        total_gb = psutil.virtual_memory().total / (1024**3)
        return max(1.0, total_gb * CPU_RAM_SAFETY_FRACTION)

    def _estimate_cpu_peak_gb(
        self, key: Tuple[str, int, int], next_batch_size: int
    ) -> Optional[float]:
        """Estimate cpu peak gb for BenchmarkRunner.
        
        Args:
            key (Tuple[str, int, int]): Configuration or runtime value named key.
            next_batch_size (int): Configuration or runtime value named next_batch_size.
        
        Returns:
            Optional[float]: Result produced by this operation.
        """
        observations = sorted(self.cpu_batch_memory_peaks.get(key, []), key=lambda item: item[0])
        observations = [(old_bs, peak) for old_bs, peak in observations if old_bs > 0 and peak > 0]
        if not observations:
            return None
        if len(observations) == 1:
            prev_bs, prev_peak = observations[-1]
            if next_batch_size <= prev_bs:
                return prev_peak
            return prev_peak * (next_batch_size / prev_bs) * CPU_RAM_ESTIMATE_HEADROOM

        prev_bs, prev_peak = observations[-1]
        prior_bs, prior_peak = observations[-2]
        if next_batch_size <= prev_bs:
            return prev_peak
        if prev_bs == prior_bs:
            return prev_peak * (next_batch_size / prev_bs) * CPU_RAM_ESTIMATE_HEADROOM

        slope = max(0.0, (prev_peak - prior_peak) / (prev_bs - prior_bs))
        baseline = max(0.0, prev_peak - slope * prev_bs)
        projected = baseline + slope * next_batch_size
        return max(projected, prev_peak) * CPU_RAM_ESTIMATE_HEADROOM

    def _should_skip_cpu_batch_for_ram(
        self, model_name: str, ctx: int, hor: int, bs: int
    ) -> Tuple[bool, Optional[float], float]:
        """Determine whether to skip cpu batch for ram for BenchmarkRunner.
        
        Args:
            model_name (str): Registered model adapter name.
            ctx (int): Context length for the current configuration.
            hor (int): Forecast horizon for the current configuration.
            bs (int): Configuration or runtime value named bs.
        
        Returns:
            Tuple[bool, Optional[float], float]: Result produced by this operation.
        """
        if self._execution_device() != "cpu":
            return False, None, self._cpu_ram_limit_gb()
        key = (model_name, int(ctx), int(hor))
        estimated_peak = self._estimate_cpu_peak_gb(key, int(bs))
        limit_gb = self._cpu_ram_limit_gb()
        if estimated_peak is None:
            return False, None, limit_gb
        return estimated_peak >= limit_gb, estimated_peak, limit_gb

    def _remember_cpu_peak(
        self, model_name: str, ctx: int, hor: int, bs: int, peak_gb: Optional[float]
    ) -> None:
        """Retain a successful CPU peak-RSS observation for the next-batch memory projection.
        
        Args:
            model_name (str): Registered model adapter name.
            ctx (int): Context length for the current configuration.
            hor (int): Forecast horizon for the current configuration.
            bs (int): Configuration or runtime value named bs.
            peak_gb (Optional[float]): Configuration or runtime value named peak_gb.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        if (
            self._execution_device() != "cpu"
            or peak_gb is None
            or not np.isfinite(peak_gb)
            or peak_gb <= 0
        ):
            return
        key = (model_name, int(ctx), int(hor))
        observations = [
            (old_bs, old_peak)
            for old_bs, old_peak in self.cpu_batch_memory_peaks.get(key, [])
            if old_bs != int(bs)
        ]
        observations.append((int(bs), float(peak_gb)))
        self.cpu_batch_memory_peaks[key] = sorted(observations, key=lambda item: item[0])[-4:]

    def _gpu_memory_limit_gb(self) -> float:
        """Return the safe accelerator-memory ceiling after applying configured headroom.
        
        Returns:
            float: Computed scalar described by this callable.
        """
        configured = os.environ.get("GPU_MEMORY_HARD_LIMIT_GB")
        if configured:
            try:
                value = float(configured)
                if value > 0:
                    return value
            except ValueError:
                logger.warning(
                    "[GPU MEMORY] Invalid GPU_MEMORY_HARD_LIMIT_GB=%r; using device fraction.",
                    configured,
                )
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return max(1.0, total_gb * GPU_MEMORY_SAFETY_FRACTION)

    def _estimate_gpu_peak_gb(
        self, key: Tuple[str, int, int], next_batch_size: int
    ) -> Optional[float]:
        """Estimate gpu peak gb for BenchmarkRunner.
        
        Args:
            key (Tuple[str, int, int]): Configuration or runtime value named key.
            next_batch_size (int): Configuration or runtime value named next_batch_size.
        
        Returns:
            Optional[float]: Result produced by this operation.
        """
        observations = sorted(self.gpu_batch_memory_peaks.get(key, []), key=lambda item: item[0])
        observations = [(old_bs, peak) for old_bs, peak in observations if old_bs > 0 and peak > 0]
        if not observations:
            return None
        if len(observations) == 1:
            prev_bs, prev_peak = observations[-1]
            if next_batch_size <= prev_bs:
                return prev_peak
            return prev_peak * (next_batch_size / prev_bs) * GPU_MEMORY_ESTIMATE_HEADROOM

        prev_bs, prev_peak = observations[-1]
        prior_bs, prior_peak = observations[-2]
        if next_batch_size <= prev_bs:
            return prev_peak
        if prev_bs == prior_bs:
            return prev_peak * (next_batch_size / prev_bs) * GPU_MEMORY_ESTIMATE_HEADROOM

        slope = max(0.0, (prev_peak - prior_peak) / (prev_bs - prior_bs))
        baseline = max(0.0, prev_peak - slope * prev_bs)
        projected = baseline + slope * next_batch_size
        return max(projected, prev_peak) * GPU_MEMORY_ESTIMATE_HEADROOM

    def _should_skip_gpu_batch_for_memory(
        self, model_name: str, ctx: int, hor: int, bs: int
    ) -> Tuple[bool, Optional[float], Optional[float]]:
        """Determine whether to skip gpu batch for memory for BenchmarkRunner.
        
        Args:
            model_name (str): Registered model adapter name.
            ctx (int): Context length for the current configuration.
            hor (int): Forecast horizon for the current configuration.
            bs (int): Configuration or runtime value named bs.
        
        Returns:
            Tuple[bool, Optional[float], Optional[float]]: Result produced by this operation.
        """
        if self._execution_device() != "cuda" or not torch.cuda.is_available():
            return False, None, None
        key = (model_name, int(ctx), int(hor))
        estimated_peak = self._estimate_gpu_peak_gb(key, int(bs))
        limit_gb = self._gpu_memory_limit_gb()
        if estimated_peak is None:
            return False, None, limit_gb
        return estimated_peak >= limit_gb, estimated_peak, limit_gb

    def _remember_gpu_peak(
        self, model_name: str, ctx: int, hor: int, bs: int, peak_gb: Optional[float]
    ) -> None:
        """Retain a successful CUDA peak-reserved observation for the next-batch memory projection.
        
        Args:
            model_name (str): Registered model adapter name.
            ctx (int): Context length for the current configuration.
            hor (int): Forecast horizon for the current configuration.
            bs (int): Configuration or runtime value named bs.
            peak_gb (Optional[float]): Configuration or runtime value named peak_gb.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        if (
            self._execution_device() != "cuda"
            or peak_gb is None
            or not np.isfinite(peak_gb)
            or peak_gb <= 0
        ):
            return
        key = (model_name, int(ctx), int(hor))
        observations = [
            (old_bs, old_peak)
            for old_bs, old_peak in self.gpu_batch_memory_peaks.get(key, [])
            if old_bs != int(bs)
        ]
        observations.append((int(bs), float(peak_gb)))
        self.gpu_batch_memory_peaks[key] = sorted(observations, key=lambda item: item[0])[-4:]

    @staticmethod
    def _shutdown_dataloader_workers(loader: DataLoader) -> None:
        """Shut down persistent DataLoader workers explicitly so model processes release resources promptly.
        
        Args:
            loader (DataLoader): Configuration or runtime value named loader.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        iterator = getattr(loader, "_iterator", None)
        if iterator is None:
            return

        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            try:
                shutdown()
            except Exception as exc:
                logger.warning("[DATALOADER] Worker shutdown warning: %s", exc)
        try:
            loader._iterator = None
        except Exception:
            pass

    @staticmethod
    def _metadata_to_numpy(values: Any) -> np.ndarray:
        """Convert DataLoader-collated metadata into a flat int64 array.
        
        Args:
            values (Any): Values to normalize, summarize, or serialize.
        
        Returns:
            np.ndarray: NumPy data produced by this operation.
        """
        if torch.is_tensor(values):
            return values.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
        if isinstance(values, dict):
            if len(values) == 1:
                return BenchmarkRunner._metadata_to_numpy(next(iter(values.values())))
            ordered_values = [values[key] for key in sorted(values)]
            return np.asarray(ordered_values, dtype=np.int64).reshape(-1)
        return np.asarray(values, dtype=np.int64).reshape(-1)

    @staticmethod
    def _format_repeat_ids(values: pd.Series) -> str:
        """Format repeat ids for BenchmarkRunner.
        
        Args:
            values (pd.Series): Values to normalize, summarize, or serialize.
        
        Returns:
            str: Result produced by this operation.
        """
        clean_values = values.dropna().astype(int).tolist()
        if not clean_values:
            return ""
        return "|".join(str(value) for value in sorted(set(clean_values)))

    @staticmethod
    def _format_missing_repeats(success_values: pd.Series, expected_repeats: int) -> str:
        """Format missing repeats for BenchmarkRunner.
        
        Args:
            success_values (pd.Series): Configuration or runtime value named success_values.
            expected_repeats (int): Number of measured repeats required by the protocol.
        
        Returns:
            str: Result produced by this operation.
        """
        seen = set(success_values.dropna().astype(int).tolist())
        missing = [repeat_id for repeat_id in range(expected_repeats) if repeat_id not in seen]
        return "|".join(str(value) for value in missing)

    def _summarize_batch_rows(self, rows: List[ResultRow]) -> Optional[Dict[str, Any]]:
        """Summarize batch rows for BenchmarkRunner.
        
        Args:
            rows (List[ResultRow]): Per-repeat or summary records to process.
        
        Returns:
            Optional[Dict[str, Any]]: Result produced by this operation.
        """
        if not rows:
            return None

        df = pd.DataFrame([row.model_dump(mode="json") for row in rows])
        first_row = df.iloc[0]
        status_series = df["status"].map(
            lambda value: value.value if isinstance(value, ExecutionStatus) else str(value)
        )
        success_df = df[status_series == ExecutionStatus.SUCCESS.value].copy()

        summary_row: Dict[str, Any] = {
            "benchmark_seed": int(first_row.get("benchmark_seed", 12345)),
            "model_name": first_row["model_name"],
            "model_artifact": first_row.get("model_artifact", ""),
            "model_variant": first_row.get("model_variant", ""),
            "device_category": first_row["device_category"],
            "execution_device": first_row.get("execution_device", ""),
            "gpu_memory_scope": first_row.get("gpu_memory_scope", ""),
            "power_scope": first_row.get("power_scope", "unavailable"),
            "jetson_oc_counter_available": bool(
                first_row.get("jetson_oc_counter_available", False)
            ),
            "context_len": int(first_row["context_len"]),
            "horizon_len": int(first_row["horizon_len"]),
            "batch_size": int(first_row["batch_size"]),
            "power_mode": first_row["power_mode"],
            "power_mode_verification": first_row.get("power_mode_verification", "NOT_VERIFIED"),
            "power_mode_stabilization_sec": float(
                first_row.get("power_mode_stabilization_sec", 0.0)
            ),
            "dataset_name": first_row["dataset_name"],
            "pollutant": first_row.get("pollutant", ""),
            "measurement_scope": first_row["measurement_scope"],
            "source_count": int(first_row["source_count"]),
            "total_windows": int(first_row["total_windows"]),
            "window_stride": int(first_row["window_stride"]),
            "num_workers": int(first_row["num_workers"]),
            "warmup_scope": first_row.get("warmup_scope", "full"),
            "warmup_batches_per_iteration": int(first_row.get("warmup_batches_per_iteration", 0)),
            "warmup_windows_per_iteration": int(first_row.get("warmup_windows_per_iteration", 0)),
            "full_batches_per_iteration": int(first_row.get("full_batches_per_iteration", 0)),
            "status_values": "|".join(sorted(status_series.dropna().unique())),
            "row_count_total": int(len(df)),
            "success_count": int((status_series == ExecutionStatus.SUCCESS.value).sum()),
            "oom_count": int((status_series == ExecutionStatus.OOM.value).sum()),
            "error_count": int((status_series == ExecutionStatus.ERROR.value).sum()),
            "swap_contaminated_count": int(
                (status_series == ExecutionStatus.SWAP_CONTAMINATED.value).sum()
            ),
            "hardware_throttled_count": int(
                (status_series == ExecutionStatus.HARDWARE_THROTTLED.value).sum()
            ),
            "expected_repeats": int(self.config.repeats),
            "oom_ceiling_batch_size": "",
        }

        repeat_series = pd.to_numeric(df["repeat_id"], errors="coerce")
        success_repeat_series = pd.to_numeric(success_df["repeat_id"], errors="coerce")
        summary_row["repeat_ids_seen"] = self._format_repeat_ids(repeat_series)
        summary_row["success_repeat_ids"] = self._format_repeat_ids(success_repeat_series)
        summary_row["is_complete_success"] = bool(
            summary_row["success_count"] == self.config.repeats
        )
        summary_row["missing_success_repeats"] = self._format_missing_repeats(
            success_repeat_series, self.config.repeats
        )

        if "oom_ceiling_batch_size" in df.columns:
            oom_values = pd.to_numeric(df["oom_ceiling_batch_size"], errors="coerce").dropna()
            if not oom_values.empty:
                summary_row["oom_ceiling_batch_size"] = int(oom_values.min())

        for metric in SUMMARY_METRICS:
            metric_mean_key = f"{metric}_mean"
            metric_std_key = f"{metric}_std"
            metric_ci95_key = f"{metric}_ci95"

            if metric not in success_df.columns or success_df.empty:
                summary_row[metric_mean_key] = float("nan")
                summary_row[metric_std_key] = float("nan")
                summary_row[metric_ci95_key] = float("nan")
                continue

            values = pd.to_numeric(success_df[metric], errors="coerce").dropna()
            if values.empty:
                summary_row[metric_mean_key] = float("nan")
                summary_row[metric_std_key] = float("nan")
                summary_row[metric_ci95_key] = float("nan")
                continue

            summary_row[metric_mean_key] = float(values.mean())
            if len(values) > 1:
                std_value = float(values.std(ddof=1))
                ci95_value = float(t_critical_95(len(values)) * std_value / np.sqrt(len(values)))
            else:
                std_value = 0.0
                ci95_value = 0.0
            summary_row[metric_std_key] = std_value
            summary_row[metric_ci95_key] = ci95_value

        return summary_row

    @staticmethod
    def _align_monitor_metrics(
        metrics: Dict[str, Any], start_ns: int, end_ns: int
    ) -> Dict[str, Any]:
        """Restrict telemetry statistics to the exact timed wall-clock interval.
        
        Args:
            metrics (Dict[str, Any]): Configuration or runtime value named metrics.
            start_ns (int): Inclusive measurement start timestamp in nanoseconds.
            end_ns (int): Inclusive measurement end timestamp in nanoseconds.
        
        Returns:
            Dict[str, Any]: Named values produced by this operation.
        """
        aligned = dict(metrics)
        rows = list(metrics.get("telemetry_samples", []))
        if not rows:
            aligned.pop("telemetry_samples", None)
            return aligned

        power_rows = [row for row in rows if np.isfinite(float(row.get("power_w", float("nan"))))]
        if power_rows:
            aligned.update(
                summarize_power_interval(
                    [float(row["power_w"]) for row in power_rows],
                    [int(row["timestamp_ns"]) for row in power_rows],
                    start_ns,
                    end_ns,
                )
            )

        in_window = [row for row in rows if start_ns <= int(row.get("timestamp_ns", -1)) <= end_ns]
        for index, row in enumerate(in_window):
            row["sample_index"] = index
            row["elapsed_ms"] = (int(row["timestamp_ns"]) - start_ns) / 1e6
        metric_specs = {
            "avg_gpu_compute_util_percent": ("gpu_compute_util_percent", "mean"),
            "gpu_compute_util_std_percent": ("gpu_compute_util_percent", "std"),
            "avg_gpu_memory_util_percent": ("gpu_memory_util_percent", "mean"),
            "gpu_memory_util_std_percent": ("gpu_memory_util_percent", "std"),
            "gpu_memory_total_gb": ("gpu_memory_total_gb", "mean"),
            "avg_gpu_memory_used_gb": ("gpu_memory_used_gb", "mean"),
            "peak_gpu_memory_used_gb": ("gpu_memory_used_gb", "max"),
            "min_gpu_memory_free_gb": ("gpu_memory_free_gb", "min"),
            "avg_gpu_temp_c": ("gpu_temp_c", "mean"),
            "max_gpu_temp_c": ("gpu_temp_c", "max"),
            "avg_cpu_temp_c": ("cpu_temp_c", "mean"),
            "avg_gpu_soc_power_w": ("gpu_soc_power_w", "mean"),
            "peak_gpu_soc_power_w": ("gpu_soc_power_w", "max"),
            "avg_cpu_cv_power_w": ("cpu_cv_power_w", "mean"),
            "peak_cpu_cv_power_w": ("cpu_cv_power_w", "max"),
            "avg_system_5v_power_w": ("system_5v_power_w", "mean"),
            "peak_system_5v_power_w": ("system_5v_power_w", "max"),
        }
        reducers = {"mean": np.mean, "std": np.std, "max": np.max, "min": np.min}
        for metric_name, (sample_name, reducer_name) in metric_specs.items():
            values = np.asarray(
                [row.get(sample_name, float("nan")) for row in in_window],
                dtype=np.float64,
            )
            values = values[np.isfinite(values)]
            aligned[metric_name] = (
                float(reducers[reducer_name](values)) if values.size else float("nan")
            )
        aligned.pop("telemetry_samples", None)
        return aligned

    def _execution_device(self) -> str:
        """Return the concrete adapter device string selected for the current hardware category.
        
        Returns:
            str: Result produced by this operation.
        """
        return "cpu" if self.config.device_category in CPU_EXECUTION_CATEGORIES else "cuda"

    @staticmethod
    def _safe_path_part(value: str) -> str:
        """Sanitize arbitrary metadata into one stable filesystem path component.
        
        Args:
            value (str): Configuration or runtime value named value.
        
        Returns:
            str: Result produced by this operation.
        """
        safe_value = "".join(
            char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(value)
        ).strip("._")
        return safe_value or "unknown"

    def _prediction_npz_path(
        self, model_name: str, source: SourceSeries, ctx: int, hor: int
    ) -> Path:
        """Construct the model/source/context/horizon path for one Phase-1 prediction artifact.
        
        Args:
            model_name (str): Registered model adapter name.
            source (SourceSeries): One independently validated source series.
            ctx (int): Context length for the current configuration.
            hor (int): Forecast horizon for the current configuration.
        
        Returns:
            Path: Result produced by this operation.
        """
        relative = Path(source.source_relative_path)
        safe_parent = [self._safe_path_part(part) for part in relative.parent.parts if part != "."]
        source_dir = self.output_dir / "benchmark_predictions" / self._safe_path_part(model_name)
        if safe_parent:
            source_dir = source_dir.joinpath(*safe_parent)
        source_dir = source_dir / self._safe_path_part(relative.stem)
        source_dir.mkdir(parents=True, exist_ok=True)
        return source_dir / f"ctx_{ctx}_hor_{hor}.npz"

    def _measure_idle_baseline(self) -> float:
        """Measure idle power baseline with a visible countdown. The monitor actively samples during the entire duration.
        
        Returns:
            float: Computed scalar described by this callable.
        """
        duration = self.config.idle_baseline_duration_sec
        logger.info(f"[IDLE] Measuring idle baseline power for {duration}s...")
        psutil.cpu_percent(interval=None)
        self.power_monitor.start()
        self._countdown_sleep(duration, desc="Idle baseline")
        idle_metrics = self.power_monitor.stop()
        self.idle_cpu_util_percent = float(psutil.cpu_percent(interval=None))
        self.idle_gpu_util_percent = float(
            idle_metrics.get("avg_gpu_compute_util_percent", float("nan"))
        )
        baseline = float(idle_metrics.get("avg_power_w", float("nan")))
        if np.isfinite(baseline):
            logger.info("[IDLE] Baseline power: %.2fW", baseline)
        else:
            logger.warning("[IDLE] Baseline power unavailable; dynamic energy fields will be NaN.")
        logger.info(
            "[IDLE] Utilization: host CPU %.1f%%, GPU %s",
            self.idle_cpu_util_percent,
            (
                f"{self.idle_gpu_util_percent:.1f}%"
                if np.isfinite(self.idle_gpu_util_percent)
                else "N/A"
            ),
        )
        if bool(idle_metrics.get("hardware_throttled", False)):
            raise RuntimeError(
                "Hardware throttling was active or newly recorded during the idle baseline; "
                "cool the device and correct its power/cooling setup before benchmarking."
            )
        if self.idle_cpu_util_percent > self.config.max_idle_cpu_util_percent:
            raise RuntimeError(
                f"Idle host CPU utilization {self.idle_cpu_util_percent:.1f}% exceeds "
                f"the {self.config.max_idle_cpu_util_percent:.1f}% contamination limit."
            )
        if (
            self._execution_device() == "cuda"
            and np.isfinite(self.idle_gpu_util_percent)
            and self.idle_gpu_util_percent > self.config.max_idle_gpu_util_percent
        ):
            raise RuntimeError(
                f"Idle GPU utilization {self.idle_gpu_util_percent:.1f}% exceeds "
                f"the {self.config.max_idle_gpu_util_percent:.1f}% contamination limit."
            )
        return baseline

    def _check_swap_usage(self, baseline_bytes: int, context: str = "run") -> bool:
        """Check if swap usage has increased since baseline. Returns True if swap rose.
        
        Args:
            baseline_bytes (int): Swap usage captured before the measured operation.
            context (str): Configuration or runtime value named context. Defaults to 'run'.
        
        Returns:
            bool: Whether the condition described by this callable holds.
        """
        swap = psutil.swap_memory()
        current_swap = swap.used
        if current_swap > baseline_bytes + (100 * 1024 * 1024):
            logger.warning(
                "[SWAP] Swap usage increased by %.2fGB during %s — results may be contaminated by disk I/O",
                (current_swap - baseline_bytes) / (1024**3),
                context,
            )
            return True
        return False

    def _countdown_sleep(self, secs: float, desc: str = "Waiting"):
        """Sleep with a visible tqdm countdown so the terminal doesn't freeze.
        
        Args:
            secs (float): Duration in seconds.
            desc (str): Short user-facing label for progress output. Defaults to 'Waiting'.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        if secs <= 0:
            return
        for _ in progress(
            range(int(secs)),
            desc=desc,
            position=3,
            leave=False,
            colour="yellow",
            unit="s",
        ):
            time.sleep(1.0)

    def _write_preprocessing_report(self, sources: List[SourceSeries]) -> None:
        """Write preprocessing report for BenchmarkRunner.
        
        Args:
            sources (List[SourceSeries]): Validated independent source series; sources are never
                                          concatenated before windowing.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        rows = [source.preprocessing for source in sources]
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        self._initialize_dict_csv(self.preprocessing_csv_path, fieldnames)
        existing_paths = set()
        if self.preprocessing_csv_path.exists():
            try:
                existing = pd.read_csv(self.preprocessing_csv_path, usecols=["source_path"])
                existing_paths = set(existing["source_path"].astype(str))
            except (ValueError, pd.errors.EmptyDataError):
                existing_paths = set()
        new_rows = [row for row in rows if str(row["source_path"]) not in existing_paths]
        self._append_dict_rows(self.preprocessing_csv_path, fieldnames, new_rows)

    def _write_run_metadata(self, sources: List[SourceSeries]) -> None:
        """Write run metadata for BenchmarkRunner.
        
        Args:
            sources (List[SourceSeries]): Validated independent source series; sources are never
                                          concatenated before windowing.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        metadata_dir = self.output_dir / "benchmark_metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        model_key = "_".join(self._safe_path_part(model) for model in self.config.models)
        cpu_model = platform.processor()
        try:
            for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
                if line.lower().startswith(("model name", "hardware")):
                    cpu_model = line.split(":", 1)[-1].strip()
                    break
        except OSError:
            pass
        try:
            git_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            git_commit = ""
        group_counts: Dict[str, int] = {}
        for source in sources:
            group_counts[source.dataset_group] = group_counts.get(source.dataset_group, 0) + 1
        file_stats = [
            {
                "source_relative_path": source.source_relative_path,
                "size_bytes": Path(source.source_path).stat().st_size,
                "mtime_ns": Path(source.source_path).stat().st_mtime_ns,
            }
            for source in sources
        ]
        dataset_manifest_path = metadata_dir / "dataset_manifest.json"
        dataset_manifest: Dict[str, Any] = {}
        if dataset_manifest_path.exists():
            try:
                candidate = json.loads(dataset_manifest_path.read_text())
                if candidate.get("file_stats") == file_stats:
                    dataset_manifest = candidate
            except (OSError, json.JSONDecodeError):
                pass
        if not dataset_manifest:
            digest = hashlib.sha256()
            files = []
            for source, stat_row in zip(sources, file_stats):
                file_digest = hashlib.sha256()
                with Path(source.source_path).open("rb") as source_file:
                    for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                        file_digest.update(chunk)
                sha256 = file_digest.hexdigest()
                digest.update(source.source_relative_path.encode("utf-8"))
                digest.update(sha256.encode("ascii"))
                files.append({**stat_row, "sha256": sha256})
            dataset_manifest = {
                "dataset_sha256": digest.hexdigest(),
                "file_count": len(files),
                "file_stats": file_stats,
                "files": files,
            }
            dataset_manifest_path.write_text(
                json.dumps(dataset_manifest, indent=2, sort_keys=True) + "\n"
            )
        metadata: Dict[str, Any] = {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "benchmark_seed": self.benchmark_seed,
            "git_commit": git_commit,
            "config": self.config.model_dump(mode="json"),
            "model_artifacts": {
                model: MODEL_ARTIFACTS.get(model, {"source": "unknown"})
                for model in self.config.models
            },
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version() if cuda_benchmark_enabled() else None,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
            },
            "hardware": {
                "cpu_model": cpu_model,
                "logical_cpu_count": psutil.cpu_count(logical=True),
                "physical_cpu_count": psutil.cpu_count(logical=False),
                "total_ram_bytes": psutil.virtual_memory().total,
                "monitor_class": type(self.power_monitor).__name__,
                "telemetry_interval_ms": getattr(self.power_monitor, "interval_ms", 50),
                "cloud_provider": os.environ.get("CLOUD_PROVIDER", ""),
                "cloud_instance_type": os.environ.get("CLOUD_INSTANCE_TYPE", ""),
                "cloud_region": os.environ.get("CLOUD_REGION", ""),
                "cloud_instance_id": os.environ.get("CLOUD_INSTANCE_ID", ""),
            },
            "threading": {
                "torch_intra_op_threads": torch.get_num_threads(),
                "torch_inter_op_threads": torch.get_num_interop_threads(),
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
                "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", ""),
            },
            "methodology": {
                "phase": self.config.phase,
                "prediction_phase": f"Deterministic untimed inference at batch size {self.config.prediction_batch_size}; saves one NPZ per source and computes no metrics.",
                "benchmark_phase": f"Exactly {self.config.benchmark_batches} preloaded batches per iteration, with {self.config.warmups} warmups and {self.config.repeats} measured repeats; saves no predictions or accuracy.",
                "imputation": "Per-source global target median for model inputs; the original observed mask is retained in prediction NPZ files.",
                "raw_power": "avg_power_w and total_energy_j use the monitor-specific physical scope recorded in power_scope; they are not assumed to be wall power.",
                "dynamic_power": "dynamic_* fields subtract idle_baseline_w over telemetry_duration_sec and clamp at zero.",
                "timing_scope": f"E2E and model-call fields cover {self.config.benchmark_batches} synchronous forecast calls over preloaded contiguous NumPy batch views.",
                "timing_exclusions": "CSV parsing, imputation, window indexing, benchmark-pool construction, model construction, stabilization, warmups, progress rendering, telemetry aggregation, cooling, and persistence.",
                "repeatability": "Source order and batch order are fixed, shuffle is disabled, and Python/NumPy/PyTorch/CUDA RNGs use benchmark_seed.",
                "gpu_memory_guard": {
                    "safety_fraction": GPU_MEMORY_SAFETY_FRACTION,
                    "estimate_headroom": GPU_MEMORY_ESTIMATE_HEADROOM,
                    "hard_limit_gb": os.environ.get("GPU_MEMORY_HARD_LIMIT_GB"),
                    "policy": "Project the next batch peak from observed PyTorch peak-reserved memory after timing; do not launch configurations at or above the safe device-memory limit.",
                },
            },
            "source_manifest": {
                "source_count": len(sources),
                "group_counts": group_counts,
                "preprocessing_report": str(self.preprocessing_csv_path),
                "dataset_manifest": str(dataset_manifest_path),
                "dataset_sha256": dataset_manifest["dataset_sha256"],
            },
        }
        if cuda_benchmark_enabled():
            properties = torch.cuda.get_device_properties(0)
            metadata["hardware"].update(
                {
                    "gpu_name": properties.name,
                    "gpu_total_memory_bytes": properties.total_memory,
                    "gpu_compute_capability": f"{properties.major}.{properties.minor}",
                    "gpu_physical_index": int(
                        os.environ.get("BENCHMARK_GPU_INDEX", "0")
                    ),
                    "cuda_visible_devices": os.environ.get(
                        "CUDA_VISIBLE_DEVICES", ""
                    ),
                    "gpu_uuid": getattr(
                        self.device_profile, "selected_gpu_uuid", ""
                    ),
                    "gpu_mig_mode": getattr(
                        self.device_profile, "selected_gpu_mig_mode", "unavailable"
                    ),
                    "gpu_original_power_limit_w": getattr(
                        self.device_profile, "original_power_limit_w", None
                    ),
                    "gpu_clock_policy": getattr(
                        self.device_profile, "clock_policy", "unavailable"
                    ),
                }
            )
        metadata_tag = os.environ.get("BENCHMARK_METADATA_TAG", "").strip()
        metadata_suffix = (
            f"_{self._safe_path_part(metadata_tag)}" if metadata_tag else ""
        )
        path = metadata_dir / (
            f"{model_key or 'benchmark'}_{self.config.phase}{metadata_suffix}.json"
        )
        path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        logger.info("[METADATA] Reproducibility manifest: %s", path)

    def _preflight_device_exclusivity(self) -> None:
        """Wait for a quiet discrete GPU before changing its clocks or limits.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        if self.config.device_category not in DISCRETE_NVIDIA_CATEGORIES:
            return
        try:
            import pynvml

            gpu_index = int(os.environ.get("BENCHMARK_GPU_INDEX", "0"))
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        except Exception as exc:
            logger.warning("[PREFLIGHT] Could not query GPU exclusivity: %s", exc)
            return

        wait_sec = max(0.0, float(os.environ.get("GPU_PREFLIGHT_WAIT_SEC", "120")))
        poll_sec = max(0.2, float(os.environ.get("GPU_PREFLIGHT_POLL_SEC", "5")))
        deadline = time.monotonic() + wait_sec
        attempt = 0
        while True:
            attempt += 1
            samples = []
            try:
                for _ in range(5):
                    samples.append(
                        float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                    )
                    time.sleep(0.2)
            except Exception as exc:
                logger.warning("[PREFLIGHT] Could not query GPU exclusivity: %s", exc)
                return

            average_util = float(np.mean(samples))
            if average_util <= self.config.max_idle_gpu_util_percent:
                logger.info(
                    "[PREFLIGHT] GPU idle after %d check(s): %.1f%% utilization.",
                    attempt,
                    average_util,
                )
                return

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"GPU remained busy ({average_util:.1f}% utilization) for "
                    f"{wait_sec:.0f}s; refusing to change clocks or run a "
                    "contaminated benchmark."
                )
            if attempt == 1:
                logger.warning(
                    "[PREFLIGHT] GPU is busy at %.1f%%; waiting up to %.0fs for a "
                    "sustained idle window.",
                    average_util,
                    wait_sec,
                )
            time.sleep(min(poll_sec, remaining))

    def _make_window_dataset(
        self, sources: List[SourceSeries], ctx: int, hor: int
    ) -> MultiSourceWindowDataset:
        """Construct window dataset for BenchmarkRunner.
        
        Args:
            sources (List[SourceSeries]): Validated independent source series; sources are never
                                          concatenated before windowing.
            ctx (int): Context length for the current configuration.
            hor (int): Forecast horizon for the current configuration.
        
        Returns:
            MultiSourceWindowDataset: Result produced by this operation.
        """
        stride = self.config.window_stride or hor
        return MultiSourceWindowDataset(
            sources,
            context_len=ctx,
            horizon_len=hor,
            stride=stride,
            max_windows_per_source=self.config.max_windows,
            include_targets=False,
        )

    @staticmethod
    def _make_loader(
        dataset: MultiSourceWindowDataset,
        batch_size: int,
        num_workers: int,
        seed: int,
    ) -> DataLoader:
        """Construct loader for BenchmarkRunner.
        
        Args:
            dataset (MultiSourceWindowDataset): Window dataset used to construct deterministic batches.
            batch_size (int): Number of source-local windows passed to one model call.
            num_workers (int): PyTorch DataLoader worker-process count.
            seed (int): Root random seed used for deterministic execution.
        
        Returns:
            DataLoader: Result produced by this operation.
        """
        generator = torch.Generator()
        generator.manual_seed(seed)
        kwargs: Dict[str, Any] = {
            "batch_size": int(batch_size),
            "shuffle": False,
            "drop_last": False,
            "num_workers": int(num_workers),
            "pin_memory": False,
            "persistent_workers": int(num_workers) > 0,
            "collate_fn": identity_window_batch,
            "generator": generator,
            "worker_init_fn": seed_data_worker,
        }
        if num_workers > 0:
            kwargs["prefetch_factor"] = 1
        return DataLoader(dataset, **kwargs)

    @staticmethod
    def _normalize_predictions(predictions: Any, batch_size: int, horizon_len: int) -> np.ndarray:
        """Normalize predictions for BenchmarkRunner.
        
        Args:
            predictions (Any): Raw adapter output to normalize or persist.
            batch_size (int): Number of source-local windows passed to one model call.
            horizon_len (int): Number of future points requested from the model.
        
        Returns:
            np.ndarray: NumPy data produced by this operation.
        """
        if torch.is_tensor(predictions):
            predictions = predictions.detach().cpu().numpy()
        values = np.asarray(predictions, dtype=np.float32)
        if values.ndim == 1 and batch_size == 1:
            values = values.reshape(1, -1)
        elif values.ndim > 2:
            values = values.reshape(values.shape[0], -1)
        if values.ndim != 2 or values.shape[0] != batch_size or values.shape[1] < horizon_len:
            raise ValueError(
                "Model returned invalid forecast shape "
                f"{values.shape}; expected ({batch_size}, >= {horizon_len})"
            )
        values = np.ascontiguousarray(values[:, :horizon_len], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Model returned non-finite forecast values")
        return values

    def _save_prediction_npz(
        self,
        *,
        model_name: str,
        dataset_name: str,
        source: SourceSeries,
        ctx: int,
        hor: int,
        reference_batch_size: int,
        window_ids: np.ndarray,
        starts: np.ndarray,
        predictions: np.ndarray,
        trues: np.ndarray,
        observed_mask: np.ndarray,
    ) -> Path:
        """Persist prediction npz for BenchmarkRunner.
        
        Args:
            model_name (str): Registered model adapter name.
            dataset_name (str): Stable dataset identifier written to output metadata.
            source (SourceSeries): One independently validated source series.
            ctx (int): Context length for the current configuration.
            hor (int): Forecast horizon for the current configuration.
            reference_batch_size (int): Configuration or runtime value named reference_batch_size.
            window_ids (np.ndarray): Configuration or runtime value named window_ids.
            starts (np.ndarray): Configuration or runtime value named starts.
            predictions (np.ndarray): Raw adapter output to normalize or persist.
            trues (np.ndarray): Configuration or runtime value named trues.
            observed_mask (np.ndarray): Configuration or runtime value named observed_mask.
        
        Returns:
            Path: Result produced by this operation.
        """
        path = self._prediction_npz_path(model_name, source, ctx, hor)
        horizon_steps = np.arange(hor, dtype=np.int64)
        forecast_indices = starts[:, None] + int(ctx) + horizon_steps[None, :]
        forecast_time_ns = source.timestamps_at(forecast_indices)
        seasonality = int(self.config.accuracy_seasonality)
        context_steps = np.arange(ctx, dtype=np.int64)
        context_indices = starts[:, None] + context_steps[None, :]
        contexts = source.values[context_indices].astype(np.float64, copy=False)
        if ctx > seasonality:
            seasonal_differences = contexts[:, seasonality:] - contexts[:, :-seasonality]
            context_mae_scale = np.mean(np.abs(seasonal_differences), axis=1)
            context_mse_scale = np.mean(np.square(seasonal_differences), axis=1)
        else:
            context_mae_scale = np.full(len(starts), np.nan, dtype=np.float64)
            context_mse_scale = np.full(len(starts), np.nan, dtype=np.float64)
        np.savez(
            path,
            y_pred=np.ascontiguousarray(predictions, dtype=np.float32),
            y_true=np.ascontiguousarray(trues, dtype=np.float32),
            y_true_observed=np.ascontiguousarray(observed_mask, dtype=np.bool_),
            window_id=np.ascontiguousarray(window_ids, dtype=np.int64),
            context_start_index=np.ascontiguousarray(starts, dtype=np.int64),
            forecast_start_index=np.ascontiguousarray(starts + int(ctx), dtype=np.int64),
            forecast_time_ns=np.ascontiguousarray(forecast_time_ns, dtype=np.int64),
            horizon_step=horizon_steps,
            source_id=np.asarray(source.source_id),
            source_file=np.asarray(source.source_file),
            source_relative_path=np.asarray(source.source_relative_path),
            dataset_group=np.asarray(source.dataset_group),
            source_path=np.asarray(source.source_path),
            target_column=np.asarray(source.target_column),
            time_column=np.asarray(source.time_column),
            model_name=np.asarray(model_name),
            device_category=np.asarray(self.config.device_category.value),
            execution_device=np.asarray(self._execution_device()),
            dataset_name=np.asarray(dataset_name),
            pollutant=np.asarray(self.config.pollutant),
            imputation_policy=np.asarray("per_source_global_median"),
            benchmark_seed=np.asarray(self.benchmark_seed, dtype=np.int64),
            accuracy_seasonality=np.asarray(seasonality, dtype=np.int64),
            context_mae_scale=np.asarray(context_mae_scale, dtype=np.float32),
            context_mse_scale=np.asarray(context_mse_scale, dtype=np.float32),
            context_len=np.asarray(ctx, dtype=np.int64),
            horizon_len=np.asarray(hor, dtype=np.int64),
            window_stride=np.asarray(self.config.window_stride or hor, dtype=np.int64),
            reference_batch_size=np.asarray(reference_batch_size, dtype=np.int64),
            power_mode=np.asarray(str(self.current_power_mode)),
        )
        logger.debug(
            "[PREDICTIONS] Saved %d windows for source=%s to %s",
            len(predictions),
            source.source_id,
            path,
        )
        return path

    def _run_prediction_pass(
        self,
        model_name: str,
        ctx: int,
        hor: int,
        sources: List[SourceSeries],
        dataset_name: str,
    ) -> None:
        """Generate untimed deterministic forecasts for one model/context/horizon and save one NPZ per source.
        
        Args:
            model_name (str): Registered model adapter name.
            ctx (int): Context length for the current configuration.
            hor (int): Forecast horizon for the current configuration.
            sources (List[SourceSeries]): Validated independent source series; sources are never
                                          concatenated before windowing.
            dataset_name (str): Stable dataset identifier written to output metadata.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        dataset = self._make_window_dataset(sources, ctx, hor)
        total_windows = len(dataset)
        if total_windows == 0:
            logger.warning("[PREDICTIONS] No complete windows for ctx=%d hor=%d", ctx, hor)
            return

        reference_batch_size = self.config.prediction_batch_size
        effective_workers = 0 if model_name == "toto" else self.config.num_workers
        seed_benchmark_rng(self.benchmark_seed)
        model = get_model(
            model_name,
            device=self._execution_device(),
            context_len=ctx,
            horizon_len=hor,
            batch_size=reference_batch_size,
            num_workers=effective_workers,
        )
        loader = self._make_loader(
            dataset,
            reference_batch_size,
            effective_workers,
            self.benchmark_seed,
        )
        working_dir = self.output_dir / ".prediction_work"
        working_dir.mkdir(parents=True, exist_ok=True)
        prediction_work_path = working_dir / (
            f"{self._safe_path_part(model_name)}_ctx_{ctx}_hor_{hor}.float32"
        )
        predictions = np.memmap(
            prediction_work_path,
            mode="w+",
            dtype=np.float32,
            shape=(total_windows, hor),
        )

        logger.info(
            "[PREDICTIONS] Untimed generation: model=%s ctx=%d hor=%d bs=%d windows=%d",
            model_name,
            ctx,
            hor,
            reference_batch_size,
            total_windows,
        )
        reference_bar = progress(
            total=len(loader),
            desc=f"Predictions {model_name} ctx={ctx} hor={hor}",
            position=3,
            leave=False,
            colour="magenta",
            unit="batch",
        )
        try:
            try:
                for batch_x, _batch_y, _source_idx, _window_id, _start, global_index in loader:
                    inputs = batch_x.numpy()
                    batch_predictions, _ = model.forecast(inputs, hor)
                    batch_predictions = self._normalize_predictions(
                        batch_predictions, len(inputs), hor
                    )
                    indices = self._metadata_to_numpy(global_index)
                    predictions[indices] = batch_predictions
                    reference_bar.update(1)
            finally:
                reference_bar.close()
                self._shutdown_dataloader_workers(loader)
                del loader
                del model
                clear_gpu_memory()

            predictions.flush()
            horizon_steps = np.arange(hor, dtype=np.int64)
            saved_prediction_files = 0
            saved_prediction_windows = 0

            for source in sources:
                source_start, source_end = dataset.source_offsets[source.source_id]
                if source_end <= source_start:
                    continue
                source_starts = dataset.starts[source_start:source_end]
                forecast_indices = source_starts[:, None] + int(ctx) + horizon_steps[None, :]
                source_predictions = np.asarray(
                    predictions[source_start:source_end], dtype=np.float32
                )
                source_trues = source.values[forecast_indices]
                source_observed = source.observed_at(forecast_indices)
                self._save_prediction_npz(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    source=source,
                    ctx=ctx,
                    hor=hor,
                    reference_batch_size=reference_batch_size,
                    window_ids=dataset.window_ids[source_start:source_end],
                    starts=source_starts,
                    predictions=source_predictions,
                    trues=source_trues,
                    observed_mask=source_observed,
                )
                saved_prediction_files += 1
                saved_prediction_windows += source_end - source_start
            logger.info(
                "[PREDICTIONS] Saved %d source file(s), %d windows for model=%s ctx=%d hor=%d",
                saved_prediction_files,
                saved_prediction_windows,
                model_name,
                ctx,
                hor,
            )
        finally:
            predictions.flush()
            mmap_handle = getattr(predictions, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()
            del predictions, dataset
            prediction_work_path.unlink(missing_ok=True)
            try:
                working_dir.rmdir()
            except OSError:
                pass
            gc.collect()

    def run(self, data_path: str):
        """Execute run for BenchmarkRunner.
        
        Args:
            data_path (str): Dataset directory passed to the runner.
        
        Returns:
            Runtime value with no declared return annotation; see the implementation and callers.
        """
        return self._run_source_directory(data_path)

    def _save_rows(self, rows: List[ResultRow]):
        """Persist rows for BenchmarkRunner.
        
        Args:
            rows (List[ResultRow]): Per-repeat or summary records to process.
        
        Returns:
            None; the callable operates through state changes, validation, or persistence.
        """
        if not self.config.save_raw_performance_metrics:
            return
        df = pd.DataFrame([row.model_dump(mode="json") for row in rows])
        df.to_csv(self.csv_path, mode="a", header=False, index=False)

    def _run_source_directory(self, data_dir: str) -> None:
        """Execute the selected prediction or benchmark phase over the configured model and hardware loops.
        
        Args:
            data_dir (str): Root directory searched recursively for source CSV files.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        dataset_root = Path(data_dir).expanduser().resolve()
        dataset_name = self.config.dataset_folder or dataset_root.name
        logger.info("=" * 60)
        logger.info("Starting Unified BenchmarkRunner - Dataset directory: %s", dataset_root)
        logger.info("Device: %s", self.config.device_category.value)
        logger.info("Models: %s", self.config.models)
        logger.info("Power Modes: %s", self.config.power_modes)
        logger.info(
            "Power mode switching: %s; stabilization: %.0fs",
            "enabled" if self.config.power_mode_switching else "disabled",
            self.config.power_mode_stabilization_sec,
        )
        logger.info("DataLoader workers: %d", self.config.num_workers)
        logger.info("=" * 60)

        if self.config.phase == "prediction":
            sources = load_source_directory(
                dataset_root,
                pattern=self.config.source_pattern,
                excluded_names=self.config.excluded_source_names,
                max_sources=self.config.max_sources,
                target_column=self.config.target_column,
                time_column=self.config.time_column,
                cache_dir=self.config.data_cache_dir,
            )
            self._write_preprocessing_report(sources)
            self._write_run_metadata(sources)
            prediction_configs = (
                len(self.config.models)
                * len(self.config.context_lens)
                * len(self.config.horizon_lens)
            )
            prediction_bar = progress(
                total=prediction_configs,
                desc="Prediction configs",
                position=0,
                leave=True,
                colour="magenta",
                unit="cfg",
            )
            try:
                for model_name in self.config.models:
                    for ctx in self.config.context_lens:
                        for hor in self.config.horizon_lens:
                            self._run_prediction_pass(model_name, ctx, hor, sources, dataset_name)
                            prediction_bar.update(1)
            finally:
                prediction_bar.close()
                clear_gpu_memory()
            logger.info("[COMPLETE] Finished prediction generation")
            return

        self._preflight_device_exclusivity()
        self.device_profile.setup()
        try:
            sources = load_source_directory(
                dataset_root,
                pattern=self.config.source_pattern,
                excluded_names=self.config.excluded_source_names,
                max_sources=self.config.max_sources,
                target_column=self.config.target_column,
                time_column=self.config.time_column,
                cache_dir=self.config.data_cache_dir,
            )
            self._write_preprocessing_report(sources)
            self._write_run_metadata(sources)
            dataset_group_count = len({source.dataset_group for source in sources})
            logger.info(
                "[DATA] Loaded %d independent source(s) across %d dataset group(s)",
                len(sources),
                dataset_group_count,
            )
            self.swap_baseline_bytes = psutil.swap_memory().used

            for power_mode in progress(
                self.config.power_modes,
                desc="Power modes",
                position=0,
                leave=True,
                colour="green",
                unit="mode",
            ):
                self.current_power_mode = power_mode
                self.current_power_mode_verification = "NOT_VERIFIED"
                switching_supported = self.device_profile.supports_power_mode_switching()
                if self.config.power_mode_switching and switching_supported:
                    logger.info("[POWER MODE] Switching to power mode: %s", power_mode)
                    self.device_profile.apply_power_mode(power_mode)
                    if not self.device_profile.verify_power_mode(power_mode):
                        raise RuntimeError(
                            f"Power mode {power_mode!r} was not applied successfully."
                        )
                    logger.info(
                        "[POWER MODE] Verified mode %s; stabilizing for %.0fs before measurement.",
                        power_mode,
                        self.config.power_mode_stabilization_sec,
                    )
                    self._countdown_sleep(
                        self.config.power_mode_stabilization_sec,
                        desc=f"Power mode {power_mode} stabilization",
                    )
                    if not self.device_profile.verify_power_mode(power_mode):
                        raise RuntimeError(
                            f"Power mode {power_mode!r} changed during stabilization."
                        )
                    logger.info(
                        "[POWER MODE] Mode %s remained active after stabilization.",
                        power_mode,
                    )
                    self.current_power_mode_verification = "VERIFIED"
                elif self.config.power_mode_switching:
                    logger.info(
                        "[POWER MODE] Switching is not applicable to %s; no stabilization wait is required.",
                        self.config.device_category.value,
                    )
                    self.current_power_mode_verification = "NOT_APPLICABLE"
                else:
                    logger.warning(
                        "[POWER MODE] Switching disabled; labelling measurements with requested mode %s only.",
                        power_mode,
                    )
                    self.current_power_mode_verification = "DISABLED"
                for model_index, model_name in enumerate(
                    progress(
                        self.config.models,
                        desc="Models",
                        position=1,
                        leave=False,
                        colour="blue",
                        unit="model",
                    )
                ):
                    if model_index > 0:
                        clear_gpu_memory()
                        self._countdown_sleep(
                            self.config.sleep_between_models_sec,
                            desc="Model cooling",
                        )
                    clear_gpu_memory()

                    total_configs = (
                        len(self.config.context_lens)
                        * len(self.config.horizon_lens)
                        * len(self.config.batch_sizes)
                    )
                    config_bar = progress(
                        total=total_configs,
                        desc=f"Configs ({model_name})",
                        position=2,
                        leave=False,
                        colour="cyan",
                        unit="cfg",
                    )
                    model_config_started = False
                    try:
                        for ctx in self.config.context_lens:
                            for hor in self.config.horizon_lens:
                                index_dataset = self._make_window_dataset(sources, ctx, hor)
                                total_windows = len(index_dataset)
                                logger.info(
                                    "[WINDOWS] ctx=%d hor=%d stride=%d total=%d sources=%d",
                                    ctx,
                                    hor,
                                    index_dataset.stride,
                                    total_windows,
                                    len(sources),
                                )
                                del index_dataset
                                if total_windows == 0:
                                    logger.warning(
                                        "[WINDOWS] No complete windows for ctx=%d hor=%d; skipping.",
                                        ctx,
                                        hor,
                                    )
                                    config_bar.update(len(self.config.batch_sizes))
                                    continue

                                ceiling_triggered = False
                                ceiling_reason = ""
                                for batch_index, batch_size in enumerate(self.config.batch_sizes):
                                    config_bar.set_description(
                                        f"{model_name} ctx={ctx} hor={hor} bs={batch_size}"
                                    )
                                    config_bar.set_postfix_str(
                                        f"sources={len(sources)} windows={total_windows}"
                                    )
                                    skip_for_gpu, estimated_gpu_gb, gpu_limit_gb = (
                                        self._should_skip_gpu_batch_for_memory(
                                            model_name, ctx, hor, batch_size
                                        )
                                    )
                                    skip_for_ram, estimated_peak_gb, ram_limit_gb = (
                                        self._should_skip_cpu_batch_for_ram(
                                            model_name, ctx, hor, batch_size
                                        )
                                    )
                                    if skip_for_gpu:
                                        logger.warning(
                                            "[GPU MEMORY] Skipping %s ctx=%d hor=%d bs=%d: "
                                            "projected peak %.2fGB exceeds safe limit %.2fGB.",
                                            model_name,
                                            ctx,
                                            hor,
                                            batch_size,
                                            estimated_gpu_gb,
                                            gpu_limit_gb,
                                        )
                                        self._record_source_oom(
                                            model_name,
                                            ctx,
                                            hor,
                                            batch_size,
                                            dataset_name,
                                            len(sources),
                                            total_windows,
                                        )
                                        ceiling_triggered = True
                                        ceiling_reason = "OOM"
                                    elif skip_for_ram:
                                        logger.warning(
                                            "[CPU RAM] Skipping %s ctx=%d hor=%d bs=%d: "
                                            "estimated %.2fGB exceeds %.2fGB.",
                                            model_name,
                                            ctx,
                                            hor,
                                            batch_size,
                                            estimated_peak_gb,
                                            ram_limit_gb,
                                        )
                                        self._record_source_oom(
                                            model_name,
                                            ctx,
                                            hor,
                                            batch_size,
                                            dataset_name,
                                            len(sources),
                                            total_windows,
                                        )
                                        ceiling_triggered = True
                                        ceiling_reason = "OOM"
                                    else:
                                        if model_config_started:
                                            clear_gpu_memory()
                                            self._countdown_sleep(
                                                self.config.sleep_between_batches_sec,
                                                desc="Batch-size cooling",
                                            )
                                        clear_gpu_memory()
                                        model_config_started = True
                                        try:
                                            peak_cpu_ram_gb = self._run_source_batch_size(
                                                model_name=model_name,
                                                ctx=ctx,
                                                hor=hor,
                                                batch_size=batch_size,
                                                sources=sources,
                                                dataset_name=dataset_name,
                                            )
                                            self._remember_cpu_peak(
                                                model_name,
                                                ctx,
                                                hor,
                                                batch_size,
                                                peak_cpu_ram_gb,
                                            )
                                        except ConfigDurationExceeded as exc:
                                            logger.warning(
                                                "[TIMEOUT] %s ctx=%d hor=%d bs=%d: %s",
                                                model_name,
                                                ctx,
                                                hor,
                                                batch_size,
                                                exc,
                                            )
                                            self._record_source_limit(
                                                model_name,
                                                ctx,
                                                hor,
                                                batch_size,
                                                dataset_name,
                                                len(sources),
                                                total_windows,
                                                status=ExecutionStatus.TIMEOUT,
                                            )
                                            ceiling_triggered = True
                                            ceiling_reason = "TIMEOUT"
                                            clear_gpu_memory()
                                        except torch.cuda.OutOfMemoryError as exc:
                                            logger.warning(
                                                "[OOM] %s ctx=%d hor=%d bs=%d: %s",
                                                model_name,
                                                ctx,
                                                hor,
                                                batch_size,
                                                exc,
                                            )
                                            self._record_source_oom(
                                                model_name,
                                                ctx,
                                                hor,
                                                batch_size,
                                                dataset_name,
                                                len(sources),
                                                total_windows,
                                            )
                                            ceiling_triggered = True
                                            ceiling_reason = "OOM"
                                            clear_gpu_memory()
                                        except RuntimeError as exc:
                                            if not _is_batch_limit_runtime(exc):
                                                raise
                                            logger.warning(
                                                "[BATCH LIMIT] %s ctx=%d hor=%d bs=%d: %s",
                                                model_name,
                                                ctx,
                                                hor,
                                                batch_size,
                                                exc,
                                            )
                                            self._record_source_oom(
                                                model_name,
                                                ctx,
                                                hor,
                                                batch_size,
                                                dataset_name,
                                                len(sources),
                                                total_windows,
                                            )
                                            ceiling_triggered = True
                                            ceiling_reason = "OOM"
                                            clear_gpu_memory()

                                    config_bar.update(1)
                                    if ceiling_triggered:
                                        skipped = len(self.config.batch_sizes) - batch_index - 1
                                        if skipped:
                                            logger.info(
                                                "[%s] Skipping %d larger batch size(s) for ctx=%d hor=%d; "
                                                "the next horizon/context will still run.",
                                                ceiling_reason,
                                                skipped,
                                                ctx,
                                                hor,
                                            )
                                            config_bar.update(skipped)
                                        break
                    finally:
                        config_bar.close()
        finally:
            self.device_profile.teardown()
            logger.info("=" * 60)
            logger.info(
                "BenchmarkRunner finished. Batch summary: %s",
                self.batch_summary_csv_path,
            )
            logger.info("=" * 60)

    def _run_source_batch_size(
        self,
        *,
        model_name: str,
        ctx: int,
        hor: int,
        batch_size: int,
        sources: List[SourceSeries],
        dataset_name: str,
    ) -> Optional[float]:
        """Run one model/context/horizon/batch-size configuration, including preload, baseline, warmups, measured repeats, telemetry, and summary persistence.
        
        Args:
            model_name (str): Registered model adapter name.
            ctx (int): Context length for the current configuration.
            hor (int): Forecast horizon for the current configuration.
            batch_size (int): Number of source-local windows passed to one model call.
            sources (List[SourceSeries]): Validated independent source series; sources are never
                                          concatenated before windowing.
            dataset_name (str): Stable dataset identifier written to output metadata.
        
        Returns:
            Optional[float]: Result produced by this operation.
        """
        device_str = self._execution_device()
        effective_workers = 0 if model_name == "toto" else self.config.num_workers
        model = self._load_model_for_config(
            model_name=model_name,
            device=device_str,
            context_len=ctx,
            horizon_len=hor,
            batch_size=batch_size,
            num_workers=effective_workers,
        )
        model_artifact, model_variant = get_model_result_metadata(
            model_name,
            model=model,
            context_len=ctx,
            horizon_len=hor,
        )
        logger.info(
            "[MODEL VARIANT] artifact=%s revision=%s",
            model_artifact or "unavailable",
            model_variant or "unavailable",
        )
        stabilization_sec = float(os.environ.get("MODEL_STABILIZATION_SEC", "30"))
        logger.info("[MODEL] Loaded %s. Stabilizing for %.1fs...", model_name, stabilization_sec)
        self._countdown_sleep(stabilization_sec, desc="Model stabilization")

        dataset = self._make_window_dataset(sources, ctx, hor)
        total_windows = len(dataset)
        if total_windows == 0:
            del dataset, model
            clear_gpu_memory()
            return None

        prepared = PreparedBenchmarkStream(
            dataset,
            batch_size=batch_size,
            batch_count=self.config.benchmark_batches,
            max_pool_windows=int(os.environ.get("BENCHMARK_PRELOAD_POOL_WINDOWS", "65536")),
        )
        num_batches = self.config.benchmark_batches
        warmup_scope = "fixed_preloaded_batches"
        warmup_batch_limit = num_batches
        warmup_windows_per_iteration = num_batches * batch_size
        logger.info(
            "[PRELOAD] bs=%d batches=%d pool=%d/%d windows repeated=%s",
            batch_size,
            num_batches,
            prepared.pool_windows,
            total_windows,
            prepared.repeated,
        )
        self.idle_baseline_w = self._measure_idle_baseline()

        measured_rows: List[ResultRow] = []
        max_peak_cpu_ram_gb: Optional[float] = None
        max_peak_gpu_reserved_gb: Optional[float] = None
        swap_baseline_bytes = psutil.swap_memory().used
        swap_contaminated = False
        total_iterations = self.config.warmups + self.config.repeats
        config_timeout_sec = self.config.max_config_duration_sec
        config_deadline = (
            time.monotonic() + config_timeout_sec if config_timeout_sec is not None else None
        )
        previous_alarm_handler = None
        alarm_armed = False

        def raise_config_timeout(_signum, _frame):
            """Raise the configuration timeout from the process alarm handler.
            
            Args:
                _signum (Any): Configuration or runtime value named _signum.
                _frame (Any): Configuration or runtime value named _frame.
            
            Returns:
                None; the callable operates through state changes, validation, or persistence.
            """
            raise ConfigDurationExceeded(
                f"configuration exceeded {config_timeout_sec / 3600:.2f} hours"
            )

        if config_timeout_sec is not None and hasattr(signal, "setitimer"):
            previous_alarm_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, raise_config_timeout)
            signal.setitimer(signal.ITIMER_REAL, config_timeout_sec)
            alarm_armed = True
            logger.info(
                "[TIMEOUT] Configuration wall-clock ceiling: %.2f hours",
                config_timeout_sec / 3600,
            )
        iteration_bar = progress(
            total=total_iterations,
            desc=f"Iterations bs={batch_size}",
            position=3,
            leave=False,
            colour="green",
            unit="iter",
        )

        try:
            for iteration in range(total_iterations):
                is_warmup = iteration < self.config.warmups
                phase = "WARMUP" if is_warmup else "MEASURE"
                iteration_batch_total = num_batches
                iteration_bar.set_description(
                    f"{phase} bs={batch_size} batches={iteration_batch_total}/{num_batches}"
                )
                batch_description = (
                    f"Batches [{phase}] bs={batch_size}"
                    if is_warmup
                    else f"Batches [MEASURE; update after timing] bs={batch_size}"
                )
                batch_bar = progress(
                    total=iteration_batch_total,
                    desc=batch_description,
                    position=4,
                    leave=False,
                    colour="blue",
                    unit="batch",
                )
                memory_sampler = ProcessMemorySampler()
                monitor_started = False
                monitor_metrics: Dict[str, Any] = {}
                peak_cpu_ram_gb = float("nan")
                processed_windows = 0
                processed_batches = 0
                measurement_cycles = 0
                model_call_duration_sec = 0.0
                measurement_start_ns = 0
                measurement_end_ns = 0
                next_progress_refresh = time.monotonic() + PROGRESS_MIN_INTERVAL
                try:
                    if cuda_benchmark_enabled():
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()
                    if not is_warmup:
                        memory_sampler.start()
                        self.power_monitor.start()
                        monitor_started = True
                    measurement_start_ns = time.time_ns()
                    started = time.perf_counter()
                    for batch_index in range(num_batches):
                        if config_deadline is not None and time.monotonic() >= config_deadline:
                            raise_config_timeout(None, None)
                        model_inputs = prepared.batch(batch_index)
                        model_call_started = time.perf_counter()
                        predictions, _adapter_reported_time = model.forecast(model_inputs, hor)
                        if config_deadline is not None and time.monotonic() >= config_deadline:
                            raise_config_timeout(None, None)
                        model_call_duration_sec += time.perf_counter() - model_call_started
                        processed_windows += len(model_inputs)
                        processed_batches += 1
                        if live_batch_progress_enabled(is_warmup):
                            now = time.monotonic()
                            if (
                                processed_batches == iteration_batch_total
                                or now >= next_progress_refresh
                            ):
                                batch_bar.n = min(processed_batches, iteration_batch_total)
                                batch_bar.refresh()
                                next_progress_refresh = now + PROGRESS_MIN_INTERVAL
                        del predictions
                    measurement_cycles = 1
                    if cuda_benchmark_enabled():
                        torch.cuda.synchronize()
                    finished = time.perf_counter()
                    measurement_end_ns = time.time_ns()
                    if cuda_benchmark_enabled():
                        current_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
                        max_peak_gpu_reserved_gb = max(
                            max_peak_gpu_reserved_gb or 0.0,
                            current_reserved_gb,
                        )
                finally:
                    if monitor_started:
                        monitor_metrics = self.power_monitor.stop()
                        if measurement_end_ns > measurement_start_ns:
                            monitor_metrics = self._align_monitor_metrics(
                                monitor_metrics,
                                measurement_start_ns,
                                measurement_end_ns,
                            )
                        peak_cpu_ram_gb = memory_sampler.stop()
                        if np.isfinite(peak_cpu_ram_gb):
                            max_peak_cpu_ram_gb = max(
                                max_peak_cpu_ram_gb or 0.0,
                                float(peak_cpu_ram_gb),
                            )
                    # Console rendering is deliberately outside the timed and
                    # power-sampled region so UI speed cannot change throughput.
                    batch_bar.total = max(1, processed_batches)
                    batch_bar.n = processed_batches
                    batch_bar.set_postfix_str(
                        f"cycles={measurement_cycles} windows={processed_windows}"
                    )
                    batch_bar.refresh()
                    batch_bar.close()

                if not is_warmup:
                    duration = finished - started
                    if self._check_swap_usage(
                        swap_baseline_bytes,
                        context=f"{model_name} ctx={ctx} hor={hor} bs={batch_size}",
                    ):
                        swap_contaminated = True
                    oc_event_delta = int(
                        monitor_metrics.get("jetson_oc_event_delta", 0) or 0
                    )
                    hardware_throttled = bool(
                        monitor_metrics.get("hardware_throttled", False)
                    )
                    if oc_event_delta > 0:
                        logger.error(
                            "[JETSON THROTTLE] %d hardware OC event(s) occurred during "
                            "%s ctx=%d hor=%d bs=%d repeat=%d; rejecting this repeat.",
                            oc_event_delta,
                            model_name,
                            ctx,
                            hor,
                            batch_size,
                            iteration - self.config.warmups,
                        )
                        status = ExecutionStatus.HARDWARE_THROTTLED
                    elif hardware_throttled:
                        start_flags = monitor_metrics.get("rpi_throttled_start")
                        end_flags = monitor_metrics.get("rpi_throttled_end")
                        logger.error(
                            "[RPI THROTTLE] Firmware throttle flags changed or were active "
                            "during %s ctx=%d hor=%d bs=%d repeat=%d "
                            "(start=%s end=%s); rejecting this repeat.",
                            model_name,
                            ctx,
                            hor,
                            batch_size,
                            iteration - self.config.warmups,
                            f"0x{start_flags:x}" if isinstance(start_flags, int) else "unavailable",
                            f"0x{end_flags:x}" if isinstance(end_flags, int) else "unavailable",
                        )
                        status = ExecutionStatus.HARDWARE_THROTTLED
                    elif swap_contaminated:
                        status = ExecutionStatus.SWAP_CONTAMINATED
                    else:
                        status = ExecutionStatus.SUCCESS

                    energy_j = float(monitor_metrics.get("total_energy_j", float("nan")))
                    avg_power_w = float(monitor_metrics.get("avg_power_w", float("nan")))
                    telemetry_duration = float(
                        monitor_metrics.get("telemetry_duration_sec", duration)
                    )
                    dynamic_avg_power_w = (
                        max(0.0, avg_power_w - self.idle_baseline_w)
                        if np.isfinite(avg_power_w) and np.isfinite(self.idle_baseline_w)
                        else float("nan")
                    )
                    dynamic_energy_j = (
                        max(0.0, energy_j - self.idle_baseline_w * telemetry_duration)
                        if np.isfinite(energy_j)
                        and np.isfinite(self.idle_baseline_w)
                        and np.isfinite(telemetry_duration)
                        else float("nan")
                    )

                    point_count = processed_windows * hor
                    model_call_point_count = processed_windows * hor
                    pipeline_overhead_sec = max(0.0, duration - model_call_duration_sec)
                    energy_per_window = (
                        energy_j / processed_windows
                        if processed_windows and np.isfinite(energy_j)
                        else float("nan")
                    )
                    energy_per_1k_points = (
                        energy_j / (point_count / 1000)
                        if point_count and np.isfinite(energy_j)
                        else float("nan")
                    )
                    dynamic_energy_per_window = (
                        dynamic_energy_j / processed_windows
                        if processed_windows and np.isfinite(dynamic_energy_j)
                        else float("nan")
                    )
                    dynamic_energy_per_1k_points = (
                        dynamic_energy_j / (point_count / 1000)
                        if point_count and np.isfinite(dynamic_energy_j)
                        else float("nan")
                    )
                    repeat_id = iteration - self.config.warmups
                    row = ResultRow(
                        benchmark_seed=self.benchmark_seed,
                        model_name=model_name,
                        model_artifact=model_artifact,
                        model_variant=model_variant,
                        device_category=self.config.device_category.value,
                        execution_device=device_str,
                        context_len=ctx,
                        horizon_len=hor,
                        batch_size=batch_size,
                        num_workers=effective_workers,
                        repeat_id=repeat_id,
                        power_mode=self.current_power_mode,
                        power_mode_verification=self.current_power_mode_verification,
                        power_mode_stabilization_sec=(
                            self.config.power_mode_stabilization_sec
                            if self.current_power_mode_verification == "VERIFIED"
                            else 0.0
                        ),
                        dataset_name=dataset_name,
                        pollutant=self.config.pollutant,
                        measurement_scope=f"fixed_preloaded_{num_batches}_batches",
                        source_count=len(sources),
                        total_windows=total_windows,
                        processed_windows=processed_windows,
                        processed_points=point_count,
                        measurement_cycles=measurement_cycles,
                        window_stride=dataset.stride,
                        warmup_scope=warmup_scope,
                        warmup_batches_per_iteration=warmup_batch_limit,
                        warmup_windows_per_iteration=warmup_windows_per_iteration,
                        full_batches_per_iteration=num_batches,
                        status=status,
                        measurement_duration_sec=duration,
                        e2e_latency_ms=(duration / processed_batches) * 1000,
                        latency_per_window_ms=(duration / processed_windows) * 1000,
                        throughput_windows_per_sec=processed_windows / duration,
                        throughput_points_per_sec=point_count / duration,
                        model_call_duration_sec=model_call_duration_sec,
                        model_call_latency_per_batch_ms=(
                            model_call_duration_sec / processed_batches
                        )
                        * 1000,
                        model_call_latency_per_window_ms=(
                            model_call_duration_sec / processed_windows
                        )
                        * 1000,
                        model_call_throughput_windows_per_sec=(
                            processed_windows / model_call_duration_sec
                            if model_call_duration_sec > 0
                            else float("nan")
                        ),
                        model_call_throughput_points_per_sec=(
                            model_call_point_count / model_call_duration_sec
                            if model_call_duration_sec > 0
                            else float("nan")
                        ),
                        pipeline_overhead_sec=pipeline_overhead_sec,
                        pipeline_overhead_percent=(
                            pipeline_overhead_sec / duration * 100 if duration > 0 else float("nan")
                        ),
                        telemetry_duration_sec=telemetry_duration,
                        idle_baseline_w=self.idle_baseline_w,
                        idle_gpu_compute_util_percent=self.idle_gpu_util_percent,
                        idle_cpu_util_percent=self.idle_cpu_util_percent,
                        avg_power_w=avg_power_w,
                        power_scope=monitor_metrics.get("power_scope", "unavailable"),
                        peak_power_w=monitor_metrics.get("peak_power_w", float("nan")),
                        power_sample_count=int(monitor_metrics.get("power_sample_count", 0) or 0),
                        power_std_w=monitor_metrics.get("power_std_w", float("nan")),
                        power_p50_w=monitor_metrics.get("power_p50_w", float("nan")),
                        power_p95_w=monitor_metrics.get("power_p95_w", float("nan")),
                        total_energy_j=energy_j,
                        energy_per_window_j=energy_per_window,
                        energy_per_1k_points_j=energy_per_1k_points,
                        edp_joule_seconds=(
                            energy_j * duration if np.isfinite(energy_j) else float("nan")
                        ),
                        dynamic_avg_power_w=dynamic_avg_power_w,
                        dynamic_energy_j=dynamic_energy_j,
                        dynamic_energy_per_window_j=dynamic_energy_per_window,
                        dynamic_energy_per_1k_points_j=dynamic_energy_per_1k_points,
                        dynamic_edp_joule_seconds=(
                            dynamic_energy_j * duration
                            if np.isfinite(dynamic_energy_j)
                            else float("nan")
                        ),
                        avg_gpu_soc_power_w=monitor_metrics.get(
                            "avg_gpu_soc_power_w", float("nan")
                        ),
                        peak_gpu_soc_power_w=monitor_metrics.get(
                            "peak_gpu_soc_power_w", float("nan")
                        ),
                        avg_cpu_cv_power_w=monitor_metrics.get("avg_cpu_cv_power_w", float("nan")),
                        peak_cpu_cv_power_w=monitor_metrics.get(
                            "peak_cpu_cv_power_w", float("nan")
                        ),
                        avg_system_5v_power_w=monitor_metrics.get(
                            "avg_system_5v_power_w", float("nan")
                        ),
                        peak_system_5v_power_w=monitor_metrics.get(
                            "peak_system_5v_power_w", float("nan")
                        ),
                        jetson_oc_counter_available=bool(
                            monitor_metrics.get("jetson_oc_counter_available", False)
                        ),
                        jetson_oc_event_delta=oc_event_delta,
                        jetson_oc1_event_delta=int(
                            monitor_metrics.get("jetson_oc1_event_delta", 0) or 0
                        ),
                        jetson_oc2_event_delta=int(
                            monitor_metrics.get("jetson_oc2_event_delta", 0) or 0
                        ),
                        jetson_oc3_event_delta=int(
                            monitor_metrics.get("jetson_oc3_event_delta", 0) or 0
                        ),
                        avg_gpu_temp_c=monitor_metrics.get("avg_gpu_temp_c", float("nan")),
                        max_gpu_temp_c=monitor_metrics.get("max_gpu_temp_c", float("nan")),
                        avg_cpu_temp_c=monitor_metrics.get("avg_cpu_temp_c", float("nan")),
                        peak_gpu_allocated_gb=(
                            torch.cuda.max_memory_allocated() / (1024**3)
                            if cuda_benchmark_enabled()
                            else float("nan")
                        ),
                        peak_gpu_reserved_gb=(
                            torch.cuda.max_memory_reserved() / (1024**3)
                            if cuda_benchmark_enabled()
                            else float("nan")
                        ),
                        gpu_memory_scope=monitor_metrics.get(
                            "gpu_memory_scope",
                            "not_applicable"
                            if self.config.device_category in CPU_EXECUTION_CATEGORIES
                            else "unavailable",
                        ),
                        gpu_memory_total_gb=monitor_metrics.get(
                            "gpu_memory_total_gb", float("nan")
                        ),
                        avg_gpu_memory_used_gb=monitor_metrics.get(
                            "avg_gpu_memory_used_gb", float("nan")
                        ),
                        peak_gpu_memory_used_gb=monitor_metrics.get(
                            "peak_gpu_memory_used_gb", float("nan")
                        ),
                        min_gpu_memory_free_gb=monitor_metrics.get(
                            "min_gpu_memory_free_gb", float("nan")
                        ),
                        peak_cpu_ram_gb=peak_cpu_ram_gb,
                        avg_gpu_compute_util_percent=monitor_metrics.get(
                            "avg_gpu_compute_util_percent", float("nan")
                        ),
                        avg_gpu_memory_util_percent=monitor_metrics.get(
                            "avg_gpu_memory_util_percent", float("nan")
                        ),
                        gpu_compute_util_std_percent=monitor_metrics.get(
                            "gpu_compute_util_std_percent", float("nan")
                        ),
                        gpu_memory_util_std_percent=monitor_metrics.get(
                            "gpu_memory_util_std_percent", float("nan")
                        ),
                    )
                    measured_rows.append(row)
                iteration_bar.update(1)
        finally:
            if alarm_armed:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_alarm_handler)
            iteration_bar.close()
            del prepared, dataset, model
            clear_gpu_memory()

        if measured_rows:
            self._save_rows(measured_rows)
            summary = self._summarize_batch_rows(measured_rows)
            if summary is not None:
                self._append_dict_rows(
                    self.batch_summary_csv_path,
                    SUMMARY_FIELDNAMES,
                    [summary],
                )
        self._remember_gpu_peak(
            model_name,
            ctx,
            hor,
            batch_size,
            max_peak_gpu_reserved_gb,
        )
        return max_peak_cpu_ram_gb

    def _record_source_oom(
        self,
        model_name: str,
        ctx: int,
        hor: int,
        batch_size: int,
        dataset_name: str,
        source_count: int,
        total_windows: int,
    ) -> None:
        """Record source oom for BenchmarkRunner.
        
        Args:
            model_name (str): Registered model adapter name.
            ctx (int): Context length for the current configuration.
            hor (int): Forecast horizon for the current configuration.
            batch_size (int): Number of source-local windows passed to one model call.
            dataset_name (str): Stable dataset identifier written to output metadata.
            source_count (int): Configuration or runtime value named source_count.
            total_windows (int): Configuration or runtime value named total_windows.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        self._record_source_limit(
            model_name,
            ctx,
            hor,
            batch_size,
            dataset_name,
            source_count,
            total_windows,
            status=ExecutionStatus.OOM,
        )

    def _record_source_limit(
        self,
        model_name: str,
        ctx: int,
        hor: int,
        batch_size: int,
        dataset_name: str,
        source_count: int,
        total_windows: int,
        *,
        status: ExecutionStatus,
    ) -> None:
        """Record source limit for BenchmarkRunner.
        
        Args:
            model_name (str): Registered model adapter name.
            ctx (int): Context length for the current configuration.
            hor (int): Forecast horizon for the current configuration.
            batch_size (int): Number of source-local windows passed to one model call.
            dataset_name (str): Stable dataset identifier written to output metadata.
            source_count (int): Configuration or runtime value named source_count.
            total_windows (int): Configuration or runtime value named total_windows.
            status (ExecutionStatus): Configuration or runtime value named status.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        if status not in {ExecutionStatus.OOM, ExecutionStatus.TIMEOUT}:
            raise ValueError(f"Unsupported terminal configuration status: {status}")
        warmup_scope = "fixed_preloaded_batches"
        warmup_batches = self.config.benchmark_batches
        warmup_windows = batch_size * self.config.benchmark_batches
        full_batches = self.config.benchmark_batches
        model_artifact, model_variant = get_model_result_metadata(
            model_name,
            context_len=ctx,
            horizon_len=hor,
        )
        row = ResultRow(
            benchmark_seed=self.benchmark_seed,
            model_name=model_name,
            model_artifact=model_artifact,
            model_variant=model_variant,
            device_category=self.config.device_category.value,
            execution_device=self._execution_device(),
            context_len=ctx,
            horizon_len=hor,
            batch_size=batch_size,
            num_workers=self.config.num_workers,
            repeat_id=0,
            power_mode=self.current_power_mode,
            power_mode_verification=self.current_power_mode_verification,
            power_mode_stabilization_sec=(
                self.config.power_mode_stabilization_sec
                if self.current_power_mode_verification == "VERIFIED"
                else 0.0
            ),
            dataset_name=dataset_name,
            pollutant=self.config.pollutant,
            measurement_scope=f"fixed_preloaded_{self.config.benchmark_batches}_batches",
            source_count=source_count,
            total_windows=total_windows,
            window_stride=self.config.window_stride or hor,
            warmup_scope=warmup_scope,
            warmup_batches_per_iteration=warmup_batches,
            warmup_windows_per_iteration=warmup_windows,
            full_batches_per_iteration=full_batches,
            status=status,
            oom_ceiling_batch_size=(batch_size if status == ExecutionStatus.OOM else None),
        )
        self._save_rows([row])
        summary = self._summarize_batch_rows([row])
        if summary is not None:
            self._append_dict_rows(self.batch_summary_csv_path, SUMMARY_FIELDNAMES, [summary])
