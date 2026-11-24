#!/usr/bin/env python3
"""
Unified CLI Entry Point for Time-Series Forecasting Benchmarks.
Auto-detects device, parses arguments, and delegates to BenchmarkRunner.
"""

import argparse
import glob
import logging
import os
import subprocess
import sys
import warnings

warnings.filterwarnings(
    "ignore",
    message="The pynvml package is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*torch.utils._pytree._register_pytree_node.*deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*resume_download.*deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Using `json`-module for json-handling.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Triton is not supported on current platform.*",
    category=UserWarning,
)

import torch
from tqdm import tqdm

from benchmark.config import (
    BenchmarkConfig,
    CPU_EXECUTION_CATEGORIES,
    DISCRETE_NVIDIA_CATEGORIES,
    DeviceCategory,
)
from benchmark.runner import BenchmarkRunner

logger = logging.getLogger(__name__)


class TqdmLoggingHandler(logging.Handler):
    """Write console logs without corrupting active tqdm progress bars.
    
    Bases:
        logging.Handler
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a logging record according to TqdmLoggingHandler console policy.
        
        Args:
            record (logging.LogRecord): Logging record being filtered or emitted.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        try:
            tqdm.write(self.format(record), file=sys.stdout)
            self.flush()
        except Exception:
            self.handleError(record)


class ImportantConsoleFilter(logging.Filter):
    """Keep routine diagnostics in the model log instead of the terminal.
    
    Bases:
        logging.Filter
    """

    IMPORTANT_PREFIXES = (
        "[DEVICE]",
        "[POWER MODE]",
        "[COMPLETE]",
    )
    ROUTINE_VENDOR_WARNINGS = (
        "sending unauthenticated requests to the HF Hub",
        "Quantiles are now availble",
        "Passing a tuple of `past_key_values` is deprecated",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        """Select a logging record according to ImportantConsoleFilter console policy.
        
        Args:
            record (logging.LogRecord): Logging record being filtered or emitted.
        
        Returns:
            bool: Whether the condition described by this callable holds.
        """
        message = record.getMessage()
        if any(fragment in message for fragment in self.ROUTINE_VENDOR_WARNINGS):
            return False
        if record.levelno >= logging.WARNING:
            return True
        return message.startswith(self.IMPORTANT_PREFIXES)


def configure_logging(log_file: str | None = None) -> None:
    """Route concise milestone messages to the terminal and complete diagnostic output to the model log.
    
    Args:
        log_file (str | None): Configuration or runtime value named log_file. Defaults to None.
    
    Returns:
        None; the operation updates state, validates input, or persists output in place.
    """
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)

    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(ImportantConsoleFilter())
    root_logger.addHandler(console_handler)

    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def auto_detect_device() -> DeviceCategory:
    """Detect device category from hardware signatures.
    
    Detection priority:
    1. /etc/nv_tegra_release exists → Jetson (definitive, not OS-based)
    2. Device-tree model identifies Raspberry Pi 5 or Arduino UNO Q
    3. torch.cuda device name substring match → discrete GPU
    4. Fallback → CPU
    
    Manual --device flag always overrides this.
    
    Returns:
        DeviceCategory: Result produced by this operation.
    """
    # The Tegra release file is the authoritative Jetson signal.
    if os.path.exists("/etc/nv_tegra_release"):
        with open("/etc/nv_tegra_release") as f:
            content = f.read()
        logger.info(f"[AUTO-DETECT] Found /etc/nv_tegra_release: {content.strip()[:80]}")
        return DeviceCategory.JETSON

    board_model = ""
    for model_path in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            with open(model_path, "rb") as model_file:
                board_model = model_file.read().decode("utf-8", errors="replace").strip("\x00")
            if board_model:
                break
        except OSError:
            continue

    normalized_model = board_model.upper()
    if "RASPBERRY PI 5" in normalized_model:
        hailo_device_present = os.path.exists("/dev/hailo0") or bool(glob.glob("/sys/class/hailo*"))
        if hailo_device_present:
            try:
                identify = subprocess.run(
                    ["hailortcli", "fw-control", "identify"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                ).stdout.upper()
            except (OSError, subprocess.SubprocessError):
                identify = ""
            if "HAILO8L" not in identify and ("HAILO8" in identify or "HAILO-8" in identify):
                logger.info("[AUTO-DETECT] Raspberry Pi 5 with Hailo-8 26-TOPS accelerator")
                return DeviceCategory.RPI5_HAILO26
            logger.warning(
                "[AUTO-DETECT] Raspberry Pi 5 has a Hailo device, but its variant could not be verified as 26 TOPS. "
                "Using rpi5; pass --device rpi5_hailo26 when confirmed."
            )
        logger.info("[AUTO-DETECT] Board model: %s", board_model)
        return DeviceCategory.RPI5

    if "UNO Q" in normalized_model or "QRB2210" in normalized_model:
        logger.info("[AUTO-DETECT] Board model: %s", board_model)
        return DeviceCategory.UNO_Q

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0).upper()
        logger.info(f"[AUTO-DETECT] CUDA device: {torch.cuda.get_device_name(0)}")

        if "H200" in gpu_name:
            return DeviceCategory.H200
        if "H100" in gpu_name:
            return DeviceCategory.H100
        if "A6000" in gpu_name:
            return DeviceCategory.A6000
        elif "5090" in gpu_name:
            return DeviceCategory.RTX5090
        raise RuntimeError(
            f"Unrecognized CUDA device {gpu_name!r}; pass a supported --device explicitly "
            "instead of applying an unsafe default clock/power profile."
        )

    logger.info("[AUTO-DETECT] No CUDA device found. Using CPU profile.")
    return DeviceCategory.CPU


def main():
    """Parse the command line, establish deterministic runtime policy, and execute the configured benchmark runner.
    
    Returns:
        None; the callable operates through state changes, validation, or persistence.
    """
    parser = argparse.ArgumentParser(description="Unified TSFM Benchmark Runner")
    parser.add_argument(
        "--phase",
        choices=("prediction", "benchmark"),
        default="benchmark",
        help="Run prediction generation without metrics, or performance benchmarking without forecasts.",
    )
    parser.add_argument("--data", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing independent source CSV files.",
    )
    parser.add_argument(
        "--dataset-folder",
        type=str,
        default="",
        help="Selected folder inside --data-dir; recorded in result metadata.",
    )
    parser.add_argument(
        "--pollutant",
        type=str,
        default="",
        help="Selected pollutant label; recorded in result metadata.",
    )
    parser.add_argument("--models", type=str, default="ttm", help="Comma-separated model names.")
    parser.add_argument(
        "--context-lens", type=str, default="512", help="Comma-separated context lengths."
    )
    parser.add_argument(
        "--horizon-lens", type=str, default="96", help="Comma-separated horizon lengths."
    )
    parser.add_argument(
        "--batch-sizes", type=str, default="128,256,512", help="Comma-separated batch sizes."
    )
    parser.add_argument("--repeats", type=int, default=5, help="Number of measurement iterations.")
    parser.add_argument("--warmups", type=int, default=2, help="Number of warmup iterations.")
    parser.add_argument("--prediction-batch-size", type=int, default=512)
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("BENCHMARK_SEED", "12345")),
        help="Single reproducibility seed for Python, NumPy, PyTorch, CUDA, and DataLoader workers.",
    )
    parser.add_argument(
        "--accuracy-seasonality",
        type=int,
        default=24,
        help="Seasonal-naive lag used for context-derived MASE/RMSSE scales.",
    )
    parser.add_argument(
        "--benchmark-batches",
        type=int,
        default=300,
        help="Exact number of preloaded full batches per warmup and measured repeat.",
    )
    parser.add_argument(
        "--max-config-duration-sec",
        type=float,
        default=0.0,
        help="Wall-clock ceiling for one batch-size configuration; 0 disables it.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./benchmark_results", help="Output directory."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device category (jetson, h100, h200, a6000, 5090, cpu, rpi5, rpi5_hailo26, uno_q). If omitted, auto-detects.",
    )
    parser.add_argument(
        "--power-modes",
        type=str,
        default=None,
        help="Comma-separated power mode IDs (Jetson: 0=MAXN,1=15W,2=30W,3=50W). Default: device-appropriate.",
    )
    parser.add_argument(
        "--power-mode-switching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply and verify each requested hardware power mode.",
    )
    parser.add_argument(
        "--power-mode-stabilization-sec",
        type=float,
        default=100.0,
        help="Wait after switching and verifying each power mode (default: 100 seconds).",
    )
    parser.add_argument(
        "--max-sources", type=int, default=None, help="Optional maximum number of source files."
    )
    parser.add_argument(
        "--max-windows", type=int, default=None, help="Optional max complete windows per source."
    )
    parser.add_argument(
        "--source-pattern",
        type=str,
        default="**/*.csv",
        help="Recursive glob used inside --data-dir.",
    )
    parser.add_argument(
        "--data-cache-dir",
        type=str,
        default=None,
        help="Persistent NumPy preprocessing cache shared by model runs.",
    )
    parser.add_argument(
        "--exclude-source", action="append", default=None, help="Filename to exclude; repeatable."
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default=None,
        help="Target column; defaults to aggregate/target/y/value auto-detection.",
    )
    parser.add_argument(
        "--time-column",
        type=str,
        default=None,
        help="Timestamp column; defaults to time/timestamp/datetime/date auto-detection.",
    )
    parser.add_argument(
        "--window-stride", type=int, default=None, help="Window stride; defaults to horizon length."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="PyTorch DataLoader workers for benchmark batch preparation.",
    )
    parser.add_argument(
        "--save-raw-performance-metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save per-repeat performance rows; disabled by default to conserve disk space.",
    )
    parser.add_argument(
        "--sleep-between-batches",
        type=float,
        default=300.0,
        help="Mandatory cooling time (s) between batch sizes.",
    )
    parser.add_argument(
        "--idle-baseline-sec",
        type=float,
        default=30.0,
        help="Duration (s) to measure idle power baseline before each config.",
    )
    parser.add_argument("--max-idle-gpu-util", type=float, default=20.0)
    parser.add_argument("--max-idle-cpu-util", type=float, default=30.0)
    parser.add_argument(
        "--log-file", type=str, default=None, help="Optional per-run log file path."
    )

    args = parser.parse_args()
    configure_logging(args.log_file)
    os.environ["BENCHMARK_SEED"] = str(args.seed)

    if args.data and not args.data_dir:
        parser.error(
            "--data no longer accepts a combined file; pass the directory with --data-dir."
        )
    if not args.data_dir:
        parser.error("--data-dir is required.")

    if args.device:
        device_cat = DeviceCategory(args.device.lower())
        logger.info(f"[DEVICE] Using manually specified device: {device_cat.value}")
    else:
        device_cat = auto_detect_device()
        logger.info(f"[DEVICE] Auto-detected device: {device_cat.value}")
    os.environ["BENCHMARK_EXECUTION_DEVICE"] = (
        "cpu" if device_cat in CPU_EXECUTION_CATEGORIES else "cuda"
    )

    if args.power_modes:
        power_modes = [m.strip() for m in args.power_modes.split(",")]
    elif device_cat == DeviceCategory.JETSON:
        power_modes = ["0", "1", "2", "3"]
    else:
        power_modes = ["0"]

    if args.repeats < 1:
        logger.warning(f"[CONFIG] --repeats {args.repeats} invalid; setting to 1")
        args.repeats = 1
    if args.warmups < 0:
        logger.warning(f"[CONFIG] --warmups {args.warmups} invalid; setting to 0")
        args.warmups = 0
    if args.num_workers < 0:
        logger.warning(f"[CONFIG] --num-workers {args.num_workers} invalid; setting to 0")
        args.num_workers = 0

    config = BenchmarkConfig(
        phase=args.phase,
        device_category=device_cat,
        models=[m.strip() for m in args.models.split(",") if m.strip()],
        context_lens=[int(x) for x in args.context_lens.split(",")],
        horizon_lens=[int(x) for x in args.horizon_lens.split(",")],
        batch_sizes=[int(x) for x in args.batch_sizes.split(",")],
        repeats=args.repeats,
        warmups=args.warmups,
        prediction_batch_size=args.prediction_batch_size,
        benchmark_batches=args.benchmark_batches,
        max_config_duration_sec=(
            args.max_config_duration_sec if args.max_config_duration_sec > 0 else None
        ),
        benchmark_seed=args.seed,
        accuracy_seasonality=args.accuracy_seasonality,
        max_idle_gpu_util_percent=args.max_idle_gpu_util,
        max_idle_cpu_util_percent=args.max_idle_cpu_util,
        power_modes=power_modes,
        power_mode_switching=args.power_mode_switching,
        power_mode_stabilization_sec=args.power_mode_stabilization_sec,
        idle_baseline_duration_sec=args.idle_baseline_sec,
        output_dir=args.output_dir,
        dataset_folder=args.dataset_folder,
        pollutant=args.pollutant,
        sleep_between_batches_sec=args.sleep_between_batches,
        max_sources=args.max_sources,
        max_windows=args.max_windows,
        num_workers=args.num_workers,
        save_raw_performance_metrics=args.save_raw_performance_metrics,
        source_pattern=args.source_pattern,
        data_cache_dir=args.data_cache_dir,
        excluded_source_names=args.exclude_source or [],
        target_column=args.target_column,
        time_column=args.time_column,
        window_stride=args.window_stride,
    )

    if device_cat == DeviceCategory.JETSON:
        from benchmark.devices.jetson import JetsonProfile
        from benchmark.monitors.jetson import JetsonPowerMonitor

        device_profile = JetsonProfile()
        power_monitor = JetsonPowerMonitor(interval_ms=50)
    elif device_cat in DISCRETE_NVIDIA_CATEGORIES:
        from benchmark.devices.nvml import NVMLProfile
        from benchmark.monitors.nvml import NVMLPowerMonitor

        device_profile = NVMLProfile(category=device_cat)
        power_monitor = NVMLPowerMonitor(
            interval_ms=50,
            gpu_index=int(os.environ.get("BENCHMARK_GPU_INDEX", "0")),
        )
    elif device_cat in CPU_EXECUTION_CATEGORIES:
        from benchmark.devices.cpu import CPUProfile

        device_profile = CPUProfile(category=device_cat)
        if device_cat == DeviceCategory.CPU:
            from benchmark.monitors.rapl import RAPLPowerMonitor

            power_monitor = RAPLPowerMonitor(interval_ms=50)
        else:
            from benchmark.monitors.sysfs import SysfsPowerMonitor

            power_monitor = SysfsPowerMonitor(interval_ms=50)
            if device_cat in {DeviceCategory.RPI5_HAILO26, DeviceCategory.UNO_Q}:
                logger.warning(
                    "[ACCELERATOR] %s is registered, but current TSFM adapters execute through PyTorch CPU. "
                    "NPU results require a vendor-runtime model adapter and converted model artifact.",
                    device_cat.value,
                )
    else:
        raise ValueError(f"Unsupported device category: {device_cat}")

    if device_cat == DeviceCategory.JETSON:
        logger.info(
            "[METRICS] Jetson coverage: PyTorch GPU allocated/reserved memory, process-tree CPU RAM, "
            "VDD_GPU_SOC/VDD_CPU_CV/VIN_SYS_5V0 power rails, GPU/CPU temperature, GR3D utilization, and throughput."
        )
    elif device_cat == DeviceCategory.CPU:
        logger.info(
            "[METRICS] CPU coverage: process-tree RAM, wrap-aware Intel RAPL package power/energy, "
            "CPU package temperature, and throughput. GPU temperature is N/A because inference_device=cpu."
        )
    elif device_cat in CPU_EXECUTION_CATEGORIES:
        logger.info(
            "[METRICS] ARM CPU coverage: process-tree RAM, configured sysfs board-power sensor, "
            "CPU temperature, and throughput. Accelerator telemetry requires a vendor-runtime adapter."
        )

    runner = BenchmarkRunner(config, device_profile, power_monitor)

    runner.run(args.data_dir)
    logger.info("[COMPLETE] Finished source-directory benchmark: %s", args.data_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("[FATAL] Benchmark entry point crashed")
        raise
