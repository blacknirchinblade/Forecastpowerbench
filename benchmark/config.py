"""Define validated benchmark protocol, execution-status, device-category, and result-row schemas.
"""

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, PositiveInt


class DeviceCategory(str, Enum):
    """Define the DeviceCategory type and its associated runtime state or contract.
    
    Bases:
        str, Enum
    """
    JETSON = "jetson"
    A6000 = "a6000"
    H100 = "h100"
    H200 = "h200"
    RTX5090 = "5090"
    CPU = "cpu"
    RPI5 = "rpi5"
    RPI5_HAILO26 = "rpi5_hailo26"
    UNO_Q = "uno_q"


CPU_EXECUTION_CATEGORIES = frozenset(
    {
        DeviceCategory.CPU,
        DeviceCategory.RPI5,
        DeviceCategory.RPI5_HAILO26,
        DeviceCategory.UNO_Q,
    }
)

DISCRETE_NVIDIA_CATEGORIES = frozenset(
    {
        DeviceCategory.A6000,
        DeviceCategory.H100,
        DeviceCategory.H200,
        DeviceCategory.RTX5090,
    }
)


class ExecutionStatus(str, Enum):
    """Define the ExecutionStatus type and its associated runtime state or contract.
    
    Bases:
        str, Enum
    """
    SUCCESS = "SUCCESS"
    OOM = "OOM"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"
    SWAP_CONTAMINATED = "SWAP_CONTAMINATED"
    HARDWARE_THROTTLED = "HARDWARE_THROTTLED"


class BenchmarkConfig(BaseModel):
    """Validate and carry the complete two-phase benchmark protocol.
    
    Bases:
        BaseModel
    
    Attributes:
        accuracy_seasonality: Accuracy seasonality state retained by each instance.
        batch_sizes: Batch sizes state retained by each instance.
        benchmark_batches: Benchmark batches state retained by each instance.
        benchmark_seed: Benchmark seed state retained by each instance.
        context_lens: Context lens state retained by each instance.
        data_cache_dir: Data cache dir state retained by each instance.
        dataset_folder: Dataset folder state retained by each instance.
        device_category: Device category state retained by each instance.
        excluded_source_names: Excluded source names state retained by each instance.
        horizon_lens: Horizon lens state retained by each instance.
        idle_baseline_duration_sec: Idle baseline duration sec state retained by each instance.
        max_config_duration_sec: Max config duration sec state retained by each instance.
        max_idle_cpu_util_percent: Max idle cpu util percent state retained by each instance.
        max_idle_gpu_util_percent: Max idle gpu util percent state retained by each instance.
        max_sources: Optional maximum number of source files retained after stable sorting.
        max_windows: Max windows state retained by each instance.
        models: Models state retained by each instance.
        num_workers: PyTorch DataLoader worker-process count.
        output_dir: Directory containing run artifacts.
        phase: Phase state retained by each instance.
        pollutant: Pollutant state retained by each instance.
        power_mode_stabilization_sec: Power mode stabilization sec state retained by each instance.
        power_mode_switching: Power mode switching state retained by each instance.
        power_modes: Power modes state retained by each instance.
        prediction_batch_size: Fixed batch size used only for deterministic Phase-1 prediction generation.
        repeats: Repeats state retained by each instance.
        save_raw_performance_metrics: Save raw performance metrics state retained by each instance.
        sleep_between_batches_sec: Sleep between batches sec state retained by each instance.
        sleep_between_models_sec: Sleep between models sec state retained by each instance.
        source_pattern: Source pattern state retained by each instance.
        target_column: Optional explicit forecast-target column name.
        time_column: Optional explicit timestamp column name.
        warmups: Warmups state retained by each instance.
        window_stride: Window stride state retained by each instance.
    """
    phase: str = Field(default="benchmark", pattern="^(prediction|benchmark)$")
    device_category: DeviceCategory
    models: List[str]
    context_lens: List[PositiveInt]
    horizon_lens: List[PositiveInt]
    batch_sizes: List[PositiveInt]
    repeats: int = Field(default=5, ge=1)
    warmups: int = Field(default=2, ge=0)
    prediction_batch_size: int = Field(default=512, ge=1)
    benchmark_batches: int = Field(default=300, ge=1)
    max_config_duration_sec: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Optional wall-clock ceiling for one batch-size configuration.",
    )
    benchmark_seed: int = Field(
        default=12345,
        ge=0,
        description="Single root seed used by Python, NumPy, PyTorch, CUDA, and loaders.",
    )
    accuracy_seasonality: int = Field(
        default=24,
        ge=1,
        description="Seasonal-naive lag used to save context-derived MASE/RMSSE scales.",
    )

    power_modes: List[str] = Field(
        default_factory=lambda: ["0"],
        description="Power mode IDs to sweep (Jetson: 0=MAXN, 1=15W, 2=30W, 3=50W)",
    )
    power_mode_switching: bool = True
    power_mode_stabilization_sec: float = Field(
        default=100.0,
        ge=0.0,
        description="Wait after each verified power-mode switch before baseline and inference.",
    )
    idle_baseline_duration_sec: float = Field(
        default=30.0,
        ge=5.0,
        description="Duration to measure idle power baseline before each config.",
    )
    max_idle_gpu_util_percent: float = Field(default=20.0, ge=0.0, le=100.0)
    max_idle_cpu_util_percent: float = Field(default=30.0, ge=0.0, le=100.0)

    sleep_between_models_sec: float = Field(
        default=300.0, ge=0.0, description="Mandatory cooling period between models."
    )
    sleep_between_batches_sec: float = Field(
        default=300.0, ge=0.0, description="Mandatory cooling period between batch sizes."
    )

    save_raw_performance_metrics: bool = False
    output_dir: str = "./benchmark_results"
    dataset_folder: str = ""
    pollutant: str = ""
    max_sources: Optional[int] = Field(default=None, ge=1)
    max_windows: Optional[int] = Field(
        default=None, ge=1, description="Optional cap on windows per source."
    )
    num_workers: int = Field(
        default=0, ge=0, description="PyTorch DataLoader workers for benchmark batch preparation."
    )
    source_pattern: str = "**/*.csv"
    data_cache_dir: Optional[str] = None
    excluded_source_names: List[str] = Field(default_factory=list)
    target_column: Optional[str] = None
    time_column: Optional[str] = None
    window_stride: Optional[int] = Field(
        default=None, ge=1, description="Window stride; defaults to horizon length."
    )


class ResultRow(BaseModel):
    # Identifiers
    """Represent one measured repeat or terminal configuration.
    
    Bases:
        BaseModel
    
    Attributes:
        avg_cpu_cv_power_w: Avg cpu cv power w state retained by each instance.
        avg_cpu_temp_c: Avg cpu temp c state retained by each instance.
        avg_gpu_compute_util_percent: Avg gpu compute util percent state retained by each instance.
        avg_gpu_memory_used_gb: Avg gpu memory used gb state retained by each instance.
        avg_gpu_memory_util_percent: Avg gpu memory util percent state retained by each instance.
        avg_gpu_soc_power_w: Avg gpu soc power w state retained by each instance.
        avg_gpu_temp_c: Avg gpu temp c state retained by each instance.
        avg_power_w: Avg power w state retained by each instance.
        avg_system_5v_power_w: Avg system 5v power w state retained by each instance.
        batch_size: Number of source-local windows passed to one model call.
        benchmark_seed: Benchmark seed state retained by each instance.
        context_len: Number of historical points supplied for each forecast window.
        dataset_name: Stable dataset identifier written to output metadata.
        device_category: Device category state retained by each instance.
        dynamic_avg_power_w: Dynamic avg power w state retained by each instance.
        dynamic_edp_joule_seconds: Dynamic edp joule seconds state retained by each instance.
        dynamic_energy_j: Dynamic energy j state retained by each instance.
        dynamic_energy_per_1k_points_j: Dynamic energy per 1k points j state retained by each instance.
        dynamic_energy_per_window_j: Dynamic energy per window j state retained by each instance.
        e2e_latency_ms: E2e latency ms state retained by each instance.
        edp_joule_seconds: Edp joule seconds state retained by each instance.
        energy_per_1k_points_j: Energy per 1k points j state retained by each instance.
        energy_per_window_j: Energy per window j state retained by each instance.
        execution_device: Execution device state retained by each instance.
        full_batches_per_iteration: Full batches per iteration state retained by each instance.
        gpu_compute_util_std_percent: Gpu compute util std percent state retained by each instance.
        gpu_memory_scope: Gpu memory scope state retained by each instance.
        gpu_memory_total_gb: Gpu memory total gb state retained by each instance.
        gpu_memory_util_std_percent: Gpu memory util std percent state retained by each instance.
        horizon_len: Number of future points requested from the model.
        idle_baseline_w: Idle baseline w state retained by each instance.
        idle_cpu_util_percent: Idle cpu util percent state retained by each instance.
        idle_gpu_compute_util_percent: Idle gpu compute util percent state retained by each instance.
        jetson_oc1_event_delta: Jetson oc1 event delta state retained by each instance.
        jetson_oc2_event_delta: Jetson oc2 event delta state retained by each instance.
        jetson_oc3_event_delta: Jetson oc3 event delta state retained by each instance.
        jetson_oc_counter_available: Jetson oc counter available state retained by each instance.
        jetson_oc_event_delta: Jetson oc event delta state retained by each instance.
        latency_per_window_ms: Latency per window ms state retained by each instance.
        max_gpu_temp_c: Max gpu temp c state retained by each instance.
        measurement_cycles: Measurement cycles state retained by each instance.
        measurement_duration_sec: Measurement duration sec state retained by each instance.
        measurement_scope: Measurement scope state retained by each instance.
        min_gpu_memory_free_gb: Min gpu memory free gb state retained by each instance.
        model_call_duration_sec: Model call duration sec state retained by each instance.
        model_call_latency_per_batch_ms: Model call latency per batch ms state retained by each instance.
        model_call_latency_per_window_ms: Model call latency per window ms state retained by each instance.
        model_call_throughput_points_per_sec: Model call throughput points per sec state retained by each instance.
        model_call_throughput_windows_per_sec: Model call throughput windows per sec state retained by each instance.
        model_name: Registered model adapter name.
        num_workers: PyTorch DataLoader worker-process count.
        oom_ceiling_batch_size: Oom ceiling batch size state retained by each instance.
        peak_cpu_cv_power_w: Peak cpu cv power w state retained by each instance.
        peak_cpu_ram_gb: Peak cpu ram gb state retained by each instance.
        peak_gpu_allocated_gb: Peak gpu allocated gb state retained by each instance.
        peak_gpu_memory_used_gb: Peak gpu memory used gb state retained by each instance.
        peak_gpu_reserved_gb: Peak gpu reserved gb state retained by each instance.
        peak_gpu_soc_power_w: Peak gpu soc power w state retained by each instance.
        peak_power_w: Peak power w state retained by each instance.
        peak_system_5v_power_w: Peak system 5v power w state retained by each instance.
        pipeline_overhead_percent: Pipeline overhead percent state retained by each instance.
        pipeline_overhead_sec: Pipeline overhead sec state retained by each instance.
        pollutant: Pollutant state retained by each instance.
        power_mode: Requested device power-mode identifier.
        power_mode_stabilization_sec: Power mode stabilization sec state retained by each instance.
        power_mode_verification: Power mode verification state retained by each instance.
        power_p50_w: Power p50 w state retained by each instance.
        power_p95_w: Power p95 w state retained by each instance.
        power_sample_count: Power sample count state retained by each instance.
        power_scope: Power scope state retained by each instance.
        power_std_w: Power std w state retained by each instance.
        processed_points: Processed points state retained by each instance.
        processed_windows: Processed windows state retained by each instance.
        repeat_id: Repeat id state retained by each instance.
        source_count: Source count state retained by each instance.
        status: Status state retained by each instance.
        telemetry_duration_sec: Telemetry duration sec state retained by each instance.
        throughput_points_per_sec: Throughput points per sec state retained by each instance.
        throughput_windows_per_sec: Throughput windows per sec state retained by each instance.
        total_energy_j: Total energy j state retained by each instance.
        total_windows: Total windows state retained by each instance.
        warmup_batches_per_iteration: Warmup batches per iteration state retained by each instance.
        warmup_scope: Warmup scope state retained by each instance.
        warmup_windows_per_iteration: Warmup windows per iteration state retained by each instance.
        window_stride: Window stride state retained by each instance.
    """
    benchmark_seed: int = 12345
    model_name: str
    model_artifact: str = ""
    model_variant: str = ""
    device_category: str
    execution_device: str = ""
    context_len: int
    horizon_len: int
    batch_size: int
    num_workers: int = 0
    repeat_id: int
    power_mode: str = "N/A"
    power_mode_verification: str = "NOT_VERIFIED"
    power_mode_stabilization_sec: float = 0.0
    dataset_name: str = ""
    pollutant: str = ""
    measurement_scope: str = "fixed_preloaded_batches"
    source_count: int = 0
    total_windows: int = 0
    processed_windows: int = 0
    processed_points: int = 0
    measurement_cycles: int = 0
    window_stride: int = 0
    warmup_scope: str = "fixed_preloaded_batches"
    warmup_batches_per_iteration: int = 0
    warmup_windows_per_iteration: int = 0
    full_batches_per_iteration: int = 0

    # Execution Status
    status: ExecutionStatus
    oom_ceiling_batch_size: Optional[int] = None

    # Latency & Throughput
    measurement_duration_sec: float = float("nan")
    e2e_latency_ms: float = float("nan")
    latency_per_window_ms: float = float("nan")
    throughput_windows_per_sec: float = float("nan")
    throughput_points_per_sec: float = float("nan")
    model_call_duration_sec: float = float("nan")
    model_call_latency_per_batch_ms: float = float("nan")
    model_call_latency_per_window_ms: float = float("nan")
    model_call_throughput_windows_per_sec: float = float("nan")
    model_call_throughput_points_per_sec: float = float("nan")
    pipeline_overhead_sec: float = float("nan")
    pipeline_overhead_percent: float = float("nan")

    # Power & Energy
    telemetry_duration_sec: float = float("nan")
    idle_baseline_w: float = float("nan")
    idle_gpu_compute_util_percent: float = float("nan")
    idle_cpu_util_percent: float = float("nan")
    avg_power_w: float = float("nan")
    power_scope: str = "unavailable"
    peak_power_w: float = float("nan")
    power_sample_count: int = 0
    power_std_w: float = float("nan")
    power_p50_w: float = float("nan")
    power_p95_w: float = float("nan")
    total_energy_j: float = float("nan")
    energy_per_window_j: float = float("nan")
    energy_per_1k_points_j: float = float("nan")
    edp_joule_seconds: float = float("nan")
    dynamic_avg_power_w: float = float("nan")
    dynamic_energy_j: float = float("nan")
    dynamic_energy_per_window_j: float = float("nan")
    dynamic_energy_per_1k_points_j: float = float("nan")
    dynamic_edp_joule_seconds: float = float("nan")
    # Jetson AGX Orin rails. VDD_GPU_SOC includes GPU and SoC engines, so it
    # must not be labelled as GPU-only power.
    avg_gpu_soc_power_w: float = float("nan")
    peak_gpu_soc_power_w: float = float("nan")
    avg_cpu_cv_power_w: float = float("nan")
    peak_cpu_cv_power_w: float = float("nan")
    avg_system_5v_power_w: float = float("nan")
    peak_system_5v_power_w: float = float("nan")
    jetson_oc_counter_available: bool = False
    jetson_oc_event_delta: int = 0
    jetson_oc1_event_delta: int = 0
    jetson_oc2_event_delta: int = 0
    jetson_oc3_event_delta: int = 0

    # Thermal
    avg_gpu_temp_c: float = float("nan")
    max_gpu_temp_c: float = float("nan")
    avg_cpu_temp_c: float = float("nan")

    # Memory & Utilization
    peak_gpu_allocated_gb: float = float("nan")
    peak_gpu_reserved_gb: float = float("nan")
    gpu_memory_scope: str = ""
    gpu_memory_total_gb: float = float("nan")
    avg_gpu_memory_used_gb: float = float("nan")
    peak_gpu_memory_used_gb: float = float("nan")
    min_gpu_memory_free_gb: float = float("nan")
    peak_cpu_ram_gb: float = float("nan")
    avg_gpu_compute_util_percent: float = float("nan")
    avg_gpu_memory_util_percent: float = float("nan")
    gpu_compute_util_std_percent: float = float("nan")
    gpu_memory_util_std_percent: float = float("nan")
