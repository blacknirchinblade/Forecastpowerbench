# Measurement Methodology

This document is the normative definition of PowerBench performance measurement.
When a plot label or paper description conflicts with this document and the recorded
run metadata, the metadata and implementation must be inspected before publication.

## Workload Symbols

For one measured repeat:

- `B` is the number of benchmark batches, default 300.
- `N` is the configured batch size.
- `H` is forecast horizon, default 96.
- `T` is measured end-to-end duration in seconds.
- `M` is the sum of synchronous adapter-call durations.
- `W = B*N` is the number of processed windows.
- `P = W*H` is the number of produced forecast points.

Only complete full batches count. The prepared stream repeats data deterministically
when necessary, but every configured workload executes the same number of calls and
windows.

## Timing Boundary

The end-to-end interval begins immediately before selecting the first prepared batch
and ends after the final adapter call and device synchronization. It includes:

- one zero-copy NumPy slice per batch;
- the complete model adapter call;
- required tensor construction and host-to-device transfer;
- model forward pass, generation, or sampling defined by that adapter;
- point-forecast statistic extraction;
- device-to-host transfer and normalized NumPy return;
- minimal loop bookkeeping and forecast deletion.

It excludes:

- source discovery, CSV parsing, validation, and median imputation;
- window-index construction and benchmark-pool materialization;
- model/checkpoint loading, compilation before warmup, and stabilization;
- idle baseline and warmup iterations;
- progress rendering and normal log output;
- telemetry aggregation, summary statistics, disk writes, and prediction saving;
- cache cleanup, power-mode stabilization, and cooldown.

CUDA is synchronized immediately before timing and after the last call. Without the
final synchronization, host time would under-report asynchronous GPU execution.

## Timing And Throughput Metrics

| Metric | Definition |
|---|---|
| `measurement_duration_sec` | `T`, wall-clock duration of the measured interval. |
| `e2e_latency_ms` | `1000*T/B`, end-to-end latency per benchmark batch. |
| `latency_per_window_ms` | `1000*T/W`. |
| `throughput_windows_per_sec` | `W/T`. |
| `throughput_points_per_sec` | `P/T`. |
| `model_call_duration_sec` | `M`, sum of all synchronous adapter-call durations. |
| `model_call_latency_per_batch_ms` | `1000*M/B`. |
| `model_call_latency_per_window_ms` | `1000*M/W`. |
| `model_call_throughput_windows_per_sec` | `W/M`. |
| `model_call_throughput_points_per_sec` | `P/M`. |
| `pipeline_overhead_sec` | `max(0, T-M)`. |
| `pipeline_overhead_percent` | `100*max(0, T-M)/T`. |

`model_call_*` still includes adapter conversion and transfer. It exists to separate
runner-loop overhead from the adapter, not to claim pure kernel timing.

## Telemetry Lifecycle

Each monitor samples on a background thread at 50 ms (20 Hz). The runner records a
wall-clock nanosecond boundary for the measured interval. At stop:

1. finite utilization, memory, temperature, and rail samples outside the closed
   interval are discarded;
2. power alone receives exact start/end values by linear interpolation (or nearest
   endpoint hold when the boundary is outside the sampled range);
3. non-power statistics use only finite samples actually observed inside the
   interval;
4. power is integrated against the endpoint-enriched timestamps with the
   trapezoidal rule;
5. raw samples are discarded after calculation unless debugging code explicitly
   changes the policy.

Sampling overhead exists on the host and is held constant across batch sizes for a
device. Monitor start/stop and summary calculations are outside `T`; the sampling
thread itself overlaps inference because system behavior must be observed during the
actual workload.

## Idle Baseline

A fresh idle baseline is collected per executed configuration with the model and
prepared input pool resident. This captures device/process residency without model
forward work. It is not an empty-system or wall-plug baseline.

Raw energy remains the primary sensor integral. Dynamic fields subtract the baseline
at the same sensor scope and clamp negative values to zero:

```text
dynamic_avg_power_w = max(0, avg_power_w - idle_baseline_w)
dynamic_energy_j = max(0, total_energy_j
                          - idle_baseline_w * telemetry_duration_sec)
```

## Power And Energy Metrics

| Metric | Definition |
|---|---|
| `power_scope` | Named physical boundary of all generic power fields. |
| `idle_baseline_w` | Time-weighted baseline power with model and pool resident. |
| `avg_power_w` | Time-weighted mean power in the aligned interval. |
| `peak_power_w` | Maximum aligned power sample. |
| `power_sample_count` | Number of aligned samples. |
| `power_std_w` | Population standard deviation of aligned power samples. |
| `power_p50_w`, `power_p95_w` | Aligned-sample 50th and 95th percentiles. |
| `total_energy_j` | Trapezoidal integral of raw power. |
| `energy_per_window_j` | `total_energy_j/W`. |
| `energy_per_1k_points_j` | `total_energy_j/(P/1000)`. |
| `edp_joule_seconds` | `total_energy_j*T`. |
| `dynamic_avg_power_w` | Baseline-corrected average power above. |
| `dynamic_energy_j` | Baseline-corrected energy above. |
| `dynamic_energy_per_window_j` | `dynamic_energy_j/W`. |
| `dynamic_energy_per_1k_points_j` | `dynamic_energy_j/(P/1000)`. |
| `dynamic_edp_joule_seconds` | `dynamic_energy_j*T`. |

## Physical Power Scopes

| Device/monitor | Scope | Interpretation |
|---|---|---|
| H100/H200/RTX 5090/A6000 NVML | `gpu_board` | NVIDIA board power, not CPU, RAM, storage, cloud host, cooling, or whole-system wall power. |
| Intel CPU RAPL | `cpu_package` | Wrap-aware package energy/power, not total server wall power. |
| Jetson | named rail sum | Sum of VDD_GPU_SOC, VDD_CPU_CV, and VIN_SYS_5V0 readings; not GPU-only and not an external wall meter. |
| ARM sysfs | `configured_board_sensor` | Only the explicitly readable configured sensor; blank when unavailable. |

Jetson also retains rail-specific average and peak columns. `VDD_GPU_SOC` covers GPU
plus SoC and must not be renamed GPU-only power. Values from different scopes can be
reported side by side only when the scope distinction remains explicit.

## Memory Metrics

Memory is intentionally measured at multiple layers:

| Metric | Definition |
|---|---|
| `peak_gpu_allocated_gb` | Peak live tensor bytes observed by PyTorch's CUDA allocator. |
| `peak_gpu_reserved_gb` | Peak CUDA caching-allocator reservation. |
| `gpu_memory_total_gb` | Physical dedicated/shared capacity reported by the monitor. |
| `avg_gpu_memory_used_gb` | Mean aligned physical GPU/shared memory in use. |
| `peak_gpu_memory_used_gb` | Maximum aligned physical GPU/shared memory in use. |
| `min_gpu_memory_free_gb` | Minimum aligned physical free memory. |
| `gpu_memory_scope` | `dedicated_vram`, `shared_unified_system_memory`, or `not_applicable`. |
| `peak_cpu_ram_gb` | Peak RSS of the benchmark process plus recursive child processes. |

Allocator memory and physical VRAM answer different questions. Reserved memory may
remain flat because PyTorch reuses cached blocks; physical memory includes CUDA
context and libraries; process RSS includes the resident model and prepared pool.
Therefore memory is not required to rise monotonically at every batch size. A flat
curve is not itself proof of a benchmark defect.

`ProcessMemorySampler` uses psutil every 50 ms and recursively sums child RSS. It is
started and stopped with telemetry for each measured repeat.

## Utilization And Thermal Metrics

| Metric | Definition |
|---|---|
| `avg_gpu_compute_util_percent` | Mean aligned GPU/GR3D compute utilization. |
| `avg_gpu_memory_util_percent` | Mean physical capacity usage, `used/total*100`; not memory-controller bandwidth. |
| `gpu_compute_util_std_percent` | Population standard deviation of compute samples. |
| `gpu_memory_util_std_percent` | Population standard deviation of capacity-use samples. |
| `avg_gpu_temp_c`, `max_gpu_temp_c` | Mean and maximum aligned GPU temperature. |
| `avg_cpu_temp_c` | Mean aligned CPU/package temperature. |

Unsupported sensors remain blank/NaN and retain an explicit zero sample count. A
different available sensor is never relabeled to fill a missing physical scope.

## Repeat Aggregation

For each configuration and numeric metric `x`, successful repeats produce:

```text
x_mean = arithmetic mean
x_std  = sample standard deviation with ddof=1
x_ci95 = t(0.975, n-1) * x_std / sqrt(n)
```

With five repeats, the Student-t multiplier is approximately 2.776. With two CPU
repeats, uncertainty is much less stable and should be interpreted accordingly. For
one valid observation, the implementation writes `0.0` for standard deviation and
CI half-width because no between-repeat dispersion can be estimated; such a row must
not be interpreted as evidence of zero uncertainty.

Power-sample standard deviations and repeat-level summary standard deviations are
different quantities: `power_std_w` describes variation among 50 ms samples inside
one repeat; `power_std_w_std` in a summary describes variation of that statistic
across repeats.

## Validity Status

| Status | Meaning |
|---|---|
| `SUCCESS` | Repeat completed with required measurements and no detected contamination. |
| `OOM` | Actual or safely projected memory ceiling. |
| `TIMEOUT` | Configuration exceeded its allowed execution time. |
| `ERROR` | Model/runtime or validation failure not classified elsewhere. |
| `SWAP_CONTAMINATED` | Swap changed during the protected interval/run. |
| `HARDWARE_THROTTLED` | Jetson OC or Raspberry Pi throttle evidence invalidated the repeat. |

Summary identifiers include status counts, repeat IDs, missing repeat IDs, and
`is_complete_success`. Invalid repeats do not silently contribute to successful
metric averages.

## Memory Safety Guard

After completed CUDA batch sizes, PowerBench observes peak reserved memory outside
the timed/sampled interval. Using the two latest observations, it fits a non-negative
linear projection, multiplies by default headroom 1.08, and compares against 90% of
physical memory (or an explicit hard limit). A projected violation records the OOM
ceiling without executing the unsafe configuration.

This guard is model- and observation-driven. It does not change any successful
configuration's model call or measured interval. Its purpose is to prevent native
allocator/Triton failures that could corrupt a long multi-model study.

## Accuracy Boundary

Accuracy never belongs to a performance configuration. Phase 1 predictions are
generated once at fixed batch 512 and later evaluated from NPZ artifacts. The
notebook uses `utils.metrics.compute_metrics`; imputed forecast targets are excluded
by `y_true_observed`.

Let `V` be the finite, originally observed forecast points, `n=|V|`, and
`e_i=y_pred_i-y_true_i`:

```text
MAE   = mean_V(|e|)
MSE   = mean_V(e^2)
RMSE  = sqrt(MSE)
MAPE  = 100*mean_V(|e/y|)
sMAPE = 100*mean_V(2|e|/(|y|+|y_pred|))
R2    = 1 - sum_V(e^2)/sum_V((y-mean_V(y))^2)
ND    = sum_V(|e|)/sum_V(|y|)
WAPE  = 100*sum_V(|e|)/sum_V(|y|)
NRMSE = RMSE/(mean_V(|y|) + 1e-8)
Bias  = mean_V(e)
```

These shared formulas reproduce the historical TFB/TSFM-Bench-compatible
`tsfm_metrics.py` contract. Therefore MAPE is undefined when a retained actual is
zero, and an sMAPE term is undefined when both actual and prediction are zero.
`R2` is NaN for fewer than two valid points or zero target variance. For a
positive target absolute sum, WAPE is exactly `100*ND`; they must not be plotted
or discussed as independent evidence. The `1e-8` NRMSE stabilizer is retained
solely for formula compatibility and makes all-zero targets produce a large
finite NRMSE rather than NaN.

For configured seasonality `m=24`, window `w` receives context-only scales:

```text
a_w = mean_{t=m..C-1}(|x[w,t] - x[w,t-m]|)
q_w = mean_{t=m..C-1}((x[w,t] - x[w,t-m])^2)

MASE  = mean_{(w,h) in V, a_w>0}(|e[w,h]|/a_w)
RMSSE = sqrt(mean_{(w,h) in V, q_w>0}(e[w,h]^2/q_w))
```

The per-window scale is broadcast over valid observed points in that window's
horizon. It is derived from historical context, never future targets. Undefined or
zero seasonal scales are excluded; the metric is NaN when no valid scale remains.

Raw forecasts are not clipped or calibrated. Any physically non-negative PM2.5
analysis must be labeled secondary and cannot replace raw primary metrics.

## Metric Selection Rationale

| Metric family | Why it is retained | Important limitation |
|---|---|---|
| Windows/s and ms/window | Directly compare completed forecasting work across heterogeneous model APIs. | Adapter-inclusive, not kernel-only. |
| Points/s | Makes horizon-dependent produced work explicit. | Redundant with windows/s when horizon is fixed. |
| Raw energy/window | Sensor integral normalized by a common forecasting unit. | Comparable only with the physical `power_scope` retained. |
| Dynamic energy/window | Estimates energy above model-resident idle state. | Depends on baseline stability; raw energy remains primary. |
| EDP | Exposes the joint energy/latency trade-off. | Scale depends on the selected work unit and sensor scope. |
| Allocator and physical memory | Separates tensor demand, cached reservation, and device occupancy. | No individual layer is a complete memory explanation. |
| Process-tree RSS | Captures model, prepared pool, and child-process host memory. | Sampling can miss a very short-lived peak. |
| Mean, sample std, t-CI | Reports central tendency and finite-repeat uncertainty. | Two-repeat CPU intervals are weak; one repeat is not estimable. |
| MAE/RMSE | Preserve error units and penalize absolute/large errors differently. | Scale-dependent across pollutants or datasets. |
| sMAPE/NRMSE/WAPE | Add scale-normalized views. | Ratio metrics are unstable near zero; WAPE is ND expressed as a percentage for positive denominators. |
| MASE/RMSSE | Compare against a seasonal-naive context scale. | Undefined for zero seasonal variation. |
| Bias | Detects systematic over- or under-prediction. | Positive and negative errors can cancel. |

## Interpretation Rules

- Throughput should often rise and then saturate, but architecture, compilation,
  memory pressure, sampling, and transfer can create non-monotonic local points.
- Power and memory need not be strictly monotonic because clocks, allocator reuse,
  kernel selection, and workload duration vary.
- Compare raw and dynamic energy only within identical sensor scopes.
- Do not compare pre- and post-intervention timing for an adapter whose timed
  transport changed.
- Do not call Raspberry Pi/Hailo rows accelerated unless a model-specific HEF and
  HailoRT adapter actually executed and metadata says so.
- Do not infer forecast accuracy from Phase 2 or benchmark batch size.
