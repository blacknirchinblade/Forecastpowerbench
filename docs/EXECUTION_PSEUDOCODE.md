# Execution Pseudocode

This document abstracts the implemented PowerBench control flow without hiding the
measurement boundary. Names in monospace are real first-party callables; model-specific
tensor conversion remains inside each adapter's `forecast` method.

## Call Ladder

```text
run_unified_benchmark.sh
  -> per-model virtual-environment Python
  -> experiments.forecasting.main()
     -> BenchmarkConfig(...)
     -> select DeviceProfile and PowerMonitor
     -> BenchmarkRunner(config, profile, monitor)
        -> BenchmarkRunner.run(DATA_DIR)
           -> BenchmarkRunner._run_source_directory(DATA_DIR)
              -> load_source_directory(...)
                 -> discover_source_files(...)
                 -> load_source_file(...) or validated NPZ cache
              -> Phase 1: _run_prediction_pass(...)
              -> Phase 2: _run_source_batch_size(...)
                 -> PreparedBenchmarkStream(...)
                 -> models.factory.get_model(...)
                 -> adapter.forecast(batch, horizon)
                 -> PowerMonitor.start()/stop()
                 -> summarize_power_interval(...)
                 -> _summarize_batch_rows(...)
                 -> _append_dict_rows(...)
```

## Data Preparation

```text
function LOAD_DATASET(data_directory, source_pattern):
    paths <- stable_sort(discover_source_files(data_directory, source_pattern))
    sources <- empty ordered collection

    for each path in paths:
        frame <- pandas.read_csv(path)
        time_column, target_column <- resolve configured or inferred columns
        frame <- parse UTC timestamps, stable-sort, collapse duplicate times by median
        observed_mask <- is_finite(target)
        target <- replace infinity with missing
        target <- fill missing values with this source's global finite median
        segments <- split wherever timestamp cadence is discontinuous
        append contiguous float32 SourceSeries to sources

    return sources

function BUILD_WINDOWS(sources, context C, horizon H, stride S):
    for each source independently:
        for each contiguous timestamp segment:
            retain starts where start + C + H <= segment_end
            discard incomplete final remainder
    return MultiSourceWindowDataset(global_index -> source-local complete window)
```

CSV files are never concatenated before windowing, and no window crosses a source or
timestamp-gap boundary.

## Phase 1: Deterministic Prediction Generation

```text
seed Python, NumPy, PyTorch, CUDA, and DataLoader workers with BENCHMARK_SEED
sources <- LOAD_DATASET(DATA_DIR, SOURCE_PATTERN)

for model in selected_models:
    for context C in context_lengths:
        for horizon H in horizon_lengths:
            dataset <- BUILD_WINDOWS(sources, C, H, stride=H unless configured)
            model_adapter <- get_model(model, C, H, batch_size=512)
            loader <- DataLoader(dataset, batch_size=512, shuffle=false, seeded workers)
            prediction_memmap <- float32[len(dataset), H]

            for batch in loader:
                inputs <- contiguous NumPy context batch
                predictions, ignored_time <- model_adapter.forecast(inputs, H)
                validate shape == [batch_windows, H] and finite values
                write predictions into global-index positions in prediction_memmap

            for source in sources:
                select only this source's windows, targets, timestamps, and observed mask
                save one model/source/context/horizon NPZ atomically

            close workers and memory map; release model and accelerator cache
```

There is no power monitor, RSS sampler, performance result row, or accuracy calculation
in Phase 1.

## Phase 2: Outer Benchmark Loop

```text
verify device exclusivity
device_profile.setup()
sources <- LOAD_DATASET(DATA_DIR, SOURCE_PATTERN)
swap_baseline <- current swap usage

try:
    for power_mode in configured_power_modes:
        device_profile.apply_power_mode(power_mode)
        assert device_profile.verify_power_mode(power_mode)
        wait POWER_MODE_STABILIZATION_SEC
        assert device_profile.verify_power_mode(power_mode) again

        for model in selected_models:
            if not first model: clear accelerator and cool between models

            for context C in context_lengths:
                for horizon H in horizon_lengths:
                    total_windows <- len(BUILD_WINDOWS(sources, C, H))

                    for batch_size B in ascending_batch_sizes:
                        if projected CPU/GPU memory exceeds safe capacity:
                            record OOM ceiling and stop only this C/H batch sweep

                        if not first configuration: clear accelerator and cool 5 minutes

                        try:
                            RUN_ONE_CONFIGURATION(model, C, H, B, sources)
                        catch timeout or OOM/batch-limit runtime:
                            record terminal row
                            clear accelerator
                            stop only larger batches for this C/H
finally:
    device_profile.teardown()  # restore host state even on failure
```

An OOM or timeout advances to the next horizon/context rather than terminating the
remaining model protocol.

## One Batch-Size Configuration

```text
function RUN_ONE_CONFIGURATION(model_name, C, H, B, sources):
    seed all RNGs with the same BENCHMARK_SEED
    model <- get_model(model_name, device, C, H, B, workers)
    wait MODEL_STABILIZATION_SEC

    dataset <- BUILD_WINDOWS(sources, C, H)
    prepared <- PreparedBenchmarkStream(
        dataset,
        batch_size=B,
        batch_count=BENCHMARK_BATCHES,       # normally 300
        bounded_preload_pool
    )
    # Preparation is complete before timing. prepared.batch(i) is a contiguous view.

    idle_baseline <- power_monitor.measure_idle_baseline(IDLE_BASELINE_SEC)
    repeat_rows <- empty in-memory collection
    arm optional wall-clock timeout for this complete configuration

    for iteration in range(WARMUPS + REPEATS):
        warmup <- iteration < WARMUPS
        processed_windows <- 0
        model_call_duration <- 0

        if not warmup:
            rss_sampler.start()
            power_monitor.start()            # parallel 50 ms collection begins

        measurement_start_ns <- wall_clock_nanoseconds()
        started <- monotonic_high_resolution_clock()

        for batch_index in range(BENCHMARK_BATCHES):
            input_view <- prepared.batch(batch_index)
            call_started <- monotonic_high_resolution_clock()
            predictions, adapter_time <- model.forecast(input_view, H)
            model_call_duration += monotonic_clock() - call_started
            processed_windows += len(input_view)
            delete predictions

        synchronize CUDA when CUDA is active
        finished <- monotonic_high_resolution_clock()
        measurement_end_ns <- wall_clock_nanoseconds()

        if not warmup:
            telemetry <- power_monitor.stop()
            telemetry <- align telemetry to [measurement_start_ns, measurement_end_ns]
            peak_rss <- rss_sampler.stop()

        update progress display outside the timed and sampled region

        if warmup:
            discard timing and continue

        duration <- finished - started
        status <- SUCCESS unless swap/throttle validation rejects the repeat
        repeat_row <- DERIVE_REPEAT_METRICS(
            duration, model_call_duration, processed_windows, H,
            telemetry, idle_baseline, peak_rss, allocator_peaks, status
        )
        append repeat_row to repeat_rows

    disarm timeout; destroy prepared data/model; clear accelerator
    summary <- mean, sample_std(ddof=1), Student_t_95_percent_CI(repeat_rows)
    append summary once to benchmark_results_batch_summary.csv
```

Warmups execute the same number and order of prepared batches as measured repeats but
do not start telemetry and do not produce result rows.

## Parallel Telemetry Loop

```text
function POWER_MONITOR_THREAD(interval = 50 ms):
    next_deadline <- monotonic_clock()
    while running:
        timestamp_ns <- wall_clock_nanoseconds()
        sample power, utilization, memory, and temperatures supported by this device
        append numeric sample and timestamp to in-memory buffers
        next_deadline += interval
        sleep(max(0, next_deadline - monotonic_clock()))

function STOP_AND_SUMMARIZE(start_ns, end_ns):
    stop and join collection thread/process
    retain finite non-power samples observed inside [start_ns, end_ns]
    interpolate or endpoint-hold power at exact interval boundaries
    energy_j <- trapezoidal_integral(endpoint_enriched_power_w, elapsed_seconds)
    avg_power_w <- energy_j / telemetry_duration
    calculate sample count, std, p50, p95, peak, memory, utilization, temperatures
    return aggregate metrics; do not persist raw telemetry rows
```

The discrete-GPU backend uses NVML, Jetson uses `tegrastats` plus sysfs/OC counters,
and CPU-class devices use RAPL or available sysfs sensors. Each row records its physical
power and memory scope so unlike scopes are not silently compared.

## Measurement Boundary

Included in end-to-end repeat duration:

- obtaining each preloaded batch view;
- adapter-side input conversion required by the released model API;
- synchronous `model.forecast` calls;
- output creation and immediate deletion;
- final CUDA synchronization.

Excluded from end-to-end repeat duration:

- CSV reading, imputation, window indexing, batch-pool construction, and model loading;
- idle baseline, warmup metrics, cooling, power-mode switching, and stabilization;
- progress redraws, repeat metric arithmetic, confidence intervals, and CSV writes;
- prediction persistence and accuracy calculation.
