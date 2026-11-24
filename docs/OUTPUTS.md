# Output Reference

Every run writes self-describing predictions, performance summaries, provenance, and
logs under one output root. Raw 50 ms telemetry is intentionally transient and raw
per-repeat CSV output is disabled by default.

## Directory Layout

```text
<OUTPUT_DIR>/
|-- benchmark_predictions/
|   `-- <model>/<dataset-folder>/<source>/ctx_<C>_hor_<H>.npz
|-- benchmark_results_batch_summary.csv
|-- benchmark_results_raw.csv              # optional
|-- dataset_preprocessing_report.csv
|-- benchmark_metadata/
|   |-- dataset_manifest.json
|   |-- <model>_prediction.json
|   `-- <model>_benchmark.json
|-- model_logs/
|   |-- <model>_prediction.log
|   `-- <model>_benchmark.log
|-- source_files.sha256
`-- environment/package snapshots produced by the launcher
```

The exact output root is printed in the launch header. If an existing directory
would mix independent fresh studies, the launcher selects a timestamped run folder.
Resume mode continues the selected output root instead.

## Prediction NPZ

One artifact represents one model, source CSV, context, and horizon. Independent
source files remain independently addressable for source-level accuracy and audit.

### Numerical arrays

| Key | Meaning |
|---|---|
| `y_pred` | Raw model point forecasts, shape `(windows, horizon)`. |
| `y_true` | Forecast targets after numerical preprocessing, same shape. |
| `y_true_observed` | Boolean mask identifying targets observed in the original source. |
| `window_id` | Deterministic source-local window identifier. |
| `context_start_index` | Source-row index at which each context starts. |
| `forecast_start_index` | Source-row index at which each forecast starts. |
| `forecast_time_ns` | Nanosecond forecast timestamps, shape `(windows, horizon)`. |
| `horizon_step` | Zero-based offsets within the forecast horizon. |
| `context_mae_scale`, `context_mse_scale` | Context-derived seasonal-naive denominators for later MASE/RMSSE. |

### Identity and protocol metadata

Artifacts also record `source_id`, source relative/absolute path and filename,
dataset group/name, selected columns, pollutant, model/device/execution device,
context, horizon, stride, fixed reference batch, seed, seasonality, imputation
policy, and active power-mode label. Strings/scalars are stored in an NPZ-safe
representation and validated on load.

Forecasts must be finite and match the expected shape. Imputed future targets remain
in `y_true` for shape consistency but are excluded from accuracy by
`y_true_observed`.

## Performance Summary CSV

`benchmark_results_batch_summary.csv` has one row per model/device/power-mode/context/
horizon/batch-size configuration. It is the primary performance result.

Column groups include:

| Group | Examples |
|---|---|
| Provenance | seed, model artifact and variant, device category, execution device. |
| Configuration | power mode, context, horizon, batch size, benchmark batches, expected repeats. |
| Dataset workload | source count, total windows, stride, full/warmup batch counts, processed windows/points. |
| Timing/throughput | duration, end-to-end/model-call latency and throughput, pipeline overhead. |
| Power/energy | scope, idle/raw/dynamic power, percentiles, energy normalization, EDP. |
| Memory | CUDA allocated/reserved, physical used/free/total, memory scope, process-tree RSS. |
| Utilization/thermal | GPU compute/capacity use, CPU/GPU temperatures, Jetson rails. |
| Statistics | `{metric}_mean`, `{metric}_std`, `{metric}_ci95`. |
| Validity | status values, repeat IDs, missing repeats, success/OOM/error/swap/throttle counts, completion flag, OOM ceiling. |

The definitive formulas and scope rules are in [Measurement](MEASUREMENT.md).
Accuracy columns are forbidden in this file by design.

## Optional Raw Repeat CSV

Set `SAVE_RAW_PERFORMANCE_METRICS=true` only when repeat-level rows are necessary.
`benchmark_results_raw.csv` then stores one `ResultRow` per repeat/status. It does
not store every 50 ms sensor sample. The summary remains the publication-facing
table.

Keeping raw repeats off reduces I/O and storage and does not change in-memory repeat
calculation or summary statistics.

## Preprocessing Report

`dataset_preprocessing_report.csv` is a source-level audit. It records selected
columns, row counts, invalid timestamps, duplicate collapse, non-finite/missing
values, median imputation, inferred interval, gap segments, and usable window
information. It is the evidence that files were treated independently and that
windowing did not bridge data discontinuities.

## Dataset Manifest

`benchmark_metadata/dataset_manifest.json` records every ordered relative source
path and SHA-256 plus one ordered whole-dataset fingerprint. The fingerprint changes
when source identity, order, or contents change.

The manifest is the correct way to establish that predictions and benchmark rows
used the same dataset revision; folder names alone are insufficient.

## Model Metadata

Per-model, per-phase JSON includes:

- resolved `BenchmarkConfig` and methodology constants;
- model artifact/checkpoint/revision and adapter metadata;
- execution device, profile, monitor, and physical scopes;
- Python, PyTorch, CUDA, cuDNN, NumPy, and pandas versions;
- deterministic/backend flags and threading variables;
- host CPU count, RAM, GPU identity/capacity, and relevant device state;
- dataset fingerprint, source groups/count, pollutant, and preprocessing policy;
- Git commit when available and benchmark configuration.

Prediction and benchmark metadata are separate because a phase may be resumed or
run independently. Both must be retained with published outputs.

Supplemental batch sweeps write timestamped benchmark metadata and model logs instead
of overwriting the base sweep provenance. Their rows are appended to the same batch
summary only after protocol-identity checks; interrupted requested rows are archived
under `benchmark_metadata/supplemental_backups/` before replacement.

## Logs

The terminal receives concise milestones and progress. Detailed diagnostics are
written per model and phase. This prevents terminal output from becoming unreadable
while retaining model-load revisions, effective device configuration, memory guard
decisions, monitor state, exception tracebacks, and save/validation events.

Measured repeats suppress progress redraw; logging inside upstream model code should
also be treated as a potential contamination and reviewed in model logs.

## Accuracy Notebook

`evaluate_saved_predictions.ipynb` reads Phase 1 NPZ files and calls the repository's
`utils.metrics.compute_metrics`. It should produce source-level results and an
equal-source macro average. It does not alter performance summaries.

The publication plotting notebook `plot_publication_results.ipynb` is a consumer
of saved summaries/prediction metrics, not part of measured execution.

## Validation Contract

`scripts/validate_results.py` checks, as applicable:

- prediction file count, required keys, shape, finite output, source metadata, seed,
  and fixed prediction batch;
- matching dataset/protocol identity;
- complete expected successful repeats;
- absence of swap/hardware-throttle contamination;
- required telemetry fields/sample counts for supported scopes;
- valid status and summary completeness fields;
- absence of accuracy metrics in performance output.

A shell exit code of zero before this validator runs is not sufficient evidence that
a research run is complete.

## Artifact Retention

For reproducibility, archive together:

1. prediction NPZ files used for accuracy;
2. batch summary and optional raw-repeat table;
3. preprocessing report and dataset manifest;
4. model phase metadata, package snapshots, and source hashes;
5. model logs and the exact device-local configuration with secrets removed;
6. the repository commit or source archive.

Do not publish `.benchmark-device.env` directly if it contains host paths or
credentials. Passwords should never be stored there.
