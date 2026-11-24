# Operations Guide

This guide covers production setup, execution, recovery, verification, and safe
extension. Research runs should use `run_unified_benchmark.sh`, not invoke the Python
entry point manually.

## Prerequisites

- Linux with Bash and Python 3 suitable for the selected model environments.
- Working device runtime: NVIDIA driver/CUDA for discrete GPUs, JetPack tools for
  Jetson, or readable CPU/board telemetry interfaces.
- Sufficient storage for model caches and Phase 1 prediction NPZ files.
- Stable power, cooling, and an otherwise idle benchmark device.
- Passwordless access is not required, but one interactive sudo authentication may
  be needed for power/governor telemetry. The launcher keepalive retains that
  authenticated session; never put a sudo password in an environment file.

## Device-Local Configuration

Create a local file from the template:

```bash
cp .benchmark-device.env.example .benchmark-device.env
```

This file is intentionally host-specific and should be excluded from synchronization.
Typical research settings are:

```bash
DATA_DIR=./dataset
DATASET_FOLDER=cnemc_imputed
POLLUTANT=PM2.5
CTX_LENS=512
HORIZON_LENS=96
PREDICTION_BATCH_SIZE=512
BENCHMARK_BATCHES=300
WARMUPS=2
BENCHMARK_SEED=12345
SAVE_RAW_PERFORMANCE_METRICS=false
```

When invoking the Python entry point directly, omitting `--device` enables hardware
auto-detection. For the production launcher, set `DEVICE` or `BENCHMARK_DEVICE`;
its safe fallback is `cpu`. Run the resolved-configuration check before a long
experiment:

```bash
bash run_unified_benchmark.sh --check-config
```

## Production Commands

Full two-phase protocol:

```bash
bash run_unified_benchmark.sh
```

Generate or resume predictions only:

```bash
bash run_unified_benchmark.sh --phase prediction
```

Run or resume performance benchmarking independently:

```bash
bash run_unified_benchmark.sh --phase benchmark
```

Prediction and benchmark modes have separate artifact requirements and resume
planners. `--phase prediction` needs no benchmark summary. `--phase benchmark`
needs no prediction directory because Phase 2 constructs deterministic input batches
directly from the configured dataset and does not calculate or save forecasts.

Use tmux/SSH for long runs. Progress width follows the active terminal and refreshes
every two seconds outside measured repeats. Redirected/non-interactive output disables
dynamic bars to prevent escape-code corruption.

## Environment Construction

During preflight, each model is checked in `.venv_<model>`. Missing or invalid
environments are created by `scripts/<model>.sh`; valid environments report
`READY (existing)`. This is expected reuse, not duplicate creation.

Do not delete all environments between ordinary resumes. Rebuild one environment
only when its setup script, device runtime, Python ABI, or pinned dependencies have
changed. The launcher records its package snapshot for provenance.

## Resume Policy

`AUTO_RESUME=true` is the default for all phase modes.

### Prediction phase

The planner validates model artifacts in configured order. Complete earlier models
remain untouched. Partial artifacts for the first incomplete model and later models
are backed up/removed from active state, and execution restarts at that model. A
prediction error stops the phase immediately so a later success cannot hide it.

### Benchmark phase

The planner validates existing summary rows. Complete earlier models are preserved;
rows/log state for the interrupted model and later models are backed up and filtered.
The interrupted model restarts from its model boundary. Prediction files are neither
required nor modified by benchmark execution or resume.

### Supplemental batch sizes

To add smaller or otherwise omitted batch sizes to a completed Phase-2 sweep, use the
explicit supplemental mode. For example, this measures only batch sizes 16 and 32 and
merges their rows into the existing model-wise summary:

```bash
SUPPLEMENTAL_BATCHES=true BATCH_SIZES=16,32 \
  bash run_unified_benchmark.sh --phase benchmark
```

The launcher requires a non-empty benchmark summary. It verifies the device, dataset,
seed, context, horizon, power mode, repeat
count, and fixed batches-per-iteration before modifying anything. Existing batch
sizes such as 64 and above are never filtered. Partial rows for only the requested
supplemental configurations are backed up under
`benchmark_metadata/supplemental_backups/`, then rerun at model boundaries. Completed
supplemental models are skipped after interruption, and supplemental logs/metadata
use separate timestamped names so the original run provenance remains intact.

Do not combine `SUPPLEMENTAL_BATCHES=true` with `RESET_BENCHMARK_RESULTS=true`. Use
the same production cooldown, warmup, repeat, telemetry, and power-mode settings as
the base sweep; a mismatch is rejected rather than silently aggregated.

Controls:

| Variable | Effect |
|---|---|
| `AUTO_RESUME=true` | Continue from the first incomplete model. |
| `AUTO_RESUME=false` | Fail when partial state exists; operator must resolve it. |
| `RESET_BENCHMARK_RESULTS=true` | Restart all of Phase 2 while retaining predictions. |
| `SUPPLEMENTAL_BATCHES=true` | Append only the configured `BATCH_SIZES` to a compatible existing Phase-2 sweep. |

## Important Controls

The table reports the reference profile used in the study. Every value is
configurable through the environment; the resolved values are written to metadata.
The 300-batch workload is a study design choice, not a model requirement. Cooling
and stabilization intervals are conservative starting points: extend them whenever
temperature, utilization, power, or clocks have not returned to a stable baseline.
Shorter intervals require device-specific steady-state evidence and must be reported.

| Variable | Production default | Purpose |
|---|---:|---|
| `RUN_PHASE` / `--phase` | `all` | Select both phases, prediction, or benchmark. |
| `MODEL_LIST` | 12 core adapters | Ordered selected models; checkpoint variants are opt-in. |
| `SOURCE_PATTERN` | derived | Override recursive source glob. |
| `TIME_COLUMN`, `TARGET_COLUMN` | auto | Explicit CSV column selection. |
| `CTX_LENS`, `HORIZON_LENS` | `512`, `96` | Forecast shapes. |
| `BATCH_SIZES` | device sweep | Comma-separated Phase 2 batch sizes. |
| `SUPPLEMENTAL_BATCHES` | `false` | Safely add only `BATCH_SIZES` to existing compatible results. |
| `PREDICTION_BATCH_SIZE` | `512` | Fixed Phase 1 batch. |
| `BENCHMARK_BATCHES` | `300` | Full batches per warmup/repeat. |
| `WARMUPS` | `2` | Discarded complete iterations. |
| `REPEATS` | GPU/Jetson `5`, CPU `2` | Measured iterations. |
| `CPU_CONFIG_TIMEOUT_SEC` | `18000` | Five-hour CPU configuration/prediction-model ceiling. |
| `PHASE_COOLDOWN_SEC` | `1200` | Phase 1 to Phase 2 cooling. |
| `SLEEP_BETWEEN_BATCHES_SEC` | `300` | Batch-size cooling. |
| `MODEL_COOLDOWN_SEC` | `300` | Model cooling. |
| `POWER_MODE_STABILIZATION_SEC` | `100` | Applied-mode stabilization before verification. |
| `POWER_MODES` | `0` | Selected device profile mode; Jetson 0 is MAXN. |
| `BENCHMARK_PRELOAD_POOL_WINDOWS` | `65536` | Pool bound, while retaining at least one full batch. |
| `GPU_MEMORY_SAFETY_FRACTION` | `0.90` | Projected physical-memory safety ceiling. |
| `GPU_MEMORY_ESTIMATE_HEADROOM` | `1.08` | Projection multiplier. |
| `GPU_MEMORY_HARD_LIMIT_GB` | unset | Optional absolute CUDA limit. |
| `CPU_THREADS` | logical CPU count | Intra-op/BLAS thread target. |
| `BENCHMARK_SEED` | `12345` | Global reproducibility seed. |
| `ACCURACY_SEASONALITY` | `24` | Context-derived MASE/RMSSE lag saved in predictions. |
| `TQDM_MININTERVAL` | `2` | Progress refresh outside timing. |
| `SAVE_RAW_PERFORMANCE_METRICS` | `false` | Persist per-repeat rows, not telemetry samples. |

An explicit environment override is recorded in metadata. Keep one declared profile
within a comparison and do not shorten cooldowns or warmups merely to accelerate a
publication run.

The default model order is `units`, `kairos`, `ttm`, `moirai2`, `tempopfn`,
`timesfm_2p5`, `flowstate`, `moment`, `sundial`, `toto`, `chronos2`, and `tirex`.
Additional checkpoint variants listed in `models/README.md` are opt-in. Set
`MODEL_LIST` only when the study requires another fixed order, and retain the
recorded override with the results.

## Failure Triage

### Native segmentation fault

1. Read the model's phase log and kernel log (`dmesg`, NVIDIA Xid where accessible).
2. Distinguish allocator/OOM evidence from Python transport/native-extension faults.
3. Confirm no partial model rows are being accepted by resume.
4. Reproduce the adapter call outside telemetry at the failing shape.
5. Prove prediction equivalence before changing transport; rerun all affected timing
   rows if timed overhead changes.

Do not treat a segmentation fault as a normal OOM unless evidence supports it.

### OOM or projected memory ceiling

An OOM skips larger sizes only for the same model/context/horizon. Verify the recorded
ceiling and physical-memory fields. Do not manually force unsafe sizes for a main run;
native CUDA/Triton failure can terminate the process without a recoverable exception.

### Missing power values

Check `power_scope` and sample count first. On CPU, ensure RAPL paths are readable and
sudo preparation succeeded. On ARM, a readable calibrated board sensor may not exist;
PowerBench leaves unsupported values blank rather than fabricating them. Missing
whole-board power cannot be inferred from temperature or utilization.

### Swap contamination

Stop unrelated memory-heavy processes and disable/clear swap according to host policy
before starting. PowerBench intentionally rejects swap growth because paging changes
latency, energy, and RSS interpretation.

### Jetson throttling or OC

Use adequate PSU/cabling and active cooling. Keep mode 0 MAXN and hardware protection
enabled. A new OC counter means the configuration is invalid; removing the detector
would hide a real hardware limitation, not fix performance.

### GPU preflight busy

Close or idle unrelated CUDA applications and disconnect GPU-heavy remote viewers.
PowerBench will not stop the display manager. Retry only after the sustained-idle
check can pass.

### Scrambled progress output

Use a real TTY and resize tmux after attaching. Avoid piping dynamic output through a
consumer that falsely reports TTY capability. Detailed logs remain in `model_logs`
even when dynamic bars are disabled.

## Verification And Tests

Run first-party tests:

```bash
python3 -m pytest -q tests
```

The suite covers data leakage/windowing, prepared batches, reproducibility, monitor
statistics, progress isolation, adapter interventions, device profiles, GPU preflight
and memory guard, resume planners, timeout behavior, and final validation.

Before a publication run:

1. run `--check-config` and archive its resolved values;
2. run a smoke test on the same host/runtime;
3. inspect one prediction NPZ and preprocessing report;
4. execute one representative full 300-batch configuration per model family;
5. verify monitor sample counts, power scope, memory scope, temperatures, and status;
6. clear smoke outputs, then begin the uninterrupted production run;
7. retain metadata, source hashes, logs, and validator output.

## Adding Or Changing Code Safely

A change that touches model input, point-statistic extraction, precision, sampling,
compilation, or upstream revision can change predictions and requires regenerated
Phase 1 artifacts. A change that only removes redundant work inside the timed adapter
may preserve predictions but still requires all affected Phase 2 rows to be rerun.

A monitor/progress/persistence optimization is acceptable only after proving it does
not enter the timed path, alter the batch stream, or reduce required telemetry. Add a
regression test for that boundary.

See [Architecture](ARCHITECTURE.md) for formal extension points and
[Models And Devices](MODELS_AND_DEVICES.md) for disclosure requirements.
