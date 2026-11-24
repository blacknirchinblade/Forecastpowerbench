# Experimental Setup

This document is the run-reporting companion to
[Research Methodology](METHODOLOGY.md). A paper should report the values captured
by each run's metadata rather than relying on machine nicknames.

## Study Configuration

The table reports the resolved settings used in this study. They are configurable
launcher controls rather than fixed implementation constants. The batch count is
held constant within the study; waiting intervals should be increased on hardware
that has not reached a repeatable idle thermal and power state.

| Item | Reported study value |
|---|---|
| Dataset selector | `cnemc_imputed/**/*_PM2.5.csv` |
| Target | PM2.5 |
| Independent-series policy | One CSV per source; no cross-source windows |
| Missing-value policy | Per-source global median; observed mask retained |
| Context / horizon / stride | 512 / 96 / 96 |
| Prediction batch size | 512 |
| Benchmark batch sizes | 64, 128, 256, ... until actual/projected OOM |
| Prepared workload | 300 complete batches per iteration |
| Seed | 12345 |
| Shuffle | Disabled |
| Model stabilization | 30 s |
| Idle baseline | 30 s per executed configuration |
| Warmup | 2 complete 300-batch iterations |
| Measured repeats | 5 on GPU/Jetson; 2 on CPU |
| Telemetry period | 50 ms (20 Hz) |
| Power-mode stabilization | 100 s |
| Phase cooldown | 1,200 s |
| Batch/model cooldown | 300 s |
| Raw telemetry persistence | Disabled by default |
| Prediction persistence in Phase 2 | Disabled |

The frozen RTX 5090 dataset manifest used during development contains 1,471 source
CSVs and has SHA-256
`569bd6460ec3c6c052e56d6e6cd08a52912522d8fc49b370bc311952b52aeda8`.
Publication results must quote the hash from the actual run metadata, especially if
the dataset is updated.

## Hardware Platforms

The study spans four deployment categories:

| Category | Platform | Execution / memory model | Power scope |
|---|---|---|---|
| New inference GPU | NVIDIA GeForce RTX 5090 | Discrete CUDA GPU; dedicated VRAM | NVML `gpu_board` |
| Legacy high-memory GPU | NVIDIA RTX A6000 | Discrete CUDA GPU; dedicated VRAM | NVML `gpu_board` |
| Edge GPU | NVIDIA Jetson AGX Orin | Integrated CUDA GPU; shared unified memory | Named onboard rail sum |
| CPU baseline | x86 CPU | CPU execution; system DRAM | Intel RAPL `cpu_package` |

One captured RTX 5090 run reported 33,633,533,952 bytes (approximately 31.3 GiB)
of GPU memory and compute capability 12.0, hosted by an Intel Core i9-14900K with
24 physical cores, 32 logical CPUs, and approximately 188.4 GiB RAM.

The final paper must copy the following from each platform's own run metadata:

- exact accelerator/CPU name, memory capacity, and compute capability;
- host CPU, physical/logical core count, and system RAM;
- operating system/kernel and architecture;
- Python, NumPy, pandas, PyTorch, CUDA, cuDNN, driver, and model-library versions;
- effective power mode, clocks, and monitor backend;
- storage/cache path and thread configuration;
- source revision, environment manifest, dataset hash, and model/checkpoint hashes.

Do not infer A6000 host details, Jetson memory size/JetPack release, or CPU-baseline
inventory from another machine's metadata.

## Device Controls

### Discrete NVIDIA GPUs

The device profile verifies identity and memory capacity, requests persistence and
maximum supported clocks where the driver permits, checks for sustained
pre-existing load, and records the effective state. Display-manager services are
not modified. CUDA allocator statistics are reset immediately before each measured
iteration, and CUDA is synchronized at timing boundaries.

### Jetson AGX Orin

The production profile requests and verifies `nvpmodel` mode 0 (MAXN), applies
`jetson_clocks`, requests maximum fan cooling, and checks state before and after the
100-second stabilization. `tegrastats` and sysfs provide utilization, temperature,
memory, rail power, and over-current evidence. Any over-current increment rejects
the affected repeat; protection is never disabled. SSD-backed caches are preferred
when mounted, with eMMC fallback.

### CPU Baseline

The CPU profile requests the `performance` governor where permission permits,
exposes all logical CPUs to the model runtime, and uses one PyTorch inter-op thread
to avoid nested inter-op oversubscription. Model-specific intra-op behavior remains
part of the declared backend. Intel RAPL supplies package energy when readable.

## Software And Instrumentation

| Component | Role |
|---|---|
| Bash | Device-local configuration, per-model process isolation, environment validation, resume, cooldown |
| Python `argparse` / Pydantic | Typed protocol configuration and result schema |
| pandas | Source CSV parsing and timestamp normalization outside timing |
| NumPy | Contiguous source arrays, masks, prepared pools, prediction artifacts, metric arithmetic |
| PyTorch | Model execution, DataLoader/model APIs, CUDA synchronization and allocator peaks |
| `time.perf_counter` | Monotonic end-to-end and adapter-call timing |
| `time.time_ns` | Wall-clock telemetry alignment |
| `nvidia-ml-py` / NVML | Discrete-GPU power, utilization, memory, and temperature |
| `tegrastats`, sysfs, `nvpmodel`, `jetson_clocks` | Jetson telemetry and mode verification |
| Intel RAPL | CPU package energy and derived power |
| psutil | Recursive process-tree RSS and host-state checks |
| tqdm / logging | Console progress and model logs outside measured intervals |
| CSV / JSON / NPZ | Summary tables, provenance, and per-source forecasts |

Each model is launched in a separate virtual environment and interpreter. This
prevents incompatible dependencies and stale CUDA allocator state from leaking
between models. The model adapters normalize heterogeneous public APIs to one
`predict(batch, horizon)` contract, but model-specific semantics remain disclosed
in [Models And Devices](MODELS_AND_DEVICES.md).

## Execution Example: TTM r3 On RTX 5090

1. The launcher validates `.venv_ttm`, records package/source manifests, and starts
   a fresh Python process.
2. The device factory selects the CUDA/NVML profile and monitor.
3. pandas and NumPy prepare each PM2.5 source independently. Complete 512-to-96
   windows are indexed without crossing sources or gaps.
4. Phase 1 loads `ibm-research/ttm-r3` revision `1024-96-r3`. The adapter
   first-value-pads the 512-point study context to the native 1,024-point context
   and emits a 96-step point forecast. Per-source NPZ files are written without
   telemetry.
5. After process teardown and the phase cooldown, Phase 2 constructs the 300-batch
   deterministic pool for one batch size.
6. The model, checkpoint, and pool become resident; the device stabilizes and an
   idle baseline is sampled.
7. Two full warmups run and are discarded.
8. Each measured repeat starts 50 ms NVML and process-RSS sampling, synchronizes
   CUDA, executes 300 synchronous adapter calls, synchronizes again, and stops
   sampling.
9. Power samples are boundary-aligned and integrated; all other telemetry is
   reduced from in-window samples. Repeat metrics are retained in memory.
10. After all repeats, successful observations are summarized using a mean, sample
    standard deviation, and 95% Student-t interval. The configuration row and
    provenance are persisted outside timing.
11. The process releases the pool and model, clears device caches, cools, and moves
    to the next batch size.

## Required Result Tables

At minimum, a publication table must retain:

- hardware and model identity, checkpoint revision, backend, precision, context,
  horizon, batch size, seed, and successful repeat count;
- end-to-end windows/s and latency/window with standard deviation or 95% CI;
- raw and dynamic joules/window with physical `power_scope`;
- average/peak power and power sample count;
- allocator allocation/reservation, physical accelerator memory, and process RSS;
- utilization and temperature where the named sensor exists;
- status counts, completeness, and contamination flags.

Accuracy belongs in a separate model-level table derived from Phase 1 predictions.
ND and WAPE must not both be interpreted as independent metrics because WAPE is
ND expressed as a percentage whenever the denominator is positive.

## Reproduction Checklist

Before accepting a run:

1. archive launcher configuration, git commit, source manifest, package manifest,
   dataset hash, model revisions, and device metadata;
2. verify source counts and preprocessing reports;
3. verify power mode after stabilization;
4. verify every successful row has the configured repeat identifiers;
5. inspect OOM, timeout, swap, throttle, and telemetry status columns;
6. confirm prediction artifacts are complete before using accuracy metrics;
7. retain power scope in every figure and table;
8. do not merge smoke-test, pre-intervention, or numerically different backend
   results with production measurements.
