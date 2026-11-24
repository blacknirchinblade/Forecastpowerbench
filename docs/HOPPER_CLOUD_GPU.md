# NVIDIA H100 And H200 Cloud GPUs

This guide runs PowerBench on a provider-independent, single NVIDIA Hopper GPU.
Upload the repository and dataset, create a host-local `.benchmark-device.env`,
run qualification, and then start the production protocol. No provider API or
managed-service integration is required.

## Execution Classification

| Property | PowerBench treatment |
|---|---|
| Device categories | `h100` or `h200` |
| Architecture | NVIDIA Hopper data-center GPU |
| Execution | CUDA on one selected physical GPU |
| Memory | Full dedicated HBM; MIG slices are rejected |
| Telemetry | Selected-index NVML at 50 ms |
| Power scope | Selected GPU board power |
| Device control | Selected-index persistence, graphics clocks, and power limit |
| Parallelism | One GPU per benchmark process; no model/data/tensor parallelism |

NVIDIA lists H100 SXM with 80 GB and H100 NVL with 94 GB of memory. H200 SXM
and NVL have 141 GB HBM3e. Both product families support MIG, so merely seeing
CUDA does not prove that the process owns a complete GPU.

References:

- [NVIDIA H100 specifications](https://www.nvidia.com/en-sg/data-center/h100/)
- [NVIDIA H200 specifications](https://www.nvidia.com/en-us/data-center/h200/)
- [NVIDIA supported MIG GPUs](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-gpus.html)

## Required Resource

Use a machine or container with:

- exactly one GPU selected for this benchmark process;
- a complete non-MIG H100 or H200 allocation;
- exclusive use of that selected GPU during measurement;
- enough host RAM and local storage for environments, model caches, predictions,
  and summaries;
- a compatible NVIDIA data-center driver and CUDA-enabled PyTorch stack;
- permission to query NVML and, preferably, set persistence, clock, and power
  controls through `nvidia-smi`.

PowerBench rejects:

- `h100` when the selected product name is not H100, visible memory is below
  70 GiB, or MIG mode is enabled;
- `h200` when the selected product name is not H200, visible memory is below
  120 GiB, or MIG mode is enabled.

These conservative thresholds accept full advertised devices while rejecting
normal H100/H200 MIG slices. They are identity guards, not benchmark memory
limits.

## Single-GPU Configuration

Create `.benchmark-device.env` after uploading the project. For H100:

```bash
BENCHMARK_DEVICE=h100
BENCHMARK_GPU_INDEX=0

CLOUD_PROVIDER=<provider>
CLOUD_INSTANCE_TYPE=<instance-or-node-type>
CLOUD_REGION=<region>
CLOUD_INSTANCE_ID=<instance-id>

DATASET_FOLDER=cnemc_imputed
POLLUTANT=PM2.5
BENCHMARK_SEED=12345
ACCURACY_SEASONALITY=24

GPU_LOCK_CLOCKS=true
POWER_MODES=0
SUDO_KEEPALIVE=true
GPU_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu129
```

For H200, change only:

```bash
BENCHMARK_DEVICE=h200
```

`BENCHMARK_GPU_INDEX` is the physical index displayed by `nvidia-smi`. The
launcher sets `CUDA_VISIBLE_DEVICES` to only this index. PyTorch therefore uses
logical `cuda:0`, while NVML, idle checks, controls, and metadata address the
same physical GPU. Even when a host has several GPUs, one PowerBench process
uses only the selected card.

Cloud provenance variables are optional but strongly recommended for a paper.
They do not change model execution.

## Preflight

```bash
nvidia-smi -L
nvidia-smi \
  --query-gpu=index,name,uuid,memory.total,mig.mode.current,compute_cap,driver_version,power.limit \
  --format=csv

bash run_unified_benchmark.sh --check-config
```

Confirm that the selected row reports the requested H100/H200, MIG is disabled,
no unrelated process uses that GPU, its UUID remains stable, and disk space is
sufficient for all environments and artifacts.

The shared installer defaults H100/H200 to official PyTorch `cu129` wheels.
CUDA 13 wheels require an R580-or-newer driver family and must not be used on a
CUDA 12.x driver. Preflight executes a real CUDA matrix multiplication in every
model environment, so an incompatible or CPU-only wheel fails before prediction.

TempoPFN additionally validates its FLA/Triton fused RMSNorm-gate kernel on
Hopper. The default `TEMPOPFN_TRITON_NORM_POLICY=auto` maps H100/H200 to one
fixed conservative fused launch (`BT=64`, four warps) rather than benchmarking
multiple Triton candidates as batch shapes change. This avoids a CUDA
illegal-address failure observed during large batch sweeps while preserving the
same fused operation. Do not set the policy back to `autotune` for a production
run unless the complete batch sweep has passed on the exact Torch, Triton, FLA,
driver, and GPU combination.

Some cloud containers expose read-only NVML control even to container root. If
`nvidia-smi -lgc` returns exit status 4, set `GPU_LOCK_CLOCKS=false`; PowerBench
then records `gpu_clock_policy=dynamic_requested` instead of claiming locked
clocks. Baseline power mode `0` is still read and verified, but an already-active
power limit is not redundantly written.

## Complete Qualification

```bash
bash smoke_test.sh
```

This is a functional release gate, not a reduced research benchmark. It:

1. creates or repairs every selected model environment;
2. verifies every model import;
3. preprocesses a fresh source subset;
4. generates and validates predictions for every model;
5. verifies that prediction processes release CUDA allocations;
6. exercises telemetry, warmup, repeated measurement, two batch sizes, and
   configuration transitions;
7. validates the final summary and required artifacts;
8. writes `QUALIFICATION_PASSED.txt` only after everything succeeds.

Outputs are isolated under `qualification_results/` and record
`research_result=false`. Never merge them into production summaries.

## Production Run

Start the full protocol only after qualification passes:

```bash
bash run_unified_benchmark.sh
```

Do not reuse qualification timings, shorten the reference cooldowns without
steady-state evidence, or run
unrelated GPU workloads. If the platform denies clock or power controls,
preserve the warning and report dynamic clocks and the effective power limit
rather than claiming a locked-frequency experiment.

## Code Path

```text
BENCHMARK_DEVICE=h100 or h200
  -> dedicated DeviceCategory
  -> selected physical GPU mapped to PyTorch cuda:0
  -> NVML identity, full-memory, and non-MIG validation
  -> selected-index setup, idle preflight, and 50 ms telemetry
  -> unchanged model adapters and BenchmarkRunner
  -> device-labeled summaries and reproducibility manifests
```

H100 and H200 use the discrete NVIDIA execution architecture shared with RTX
5090 and RTX A6000. They do not use the Jetson, RAPL, or ARM sysfs paths.
