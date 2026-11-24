# Models And Devices

This register defines the exact forecast semantics and execution backend used by
PowerBench. These choices are part of the research method. No adapter changes learned
parameters or architecture, but required input shaping, point-statistic selection,
sampling, compilation, and compatibility backends can affect predictions or
throughput and must be disclosed.

## Model Adapter Register

| Model | Forecast definition and input path | Execution disclosure |
|---|---|---|
| UniTS | Released x128 long-term forecast path; deterministic unseen-domain prompt/mask centroid; non-aligned horizons are generated to the next native 16-point patch and sliced. | Eager PyTorch `float32`; no compilation or sampling shortcut. |
| Kairos-23M | Per-series z-score, first-value left-pad to 512, `generation=False`, arithmetic mean across direct-head trained quantiles, inverse transform. | One direct model evaluation. Positivity inference and flipped-input averaging are not used. |
| TTM r3 | `ibm-research/ttm-r3`, revision `1024-96-r3`; 512-point input is left-padded with its first value to native 1024; native 96-step point output. | Direct eager PyTorch inference. |
| TTM r2 | `ibm-granite/granite-timeseries-ttm-r2`, revision `512-192-r2`; native 512-point input and 192-step point output, sliced to the requested 96-step study horizon. | Direct eager PyTorch inference; shares the identical `.venv_ttm` software environment with TTM r3 while retaining separate outputs and provenance. |
| Chronos-2 | Public tensor `predict_quantiles` API and its documented point forecast; scoped identity-collation correction for an already prebatched dataset item. | Native model backend; correction removes redundant Python conversion without changing tensors or forecasts. |
| Moirai 2.0 | `Salesforce/moirai-2.0-R-small`; direct batched Uni2TS forward and exact native 0.5 quantile. | Dense fixed-length tensor transport replaces equivalent generic list preparation. |
| TempoPFN | Released robust scaler and native median quantile. | CUDA uses FLA/Triton with `bfloat16`; Hopper retains the fused norm-gate kernel with one fixed conservative launch configuration to avoid changing-shape autotuner illegal-address faults; CPU-class hosts use explicit `float32` PyTorch compatibility operations. |
| TiRex | Mean returned by the released forecast API. | GPU uses model backend; CPU selects the official PyTorch backend with `compile=False`. |
| TimesFM 2.5 | Continuous quantile head with input normalization, positivity inference, quantile-crossing correction, flip invariance, and API point forecast. | Pinned source/compilation path; compile/load occurs before stabilization and timing. |
| FlowState | Per-series z-score, direct `prediction_outputs`, inverse transform. | Direct eager inference; no adapter-side internal chunking of benchmark batches. |
| MOMENT | Released pretrained reconstruction head; 512 observed points followed by a patch-aligned masked horizon, returning only requested horizon. | Direct eager inference; a semantic zero-shot adaptation, not a speed shortcut. |
| Sundial | Ten seeded autoregressive samples per series and arithmetic sample mean. | Sampling is inside measured model work; changing sample count changes the method. |
| Toto | Default `num_samples=None` and returned mean. Optional `TOTO_NUM_SAMPLES>0` uses a seeded sampled median with bounded samples per batch. | Default is not Monte Carlo inference. Compilation is attempted before measurement. |

Model parameter count alone cannot predict throughput. Tokenization/patching,
autoregressive steps, sample count, flip invariance, output heads, tensor transport,
kernel support, and compilation all change work per window.

## Device Execution Matrix

| Device category | Execution backend | Profile | Monitor | Memory scope |
|---|---|---|---|---|
| NVIDIA H100 | CUDA | Selected-index NVML persistence, supported max graphics clock, full-memory/MIG validation, power-limit verification | NVML at 50 ms | Dedicated HBM + PyTorch allocator + process RSS |
| NVIDIA H200 | CUDA | Selected-index NVML persistence, supported max graphics clock, full-memory/MIG validation, power-limit verification | NVML at 50 ms | Dedicated HBM3e + PyTorch allocator + process RSS |
| RTX 5090 | CUDA | NVML persistence, supported max graphics clock request, power-limit verification | NVML at 50 ms | Dedicated VRAM + PyTorch allocator + process RSS |
| RTX A6000 | CUDA | Same NVML policy; device-supported values are queried, not hardcoded | NVML at 50 ms | Dedicated VRAM + PyTorch allocator + process RSS |
| Jetson AGX Orin | CUDA integrated GPU | `nvpmodel` mode 0 MAXN, `jetson_clocks`, CPU/GPU/EMC verification, max fan | `tegrastats`/sysfs at 50 ms | Shared unified system memory + allocator + process RSS |
| x86 CPU | PyTorch CPU | Performance governor where permitted; all configured logical CPU threads | Intel RAPL, thermal sysfs, psutil at 50 ms | Process and recursive-child RSS |
| Raspberry Pi 5 | PyTorch ARM CPU | CPU profile; throttle verification | Configured sysfs sensor, thermal sysfs, `vcgencmd` | Process and recursive-child RSS |
| RPi 5 + Hailo-8 / UNO Q | PyTorch CPU for current adapters | Accelerator identity may be recorded | CPU/sysfs telemetry | CPU memory; no claimed NPU allocation |

Device detection occurs in `experiments/forecasting.py`. Explicit `--device` values
are recorded, and `execution_device` states where inference actually occurred.

## Device Profiles

### Discrete NVIDIA GPUs

`NVMLProfile` captures the prior device state, requests persistence mode, restores
the configured/default power limit, requests the maximum supported graphics-clock
range, and verifies effective power state. Unsupported clock control is logged rather
than misreported. Teardown restores the captured state.

On multi-GPU hosts, `BENCHMARK_GPU_INDEX` selects one physical card for CUDA,
NVML, `nvidia-smi`, and idle preflight. H100 and H200 additionally require the
matching product identity, full visible memory, and disabled MIG mode. These
guards prevent a cloud MIG slice from being reported as a complete device.

Before Phase 2, a sustained-idle preflight checks for unrelated GPU work. Transient
activity is retried for up to 120 seconds by default; persistent activity is rejected.
The launcher never stops a display manager. Operators must leave unrelated graphical
or compute applications idle.

### Jetson AGX Orin

Mode `0` is treated as true MAXN, not as a generic 50 W mode. `JetsonProfile` applies
the mode, locks supported CPU/GPU/EMC clocks with `jetson_clocks`, sets maximum fan,
logs effective frequencies, and verifies the power mode before and after the study's
configured 100 s stabilization period. It restores captured state on teardown.

Hardware over-current protection is never disabled. OC event counters are sampled
around every measured repeat; a new event invalidates the repeat as
`HARDWARE_THROTTLED`. This protects result integrity when the board cannot sustain the
requested MAXN workload.

### CPU

`CPUProfile` requests the `performance` governor where permissions permit and logs
effective per-core settings. The launcher exposes all logical CPU threads by default,
sets common BLAS/OpenMP variables consistently, and sets PyTorch inter-op threads to
one. These controls avoid nested thread oversubscription while allowing full CPU
parallelism. The original governor is restored.

### Raspberry Pi And Future Accelerators

The Raspberry Pi profile samples `vcgencmd get_throttled` before and after measured
intervals. Current undervoltage, frequency capping, thermal throttling, soft thermal
limits, or a newly set historical event invalidates the result. Adequate power and
active cooling are prerequisites.

The `rpi5_hailo26` category requires a visible `/dev/hailo0` and successful HailoRT
identity check for Hailo-8 rather than Hailo-8L. That proves hardware/runtime
readiness only. No current model has a validated HEF artifact or HailoRT adapter, so
current rows remain CPU inference. A publishable accelerator result requires
model-specific conversion, numerical-equivalence validation, and metadata proving
the accelerator backend executed.

## Adapter Interventions

### TempoPFN Hopper fused-normalization policy

TempoPFN uses Flash Linear Attention's fused RMSNorm-gate operation. Its default
Triton autotuner benchmarks several launch candidates whenever a shape key
changes. During large H100 batch sweeps, a candidate can trigger a CUDA illegal
memory access and poison the process context before the benchmark reaches the
normal OOM boundary.

On compute capability 9.x (H100/H200), PowerBench therefore keeps the same fused
operation and mathematical model but fixes the launch to `BT=64` and four warps
before model warmup. Other CUDA architectures retain upstream autotuning. The
policy is logged as `hopper_fixed` and can be overridden for diagnosis:

```bash
TEMPOPFN_TRITON_NORM_POLICY=autotune
```

The setup script verifies the fixed fused output against an explicit PyTorch
RMSNorm-plus-SiLU-gate reference at 65,537 rows. This intervention changes only
kernel launch selection, not checkpoint parameters, forecast definition, or
point-statistic selection. It must nevertheless be disclosed because kernel
selection can affect TempoPFN throughput.

### UniTS unseen-domain token

The released UniTS x128 checkpoint contains dataset-specific prompt/mask tokens but
no CPCB token. Its public unseen-dataset workflow otherwise requires a separately
trained shared-token checkpoint that is not among the release assets. PowerBench
initializes the unseen-domain prompt and mask deterministically from the centroid of
released forecasting-domain tokens. All active model tensors must load; unrelated
classification/pretraining heads are ignored. Metadata records
`forecast_prompt_centroid`.

This is a training-free adaptation and must be disclosed with UniTS results.

UniTS uses non-overlapping 16-point patches and one GEN token per complete
forecast patch. For a requested horizon \(H\), the adapter configures the
unchanged backbone with \(H_{\mathrm{model}}=16\lceil H/16\rceil\), then returns
the first \(H\) predictions. This avoids both the upstream zero-GEN-token failure
for \(H<16\) and an ungenerated zero tail for non-multiple horizons. Requests
already divisible by 16, including the main 96-step protocol, are unchanged.

### Chronos-2 identity collation

Chronos `predict_quantiles` creates an internal `DataLoader(batch_size=None)` whose
dataset already yields one complete dictionary of tensors and index ranges. With no
collator, PyTorch recursively applies `default_convert` to this already prebatched
object. At large batch sizes, redundant traversal of thousands of metadata tuples
caused a native segmentation fault on the RTX 5090.

PowerBench passes one contiguous `float32` tensor to the public API and installs a
Chronos-module-scoped loader wrapper. Only for Chronos loaders requesting
`batch_size=None` without an explicit collator, it supplies identity collation:

```python
DataLoader(
    dataset,
    batch_size=None,
    collate_fn=lambda prepared_batch: prepared_batch,
    pin_memory=True,
)
```

Normal batched loaders and explicit collators are unchanged. The dataset, order,
preprocessing, pinned memory, model forward, `cross_learning=False`, horizon,
quantiles, and point extraction are unchanged. An A/B test on identical `8 x 1 x
512` input produced bit-identical `(8, 1, 96)` point forecasts
(`max_abs_diff=0.0`); 100 consecutive `2048 x 512 -> 96` GPU forecasts completed
without allocator growth.

This correction changes timed pipeline overhead, so pre-intervention Chronos timing
is not comparable. Prediction artifacts may be reused only when their metadata/model
and numerical-equivalence conditions match.

### Moirai dense tensor transport

The generic Uni2TS `predict` path accepts variable-length lists, converts every
series through Python/NumPy, creates observed/padding arrays, restacks them, and then
copies tensors to device. PowerBench contexts are already finite, median-imputed,
contiguous, and fixed length. The adapter therefore constructs the equivalent inputs
directly:

- `past_target`: `(batch, context, 1)`;
- `past_observed_target`: all true;
- `past_is_pad`: all false.

It calls the unchanged `Moirai2Forecast.forward`, selects the unchanged native 0.5
quantile, and validates output shape. An A/B test with identical `8 x 512` input
produced bit-identical full quantile tensors (`max_abs_diff=0.0`). This avoids
redundant timed list preparation and an intermittent long-run list-path failure.
Pre-intervention Moirai timing is not comparable.

### Kairos prior-paper compatibility

PowerBench retains the prior study's `generation=False` method. Each context is
z-normalized, first-value padded when shorter than 512, and evaluated once. The
direct head returns trained quantiles; their arithmetic mean is the declared point
estimator and is inverse transformed.

Earlier reference code passed `preserve_positivity` and
`average_with_flipped_input`, but the vendored direct-forward signature consumed
those through `**kwargs` without using them. They were removed to avoid claiming
operations that never occurred. The effective call remains:

```python
model(past_target=batch, prediction_length=horizon, generation=False)
```

This is not generated Kairos inference and can be faster than models performing more
passes. The vendored `modeling_kairos.py` SHA-256 is
`5652ac5dfd5d0c689af4736a6a4954c2b64ecc9da35a18006422698b23ace93e`.

### TempoPFN ARM compatibility backend

TempoPFN's released CUDA implementation uses Flash Linear Attention and Triton,
which have no supported Raspberry Pi ARM64 runtime. CPU execution retains the
checkpoint, topology, robust scaler, equal-length batches, and median output, while
unavailable fused operations use explicit `float32` PyTorch implementations: RMS
normalization/gating, depthwise short convolution, SwiGLU projection, and naive
recurrent gated-delta product.

The compatibility path pins `transformers==4.56.1` and FLA revision
`ccb0ff944cbff035fa59e31adb1621bdc98a57b91b`. It marks compiler capability absent
during import and exposes inert decorator/type namespaces only for unused kernel
definitions. `torch.compile` is disabled only while those definitions import and is
then restored. A Raspberry Pi functional test loaded the released checkpoint and
returned a finite `float32` `1 x 512 -> 1 x 96` forecast.

CPU and CUDA TempoPFN are different numerical backends and are not bitwise-kernel
equivalent. No Hailo operation is used.

## Model Environment Isolation

Each `scripts/<model>.sh` creates or validates `.venv_<model>` and installs the
device-compatible PyTorch stack plus pinned model dependencies. TimesFM source is
pinned to commit `3dae50b20d7a724981e8ea36cda75578f80dd2dc`; setup does not update a
persistent checkout during a run. Jetson-specific repair logic verifies CUDA-enabled
PyTorch and repairs known NumPy/Pandas ABI incompatibility for Moirai.

An environment reported as `READY (existing)` has passed the launcher's import and
device checks; it is not recreated on every run. Remove an environment only when
changing its dependency definition or deliberately rebuilding the study software
state.

## Publication Disclosure Checklist

For each reported model/device combination, state:

- upstream model/checkpoint and exact revision;
- context, horizon, point forecast statistic, sample count, and fixed seed;
- adapter normalization, padding, masking, flip/positivity behavior, and compilation;
- actual execution backend and precision;
- any local transport or compatibility intervention above;
- hardware profile and effective power mode/clocks;
- monitor backend and physical power/memory scope;
- environment manifest and source checksum;
- whether all repeats passed contamination validation.

PowerBench does not claim authorship or endorsement of upstream models, PyTorch,
Chronos, NVML, or vendor software. Publications must cite and acknowledge their
original authors and clearly identify PowerBench changes as local benchmark adapters,
not upstream patches.
