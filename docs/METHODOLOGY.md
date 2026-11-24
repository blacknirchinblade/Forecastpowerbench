# Research Methodology

This chapter describes the publishable PowerBench protocol. It is intentionally
separate from operational instructions: the purpose here is to define the
experimental unit, controlled variables, measurement boundaries, estimands, and
validity criteria precisely enough for an independent implementation.

## Research Questions

PowerBench addresses three distinct questions:

1. How does inference batch size affect the throughput and latency of a zero-shot
   time-series foundation model (TSFM) on a fixed hardware platform?
2. What power, energy, memory, utilization, and thermal cost accompanies that
   performance at the sensor boundary available on each platform?
3. What forecast accuracy does each model attain on the fixed prediction corpus,
   independently of the performance batch-size sweep?

The third question is deliberately isolated from the first two. Accuracy is a
property of the model, inputs, checkpoint, point-estimator rule, and numerical
backend; it is not treated as a batch-size-dependent systems metric.

## Forecasting Task

Let source series \(s\) contain observations \(x^{(s)}_1,\ldots,x^{(s)}_{L_s}\).
For context length \(C\), horizon \(H\), and stride \(S\), window \(w\) contains

\[
\mathbf{x}_{s,w} =
  [x^{(s)}_{a_w},\ldots,x^{(s)}_{a_w+C-1}]
\quad\text{and}\quad
\mathbf{y}_{s,w} =
  [x^{(s)}_{a_w+C},\ldots,x^{(s)}_{a_w+C+H-1}],
\]

where \(a_w = wS\). Only complete windows are retained. The primary study uses
\(C=512\), \(H=96\), and \(S=96\). Each CSV is an independent source: windows
cannot cross files, timestamp discontinuities, or incomplete source tails.

## Dataset Preparation

The default dataset selector is
`<DATA_DIR>/<DATASET_FOLDER>/**/*_<POLLUTANT>.csv`; the production study selects
CNEMC PM2.5 sources. Discovery is lexicographically sorted and shuffling is
disabled.

For each source independently, PowerBench:

1. resolves and parses the timestamp and target columns;
2. removes rows with invalid timestamps and stably sorts the remainder;
3. collapses duplicate timestamps using the median target value;
4. computes one source-level median from finite target values and uses it to
   replace non-finite targets;
5. retains an observed-value mask so imputed forecast targets are excluded from
   later accuracy calculations;
6. infers the regular sampling interval and splits the source at timestamp gaps;
7. emits only complete source-local context-and-horizon windows.

The source-level median is calculated from the complete source. This is a
transductive imputation policy, not a causal online-imputation claim, and must be
reported as such. It is held fixed across all models and devices so the systems
comparison uses identical numerical inputs. The preprocessing report records
removed timestamps, duplicate collapse, imputation count, inferred interval,
segment count, and source identity.

Validated sources are stored as contiguous `float32` NumPy arrays with compact
`bool` observation masks and `int64` timestamps or segment bounds. A
content-derived cache key includes the preprocessing schema, relative path, file
size, modification time, and selected columns. Cache validation and window
construction occur before any measured interval.

## Two-Phase Experimental Design

Workload counts, warmups, repeats, stabilization, and cooldown durations are
configurable protocol parameters. We use the conservative values below for the
reported study and keep them unchanged within each cross-hardware comparison.
Other deployments should lengthen thermal waits when steady-state checks require it
and disclose any deviation.

### Phase 1: Prediction Generation

Each model generates forecasts once using batch size 512, seed 12345, deterministic
source and window order, and no shuffling. Telemetry, throughput measurement,
accuracy calculation, and benchmark-result persistence are disabled. Predictions,
targets, observed masks, source/window identifiers, context-derived seasonal
scales, and provenance are written to one compressed NPZ artifact per
model/source/context/horizon.

Separating prediction generation prevents NPZ serialization, file-system activity,
and accuracy arithmetic from contaminating systems measurements. It also prevents
the batch-size sweep from creating multiple nominally equivalent prediction
corpora.

### Phase Transition

Model processes are terminated, references are released, CUDA caches are cleared
where applicable, and the device cools for 1,200 seconds before benchmarking. This
transition reduces dependence on the thermal and allocator state left by Phase 1.

### Phase 2: Systems Benchmarking

For each model, device, power mode, context, horizon, and batch size, a
`PreparedBenchmarkStream` is materialized before inference. It deterministically
cycles valid windows when necessary and provides the configured 300 complete
batches per iteration. Batch retrieval is a contiguous NumPy view; CSV parsing, imputation,
window discovery, and pool construction are therefore outside timing.

The default sweep starts at 64 and doubles until an observed or safely projected
out-of-memory condition. Supplemental sweeps, such as batch sizes 16 and 32,
validate the saved protocol identity and append only missing configurations.

For each configuration the production protocol performs:

1. device power-mode application and verification, followed by 100 seconds of
   stabilization;
2. model loading and 30 seconds of model stabilization;
3. a 30-second idle baseline with the model and prepared pool resident;
4. two complete, unmeasured warmup iterations of 300 batches;
5. five measured repeats on GPU/Jetson or two on the CPU baseline;
6. summary calculation and persistence only after the configuration completes;
7. resource release, device cache cleanup, and a 300-second cooldown before the
   next batch size or model.

The warmups populate lazy kernels, graph/compiler caches, allocator pools, and
model-specific runtime state. They are full-workload warmups rather than one-call
priming so short and long model paths receive the same preparation. Warmup results
are discarded.

## Determinism And Controlled Variables

Seed 12345 controls `PYTHONHASHSEED`, Python `random`, NumPy, PyTorch CPU/CUDA
generators, and DataLoader workers. Source order, window order, and prepared-pool
cycling are deterministic; shuffling is false. Model checkpoint/revision, context,
horizon, point-estimator rule, preprocessing, seed, and workload count are fixed
within a comparison.

These controls reduce avoidable variance but do not imply bitwise equality across
runs or across different hardware, compiler, kernel, or precision backends when a
runtime uses nondeterministic kernels. Any adapter-specific
padding, normalization, quantile reduction, sampling, compilation, or compatibility
backend is recorded and disclosed.

## Measurement Boundary

For one repeat in the reported profile, let \(B=300\) be the configured number of
batches, \(N\) the batch size,
\(H\) the horizon, \(W=BN\) the number of forecast windows, \(P=WH\) the number of
forecast points, and \(T\) the measured end-to-end duration.

The interval starts immediately before the first prepared-batch view and ends after
the final adapter call and device synchronization. It includes batch-view
selection, adapter conversion, host-to-device transfer, model execution, point
forecast extraction, device-to-host transfer, normalized NumPy return, and minimal
loop bookkeeping.

It excludes source I/O, preprocessing, window/pool construction, model loading,
compilation before warmup, stabilization, idle-baseline collection, warmups,
progress rendering, telemetry reduction, persistence, cleanup, and cooldown. CUDA
is synchronized immediately before and after the interval to prevent asynchronous
execution from being under-counted.

Power and process-memory monitors sample concurrently at 50 ms (20 Hz). Monitor
start/stop and all reduction arithmetic are outside \(T\). Thus the observed
sampling work is present consistently during inference, while post-processing is
not charged to throughput.

## Primary Performance Estimands

\[
\operatorname{Throughput}_{w}=\frac{W}{T},\qquad
\operatorname{Throughput}_{p}=\frac{P}{T},
\]

\[
\operatorname{Latency}_{batch}=\frac{1000T}{B},\qquad
\operatorname{Latency}_{window}=\frac{1000T}{W}.
\]

These are end-to-end adapter metrics. If \(M\) is the sum of synchronous adapter
call durations, `model_call_*` metrics replace \(T\) with \(M\), and

\[
T_{\mathrm{overhead}}=\max(0,T-M).
\]

\(M\) is not pure GPU-kernel time: it still includes model-specific conversion,
transfer, forecast extraction, and host return. End-to-end window throughput is the
primary deployment-facing estimand because all models solve the same window-level
task even when their internal APIs differ.

## Telemetry Alignment And Energy

The runner records wall-clock nanosecond boundaries for each measured interval.
Only finite utilization, memory, and temperature samples whose timestamps lie in
the closed interval are reduced. For power integration, the start and end values
are linearly interpolated from surrounding samples, or held at the nearest observed
value when the interval extends beyond the sampled range. With endpoint-enriched
power samples \((t_i,p_i)\),

\[
E=\sum_{i=0}^{K-2}\frac{p_i+p_{i+1}}{2}(t_{i+1}-t_i),
\qquad
\overline{P}=\frac{E}{t_{K-1}-t_0}.
\]

Trapezoidal timestamp integration is used instead of multiplying a simple mean by
nominal duration because background scheduling makes sampling intervals imperfectly
uniform.

The idle baseline \(P_{\mathrm{idle}}\) is measured with the model and prepared
pool resident. Raw sensor energy remains primary; baseline-corrected quantities are

\[
P_{\mathrm{dyn}}=\max(0,\overline{P}-P_{\mathrm{idle}}),\qquad
E_{\mathrm{dyn}}=\max(0,E-P_{\mathrm{idle}}T_{\mathrm{tel}}).
\]

Energy efficiency is normalized as \(E/W\) joules per forecast window and
\(E/(P/1000)\) joules per 1,000 forecast points. The energy-delay product is
\(E T\). These normalizations expose whether higher instantaneous power is offset
by shorter execution.

Power scope is never assumed equivalent across devices: discrete NVIDIA GPUs report
NVML board power, the CPU baseline reports RAPL package energy, and Jetson reports
the configured sum of `VDD_GPU_SOC`, `VDD_CPU_CV`, and `VIN_SYS_5V0`. None is
silently labeled whole-system wall power.

## Memory, Utilization, And Temperature

PowerBench reports complementary memory layers:

- PyTorch peak live allocation and peak caching-allocator reservation;
- mean/peak physical GPU or shared-memory use from the device monitor;
- minimum physical GPU free memory;
- peak RSS of the benchmark process and recursive children.

Allocator and physical memory metrics answer different questions and need not be
monotonic with batch size because model residency, library contexts, allocator
reuse, kernel selection, and unified-memory behavior differ. GPU/GR3D utilization,
memory-capacity utilization, GPU/CPU temperature, and device-specific rail metrics
are reduced only from in-window samples. Unsupported sensors remain NaN with an
explicit zero sample count rather than being replaced by a different physical
quantity.

## Repeat Aggregation And Uncertainty

For metric observations \(z_1,\ldots,z_n\) from successful repeats,

\[
\bar z=\frac{1}{n}\sum_{r=1}^{n}z_r,\qquad
s_z=\sqrt{\frac{1}{n-1}\sum_{r=1}^{n}(z_r-\bar z)^2},
\]

\[
\operatorname{CI95}_{\mathrm{half}}
=t_{0.975,n-1}\frac{s_z}{\sqrt{n}}.
\]

Five-repeat GPU/Jetson rows use \(t_{0.975,4}=2.776\). Two-repeat CPU rows use
\(t_{0.975,1}=12.706\), so CPU confidence intervals are appropriately wide and
should be interpreted as preliminary uncertainty estimates. Within-repeat
`power_std_w` describes 50 ms temporal variation; the similarly suffixed summary
standard deviation describes between-repeat variation and is not interchangeable.

Only successful repeats contribute to means. A result is complete only when all
configured repeat identifiers are present and valid. One successful repeat yields
a stored zero standard deviation/CI for schema compatibility, but uncertainty is
statistically not estimable and must not be described as zero.

## Accuracy Metrics

Accuracy is computed after Phase 1 from finite, originally observed forecast
targets. Let \(\mathcal V\) be that valid set, \(n=|\mathcal V|\), and
\(e_i=\hat y_i-y_i\):

\[
\mathrm{MAE}=\frac{1}{n}\sum |e_i|,\quad
\mathrm{MSE}=\frac{1}{n}\sum e_i^2,\quad
\mathrm{RMSE}=\sqrt{\mathrm{MSE}},
\]

\[
\mathrm{sMAPE}=100\frac{1}{n}\sum
\frac{2|e_i|}{|y_i|+|\hat y_i|},\quad
\mathrm{NRMSE}=\frac{\mathrm{RMSE}}{\frac{1}{n}\sum|y_i|+10^{-8}},
\]

\[
\mathrm{ND}=\frac{\sum|e_i|}{\sum|y_i|},\quad
\mathrm{WAPE}=100\,\mathrm{ND},\qquad
\mathrm{Bias}=\frac{1}{n}\sum e_i.
\]

These formulas match the historical TFB/TSFM-Bench-compatible metric module.
Consequently, MAPE is undefined for a retained zero actual and sMAPE is undefined
when both values are zero. \(R^2\) is undefined for fewer than two valid
observations or zero target variance. The NRMSE \(10^{-8}\) stabilizer is retained
for exact compatibility.

For seasonality \(m=24\), each window derives historical scales from its context:

\[
a_w=\frac{1}{C-m}\sum_{t=m+1}^{C}|x_{w,t}-x_{w,t-m}|,
\quad
q_w=\frac{1}{C-m}\sum_{t=m+1}^{C}(x_{w,t}-x_{w,t-m})^2.
\]

The implementation broadcasts these scales across each window's valid horizon
points:

\[
\mathrm{MASE}=\operatorname{mean}_{(w,h)\in\mathcal V,a_w>0}
\frac{|e_{w,h}|}{a_w},\qquad
\mathrm{RMSSE}=\sqrt{\operatorname{mean}_{(w,h)\in\mathcal V,q_w>0}
\frac{e_{w,h}^{2}}{q_w}}.
\]

For positive denominators WAPE is ND expressed in percent, so they must not be
presented as independent evidence. Raw forecasts are neither clipped nor
calibrated.

## Complementary Multi-Objective Analyses

We report the alpha-conditioned Energy-Accuracy Score (EAS), Multi-Objective
Alpha Score (MOAS), and Preference-Robust Rank Acceptability (PRRA). They answer
different questions, and none replaces the underlying accuracy, energy,
latency, and memory measurements.

### Alpha-conditioned Energy-Accuracy Score

For the frozen study population, error \(E\) is globally min-max normalized:

\[
\widetilde E=\frac{E-E_{\min}}{E_{\max}-E_{\min}}.
\]

Energy per forecast \(J\) is normalized after a base-10 logarithm because its
positive values can span orders of magnitude:

\[
\widetilde J=
\frac{\log_{10}J-\log_{10}J_{\min}}
{\log_{10}J_{\max}-\log_{10}J_{\min}}.
\]

The higher-is-better score is

\[
\mathrm{EAS}_{\alpha}
=1-\left[\alpha\widetilde E+(1-\alpha)\widetilde J\right],
\qquad 0\leq\alpha\leq1.
\]

Thus, \(\alpha=1\) expresses an accuracy-only preference,
\(\alpha=0\) an energy-only preference, and the predeclared
\(\alpha=0.5\) weights their normalized costs equally. Alpha is not fitted to
the observed winner. We report the complete \(\alpha\in[0,1]\) sensitivity
curve and ranking crossover values in addition to the predeclared score.
Because EAS depends on min-max bounds and includes only error and energy, its
reported model/hardware population must be frozen before normalization.

### Multi-Objective Alpha Score

MOAS accepts independently declared non-negative priorities
\(\boldsymbol\lambda=(\lambda_E,\lambda_J,\lambda_T,\lambda_M)\) for error,
energy, latency, and memory, then removes arbitrary priority scale through

\[
\bar\lambda_k=\frac{\lambda_k}{\sum_j\lambda_j}.
\]

Each lower-is-better criterion is normalized within hardware:

\[
\widetilde{x}_{hmk}=
\frac{g_k(x_{hmk})-\min_{m'}g_k(x_{hm'k})}
{\max_{m'}g_k(x_{hm'k})-\min_{m'}g_k(x_{hm'k})},
\]

where error uses the identity transform and positive energy, latency, and
memory use \(\log_{10}\). The higher-is-better score is

\[
\mathrm{MOAS}_{hm}(\boldsymbol\lambda)
=1-\sum_{k\in\{E,J,T,M\}}\bar\lambda_k\widetilde{x}_{hmk}.
\]

The balanced default is \((1,1,1,1)\). Accuracy-, energy-, latency-, and
memory-priority scenarios use \((4,1,1,1)\) and its permutations. These
priorities are declared before comparison and are not learned from the results.
Within-hardware normalization prevents unlike device measurement scopes from
being directly mixed. MOAS is our fixed-priority sensitivity metric, not a
standard published metric.

### Preference-Robust Rank Acceptability

No deterministic scalar can rank accuracy, energy, latency, and memory without
encoding a preference about compensation among them. As a complementary
analysis, we use preference-robust rank acceptability inspired by Stochastic
Multicriteria Acceptability Analysis. PRRA is our adaptation, not an existing
standardized metric. Its complete derivation and intellectual provenance are
documented in [Robust Ranking](ROBUST_RANKING.md).

For each model, a fixed rule selects its maximum-throughput successful operating
point on each hardware. For lower-is-better criterion \(k\), its within-hardware
fractional rank cost is

\[
c_{hmk}=\frac{\operatorname{rank}_{h}(x_{hmk})-1}{M_h-1}.
\]

We draw \(B=20{,}000\) symmetric preference vectors,
\(\mathbf w^{(b)}\sim\operatorname{Dirichlet}(1,1,1,1)\), over NRMSE, energy per
forecast, latency per forecast, and relevant peak memory. Each draw ranks

\[
L_{hm}^{(b)}=\sum_k w_k^{(b)}c_{hmk}.
\]

The normalized rank acceptability is

\[
A_{hm}=1-\frac{\mathbb E_b[r_{hm}^{(b)}]-1}{M_h-1}.
\]

We report \(A_{hm}\), expected rank, top-1 and top-3 preference acceptability,
and Pareto-efficient status. Aggregation uses only complete models and gives
each hardware equal weight. Ranking is performed within hardware so physically
different power and memory scopes are never directly normalized together.

These probabilities describe variation over criterion preferences, not sampling
uncertainty or the probability that a model is truly superior. Rank transforms
reduce outlier sensitivity but discard effect magnitude and remain dependent on
the candidate set. Raw metrics, confidence intervals, and Pareto plots are
therefore primary evidence.

EAS, MOAS, and PRRA are reported side by side. EAS exposes the error--energy
trade-off for a declared alpha, MOAS evaluates one declared four-objective
deployment profile, and PRRA tests ranking stability across uncertain
preferences. A model should not be called universally best solely because it
leads any scalar summary.

## Failure And Contamination Policy

Every repeat receives one of `SUCCESS`, `OOM`, `TIMEOUT`, `ERROR`,
`SWAP_CONTAMINATED`, or `HARDWARE_THROTTLED`. Swap growth, Jetson over-current
events, Raspberry Pi throttle flags, missing required telemetry, malformed
predictions, and incomplete repeat sets are surfaced rather than silently averaged.

For CUDA sweeps, a guard fits a non-negative projection to recent peak reserved
memory, applies 1.08 headroom, and compares it with 90% of physical memory (or an
explicit limit). This check occurs outside measurement and prevents native
allocator/compiler failures from corrupting later configurations.

## Interpretation And Fairness

- Compare models at the same dataset, context, horizon, window count, power mode,
  and physical device.
- Retain `power_scope` in every cross-device power or energy table; board/package
  sensors are not equivalent to wall power.
- Report backend, precision, checkpoint revision, point estimator, adapter
  interventions, and compilation state because parameter count alone does not
  determine throughput.
- Expect throughput to increase and eventually saturate in broad terms, but do not
  force monotonicity: kernel changes, transfers, runtime compilation, memory
  pressure, and measurement uncertainty can create local reversals.
- Do not interpret Phase 2 batch size as an accuracy variable.
- Do not mix rows collected before and after a timed adapter intervention.

The exact column definitions and implementation edge cases are normative in
[Measurement](MEASUREMENT.md); execution order is specified in
[Workflow](WORKFLOW.md).
