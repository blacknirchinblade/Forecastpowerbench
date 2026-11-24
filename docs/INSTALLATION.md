# Installation

## Prerequisites

- Linux with Python 3.11, Git, `curl`, and `uv`;
- sufficient local storage for isolated model environments and checkpoints;
- NVIDIA driver and `nvidia-smi` for discrete GPUs;
- an NVIDIA Jetson software stack with `nvpmodel`, `jetson_clocks`, and
  `tegrastats` for Jetson;
- readable Intel RAPL or configured sysfs power sensors when those measurements
  are required on CPU-class devices.

Install `uv` using its official installer if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Install The Python Package

For development, analysis, or programmatic use, install the checkout in editable
mode. The command can then be called from any directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[analysis,test]"
powerbench --help
```

The core package intentionally does not declare PyTorch or model-framework
dependencies. Those versions differ by hardware and model, so the production
launcher installs them into isolated `.venv_<model>` environments using the files
under `scripts/`. Installing the package does not download datasets or weights.

## Configure A Host

```bash
cp .benchmark-device.env.example .benchmark-device.env
```

Set `BENCHMARK_DEVICE` and dataset controls in the local file. Do not commit or
synchronize `.benchmark-device.env`; power modes, GPU index, sensor paths, cache
mounts, and cloud provenance are host-specific.

Place source CSV files according to [the dataset contract](../dataset/README.md),
then inspect the resolved configuration:

```bash
bash run_unified_benchmark.sh --check-config
```

Launchers are location-independent: each changes to its own repository directory
before resolving relative paths. Therefore the following is equivalent and does
not write runtime files into `/tmp`:

```bash
cd /tmp
REPOSITORY="$(realpath ../benchmark-checkout)"
bash "$REPOSITORY/run_unified_benchmark.sh" --check-config
```

## Qualify Before Measurement

```bash
bash smoke_test.sh
```

The qualification creates or repairs one environment per model, downloads released
weights, exercises prediction and benchmark phases, and validates the resulting
artifacts. Qualification outputs are functional checks, not research measurements.

After the TTM model environment exists, run the first-party regression suite as
described in [`tests/README.md`](../tests/README.md).

## Run The Study

```bash
bash run_unified_benchmark.sh --phase prediction
bash run_unified_benchmark.sh --phase benchmark
```

Use `--phase all` for the complete sequence. Both phases resume independently at
model boundaries. See [Operations](OPERATIONS.md) before changing protocol controls.
