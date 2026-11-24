# Repository Structure

The review artifact contains executable source, tests, configuration templates,
and documentation. Runtime data and generated artifacts are intentionally absent.

```text
.
|-- README.md                         Project overview and quick start
|-- .benchmark-device.env.example     Shareable host configuration template
|-- pyproject.toml                    Package metadata, entry points, and lint policy
|-- MANIFEST.in                      Source-distribution file policy
|-- run_unified_benchmark.sh          Two-phase benchmark launcher
|-- run_horizon_sweep.sh              Context/horizon experiment launcher
|-- run_ttm_r3_variant_sweep.sh       TTM-R3 routing experiment launcher
|-- smoke_test.sh                     End-to-end qualification gate
|-- analysis/                         Publication analysis and ranking functions
|-- benchmark/                        Protocol, data, runner, devices, telemetry
|-- powerbench/                       Installable package facade and CLI
|-- experiments/forecasting.py        Per-model Python entry point
|-- models/                           Adapter contract, registry, implementations
|-- scripts/                          Environment setup, resume, validation
|-- tests/                            First-party regression suite
|-- utils/                            Metrics and required upstream support code
|-- dataset/README.md                 Input-data contract; data are not committed
|-- docs/                             Methodology and operations documentation
|-- evaluate_saved_predictions.ipynb  Post-hoc forecast evaluation
`-- plot_publication_results.ipynb    Saved-result visualization
```

## Runtime Artifacts

The following paths are created locally and excluded from Git:

- `.benchmark-device.env` for host-specific settings;
- `.venv_<model>/` for isolated model environments;
- `.model_sources/` and model caches for pinned upstream downloads;
- `dataset/<dataset>/` and `dataset/.powerbench_cache/` for inputs and preprocessing;
- `results_unified/`, `all_hardwares_results/`, and `test_results/` for outputs;
- Python, pytest, Ruff, notebook, and operating-system caches;
- model weight formats such as `.pth`, `.pt`, `.ckpt`, and `.safetensors`.

## Naming Policy

Names that identify an evaluated checkpoint or published architecture are retained,
including TTM-R3, TTM-R2, Moirai 2.0, TimesFM 2.5, and Toto 2.0. They are scientific
provenance, not development revisions. Temporary suffixes, copied files, generated
schema labels, dated smoke artifacts, and hardware-local result names are excluded
from the source artifact.

## Source Boundaries

`powerbench/`, `analysis/`, `benchmark/`, `experiments/`, `models/`, and the top-level launchers are first-party
benchmark code. Portions of `utils/` are required upstream implementations used by
adapters. Their origin and role are listed in `THIRD_PARTY.md`; local adapters remain
responsible for transport, point-forecast extraction, and benchmark integration.
