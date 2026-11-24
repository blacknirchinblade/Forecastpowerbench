# Scripts

Model setup scripts create `.venv_<model>` environments, install device-aware
PyTorch and model dependencies, download released artifacts, and run import/device
validation. Git-sourced dependencies are revision-pinned; the launcher records the
fully resolved package set for every environment. `torch_device.sh` centralizes CPU,
discrete-GPU, Hopper, Jetson, and ARM wheel selection.

Operational utilities:

- `prediction_resume.py` identifies complete Phase-1 model outputs;
- `benchmark_resume.py` identifies complete Phase-2 model summaries;
- `supplemental_batches.py` validates and plans additional batch sizes;
- `validate_results.py` enforces prediction and benchmark artifact contracts;
- `inspect_ttm_r3_variants.py` reports TTM-R3 context/horizon routing.

Invoke setup scripts through `run_unified_benchmark.sh` or `smoke_test.sh` so host
configuration, logging, and environment qualification remain consistent.
