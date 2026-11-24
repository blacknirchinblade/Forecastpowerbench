# Tests

The first-party suite covers data isolation, prepared batches, reproducibility,
accuracy formulas, adapter semantics, hardware profiles, monitor aggregation,
memory guards, resume behavior, supplemental sweeps, and artifact validation.

Run it from the generated TTM environment, which contains PyTorch, Granite TSFM,
and the benchmark runtime dependencies required during test collection. Add only
the test tools to that environment:

```bash
source .venv_ttm/bin/activate
uv pip install -e ".[test]"
ruff check analysis benchmark experiments models scripts tests
python -m pytest -q tests
```

Adapter tests use lightweight fakes where possible. They do not replace
`smoke_test.sh`, which qualifies real model environments and checkpoints on the
target hardware.
