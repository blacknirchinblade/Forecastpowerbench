#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"
source "${SCRIPT_DIR}/torch_device.sh"

ENV_NAME="tempopfn"
VENV_DIR=".venv_${ENV_NAME}"

echo "=== Setting up UV environment for TempoPFN: ${VENV_DIR} ==="

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment with uv-managed Python 3.11..."
    uv venv "$VENV_DIR" --python 3.11 --managed-python
else
    echo "Environment ${VENV_DIR} already exists."
fi

source "${VENV_DIR}/bin/activate"

echo "Checking Python development headers required by Triton..."
PYTHON_INCLUDE_DIR="$(python - <<'EOF'
import os
import sys
import sysconfig

version_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
system_prefixes = ("/usr", "/usr/local")

def is_system_path(path):
    path = os.path.abspath(path)
    return path in system_prefixes or any(path.startswith(prefix + os.sep) for prefix in system_prefixes)

candidate_dirs = [
    os.path.join(sys.prefix, "include", version_dir),
    os.path.join(os.environ.get("VIRTUAL_ENV", ""), "include", version_dir),
]

base_include = os.path.join(sys.base_prefix, "include", version_dir)
if not is_system_path(sys.base_prefix):
    candidate_dirs.append(base_include)

sysconfig_include = sysconfig.get_paths().get("include") or sysconfig.get_config_var("INCLUDEPY")
if sysconfig_include and not is_system_path(sysconfig_include):
    candidate_dirs.append(sysconfig_include)

for candidate in dict.fromkeys(path for path in candidate_dirs if path):
    path = os.path.join(candidate, "Python.h")
    if os.path.exists(path):
        print(candidate)
        sys.exit(0)

print("", file=sys.stderr)
print("ERROR: Missing compatible Python.h for this interpreter.", file=sys.stderr)
print(f"Python executable: {sys.executable}", file=sys.stderr)
print(f"Expected Python include version: {version_dir}", file=sys.stderr)
print("Checked:", file=sys.stderr)
for candidate in dict.fromkeys(path for path in candidate_dirs if path):
    print(f"  {os.path.join(candidate, 'Python.h')}", file=sys.stderr)
if is_system_path(sys.base_prefix):
    print(f"Skipped system Python base: {sys.base_prefix}", file=sys.stderr)
print("", file=sys.stderr)
print("TempoPFN uses Flash Linear Attention/Triton CUDA kernels. Triton JIT-compiles", file=sys.stderr)
print("a small CUDA driver helper at runtime, and that compile requires Python.h.", file=sys.stderr)
print("", file=sys.stderr)
print("Recreate this model environment with uv-managed Python so headers come", file=sys.stderr)
print("from the created environment instead of the host system:", file=sys.stderr)
print(f"  rm -rf {os.environ.get('VIRTUAL_ENV', '.venv_tempopfn')}", file=sys.stderr)
print("  bash scripts/tempopfn.sh", file=sys.stderr)
sys.exit(1)
EOF
)"

export CPATH="${PYTHON_INCLUDE_DIR}"
echo "Python.h found: ${PYTHON_INCLUDE_DIR}/Python.h"

echo "Step 1: Installing device-aware PyTorch stack..."
install_torch_stack

echo "Step 2: Installing TempoPFN core dependencies..."
uv pip install \
    pandas \
    scikit-learn \
    pyarrow \
    fastparquet \
    numpy \
    psutil \
    torchmetrics \
    gpytorch \
    gluonts \
    matplotlib \
    notebook \
    datasets \
    ujson \
    huggingface_hub \
    "transformers==4.56.1" \
    pyyaml \
    nvidia-ml-py \
    "pydantic>=2.0.0"

echo "Step 3: Installing the validated Flash Linear Attention revision..."
uv pip install einops ninja packaging
# Keep FLA from silently replacing the device-aware Torch/Triton stack or the
# TempoPFN-validated Transformers version selected above.
uv pip install --no-deps git+https://github.com/fla-org/flash-linear-attention@ccb0ff944cbff035fa59ac47a4cc8fd2e079bb17

# PyTorch Inductor imports triton.backends even for some CPU-side module
# imports. TempoPFN's CPU fallback must therefore use a real Triton package on
# x86_64 instead of exposing its minimal kernel stub as the top-level package.
BENCHMARK_DEVICE_KIND="$(detect_benchmark_device)"
if [ "$BENCHMARK_DEVICE_KIND" = "cpu" ] && [ "$(uname -m)" = "x86_64" ]; then
    echo "Installing the validated x86 CPU Triton compatibility package..."
    uv pip install "triton==3.7.1"
fi

echo "Step 4: Re-installing device-aware PyTorch stack..."
# Dependencies like gpytorch or flash-linear-attention might force a downgrade
# to the CPU-only torch wheel on aarch64. We force reinstall the CUDA wheels
# from the official PyTorch index to override this.
install_torch_stack --reinstall

echo "Environment setup complete."
echo "Active environment: $VIRTUAL_ENV"
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"

# Validate imports and device visibility.
echo "Validating runtime imports and device visibility..."
python - <<EOF
import torch
import warnings
warnings.filterwarnings("ignore", message=".*Found GPU.*compute capability.*")
warnings.filterwarnings("ignore", message="The pynvml package is deprecated.*")

import numpy
import pandas
import transformers
print("Torch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("Transformers:", transformers.__version__)
if transformers.__version__ != "4.56.1":
    raise RuntimeError(
        "TempoPFN environment drifted from transformers==4.56.1; "
        f"found {transformers.__version__}"
    )

if "${BENCHMARK_DEVICE_KIND}" == "cpu" and __import__("platform").machine() == "x86_64":
    import triton
    import triton.backends.compiler
    print("Triton CPU import compatibility:", triton.__version__)

try:
    from torchvision.transforms import InterpolationMode
    print("Torchvision verification: SUCCESS")
except Exception as e:
    print("Torchvision verification: FAILED ->", str(e))

try:
    import pynvml
    pynvml.nvmlInit()
    print("NVML/GPU Metrics verification: SUCCESS")
    pynvml.nvmlShutdown()
except Exception as e:
    print("NVML verification: FAILED ->", str(e))

if "${BENCHMARK_DEVICE_KIND}" in {"h100", "h200"}:
    from models.tempopfn import TempoPFNModel, _HopperSafeRMSNormGated
    from fla.modules import FusedRMSNormGated

    policy = TempoPFNModel._configure_norm_policy()
    if policy != "hopper_torch":
        raise RuntimeError(f"Expected Hopper PyTorch norm policy, found {policy}")

    rows = 65537
    hidden_size = 128
    generator = torch.Generator(device="cuda").manual_seed(12345)
    x = torch.randn(
        rows,
        hidden_size,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    gate = torch.randn(
        rows,
        hidden_size,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    fused_norm = FusedRMSNormGated(
        hidden_size,
        eps=1e-5,
    ).cuda().bfloat16().eval()
    norm = _HopperSafeRMSNormGated(fused_norm).eval()
    with torch.inference_mode():
        actual = norm(x, gate)
        reference = x.float()
        reference = reference * torch.rsqrt(
            reference.square().mean(dim=-1, keepdim=True) + fused_norm.eps
        )
        reference = reference * fused_norm.weight.float()
        reference = reference * torch.nn.functional.silu(gate.float())
        reference = reference.to(actual.dtype)
    torch.cuda.synchronize()
    if not torch.isfinite(actual).all():
        raise RuntimeError("Hopper PyTorch norm validation produced non-finite values")
    if not torch.allclose(actual, reference, rtol=3e-2, atol=3e-2):
        max_error = (actual.float() - reference.float()).abs().max().item()
        raise RuntimeError(
            f"Hopper PyTorch norm validation mismatch; max_abs_error={max_error}"
        )
    print(
        "Hopper TempoPFN PyTorch norm verification: SUCCESS "
        f"(policy={policy}, rows={rows})"
    )
EOF

echo "TempoPFN Setup complete."
