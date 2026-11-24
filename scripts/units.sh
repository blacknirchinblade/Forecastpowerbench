#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"
source "${SCRIPT_DIR}/torch_device.sh"

ENV_NAME="units"
VENV_DIR=".venv_${ENV_NAME}"
CHECKPOINT_PATH="utils/units/checkpoint/units_x128_pretrain_checkpoint.pth"
CHECKPOINT_URL="https://github.com/mims-harvard/UniTS/releases/download/ckpt/units_x128_pretrain_checkpoint.pth"
CHECKPOINT_SHA256="35d17336c8857d2bacefe403fe7f21ec33ffae9f67d497d569e6af0cb38b5474"

echo "=== Setting up UV environment for UniTS: ${VENV_DIR} ==="

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment with Python 3.11..."
    uv venv "$VENV_DIR" --python 3.11
else
    echo "Environment ${VENV_DIR} already exists."
fi

source "${VENV_DIR}/bin/activate"

echo "Verifying the released UniTS checkpoint..."
if ! printf '%s  %s\n' "$CHECKPOINT_SHA256" "$CHECKPOINT_PATH" | sha256sum -c - >/dev/null 2>&1; then
    command -v curl >/dev/null 2>&1 || {
        echo "ERROR: curl is required to download the UniTS checkpoint." >&2
        exit 1
    }
    mkdir -p "$(dirname "$CHECKPOINT_PATH")"
    rm -f "${CHECKPOINT_PATH}.part"
    curl -fL --retry 3 -o "${CHECKPOINT_PATH}.part" "$CHECKPOINT_URL"
    printf '%s  %s\n' "$CHECKPOINT_SHA256" "${CHECKPOINT_PATH}.part" | sha256sum -c -
    mv "${CHECKPOINT_PATH}.part" "$CHECKPOINT_PATH"
fi

echo "Step 1: Installing device-aware PyTorch stack..."
install_torch_stack

echo "Step 2: Installing UniTS dependencies..."
# NumPy remains below 2.0 for the Pandas/scikit-learn C-API ABI.
uv pip install \
    "einops>=0.4.0" \
    "matplotlib" \
    "numpy>=1.26.0,<2.0" \
    "pandas>=2.0.0" \
    "patool" \
    "reformer-pytorch" \
    "scikit-learn>=1.2.0" \
    "scipy>=1.10.0" \
    "sktime>=0.20.0" \
    "sympy" \
    "tqdm" \
    pyyaml \
    timm \
    pyarrow \
    fastparquet \
    psutil \
    nvidia-ml-py \
    "pydantic>=2.0.0"

echo "Step 3: Re-installing device-aware PyTorch stack..."
# UniTS dependencies might try to downgrade torch. Reinstall the device-aware stack.
install_torch_stack --reinstall

echo "Environment setup complete."
echo "Active environment: $VIRTUAL_ENV"
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"

# Validate imports and device visibility.
echo "Validating runtime imports and device visibility..."
python - <<EOF
import warnings
warnings.filterwarnings("ignore", message=".*Found GPU.*compute capability.*")
warnings.filterwarnings("ignore", message="The pynvml package is deprecated.*")

import torch
import numpy
import pandas
print("Torch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)

try:
    from torchvision.transforms import InterpolationMode
    print("Torchvision verification: SUCCESS")
except Exception as e:
    print("Torchvision verification: FAILED ->", str(e))

try:
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2, 3]})
    print("Pandas/NumPy ABI verification: SUCCESS")
except Exception as e:
    print("Pandas/NumPy ABI verification: FAILED ->", str(e))
    
try:
    import pynvml
    pynvml.nvmlInit()
    print("NVML/GPU Metrics verification: SUCCESS")
    pynvml.nvmlShutdown()
except Exception as e:
    print("NVML verification: FAILED ->", str(e))
EOF
echo "UniTS Setup complete."
