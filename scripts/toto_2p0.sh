#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"
source "${SCRIPT_DIR}/torch_device.sh"

ENV_NAME="toto_2p0"
VENV_DIR=".venv_${ENV_NAME}"

echo "=== Setting up UV environment for Toto 2.0: ${VENV_DIR} ==="

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment with Python 3.12..."
    uv venv "$VENV_DIR" --python 3.12
else
    echo "Environment ${VENV_DIR} already exists."
fi

source "${VENV_DIR}/bin/activate"

echo "Step 1: Installing device-aware PyTorch stack..."
install_torch_stack

echo "Step 2: Installing official Toto 2.0 package and benchmark dependencies..."
uv pip install \
    "toto-models==1.0.0" \
    "pandas>=2.2,<2.4" \
    "pyarrow" \
    "fastparquet" \
    "psutil" \
    "nvidia-ml-py" \
    "pydantic>=2.0"

echo "Step 3: Restoring the device-aware PyTorch stack..."
install_torch_stack --reinstall

python - <<'PY'
import torch
from toto2 import Toto2Model

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
print("Toto2Model import: OK")
PY

echo "Toto 2.0 setup complete."
