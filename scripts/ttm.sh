#!/usr/bin/env bash
# Environment setup for TinyTimeMixer.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"
source "${SCRIPT_DIR}/torch_device.sh"

ENV_NAME="ttm"
VENV_DIR=".venv_${ENV_NAME}"

echo "=== Setting up UV environment for TTM: ${VENV_DIR} ==="

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment with Python 3.11..."
    uv venv "$VENV_DIR" --python 3.11
else
    echo "Environment ${VENV_DIR} already exists."
fi

source "${VENV_DIR}/bin/activate"

echo "Installing device-aware PyTorch stack..."
install_torch_stack "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0"

echo "Installing TTM core dependencies..."
uv pip install \
    pandas \
    scikit-learn \
    pyarrow \
    fastparquet \
    numpy \
    psutil \
    nvidia-ml-py \
    "pydantic>=2.0.0"

echo "Installing granite-tsfm from source..."
GRANITE_TSFM_REVISION="${GRANITE_TSFM_REVISION:-9c9eb7dd1c1e2ad294afffc00c9460c6b7017f4b}"
uv pip install "granite-tsfm[notebooks] @ git+https://github.com/ibm-granite/granite-tsfm.git@${GRANITE_TSFM_REVISION}"

# granite-tsfm can pull a CPU-only torch wheel from PyPI. Reinstall through
# the shared helper so CPU, Jetson, and discrete GPUs get the right stack.
echo "============================================================"
echo "Re-installing device-aware PyTorch stack..."
echo "============================================================"
install_torch_stack --reinstall "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0"

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

import numpy
import pandas
import pynvml
print("Torch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("NVML Python bindings: OK")
EOF

echo "TTM Setup complete."
