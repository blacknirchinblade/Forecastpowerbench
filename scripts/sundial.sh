#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"
source "${SCRIPT_DIR}/torch_device.sh"

ENV_NAME="sundial"
VENV_DIR=".venv_${ENV_NAME}"

echo "=== Setting up UV environment for Sundial: ${VENV_DIR} ==="

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment with Python 3.11..."
    uv venv "$VENV_DIR" --python 3.11
else
    echo "Environment ${VENV_DIR} already exists."
fi

source "${VENV_DIR}/bin/activate"

echo "Step 1: Installing device-aware PyTorch stack..."
install_torch_stack

echo "Step 2: Installing Sundial core dependencies..."
uv pip install \
    pandas \
    scikit-learn \
    pyarrow \
    fastparquet \
    numpy \
    psutil \
    "transformers==4.40.1" \
    nvidia-ml-py \
    "pydantic>=2.0.0"

echo "Step 3: Re-installing device-aware PyTorch stack..."
# transformers and other dependencies might force a downgrade to the CPU-only 
# torch wheel on aarch64. We force reinstall the CUDA wheels from the official 
# PyTorch index to override this.
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
    import pynvml
    pynvml.nvmlInit()
    print("NVML/GPU Metrics verification: SUCCESS")
    pynvml.nvmlShutdown()
except Exception as e:
    print("NVML verification: FAILED ->", str(e))
EOF

echo "Sundial Setup complete."
