#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"
source "${SCRIPT_DIR}/torch_device.sh"

ENV_NAME="flowstate"
VENV_DIR=".venv_${ENV_NAME}"

echo "=== Setting up UV environment for FlowState: ${VENV_DIR} ==="

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment with Python 3.11..."
    uv venv "$VENV_DIR" --python 3.11
else
    echo "Environment ${VENV_DIR} already exists."
fi

source "${VENV_DIR}/bin/activate"

echo "Step 1: Installing device-aware PyTorch stack..."
install_torch_stack "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0"

echo "Step 2: Installing FlowState/Granite TSFM core dependencies..."
uv pip install \
    pandas \
    scikit-learn \
    pyarrow \
    fastparquet \
    numpy \
    psutil \
    nvidia-ml-py \
    "pydantic>=2.0.0"

echo "Step 3: Installing granite-tsfm from source..."
GRANITE_TSFM_REVISION="${GRANITE_TSFM_REVISION:-9c9eb7dd1c1e2ad294afffc00c9460c6b7017f4b}"
uv pip install "granite-tsfm[notebooks] @ git+https://github.com/ibm-granite/granite-tsfm.git@${GRANITE_TSFM_REVISION}"

echo "Step 4: Re-installing device-aware PyTorch stack..."
# granite-tsfm dependencies might force a downgrade to CPU-only torch. Reinstall
# through the shared helper so each device gets the correct wheel set.
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

echo "FlowState Setup complete."
