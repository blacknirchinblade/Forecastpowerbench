#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"
source "${SCRIPT_DIR}/torch_device.sh"

ENV_NAME="moment"
VENV_DIR=".venv_${ENV_NAME}"

echo "=== Setting up UV environment for MOMENT: ${VENV_DIR} ==="

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment with Python 3.11..."
    uv venv "$VENV_DIR" --python 3.11
else
    echo "Environment ${VENV_DIR} already exists."
fi

source "${VENV_DIR}/bin/activate"

echo "Step 1: Installing device-aware PyTorch stack..."
install_torch_stack

echo "Step 2: Installing MOMENT core dependencies..."
uv pip install \
    pandas \
    scikit-learn \
    pyarrow \
    fastparquet \
    psutil \
    momentfm \
    "huggingface-hub>=0.24.0" \
    "numpy==1.25.2" \
    "transformers==4.33.3" \
    nvidia-ml-py \
    "pydantic>=2.0.0"

echo "Step 3: Installing MOMENT from source..."
MOMENT_REVISION="${MOMENT_REVISION:-38f7310ad594100747ca2a8357e9c7ca7d323e0e}"
uv pip install "git+https://github.com/moment-timeseries-foundation-model/moment.git@${MOMENT_REVISION}"

echo "Step 4: Re-installing device-aware PyTorch stack..."
# momentfm and transformers might force a downgrade to the CPU-only torch wheel 
# on aarch64. We force reinstall the CUDA wheels from the official PyTorch index 
# to override this.
install_torch_stack --reinstall

echo "Step 5: Restoring MOMENT dependency pins..."
# momentfm pins these versions; the PyTorch reinstall may upgrade NumPy.
uv pip install "numpy==1.25.2" "transformers==4.33.3"

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

echo "MOMENT Setup complete."
