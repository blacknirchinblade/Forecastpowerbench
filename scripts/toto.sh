#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"
source "${SCRIPT_DIR}/torch_device.sh"

ENV_NAME="toto"
VENV_DIR=".venv_${ENV_NAME}"

echo "=== Setting up UV environment for Toto: ${VENV_DIR} ==="

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment with Python 3.11..."
    uv venv "$VENV_DIR" --python 3.11
else
    echo "Environment ${VENV_DIR} already exists."
fi

source "${VENV_DIR}/bin/activate"

echo "Step 1: Installing device-aware PyTorch stack..."
install_torch_stack "torch"

echo "Step 2: Installing Toto core and pinned dependencies..."
uv pip install \
    "aioboto3==12.4.0" \
    "beartype==0.18.5" \
    "boto3==1.34.69" \
    "datasets==2.17.1" \
    "dill==0.3.8" \
    "einops==0.7.0" \
    "numpy==1.26.4" \
    "gluonts[torch]==0.16.2" \
    "jaxtyping==0.2.29" \
    "mypy==1.11.1" \
    "pandas==2.2.3" \
    "pytest==8.3.5" \
    "pytest-env==1.1.5" \
    "rotary-embedding-torch==0.8.6" \
    "scikit-learn==1.5.0" \
    "types-tabulate==0.9.0.20241207" \
    "transformers==4.52.1" \
    "tqdm==4.66.3" \
    "pyyaml==6.0.2" \
    "types-PyYAML==6.0.12.20240917" \
    "black==24.10.0" \
    "isort==5.13.2" \
    "jupyter==1.1.1" \
    "matplotlib==3.9.2" \
    "tabulate==0.9.0" \
    "lightning==2.3.3" \
    "rich>=10.2.2" \
    "tensorboard==2.20.0" \
    pyarrow \
    fastparquet \
    psutil \
    nvidia-ml-py \
    "pydantic>=2.0.0"

echo "Step 3: Re-installing device-aware PyTorch stack..."
# Toto's dependencies (like lightning or gluonts) might force a downgrade 
# to the CPU-only torch wheel on aarch64. We force reinstall the CUDA wheels 
# from the official PyTorch index to override this.
install_torch_stack --reinstall "torch"

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

echo "Toto Setup complete."
