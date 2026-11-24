#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"
source "${SCRIPT_DIR}/torch_device.sh"

ENV_NAME="moirai2"
VENV_DIR=".venv_${ENV_NAME}"

echo "=== Setting up UV environment for Moirai: ${VENV_DIR} ==="

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment with Python 3.11..."
    uv venv "$VENV_DIR" --python 3.11
else
    echo "Environment ${VENV_DIR} already exists."
fi

source "${VENV_DIR}/bin/activate"

echo "Step 1: Installing device-aware PyTorch stack..."
install_torch_stack

echo "Step 2: Installing Moirai/uni2ts core and pinned dependencies..."
# JAX is unnecessary for this PyTorch inference path and problematic on ARM64.
uv pip install \
    "numpy~=1.26.0" \
    "scipy~=1.11.3" \
    "gluonts~=0.14.3" \
    "lightning>=2.0" \
    "einops==0.7.*" \
    "jaxtyping~=0.2.24" \
    "datasets~=2.17.1" \
    orjson \
    multiprocess \
    "python-dotenv==1.0.0" \
    "hydra-core==1.3" \
    tensorboard \
    "huggingface_hub>=0.23.0" \
    safetensors \
    pandas \
    scikit-learn \
    pyarrow \
    fastparquet \
    psutil \
    nvidia-ml-py \
    uni2ts \
    "pydantic>=2.0.0"

echo "Step 3: Re-installing device-aware PyTorch stack..."
# uni2ts and lightning dependencies might force a downgrade to CPU-only torch.
# Reinstall through the shared helper so each device gets the correct wheel set.
install_torch_stack --reinstall

echo "Step 4: Restoring Moirai NumPy/Pandas ABI compatibility..."
# The PyTorch reinstall can upgrade NumPy to 2.x. uni2ts resolves pandas 2.1.x,
# which requires the NumPy 1.26 ABI.
restore_model_abi_if_needed "${ENV_NAME}"

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

echo "Moirai Setup complete."
