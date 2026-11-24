#!/bin/bash

# Shared PyTorch installer for benchmark setup scripts.
# DEVICE/BENCHMARK_DEVICE can be: jetson, cpu, h100, h200, 5090, a6000, rpi5,
# rpi5_hailo26, or uno_q.

export PATH="/mnt/ssd/bin:$HOME/.local/bin:$PATH"

detect_benchmark_device() {
    local requested="${BENCHMARK_DEVICE:-${DEVICE:-}}"
    requested="$(printf "%s" "$requested" | tr '[:upper:]' '[:lower:]')"

    if [ -n "$requested" ]; then
        echo "$requested"
        return
    fi
    if [ -f /etc/nv_tegra_release ]; then
        echo "jetson"
        return
    fi
    local board_model=""
    if [ -r /proc/device-tree/model ]; then
        board_model=$(tr -d '\000' < /proc/device-tree/model 2>/dev/null || true)
    elif [ -r /sys/firmware/devicetree/base/model ]; then
        board_model=$(tr -d '\000' < /sys/firmware/devicetree/base/model 2>/dev/null || true)
    fi
    case "$(printf "%s" "$board_model" | tr '[:lower:]' '[:upper:]')" in
        *"RASPBERRY PI 5"*)
            if [ -e /dev/hailo0 ]; then
                echo "rpi5_hailo26"
            else
                echo "rpi5"
            fi
            return
            ;;
        *"UNO Q"*|*"QRB2210"*)
            echo "uno_q"
            return
            ;;
    esac
    if [ "${CUDA_VISIBLE_DEVICES:-__unset__}" = "" ]; then
        echo "cpu"
        return
    fi
    if command -v nvidia-smi >/dev/null 2>&1 || [ -d /proc/driver/nvidia ]; then
        echo "gpu"
        return
    fi

    echo "cpu"
}

install_torch_stack() {
    local reinstall=false
    if [ "${1:-}" = "--reinstall" ]; then
        reinstall=true
        shift
    fi

    local device
    device="$(detect_benchmark_device)"

    local reinstall_args=()
    if [ "$reinstall" = "true" ]; then
        reinstall_args=(--reinstall)
    fi

    local packages=("$@")
    if [ "${#packages[@]}" -eq 0 ]; then
        packages=(torch torchvision torchaudio)
    fi

    case "$device" in
        jetson)
            local jetson_index="${JETSON_TORCH_INDEX_URL:-https://download.pytorch.org/whl/nightly/cu132}"
            # Jetson Orin on L4T R39 / CUDA 13.2 needs cu132 wheels; the cu130
            # pip wheels warn for sm_87 and are not a trustworthy benchmark base.
            # shellcheck disable=SC2086
            uv pip install "${reinstall_args[@]}" --pre ${JETSON_TORCH_PACKAGES:-torch torchvision torchaudio} --index-url "$jetson_index"
            ;;
        cpu)
            local cpu_index="${CPU_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
            # CPU envs should not accidentally pull CUDA wheels.
            # shellcheck disable=SC2086
            uv pip install "${reinstall_args[@]}" ${CPU_TORCH_PACKAGES:-torch torchvision torchaudio} --index-url "$cpu_index"
            ;;
        rpi5|rpi5_hailo26|uno_q)
            local arm_cpu_index="${ARM_CPU_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
            # PyPI's generic Linux ARM64 resolver may select NVIDIA CUDA
            # dependencies. These categories execute the TSFM adapters on CPU;
            # vendor NPU runtimes are installed and validated separately.
            uv pip install "${reinstall_args[@]}" "${packages[@]}" --index-url "$arm_cpu_index"
            ;;
        h100|h200)
            # Hopper cloud hosts commonly expose a CUDA 12.9-compatible driver.
            # Keep the wheel family below CUDA 13 unless the operator explicitly
            # overrides GPU_TORCH_INDEX_URL on an R580-or-newer driver.
            local hopper_index="${GPU_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu129}"
            uv pip install "${reinstall_args[@]}" "${packages[@]}" --index-url "$hopper_index"
            ;;
        *)
            local gpu_index="${GPU_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
            uv pip install "${reinstall_args[@]}" "${packages[@]}" --index-url "$gpu_index"
            ;;
    esac
}

restore_model_abi_if_needed() {
    local model_name="${1:-}"
    case "$model_name" in
        moirai2)
            # uni2ts currently resolves pandas 2.1.x on this stack; keep NumPy
            # on the 1.26 ABI after any Torch reinstall that may upgrade NumPy.
            uv pip install "numpy~=1.26.0" "pandas==2.1.4"
            ;;
    esac
}
