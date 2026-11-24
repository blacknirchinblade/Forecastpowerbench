#!/bin/bash
set -euo pipefail

# Handles per-model environment isolation, independent source files,
# pre-flight validation, and power mode sweeps.

# Configuration. The launcher is identical on every host; only this small
# local file identifies which benchmark device the host should use.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEVICE_CONFIG_FILE="${DEVICE_CONFIG_FILE:-$SCRIPT_DIR/.benchmark-device.env}"
if [ -f "$DEVICE_CONFIG_FILE" ]; then
    # shellcheck disable=SC1090
    source "$DEVICE_CONFIG_FILE"
fi
cd "$SCRIPT_DIR"

RUN_PHASE="${RUN_PHASE:-all}"
TTM_PREFER_KNOWN_MAPPINGS="${TTM_PREFER_KNOWN_MAPPINGS:-false}"
CHECK_CONFIG=false
while [ "$#" -gt 0 ]; do
    case "$1" in
        --phase)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] --phase requires all, prediction, or benchmark."
                exit 2
            fi
            RUN_PHASE="$2"
            shift 2
            ;;
        --ttm-prefer-known-mappings)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] --ttm-prefer-known-mappings requires true or false."
                exit 2
            fi
            TTM_PREFER_KNOWN_MAPPINGS="$2"
            shift 2
            ;;
        --check-config)
            CHECK_CONFIG=true
            shift
            ;;
        --help|-h)
            cat <<'EOF'
Usage: bash run_unified_benchmark.sh [--phase all|prediction|benchmark]
                                     [--ttm-prefer-known-mappings true|false]
                                     [--check-config]

  all         Generate predictions, cool, then benchmark (default).
  prediction  Independently generate/validate predictions; auto-resume Phase 1.
  benchmark   Independently measure/validate performance; auto-resume Phase 2.

TTM-R3 automatic model selection:
  false  Select the smallest/closest compatible official R3 variant (default).
  true   Use IBM's legacy known-mapping policy.

Supplement an existing benchmark without modifying other batch sizes:
  SUPPLEMENTAL_BATCHES=true BATCH_SIZES=16,32 \
    bash run_unified_benchmark.sh --phase benchmark
EOF
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            exit 2
            ;;
    esac
done
case "$RUN_PHASE" in
    all|prediction|benchmark) ;;
    *)
        echo "[ERROR] Invalid phase '$RUN_PHASE'; expected all, prediction, or benchmark."
        exit 2
        ;;
esac
case "$TTM_PREFER_KNOWN_MAPPINGS" in
    true|false) ;;
    *)
        echo "[ERROR] TTM_PREFER_KNOWN_MAPPINGS must be true or false."
        exit 2
        ;;
esac
export TTM_PREFER_KNOWN_MAPPINGS

export PATH="/mnt/ssd/bin:$HOME/.local/bin:$PATH"
BASE_PATH="$PATH"
# Stable model order preserves model-boundary resume behavior.
# DEFAULT_MODELS="units kairos ttm_r3_auto ttm_r3_auto_lite moirai2 tempopfn timesfm_2p5 flowstate sundial toto chronos2 tirex"
DEFAULT_MODELS="units kairos ttm_r3_auto ttm_r3_auto_lite moirai2 tempopfn timesfm_2p5 flowstate sundial toto chronos2 tirex"
read -r -a MODELS <<< "${MODEL_LIST:-$DEFAULT_MODELS}"
DATA_DIR="${DATA_DIR:-./dataset}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-}"
DATASET_FOLDER="${DATASET_FOLDER:-cnemc_imputed}"
POLLUTANT="${POLLUTANT:-PM2.5}"
SOURCE_PATTERN="${SOURCE_PATTERN:-${DATASET_FOLDER%/}/**/*_${POLLUTANT}.csv}"
EXCLUDE_SOURCE="${EXCLUDE_SOURCE:-}"
TARGET_COLUMN="${TARGET_COLUMN:-}"
TIME_COLUMN="${TIME_COLUMN:-}"
WINDOW_STRIDE="${WINDOW_STRIDE:-}"
CTX_LENS="${CTX_LENS:-512}"
HORIZON_LENS="${HORIZON_LENS:-96}"
BATCH_SIZES="${BATCH_SIZES:-64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144}"
WARMUPS="${WARMUPS:-2}"
REPEATS_WAS_SET="${REPEATS+x}"
REPEATS="${REPEATS:-5}"
CPU_CONFIG_TIMEOUT_SEC="${CPU_CONFIG_TIMEOUT_SEC:-18000}"
PREDICTION_BATCH_SIZE="${PREDICTION_BATCH_SIZE:-128}"
BENCHMARK_BATCHES="${BENCHMARK_BATCHES:-300}"
BENCHMARK_SEED="${BENCHMARK_SEED:-12345}"
ACCURACY_SEASONALITY="${ACCURACY_SEASONALITY:-24}"
PHASE_COOLDOWN_SEC="${PHASE_COOLDOWN_SEC:-1200}"
MODEL_COOLDOWN_SEC="${MODEL_COOLDOWN_SEC:-300}"
MAX_IDLE_GPU_UTIL="${MAX_IDLE_GPU_UTIL:-20.0}"
MAX_IDLE_CPU_UTIL="${MAX_IDLE_CPU_UTIL:-30.0}"
GPU_PREFLIGHT_WAIT_SEC="${GPU_PREFLIGHT_WAIT_SEC:-120}"
GPU_PREFLIGHT_POLL_SEC="${GPU_PREFLIGHT_POLL_SEC:-5}"
export GPU_PREFLIGHT_WAIT_SEC GPU_PREFLIGHT_POLL_SEC
OUTPUT_DIR="${OUTPUT_DIR:-./results_unified}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-$OUTPUT_DIR/data_cache}"
DEVICE="${DEVICE:-${BENCHMARK_DEVICE:-cpu}}" # Options: jetson, h100, h200, a6000, 5090, cpu, rpi5, rpi5_hailo26, uno_q
export DEVICE
case "$(printf "%s" "$DEVICE" | tr '[:upper:]' '[:lower:]')" in
    jetson|h100|h200|a6000|5090|cpu|rpi5|rpi5_hailo26|uno_q) ;;
    *)
        echo "[ERROR] Unsupported DEVICE=$DEVICE"
        exit 2
        ;;
esac
case "$(printf "%s" "$DEVICE" | tr '[:upper:]' '[:lower:]')" in
    cpu)
        if [ -z "$REPEATS_WAS_SET" ]; then
            REPEATS=2
        fi
        CONFIG_TIMEOUT_SEC="$CPU_CONFIG_TIMEOUT_SEC"
        ;;
    *)
        CONFIG_TIMEOUT_SEC=0
        ;;
esac
if [ -z "${POWER_MODES:-}" ]; then
    case "$(printf "%s" "$DEVICE" | tr '[:upper:]' '[:lower:]')" in
        # Use true MAXN on Jetson. Hardware OC protection remains enabled and
        # telemetry invalidates any repeat in which an OC counter increments.
        jetson) POWER_MODES="0" ;;
        *) POWER_MODES="0" ;;
    esac
fi
POWER_MODE_SWITCHING="${POWER_MODE_SWITCHING:-true}"
POWER_MODE_STABILIZATION_SEC="${POWER_MODE_STABILIZATION_SEC:-100}"
IDLE_BASELINE_SEC="${IDLE_BASELINE_SEC:-30.0}"
SLEEP_BETWEEN_BATCHES_SEC="${SLEEP_BETWEEN_BATCHES_SEC:-300}"
case "$(printf "%s" "$DEVICE" | tr '[:upper:]' '[:lower:]')" in
    cpu)
        DEFAULT_NUM_WORKERS=0
        DEFAULT_CPU_THREADS=$(nproc 2>/dev/null || echo 1)
        ;;
    rpi5|rpi5_hailo26|uno_q)
        DEFAULT_NUM_WORKERS=0
        DEFAULT_CPU_THREADS=$(nproc 2>/dev/null || echo 1)
        ;;
    *)
        # Batches are vectorized from RAM-resident NumPy arrays. Extra loader
        # workers only prefetch large copies and can distort host RAM metrics.
        DEFAULT_NUM_WORKERS=0
        DEFAULT_CPU_THREADS=$(nproc 2>/dev/null || echo 1)
        ;;
esac
NUM_WORKERS="${NUM_WORKERS:-$DEFAULT_NUM_WORKERS}"
CPU_THREADS="${CPU_THREADS:-$DEFAULT_CPU_THREADS}"
CPU_RAM_SAFETY_FRACTION="${CPU_RAM_SAFETY_FRACTION:-0.82}"
CPU_RAM_ESTIMATE_HEADROOM="${CPU_RAM_ESTIMATE_HEADROOM:-1.08}"
GPU_MEMORY_SAFETY_FRACTION="${GPU_MEMORY_SAFETY_FRACTION:-0.90}"
GPU_MEMORY_ESTIMATE_HEADROOM="${GPU_MEMORY_ESTIMATE_HEADROOM:-1.08}"
BENCHMARK_GPU_INDEX="${BENCHMARK_GPU_INDEX:-0}"
CLOUD_PROVIDER="${CLOUD_PROVIDER:-}"
CLOUD_INSTANCE_TYPE="${CLOUD_INSTANCE_TYPE:-}"
CLOUD_REGION="${CLOUD_REGION:-}"
CLOUD_INSTANCE_ID="${CLOUD_INSTANCE_ID:-}"
if ! [[ "$BENCHMARK_GPU_INDEX" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] BENCHMARK_GPU_INDEX must be a non-negative integer."
    exit 2
fi
export GPU_MEMORY_SAFETY_FRACTION GPU_MEMORY_ESTIMATE_HEADROOM
export BENCHMARK_GPU_INDEX
export CLOUD_PROVIDER CLOUD_INSTANCE_TYPE CLOUD_REGION CLOUD_INSTANCE_ID
export GPU_TORCH_INDEX_URL
MAX_SOURCES="${MAX_SOURCES:-}"
MAX_WINDOWS="${MAX_WINDOWS:-}"
SAVE_RAW_PERFORMANCE_METRICS="${SAVE_RAW_PERFORMANCE_METRICS:-false}"
APPEND_RESULTS="${APPEND_RESULTS:-false}"
TQDM_MININTERVAL="${TQDM_MININTERVAL:-2.0}"
SUDO_KEEPALIVE="${SUDO_KEEPALIVE:-true}"
DISABLE_SWAP_DURING_BENCHMARK="${DISABLE_SWAP_DURING_BENCHMARK:-true}"
OPEN_FILES_LIMIT="${OPEN_FILES_LIMIT:-1048576}"
JETSON_TORCH_INDEX_URL="${JETSON_TORCH_INDEX_URL:-https://download.pytorch.org/whl/nightly/cu132}"
JETSON_TORCH_PACKAGES="${JETSON_TORCH_PACKAGES:-torch torchvision torchaudio}"
POWER_SENSOR_PATH="${POWER_SENSOR_PATH:-}"
POWER_SENSOR_SCALE="${POWER_SENSOR_SCALE:-1e-6}"
GPU_LOCK_CLOCKS="${GPU_LOCK_CLOCKS:-true}"
AUTO_RESUME="${AUTO_RESUME:-true}"
SUPPLEMENTAL_BATCHES="${SUPPLEMENTAL_BATCHES:-false}"
export GPU_LOCK_CLOCKS
export POWER_SENSOR_PATH POWER_SENSOR_SCALE
export BENCHMARK_SEED
# PYTHONHASHSEED must be set before each model interpreter starts.
export PYTHONHASHSEED="$BENCHMARK_SEED"
BASE_OUTPUT_DIR="$OUTPUT_DIR"
RUN_STAMP=$(date +%Y%m%d_%H%M%S)

case "$SUPPLEMENTAL_BATCHES" in
    true|false) ;;
    *)
        echo "[ERROR] SUPPLEMENTAL_BATCHES must be true or false."
        exit 2
        ;;
esac
if [ "$SUPPLEMENTAL_BATCHES" = "true" ] && [ "$RUN_PHASE" != "benchmark" ]; then
    echo "[ERROR] SUPPLEMENTAL_BATCHES=true requires --phase benchmark."
    exit 2
fi

if [ "$RUN_PHASE" = "all" ] && [ "$AUTO_RESUME" != "true" ] \
    && [ "$APPEND_RESULTS" != "true" ] \
    && { [ -f "$OUTPUT_DIR/benchmark_results_batch_summary.csv" ] || [ -d "$OUTPUT_DIR/benchmark_predictions" ]; }; then
    OUTPUT_DIR="$BASE_OUTPUT_DIR/run_$RUN_STAMP"
fi

MODEL_LOG_DIR="$OUTPUT_DIR/model_logs"
export TQDM_MININTERVAL

normalized_device() {
    printf "%s" "$DEVICE" | tr '[:upper:]' '[:lower:]'
}

model_environment_name() {
    case "$1" in
        ttm_r2|ttm_r3_auto|ttm_r3_512_96_dec|ttm_r3_512_96_dec_lite|ttm_r3_1024_720|ttm_r3_auto_lite)
            printf '%s\n' "ttm"
            ;;
        chronos_bolt_base) printf '%s\n' "chronos2" ;;
        kairos_50m) printf '%s\n' "kairos" ;;
        toto_2p0_4m|toto_2p0_2p5b) printf '%s\n' "toto_2p0" ;;
        *) printf '%s\n' "$1" ;;
    esac
}

is_cpu_device() {
    case "$(normalized_device)" in
        cpu|rpi5|rpi5_hailo26|uno_q)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

is_jetson_device() {
    [ "$(normalized_device)" = "jetson" ]
}

is_discrete_nvidia_device() {
    case "$(normalized_device)" in
        h100|h200|a6000|5090)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

select_jetson_cache_root() {
    local repo_source target source rotational
    repo_source=$(findmnt -T "$SCRIPT_DIR" -n -o SOURCE 2>/dev/null || true)
    case "$repo_source" in
        /dev/nvme*)
            printf '%s\n' "$SCRIPT_DIR/.powerbench_cache"
            return
            ;;
    esac

    while read -r target source; do
        case "$source" in
            /dev/nvme*)
                if [ -d "$target" ] && [ -w "$target" ]; then
                    printf '%s\n' "$target/powerbench/.powerbench_cache"
                    return
                fi
                ;;
        esac
    done < <(findmnt -rn -o TARGET,SOURCE 2>/dev/null || true)

    while read -r target source; do
        case "$source" in
            /dev/mmc*|tmpfs|overlay|"")
                continue
                ;;
        esac
        rotational=$(lsblk -dn -o ROTA "$source" 2>/dev/null | head -1 || true)
        if [ "$rotational" = "0" ] && [ -d "$target" ] && [ -w "$target" ]; then
            printf '%s\n' "$target/powerbench/.powerbench_cache"
            return
        fi
    done < <(findmnt -rn -o TARGET,SOURCE 2>/dev/null || true)

    # No writable SSD was detected; keep the benchmark functional on eMMC.
    printf '%s\n' "$SCRIPT_DIR/.powerbench_cache"
}

BENCHMARK_CACHE_ROOT="${BENCHMARK_CACHE_ROOT:-}"
if is_jetson_device && [ -z "$BENCHMARK_CACHE_ROOT" ]; then
    BENCHMARK_CACHE_ROOT=$(select_jetson_cache_root)
fi
if [ -n "$BENCHMARK_CACHE_ROOT" ]; then
    BENCHMARK_CACHE_ROOT=$(mkdir -p "$BENCHMARK_CACHE_ROOT" && cd "$BENCHMARK_CACHE_ROOT" && pwd)
    mkdir -p \
        "$BENCHMARK_CACHE_ROOT/huggingface" \
        "$BENCHMARK_CACHE_ROOT/torch" \
        "$BENCHMARK_CACHE_ROOT/uv" \
        "$BENCHMARK_CACHE_ROOT/xdg" \
        "$BENCHMARK_CACHE_ROOT/pip" \
        "$BENCHMARK_CACHE_ROOT/tmp"
    export HF_HOME="$BENCHMARK_CACHE_ROOT/huggingface"
    export HF_HUB_CACHE="$HF_HOME/hub"
    export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
    export TORCH_HOME="$BENCHMARK_CACHE_ROOT/torch"
    export UV_CACHE_DIR="$BENCHMARK_CACHE_ROOT/uv"
    export XDG_CACHE_HOME="$BENCHMARK_CACHE_ROOT/xdg"
    export PIP_CACHE_DIR="$BENCHMARK_CACHE_ROOT/pip"
    export TMPDIR="$BENCHMARK_CACHE_ROOT/tmp"
    export TMP="$TMPDIR"
    export TEMP="$TMPDIR"
fi

if is_cpu_device; then
    export CUDA_VISIBLE_DEVICES=""
    export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:CUDA initialization:UserWarning}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CPU_THREADS}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$CPU_THREADS}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$CPU_THREADS}"
    export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$CPU_THREADS}"
    export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-$CPU_THREADS}"
    export TORCH_NUM_INTEROP_THREADS="${TORCH_NUM_INTEROP_THREADS:-1}"
    export CPU_RAM_SAFETY_FRACTION
    export CPU_RAM_ESTIMATE_HEADROOM
elif ! is_jetson_device; then
    # Keep PyTorch logical device 0 and NVML/nvidia-smi on the same physical GPU.
    # This keeps a single selected card visible on multi-GPU cloud nodes.
    export CUDA_VISIBLE_DEVICES="$BENCHMARK_GPU_INDEX"
fi

TOTAL=${#MODELS[@]}
INTERACTIVE_OUTPUT=false
if [ -t 1 ] && [ "${TERM:-dumb}" != "dumb" ]; then
    INTERACTIVE_OUTPUT=true
fi

refresh_terminal_geometry() {
    TERM_COLS=$(tput cols 2>/dev/null || printf '120')
    if ! [[ "$TERM_COLS" =~ ^[0-9]+$ ]] || [ "$TERM_COLS" -lt 20 ]; then
        TERM_COLS=120
    fi
    BAR_WIDTH=$((TERM_COLS - 22))
    if [ "$BAR_WIDTH" -gt 40 ]; then BAR_WIDTH=40; fi
    if [ "$BAR_WIDTH" -lt 10 ]; then BAR_WIDTH=10; fi
}
refresh_terminal_geometry

if [ "$INTERACTIVE_OUTPUT" = "true" ]; then
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    YELLOW='\033[0;33m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    GREEN=''
    RED=''
    YELLOW=''
    CYAN=''
    BOLD=''
    NC=''
fi

SPINNER_FRAMES=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')
SPINNER_PID=""
SUDO_KEEPALIVE_PID=""
SWAP_DISABLED_FOR_RUN=false

SCRIPT_START=$(date +%s)

RAW_PERFORMANCE_ARGS=(--no-save-raw-performance-metrics)
if [ "$SAVE_RAW_PERFORMANCE_METRICS" = "true" ]; then
    RAW_PERFORMANCE_ARGS=(--save-raw-performance-metrics)
fi

POWER_MODE_ARGS=(--power-mode-switching)
if [ "$POWER_MODE_SWITCHING" != "true" ]; then
    POWER_MODE_ARGS=(--no-power-mode-switching)
fi

DATA_ARGS=(
    --data-dir "$DATA_DIR"
    --data-cache-dir "$DATA_CACHE_DIR"
    --dataset-folder "$DATASET_FOLDER"
    --pollutant "$POLLUTANT"
    --source-pattern "$SOURCE_PATTERN"
)
if [ -n "$EXCLUDE_SOURCE" ]; then
    DATA_ARGS+=(--exclude-source "$EXCLUDE_SOURCE")
fi
if [ -n "$MAX_SOURCES" ]; then
    DATA_ARGS+=(--max-sources "$MAX_SOURCES")
fi
if [ -n "$MAX_WINDOWS" ]; then
    DATA_ARGS+=(--max-windows "$MAX_WINDOWS")
fi
if [ -n "$TARGET_COLUMN" ]; then
    DATA_ARGS+=(--target-column "$TARGET_COLUMN")
fi
if [ -n "$TIME_COLUMN" ]; then
    DATA_ARGS+=(--time-column "$TIME_COLUMN")
fi
if [ -n "$WINDOW_STRIDE" ]; then
    DATA_ARGS+=(--window-stride "$WINDOW_STRIDE")
fi

draw_progress_bar() {
    local current=$1
    local total=$2
    if [ "$INTERACTIVE_OUTPUT" != "true" ]; then
        printf "  Progress: %d/%d\n" "$current" "$total"
        return
    fi
    refresh_terminal_geometry
    local pct=$((current * 100 / total))
    local filled=$((current * BAR_WIDTH / total))
    local empty=$((BAR_WIDTH - filled))

    printf "\r  ["
    printf "${GREEN}"
    for ((i=0; i<filled; i++)); do printf "█"; done
    printf "${NC}"
    for ((i=0; i<empty; i++)); do printf "░"; done
    printf "] ${BOLD}%3d%%${NC} (%d/%d)" "$pct" "$current" "$total"
}

draw_mini_progress_bar() {
    local current=$1
    local total=$2
    local width=${3:-20}
    local pct=$((current * 100 / total))
    local filled=$((current * width / total))
    local empty=$((width - filled))

    printf "["
    printf "${CYAN}"
    for ((i=0; i<filled; i++)); do printf "="; done
    printf "${NC}"
    for ((i=0; i<empty; i++)); do printf "-"; done
    printf "] %3d%%" "$pct"
}

estimate_time_remaining() {
    local elapsed=$1
    local current=$2
    local total=$3
    
    if [ "$current" -eq 0 ] || [ "$current" -eq "$total" ]; then
        echo "-- --"
        return
    fi
    
    local rate=$((elapsed / current))
    local remaining=$((rate * (total - current)))
    format_duration "$remaining"
}

start_spinner() {
    local msg="$1"
    if [ "$INTERACTIVE_OUTPUT" != "true" ]; then
        printf "    %s\n" "$msg"
        return
    fi
    (
        local frame=0
        while true; do
            local spin_char="${SPINNER_FRAMES[$((frame % ${#SPINNER_FRAMES[@]}))]}"
            printf "\r    ${CYAN}%s${NC} %s" "$spin_char" "$msg"
            frame=$((frame + 1))
            sleep 0.15
        done
    ) &
    SPINNER_PID=$!
    tput civis 2>/dev/null || true
}

stop_spinner() {
    if [ "$INTERACTIVE_OUTPUT" != "true" ]; then
        return
    fi
    if [ -n "$SPINNER_PID" ] && kill -0 "$SPINNER_PID" 2>/dev/null; then
        kill "$SPINNER_PID" 2>/dev/null
        wait "$SPINNER_PID" 2>/dev/null || true
    fi
    SPINNER_PID=""
    tput cnorm 2>/dev/null || true
    printf "\r%-90s\r" ""
}

should_start_sudo_keepalive() {
    local normalized_device
    normalized_device=$(printf "%s" "$DEVICE" | tr '[:upper:]' '[:lower:]')

    if [ "$SUDO_KEEPALIVE" = "true" ]; then
        return 0
    fi
    if [ "$SUDO_KEEPALIVE" = "false" ]; then
        return 1
    fi

    # Auto mode: only Jetson telemetry uses sudo in this benchmark path.
    [ "$normalized_device" = "jetson" ]
}

check_jetson_torch() {
    local venv_python="$1"

    "$venv_python" - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    print(f"torch import failed: {exc}", file=sys.stderr)
    sys.exit(1)

version = getattr(torch, "__version__", "")
if "+cu132" not in version:
    print(f"torch {version!r} is not a Jetson CUDA 13.2 build", file=sys.stderr)
    sys.exit(2)
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is false", file=sys.stderr)
    sys.exit(3)

try:
    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    x = torch.ones((16, 16), device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    if float(y[0, 0].item()) != 16.0:
        raise RuntimeError("CUDA matmul returned an unexpected value")
except Exception as exc:
    print(f"Jetson CUDA kernel smoke failed: {exc}", file=sys.stderr)
    sys.exit(4)

print(f"torch {version} OK on {device_name} sm_{capability[0]}{capability[1]}")
PY
}

check_jetson_python_abi() {
    local venv_python="$1"

    PYTHONPATH="$PWD" "$venv_python" - <<'PY'
import sys

try:
    import numpy
    import pandas
    from benchmark.runner import BenchmarkRunner  # noqa: F401
except Exception as exc:
    print(f"Python ABI/import smoke failed: {exc}", file=sys.stderr)
    sys.exit(1)

print(f"numpy {numpy.__version__}, pandas {pandas.__version__}, benchmark import OK")
PY
}

check_discrete_cuda() {
    local venv_python="$1"

    "$venv_python" - "$(normalized_device)" <<'PY'
import sys

import torch

category = sys.argv[1]
if not torch.cuda.is_available():
    raise SystemExit(
        f"CUDA unavailable: torch={torch.__version__}, "
        f"wheel_cuda={torch.version.cuda}. Install a wheel compatible with the host driver."
    )

name = torch.cuda.get_device_name(0)
expected_tokens = {
    "h100": "H100",
    "h200": "H200",
    "a6000": "A6000",
    "5090": "5090",
}
expected = expected_tokens[category]
if expected not in name.upper():
    raise SystemExit(
        f"Selected CUDA device {name!r} does not match category {category!r}."
    )

x = torch.ones((16, 16), device="cuda")
y = x @ x
torch.cuda.synchronize()
if float(y[0, 0].item()) != 16.0:
    raise SystemExit("CUDA matrix-multiplication smoke returned an invalid value.")

capability = torch.cuda.get_device_capability(0)
print(
    f"CUDA runtime OK: torch={torch.__version__}, wheel_cuda={torch.version.cuda}, "
    f"device={name}, sm_{capability[0]}{capability[1]}"
)
PY
}

check_model_import() {
    local model="$1"
    local venv_python="$2"

    PYTHONPATH="$PWD" "$venv_python" - "$model" <<'PY'
import importlib
import os
import sys

cpu_execution = os.environ.get("DEVICE", "").lower() in {
    "cpu", "rpi5", "rpi5_hailo26", "uno_q"
}
modules = {
    "units": "models.units",
    "ttm": "models.ttm",
    "ttm_r3_auto": "models.ttm_r3_auto",
    "ttm_r2": "models.ttm_r2",
    "ttm_r3_512_96_dec": "models.ttm_r3_512_96_dec",
    "ttm_r3_512_96_dec_lite": "models.ttm_r3_512_96_dec_lite",
    "ttm_r3_1024_720": "models.ttm_r3_1024_720",
    "ttm_r3_auto_lite": "models.ttm_r3_auto_lite",
    "flowstate": "models.flowstate",
    "chronos2": "models.chronos2",
    "chronos_bolt_base": "models.chronos_bolt_base",
    "moirai2": "models.moirai_2p0",
    "kairos": "models.kairos",
    "kairos_50m": "models.kairos_50m",
    "sundial": "models.sundial",
    "moment": "models.moment",
    "tempopfn": "models.tempopfn_cpu" if cpu_execution else "models.tempopfn",
    "tirex": "models.tirex_cpu" if cpu_execution else "models.tirex",
    "toto": "models.toto",
    "toto_2p0_4m": "models.toto_2p0_4m",
    "toto_2p0_2p5b": "models.toto_2p0_2p5b",
    "timesfm_2p5": "models.timesfm_2p5",
}
model = sys.argv[1]
module = modules.get(model)
if module is None:
    raise SystemExit(f"No import smoke mapping for model {model!r}")
importlib.import_module(module)
print(f"{module} import OK")
PY
}

check_model_runtime() {
    local model="$1"
    local venv_dir="$2"
    local env_log="$3"
    local venv_python="$venv_dir/bin/python"

    ensure_jetson_torch "$model" "$venv_dir" "$env_log" || return 1
    if is_discrete_nvidia_device; then
        check_discrete_cuda "$venv_python" >> "$env_log" 2>&1 || return 1
    fi
    check_model_import "$model" "$venv_python" >> "$env_log" 2>&1
}

repair_jetson_model_abi() {
    local model="$1"
    local venv_python="$2"
    local env_log="$3"

    case "$model" in
        moirai2)
            echo "[JETSON] Restoring Moirai NumPy/Pandas ABI compatibility." >> "$env_log"
            uv pip install --python "$venv_python" --reinstall "numpy~=1.26.0" "pandas==2.1.4" >> "$env_log" 2>&1
            ;;
        *)
            return 1
            ;;
    esac
}

ensure_jetson_torch() {
    local model="$1"
    local venv_dir="$2"
    local env_log="$3"
    local venv_python="$venv_dir/bin/python"
    JETSON_TORCH_RESULT="unchanged"

    if ! is_jetson_device; then
        return 0
    fi
    if [ ! -x "$venv_python" ]; then
        return 1
    fi

    if check_jetson_torch "$venv_python" >> "$env_log" 2>&1 \
        && check_jetson_python_abi "$venv_python" >> "$env_log" 2>&1; then
        return 0
    fi

    JETSON_TORCH_RESULT="repaired"
    {
        echo "[JETSON] Repairing $model torch stack with: $JETSON_TORCH_INDEX_URL"
        echo "[JETSON] Packages: $JETSON_TORCH_PACKAGES"
    } >> "$env_log"

    # shellcheck disable=SC2086
    if uv pip install --python "$venv_python" --pre --reinstall $JETSON_TORCH_PACKAGES --index-url "$JETSON_TORCH_INDEX_URL" >> "$env_log" 2>&1 \
        && { check_jetson_python_abi "$venv_python" >> "$env_log" 2>&1 || repair_jetson_model_abi "$model" "$venv_python" "$env_log"; } \
        && check_jetson_torch "$venv_python" >> "$env_log" 2>&1 \
        && check_jetson_python_abi "$venv_python" >> "$env_log" 2>&1; then
        return 0
    fi

    return 1
}

start_sudo_keepalive() {
    if ! should_start_sudo_keepalive; then
        return
    fi
    if ! command -v sudo >/dev/null 2>&1; then
        echo -e "${YELLOW}[SUDO] sudo is not installed; continuing without keepalive.${NC}"
        return
    fi

    if ! sudo -n true 2>/dev/null; then
        echo -e "${YELLOW}[SUDO] Benchmark telemetry may need sudo. Please enter it once; the script will keep the session alive.${NC}"
        sudo -v
    fi

    (
        while true; do
            sudo -n true 2>/dev/null || exit 0
            sleep 60
        done
    ) &
    SUDO_KEEPALIVE_PID=$!
}

prepare_jetson_runtime() {
    if ! is_jetson_device; then
        return
    fi

    local requested_mode="${POWER_MODES%%,*}"
    requested_mode="${requested_mode%% *}"
    if ! [[ "$requested_mode" =~ ^[0-9]+$ ]]; then
        echo -e "${RED}[JETSON] Invalid first POWER_MODES value: $requested_mode${NC}"
        exit 1
    fi

    echo -e "${CYAN}[JETSON] Preparing power mode $requested_mode, maximum clocks, and maximum fan before workload execution.${NC}"
    if ! sudo -n nvpmodel -m "$requested_mode"; then
        echo -e "${RED}[JETSON] Failed to apply nvpmodel mode $requested_mode.${NC}"
        exit 1
    fi
    if ! sudo -n jetson_clocks; then
        echo -e "${RED}[JETSON] Failed to lock Jetson clocks.${NC}"
        exit 1
    fi
    sudo -n jetson_clocks --fan >/dev/null 2>&1 || true

    local mode_output effective_mode clock_output
    mode_output=$(sudo -n nvpmodel -q 2>&1)
    effective_mode=$(printf '%s\n' "$mode_output" | awk '/^[0-9]+$/{print; exit}')
    clock_output=$(sudo -n jetson_clocks --show 2>&1)
    if [ "$effective_mode" != "$requested_mode" ]; then
        echo -e "${RED}[JETSON] Power-mode verification failed: requested=$requested_mode effective=${effective_mode:-unknown}.${NC}"
        exit 1
    fi
    if ! printf '%s\n' "$clock_output" | grep -Eq '^FAN .*pwm1=255'; then
        echo -e "${RED}[JETSON] Maximum fan verification failed.${NC}"
        exit 1
    fi

    echo -e "${GREEN}[JETSON] Runtime ready: $(printf '%s\n' "$mode_output" | head -1), clocks locked, fan PWM 255.${NC}"
    printf '%s\n' "$clock_output" \
        | grep -E '^(Online CPUs:|cpu0:|GPU MinFreq=|EMC MinFreq=|FAN |NV Power Mode:)' \
        | sed 's/^/[JETSON] /'
}

configure_open_file_limit() {
    local current_limit
    current_limit=$(ulimit -n)

    if [ "$current_limit" = "unlimited" ]; then
        echo -e "${CYAN}[LIMITS] Open-file limit is unlimited.${NC}"
        return
    fi

    if [ "$current_limit" -ge "$OPEN_FILES_LIMIT" ]; then
        echo -e "${CYAN}[LIMITS] Open-file limit: $current_limit${NC}"
        return
    fi

    if ulimit -n "$OPEN_FILES_LIMIT" 2>/dev/null; then
        echo -e "${CYAN}[LIMITS] Raised open-file limit: $current_limit -> $(ulimit -n)${NC}"
    elif command -v prlimit >/dev/null 2>&1 \
        && command -v sudo >/dev/null 2>&1 \
        && sudo -n prlimit --pid "$$" \
            --nofile="${OPEN_FILES_LIMIT}:${OPEN_FILES_LIMIT}" >/dev/null 2>&1; then
        echo -e "${CYAN}[LIMITS] Privileged open-file limit: $current_limit -> $(ulimit -n)${NC}"
    elif [ "$NUM_WORKERS" -eq 0 ]; then
        echo -e "${CYAN}[LIMITS] Open-file limit remains $current_limit; sufficient for the sequential in-memory loader.${NC}"
    else
        echo -e "${YELLOW}[LIMITS] Could not raise open-file limit from $current_limit to $OPEN_FILES_LIMIT. DataLoader workers may fail; try: ulimit -n $OPEN_FILES_LIMIT${NC}"
    fi
}

configure_cpu_benchmark_permissions() {
    local normalized_device
    normalized_device=$(printf "%s" "$DEVICE" | tr '[:upper:]' '[:lower:]')
    if ! is_cpu_device; then
        return
    fi
    if ! command -v sudo >/dev/null 2>&1; then
        echo -e "${YELLOW}[CPU] sudo is not installed; CPU power/governor telemetry may be unavailable.${NC}"
        return
    fi
    if ! sudo -n true 2>/dev/null; then
        echo -e "${YELLOW}[CPU] sudo is not cached; power/governor telemetry may be unavailable.${NC}"
        return
    fi

    local telemetry_paths=()
    local path
    if [ "$normalized_device" = "cpu" ]; then
        for path in /sys/class/powercap/intel-rapl:*/energy_uj /sys/class/powercap/intel-rapl:*:*/energy_uj; do
            [ -e "$path" ] && telemetry_paths+=("$path")
        done
    else
        for path in /sys/class/hwmon/hwmon*/power*_input; do
            [ -e "$path" ] && telemetry_paths+=("$path")
        done
    fi
    if [ "${#telemetry_paths[@]}" -gt 0 ]; then
        if sudo -n chmod a+r "${telemetry_paths[@]}" 2>/dev/null; then
            echo -e "${CYAN}[POWER] Telemetry sensor files made readable for this run.${NC}"
        else
            echo -e "${YELLOW}[POWER] Could not make telemetry sensors readable; power metrics may be NaN.${NC}"
        fi
    else
        echo -e "${YELLOW}[POWER] No onboard power sensor found; configure POWER_SENSOR_PATH for board-level telemetry.${NC}"
    fi

    local governor_paths=(/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor)
    if sudo -n chmod a+rw "${governor_paths[@]}" 2>/dev/null; then
        echo -e "${CYAN}[CPU] CPU governor controls made writable for this run.${NC}"
    else
        echo -e "${YELLOW}[CPU] Could not make CPU governor controls writable; frequency lock may be skipped.${NC}"
    fi
}

stop_sudo_keepalive() {
    if [ -n "$SUDO_KEEPALIVE_PID" ] && kill -0 "$SUDO_KEEPALIVE_PID" 2>/dev/null; then
        kill "$SUDO_KEEPALIVE_PID" 2>/dev/null || true
        wait "$SUDO_KEEPALIVE_PID" 2>/dev/null || true
    fi
    SUDO_KEEPALIVE_PID=""
}

prepare_swap_guard() {
    if [ "$DISABLE_SWAP_DURING_BENCHMARK" != "true" ]; then
        return
    fi
    if ! command -v swapon >/dev/null 2>&1; then
        echo -e "${YELLOW}[SWAP] swapon command not found; swap guard skipped.${NC}"
        return
    fi
    if [ -z "$(swapon --noheadings --show=NAME 2>/dev/null)" ]; then
        echo -e "${CYAN}[SWAP] No active swap detected.${NC}"
        return
    fi
    if ! command -v sudo >/dev/null 2>&1 || ! sudo -n true 2>/dev/null; then
        echo -e "${YELLOW}[SWAP] Active swap detected, but sudo is unavailable; results may be marked SWAP_CONTAMINATED if swap grows.${NC}"
        return
    fi

    echo -e "${CYAN}[SWAP] Temporarily disabling swap for cleaner benchmark timing...${NC}"
    if sudo -n swapoff -a; then
        SWAP_DISABLED_FOR_RUN=true
        echo -e "${CYAN}[SWAP] Swap disabled for this benchmark run.${NC}"
    else
        echo -e "${YELLOW}[SWAP] Could not disable swap; results may be marked SWAP_CONTAMINATED if swap grows.${NC}"
    fi
}

restore_swap_guard() {
    if [ "$SWAP_DISABLED_FOR_RUN" = "true" ]; then
        echo -e "${CYAN}[SWAP] Restoring swap...${NC}"
        sudo -n swapon -a 2>/dev/null || echo -e "${YELLOW}[SWAP] Could not restore swap automatically; run sudo swapon -a.${NC}"
        SWAP_DISABLED_FOR_RUN=false
    fi
}

format_duration() {
    local secs=$1
    if [ "$secs" -lt 60 ]; then
        echo "${secs}s"
    elif [ "$secs" -ge 3600 ]; then
        local hours=$((secs / 3600))
        local mins=$(((secs % 3600) / 60))
        local remaining=$((secs % 60))
        echo "${hours}h${mins}m${remaining}s"
    else
        local mins=$((secs / 60))
        local remaining=$((secs % 60))
        echo "${mins}m${remaining}s"
    fi
}

cooldown_countdown() {
    local seconds=${1%.*}
    local label=$2
    if [ "$seconds" -le 0 ]; then
        return
    fi
    if [ "$INTERACTIVE_OUTPUT" != "true" ]; then
        echo "  $label: waiting $(format_duration "$seconds")"
        sleep "$seconds"
        return
    fi
    local remaining=$seconds
    while [ "$remaining" -gt 0 ]; do
        printf "\r  ${CYAN}%-28s${NC} %s remaining" "$label" "$(format_duration "$remaining")"
        local step=10
        if [ "$remaining" -lt "$step" ]; then step=$remaining; fi
        sleep "$step"
        remaining=$((remaining - step))
    done
    printf "\r  ${GREEN}%-28s complete${NC}%30s\n" "$label" ""
}

interrupt_benchmark() {
    trap - INT TERM HUP TSTP
    echo ""
    echo -e "${YELLOW}[INTERRUPT] Stopping the active model process. Re-run with --phase benchmark to resume safely.${NC}"
    local child_pid
    while read -r child_pid; do
        [ -n "$child_pid" ] && kill -TERM "$child_pid" 2>/dev/null || true
    done < <(jobs -pr)
    sleep 1
    while read -r child_pid; do
        [ -n "$child_pid" ] && kill -KILL "$child_pid" 2>/dev/null || true
    done < <(jobs -pr)
    exit 130
}

trap interrupt_benchmark INT TERM HUP TSTP
trap 'stop_spinner; restore_swap_guard; stop_sudo_keepalive; tput cnorm 2>/dev/null || true' EXIT

echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}  Unified TSFM Benchmark Suite${NC}"
echo -e "${BOLD}============================================================${NC}"
echo -e "  Device:       ${CYAN}$DEVICE${NC}"
echo -e "  Run Phase:    ${CYAN}$RUN_PHASE${NC}"
if [ -n "$CLOUD_PROVIDER" ]; then
    echo -e "  Cloud:        ${CYAN}$CLOUD_PROVIDER${NC} ${CLOUD_INSTANCE_TYPE:-unknown-type} ${CLOUD_REGION:-unknown-region}"
fi
case "$(normalized_device)" in
    rpi5_hailo26|uno_q)
        echo -e "  Backend:      ${YELLOW}PyTorch CPU${NC} (vendor NPU adapter required for accelerator results)"
        ;;
esac
echo -e "  Data Dir:     ${CYAN}$DATA_DIR${NC}"
echo -e "  Dataset:      ${CYAN}$DATASET_FOLDER${NC}"
echo -e "  Pollutant:    ${CYAN}$POLLUTANT${NC}"
echo -e "  Data Cache:   ${CYAN}$DATA_CACHE_DIR${NC}"
if [ -n "$BENCHMARK_CACHE_ROOT" ]; then
    echo -e "  System Cache: ${CYAN}$BENCHMARK_CACHE_ROOT${NC}"
    echo -e "  Temp Dir:     ${CYAN}$TMPDIR${NC}"
    CACHE_FILESYSTEM=$(findmnt -T "$BENCHMARK_CACHE_ROOT" -n -o SOURCE,FSTYPE 2>/dev/null || echo "unknown")
    echo -e "  Cache Device: ${CYAN}$CACHE_FILESYSTEM${NC}"
fi
echo -e "  Source Glob:  ${CYAN}$SOURCE_PATTERN${NC}"
if [ -n "$EXCLUDE_SOURCE" ]; then
    echo -e "  Excluded:     ${CYAN}$EXCLUDE_SOURCE${NC}"
fi
echo -e "  Models:       ${BOLD}$TOTAL selected${NC} (listed in pre-flight below)"
echo -e "  TTM Mapping:  ${CYAN}prefer_known_mappings=$TTM_PREFER_KNOWN_MAPPINGS${NC}"
echo -e "  Power Modes:  ${CYAN}$POWER_MODES${NC}"
echo -e "  Mode Switch:  ${CYAN}$POWER_MODE_SWITCHING${NC}"
echo -e "  Mode Settle:  ${CYAN}$POWER_MODE_STABILIZATION_SEC s${NC}"
echo -e "  Workers:      ${CYAN}$NUM_WORKERS${NC}"
if is_cpu_device; then
    echo -e "  CPU Threads:  ${CYAN}$CPU_THREADS${NC}"
    echo -e "  CPU RAM Guard:${CYAN} ${CPU_RAM_SAFETY_FRACTION} safety, ${CPU_RAM_ESTIMATE_HEADROOM} headroom${NC}"
    if [ "$(normalized_device)" = "cpu" ]; then
        echo -e "  CPU Timeout:  ${CYAN}$(format_duration "$CONFIG_TIMEOUT_SEC") per prediction model and batch-size configuration${NC}"
    fi
fi
echo -e "  Max Sources:  ${CYAN}${MAX_SOURCES:-all}${NC}"
echo -e "  Max Windows:  ${CYAN}${MAX_WINDOWS:-all per source}${NC}"
echo -e "  Prediction:   ${CYAN}batch size $PREDICTION_BATCH_SIZE, metrics disabled${NC}"
echo -e "  Benchmark:    ${CYAN}$BENCHMARK_BATCHES preloaded batches, $WARMUPS warmups, $REPEATS repeats${NC}"
echo -e "  Seed:         ${CYAN}$BENCHMARK_SEED${NC} (Python, NumPy, PyTorch, CUDA, loaders)"
echo -e "  Seasonality:  ${CYAN}$ACCURACY_SEASONALITY${NC} (context-derived MASE/RMSSE scales)"
echo -e "  Raw Rows:     ${CYAN}$SAVE_RAW_PERFORMANCE_METRICS${NC}"
echo -e "  Auto Resume:  ${CYAN}$AUTO_RESUME${NC} (prediction/benchmark model boundary)"
echo -e "  Supplemental: ${CYAN}$SUPPLEMENTAL_BATCHES${NC} (append only requested Phase-2 batch sizes)"
echo -e "  Cooling:      ${CYAN}${PHASE_COOLDOWN_SEC}s phase, ${SLEEP_BETWEEN_BATCHES_SEC}s batch, ${MODEL_COOLDOWN_SEC}s model${NC}"
echo -e "  Idle Limits:  ${CYAN}GPU ${MAX_IDLE_GPU_UTIL}%, CPU ${MAX_IDLE_CPU_UTIL}%${NC}"
if ! is_cpu_device && ! is_jetson_device; then
    echo -e "  GPU Index:    ${CYAN}$BENCHMARK_GPU_INDEX${NC} (physical NVML index; CUDA logical index 0)"
    echo -e "  GPU Clocks:   ${CYAN}${GPU_LOCK_CLOCKS}${NC} (lock maximum supported graphics clock)"
fi
echo -e "  Bar Refresh:  ${CYAN}$TQDM_MININTERVAL s${NC}"
echo -e "  Sudo Keepalive: ${CYAN}$SUDO_KEEPALIVE${NC}"
echo -e "  Swap Guard:   ${CYAN}$DISABLE_SWAP_DURING_BENCHMARK${NC}"
if is_jetson_device; then
    echo -e "  Jetson Torch: ${CYAN}$JETSON_TORCH_INDEX_URL${NC}"
fi
echo -e "  Open Files:   ${CYAN}$OPEN_FILES_LIMIT${NC}"
echo -e "  Output Dir:   ${CYAN}$OUTPUT_DIR${NC}"
echo -e "  Model Logs:   ${CYAN}$MODEL_LOG_DIR${NC}"
echo -e "${BOLD}============================================================${NC}"

if [ "$CHECK_CONFIG" = "true" ]; then
    echo -e "${GREEN}[CHECK] Device-aware launcher configuration is valid.${NC}"
    exit 0
fi

if [ "$OUTPUT_DIR" != "$BASE_OUTPUT_DIR" ]; then
    echo -e "${YELLOW}[RESULTS] Existing files found in ${BASE_OUTPUT_DIR}; writing this run to ${OUTPUT_DIR} to avoid mixing stale and fresh benchmarks.${NC}"
fi

# Verify prerequisites.
if ! command -v uv &> /dev/null; then
    echo "[ERROR] 'uv' package manager is not installed."
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "[ERROR] Dataset directory $DATA_DIR does not exist."
    exit 1
fi
SOURCE_COUNT=$(python3 - "$DATA_DIR" "$SOURCE_PATTERN" "$EXCLUDE_SOURCE" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
pattern = sys.argv[2]
excluded = sys.argv[3]
count = 0
for path in root.glob(pattern):
    if not path.is_file() or (excluded and path.name == excluded):
        continue
    relative = path.relative_to(root)
    if any(part.startswith(".") for part in relative.parts):
        continue
    count += 1
print(count)
PY
)
if [ "$SOURCE_COUNT" -eq 0 ]; then
    echo "[ERROR] No source files matched $DATA_DIR/$SOURCE_PATTERN."
    exit 1
fi
echo "[DATA] Using $SOURCE_COUNT independent source file(s) from $DATA_DIR"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$MODEL_LOG_DIR"

start_sudo_keepalive
configure_open_file_limit
prepare_jetson_runtime
prepare_swap_guard
configure_cpu_benchmark_permissions

# Prepare isolated model environments.
echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}  Pre-Flight Environment Check${NC}"
echo -e "${BOLD}============================================================${NC}"

PREFLIGHT_PASS=0
PREFLIGHT_FAIL=0
PREFLIGHT_FAILED=()

for MODEL in "${MODELS[@]}"; do
    MODEL_ENV_NAME=$(model_environment_name "$MODEL")
    VENV_DIR=".venv_${MODEL_ENV_NAME}"
    SETUP_SCRIPT="scripts/${MODEL_ENV_NAME}.sh"
    ENV_LOG="$MODEL_LOG_DIR/${MODEL}_env_setup.log"

    printf "  ${BOLD}%-12s${NC} " "$MODEL"

    ENV_CREATED=false
    if [ ! -d "$VENV_DIR" ]; then
        if [ ! -f "$SETUP_SCRIPT" ]; then
            echo -e "${RED}FAIL${NC}  (no setup script)"
            PREFLIGHT_FAIL=$((PREFLIGHT_FAIL + 1))
            PREFLIGHT_FAILED+=("$MODEL")
            continue
        fi

        start_spinner "Creating environment..."
        if bash "$SETUP_SCRIPT" > "$ENV_LOG" 2>&1; then
            stop_spinner
            printf "  ${BOLD}%-12s${NC} " "$MODEL"
            ENV_CREATED=true
        else
            stop_spinner
            printf "  ${BOLD}%-12s${NC} " "$MODEL"
            echo -e "${RED}FAIL${NC}  (env creation failed — see $ENV_LOG)"
            PREFLIGHT_FAIL=$((PREFLIGHT_FAIL + 1))
            PREFLIGHT_FAILED+=("$MODEL")
            continue
        fi
    fi

    if check_model_runtime "$MODEL" "$VENV_DIR" "$ENV_LOG"; then
        if [ "$ENV_CREATED" = "true" ]; then
            echo -e "${GREEN}READY${NC} (created, import verified)"
        elif [ "${JETSON_TORCH_RESULT:-unchanged}" = "repaired" ]; then
            echo -e "${GREEN}READY${NC} (repaired, import verified)"
        else
            echo -e "${GREEN}READY${NC} (existing, import verified)"
        fi
        PREFLIGHT_PASS=$((PREFLIGHT_PASS + 1))
        continue
    fi

    if [ "$ENV_CREATED" = "false" ] && [ -f "$SETUP_SCRIPT" ]; then
        echo -e "${YELLOW}REPAIR${NC} (model import failed; rebuilding environment)"
        if bash "$SETUP_SCRIPT" > "$ENV_LOG" 2>&1 \
            && check_model_runtime "$MODEL" "$VENV_DIR" "$ENV_LOG"; then
            printf "  ${BOLD}%-12s${NC} " "$MODEL"
            echo -e "${GREEN}READY${NC} (repaired, import verified)"
            PREFLIGHT_PASS=$((PREFLIGHT_PASS + 1))
            continue
        fi
    fi

    printf "  ${BOLD}%-12s${NC} " "$MODEL"
    echo -e "${RED}FAIL${NC} (model import validation failed; see $ENV_LOG)"
    PREFLIGHT_FAIL=$((PREFLIGHT_FAIL + 1))
    PREFLIGHT_FAILED+=("$MODEL")
done

echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}  Pre-Flight Summary${NC}"
echo -e "${BOLD}============================================================${NC}"
echo -e "  Ready: ${GREEN}$PREFLIGHT_PASS${NC}"
echo -e "  Failed: ${RED}$PREFLIGHT_FAIL${NC}"
if [ $PREFLIGHT_FAIL -gt 0 ]; then
    echo -e "  Failed Models: ${RED}${PREFLIGHT_FAILED[*]}${NC}"
    echo "Fix these before running the full benchmark."
    exit 1
fi
echo ""
echo -e "${BOLD}All model environments are ready. Starting full benchmark...${NC}"
echo ""

METADATA_DIR="$OUTPUT_DIR/benchmark_metadata"
ENVIRONMENT_DIR="$METADATA_DIR/environments"
mkdir -p "$ENVIRONMENT_DIR"

has_prediction_state() {
    [ -d "$OUTPUT_DIR/benchmark_predictions" ] \
        || compgen -G "$OUTPUT_DIR/benchmark_metadata/*_prediction.json" >/dev/null \
        || compgen -G "$OUTPUT_DIR/model_logs/*_prediction.log" >/dev/null \
        || compgen -G "$OUTPUT_DIR/.prediction_work/*" >/dev/null
}

for MODEL in "${MODELS[@]}"; do
    MODEL_ENV_NAME=$(model_environment_name "$MODEL")
    VENV_PYTHON=".venv_${MODEL_ENV_NAME}/bin/python"
    uv pip freeze --python "$VENV_PYTHON" > "$ENVIRONMENT_DIR/${MODEL}_packages.txt"
    "$VENV_PYTHON" -VV > "$ENVIRONMENT_DIR/${MODEL}_python.txt" 2>&1
done
{
    printf '%s\n' \
        run_unified_benchmark.sh run_horizon_sweep.sh \
        run_ttm_r3_variant_sweep.sh smoke_test.sh pyproject.toml
    find analysis benchmark experiments models powerbench scripts tests -type f \
        \( -name '*.py' -o -name '*.sh' \) -print
    find utils -type f ! -path '*/__pycache__/*' ! -name '*.pyc' -print
} | sort -u | while IFS= read -r source_file; do
    [ -f "$source_file" ] && sha256sum "$source_file"
done > "$METADATA_DIR/source_files.sha256"
echo -e "${CYAN}[REPRODUCIBILITY] Saved package and source manifests under $METADATA_DIR.${NC}"

# Generate forecasts before performance measurement.
MODEL_CSV=$(IFS=,; echo "${MODELS[*]}")
VALIDATOR_ENV_NAME=$(model_environment_name "${MODELS[0]}")
VALIDATOR_PYTHON=".venv_${VALIDATOR_ENV_NAME}/bin/python"
if [ "$RUN_PHASE" != "benchmark" ]; then
PREDICTION_RESUME_START_INDEX=0
if [ "$RUN_PHASE" != "benchmark" ] && has_prediction_state; then
    if [ "$AUTO_RESUME" != "true" ]; then
        echo -e "${RED}[STOP] Existing prediction state found in $OUTPUT_DIR.${NC}"
        echo "Set AUTO_RESUME=true to continue or use a different OUTPUT_DIR for a fresh run."
        exit 1
    fi
    PREDICTION_RESUME_START_INDEX=$("$VALIDATOR_PYTHON" scripts/prediction_resume.py \
        --output-dir "$OUTPUT_DIR" \
        --models "$MODEL_CSV" \
        --device "$DEVICE" \
        --seed "$BENCHMARK_SEED" \
        --dataset "$DATASET_FOLDER" \
        --context-lens "$CTX_LENS" \
        --horizon-lens "$HORIZON_LENS" \
        --prediction-batch-size "$PREDICTION_BATCH_SIZE" \
        --prepare)
    if ! [[ "$PREDICTION_RESUME_START_INDEX" =~ ^[0-9]+$ ]] \
        || [ "$PREDICTION_RESUME_START_INDEX" -gt "$TOTAL" ]; then
        echo -e "${RED}[STOP] Prediction resume planner returned an invalid model index.${NC}"
        exit 1
    fi
fi
PREDICTION_TIMEOUT_PREFIX=()
if [ "$(normalized_device)" = "cpu" ] && [ "$CPU_CONFIG_TIMEOUT_SEC" -gt 0 ] && command -v timeout >/dev/null 2>&1; then
    PREDICTION_TIMEOUT_PREFIX=(
        timeout --foreground --signal=TERM --kill-after=60s "${CPU_CONFIG_TIMEOUT_SEC}s"
    )
fi
BASELINE_COMPUTE_PIDS=""
if command -v nvidia-smi >/dev/null 2>&1; then
    BASELINE_COMPUTE_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk '/^[[:space:]]*[0-9]+[[:space:]]*$/{gsub(/[[:space:]]/, ""); print}' | sort -u || true)
fi
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}  Phase 1/2: Prediction Generation${NC}"
echo -e "${BOLD}============================================================${NC}"
PREDICTION_FAILURES=()
for i in "${!MODELS[@]}"; do
    if [ "$i" -lt "$PREDICTION_RESUME_START_INDEX" ]; then
        continue
    fi
    MODEL="${MODELS[$i]}"
    NUM=$((i + 1))
    MODEL_ENV_NAME=$(model_environment_name "$MODEL")
    VENV_PYTHON=".venv_${MODEL_ENV_NAME}/bin/python"
    MODEL_LOG="$MODEL_LOG_DIR/${MODEL}_prediction.log"
    printf "  ${BOLD}[%2d/%d] %-12s${NC} bs=%s | predictions only\n" \
        "$NUM" "$TOTAL" "$MODEL" "$PREDICTION_BATCH_SIZE"
    set +e
    PYTHONFAULTHANDLER=1 \
    VIRTUAL_ENV="$(pwd)/.venv_${MODEL_ENV_NAME}" \
    PATH="$(pwd)/.venv_${MODEL_ENV_NAME}/bin:$BASE_PATH" \
    "${PREDICTION_TIMEOUT_PREFIX[@]}" \
    "$VENV_PYTHON" -u -m experiments.forecasting \
        "${DATA_ARGS[@]}" \
        --phase prediction \
        --models "$MODEL" \
        --context-lens "$CTX_LENS" \
        --horizon-lens "$HORIZON_LENS" \
        --batch-sizes "$PREDICTION_BATCH_SIZE" \
        --prediction-batch-size "$PREDICTION_BATCH_SIZE" \
        --seed "$BENCHMARK_SEED" \
        --accuracy-seasonality "$ACCURACY_SEASONALITY" \
        --output-dir "$OUTPUT_DIR" \
        --device "$DEVICE" \
        --num-workers "$NUM_WORKERS" \
        --log-file "$MODEL_LOG" \
        --no-power-mode-switching
    PREDICTION_EXIT=$?
    set -e
    if [ "$PREDICTION_EXIT" -eq 0 ]; then
        echo -e "           ${GREEN}✓ predictions saved${NC}"
    else
        rm -rf "$OUTPUT_DIR/benchmark_predictions/$MODEL"
        rm -f "$OUTPUT_DIR/.prediction_work/${MODEL}_ctx_"*.float32
        rm -f "$OUTPUT_DIR/benchmark_metadata/${MODEL}_prediction.json"
        if [ "$PREDICTION_EXIT" -eq 124 ] || [ "$PREDICTION_EXIT" -eq 137 ]; then
            echo -e "           ${YELLOW}TIMEOUT${NC} after $(format_duration "$CPU_CONFIG_TIMEOUT_SEC"); partial predictions removed"
        fi
        echo -e "           ${RED}✗ failed${NC} | log: $MODEL_LOG"
        PREDICTION_FAILURES+=("$MODEL")
        echo -e "           ${YELLOW}Stopping at the first failed model; rerun --phase prediction to resume here.${NC}"
        break
    fi
done

if [ "${#PREDICTION_FAILURES[@]}" -gt 0 ]; then
    echo -e "${RED}[STOP] Prediction failures: ${PREDICTION_FAILURES[*]}. Benchmark phase not started.${NC}"
    exit 1
fi
HF_CACHE_ROOT="${HF_HOME:-$HOME/.cache/huggingface}/hub"
if [ -d "$HF_CACHE_ROOT" ]; then
    while IFS= read -r ref_file; do
        printf '%s\t%s\n' "${ref_file#"$HF_CACHE_ROOT"/}" "$(cat "$ref_file")"
    done < <(find "$HF_CACHE_ROOT" -path '*/refs/*' -type f | sort) \
        > "$METADATA_DIR/huggingface_cache_refs.tsv"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    ACTIVE_COMPUTE_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk '/^[[:space:]]*[0-9]+[[:space:]]*$/{gsub(/[[:space:]]/, ""); print}' || true)
    NEW_COMPUTE_PIDS=$(comm -13 \
        <(printf '%s\n' "$BASELINE_COMPUTE_PIDS" | sed '/^[[:space:]]*$/d' | sort -u) \
        <(printf '%s\n' "$ACTIVE_COMPUTE_PIDS" | sed '/^[[:space:]]*$/d' | sort -u))
    if [ -n "$NEW_COMPUTE_PIDS" ]; then
        echo -e "${RED}[GPU] New compute processes remain after predictions: $NEW_COMPUTE_PIDS${NC}"
        exit 1
    fi
    echo -e "${GREEN}[GPU] Prediction processes exited; no new CUDA allocations remain.${NC}"
fi
fi

RESUME_START_INDEX=0
SUPPLEMENTAL_PENDING_MODELS=""
SUPPLEMENTAL_METADATA_TAG=""
SUPPLEMENTAL_PLANNER_ARGS=()
if [ "$RUN_PHASE" != "prediction" ]; then
    if [ "$RUN_PHASE" = "all" ]; then
        if ! "$VALIDATOR_PYTHON" scripts/validate_results.py \
            --scope predictions \
            --output-dir "$OUTPUT_DIR" \
            --models "$MODEL_CSV" \
            --device "$DEVICE" \
            --seed "$BENCHMARK_SEED" \
            --prediction-batch-size "$PREDICTION_BATCH_SIZE"; then
            echo -e "${RED}[STOP] Prediction validation failed; benchmark phase was not started.${NC}"
            exit 1
        fi
        echo -e "${GREEN}[PHASE 1] Saved predictions validated and will not be modified.${NC}"
    else
        echo -e "${GREEN}[PHASE 2] Standalone benchmark mode; prediction artifacts are not required or modified.${NC}"
    fi

    if [ "$SUPPLEMENTAL_BATCHES" = "true" ]; then
        if [ "${RESET_BENCHMARK_RESULTS:-false}" = "true" ]; then
            echo -e "${RED}[STOP] Supplemental mode cannot be combined with RESET_BENCHMARK_RESULTS=true.${NC}"
            exit 1
        fi
        if [ ! -s "$OUTPUT_DIR/benchmark_results_batch_summary.csv" ] \
            || [ "$(wc -l < "$OUTPUT_DIR/benchmark_results_batch_summary.csv")" -le 1 ]; then
            echo -e "${RED}[STOP] Supplemental mode requires an existing non-empty Phase-2 summary.${NC}"
            exit 1
        fi
        SUPPLEMENTAL_PLANNER_ARGS=(
            --output-dir "$OUTPUT_DIR"
            --device "$DEVICE"
            --seed "$BENCHMARK_SEED"
            --dataset "$DATASET_FOLDER"
            --context-lens "$CTX_LENS"
            --horizon-lens "$HORIZON_LENS"
            --batch-sizes "$BATCH_SIZES"
            --power-modes "$POWER_MODES"
            --repeats "$REPEATS"
            --benchmark-batches "$BENCHMARK_BATCHES"
        )
        SUPPLEMENTAL_PENDING_MODELS=$("$VALIDATOR_PYTHON" scripts/supplemental_batches.py \
            "${SUPPLEMENTAL_PLANNER_ARGS[@]}" \
            --models "$MODEL_CSV" \
            --prepare)
        SUPPLEMENTAL_BATCH_TAG=$(printf '%s' "$BATCH_SIZES" | tr ',' '_' | tr -cd '[:alnum:]_-')
        SUPPLEMENTAL_METADATA_TAG="supplemental_bs_${SUPPLEMENTAL_BATCH_TAG}_${RUN_STAMP}"
        if [ -n "$SUPPLEMENTAL_PENDING_MODELS" ]; then
            echo -e "${GREEN}[SUPPLEMENTAL] Existing rows are protected; pending models: ${SUPPLEMENTAL_PENDING_MODELS//,/ }.${NC}"
        else
            echo -e "${GREEN}[SUPPLEMENTAL] Requested batch sizes are already complete for every selected model.${NC}"
        fi
    elif [ "${RESET_BENCHMARK_RESULTS:-false}" = "true" ]; then
        rm -f "$OUTPUT_DIR/benchmark_results_batch_summary.csv" \
            "$OUTPUT_DIR/benchmark_results_raw.csv"
        rm -f "$MODEL_LOG_DIR"/*_benchmark.log
        echo -e "${YELLOW}[RESUME] Cleared benchmark rows/logs; prediction artifacts, if present, were preserved.${NC}"
    elif [ -f "$OUTPUT_DIR/benchmark_results_batch_summary.csv" ] \
        && [ "$(wc -l < "$OUTPUT_DIR/benchmark_results_batch_summary.csv")" -gt 1 ]; then
        if [ "$AUTO_RESUME" != "true" ]; then
            echo -e "${RED}[STOP] Existing benchmark rows found in $OUTPUT_DIR.${NC}"
            echo "Set AUTO_RESUME=true to continue or RESET_BENCHMARK_RESULTS=true to restart phase 2."
            exit 1
        fi
        RESUME_START_INDEX=$("$VALIDATOR_PYTHON" scripts/benchmark_resume.py \
            --output-dir "$OUTPUT_DIR" \
            --models "$MODEL_CSV" \
            --device "$DEVICE" \
            --seed "$BENCHMARK_SEED" \
            --dataset "$DATASET_FOLDER" \
            --context-lens "$CTX_LENS" \
            --horizon-lens "$HORIZON_LENS" \
            --batch-sizes "$BATCH_SIZES" \
            --power-modes "$POWER_MODES" \
            --repeats "$REPEATS" \
            --benchmark-batches "$BENCHMARK_BATCHES" \
            --prepare)
        if ! [[ "$RESUME_START_INDEX" =~ ^[0-9]+$ ]] \
            || [ "$RESUME_START_INDEX" -gt "$TOTAL" ]; then
            echo -e "${RED}[STOP] Resume planner returned an invalid model index.${NC}"
            exit 1
        fi
    fi
fi

if [ "$RUN_PHASE" = "prediction" ]; then
    if ! "$VALIDATOR_PYTHON" scripts/validate_results.py \
        --scope predictions \
        --output-dir "$OUTPUT_DIR" \
        --models "$MODEL_CSV" \
        --device "$DEVICE" \
        --seed "$BENCHMARK_SEED" \
        --prediction-batch-size "$PREDICTION_BATCH_SIZE"; then
        exit 1
    fi
    echo -e "${GREEN}${BOLD}Prediction phase completed successfully; benchmark phase was not started.${NC}"
    exit 0
fi

if [ "$SUPPLEMENTAL_BATCHES" != "true" ] || [ -n "$SUPPLEMENTAL_PENDING_MODELS" ]; then
    if [ "$RUN_PHASE" = "benchmark" ]; then
        cooldown_countdown "$PHASE_COOLDOWN_SEC" "Pre-benchmark cooling"
    else
        cooldown_countdown "$PHASE_COOLDOWN_SEC" "Prediction-benchmark cooling"
    fi
fi

# Benchmark performance without prediction persistence or accuracy evaluation.
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}  Phase 2/2: Performance Benchmark${NC}"
echo -e "${BOLD}============================================================${NC}"
supplemental_model_pending() {
    case ",$SUPPLEMENTAL_PENDING_MODELS," in
        *",$1,"*) return 0 ;;
        *) return 1 ;;
    esac
}

PASS_COUNT=$RESUME_START_INDEX
FAIL_COUNT=0
FAILED_MODELS=()
PASSED_MODELS=()
RESULT_VALIDATION_FAILED=false
RESULT_VALIDATION_RAN=false
for ((completed_index=0; completed_index<RESUME_START_INDEX; completed_index++)); do
    PASSED_MODELS+=("${MODELS[$completed_index]}")
done
if [ "$SUPPLEMENTAL_BATCHES" = "true" ]; then
    PASS_COUNT=0
    for MODEL in "${MODELS[@]}"; do
        if supplemental_model_pending "$MODEL"; then
            continue
        fi
        PASS_COUNT=$((PASS_COUNT + 1))
        PASSED_MODELS+=("$MODEL")
    done
fi

for i in "${!MODELS[@]}"; do
    if [ "$i" -lt "$RESUME_START_INDEX" ]; then
        continue
    fi
    MODEL="${MODELS[$i]}"
    if [ "$SUPPLEMENTAL_BATCHES" = "true" ] && ! supplemental_model_pending "$MODEL"; then
        continue
    fi
    NUM=$((i + 1))
    MODEL_ENV_NAME=$(model_environment_name "$MODEL")
    VENV_DIR=".venv_${MODEL_ENV_NAME}"
    if [ "$SUPPLEMENTAL_BATCHES" = "true" ]; then
        MODEL_LOG="$MODEL_LOG_DIR/${MODEL}_benchmark_${SUPPLEMENTAL_METADATA_TAG}.log"
    else
        MODEL_LOG="$MODEL_LOG_DIR/${MODEL}_benchmark.log"
    fi

    draw_progress_bar "$i" "$TOTAL"
    echo ""

    IFS=',' read -r -a BATCH_SIZE_ITEMS <<< "$BATCH_SIZES"
    BATCH_SIZE_COUNT=${#BATCH_SIZE_ITEMS[@]}
    FIRST_BATCH_SIZE=${BATCH_SIZE_ITEMS[0]}
    LAST_BATCH_SIZE=${BATCH_SIZE_ITEMS[$((BATCH_SIZE_COUNT - 1))]}
    printf "  ${BOLD}[%2d/%d] %-12s${NC} | ctx=%s hor=%s | %d batch sizes (%s..%s)\n" \
        "$NUM" "$TOTAL" "$MODEL" "$CTX_LENS" "$HORIZON_LENS" \
        "$BATCH_SIZE_COUNT" "$FIRST_BATCH_SIZE" "$LAST_BATCH_SIZE"
    printf "           ${CYAN}warmups=%s repeats=%s workers=%s | log: %s${NC}\n" \
        "$WARMUPS" "$REPEATS" "$NUM_WORKERS" "$(basename "$MODEL_LOG")"
    
    EXIT_CODE=
    VENV_PYTHON="$VENV_DIR/bin/python"
    if [ ! -x "$VENV_PYTHON" ]; then
        if command -v python3 >/dev/null 2>&1; then
            VENV_PYTHON="$(command -v python3)"
        elif command -v python >/dev/null 2>&1; then
            VENV_PYTHON="$(command -v python)"
        else
            echo "           ${RED}No Python interpreter found for $MODEL${NC}"
            EXIT_CODE=127
        fi
    fi

    MODEL_START=$(date +%s)
    SUITE_ELAPSED=$((MODEL_START - SCRIPT_START))
    echo -e "           ${CYAN}Live:${NC} power mode, config, iteration, and batch progress (${TQDM_MININTERVAL}s refresh)."

    set +e
    if [ -z "${EXIT_CODE:-}" ]; then
    PYTHONFAULTHANDLER=1 \
    BENCHMARK_METADATA_TAG="$SUPPLEMENTAL_METADATA_TAG" \
    VIRTUAL_ENV="$(pwd)/$VENV_DIR" \
    PATH="$(pwd)/$VENV_DIR/bin:$BASE_PATH" \
    "$VENV_PYTHON" -u -m experiments.forecasting \
        "${DATA_ARGS[@]}" \
        --phase benchmark \
        --models "$MODEL" \
        --context-lens "$CTX_LENS" \
        --horizon-lens "$HORIZON_LENS" \
        --batch-sizes "$BATCH_SIZES" \
        --warmups $WARMUPS \
        --benchmark-batches "$BENCHMARK_BATCHES" \
        --max-config-duration-sec "$CONFIG_TIMEOUT_SEC" \
        --seed "$BENCHMARK_SEED" \
        --accuracy-seasonality "$ACCURACY_SEASONALITY" \
        --repeats $REPEATS \
        --power-modes "$POWER_MODES" \
        --power-mode-stabilization-sec "$POWER_MODE_STABILIZATION_SEC" \
        --idle-baseline-sec $IDLE_BASELINE_SEC \
        --max-idle-gpu-util "$MAX_IDLE_GPU_UTIL" \
        --max-idle-cpu-util "$MAX_IDLE_CPU_UTIL" \
        --sleep-between-batches "$SLEEP_BETWEEN_BATCHES_SEC" \
        --output-dir "$OUTPUT_DIR" \
        --device "$DEVICE" \
        --num-workers "$NUM_WORKERS" \
        --log-file "$MODEL_LOG" \
        "${POWER_MODE_ARGS[@]}" \
        "${RAW_PERFORMANCE_ARGS[@]}"
    EXIT_CODE=$?
    fi
    set -e

    if [ "$EXIT_CODE" -eq 0 ] && [ "$SUPPLEMENTAL_BATCHES" = "true" ]; then
        set +e
        MODEL_PENDING_AFTER=$("$VALIDATOR_PYTHON" scripts/supplemental_batches.py \
            "${SUPPLEMENTAL_PLANNER_ARGS[@]}" \
            --models "$MODEL" 2>> "$MODEL_LOG")
        SUPPLEMENTAL_CHECK_EXIT=$?
        set -e
        if [ "$SUPPLEMENTAL_CHECK_EXIT" -ne 0 ] || [ -n "$MODEL_PENDING_AFTER" ]; then
            printf '%s [SUPPLEMENTAL] Requested rows failed post-run validation.\n' \
                "$(date '+%Y-%m-%d %H:%M:%S')" >> "$MODEL_LOG"
            EXIT_CODE=1
        fi
    fi

    MODEL_END=$(date +%s)
    ELAPSED=$((MODEL_END - MODEL_START))
    DURATION=$(format_duration "$ELAPSED")
    SUITE_ELAPSED=$((MODEL_END - SCRIPT_START))
    SUITE_DURATION=$(format_duration "$SUITE_ELAPSED")

    printf "  ${BOLD}[%2d/%d] %-12s${NC}" "$NUM" "$TOTAL" "$MODEL"
    if [ "$EXIT_CODE" -eq 0 ]; then
        echo -e " ${GREEN}✓ PASS${NC} | ${CYAN}Time:${NC} ${DURATION} | ${CYAN}Total:${NC} ${SUITE_DURATION}"
        echo -e "           ${CYAN}Log:${NC} $MODEL_LOG"
        PASS_COUNT=$((PASS_COUNT + 1))
        PASSED_MODELS+=("$MODEL")
    else
        printf '%s [LAUNCHER] Benchmark process exited with code %s\n' \
            "$(date '+%Y-%m-%d %H:%M:%S')" "$EXIT_CODE" >> "$MODEL_LOG"
        echo -e " ${RED}✗ FAIL${NC} | ${CYAN}Time:${NC} ${DURATION} | ${CYAN}Total:${NC} ${SUITE_DURATION}"
        echo -e "           ${RED}Error in:${NC} $MODEL"
        if [ -f "$MODEL_LOG" ]; then
            echo -e "           ${CYAN}Last 5 lines from log:${NC}"
            tail -5 "$MODEL_LOG" | sed 's/^/           /'
        fi
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_MODELS+=("$MODEL")
    fi
    
    echo -e "           ${BOLD}Progress:${NC} "
    draw_mini_progress_bar "$NUM" "$TOTAL" 30
    REMAINING=$(estimate_time_remaining "$SUITE_ELAPSED" "$NUM" "$TOTAL")
    printf " | ${CYAN}ETA:${NC} ~${REMAINING}\n"
    echo ""

    if [ "$EXIT_CODE" -ne 0 ]; then
        echo -e "           ${YELLOW}Stopping at the failed model; rerun phase 2 to resume here.${NC}"
        break
    fi
    if [ "$NUM" -lt "$TOTAL" ]; then
        cooldown_countdown "$MODEL_COOLDOWN_SEC" "Model cooling"
    fi
done

if [ "$FAIL_COUNT" -eq 0 ]; then
    RESULT_VALIDATION_RAN=true
    MODEL_CSV=$(IFS=,; echo "${MODELS[*]}")
    VALIDATOR_ENV_NAME=$(model_environment_name "${MODELS[0]}")
    VALIDATOR_PYTHON=".venv_${VALIDATOR_ENV_NAME}/bin/python"
    RESULT_VALIDATION_SCOPE="all"
    if [ "$RUN_PHASE" = "benchmark" ]; then
        RESULT_VALIDATION_SCOPE="benchmark"
    fi
    if ! "$VALIDATOR_PYTHON" scripts/validate_results.py \
        --scope "$RESULT_VALIDATION_SCOPE" \
        --output-dir "$OUTPUT_DIR" \
        --models "$MODEL_CSV" \
        --device "$DEVICE" \
        --seed "$BENCHMARK_SEED" \
        --prediction-batch-size "$PREDICTION_BATCH_SIZE"; then
        RESULT_VALIDATION_FAILED=true
    fi
fi

draw_progress_bar "$TOTAL" "$TOTAL"
echo ""
echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}  Benchmark Summary${NC}"
echo -e "${BOLD}============================================================${NC}"
echo -e "  Models Tested:"
for MODEL in "${MODELS[@]}"; do
    if [[ " ${PASSED_MODELS[*]} " =~ " ${MODEL} " ]]; then
        echo -e "    ✓ ${GREEN}${MODEL}${NC}"
    elif [[ " ${FAILED_MODELS[*]} " =~ " ${MODEL} " ]]; then
        echo -e "    ✗ ${RED}${MODEL}${NC}"
    fi
done
echo ""
echo -e "  Results:"
echo -e "    Passed: ${GREEN}${BOLD}$PASS_COUNT${NC}/$TOTAL"
echo -e "    Failed: ${RED}${BOLD}$FAIL_COUNT${NC}/$TOTAL"
if [ "$FAIL_COUNT" -gt 0 ]; then
    echo -e "    Failed Models: ${RED}${FAILED_MODELS[*]}${NC}"
fi
if [ "$RESULT_VALIDATION_RAN" != "true" ]; then
    echo -e "    Validation: ${YELLOW}NOT RUN (model failure)${NC}"
elif [ "$RESULT_VALIDATION_FAILED" = "true" ]; then
    echo -e "    Validation: ${RED}FAIL${NC}"
else
    echo -e "    Validation: ${GREEN}PASS${NC}"
fi
SCRIPT_END=$(date +%s)
TOTAL_DURATION=$((SCRIPT_END - SCRIPT_START))
TOTAL_FMT=$(format_duration "$TOTAL_DURATION")

echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}  Execution Summary${NC}"
echo -e "${BOLD}============================================================${NC}"
echo -e "  ${CYAN}Total Duration:${NC}  ${BOLD}$TOTAL_FMT${NC}"
echo -e "  ${CYAN}Data Directory:${NC}  $DATA_DIR"
echo -e "  ${CYAN}Output Directory:${NC} $OUTPUT_DIR"
if [ "$SAVE_RAW_PERFORMANCE_METRICS" = "true" ]; then
    echo -e "  ${CYAN}Raw Results:${NC}      ${BOLD}$OUTPUT_DIR/benchmark_results_raw.csv${NC}"
else
    echo -e "  ${CYAN}Raw Results:${NC}      disabled (set SAVE_RAW_PERFORMANCE_METRICS=true to enable)"
fi
echo -e "  ${CYAN}Batch Summary:${NC}    ${BOLD}$OUTPUT_DIR/benchmark_results_batch_summary.csv${NC}"
echo -e "  ${CYAN}Preprocessing:${NC}    ${BOLD}$OUTPUT_DIR/dataset_preprocessing_report.csv${NC}"
if [ "$RUN_PHASE" = "benchmark" ]; then
    echo -e "  ${CYAN}Predictions:${NC}      not required; existing artifacts were not modified"
else
    echo -e "  ${CYAN}Predictions:${NC}      ${BOLD}$OUTPUT_DIR/benchmark_predictions/<model>/<dataset-folder>/<source>/*.npz${NC}"
fi
echo -e "  ${CYAN}Log Directory:${NC}    $MODEL_LOG_DIR"
echo -e "  ${CYAN}Device:${NC}          $DEVICE"
echo -e "${BOLD}============================================================${NC}"
echo ""
if [ "$FAIL_COUNT" -gt 0 ] || [ "$RESULT_VALIDATION_FAILED" = "true" ]; then
    echo -e "${RED}${BOLD}Benchmark suite completed with failures or invalid results.${NC}"
    echo ""
    exit 1
fi
echo -e "${GREEN}${BOLD}Benchmark suite completed successfully!${NC}"
echo ""
