#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Keep exploratory horizons isolated from publication-run predictions/results.
export OUTPUT_DIR="${OUTPUT_DIR:-./results_horizon_sweep}"
export DATA_CACHE_DIR="${DATA_CACHE_DIR:-./results_unified/data_cache}"
export CTX_LENS="${CTX_LENS:-32,64,128,256,512,1024,2048}"
export HORIZON_LENS="${HORIZON_LENS:-1,6,12,24,48,96,192,360,720}"
export BATCH_SIZES="${BATCH_SIZES:-512}"
export WARMUPS="${WARMUPS:-2}"
export REPEATS="${REPEATS:-3}"
export PHASE_COOLDOWN_SEC="${PHASE_COOLDOWN_SEC:-60}"
export MODEL_COOLDOWN_SEC="${MODEL_COOLDOWN_SEC:-80}"
export POWER_MODE_STABILIZATION_SEC="${POWER_MODE_STABILIZATION_SEC:-60}"
export SLEEP_BETWEEN_BATCHES_SEC="${SLEEP_BETWEEN_BATCHES_SEC:-60}"

exec bash "$SCRIPT_DIR/run_unified_benchmark.sh" "$@"
,,,
