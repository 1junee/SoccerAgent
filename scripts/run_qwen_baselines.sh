#!/usr/bin/env bash
set -euo pipefail

# run_qwen_baselines.sh
# Run baseline.py with Qwen for qa/q1.json .. qa/q14.json
# - Ensures working dir is the SoccerBench dataset so relative materials paths resolve
# - Creates fresh CSVs by default to avoid length mismatches; set RESUME=true to append
# - Supports parallel execution per GPU (one run per GPU at a time)
#
# Usage:
#   bash SoccerAgent/scripts/run_qwen_baselines.sh [--start N] [--end M]
#
# Env vars (override as needed):
#   ROOT          Project root (auto from this script)
#   DATA_DIR      Default: "$ROOT/database/datasets/SoccerBench"
#   MODEL_PATH    Default: "$ROOT/Qwen2.5-VL"
#   QUESTIONS_DIR Default: "qa" (inside DATA_DIR)
#   OUTPUT_PREFIX Default: "qwen_"
#   RESUME        Default: "false" (fresh CSV). Set to "true" to resume appending.
#   CUDA_VISIBLE_DEVICES  Default: "0" (comma-separated list for multi-GPU)
#   MAX_JOBS      Default: number of visible GPUs
#   TF_CPP_MIN_LOG_LEVEL  Default: "2" (quieter TF logs)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
DATA_DIR="${DATA_DIR:-$ROOT/database/datasets/SoccerBench}"
MODEL_PATH="${MODEL_PATH:-$ROOT/Qwen2.5-VL}"
QUESTIONS_DIR="${QUESTIONS_DIR:-qa}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-qwen_}"
RESUME="${RESUME:-false}"

: "${CUDA_VISIBLE_DEVICES:=0}"; export CUDA_VISIBLE_DEVICES
: "${MAX_JOBS:=}" # compute later if empty
: "${TF_CPP_MIN_LOG_LEVEL:=2}"; export TF_CPP_MIN_LOG_LEVEL

START=1
END=14
while [[ $# -gt 0 ]]; do
  case "$1" in
    --start) START="$2"; shift 2 ;;
    --end)   END="$2";   shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

echo "==> Project root   : $ROOT"
echo "==> Dataset dir    : $DATA_DIR"
echo "==> Questions dir  : $QUESTIONS_DIR"
echo "==> Model path     : $MODEL_PATH"
echo "==> Output prefix  : $OUTPUT_PREFIX"
echo "==> Range          : q${START}..q${END}"
echo "==> Devices        : CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "==> Resume         : $RESUME"

cd "$DATA_DIR"

# Parse visible GPUs and set max parallel jobs
IFS=',' read -r -a GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
if [[ -z "${MAX_JOBS}" ]]; then
  MAX_JOBS=${#GPU_LIST[@]}
  [[ $MAX_JOBS -eq 0 ]] && MAX_JOBS=1
fi
echo "==> Max parallel jobs: $MAX_JOBS"

# Pre-clean outputs if not resuming
if [[ "$RESUME" != "true" ]]; then
  for i in $(seq "$START" "$END"); do
    OUTFILE="${OUTPUT_PREFIX}q${i}.csv"
    rm -f "$OUTFILE"
  done
fi

run_one() {
  local qidx="$1" dev="$2"
  local qfile="$QUESTIONS_DIR/q${qidx}.json"
  local outfile="${OUTPUT_PREFIX}q${qidx}.csv"
  if [[ ! -f "$qfile" ]]; then
    echo "[Skip] Missing $qfile"
    return 0
  fi
  echo "[Launch][GPU $dev] q${qidx}: $qfile -> $outfile"
  CUDA_VISIBLE_DEVICES="$dev" TF_CPP_MIN_LOG_LEVEL="$TF_CPP_MIN_LOG_LEVEL" \
  python "$ROOT/baseline/baseline.py" \
    --model qwen \
    --input_file "$qfile" \
    --output_file "$outfile" \
    --model_path "$MODEL_PATH" \
  > "${outfile%.csv}.log" 2>&1 &
  echo $!
}

declare -a PIDS=()
declare -a GPUS=()
next_gpu_idx=0

for i in $(seq "$START" "$END"); do
  # Throttle to MAX_JOBS concurrent processes
  while (( ${#PIDS[@]} >= MAX_JOBS )); do
    if wait -n 2>/dev/null; then
      :
    else
      # Portable fallback: wait for the oldest PID
      wait "${PIDS[0]}" || true
    fi
    # Compact arrays to remove finished PIDs
    tmp_pids=()
    tmp_gpus=()
    for idx in "${!PIDS[@]}"; do
      if kill -0 "${PIDS[$idx]}" 2>/dev/null; then
        tmp_pids+=("${PIDS[$idx]}")
        tmp_gpus+=("${GPUS[$idx]}")
      fi
    done
    PIDS=("${tmp_pids[@]}")
    GPUS=("${tmp_gpus[@]}")
  done

  # Choose GPU in round-robin
  dev="${GPU_LIST[$(( next_gpu_idx % ${#GPU_LIST[@]} ))]}"
  next_gpu_idx=$(( next_gpu_idx + 1 ))

  pid=$(run_one "$i" "$dev") || pid=""
  if [[ -n "$pid" ]]; then
    PIDS+=("$pid")
    GPUS+=("$dev")
  fi
done

# Wait for all remaining jobs
for pid in "${PIDS[@]}"; do
  wait "$pid" || true
done

echo "\nAll done. Outputs in: $DATA_DIR"
