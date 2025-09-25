#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="/home/heodnjswns/SoccerAgent"
INPUT_BASE="$REPO_DIR/database/SoccerBench/subqa/random_1per"
OUTPUT_BASE="$REPO_DIR/database/SoccerBench/run_outputs/scAgent/subqa/random_1per"

declare -a QA_LIST=(q4 q6 q8 q10)
# declare -a QA_LIST=(q1 q2 q4 q6 q8 q10)

cd "$REPO_DIR"

for qa in "${QA_LIST[@]}"; do
    input_file="$INPUT_BASE/${qa}.json"
    output_dir="$OUTPUT_BASE/${qa}"
    output_file="$output_dir/${qa}_output.json"
    log_file="$output_dir/run.log"

    mkdir -p "$output_dir"

    echo "[INFO] Processing $qa"
    CUDA_VISIBLE_DEVICES="2,3" python platform_full_version.py \
        --input_file "$input_file" \
        --output_file "$output_file" \
        --sample_mode \
        --sample_ratio 0.01 \
        >"$log_file" 2>&1
done

echo "[INFO] Completed all runs. Outputs stored under $OUTPUT_BASE"
