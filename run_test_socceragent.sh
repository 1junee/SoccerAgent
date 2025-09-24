#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/platform_full_version_$(date +%Y%m%d_%H%M%S).log"

echo "Logging to: ${LOG_FILE}"

python -m platform_full_version \
  --input_file /home/work/wonjun/study/agent/SoccerAgent/database/testqa.json \
  --output_file /home/work/wonjun/study/agent/SoccerAgent/database/testresult.json \
  "$@" \
  2>&1 | tee "${LOG_FILE}"
