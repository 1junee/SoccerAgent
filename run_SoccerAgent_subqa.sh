#!/usr/bin/env bash
# Sample 10 random questions from selected SoccerBench QA files
# and run platform_full_version.py for each sample individually.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

QA_DIR="${SCRIPT_DIR}/database/SoccerBench/qa"
QA_FILES=(q1.json q2.json q4.json q6.json q8.json q10.json)

RESULT_ROOT="${SCRIPT_DIR}/result_logs/sub_qa"
mkdir -p "${RESULT_ROOT}"

MANIFEST="$(mktemp)"
python - <<'PY' "${QA_DIR}" "${MANIFEST}" "${QA_FILES[@]}"
import json, random, os, sys

qa_dir = sys.argv[1]
manifest_path = sys.argv[2]
qa_files = sys.argv[3:]

pool = []
for qa_file in qa_files:
    path = os.path.join(qa_dir, qa_file)
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    pool.extend((qa_file, idx) for idx in range(len(data)))

if len(pool) < 10:
    raise SystemExit("Not enough QA entries to sample 10 items.")

sample = random.sample(pool, 10)
with open(manifest_path, 'w', encoding='utf-8') as mf:
    for qa_file, idx in sample:
        mf.write(f"{qa_file}\t{idx}\n")
PY

while IFS=$'\t' read -r qa_file idx; do
    [[ -z "${qa_file:-}" ]] && continue

    question_id="${qa_file%.json}"
    run_dir="${RESULT_ROOT}/${question_id}/${question_id}_${idx}"
    mkdir -p "${run_dir}"

    log_path="${run_dir}/run.log"
    summary_path="${run_dir}/result.json"
    pipeline_dump="${run_dir}/process.txt"

    tmp_input="$(mktemp --suffix=.json)"
    tmp_output="$(mktemp --suffix=.json)"

    python - <<'PY' "${QA_DIR}/${qa_file}" "${idx}" "${tmp_input}"
import json, sys

qa_path = sys.argv[1]
index = int(sys.argv[2])
output_path = sys.argv[3]

with open(qa_path, encoding='utf-8') as f:
    data = json.load(f)

try:
    entry = data[index]
except IndexError:
    raise SystemExit(f"Index {index} out of range for {qa_path}")

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump([entry], f, ensure_ascii=False, indent=2)
PY

    OMP_NUM_THREADS=1 python -m platform_full_version \
        --input_file "${tmp_input}" \
        --output_file "${tmp_output}" \
        "$@" \
        > "${log_path}" 2>&1 || true

    if [[ -s "${tmp_output}" ]]; then
        cp "${tmp_output}" "${summary_path}"
    else
        echo "{}" > "${summary_path}"
    fi

    python - <<'PY' "${tmp_output}" "${pipeline_dump}"
import json, sys, os

output_path = sys.argv[1]
process_path = sys.argv[2]

if not os.path.isfile(output_path):
    sys.exit(0)

try:
    with open(output_path, encoding='utf-8') as f:
        data = json.load(f)
except Exception:
    sys.exit(0)

with open(process_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
PY

    rm -f "${tmp_input}" "${tmp_output}"
done < "${MANIFEST}"

rm -f "${MANIFEST}"

echo "[DONE] Results stored under ${RESULT_ROOT}"
