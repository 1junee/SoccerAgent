#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="sc"
PYTHON_VERSION="3.10"
REQ_FILE="requirements.txt"

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda command not found. Install Miniconda or Anaconda first." >&2
    exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure conda shell functions (e.g. conda activate) are available.
source "$(conda info --base)/etc/profile.d/conda.sh"

# Accept Anaconda Terms of Service for default channels if needed (Conda 23.11+).
if conda tos --help >/dev/null 2>&1; then
    echo "Ensuring Anaconda Terms of Service are accepted for required channels..."
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
else
    echo "Note: 'conda tos' command unavailable; ensure required channels' ToS are accepted manually." >&2
fi

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "Conda environment '${ENV_NAME}' already exists. Skipping creation."
else
    echo "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

conda activate "${ENV_NAME}"

REQ_PATH="${SCRIPT_DIR}/${REQ_FILE}"
if [[ ! -f "${REQ_PATH}" ]]; then
    echo "Error: ${REQ_FILE} not found in ${SCRIPT_DIR}." >&2
    exit 1
fi

echo "Upgrading pip inside '${ENV_NAME}'..."
pip install --upgrade pip

# Pre-install torch packages before the rest to satisfy flash_attn metadata build.
mapfile -t TORCH_PACKAGES < <(grep -E '^(torch==|torchaudio==|torchvision==)' "${REQ_PATH}" || true)
if (( ${#TORCH_PACKAGES[@]} > 0 )); then
    echo "Pre-installing torch packages: ${TORCH_PACKAGES[*]}"
    pip install "${TORCH_PACKAGES[@]}"
fi

echo "Installing packages from ${REQ_FILE}..."
pip install -r "${REQ_PATH}"

echo "Environment '${ENV_NAME}' is ready."
