#!/usr/bin/env bash
# Download SoccerWiki and SoccerBench datasets into the expected folders
# - Uses huggingface-cli (from huggingface_hub)
# - Optional env: HUGGINGFACE_TOKEN or HUGGINGFACE_HUB_TOKEN for private/authenticated access
# - Idempotent: For SoccerWiki, only missing files are downloaded (skip existing files)

set -euo pipefail

log()  { printf "[INFO] %s\n" "$*"; }
warn() { printf "[WARN] %s\n" "$*" >&2; }
err()  { printf "[ERROR] %s\n" "$*" >&2; exit 1; }

need_cmd() { command -v "$1" >/dev/null 2>&1 || err "'$1' command is required"; }
have_cmd() { command -v "$1" >/dev/null 2>&1; }

# Pick a Python executable
pick_python() {
  if have_cmd python; then
    PYTHON_BIN=python
  elif have_cmd python3; then
    PYTHON_BIN=python3
  else
    err "Python is required (python or python3 not found)"
  fi
}

# Ensure huggingface tools are available; installs in active env if missing
ensure_hf_tools() {
  pick_python
  # Detect if running in a virtualenv/conda env
  IN_VENV=$("$PYTHON_BIN" - <<'PY'
import sys
print((hasattr(sys, 'real_prefix')) or (getattr(sys, 'base_prefix', sys.prefix) != sys.prefix))
PY
)
  # Ensure pip exists
  "$PYTHON_BIN" -m ensurepip --upgrade >/dev/null 2>&1 || true
  # Choose pip install flags: prefer env-local install when in venv/conda
  if [ "$IN_VENV" = "True" ]; then PIP_FLAGS=""; else PIP_FLAGS="--user"; fi

  # Install/upgrade huggingface_hub Python package if missing or too old
  if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import pkg_resources
pkg_resources.require('huggingface_hub>=0.22.0')
PY
  then
    log "Installing/upgrading huggingface_hub>=0.22.0 in active Python env"
    "$PYTHON_BIN" -m pip install --upgrade $PIP_FLAGS pip >/dev/null 2>&1 || true
    "$PYTHON_BIN" -m pip install --upgrade $PIP_FLAGS "huggingface_hub>=0.22.0"
  fi

  # Ensure CLI is available on PATH (console_script from huggingface_hub)
  if ! have_cmd huggingface-cli; then
    # Add common user bin to PATH for this shell session
    export PATH="$HOME/.local/bin:$HOME/Library/Python/3.*/bin:$PATH"
  fi
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Paths
DB_DIR="$ROOT_DIR/database"
WIKI_DIR="$DB_DIR/SoccerWiki"
BENCH_DIR="$ROOT_DIR/benchmark/SoccerBench"

# Repos (SoccerWiki may exist under different orgs; try official first)
WIKI_REPO_OFFICIAL="SJTU-AI4Sports/SoccerWiki"
WIKI_REPO_FALLBACK="Homie0609/SoccerWiki"
BENCH_REPO="Homie0609/SoccerBench"

# Revision/branch (default main)
WIKI_REVISION="main"
BENCH_REVISION="main"

sync_soccerwiki_skip_existing() {
  # Idempotent sync using huggingface_hub from Python: only download files that are missing locally
  pick_python

  local repo_id_official="$WIKI_REPO_OFFICIAL"
  local repo_id_fallback="$WIKI_REPO_FALLBACK"
  local dest_dir="$WIKI_DIR"
  local revision="$WIKI_REVISION"

  mkdir -p "$dest_dir"

  local token="${HUGGINGFACE_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"

  "$PYTHON_BIN" - <<PY
import os
import sys
import shutil
from typing import List

def log(*a):
    print("[INFO]", *a)

def warn(*a):
    print("[WARN]", *a, file=sys.stderr)

def sync_repo(repo_id: str, dest_dir: str, revision: str, token: str | None) -> tuple[int, int]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as e:
        warn("huggingface_hub is required:", e)
        raise

    api = HfApi(token=token or None)
    try:
        files: List[str] = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    except Exception as e:
        warn(f"Failed to list files for {repo_id}@{revision}:", e)
        raise

    wanted = [p for p in files if p.startswith("data/") or p.startswith("pic/")]
    downloaded = 0
    skipped = 0

    for rel in wanted:
        dst = os.path.join(dest_dir, rel)
        if os.path.exists(dst):
            skipped += 1
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            cache_path = hf_hub_download(repo_id=repo_id, filename=rel, repo_type="dataset", revision=revision, token=token or None, resume_download=True)
        except Exception as e:
            warn(f"Download failed for {rel}: {e}")
            continue
        try:
            shutil.copy2(cache_path, dst)
            downloaded += 1
        except Exception as e:
            warn(f"Copy failed for {rel}: {e}")
    return downloaded, skipped

repo_id_official = os.environ.get("WIKI_REPO_OFFICIAL") or "SJTU-AI4Sports/SoccerWiki"
repo_id_fallback = os.environ.get("WIKI_REPO_FALLBACK") or "Homie0609/SoccerWiki"
dest_dir = os.environ.get("WIKI_DIR") or "./SoccerWiki"
revision = os.environ.get("WIKI_REVISION") or "main"
token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

log(f"Syncing SoccerWiki (skip existing) to: {dest_dir}")

try:
    dl, sk = sync_repo(repo_id_official, dest_dir, revision, token)
    log(f"Official repo synced: downloaded={dl}, skipped={sk}")
except Exception as e:
    warn(f"Official repo failed ({repo_id_official}): {e}\nTrying fallback: {repo_id_fallback}")
    dl, sk = sync_repo(repo_id_fallback, dest_dir, revision, token)
    log(f"Fallback repo synced: downloaded={dl}, skipped={sk}")

PY
}

sync_soccerbench_skip_existing() {
  # Idempotent sync for SoccerBench QA JSONs (skip existing)
  pick_python

  local repo_id="$BENCH_REPO"
  local dest_dir="$BENCH_DIR"
  local revision="$BENCH_REVISION"

  mkdir -p "$dest_dir/qa"

  local token="${HUGGINGFACE_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"

  "$PYTHON_BIN" - <<PY
import os, sys, shutil

def log(*a):
    print("[INFO]", *a)

def warn(*a):
    print("[WARN]", *a, file=sys.stderr)

def main(repo_id: str, dest_dir: str, revision: str, token: str | None):
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as e:
        warn("huggingface_hub is required:", e)
        raise
    api = HfApi(token=token or None)
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    except Exception as e:
        warn(f"Failed to list files for {repo_id}@{revision}: {e}")
        raise

    wanted = [p for p in files if p.startswith("qa/") and p.endswith(".json")]
    downloaded = skipped = 0
    for rel in wanted:
        dst = os.path.join(dest_dir, rel)
        if os.path.exists(dst):
            skipped += 1
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            cache = hf_hub_download(repo_id=repo_id, filename=rel, repo_type="dataset", revision=revision, token=token or None, resume_download=True)
        except Exception as e:
            warn(f"Download failed for {rel}: {e}")
            continue
        try:
            shutil.copy2(cache, dst)
            downloaded += 1
        except Exception as e:
            warn(f"Copy failed for {rel}: {e}")
    log(f"SoccerBench QA synced: downloaded={downloaded}, skipped={skipped}")

token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
main(repo_id=os.environ.get("BENCH_REPO", "Homie0609/SoccerBench"), dest_dir=os.environ.get("BENCH_DIR", "./SoccerBench"), revision=os.environ.get("BENCH_REVISION", "main"), token=token)
PY
}

main() {
  ensure_hf_tools
  need_cmd huggingface-cli
  mkdir -p "$WIKI_DIR" "$BENCH_DIR"

  # Optional non-interactive login for private/authenticated repos
  local token="${HUGGINGFACE_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
  if [[ -n "$token" ]]; then
    log "Logging into Hugging Face CLI non-interactively"
    huggingface-cli login --token "$token" --add-to-git-credential || warn "HF login failed; proceeding unauthenticated"
  else
    log "No HF token provided; attempting public download"
  fi

  # SoccerWiki: sync only missing files (data/** and pic/**)
  log "Syncing SoccerWiki (skip existing files)"
  WIKI_REPO_OFFICIAL="$WIKI_REPO_OFFICIAL" \
  WIKI_REPO_FALLBACK="$WIKI_REPO_FALLBACK" \
  WIKI_DIR="$WIKI_DIR" \
  WIKI_REVISION="$WIKI_REVISION" \
  HUGGINGFACE_TOKEN="$token" \
  HUGGINGFACE_HUB_TOKEN="$token" \
    sync_soccerwiki_skip_existing
  log "SoccerWiki sync complete: $WIKI_DIR"

  # SoccerBench QA JSONs: sync only missing
  log "Syncing SoccerBench QA (skip existing files) from: $BENCH_REPO"
  BENCH_REPO="$BENCH_REPO" \
  BENCH_DIR="$BENCH_DIR" \
  BENCH_REVISION="$BENCH_REVISION" \
  HUGGINGFACE_TOKEN="$token" \
  HUGGINGFACE_HUB_TOKEN="$token" \
    sync_soccerbench_skip_existing
  log "SoccerBench sync complete: $BENCH_DIR/qa"

  cat <<EOF

Done.
- SoccerWiki (skip existing): $WIKI_DIR
 - SoccerBench QA (skip existing): $BENCH_DIR/qa

If Game_dataset is archived, unzip it into: $DB_DIR/Game_dataset
  e.g., tar -xzf $DB_DIR/Game_dataset.tar.gz -C $DB_DIR

EOF
}

main "$@"
