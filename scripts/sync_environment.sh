#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/prepare_rust.sh
if [[ -f uv.lock ]]; then
  uv sync --locked "$@"
else
  echo "No committed uv.lock. Resolving dependencies for bootstrap validation. Publication requires the lock." >&2
  uv sync "$@"
fi
