#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../docs"
if [[ -f package-lock.json ]]; then
  npm ci --no-audit --no-fund
else
  echo "No committed npm lock. Resolving dependencies for bootstrap validation." >&2
  npm install --no-audit --no-fund
fi
