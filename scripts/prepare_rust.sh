#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
rustc --version
cargo --version
if [[ -f Cargo.lock ]]; then
  cargo metadata --locked --format-version 1 >/dev/null
else
  echo "Resolving Cargo dependencies for bootstrap CI; release requires a committed lock." >&2
  cargo generate-lockfile
fi
