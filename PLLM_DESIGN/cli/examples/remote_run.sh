#!/usr/bin/env bash
# Client host; this command MUST NOT start, contact, or refill from Preparation.
set -euo pipefail
cd "$(dirname "$0")"
pllm run build/qwen/plan.lock.json \
  --deployment remote.yaml --prepared-only --input-file question.txt
