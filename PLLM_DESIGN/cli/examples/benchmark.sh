#!/usr/bin/env bash
# Fresh material for one warmup plus five timed requests; never reused across them.
set -euo pipefail
cd "$(dirname "$0")"
pllm bench run pllm.yaml \
  --set budget__requests=6 \
  --input-tokens 32 --output-tokens 16 \
  --warmup 1 --repeats 5 --output runs/smoke
pllm bench search pllm.yaml \
  --set budget__requests=6 \
  --grid 'pipeline__components__kernels__threads=[1,2,4,8]' \
  --input-tokens 32 --output-tokens 16 \
  --warmup 1 --repeats 5 --output runs/thread-search
