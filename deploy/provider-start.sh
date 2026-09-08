#!/bin/sh
set -eu
: "${PLLM_API_KEY:?Set PLLM_API_KEY in the service environment}"
: "${PLLM_MODEL_SOURCE:?Set PLLM_MODEL_SOURCE}"
exec /opt/pllm/.venv/bin/pllm serve "$PLLM_MODEL_SOURCE" \
  --weights public --model-id "${PLLM_MODEL_ID:-private-model}" \
  --host 127.0.0.1 --port 8000 --local-files-only \
  --compiled-cache-dir /var/cache/pllm
