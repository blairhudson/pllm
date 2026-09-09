#!/bin/sh
set -eu
: "${PLLM_API_KEY:?Set PLLM_API_KEY in the service environment}"
: "${PLLM_MODEL_SOURCE:?Set PLLM_MODEL_SOURCE}"
: "${PLLM_INFERENCE_URL:?Set PLLM_INFERENCE_URL}"
: "${PLLM_PUSH_API_KEY:?Set PLLM_PUSH_API_KEY}"
[ "$PLLM_API_KEY" != "$PLLM_PUSH_API_KEY" ] || { echo "Use distinct preparation and push credentials" >&2; exit 2; }
exec /opt/pllm/.venv/bin/pllm preparation serve "$PLLM_MODEL_SOURCE" \
  --api-key "$PLLM_API_KEY" --model-id "${PLLM_MODEL_ID:-private-model}" \
  --host 127.0.0.1 --port 8001 --local-files-only \
  --compiled-cache-dir /var/cache/pllm-preparation \
  --inference-url "$PLLM_INFERENCE_URL" --push-api-key "$PLLM_PUSH_API_KEY" \
  --push-timeout "${PLLM_PUSH_TIMEOUT:-10}"
