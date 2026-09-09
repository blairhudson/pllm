#!/bin/sh
set -eu
: "${PLLM_API_KEY:?Set PLLM_API_KEY in the service environment}"
: "${PLLM_MODEL_SOURCE:?Set PLLM_MODEL_SOURCE}"
: "${PLLM_PROVIDER_PUSH_API_KEY:?Set PLLM_PROVIDER_PUSH_API_KEY}"
[ "$PLLM_API_KEY" != "$PLLM_PROVIDER_PUSH_API_KEY" ] || { echo "Use distinct client and push credentials" >&2; exit 2; }
exec /opt/pllm/.venv/bin/pllm serve "$PLLM_MODEL_SOURCE" \
  --weights public --model-id "${PLLM_MODEL_ID:-private-model}" \
  --host 127.0.0.1 --port 8000 --local-files-only \
  --compiled-cache-dir /var/cache/pllm \
  --provider-push-api-key "$PLLM_PROVIDER_PUSH_API_KEY" \
  --rendezvous-timeout "${PLLM_RENDEZVOUS_TIMEOUT:-30}" \
  --rendezvous-capacity "${PLLM_RENDEZVOUS_CAPACITY:-32768}" \
  --rendezvous-max-bytes "${PLLM_RENDEZVOUS_MAX_BYTES:-268435456}"
