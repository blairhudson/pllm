#!/bin/sh
set -eu
: "${PLLM_API_KEY:?Set PLLM_API_KEY}"
case "${1:-provider}" in
 provider)
  : "${PLLM_MODEL_SOURCE:?Set PLLM_MODEL_SOURCE to a mounted checkpoint}"
  exec pllm serve "$PLLM_MODEL_SOURCE" --weights public --model-id "${PLLM_MODEL_ID:-private-model}" --host 0.0.0.0 --port 8000 --local-files-only --compiled-cache-dir /var/cache/pllm
  ;;
 gateway)
  : "${PLLM_BASE_URL:?Set PLLM_BASE_URL to the provider URL}"
  : "${PLLM_LOCAL_API_KEY:?Set PLLM_LOCAL_API_KEY}"
  [ "$PLLM_API_KEY" != "$PLLM_LOCAL_API_KEY" ] || { echo "Use different provider and gateway credentials" >&2; exit 2; }
  exec pllm sidecar --server "$PLLM_BASE_URL" --api-key "$PLLM_API_KEY" --local-api-key "$PLLM_LOCAL_API_KEY" --host 0.0.0.0 --port 8080 --transport http --correlation-mode bfv
  ;;
 *) exec "$@" ;;
esac
