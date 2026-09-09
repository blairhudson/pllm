#!/bin/sh
set -eu
: "${PLLM_API_KEY:?Set PLLM_API_KEY}"
case "${1:-provider}" in
 provider)
  : "${PLLM_MODEL_SOURCE:?Set PLLM_MODEL_SOURCE to a mounted checkpoint}"
  : "${PLLM_PROVIDER_PUSH_API_KEY:?Set dedicated provider push credential}"
  [ "$PLLM_API_KEY" != "$PLLM_PROVIDER_PUSH_API_KEY" ] || { echo "Use different client and push credentials" >&2; exit 2; }
  exec pllm serve "$PLLM_MODEL_SOURCE" --weights public --model-id "${PLLM_MODEL_ID:-private-model}" --host 0.0.0.0 --port 8000 --local-files-only --compiled-cache-dir /var/cache/pllm --provider-push-api-key "$PLLM_PROVIDER_PUSH_API_KEY"
  ;;
 preparation)
  : "${PLLM_MODEL_SOURCE:?Set PLLM_MODEL_SOURCE to a mounted checkpoint}"
  : "${PLLM_INFERENCE_URL:?Set fixed inference URL}"
  : "${PLLM_PUSH_API_KEY:?Set dedicated provider push credential}"
  [ "$PLLM_API_KEY" != "$PLLM_PUSH_API_KEY" ] || { echo "Use different preparation client and push credentials" >&2; exit 2; }
  exec pllm preparation serve "$PLLM_MODEL_SOURCE" --api-key "$PLLM_API_KEY" --model-id "${PLLM_MODEL_ID:-private-model}" --host 0.0.0.0 --port 8001 --local-files-only --compiled-cache-dir /var/cache/pllm --inference-url "$PLLM_INFERENCE_URL" --push-api-key "$PLLM_PUSH_API_KEY"
  ;;
 gateway)
  : "${PLLM_BASE_URL:?Set PLLM_BASE_URL to the provider URL}"
  : "${PLLM_PREPARATION_BASE_URL:?Set PLLM_PREPARATION_BASE_URL}"
  : "${PLLM_PREPARATION_API_KEY:?Set PLLM_PREPARATION_API_KEY}"
  : "${PLLM_LOCAL_API_KEY:?Set PLLM_LOCAL_API_KEY}"
  [ "$PLLM_API_KEY" != "$PLLM_LOCAL_API_KEY" ] || { echo "Use different provider and gateway credentials" >&2; exit 2; }
  [ "$PLLM_API_KEY" != "$PLLM_PREPARATION_API_KEY" ] || { echo "Use different provider and preparation credentials" >&2; exit 2; }
  exec pllm sidecar --server "$PLLM_BASE_URL" --api-key "$PLLM_API_KEY" --preparation-url "$PLLM_PREPARATION_BASE_URL" --preparation-api-key "$PLLM_PREPARATION_API_KEY" --local-api-key "$PLLM_LOCAL_API_KEY" --host 0.0.0.0 --port 8080 --transport http
  ;;
 *) exec "$@" ;;
esac
