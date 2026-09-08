# Deploy with Docker

A provider container and a separate customer gateway.


The supplied Compose file uses a local model snapshot, a read only model mount, and a writable compiled cache. It binds both service ports to host loopback. Put a TLS reverse proxy in front of the provider when it must be reached from another machine.

## Build and start

From the release root, set `PLLM_MODEL_SOURCE` to an absolute model directory and set both credentials.

```bash
: "${PLLM_MODEL_SOURCE:?Set the absolute checkpoint directory}"
: "${PLLM_API_KEY:?Set the provider credential}"
: "${PLLM_LOCAL_API_KEY:?Set a different local gateway credential}"
docker compose -f deploy/compose.yaml up --build
```

The Dockerfile installs the supplied runtime with UV. It does not pull an unrelated registry package named `pllm`. The build needs package network access; image build and HE wheel compatibility must be tested on the target architecture.

## Keep the gateway trusted

The example runs both services on one host for evaluation. On separate machines, run the gateway on the customer's host and point it at the provider's TLS URL. Do not place both in a provider controlled environment and retain the same confidentiality claim.

## Data and restart behaviour

Keep compiled public matrices on a persistent volume. Do not persist and restore consumed correlation inventory as though it were reusable model cache. Key and session changes invalidate preparation. Reconnecting after a failed stage must not resend a private activation with a previously used mask.

The current reference has no complete snapshot rollback defence. Avoid VM snapshot restore for active private sessions and treat restarted sessions as fresh work.
