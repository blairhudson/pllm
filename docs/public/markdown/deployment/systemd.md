# Deploy with systemd

Install separate inference and preparation services with explicit caches and secrets.


## Install

Install this source at `/opt/pllm`, create a dedicated `pllm` user, and build the
locked environment before enabling read-only service protection:

```bash
cd /opt/pllm
uv sync --locked
/opt/pllm/.venv/bin/pllm build
```

Copy `deploy/pllm-provider.service` and
`deploy/pllm-preparation.service` to `/etc/systemd/system/`. The start scripts pass
actual `--compiled-cache-dir` values under `/var/cache/pllm` and
`/var/cache/pllm-preparation`.

Current runtime cache control is `PLLM_COMPILED_CACHE` or
`--compiled-cache-dir`. The explicit role-specific cache arguments in the supplied
start scripts are authoritative.

## Configure role-specific environment

Create `/etc/pllm/provider.env` from `deploy/provider.env.example` and
`/etc/pllm/preparation.env` from `deploy/preparation.env.example`.

Provider environment:

- `PLLM_API_KEY` authenticates clients.
- `PLLM_PROVIDER_PUSH_API_KEY` authenticates preparation correction pushes.
- `PLLM_RENDEZVOUS_*` bounds live correction storage and non-preloaded matching.
- `PLLM_PREPARED_SESSION_IDLE` and `PLLM_PREPARED_SESSION_CAPACITY` can override
  memory-only inventory lifetime/capacity when added.

Preparation environment:

- `PLLM_API_KEY` authenticates client seed requests and must differ from provider.
- `PLLM_INFERENCE_URL` is the fixed inference HTTPS origin.
- `PLLM_PUSH_API_KEY` equals only inference's provider-push value.
- `PLLM_PUSH_TIMEOUT` bounds correction send and acknowledgement.

Both roles need the same public checkpoint and model ID. Keep preparation inside
the customer boundary or at a separately trusted non-colluding operator.

```bash
sudo chmod 600 /etc/pllm/provider.env /etc/pllm/preparation.env
sudo systemctl daemon-reload
sudo systemctl enable --now pllm-provider pllm-preparation
sudo systemctl status pllm-provider pllm-preparation
```

## Network and lifetime

Supplied units bind services to loopback. Put explicitly configured TLS proxies in
front of remote endpoints. Preserve `Authorization` and WebSocket subprotocols for
`/v1/he/corrections/ws` and `/v1/he/ws/*`; allow bounded binary bodies; disable
payload logs and buffering. Decode WebSocket idle timeout must exceed legitimate
online gaps. Correction push timeout covers offline send/ACK, not decode.

Systemd restart discards READY inventory. The same is true after provider
prepared-session idle expiry. Clients must prepare again before opening an online
response. Never persist or restore tickets, root seeds, masks, or reservations to
work around restart loss.

These units provide baseline process isolation controls, not a production
certification. Review filesystem paths, capabilities, outbound network policy,
resource limits, telemetry, proxy behavior, and secret rotation for the target
host.
