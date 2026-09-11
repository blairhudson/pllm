# Deployment templates

These files are evaluation and deployment starting points, not evidence of a live
or production-hardened service. The runtime image builds the Maturin package and
keeps the compiler out of its final stage. Validate the image, checkpoint, network,
and optional dependency availability on every target architecture.

## Roles

`compose.yaml` starts three runtime roles on one host:

| Service | Trust placement | Online responsibility |
| --- | --- | --- |
| `gateway` | Customer boundary | Accepts plaintext, owns client state, calls inference |
| `preparation` | Customer boundary or separately trusted operator | None; prepares inventory only while client is idle |
| `provider` | Inference host | Consumes one-time tickets and executes masked public-body stages |

Inference and preparation mount the same public checkpoint. Public client bundles
carry token lookup and output-head matrices; the remote services execute the
transformer body. Never mount gateway state, root seeds, masks, or plaintext logs
into the provider container.

## Environment names

Use four distinct credential values. Similar names reflect process boundaries:

| Compose input | Runtime destination |
| --- | --- |
| `PLLM_API_KEY` | Inference's client credential and gateway's remote credential |
| `PLLM_PREPARATION_API_KEY` | Preparation's `PLLM_API_KEY` and gateway's preparation credential |
| `PLLM_PROVIDER_PUSH_API_KEY` | Inference's push credential and preparation's `PLLM_PUSH_API_KEY` |
| `PLLM_LOCAL_API_KEY` | Application-to-gateway credential |

Preparation also reads `PLLM_INFERENCE_URL`; clients use
`PLLM_PREPARATION_BASE_URL`. Do not configure preparation with
`PLLM_PROVIDER_PUSH_API_KEY` directly unless a wrapper maps it to
`PLLM_PUSH_API_KEY` as Compose does.

Start Compose from the release root with an absolute public checkpoint path:

```bash
export PLLM_MODEL_SOURCE="${PLLM_MODEL_SOURCE:?Set an absolute checkpoint directory}"
export PLLM_MODEL_ID="${PLLM_MODEL_ID:-demo-model}"
export PLLM_API_KEY="${PLLM_API_KEY:?Set the inference client credential}"
export PLLM_PREPARATION_API_KEY="${PLLM_PREPARATION_API_KEY:?Set the preparation client credential}"
export PLLM_PROVIDER_PUSH_API_KEY="${PLLM_PROVIDER_PUSH_API_KEY:?Set the provider push credential}"
export PLLM_LOCAL_API_KEY="${PLLM_LOCAL_API_KEY:?Set the local gateway credential}"
docker compose -f deploy/compose.yaml up --build
```

All published example ports bind to loopback. The single-host topology demonstrates
transport and lifecycle behavior; it does not demonstrate operator non-collusion.

## Prepared inventory network

Before `READY`, the client authorizes an inventory and sends root-seed batches to
preparation. Preparation pushes `W*r-s` one way to inference over
`/v1/he/corrections/ws` and waits for acknowledgements. During chat, preparation
is idle and the gateway communicates only with inference. Prefill uses HTTP packed
stage batches; decode normally uses the persistent `/v1/he/ws/{session_id}`
client-to-inference WebSocket.

Remote deployments need TLS and network policy on both service origins. A reverse
proxy in front of inference must:

- preserve `Authorization` and binary WebSocket subprotocols;
- pass `/v1/he/corrections/ws` and `/v1/he/ws/*` upgrades;
- allow compact prefill HTTP bodies and correction frames up to configured bounds;
- disable request, response, and frame payload logging;
- avoid buffering large binary preparation and prefill payloads;
- keep decode WebSockets open longer than the largest expected gap between online
  stage requests.

A correction WebSocket may close while preparation is idle; the next independent
inventory can reconnect. Its active timeout must still cover connection setup,
one bounded correction send, inference processing, and acknowledgement. Do not
tune it from decode latency because correction transport is offline.

## Distinct timeouts and bounds

Defaults are development starting points:

| Control | Default | Meaning |
| --- | --- | --- |
| Client `PLLM_TIMEOUT` | 300 s | HTTP operations, including model bundle and offline preparation calls |
| Decode WebSocket connect | Fixed 30 s | Client upgrade; not controlled by `PLLM_TIMEOUT` |
| Decode WebSocket receive | No client setting | Bound through external application/proxy policy |
| Preparation `PLLM_PUSH_TIMEOUT` | 10 s | Correction connection/acknowledgement operation |
| Provider `PLLM_RENDEZVOUS_TIMEOUT` | 30 s | Unmatched non-preloaded correction/activation entry; not READY inventory retention |
| Provider `PLLM_PREPARED_SESSION_IDLE` | 300 s | Memory-only inventory idle lifetime |
| Provider `PLLM_RENDEZVOUS_CAPACITY` | 32,768 rows | Aggregate live correction-entry bound |
| Provider `PLLM_RENDEZVOUS_MAX_BYTES` | 256 MiB | Aggregate live correction payload bound |
| Provider `PLLM_PREPARED_SESSION_CAPACITY` | 4,096 | Live inventory/session bound |

Size row and byte bounds together. Approximate correction storage from inventory
rows, every remote stage's output width, and its 2/3/4-byte ring, then include
protocol overhead and concurrent inventories. Larger values increase denial-of-
service exposure and memory use; smaller values fail inventory loading. Set
prepared-session idle above expected between-chat reuse and worst legitimate
between-stage gap, but do not treat longer retention as durability. Restart still
discards all prepared rows.

Starting an execution reserves rows. Cancellation, timeout, failure, or early
completion burns unused reserved rows. Refill runs only when the client is idle.
Explicit HTTP transport uses `PLLM_TIMEOUT` per stage but gives up persistent
decode; default `auto` only falls back when the initial WebSocket upgrade fails.

## Images and systemd

Build static docs separately:

```bash
docker build -f deploy/Dockerfile.site -t pllm-docs .
```

For systemd, install at `/opt/pllm`, copy deployment files to
`/opt/pllm/deploy`, create the `pllm` user, run `uv sync --locked`, and verify
`/opt/pllm/.venv/bin/pllm build` before enabling units. Put provider values in
`/etc/pllm/provider.env` and preparation values in
`/etc/pllm/preparation.env`, both mode `0600`. Keep checkpoints outside protected
home directories.

Runtime image builds require committed `Cargo.lock` and `uv.lock`; static docs
build requires `docs/package-lock.json` and registry access. Pin deployed image
digests after validating locally built artifacts.
