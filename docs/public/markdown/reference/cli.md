# CLI reference

Current public-path commands, role-specific flags, and legacy boundaries.


Use `pllm <command> --help` from the installed revision as the authoritative flag
list. Argument abbreviation is disabled.

| Command | Current purpose |
| --- | --- |
| `pllm serve` | Run inference and load checkpoint models |
| `pllm preparation serve` | Run trusted public-weight preparation |
| `pllm configure` | Save client defaults |
| `pllm chat` | Prepare inventory and run interactive local chat |
| `pllm sidecar` | Expose a customer-side Responses gateway |
| `pllm benchmark dashboard` | Start current three-role loopback dashboard |
| `pllm build` | Inspect installed native backend capabilities |
| `pllm security` | Print protocol claim summaries |

`secure`, `market`, and `provider` commands are research/experimental surfaces,
not required for the current public prepared-inventory path.

## Inference

```text
pllm serve MODEL_PATH_OR_HUB_ID --weights public \
  --provider-push-api-key KEY
```

| Option | Meaning |
| --- | --- |
| `--model-id` | Public model identity; supply once for each model source |
| `--weights public\|confidential` | Required checkpoint disclosure policy |
| `--client-trust honest\|guarded\|untrusted` | Client assumption for confidential paths |
| `--activation-protection` | Explicit protocol override |
| `--api-key` | Client-to-inference credential; generated and printed if omitted |
| `--provider-push-api-key` | Required preparation-to-inference credential for public serving |
| `--host`, `--port` | Bind address; defaults `127.0.0.1:8000` |
| `--revision`, `--hf-token`, `--hf-cache-dir` | Hub resolution controls |
| `--local-files-only` | Reject missing local checkpoint files instead of downloading |
| `--compiled-cache-dir` | Compiled matrix cache |
| `--quantization-chunk-rows` | Maximum compile working rows; default 64 |
| `--weight-bits`, `--activation-bits` | Quantization widths; default 8/8 |
| `--engine-threads` | Persistent native executor threads, 1 through 32 |
| `--max-batch-size`, `--max-wait-ms`, `--fixed-batch-wait` | Cross-request provider batching |
| `--allow-test-correlations` | Insecure test mode; never use for private data |

Public weights allow only `seeded-preparation`. Confidential weights can select
`precomputed-masks`, `guarded-blinded-masks`, `blinded-masks`, or
`encrypted-activations` where implemented. `authenticated-shares` fails closed for
arbitrary Hugging Face graphs.

## Prepared inventory controls

| Inference option | Environment | Default | Scope |
| --- | --- | --- | --- |
| `--rendezvous-timeout` | `PLLM_RENDEZVOUS_TIMEOUT` | 30 s | Unmatched non-preloaded entry |
| `--rendezvous-capacity` | `PLLM_RENDEZVOUS_CAPACITY` | 32,768 | Aggregate live correction rows |
| `--rendezvous-max-bytes` | `PLLM_RENDEZVOUS_MAX_BYTES` | 256 MiB | Aggregate live correction payload |
| `--prepared-session-capacity` | `PLLM_PREPARED_SESSION_CAPACITY` | 4,096 | Live inventory/session count |
| `--prepared-session-idle` | `PLLM_PREPARED_SESSION_IDLE` | 300 s | Memory-only inventory idle expiry |

Rendezvous timeout is not READY inventory retention. Size row and byte limits
together from model stages, ring widths, inventory rows, and concurrency.

## Preparation

```text
pllm preparation serve MODEL_PATH_OR_HUB_ID \
  --api-key KEY \
  --inference-url URL \
  --push-api-key KEY
```

This role loads public models only. Its listener credential uses `PLLM_API_KEY`;
fixed inference origin uses `PLLM_INFERENCE_URL`; push credential uses
`PLLM_PUSH_API_KEY`. `--push-timeout` / `PLLM_PUSH_TIMEOUT` defaults to 10 seconds
for correction connection/send/ACK operations. Its default bind is
`127.0.0.1:8001`.

Inference expects the same push value under `PLLM_PROVIDER_PUSH_API_KEY`. Do not
reuse either service-facing credential for another channel.

## Configure

```text
pllm configure --server URL --api-key KEY \
  --preparation-url URL --preparation-api-key KEY --model MODEL_ID
```

Current public controls include `--transport`, `--prepared-inventory-rows`,
`--bundle-cache-mode`, `--bundle-cache-dir`, and `--timeout`. The corresponding
environment values are `PLLM_TRANSPORT`, `PLLM_PREPARED_INVENTORY_ROWS`,
`PLLM_BUNDLE_CACHE_MODE`, `PLLM_BUNDLE_CACHE_DIR`, and `PLLM_TIMEOUT`.

`PLLM_TIMEOUT` applies to direct-client HTTP operations. Persistent decode uses a
fixed 30-second WebSocket connect timeout and currently has no per-receive
`PLLM_TIMEOUT` deadline. Proxy WebSocket idle settings are separate.

`--correlation-mode`, `--correlation-prefetch`, and `--token-cache-size` remain for
legacy confidential-weight/BFV paths. They do not tune public seeded inventory or
local public token lookup.

## Chat and sidecar

```text
pllm chat [--server URL] [--preparation-url URL] [--model MODEL_ID]
pllm sidecar --local-api-key KEY --host 127.0.0.1 --port 8080
```

Chat prepares before prompting and calculates required rows before every response.
`--no-stream` changes rendering; `--max-output-tokens` also changes row
reservation.

Sidecar reads saved service settings, prepares its default public model during
startup, and exposes `/v1/responses`, `/v1/models`, retrieval/cancellation, and
authenticated `/v1/preprocess`. `auto` selects persistent WebSocket decode;
explicit `http` is a compatibility choice for networks without client WebSocket
support.

## Dashboard

```text
pllm benchmark dashboard [--model SOURCE] [--model-id ID] \
  [--max-output-tokens N] [--history-db PATH] [--no-open]
```

Defaults are loopback port 8791 and `Qwen/Qwen2.5-0.5B-Instruct`. `--tiny` creates
random tiny weights for transport smoke only. Non-loopback dashboard binds are
rejected. `--history-db` selects the insert-only SQLite run store; `:memory:` keeps
history ephemeral. See [metric definitions](/docs/reference/dashboard).
