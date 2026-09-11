# Deployment overview

Place three trust roles, route two online transports, and size inventory safely.


## Place roles by data, not convenience

| Component | Plaintext or seed access | Intended placement |
| --- | --- | --- |
| Client SDK or local gateway | Prompt, tokens, activations, root seeds, masks, decoded output | Customer-controlled host |
| Trusted preparation | Stage root seeds before chat; public transformer body | Customer host or separately trusted non-colluding operator |
| Inference | Prepared corrections and masked online values; public transformer body | Compute provider |
| Static documentation | No runtime input | Independent static host |

The privacy claim assumes preparation and inference follow the protocol and do not
collude. Putting both roles under one untrusted operator does not preserve that
claim. Self-hosting preparation keeps seed-side trust in the customer boundary.

## Lifecycle and routes

Before chat, client-to-preparation HTTP carries inventory authorization and root
seed batches. Preparation pushes corrections one way to inference over
`/v1/he/corrections/ws` and waits for acknowledgements. Inference seals the
complete inventory `READY`.

Online, the client contacts inference only:

- Prefill uses binary HTTP on `/v1/he/sessions/{session}/stages/{stage}` with a
  compact ticket vector and packed masked matrix.
- Decode normally uses persistent `/v1/he/ws/{session}` with one ticket per stage.

Preparation stays idle. Refill can start only between online responses.

## TLS and proxy behavior

Loopback HTTP is accepted for local evaluation. Remote preparation and inference
must use distinct HTTPS origins and credentials. Reverse proxies in front of
inference must preserve `Authorization`, pass both WebSocket routes and their
subprotocols, disable binary payload logging, avoid buffering large bodies, and
enforce explicit frame/body bounds.

Set HTTP timeouts high enough for compact prefill and WebSocket idle timeout above
the longest legitimate gap between decode stage requests. The correction socket
can reconnect for a later inventory after an idle close; its active lifetime must
cover one correction send, inference processing, and acknowledgement.

## Independent timeout controls

| Owner | Control | Source default | Scope |
| --- | --- | --- | --- |
| Client | `PLLM_TIMEOUT` | 300 s | Client HTTP, bundle, and offline preparation requests |
| Client | fixed in runtime | 30 s | Decode WebSocket upgrade only |
| Client | no setting | No application deadline | Decode WebSocket receive; bound with external application/proxy policy |
| Preparation | `--push-timeout` / `PLLM_PUSH_TIMEOUT` | 10 s | Correction connection/send/ACK |
| Inference | `--rendezvous-timeout` / `PLLM_RENDEZVOUS_TIMEOUT` | 30 s | Unmatched non-preloaded entries |
| Inference | `--prepared-session-idle` / `PLLM_PREPARED_SESSION_IDLE` | 300 s | Memory-only READY inventory lifetime |
| Proxy | HTTP/WebSocket settings | None supplied by PLLM | External transport lifetime |

Do not solve an inventory-expiry error by only extending client HTTP timeout.
Do not keep the offline correction socket open for decode traffic; it has no
online role.

Default `auto` transport attempts persistent decode and falls back to HTTP only if
the initial WebSocket upgrade fails. Explicit HTTP transport gets per-stage client
HTTP deadlines but gives up persistent decode.

## Capacity and restart

Inventory is never durable. Client or inference restart and inference idle expiry
discard it. Starting a response reserves its row range. Cancellation, failure,
timeout, or early completion burns every unused reserved row. Unreserved rows may
serve later responses.

Provider defaults allow 32,768 live correction entries, 256 MiB aggregate
correction payload, and 4,096 live prepared sessions. These are safety bounds, not
universal model sizing. Estimate rows times every remote stage's output width and
2/3/4-byte ring, then include concurrent inventories and overhead. Tune row and
byte limits together and monitor rejection counters.

## Templates

Use [Docker](/docs/deployment/docker) for the bundled one-host evaluation or
[systemd](/docs/deployment/systemd) for separate Linux services. Templates have
not been certified as a production deployment. Verify target architecture,
checkpoint quality, secrets, TLS, limits, logging, monitoring, restart behavior,
and non-collusion placement.

The [documentation site](/docs/deployment/site) is a static export and never acts
as a gateway. Browser search uses a static local index.
