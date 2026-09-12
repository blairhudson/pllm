# Troubleshooting

Diagnose role, inventory, timeout, transport, and checkpoint failures safely.


## Inference rejects `/v1/responses`

An ordinary SDK sent plaintext to the private inference service. Start
`pllm sidecar` inside the client boundary and use
`http://127.0.0.1:8080/v1`. Do not enable a trusted plaintext backend merely to
make port 8000 accept the request.

## A service returns 401

Four channels use distinct values: application-to-sidecar
`PLLM_LOCAL_API_KEY`, client-to-inference `PLLM_API_KEY`, client-to-preparation
`PLLM_PREPARATION_API_KEY`, and preparation-to-inference push credential.

Inside preparation, the listener reads `PLLM_API_KEY` and push reads
`PLLM_PUSH_API_KEY`. Inference reads `PLLM_PROVIDER_PUSH_API_KEY`. Compose maps the
friendlier outer names. Verify role mapping without printing values into logs.

## Public serving fails at startup

`--weights public` requires `--provider-push-api-key` or
`PLLM_PROVIDER_PUSH_API_KEY`. Preparation separately requires its fixed inference
URL and push credential. Both services must load the same public checkpoint and
quantization settings.

## Inventory never reaches READY

Check model/body/stage commitment parity first. Then inspect preparation correction
attempts, uploaded bytes, send duration, and failures alongside inference channel
connections, frames, accepted bytes, and processing failures.

Confirm `/v1/he/corrections/ws` accepts binary WebSocket upgrades, preserves
`Authorization`, negotiates `pllm-correction-v1`, and allows the configured frame
size. Preparation `--push-timeout` must cover send, inference processing, and ACK.
Provider row and aggregate-byte limits must fit the entire live inventory.

Do not retry a partially accepted inventory under reused tickets. Cancel it and
prepare a new independently seeded inventory.

## Inventory was READY but vanished

Inference restart and `--prepared-session-idle` expiry discard memory-only
inventory. Client restart loses seeds and masks. Increase the provider idle value
only when longer in-memory reuse is intended, then prepare again. `PLLM_TIMEOUT`
does not control this lifetime.

## Response says inventory is exhausted

Current direct and sidecar clients calculate the exact row requirement and prepare
missing capacity before opening the online session. If this error persists, verify
that both services and the client are from the same release and that Preparation
can push to Inference. Authenticated `POST /v1/preprocess` remains available for
advance warming while idle. There is no preparation after online execution starts.

## Prefill gets 400 or 413

Compact prefill must fit provider `prepared_stage_batch_rows`, prepared tensor
elements, binary payload bounds, and proxy body limits. Do not split one prepared
batch into replayed row tickets after an ambiguous failure. Raise reviewed bounds
or reduce the supported context/output plan, then prepare fresh inventory.

## Decode disconnects

Check `/v1/he/ws/{session}` upgrade, `he-responses-v1` subprotocol, authorization,
and proxy idle timeout. Decode uses a persistent client-to-inference socket with a
fixed 30-second upgrade timeout and currently no per-receive `PLLM_TIMEOUT`
deadline; preparation should be idle. A dropped or ambiguous execution burns its
remaining reservation. Start a new response with unreserved or newly prepared
rows rather than replaying tickets.

## A stream ends early

Close the failed stream and treat its full reservation as consumed or burned.
Never restore tickets from a snapshot or retry one stage independently. Remaining
unreserved rows can serve a later response if still live.

## TenSEAL is unavailable

The current public seeded-inventory path does not need TenSEAL. If an explicitly
selected confidential/BFV path needs it, install the `he` extra and verify platform
wheel support. Never replace real correlations with `local-test` for private data.

## Native thread behavior differs

Source default is available CPU count capped at 32. Check `pllm build`,
`PLLM_NATIVE_THREADS`, and service `--engine-threads`. More threads can hurt small
dependent stages; benchmark real shapes. Use `--compiled-cache-dir` or
`PLLM_COMPILED_CACHE` for service caches.

## Dashboard does not start

The default checkpoint may need network access, Hugging Face credentials, and
substantial RAM for two model services plus the client bundle. Try
`pllm benchmark dashboard --tiny --no-open` to isolate packaging and transport.
The dashboard can bind only to loopback; choose another local port with `--port`.

## Quantized output differs from source model

Compare private execution with the same clear quantized graph first. Exact integer
parity does not imply equality with floating-point logits or retained language
quality. Schema 2 tied token lookup quantization also differs from schema 1 lookup;
do not compare across bundle schemas without naming that change.
