# Bounded verified-delegation SDK

This slice composes the existing GKM numerical and verification contracts with
ordinary PLLM components, two worker processes and request-end acceptance. It is
a generated-checkpoint conformance path. The official Qwen checkpoint and the
original paper's models remain outside its admitted geometry.

## Configuration and execution

`pllm.profiles.OpenWeightVerifiableCpu` selects these immutable components:

- `OpenWeightVerifiableOutsourcing(sharing="full", generation="fixed_length_greedy")`;
- `PublishedLinearVerify(acceptance="request")`;
- `TwoOnlineDelegationRoles()`;
- one-thread CPU kernels and `ClientLocalKv()`.

The composition uses the existing Pipeline, Experiment, source locks, role
supervisor, `OpenAI`/`AsyncOpenAI`, local gateway, CLI benchmark and
`BenchmarkTarget.response` paths. `examples/delegated_sdk.py` is the public driver.
The native compiler validates the entire saved request envelope before graph
construction. A smaller actual request cannot conceal an oversized saved budget.

Sources must be complete local Hugging Face-layout or Safetensors directories,
including a genuine `tokenizer.json`. Vocabulary is at most 256; tensor, graph,
publication and window limits also apply. Default `Model.tiny()` has 258 entries
and rejects without modification. Remote source resolution is not admitted.
Source F32/F16/BF16 values become original finite little-endian float32 values;
lookup preserves their bits, including signed zero. The loader bounds metadata,
tokenizer padding and encoded input before tensor execution.

The numerical path retains i16/Q11 source coefficients, checked ring64 products,
the original float32-before-division reconstruction, post-reconstruction bias,
complete head and full client-local KV. The common semantic local executor runs
attention and nonlinear operations. It does not substitute W8A8 source weights.
See [decoder reference](gkm-decoder-reference.md) and
[delegation capsule](gkm-delegation-capsule.md).

## Ownership and acceptance

The client owns issuance, reconstruction and final verification. Each worker
imports its own source-checked publication and receives only its party-local
frames. Native frames bind the publication, request, worker, phase and schedule.
Fresh process epochs and bounded one-use sessions reject substitution and replay.
Busy admissions and stale request IDs cannot retire another active request.

Tentative logits and KV stay private to request execution. No model-derived text,
stream event or reusable state is released before the final native verifier
accepts every scheduled claim. Workers' completion receipts are not acceptance.
Fixed-count greedy generation does not stop at EOS. An accepted capped response
has status `incomplete` and reason `max_output_tokens`.

The SDK supports synchronous, asynchronous and buffered streaming delivery.
Cancellation signals outstanding work before draining it, including an unstarted
stream, concurrent `aclose()` and a disconnected gateway. Native owners retire
on conversion errors, malformed frames, verification failure and cancellation.
Python call errors that occur before native method dispatch cannot retire a
native owner; dispatched signature errors do retire it.

The bounded transport uses distinct authenticated origins and credentials. HTTP
is limited to loopback. Other endpoints require verified HTTPS and IP-literal
origins. HTTP `localhost` is normalized to loopback; DNS names reject before I/O.
A request-owned selector thread keeps synchronous calls independent of the
caller's event loop. Absolute deadlines bound exchanges and retirement; the
transport disallows executor work. Its owner and stack are included in the
existing 64 MiB outer transport allowance. Native logical caps, caller buffers,
process RSS and conservative host admission remain distinct accounting scopes.

Co-located children demonstrate role separation, not operator independence.
Confidential transport, non-collusion and the paper's verification assumptions
retain their separate contracts. This implementation review is not independent
cryptographic certification.

## Measurement and remaining gates

First accepted text includes the final verification barrier. Buffered text release
is not model decode speed: generation and N−1 text-event throughput remain null
with an explicit reason. Reports retain semantic execution, verification, actual
application bodies, worker timing and resource owners separately. Unavailable
failed-call timing remains unknown rather than fabricated zero.

The local generated fixtures test lookup bytes, full logits/KV, history and
pending selection; real process controls test final corruption, replay, partial
admission and cancellation. SDK, gateway and both ordinary benchmark entry points
share the same admitted Experiment. Validation receipts and replay are retained
in [the SDK archive](../../paper/artifacts/gkm-sdk/README.md).

Seeded network frames, stochastic or early-EOS generation, conversation
continuation, remote source discovery and automatic network planning remain
unsupported. Source/publication loading and retained weights remain charged;
this slice makes no thin-client memory claim. Full-width publication, activation
bounds, original-model and common-Qwen quality, matched performance, memory-tier
eligibility and the ten-method publication comparison remain open.
