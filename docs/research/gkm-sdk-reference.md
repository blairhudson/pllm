# Bounded verified-delegation SDK

This slice composes the existing GKM numerical and verification contracts with
ordinary PLLM components, two worker processes and request-end acceptance. It is
a generated-checkpoint conformance path. The official Qwen checkpoint and the
original paper's models remain outside this default Capsule geometry.

The subsequent explicit `OpenWeightVerifiableOutsourcing(capacity="decoder")`
uses larger, compiler-derived owner bounds. Its installed-wheel evidence passes
640 Python and 134 native checks, including generated complete results beyond the
Capsule vocabulary ceiling. Pinned Qwen metadata fits those native bounds, but
its original full-checkpoint attempt is host-memory blocked before execution. A
later [source-present response](../../paper/artifacts/gkm-sdk/qwen-response.md)
passes fresh admission and completes four input tokens plus two verified outputs
through both workers, with zero observed swap growth. Independent full-checkpoint
quality and matched comparison remain open. See the
[Decoder contract](gkm-decoder-capacity.md) and
[separate archive](../../paper/artifacts/gkm-sdk/decoder-capacity.md).

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

The ordinary SDK source loader streams at most 64 KiB of original float32 values
per native call. A one-use compiler owner checks ordered names and offsets, retains
the original source digest and converts weights directly into their final i16
buffers. Lookup and local coefficients keep their original float32 bytes. Workers
take ownership of these buffers instead of copying another complete source image.
Source locks and complete headers revalidate before value reads and publication;
malformed frames, source mutation and cancellation retire the partial import.
A separate 16 MiB process-wide import payload allowance covers simultaneous
encoded weights, table, coefficients and numeric bounds through transfer. Existing
conservative SDK memory admission remains. This removes eager Python float-source
ownership; it does not establish process-RSS savings or enlarge checkpoint limits.

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
The separate [streaming-import archive](../../paper/artifacts/gkm-sdk/source-stream.md)
passes 571 installed-wheel tests and 120 native checks, including complete SDK
execution after ownership transfer. All 362 installed package files match that
wheel. These checks establish correctness and bounded ownership, not RSS savings.

Seeded network frames, stochastic or early-EOS generation, conversation
continuation, remote source discovery and automatic network planning remain
unsupported. Source/publication loading and retained weights remain charged;
this slice makes no thin-client memory claim. Full-width publication, activation
bounds, original-model and common-Qwen quality, matched performance, memory-tier
eligibility and the ten-method publication comparison remain open.
