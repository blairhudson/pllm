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
through both workers, with zero observed swap growth. Broader full-checkpoint
quality and matched comparison remain open. A separate three-prompt
[quality smoke](../../paper/artifacts/gkm-sdk/quality-smoke.json) already records
bounded FP32-reference agreement; it is not complete-state equality. See the
[Decoder contract](gkm-decoder-capacity.md) and
[original preflight](../../paper/artifacts/gkm-sdk/decoder-capacity-v1/qwen-smoke-v2/preflight.json).

## Configuration and execution

`pllm.profiles.OpenWeightVerifiableCpu` selects these immutable components:

- `OpenWeightVerifiableOutsourcing(sharing="full" | "seeded", generation="fixed_length_greedy")`;
- `PublishedLinearVerify(acceptance="request")`;
- `TwoOnlineDelegationRoles()`;
- one-thread CPU kernels and `ClientLocalKv()`.

The composition uses the existing Pipeline, Experiment, source locks, role
supervisor, `OpenAI`/`AsyncOpenAI`, local gateway, CLI benchmark and
`BenchmarkTarget.response` paths. `examples/delegated_sdk.py` is the public driver.
The native compiler validates the entire saved request envelope before graph
construction. A smaller actual request cannot conceal an oversized saved budget.

Sources must resolve to complete local Hugging Face-layout or Safetensors snapshots,
including a genuine `tokenizer.json`. Cached Hub sources retain their pinned revision.
Default Capsule vocabulary is at most 256; tensor, graph,
publication and window limits also apply. Default `Model.tiny()` has 258 entries
and rejects without modification. Remote source discovery is not admitted.
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

### Explicit seeded transport

`sharing="seeded"` selects the existing
`gkm.seeded.aes128.duplicate-le64.two-purpose.window.v1` native profile through the
same SDK and both capacity envelopes. Each fresh request window owns independent
linear and lookup AES-128 keys. Only Worker One receives those keys, once during
authenticated admission, and expands its own shares from header-only operation
frames. Worker Zero receives complete complementary shares. Counter packing,
continuous stream tails and purpose separation retain the original seeded
reference contract. No private seed or cursor becomes an Experiment parameter.

Request, source, complete schedule, role and sharing commitments precede expansion.
Malformed frames, changed profiles, replay, cancellation and incomplete stream
consumption retire the native window. Complete request-end verification remains
the acceptance barrier. Both input profiles preserve the same numerical graph;
the seeded profile adds its explicit AES PRG assumption to benchmark privacy
identity. Ordinary comparisons retain measured costs but do not rank full and
seeded cells as identical privacy contracts.

Key delivery to Worker One requires confidentiality as well as authentication;
the existing external HTTPS and trusted loopback boundaries remain. Native stream,
key, scratch and frame owners remain admitted before allocation. Python frame
copies retain their outer allowance; native zeroization cannot erase caller-owned
copies. Owner/process-local replay enforcement does not provide durable restart
accounting. Seeded transport is separate from batched prefill and accelerated
verification, neither of which is supplied by this selection.

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
share the same admitted Experiment. Ordinary tests under `tests/test_delegated_*`
cover these contracts. The streaming-import slice passed 571 installed-wheel
tests and 120 native checks, including complete SDK
execution after ownership transfer. All 362 installed package files match that
wheel. These checks establish correctness and bounded ownership, not RSS savings.

`BenchmarkTarget.response(..., state_audit=True)` additionally records salted
commitments for every selection's complete logits, active full KV, executed
history, positions/cache lengths and selected/pending token. The GKM-specific
`pllm.response_state_audit.gkm.v1` schema records fixed-count argmax with **no
sampler RNG**. Tentative hashes remain internal until final native acceptance,
request cleanup and successful benchmark checks; failed targets publish no state
commitments. Full and seeded arms each price the same 32 MiB audit allowance, and
cryptographic randomness remains fresh. The schema and measurement identity stay
distinct from prepared-response audits.

The independent test oracle in `tests/gkm_decoder_oracle.py` reads at most 128
source rows at once, uses separate Q11 weight and activation domains, and performs
exact float64 integer chunks followed by checked int64 accumulation. It preserves
the single integer-to-float32-before-division boundary. Ordinary tests compare
complete generated Qwen2/Qwen3/Llama logits, active KV, history and selections,
including independent per-selection salted commitments and both sharing choices.
The optional pinned-Qwen test adds a separately admitted 512 MiB reference owner.

The installed seeded-transport/audit slice passes 792 Python and 100 focused native
checks, with the two optional Python checkpoint cases deselected. Generated
two-process SDK, gateway and benchmark checks cover both sharing profiles.
Release Clippy, formatting, API
coverage and the 43 documentation checks pass. These checks do not supply a
full-checkpoint equality or performance result.

The reusable `examples/benchmarks/gkm_sharing.py` uses ordinary response targets,
three balanced fresh clients per sharing choice, complete accepted-state audits
and native shared 100/40 Mbps access caps. It retains fixed-count greedy output,
source-present workers and no added RTT; setup/loading and all directed response
bodies remain charged. Full and seeded privacy identities stay separate.

The first pinned-Qwen attempt has six **blocked** cells in the ordinary
[report](../evidence/gkm-sharing-qwen-blocked.json): whole-topology admission needs
10.82 GiB against 8.48–9.87 GiB available. No provider launches, request outputs,
state checks or response timings result. The separate two-profile checkpoint
oracle attempt likewise rejects before value loading: 11.25 GiB required versus
9.41–9.53 GiB available. These are host-admission outcomes, not source-model failures
or measured peak memory. Standard replay:

```sh
uv run --no-sync python -m examples.benchmarks.gkm_sharing \
  --output target/gkm-sharing-qwen.json
PLLM_RUN_GKM_CHECKPOINT=1 uv run --no-sync python -m pytest -o pythonpath=. \
  tests/test_gkm_checkpoint_conformance.py::test_pinned_qwen_checkpoint_two_role_sdk_complete_accepted_state \
  --basetemp=target/gkm-checkpoint -q
```

Stochastic or early-EOS generation, conversation
continuation, remote source discovery and automatic network planning remain
unsupported. Source/publication loading and retained weights remain charged;
this slice makes no thin-client memory claim. Broader activation/state qualification,
original-model and held-out common-Qwen quality, matched performance, memory-tier
eligibility and the ten-method publication comparison remain open.
