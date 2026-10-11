# Decode epochs and pending-prefill sharing in the SDK

Two bounded findings from the [five-method screen](../plans/all-link-next-five.md)
now have ordinary component, SDK and benchmark paths. The historical 55.22%
decode-epoch result counts covered stage bodies. It is not a measured saving for
this SDK composition.

## Public contracts

`PreparedInventory("request-sized", rows=1, allocation="demand",
refill="on-demand", decode_epoch_tokens=16)` bounds fresh material to one to
sixty-four output selections per epoch. Select `ClientPrefixReuse` and
`SymmetricPerRow(causal_reduction="prefix_f32")` under the admitted prepared
full-KV composition. Omitted options preserve existing serialized identities.

The first epoch includes prefill. A later epoch completes the current provider
session, closes its channel, burns unused rows and releases online exclusion
before issuing fresh masks/corrections. The full request keeps its compiled token
cap, decoder state, pending greedy or sampled token, RNG and application event
sequence. The next authenticated session must acknowledge the exact continuation.
Natural EOS ends generation normally. Full-cap use pays each extra admission and
pause. Full-cap memory admission is retained.

`ClientPrefixReuse(inflight=True, inflight_max_requests=8,
inflight_max_bytes=64 << 20)` admits matching pending prefills within one client.
Request bounds are 2–32; state-payload bounds are 1–256 whole MiB. Lookup validates
native source/composition/schedule/full-KV metadata before online allocation.
It can reuse a pending producer's complete prompt when that prompt is an exact
token prefix of a caller. It does not speculate arbitrary common-prefix tasks.
Completed-cache hits bypass this owner.

A producer publishes copied, validated KV/logits. Each branch receives independent
state and fresh provider material. The owner prices producer/publication overlap,
branch copies and compiled-capacity state; completed cache, model, scratch and
request metadata remain separate owners. Ready callers enter a bounded FIFO after
prefix lookup. One caller holds the active seat across all epoch pauses, preserving
global idle-only Preparation.

Producer cancellation leaves shared prefill running only while another caller
needs it; it stops that producer's private decode after publication. Waiters can
cancel before a response ID exists. Public stream close, async cancellation and
client shutdown reach those scopes, drain execution, and retire arrays after the
runtime reset. A missing session acknowledgement or constructor failure burns its
reservation. Unconfirmed provider retirement closes further prepared admission.

Verified, retained/hybrid/windowed state and other unimplemented compositions
reject through component and native schedule admission.

## Benchmark contract

`examples/benchmarks/epoch_prefix_sdk.py` constructs four ordinary targets:

* Serial full-cap control and decode-epoch candidate.
* Same-client grouped control and pending-prefill-sharing candidate.

`BenchmarkTarget.response(..., concurrent_requests=2)` admits 2–8 barrier-aligned
callers under a cold native prepared scenario and one repeated private prompt.
Public source lookup and same-cap sizing happen before the barrier, with their
costs charged. No prefill is primed. Scheduling may produce completed-cache hits;
the report does not turn those into in-flight joins.

One aggregate audit window includes source resolution, role launch, client setup,
all branches and cleanup. Shared topology is priced once; additional queued
scratch, stored response snapshots and metadata are priced separately. Metrics
retain cleanup-inclusive client and reaped-child CPU, one client process peak RSS,
actual issued/reserved/claimed/burned/discarded rows, epoch pauses, owner counters
and directed bodies. Non-additive owner gauges are not summed. Per-role group
RSS/CPU attribution, WAN and physical wire remain outside this first group target.

`pllm.topology_body_accounting.v2` adds the four instrumented POST lifecycle-control
counters exactly once, in a separate control phase. Arithmetic prefill/decode
conservation is unchanged. Historical records retain their original covered totals;
missing controls are unknown, and partial counter sets invalidate aggregate totals.
Coverage identities prevent cross-scope traffic ranking. HTTP framing, uninstrumented
control endpoints, correction acknowledgements and checkpoint distribution remain
outside these covered bodies.

Serial targets retain ordinary response metric semantics. Group and serial windows
do not rank together. Output commitments/usage validate completed group requests;
the separate integration tests establish exact logits and KV rather than inferring
them from text equality.

## Verification and replay

Run against an installed wheel, outside overlapping builds or measurements:

```sh
target/frontier-sdk-env/bin/python -I -m pytest -o pythonpath= \
  tests/test_decode_epochs_sdk.py tests/test_inflight_prefix_runtime.py \
  tests/test_all_link_component_contracts.py tests/test_inflight_prefix_sdk.py \
  tests/test_prepared_queue.py tests/test_prepared_execution_queue.py \
  tests/test_benchmark_epoch_prefix.py -q

PLLM_INFLIGHT_SDK_NATIVE=1 target/frontier-sdk-env/bin/python -I -m pytest \
  -o pythonpath= tests/test_inflight_prefix_sdk.py -k generated_compiled_metadata -q

target/frontier-sdk-env/bin/python -I examples/benchmarks/epoch_prefix_sdk.py \
  --output target/epoch-prefix-sdk.json < prompt.txt
```

Generated Qwen2/Qwen3/Llama HTTP/WebSocket controls compare exact saved logits,
active KV, outputs and pending tokens against independent serial execution.
Controls cover stochastic continuity, natural/controlled EOS, no-match and completed
cache paths, cancellation, failed preparation, malformed session acknowledgements,
constructor failure, unconfirmed retirement and client shutdown. Streaming and
non-streaming gateway controls additionally compose shared prefill with multiple
decode epochs, checking exact independent branch state and idle-only Preparation.
The installed-wheel example completes all four targets. Its generated full-cap
epoch pair retains matching outputs and issued rows while charging four sessions
and three pauses. The grouped pair records a genuine pending join, matching
outputs, fewer issued rows, and empty private owners after client shutdown.
Benchmarks keep
private prompts on stdin and archive only salted commitments and bounded counters.

These functionality and accounting checks establish benchmark availability. Matched
real-checkpoint whole-response traffic, latency and CPU/RSS gains require their own
cohort; historical research percentages cannot fill those measurements.
