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
The fixed continuation token bound must cover the initial executed history plus
the requested output cap minus one, even when the response is expected to stop
early. A fresh 36-token prompt with a 64-output allowance therefore needs at
least 99 positions; an expected short answer does not justify a 64-position bound.

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

## Whole-response measurement driver

`scripts/benchmark_epoch_prefix_sdk.py` adds a separate matched measurement over
ordinary installed streaming SDK calls. Four workload cells cover natural early
EOS, full-cap completion, two identical pending prompts and two unrelated prompts.
The grouped cells compare control, epochs, pending sharing and their composition.
No benchmark override changes generation, forces a pending join or truncates a
response early. A declared early/full-cap lifecycle or pending join that does not
occur rejects the cohort.

Every cell starts a fresh client interpreter, two fresh provider processes and an
empty client bundle cache. Shared native 100/40 Mbps caps apply to all service
streams. Public lookup and sizing precede the submission barrier and remain
charged. The installed prepared queue owns execution throughout each response,
including epoch pauses. Private prompts, credentials and the common private
cohort salt travel over stdin; archived records contain only commitments and
resource counters.

The complete-cell window includes provider startup, client imports and setup,
requests, exact-state hashing and process cleanup. It excludes the launcher's
earlier source resolution and topology preflight. Aggregate CPU includes the
launcher/relays and every reaped client/provider child through cleanup. Client
lifetime peak RSS and each provider's sampled peak remain separate; the sampler
starts after provider startup. Provider birth CPU is sampled after client exit,
before provider shutdown. Independent launcher peak RSS is unknown.

Each final response commits every logit and active KV bit, executed history,
pending selection and output. The driver checks natural EOS/cap use, actual
pending joins, zero plaintext prompt/token-ID bytes, idle-only Preparation,
reserved = claimed + burned row conservation, versioned lifecycle-control bodies,
and empty private owners after shutdown. First-text delay includes all queue and
prefix waiting. Request-to-created time includes waiting plus initialization and
admission; isolated queue time remains unknown. N−1 text-delta throughput includes
every epoch pause.

Admission reuses the installed topology/group estimator, adds 256 MiB for the
launcher, sampler and relays, and retains the shared host-pressure/swap guard.
Replicates reverse cell order under the exclusive frontier lock. Installed build,
source lock and measurement sources are frozen throughout. Pre-positioned public
tokenizer and checkpoint distribution, physical wire and Internet RTT remain
outside this local experiment.

The frozen workload uses a 128-position bound for the 64-output early-EOS cell,
and 64 positions for the eight-output single and grouped cells. Every control and
candidate within a workload uses the same bounds. The complete group, retained
responses, queued scratch and launcher are admitted together before provider
launch. Rejected preflights preserve immutable configurations, source/build locks,
per-cell requirements and rejection reasons in an incomplete report with no
response measurements. A later admission rejection preserves earlier completed
cells plus the failing cell's current requirement and headroom. Worker failures
and interruption likewise preserve progress and their error type, without copying
potentially private exception text. These reports remain incomplete.

```sh
target/frontier-sdk-env/bin/python -I -m pytest -o pythonpath= \
  tests/test_epoch_prefix_response_benchmark.py -q
target/frontier-sdk-env/bin/python scripts/benchmark_epoch_prefix_sdk.py \
  --preflight-only --output target/epoch-prefix-sdk-admission.json
target/frontier-sdk-env/bin/python scripts/benchmark_epoch_prefix_sdk.py \
  --processes 3 --output target/epoch-prefix-sdk-response.json
```

The default source is the pinned indexed-tokenizer Experiment in
`target/prepared-frontier-artifacts/indexed.json`. Each invocation requires a new
output path and creates its own private cohort salt. Incomplete or separately
invoked cohorts cannot be pooled into a successful comparison.

### Incomplete real-checkpoint attempt

The [retained Qwen2.5 attempt](../evidence/epoch-prefix-sdk-incomplete-2026-10-06.json)
completed two of 36 planned fresh-client cells. Before the third cell launched,
post-reserve headroom fell to 3.117 GB against its 3.316 GB requirement. The largest
declared cell needs 4.421 GB; the OS/application reserve remains 4 GiB. The report
keeps `complete=false`, the failed admission and the original source/build locks.
Its SHA-256 is
`304b46a8dca3cec2a68edc66da333679cc8a3bb4f545e39dffa9b20a4c5de37a`.

The completed early-EOS pair uses 36 input tokens and three outputs against a
64-output allowance. Its final logits, active KV, history, pending token and
output commitments agree exactly. Both clients retire their private owners and
the cohort observes zero swap growth.

| Recorded quantity | Full-cap material control | Four-output epochs |
| --- | ---: | ---: |
| All-link TCP-stream bytes, decimal MB | 266.096 | 213.389 |
| Covered application bodies, decimal MB | 265.313 | 212.693 |
| Preparation correction bodies, decimal MB | 85.755 | 33.135 |
| Fresh issued stage rows | 9,399 | 3,639 |
| Reserved rows burned unused | 5,760 | 0 |

This one completed pair records 19.81% fewer TCP-stream bytes after charging cold
client-bundle delivery. It finishes within the first epoch, so it does not exercise
an epoch transition. The intended three-replicate confirmation, full-cap overhead,
pending-prefix sharing and combined workload measurements remain blocked. CPU,
latency and separate RSS samples remain in the incomplete raw report; they are
not a completed confirmation cohort. Earlier interrupted attempts retain their own
private cohort identities and are not pooled with this pair.
