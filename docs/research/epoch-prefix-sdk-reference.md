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

These functionality and accounting checks establish benchmark availability. The
separate whole-response cohort below measures the installed contracts; historical
research percentages cannot fill those measurements.

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
pending-prefix sharing and combined workload measurements were blocked in this
attempt. CPU,
latency and separate RSS samples remain in the incomplete raw report; they are
not a completed confirmation cohort. Earlier interrupted attempts retain their own
private cohort identities and are not pooled with this pair.

### Completed real-checkpoint confirmation

After host memory was freed, a [new independent cohort](../evidence/epoch-prefix-sdk-qwen25-2026-10-06.json)
completed all 36 cells: three alternating replicates of twelve configurations.
No workload, output allowance, memory estimate, reserve or swap guard changed.
Each cell has a fresh installed SDK client, two fresh provider processes and empty
bundle cache under shared native 100/40 Mbps caps. The 60 final response states
match their workload controls exactly, including full logits, active KV, executed
history, pending selection and output. Every cell passes row conservation, private
boundary, idle-only Preparation and shutdown checks. Observed swap growth is zero;
minimum available host memory is 14.228 GB before the unchanged 4 GiB reserve.

The raw report's SHA-256 is
`ccd5469f114bcfe7c5000a8debdc80c66586deb507c38d15399b6bfa31a05c62`.
Its installed build digest is
`010b3fd6008c72488d1633b013e45735f391103818e0e3ac6224600737e55d44`;
source lock and all five frozen measurement-file hashes match. Earlier incomplete
attempts retain their own evidence and private cohort salts.

#### Workloads and actual material

* **Early:** 36 input tokens, three natural-EOS outputs, allowance 64. Four-output
  epochs complete within the first epoch. Fresh issued stage rows fall from 9,399
  to 3,639; unused reserved rows burned fall from 5,760 to zero. Correction bodies
  fall from 85.755 to 33.135 MB. This cell exercises early completion, not an epoch
  transition.
* **Full:** 54 input tokens and all eight allowed outputs. Both layouts issue
  5,697 stage rows. Epochs execute two sessions and one approximately 1.11-second
  pause per response; correction bodies rise from 51.830 to 51.928 MB.
* **Shared:** two simultaneous identical 54+8 requests in one client. Sharing
  records one genuine pending join per cell, one producer and two exact independent
  final states. Issued rows fall from 11,394 to 6,369; correction bodies fall from
  103.659 to 58.067 MB. Combining sharing with epochs retains the same row count,
  adds two pauses across the group and uses 58.263 MB of correction bodies.
* **Unrelated:** simultaneous 54+8 and 51+8 requests. Every option records zero
  joins and issues 11,115 stage rows. Sharing alone leaves correction bodies at
  101.128 MB; epochs raise them to 101.325 MB. No unrelated-prefix saving is claimed.

#### Measured resource and traffic medians

Each row is the median of three fresh cells. MB is decimal. Client RSS is the
process-lifetime peak; Preparation RSS is the 50 ms sampled peak after startup.
Preparation CPU is cumulative from process birth through client exit. All-link
TCP streams charge cold client-bundle delivery, arithmetic, controls and shutdown;
covered application bodies retain their narrower versioned scope.

| Workload / option | TCP streams MB | Covered bodies MB | Client RSS MB | Client CPU s | Preparation RSS MB | Preparation CPU s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Early / control | 266.096 | 265.313 | 130.43 | 6.27 | 166.07 | 6.30 |
| Early / epochs | 213.389 | 212.693 | 132.25 | 6.47 | 140.07 | 5.26 |
| Full / control | 263.413 | 262.318 | 132.83 | 7.00 | 147.55 | 5.62 |
| Full / epochs | 263.564 | 262.449 | 137.71 | 7.20 | 143.05 | 5.74 |
| Shared / control | 401.882 | 399.731 | 140.48 | 10.79 | 147.03 | 7.14 |
| Shared / epochs | 402.183 | 399.992 | 148.36 | 11.36 | 145.24 | 7.43 |
| Shared / sharing | 280.726 | 278.821 | 151.45 | 10.98 | 149.90 | 5.95 |
| Shared / combined | 281.027 | 279.083 | 152.42 | 11.45 | 143.47 | 6.13 |
| Unrelated / control | 395.173 | 393.022 | 149.67 | 10.73 | 158.32 | 7.09 |
| Unrelated / epochs | 395.474 | 393.283 | 161.04 | 11.27 | 154.12 | 7.39 |
| Unrelated / sharing | 395.173 | 393.022 | 161.61 | 12.44 | 156.86 | 7.11 |
| Unrelated / combined | 395.474 | 393.283 | 164.33 | 12.83 | 153.89 | 7.34 |

#### Measured time and complete CPU medians

First-text delay includes queue/prefix waiting and request setup. For grouped
cells, first-text delay and N−1 text-delta throughput first take the median of
both responses, then the median across three cells. They are neither summed
throughput nor a group-tail latency. Group time includes both responses and their
state audits after client initialization. Complete-cell time additionally charges
provider startup, client loading/setup and cleanup. Aggregate CPU charges launcher,
relays and all reaped client/provider children through cleanup. Separately sampled
Inference CPU/RSS and individual response timings remain in the raw report.

| Workload / option | First text s | N−1 decode tokens/s | Group time s | Complete cell s | Aggregate CPU s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Early / control | 28.549 | 4.558 | 29.198 | 65.533 | 25.826 |
| Early / epochs | 18.096 | 4.935 | 18.708 | 55.048 | 23.199 |
| Full / control | 26.185 | 5.014 | 27.587 | 63.945 | 25.718 |
| Full / epochs | 25.577 | 3.023 | 27.907 | 64.178 | 26.320 |
| Shared / control | 40.374 | 4.950 | 55.264 | 92.275 | 36.737 |
| Shared / epochs | 39.885 | 3.009 | 55.795 | 92.882 | 37.840 |
| Shared / sharing | 30.120 | 4.969 | 33.129 | 70.144 | 31.008 |
| Shared / combined | 29.624 | 3.015 | 33.664 | 70.638 | 31.850 |
| Unrelated / control | 39.277 | 4.987 | 54.145 | 91.139 | 36.478 |
| Unrelated / epochs | 38.744 | 3.021 | 54.663 | 91.682 | 37.439 |
| Unrelated / sharing | 40.861 | 4.970 | 55.717 | 92.708 | 38.218 |
| Unrelated / combined | 40.310 | 3.010 | 56.199 | 93.197 | 39.035 |

#### Decision

For this early-EOS workload, epochs reduce TCP-stream traffic **19.81%**, complete
cell time **16.00%**, aggregate CPU **10.17%** and Preparation sampled RSS
**15.66%**. Client peak RSS instead rises **1.39%**. Full-cap execution saves no
material rows: TCP streams grow **0.057%**, aggregate CPU **2.34%** and client RSS
**3.68%**. Its charged epoch pause lowers N−1 decode throughput **39.70%** despite
the nearly unchanged complete-cell time.

For two identical pending prompts, sharing reduces TCP streams **30.15%**,
complete-cell time **23.98%**, first-text delay **25.40%** and aggregate CPU
**15.60%**. Client peak RSS rises **7.81%** and client CPU **1.80%**. Combining
epochs and sharing saves **30.07%** traffic against the grouped control but keeps
the full-cap pause penalty. Unrelated sharing saves no material and raises client
CPU **15.92%**, peak RSS **7.98%** and complete-cell time **1.72%**.

These measurements support explicit workload-dependent selection: bound unused
output material when early completion warrants the pause tradeoff, and share
genuine same-client pending prefixes when the extra state fits. They do not
justify enabling either option universally or a client-memory improvement.
This three-replicate, one-host cohort retains the measurement boundaries above:
cached checkpoint and pre-positioned tokenizer distribution, physical wire,
Internet RTT, independent operators and representative generation quality remain
outside the result.
