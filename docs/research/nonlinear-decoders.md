# Experimental masked-wire decoder

`MaskedWireGatedMultiplyQ12(layout="unfused" | "fused")` executes complete
SiLU × up regions through `TwoOnlineOffsetCpu` Experiments. The compiler binds
each semantic pair and its input/output conversion. The trusted client floors
float32 inputs to signed Q24, enforces public gate/up ranges, issues fresh keys
in chunks of at most 256 lanes, relays worker peer frames and reconstructs Q12
outputs. Attention and other local operations keep their existing ownership.

Each authenticated worker holds one party's keys. Sessions bind plan, region,
incarnation, phase and monotonic element budgets; malformed input, replay and
cancellation burn the pending material. Traffic includes both key packets and
both relay legs. Issuance occurs during the response and is charged to client
CPU. Memory admission prices client chunk buffers and every worker session slot.
This in-process dealer/co-located transport is experimental and unreviewed.

Generated Qwen2/Qwen3 prefill/decode matches an independent integer oracle,
including all checked logits and KV. Separate child workers pass SDK and gateway
requests. An installed wheel, tested outside the checkout without Cargo, also
completes the ordinary benchmark with nonzero key, peer and element counters.

## Fresh checkpoint quality

The pinned Qwen2.5-0.5B W8A8 cohort freezes twelve new public prompts and compares
clear execution of the installed numeric mapping with the exact W8A8 control.
Each teacher-forced prompt checks prefill and three same-token decode positions;
each free-running generation feeds back its own greedy selections, capped at
eight outputs. Both layouts have identical numerics.

| Measurement | Exact control | Masked-wire Q12 |
| --- | ---: | ---: |
| Prefill selections | 12/12 | 10/12 |
| Same-token decode selections | 36/36 | 36/36 |
| Bit-exact logits/KV positions | 48/48 | 0/48 |
| Worst absolute logit error | 0 | 3.46194 |
| Exact eight-token trajectories | 12/12 | 9/12 |
| Matching generation positions | 96/96 | 76/96 |

No prompt exceeds the public ranges. Native numeric outputs agree with an
independent integer oracle on 132,145,152 elements. This is a narrow fidelity
diagnostic, not a representative answer-quality evaluation or a comparison with
upstream floating-point generation. The prompts are now spent as held-out data.
Protected execution costs are measured independently through the ordinary SDK.

```sh
uv run python -m benchmarks.research.nonlinear_decoder_quality \
  --output nonlinear-decoder-quality-reproduction.json
```

Evidence: `docs/evidence/research-nonlinear-decoder-quality-qwen25.json`.
Replayable protected controls and exact seeded/residue transport combinations:
`benchmarks/research/nonlinear_decoders.py`. Fused/unfused may rank within their
numeric cohort; clear-SiLU has a different numeric identity even if selections
happen to match. No cost or throughput conclusion follows from quality replay.

## Bounded cost attempt

The five-configuration Qwen2.5 sweep (clear SiLU, both masked-wire layouts,
then both exact transport compositions) used a frozen installed wheel, isolated
clients, shared native 100/40 Mbps caps and two requested outputs. It stopped
at the **3,600-second per-request timeout**. The last live request had produced
no first token. The CLI did not persist its final comparison or earlier candidate
reports, so no completed response-cost or layout ranking is available.

`docs/evidence/research-nonlinear-decoders-qwen25-timeout.json` preserves the
command, factory and wheel hashes, observed failure and unknown measurements.
The CLI now checkpoints each candidate start and completed report atomically
when `--output` is supplied. Incomplete progress documents retain the failed
candidate identity and earlier evidence without publishing rankings. They do not
resume one-use sessions or combine separate invocation salts. This does not
recover the historical attempt's lost reports.

## Bounded transport diagnosis

`benchmarks/research/masked_wire_transport.py` derives the 4,864-element gated
width and 24 regions from the pinned Qwen semantic schedule, then measures one
synthetic row through the ordinary compiler-bound runtime. It compares the same
client/worker code in process, across two authenticated loopback worker children,
and through shared native 100/40 Mbps relays. Each of the six cases has three
fresh-material samples. All **87,552 outputs** match an independent integer
oracle; the worker cases observe zero swap growth.

| Per row, one region | Unfused | Fused |
| --- | ---: | ---: |
| In-process median | 1.696 s | 3.373 s |
| HTTP loopback median | 1.994 s | 3.601 s |
| Native 100/40 median | 3.539 s | 6.321 s |
| Client issuance, capped median | 1.075 s | 1.919 s |
| Serialized party keys, both workers | 6,279,880 B | 11,097,672 B |
| Complete covered application bodies | 7,151,942 B | 11,951,836 B |
| Rate-admitted TCP-stream bytes | 7,382,678 B | 12,125,002 B |
| HTTP requests | 456 | 342 |

Each row has 19 sequential chunks. Fusion reduces the peer rounds from 11 to 8
but its wider comparisons increase both key generation and delivery. In the
capped runs, key import/start requests take 1.666/3.873 seconds, while all
advance requests take 0.792/0.522 seconds. Issuance plus start requests occupy
about **77%/92%** of row latency. Ordinary HTTP adds about 0.30/0.23 seconds
over the in-process controls; pacing adds another 1.55/2.72 seconds. The dominant
costs here are fresh keys and serial delivery, rather than the Python HTTP
envelope alone. The diagnostic executes only the nonlinear region, not body
matrix stages.

Multiplying the capped row medians by the historical 39+8 geometry's
`24 × (39 + 8 − 1) = 1,104` row-regions projects **65.1/116.3 minutes** and
503,424/377,568 HTTP requests for these gated regions alone. This is a linear
extrapolation, not measured whole-response latency or an identification of which
historical candidate timed out. Startup, checkpoint delivery, other decoder
operations, quality, full physical wire and independent deployment remain outside
the diagnostic. The original timed-out request's input count and candidate
identity were not retained.

Evidence: `docs/evidence/research-masked-wire-transport-qwen25.json`. The report
retains each sample, phase/direction body counts, native implementation/profile
digests, compiler/source identity, authenticated worker CPU deltas and memory
admission. CPU deltas include small telemetry/idle overhead; in-process CPU
includes both local workers. Reproduce from a checkout with the pinned public
configuration cached:

```sh
uv run python -m benchmarks.research.masked_wire_transport \
  --samples 3 --output masked-wire-transport-reproduction.json
```

## Bounded issuance and delivery overlap

`MaskedWireGatedMultiplyQ12(dispatch="overlap")` is a separately bound compiled
runtime option. It issues at most one next chunk while the current chunk runs,
and sends to both workers concurrently within each phase. Each worker still
owns one active chunk. Cancellation or failure joins pending work and burns
unused material; prefetched indices cannot be retried. Serial dispatch remains
the default and preserves previous plan identities. The scalar operator benchmark
requires serial dispatch because it does not execute this transport schedule.

A new twelve-case cohort measures both dispatches, both layouts and all three
transports with three fresh-material samples each. All **175,104 outputs** match
the independent integer oracle, with zero observed swap growth in worker cases.
Tests separately check compiler-bound Qwen2/Qwen3 prefill/decode logits and KV,
partial final chunks, concurrent worker requests and cancellation/issuance failures.

| Median per synthetic row/region | Unfused serial | Unfused overlap | Fused serial | Fused overlap |
| --- | ---: | ---: | ---: | ---: |
| In-process latency | 1.701 s | 1.088 s | 3.360 s | 1.952 s |
| HTTP loopback latency | 2.010 s | 1.100 s | 3.618 s | 1.958 s |
| Native 100/40 latency | 3.554 s | 2.158 s | 6.342 s | 3.768 s |
| Native 100/40 aggregate process CPU | 2.148 s | 2.146 s | 3.770 s | 3.754 s |
| Native 100/40 client issuance | 1.075 s | 1.070 s | 1.918 s | 1.914 s |

Capped row latency falls **39.3%/40.6%**. Key bytes, application bodies, TCP-stream
bytes, HTTP request counts and rounds are identical within each layout. Work is
overlapped rather than eliminated; aggregate process CPU is nearly unchanged.
Action timings sum request durations and therefore overlap in wall time. Issuance
CPU uses its thread's clock to exclude concurrent HTTP work.

Client admission prices two chunk/frame allocations instead of one: **140 MiB
versus 70 MiB**, before the existing margin. Worker capacity is unchanged. These
are allocation bounds, not measured peak differences. The same synthetic-input,
one-region, startup-excluded scope applies to this cohort. The separate complete
response pair below measures the whole-decoder tradeoff. Fresh keys and their
traffic remain expensive.

Evidence: `docs/evidence/research-masked-wire-overlap-qwen25.json`. Reproduce with:

```sh
uv run python -m benchmarks.research.masked_wire_transport \
  --compare-dispatch --samples 3 --output masked-wire-overlap-reproduction.json
```

## Matched complete Qwen2.5 responses

The pinned Qwen2.5-0.5B checkpoint completes an ordinary SDK request under each
unfused dispatch. One invocation supplies a shared private cohort salt; each
candidate runs in a fresh isolated client with two co-located worker children,
shared native 100/40 Mbps party caps, no added RTT and no warmups. The public
`Hi` request renders to **30 input tokens** and generates **two outputs** under
the same Q12 mapping. Both output digests and all privacy/runtime checks match.

| Measurement | Serial | Overlap |
| --- | ---: | ---: |
| Full request latency | 2,705.809 s | 1,634.109 s |
| Time to first token | 2,586.445 s | 1,548.352 s |
| Later-output decode time, one token | 85.663 s | 52.518 s |
| Cold client process CPU | 1,037.713 s | 1,042.234 s |
| Cold aggregate process CPU | 1,645.455 s | 1,641.285 s |
| Client process-lifetime peak RSS | 227.197 MB | 229.491 MB |
| Same-window sampled aggregate peak RSS | 1,785.266 MB | 1,784.447 MB |
| Cold all-link TCP-stream traffic | 5,716.561 MB | 5,716.561 MB |
| Observed new host swap | 0 B | 0 B |

Overlap lowers request latency **39.61%** while client peak RSS rises **1.01%**.
Aggregate CPU differs by only 0.25%; each direction's TCP-stream count is exactly
equal. This is one response per candidate, not a confidence interval. Request
latency excludes provider startup; CPU and traffic include startup through the
first response. Client RSS includes the benchmark dashboard, relay and sampler.
The aggregate RSS is a 50 ms sampled live sum, not a sum of lifetime peaks, and
excludes OS file cache. Cached checkpoint distribution, physical TCP/IP overhead,
Internet RTT and independent operator deployment remain unmeasured. The prior
quality diagnostic still applies: this changed numeric mapping does not inherit
clear-SiLU quality merely because the two dispatches return the same two tokens.

Evidence: `docs/evidence/research-nonlinear-dispatch-qwen25.json`; source/factory
and frozen-wheel hashes: `docs/evidence/research-nonlinear-dispatch-qwen25-reproduction.json`;
pre-launch admission: `docs/evidence/research-nonlinear-dispatch-qwen25-preflight.json`.
The saved reproduction record binds the wheel, sources, targets and run settings.
From a checkout installed at that runtime revision, the same CLI arguments are:

```sh
uv run pllm benchmark run \
  --experiment benchmarks/research/nonlinear_dispatch.py:serial \
  --experiment benchmarks/research/nonlinear_dispatch.py:overlap \
  --trust-python --factory --backend native --wan --isolate-candidates \
  --prompt Hi --max-output-tokens 2 --warmups 0 --repetitions 1 \
  --timeout 3600 --temperature 0 --capture-output-digest \
  --output nonlinear-dispatch-reproduction.json --format human
```

## Native issuance and codec profile

An ignored, release-only Rust diagnostic calls the production primitive issuers,
encoders, imports and evaluator with fresh one-use material. Eighteen cases cover
1, 64 and 256 lanes, both layouts and three samples. All **1,926 complete-block
outputs** match an independent integer oracle. Microprofiles issue separate keys
and retire them after encoding; their timings are not added to complete-block
measurements. No secret keys, masks or reusable handles enter the report.

| Median for 256 lanes | Unfused | Fused |
| --- | ---: | ---: |
| Complete local issue/export/import/evaluate/decode | 88.593 ms | 175.434 ms |
| Complete issuance, including resource encoding | 51.356 ms | 91.628 ms |
| Party packet export | 4.728 ms | 8.288 ms |
| Party packet import | 3.195 ms | 5.544 ms |
| Worker start | 6.327 ms | 56.390 ms |
| Remaining worker advances | 22.923 ms | 13.626 ms |

The three GapARS rescalings account for **52.1%** of unfused microprofile issuance
and **59.6%** of its key bits. Fused direct-input index reduction and sign/interval
comparisons together account for **54.6%** of issuance and **54.4%** of its key
bits. That measured implementation serialized material once to measure resources
and again on packet export. DPF construction repeated the AES key schedule and
control block when expanding both children. The following implementation pair
measures removal of that redundant work.

Evidence: `docs/evidence/research-masked-wire-issuance.json`. The native test-only
module and wrapper are source-hashed. This is a bounded local implementation
profile, not distributed worker or whole-response timing:

```sh
uv run python -m benchmarks.research.masked_wire_issuance \
  --output masked-wire-issuance-reproduction.json
```

### Paired expansion and allocation-free resource counting

The DPF issuer and full-table walker now expand both children using one AES key
schedule and three AES blocks instead of two schedules and four blocks. A fixed-seed
regression checks identical PRG words. Point/comparison evaluation, key entropy,
correction words and fresh issuance remain unchanged. The ordinary encoder also
accepts a counting sink: resource validation walks the actual fields without
allocating or writing secret bytes; export performs the sole serialization.

Six interleaved fresh native processes compare frozen control and optimized
release binaries on the same 18 cases each. All **11,556 outputs** match the
independent oracle, with identical key sizes and per-phase peer frames.

| Median, 256 lanes | Unfused control | Unfused optimized | Fused control | Fused optimized |
| --- | ---: | ---: | ---: | ---: |
| Issuance | 51.317 ms | 36.837 ms | 91.633 ms | 61.193 ms |
| Complete local execution | 88.555 ms | 71.051 ms | 175.403 ms | 142.032 ms |

Issuance falls 28.22%/33.22%; complete local time falls 19.77%/19.03%. Export and
import costs remain nearly equal. These are local implementation measurements,
excluding HTTP, peak memory and whole-response execution. Payload reduction is a
separate change. Evidence: `docs/evidence/research-masked-wire-issuance-optimized.json`.
The comparison driver freezes a control with `--save-control DIRECTORY`, then
runs `--control DIRECTORY --output RESULT.json` after building the candidate:

```sh
uv run python -m benchmarks.research.masked_wire_issuance_comparison \
  --control frozen-issuance-control --output issuance-comparison.json
```

## Fresh party-local seeded material

`MaskedWireGatedMultiplyQ12(material_encoding="seeded")` is a separate key
encoding, composable with both layouts and dispatch choices. A fresh 256-bit seed
per party and chunk expands independent DPF roots (or party zero's short-table
leaf) and party zero's uniform coefficient shares. Each worker receives only its
own seed. AES-256 counter expansion binds the seed to the role, compiler/session/
tensor commitment and issuance nonce. Correlation-dependent corrections remain
explicit. Client input, intermediate and output wire masks retain independent
OS randomness and never derive from worker seeds.

Native export checks every omitted field against the corresponding tape before
consuming the party handle. Import preflights exact lengths, reconstructs bounded
full keys and checks canonical padding. The distinct packet namespace and plan
binding reject encoding substitution; cancellation, replay, expiry and partial
chunks retain their existing burn behavior. Expanded keys and concurrent chunks
retain the full allocation bound. This is a PLLM encoding adaptation under the
existing experimental trust contract, without independent cryptographic review.

At 256 lanes, combined encoded key payload falls **330,368 to 260,928 bytes
(21.02%)** unfused and **583,936 to 508,352 bytes (12.94%)** fused. Those totals
include both seeds but exclude packet headers. Peer rounds and frames are
unchanged. SDK operator issuance timing now includes packet encoding and local
worker imports for both encodings, so reports count the extra expansion work.

The bounded transport comparison can be reproduced with:

```sh
uv run python -m benchmarks.research.masked_wire_transport \
  --compare-material --samples 3 --output masked-wire-seeded-reproduction.json
```

The twelve-case cohort checks **175,104 exact outputs** with three fresh samples
per encoding/layout/transport and zero observed swap growth in worker cases.
Both candidates include the preceding CPU implementation optimizations and use
overlapped dispatch. Its capped row results are:

| One 4,864-element region | Unfused full | Unfused seeded | Fused full | Fused seeded |
| --- | ---: | ---: | ---: | ---: |
| Native 100/40 median latency | 2.111 s | 1.784 s | 3.706 s | 3.353 s |
| Aggregate process CPU | 1.808 s | 1.843 s | 3.109 s | 3.154 s |
| Complete key packets | 6,279,880 B | 4,960,520 B | 11,097,672 B | 9,661,576 B |
| Covered application bodies | 7,151,942 B | 5,832,582 B | 11,951,836 B | 10,515,740 B |
| TCP-stream bytes | 7,382,678 B | 6,063,318 B | 12,125,002 B | 10,688,906 B |

Latency falls **15.49%/9.52%**, while aggregate CPU rises **1.92%/1.45%**.
Worker reconstruction costs remain measured rather than treating seed expansion
as free. Keys account for the entire byte reduction; peer frames and request
counts remain equal. Evidence: `docs/evidence/research-masked-wire-seeded-qwen25.json`.
This remains a synthetic row-region comparison, excluding whole responses, peak
memory, startup, checkpoint transfer, physical wire and model quality.

### Completed seeded-material response pair

The subsequent matched Qwen2.5 pair executes one cold **30-input/2-output-token**
response per isolated client from the same frozen release wheel. Both use the
optimized native implementation, unfused layout, overlapped dispatch and shared
native 100/40 Mbps access limits. Output digests match, all runtime checks pass
and observed new swap is zero.

| Complete response | Full packets | Seeded packets |
| --- | ---: | ---: |
| Request latency | 1,611.722 s | 1,373.282 s |
| Startup-inclusive TCP-stream traffic | 5.7166 GB | 4.7350 GB |
| Covered application bodies | 5.5450 GB | 4.5634 GB |
| Combined party-key packets | 4.6722 GB | 3.6906 GB |
| Aggregate cold-first CPU | 1,394.90 s | 1,433.04 s |
| Client/dashboard lifetime peak RSS | 228.84 MB | 225.72 MB |
| Sampled simultaneous total peak RSS | 1,785.14 MB | 1,766.59 MB |

Latency falls **14.79%** and TCP-stream traffic **17.17%**, while aggregate CPU
rises **2.73%**. The exact **981,603,840-byte** key reduction accounts for the
entire reduction in both application bodies and stream traffic. The 3,618,816
nonlinear elements, 14,136 chunks and 267,622,752 logical peer-frame bytes match.
Client peak RSS falls 1.36%; expanded-key admission remains unchanged.

These are single-response co-located samples. Physical network overhead,
checkpoint distribution, independent operators and representative generation
quality remain outside the measurement. The earlier dispatch cohort used another
implementation and salt; its costs are not merged into this pair. Runtime:
`0167afc`. Evidence, preflight and frozen-wheel/source hashes are retained in
`docs/evidence/research-nonlinear-material-qwen25.json`,
`docs/evidence/research-nonlinear-material-qwen25-preflight.json` and
`docs/evidence/research-nonlinear-material-qwen25-reproduction.json`.

```sh
uv run python -m pllm benchmark run \
  --experiment benchmarks/research/nonlinear_material.py:full \
  --experiment benchmarks/research/nonlinear_material.py:seeded \
  --trust-python --factory --backend native --wan --isolate-candidates \
  --prompt Hi --max-output-tokens 2 --warmups 0 --repetitions 1 \
  --timeout 3600 --temperature 0 --capture-output-digest \
  --output nonlinear-material-reproduction.json --format human
```

## Qwen3-4B allocation gate

The same composed two-worker factory was inspected with the pinned Qwen3-4B
checkpoint, a 64-token input bound, two output tokens and native WAN pacing.
Both layouts require an estimated **14.26 GiB** of host memory, including
**0.97 GiB** for the client and **6.63 GiB per worker**, plus relay memory.
Admission rejected them against **7.08–7.09 GiB** of safe headroom at inspection.
This is current-host admission while the Qwen2.5 cohort runs, not a measured
peak or an assertion about every 32 GiB host. No 4B weights or provider roles
were loaded by this check.

`docs/evidence/research-nonlinear-decoders-qwen3-preflight.json` retains both
complete Experiment specifications, allocation owners and host observations.
Recheck either saved specification with the ordinary SDK:

```python
import json
from pathlib import Path
from pllm import Experiment
from pllm.metrics import benchmark_memory

document = json.loads(Path(
    "docs/evidence/research-nonlinear-decoders-qwen3-preflight.json"
).read_text())
experiment = Experiment.from_spec(document["cases"][0]["experiment"])
print(benchmark_memory(experiment, backend="native", enforced_wan=True))
```
