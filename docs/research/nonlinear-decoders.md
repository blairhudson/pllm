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
one-region, startup-excluded scope applies; checkpoint quality and whole-response
latency have not been remeasured. Fresh keys and their traffic remain expensive.

Evidence: `docs/evidence/research-masked-wire-overlap-qwen25.json`. Reproduce with:

```sh
uv run python -m benchmarks.research.masked_wire_transport \
  --compare-dispatch --samples 3 --output masked-wire-overlap-reproduction.json
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
