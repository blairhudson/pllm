# Matched private-boundary decode SDK reference

The installed `PrivateLookupAndFullHead` component exposes two independently
selectable exact head-delivery choices. Defaults retain the historical serial/u32
configuration identity. Historical reports supply recipes only, never samples.

## Completed 150+8 SDK screen

All **15 fresh clients** completed under one installed build and private cohort
salt. **131,404,800 logit/KV values**, positions, selections, greedy RNG and output
commitments match; observed swap growth is zero. Every recipe's largest complete
worker peak fits Low (256 MiB), and therefore Medium (2 GiB) and High (16 GiB).
The parallel/uniform-residue candidate peaks at **228.67 MiB**.

Parallel dispatch alone raises median N−1 decode throughput **39.53%** over
serial/u32. Uniform residues alone raise it **12.64%**. Together they raise it
**52.42%**, from **2.182 to 3.326 tokens/s**. Every paired replicate improves:
serial/u32 spans 2.168–2.182 tokens/s; the combined choice spans 3.277–3.332.
These are measured combined costs, not multiplied isolated probe gains.

All entries below are three-client medians except maximum client RSS. CPU and
latency are seconds; MB is decimal and MiB is binary.

| Choice | Max client MiB | Client CPU | Prep CPU | Prep MB | Online TTFT | Decode tokens/s | TCP MB | Full request |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Maintained prepared | 231.50 | 14.572 | 8.587 | 215.92 | 38.770 | 4.948 | 478.654 | 99.037 |
| Coupled serial/u32 | 221.08 | 12.146 | 8.327 | 209.35 | 41.895 | 2.182 | 367.347 | 75.975 |
| Coupled parallel/u32 | 222.83 | 12.561 | 8.426 | 212.65 | 41.907 | 3.045 | 367.347 | 75.199 |
| Coupled serial/residues | 229.47 | 12.587 | 8.509 | 211.16 | 41.996 | 2.458 | 365.221 | 75.872 |
| Coupled parallel/residues | 228.67 | 12.583 | 8.486 | 212.91 | 41.943 | 3.326 | 365.221 | 75.082 |

| Additional cost | Maintained | Serial/u32 | Parallel/u32 | Serial/residues | Parallel/residues |
| --- | ---: | ---: | ---: | ---: | ---: |
| Client lifetime peak RSS median, MB | 232.59 | 231.31 | 229.65 | 226.13 | 234.09 |
| Aggregate cold-first CPU, s | 37.653 | 36.841 | 37.596 | 37.740 | 37.574 |
| Startup-through-cleanup time, s | 110.219 | 90.295 | 89.687 | 90.545 | 89.672 |
| Covered application bodies, MB | 477.105 | 365.796 | 365.796 | 363.670 | 363.670 |

Against serial/u32, the combined choice lowers complete request latency **1.17%**
and TCP traffic **0.58%**, while client CPU rises **3.60%** and median peak RSS
**1.20%**. One of three paired full-request samples regresses; the small complete
latency change does not establish the same robust effect as the decode result.
Observed Low eligibility is not a memory-reduction claim or conservative SDK
admission under a 256 MiB cap.

Against the fresh maintained prepared comparator, the combined choice lowers
all-link TCP streams **23.70%**, full request latency **24.19%**, complete target
time **18.64%** and client CPU **13.65%**. Decode remains **32.78% slower**, online
TTFT rises **8.19%**, and median client peak RSS rises **0.65%**. Aggregate CPU
ranges overlap. These explicit cross-topology comparisons charge two additional
boundary workers and retain their non-collusion contract.

This is the **first whole-SDK screen**, registered as
`qwen25-private-boundary-decode-150-8`. SOTA tables keep Prepared and
Prepared + private boundary leaders separate and label this cohort **Screen best**.
The earlier eight-recipe tier confirmation remains independently recorded; its
timings and best-incumbent comparisons are not transferred into this cohort.
Independent whole-response confirmation and longer output workloads remain open.

### Exact traffic reconciliation

Each residue candidate removes **2,125,920 bytes** from the sixteen head replies:
9,726,176 → 7,600,256 application bytes. The payload reduction is exactly 21.875%;
the complete response still includes prepared-body, lookup and lifecycle costs.
All three replicates preserve fixed preparation material/rows, per-stage directed
work, lookup bytes and head-request bytes. The raw startup and request counters
reconstruct their canonical ledgers; every directed body total fits its enclosing
TCP measurement window.

Lifecycle controls differ by zero or one byte. In this Qwen cohort, prepared
reply-size deltas and the remaining bootstrap/metadata residual are zero. The
auditor nevertheless permits only independently reconciled metadata-compatible
variation: prepared residue frames encode `server_ns` with a variable-width
MessagePack integer. Tiny runs crossed that integer-width boundary. No unknown
counter is supplied as zero, and metadata variation is not called codec savings.
TCP stream reductions retain their separate framing/control differences.

## Execution contract

```python
from pllm.roles import PrivateLookupAndFullHead

boundary = PrivateLookupAndFullHead(
    head_dispatch="parallel", head_output_encoding="uniform_residues",
)
```

Parallel dispatch sends the current head's two authenticated requests through
separate worker clients. Both outstanding calls drain before cancellation or
partial-failure retirement; no logits escape a cancelled pair. Lookup retains
its ordered, one-use protocol. There is no persistent executor or future-head
reservation.

Uniform residues use the admitted bias-free signed-i8 weight contract and
symmetric activation bound to prove `abs(Wq) <= hidden * 128 * 127`. The wire
width is `(2 * bound).bit_length()`: 25 bits at Qwen2.5's hidden width of 896.
Input shares remain ring32. Each worker narrows only its output share; the client
reconstructs exact signed integers before unchanged float32 dequantization and
sampling. All vocabulary rows remain present. The descriptor binds source,
matrix/scales, compiler schedule, encoding and dispatch; per-response frames bind
role, session, epoch, admission and sequence. Neither a peer-supplied width nor a
client-side weight scan is used. Checksums detect corruption, not malicious
computation; the existing authenticated roles and non-collusion assumptions remain.

Admission retains original full-frame pricing and additionally charges parallel
thread/frame owners and codec/layout temporaries. Generated Qwen2/Qwen3/Llama
decoder and SDK/gateway controls check complete logits/KV, selections and RNG,
including malformed frames, replay, partial failure and async cancellation.

## Matched measurement contract

Five fresh ordinary SDK cells share one installed native build and private cohort
salt: maintained prepared comparator, coupled serial/u32 control, parallel/u32,
serial/uniform_residues, and parallel/uniform_residues. Each pays the same full
logit, active-KV, position, selected-token/history and greedy-RNG audit cost.
Each has three balanced fresh replicate clients; shared native access caps remain
100 Mbps download / 40 Mbps upload. Production bounds remain 150 input + 8 output
tokens, with source and indexed tokenizer already present. Tiny generated-source
smoke uses 128+2 bounds and cannot qualify production evidence.

```sh
.venv/bin/python scripts/benchmark_private_boundary_decode.py \
  --output target/private-boundary-decode-preflight.json --preflight-only
.venv/bin/python scripts/benchmark_private_boundary_decode.py \
  --output target/private-boundary-decode-tiny.json --tiny
.venv/bin/python scripts/benchmark_private_boundary_decode.py \
  --output target/private-boundary-decode-sdk.json
```

Use fresh output names. Plans include all five conservative topology admissions
plus equal audit scratch admission. Ordinary host-pressure/swap guards stay
enabled; completed evidence with observed swap growth is rejected. No wheel build
or source download belongs inside this measurement scope.

Summary reports all seven serving metrics, covered bodies, full request time,
complete target lifecycle time and complete worker peak RSS. Low256MiB requires
the largest complete worker peak across all three replicates to fit, not median,
child-only RSS or a sum of process peaks. Owner includes SDK, supervision,
dashboard, native relay and audit; idle suite launcher stays excluded.

Every boundary cell is explicitly compared with both coupled serial/u32 and
maintained, including paired replicate deltas. Prepared arithmetic, lookup and
head-request counters must match across boundary variants. Measured head-reply
delta is separate from lifecycle-control delta and bootstrap/metadata residual.
Residual remains charged: do not describe total stream/body savings as exact codec
payload savings. Cross-topology maintained comparisons retain different worker
and trust contracts. No historical pooling or threshold weakening is permitted.

## Frozen evidence and read-only replay

The executed command was:

```sh
target/frontier-sdk-env/bin/python scripts/benchmark_private_boundary_decode.py \
  --output target/private-boundary-decode-qwen-v1.json
```

Use a new output path for another measurement. Original source hashes were checked
before archival; summary replay loads no checkpoint or native execution owners.

- [Raw SDK results](../evidence/private-boundary-decode-qwen25-sdk.json), SHA-256
  `56bcdfbcb53aba403cc406b4854cbf702de41d4b1e9cd0138d26dc7b84997df0`.
- [Frozen suite, sources and admissions](../evidence/private-boundary-decode-qwen25-sdk.plan.json), SHA-256
  `21fb23e53bde4276dcb3df4a57997a52d3689bd34e02a3b863fbb81e567341d1`.
- [Audited summary](../evidence/private-boundary-decode-qwen25-sdk.summary.json), SHA-256
  `109ac6ecafd507680a8837033f86808c5b2091e8771b3a69636bb260a590b89e`.

```python
import json
from pathlib import Path
from benchmarks.research.private_boundary_decode import audit

root = Path("docs/evidence/private-boundary-decode-qwen25-sdk")
raw = json.loads(root.with_suffix(".json").read_bytes())
saved = json.loads(root.with_suffix(".summary.json").read_bytes())
assert audit(raw) == saved
```

The suite preserves build digest
`62c17e60f9769b0b29b74cf1b45061c57391d4a638490d7e619754cae6af7e99`.
Each recorded case can be imported with `BenchmarkCase.from_spec`; a changed
installed build requires fresh measurements. Checkpoint and indexed-tokenizer
distribution, physical wire and Internet RTT remain outside this source-present
native rate-capped cohort. The allocating-output byte-dot evidence is a separate
kernel API probe in [its reference](boundary-head-dotprod-probe.md); it is not
selected in any of these SDK cases.
