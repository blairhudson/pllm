# Bounded semantic placement frontier — 2026-10-01

## Decision

**Useful bounded client-work tradeoff; no 10× network breakthrough.** Under
**at least 80% BODY linear MACs remote**, the lowest warm covered-body forecast
among existing selectable placements is **client attention**: 148,297,986 bytes,
17.14% below the 178,970,558-byte baseline. Both values reproduce archived
measurements; the other candidate forecasts are **not measured protocol traffic**.
Attention owns **both `qkv_projection` and `attention_output`**, leaving exactly
**48 remote MLP stages**. It does not satisfy an 80% **all-linear** MAC cap once
the client-owned output head is included.

Next useful new measurement: **QKV-only** for a tighter BODY budget, and
**attention-output-only** for an 80% all-linear budget. Tiny numerical and private
prepared transport parity pass. Public-checkpoint traffic and device-resource
admission remain the next gate.

## Source, bounded selections and actual rings

- Pinned public checkpoint: `Qwen/Qwen2.5-0.5B-Instruct`, revision
  `7ae557604adf67be50417f59c2c2f167def9a775`, existing W8A8 quantizer.
- Checkpoint digest:
  `f69322057253c4fb853c10e73da5d84111262e0fb1edba33933b9e5b8caf59fd`.
- Checkpoint-resolution source lock:
  `383f5ed6947ba56ceed6083a9503d6302f09559e1e57ee614409900b36084c29`.
- Archived execution source lock:
  `880f80ec61274d3c80e2a0c1336394b9e955d53ff98436a426a680a460a592e7`.
  These are distinct locks, not interchangeable identifiers.
- Archived public BODY fingerprint:
  `5d631be30158b3ea2a72cae355ce09a3b304bbfddde48b5758c339c346a34974`.
- Retained geometry/directed-ledger SHA-256:
  `d44f3111edce0d462bbffbd4860a64f42c8823045dc675d111a5b7a250e473bd`.

Collection actually re-quantized **all 96 public body stages**, one stage at a
time, one thread, local cache only. For each matrix the artifact retains its
compiler-derived input/output shape, int8 weight and fp32 scale hashes, sizes,
actual signed output bound and existing `seeded_ring_profile` width. No uniform
u16 assumption: **72 u24 stages, 24 u32 stages**. Every QKV, attention-output and
gate/up stage uses u24; every MLP-down stage uses u32 on this checkpoint.

Per layer, input × output shapes are QKV `896 × 1152`, attention output
`896 × 896`, fused gate/up `896 × 9728`, and down `4864 × 896`. Total body
MACs per row are **357,826,560**; full int8 body payload is the same byte count,
and body scales are **1,216,512 bytes**. This collection measures public arrays
and ring selection; it does not execute a protected public response.

Bounded enumeration: baseline, all 15 nonempty semantic-role subsets, and
prefix lengths 1–8. Native compiler accepts **23 placements** and rejects the
all-four-role subset with `client linear placement must retain a remote body
stage`. Both prefill and decode schedules are complete; every linear executor
matches the existing semantic ownership rule. Experiment resolution also
checks W8A8 and one thread. No compiler contract was changed. Prefix-plus-role
combinations, arbitrary layer subsets and q/k/v splitting are not selectable
here and were not manufactured.

## Forecast accounting and archive calibration

Workload: **39 input + 32 emitted tokens**, requiring **39 + 31 = 70 fresh rows
per remote stage**, zero reuse. Forecast sums the archived baseline per-stage
serialized bodies of surviving remote stages across all five covered directions:

1. client → preparation, including stage seed setup;
2. preparation → client, including preparation acknowledgements;
3. preparation → inference, including **every recurring correction upload**;
4. client → inference, online masked inputs;
5. inference → client, online masked outputs.

Cold inference → client bundle is charged separately, then added to the same
recurring preparation/online ledger. New placements use baseline's **2,814-byte
setup allowance**, not a measured value or proven upper bound. Existing measured
placements use their archived setup remainder: attention 1,490; prefix-one
2,711; prefix-two 2,608. Warm forecasts for baseline, attention, prefix-one and
prefix-two reproduce their archived all-link and online totals **exactly**.
This is calibration, not independent measurement of the new placements.

Cold payload forecast is baseline bundle **145,977,006 bytes** plus added client
body int8 weights and fp32 scales. Placement-specific metadata/layout delta is
unknown. Native body snapshots are **resident copies**, not another cold weight
transfer. Calibrations expose the omitted metadata explicitly:

| Archived placement | Warm covered bytes | Online bytes | Cold covered bytes | Cold payload forecast minus archive |
|---|---:|---:|---:|---:|
| Baseline | 178,970,558 | 113,545,024 | 324,947,564 | 0 |
| Prefix-one | 171,513,615 | 108,814,126 | 332,451,106 | −357 |
| Prefix-two | 164,056,672 | 104,083,228 | 339,954,604 | −670 |
| Attention | 148,297,986 | 93,221,048 | 338,515,970 | −4,178 |

Original cold and warm archives have different prompt digests. Each comparison
within its own cohort is source/workload locked; their identical recurring
stage totals permit byte calibration, not a same-prompt CPU comparison.
Raw-report and public-summary hashes are verified during collection. Default
replay uses the retained canonical input hash and needs neither checkpoint nor
temporary raw reports. Directed mutations that conserve all-link totals still
fail closed.

Some preparation authorization forwarding, inference → preparation correction
acks, ready/status/control bodies and HTTP/TLS/framing/retries are outside the
archived counters. **These omitted directed bodies are unknown, not zero.** The
JSON's tensor-and-root-only ledger deliberately excludes all such metadata;
its preparation → client entry contains zero *tensor/root payload*, not a claim
of zero preparation acknowledgement traffic. No full-wire or total-WAN claim.

## Budget-specific Pareto candidates

Pareto axes: warm recurring covered-body forecast, client BODY MACs and client
BODY resident array floor. That storage floor is **weights + scales + one native
int8 body snapshot**, not total client RAM/disk. Full candidate lists and exact
frontiers are in the screen JSON; the review JSON extracts budget winners.

| Compute budget | Body-array cap | Lowest warm forecast candidate | BODY remote | All-linear remote¹ | Warm bytes | Cold payload forecast |
|---|---:|---|---:|---:|---:|---:|
| ≥95% BODY remote | 64 MiB | Prefix-one | 95.83% | 81.64% | 171,513,615 | 332,450,749 |
| ≥90% BODY remote | 128 MiB | QKV-only | 93.08% | 79.29% | 162,343,158 | 333,203,364 |
| ≥80% BODY remote | 160 MiB | Attention | 87.69% | 74.70% | 148,297,986 | 338,511,792 |
| ≥70% BODY remote | 256 MiB | Prefix-seven | 70.83% | 60.34% | 126,772,678 | 377,470,580 |
| ≥60% BODY remote | 320 MiB | Down + QKV | 63.85% | 54.39% | 117,117,070 | 392,658,748 |
| ≥80% all-linear remote¹ | 160 MiB | Attention-output-only | 94.62% | 80.60% | 164,926,710 | 330,257,316 |

¹ All-linear includes client output head **136,134,656 MACs per logit row**.
Native schedules verify one head row in prefill and each decode: 32 head rows
versus 70 body rows. This yields baseline all-linear remote share **85.18%**,
already below its 100% BODY share. The separate one-head-row-per-body-row proxy
in JSON is 72.44% baseline / 63.52% attention; it is not this request's workload
ratio. Embedding lookup is not charged as a dense matrix-vector multiply.

For the ≥80% BODY regime the nondominated warm set is baseline, prefix-one,
attention-output-only, QKV-only and attention. Prefix-two is dominated by
QKV-only; prefix-three/four by attention. Prefix-one remains useful at the
smallest client BODY budget. Additional prefix-five through eight enter looser
BODY frontiers. Down-only uses about 29.23% client BODY MACs, so it fails the
80%-remote BODY cap and is dominated by prefix-seven at the 70% regime. Its
large masked-input saving is not free client work.

The full scan also includes gate/up and larger semantic subsets. Some have
large predicted traffic reductions only by moving most BODY MACs client-side;
they fail these remote-majority budgets. The ≥80% BODY winner gives only
**1.207×** baseline/winner covered-body reduction, not 10×.

### Real storage floors, not device admission

| Placement | Client body int8 weights | fp32 scales | Native body copy | Body-array floor | Body + 70-row remote masks floor |
|---|---:|---:|---:|---:|---:|
| Baseline | 0 | 0 | 0 | 0 | 135,905,280 |
| Prefix-one | 14,909,440 | 50,688 | 14,909,440 | 29,869,568 | 160,112,128 |
| Attention-output-only | 19,267,584 | 86,016 | 19,267,584 | 38,621,184 | 162,484,224 |
| QKV-only | 24,772,608 | 110,592 | 24,772,608 | 49,655,808 | 171,798,528 |
| Attention | 44,040,192 | 196,608 | 44,040,192 | 88,276,992 | 198,377,472 |
| Prefix-seven | 104,366,080 | 354,816 | 104,366,080 | 209,086,976 | 305,353,216 |
| Down + QKV | 129,368,064 | 196,608 | 129,368,064 | 258,932,736 | 342,368,256 |

The 64/128/160/256/320 MiB caps are **only BODY-array budgets**. A 64 MiB total
RAM device is not admitted by the first row: even baseline remote mask payload
alone exceeds that. Embedding/head weights and their native copies, nonlinear
tensors, quantization/import/serialization temporaries, checkpoint/cache disk,
allocator overhead and peak RAM remain unmeasured. Existing measured placements
and all tiny candidates verify that native body snapshot bytes equal their
client body int8 weight bytes after use. Tiny head snapshots are retained
separately rather than mislabelled as body copies.

Preparation remains remote and substantive: baseline **25,047,859,200** and
attention **21,965,045,760** preparation MACs per 70-row inventory. All-linear
ratios concern online decoder linear work; they do not count preparation CPU.
**No whole-CPU compute budget is admitted**. Archived CPU samples are retained
but cannot convert matrix MAC ratios into client/remote whole-response CPU
shares. Client attention, nonlinear/head work, masking, serialization, startup,
energy and concurrency still need separate measurement.

### Cold and repeated use

Baseline dominates every candidate's single-response cold payload forecast in
network/client-compute/body-storage space. Actual archived cold attention is
**13,568,406 bytes larger** than baseline despite its warm saving. Reusing an
unchanged delivered bundle amortizes public weights, **not fresh correlations**.
Every additional response still pays preparation and all online bodies.

Payload-only forecasts first beat baseline after **two responses** for output,
QKV and attention, **three** for prefix-one, prefix-seven and down+QKV. Exact
1/2/8/32-response forecasts are retained in JSON; metadata, full-wire and cache
lifecycle changes can alter these thresholds. There is no claim of a measured
multi-response saving.

## Executed parity and checks

Generated deterministic seed-23 Qwen2 checkpoint: 24 layers, hidden 32,
intermediate 64, grouped-query attention, QKV bias, tied embeddings, W8A8.
All **23 accepted placements** execute native compiled client/remote schedules
with one thread. Three public prompts × prefill plus two decode steps produce
**nine exact logit vectors per placement, 207 total**, identical SHA-256 across
placements, maximum absolute error **0**. Remote callbacks reject client-owned
stages. Source BODY fingerprints agree. Callback uses native clear quantized
matmul; this is numerical/ownership verification, not masked traffic evidence.

Separate real two-child prepared integration tests execute baseline plus all six
budget winners on the same generated checkpoint: prefix-one, output-only,
QKV-only, attention, prefix-seven and down+QKV. Matched greedy response text and
usage agree; each retains remote stage calls while reducing their count.
Plaintext prompt bytes, plaintext token IDs and preparation-during-online audit
counters are zero. This is tiny protocol parity, not a public-model quality or
cryptographic security proof.

Checks: **15 tests pass**, Ruff lint and formatting checks pass, diff whitespace
check passes. Public benchmark CLI targets also pass a four-candidate dry-run
resolution. No training or download was performed.

## Exact reproduction

Replay both checked artifacts, without source access:

```bash
rtk proxy uv run --no-sync python scripts/probe_placement_frontier.py
rtk proxy uv run --no-sync python scripts/probe_placement_frontier.py --section review
```

Re-execute generated numerical parity and regenerate the screen:

```bash
rtk proxy uv run --no-sync python scripts/probe_placement_frontier.py \
  --verify-tiny --output docs/evidence/placement-frontier-screen-2026-10-01.json
```

Optional complete pinned-source collection from the existing cache, original
raw archives and tiny parity; offline flags and numerical thread limits are set
by the executable before imports:

```bash
rtk proxy uv run --no-sync python scripts/probe_placement_frontier.py \
  --collect-source --verify-tiny \
  --archive-dir /var/folders/3d/16hqtng12ng_bfz1n99jg04w0000gn/T/opencode \
  --output docs/evidence/placement-frontier-screen-2026-10-01.json
rtk proxy uv run --no-sync python scripts/probe_placement_frontier.py \
  --section review --output docs/evidence/placement-frontier-review-2026-10-01.json
```

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 HF_HUB_OFFLINE=1 \
rtk uv run --no-sync pytest -q tests/test_placement_frontier.py
rtk uv run --no-sync ruff check scripts/probe_placement_frontier.py tests/test_placement_frontier.py
rtk uv run --no-sync ruff format --check scripts/probe_placement_frontier.py tests/test_placement_frontier.py
git diff --check
```

## Recommended exact next gate

**Matched pinned public-checkpoint measurement of baseline, QKV-only,
output-only and attention**, one thread, no download, unchanged request-sized
inventory, no compression or prefix cache. Start with this warm body gate:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
rtk proxy uv run --no-sync pllm --no-input --format json benchmark run \
  --trust-python \
  --experiment scripts/probe_placement_frontier.py:public_baseline \
  --experiment scripts/probe_placement_frontier.py:public_qkv \
  --experiment scripts/probe_placement_frontier.py:public_output \
  --experiment scripts/probe_placement_frontier.py:public_attention \
  --max-output-tokens 32 --warmups 1 --repetitions 3 \
  --inventory-policy request-sized --bundle-compression none \
  --prefill-cache-mib 0 --timeout 900 \
  --output /var/folders/3d/16hqtng12ng_bfz1n99jg04w0000gn/T/opencode/placement-frontier-public-warm.json
```

The same command with **`--warmups 0 --repetitions 1`** and output
`placement-frontier-public-cold.json` is the first-response cold cohort. These
commands have only been dry-run resolved here; public protected execution was
not run. Each candidate is started and unloaded sequentially by the benchmark.

Gate acceptance requires matching source/body/prompt/workload/inventory keys,
39+32 usage, zero plaintext prompt/token transfer, zero reuse/online prep,
per-stage reconciliation of **every covered directed body**, observed ring and
head-row shape agreement, and no omitted prep cost. New setup and cold metadata
must replace allowances. Add explicit matched public W8A8 prefill/decode logits,
complete process CPU including startup/preparation, client peak RAM and full
checkpoint/native/cache storage on target hardware before any whole-CPU or
device admission. The existing benchmark alone does not establish all of these.

For now retain attention as the ≥80% BODY warm reference; measure QKV-only and
output-only as the tighter-budget next candidates. Looser prefix-seven/down+QKV
results are conditional BODY-budget options, not whole-compute-majority claims.
