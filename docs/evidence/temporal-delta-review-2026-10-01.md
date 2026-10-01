# Temporal-delta screen — 2026-10-01

**Decision: fails the coordinate-sparsity kill gate for the bounded cohort.**
The identity `W(q_previous + delta) = Wq_previous + Wdelta` is exact, but
quantized stage inputs change too densely to support a large communication
reduction. This is a temporal-input experiment, not another sparse-weight,
low-rank, or client-side weight-predictor experiment. No protocol admission.

## Source and workload

- Cached `Qwen/Qwen2.5-0.5B-Instruct`, revision
  `7ae557604adf67be50417f59c2c2f167def9a775`; W8A8 body fingerprint
  `5d631be30158b3ea2a72cae355ce09a3b304bbfddde48b5758c339c346a34974`.
- `scripts/decoder_probe_support.py` supplies checkpoint resolution, native
  semantic binding, tokenizer/rendering, and baseline trajectory. Resolver
  inspected before execution: `HF_HUB_OFFLINE=1` makes `snapshot_download`
  cache-only. Cached snapshot resolved successfully; no checkpoint download.
- Six fixed public prompts, three screening and three disjoint deterministic
  confirmation prompts; rendered lengths `39, 40, 38` and `36, 37, 37`.
  No truncation, training, or prompt-dependent workload selection.
- Four decode steps per prompt, plus the initial prefill output. Dense baseline
  chooses greedy tokens; temporal replay consumes those same tokens. Token
  trajectory digests, checkpoint/source locks, plan, schedule, composition and
  binding digests are locked in the companion JSON.
- One native model thread; BLAS/OpenMP/Rayon limits set to one before launching
  Python. Fixture's hard-coded four-thread constructor is overridden only
  inside this probe. No shared helper modification.
- All 96 native body stages observed under the **baseline `MaskedLinearCpu`
  execution composition**. The SDK attention candidate owns **both
  `qkv_projection` and `attention_output` on the client**. Its derived remote
  scope therefore includes only **48 MLP stages**: 24 `mlp_gate_up` and 24
  `mlp_down`. Both attention roles remain in the baseline delta diagnostics.
  This scope is a semantic-role subset of those baseline measurements, **not
  an executed distributed SDK placement**. The JSON locks this distinction in
  `sdk_derived_scope`, alongside exact resource counts.

## Quantized-coordinate results

Percent of coordinates **unchanged** from the immediately preceding stage row:

| Role | Screening prefill | Confirmation prefill | Screening decode | Confirmation decode |
| --- | ---: | ---: | ---: | ---: |
| qkv | 6.36% | 6.37% | 5.58% | 5.14% |
| gate-up | 3.50% | 3.48% | 3.75% | 4.19% |
| down | 11.03% | 11.05% | 11.51% | 11.58% |
| attention output | 12.29% | 11.91% | 11.36% | 10.38% |

Prefill compares **adjacent causal token positions within each held-out public
prompt**, not repeated identical prefixes. Decode compares the last prefill
row to the first decode row, then successive decode rows. These are different
token positions; “same-token” means identical token forcing across baseline
and temporal replay, not repeating one constant token.

The JSON locks changed-fraction distributions, including min, p10, median,
p90, max and mean, for every role/phase/cohort. Each role has 2,736 screening
prefill comparisons, 2,568 confirmation prefill comparisons, and 288 decode
comparisons per cohort. No identical quantized input row was observed.
Some individual down rows are less dense, but median changed fractions remain
about 90.5–91.4%; public-capacity admission cannot depend on those private rows.

**Activation scale changes in 100% of sampled transitions**, independently of
coordinate-change accounting. A unchanged quantized row would still require
the current scale. Raw floating-point proximity is recorded by the full probe
as a relative L2 distribution and float-coordinate change distribution; it is
never used to authorize exact updates or omitted coordinates.

Differences exceed the signed-i8 range in every measured role/phase/cohort.
Observed maximum reaches `254`; symmetric W8A8 subtraction must widen before
subtracting. Nonzero delta alphabet has 508 values, requiring nine bits for a
fixed-width plaintext encoding, not eight.

## Optimistic communication bounds

For input width `n`, changed count `k`, output width `m`, and original exact-ring
width `b`:

- Dense input body: `ceil(n*b/8)`.
- Known-support plaintext delta estimate: `ceil(9*k/8)`.
- Enumerative plaintext estimate:
  `ceil((ceil(log2(C(n,k))) + 9*k)/8)`.
- Explicit-index plaintext estimate:
  `ceil(k*(ceil(log2(n)) + 9)/8)`.
- **Masked-ring sparse oracle input**:
  `ceil((ceil(log2(C(n,k))) + b*k)/8)`.
- Online bodies add one dense output; all-link bodies add two dense outputs,
  including correction/preparation. Public full-width padded index/ring
  capacity is `ceil(n*(ceil(log2(n)) + b)/8)` and exceeds the dense input body.

These are fixed-alphabet packing/support-oracle estimates, not universal
entropy lower bounds. The enumerative code gives the private changed count
away for free. Ring-width payloads are necessary for freshly masked slots;
nine-bit plaintext deltas cannot stand in for full-width random masks.

Derived **48-stage SDK-remote transition arithmetic** from sampled baseline
MLP inputs, excluding dense initial rows, **both qkv and attention output**,
framing and all other model traffic:

| Cohort / phase | Dense all-link bytes | Sparse-ring oracle all-link bytes | Oracle reduction | Online reduction |
| --- | ---: | ---: | ---: | ---: |
| Screening prefill | 239,892,480 | 234,609,079 | 2.202% | 3.517% |
| Screening decode | 25,251,840 | 24,669,855 | 2.305% | 3.680% |
| Confirmation prefill | 225,162,240 | 220,192,457 | 2.207% | 3.524% |
| Confirmation decode | 25,251,840 | 24,662,393 | 2.334% | 3.727% |

Even the unachievable known-support nine-bit input estimate leaves dense output
and correction bodies. For these 48 stages, removing all input bodies would
still leave roughly 74.7% of all-link transition arithmetic and 59.7% of online
transition arithmetic. Input-only delta encoding cannot produce a 10× result
under this output/correction contract.

Each live sequence needs a dense first row per stage; each prompt resets state.
The JSON's initialization counters cover **all 96 observed stages**, not just
the 48 derived SDK-remote stages: 7,584,768 all-link and 4,783,104 online
arithmetic bytes per three-prompt cohort. The corresponding 48-stage
initialization costs are **6,312,960 all-link and 3,953,664 online bytes per
cohort**, or **2,104,320 / 1,317,888 bytes per live sequence**, with 531,456 input
body bytes and 313,786,368 dense provider MACs per initialization sequence.
These costs further dilute any estimate.

Controls are context only: new SDK attention placement (client qkv + attention
output), `39+32`, warm all-link
**148,297,986**, online **93,221,048**; original prepared **178,970,558** /
**113,545,024**. This short variable-prefill cohort is not matched to those
controls. No scaling to a whole-model or measured-network saving is claimed.

## Exactness, retained state, and work

For every decode stage the probe checks
`I_new = I_previous + W_integer * delta` against a fresh native dense product,
both as the full signed integer and modulo the original exact ring. It
dequantizes `I_new` with **current** activation scale and unchanged weight
scales, then adds bias once. Adding a current-scale floating delta to an
old-scale floating output is generally wrong.

Native clear kernels accept i8 inputs. The probe splits widened delta into two
exact i8 limbs; neither rounding nor requantization is introduced. Results:

- **2,304** native full signed integer checks and **2,304** exact-ring checks.
- **192** independent NumPy sparse-reference checks: every stage at the first
  decode of the first prompt in each cohort. Unit tests use an independent
  scalar-integer oracle, including extreme deltas and modular wraparound.
- **30** float32 logit rows equal the dense baseline exactly; greedy selections
  also match. No tolerance-based acceptance.
- Prefill coordinate statistics are real checkpoint measurements, but prefill
  temporal execution itself is not implemented or quality-certified.

Original ring widths are 24 bits for qkv, gate-up and attention output; 32 bits
for down. Independently lifting `Wdelta` can need a larger certificate because
its domain is twice as wide: all 24 gate-up stages, qkv layers
`0,1,3,5,6,10,12,15`, and attention-output layer `1` grow from 24 to 32 bits.
Updating **modulo the original ring** is valid because the final current
product remains in its original certified domain. Individually decoding a
delta product with the old signed bound is not valid.

Ideal retained **client** temporal state per stage, per live sequence:

| Role | Previous i8 input | Exact i32 product | f32 scale | Total per stage | Stages |
| --- | ---: | ---: | ---: | ---: | ---: |
| qkv, baseline diagnostic / candidate client-owned | 896 | 4,608 | 4 | 5,508 | 24 |
| gate-up | 896 | 38,912 | 4 | 39,812 | 24 |
| down | 4,864 | 3,584 | 4 | 8,452 | 24 |
| attention output, baseline diagnostic / candidate client-owned | 896 | 3,584 | 4 | 4,484 | 24 |

Totals: **1,158,336 bytes** for 48 derived SDK-remote stages, comprising
**138,240 previous-i8-input bytes**, **1,019,904 exact-i32-product bytes** and
**192 f32-scale bytes**; **1,398,144 bytes** for all 96 baseline diagnostic
stages. Candidate client-owned attention diagnostics contribute 239,808 bytes
to the latter, not the remote-stage temporal-state total. These exclude
ordinary KV cache, model tensors, active
correlations, allocator overhead and session metadata. The diagnostic also
stores previous floats and i64 products; those are not the proposed i32 state
layout. Remote temporal history can be zero if the client retains the prior
exact product and requests a freshly masked delta product. Any alternative
remote history containing private values requires a separately specified
masked/shared representation; no such state contract is admitted here.

Hypothetical client work is `n` subtractions, `m` accumulator additions and
current-scale dequantization per transition. Hypothetical provider sparse MACs
are `k*m`, versus `n*m` dense MACs; the JSON locks those optimistic counts. They
assume plaintext support known to the provider and are not private execution
costs. The proposed client never takes over dense checkpoint matrix products.

The single-process diagnostic executes the baseline composition and emulates
its provider execution, not the SDK attention candidate or a deployment.
It performs **179,628,933,120 dense native MACs** across dense baseline and
replay, plus **17,175,674,880** native delta-limb MACs and sampled independent
sparse-reference MACs. Extra verification work is deliberately charged,
separate from hypothetical sparse provider work. No latency/WAN measurement.

## Privacy gate and scope

Indices, changed counts, packet sizes, fallback decisions and iteration counts
cannot reveal private prompt/activation properties. Fresh ordinary dense masks
remove visible zero sparsity. Reusing a mask across adjacent inputs exposes
their difference and is forbidden. Hiding only indices while retaining private
length is insufficient. Padding a straightforward index/ring encoding to the
public worst-case input width removes its sparse-length benefit. A different
oblivious access protocol would require its own fixed schedule, one-use mask
inventory, cost accounting and security review; this screen supplies none.

No HE/TEE trust shortcut, trained predictor, new public runtime selection, or
client plaintext remote history. Aggregate sparsity diagnostics are offline
research outputs, not approved telemetry for private requests. The script
serializes no raw prompt, token, activation, scale, support, logit or weight.

The confirmation cohort strengthens reproducibility for six public prompts
and four decode steps each. It does not establish behavior over long decode,
private production prompts, other checkpoints, quantization settings, or
output-sparsity/stateful cryptographic protocols. This candidate is screened
out for the bounded exact input-update communication hypothesis.

## Reproduction and checks

From repository root, existing environment and cached checkpoint:

```sh
rtk proxy env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 PYTHONPATH=python .venv/bin/python scripts/probe_temporal_deltas.py --summary
rtk proxy env PLLM_RUN_TEMPORAL_DELTA=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_temporal_deltas.py
rtk proxy .venv/bin/python -m ruff check scripts/probe_temporal_deltas.py tests/test_temporal_deltas.py
rtk proxy .venv/bin/python -m ruff format --check scripts/probe_temporal_deltas.py tests/test_temporal_deltas.py
```

Verified: **15 tests pass**, including a complete offline native execution that
reproduces every stored measurement and digest; Ruff lint and format checks
pass. The evidence JSON is a compact subset of `--summary`; the opt-in replay
compares every retained field recursively. Without the opt-in environment,
the cached-checkpoint test skips while arithmetic/scale/encoding tests run.
Scope regressions use opaque stage IDs over 24 layers to ensure semantic-role
filtering excludes both attention roles, check exactly 48 stages and all
retained-state/initialization resources, reject unrelated cohort/phase data,
and independently sum locked MLP role measurements into the derived arithmetic
totals while requiring all four baseline roles' diagnostics remain present.
This revision corrects the initial 72-stage scope that incorrectly included
qkv; execution/checkpoint/composition locks and baseline diagnostics remain
the same, while derived scope, costs and state totals are corrected.
Initial pytest replay hit the test suite's isolated XDG cache; fixed by
capturing and explicitly passing the pre-isolation HF cache path. No download
fallback was enabled. Initial formatting differences were corrected.

Created only `scripts/probe_temporal_deltas.py`, `tests/test_temporal_deltas.py`,
this review, and its companion JSON. No shared documentation edits, staging,
commits or other agents. Completion clean; no execution blocker remains.
