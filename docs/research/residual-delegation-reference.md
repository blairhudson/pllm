# Exact integer residualized public delegation

Status: **bounded checkpoint screen complete; low-priority byte result**.
Twelve of 96 public Qwen stage groups were fitted at ranks zero through four.
Two groups have positive one-request tensor-payload projections after raw factor
and width-map delivery: **27,662 bytes combined**. All checked stage integers and
original dequantization bits agree. A separate generated decoder preserves exact
logits and active KV. Private serving, full-body savings and latency remain unmeasured.

## Checkpoint result

`docs/evidence/research-residual-delegation.json` retains the initial four-worker
layer-zero, rank-0/1/2, one-seed screen separately from twelve later workers:
layers 0/12/23, all four stage families, ranks 0–4, two public seeds. Workers ran
serially with frozen checkpoint/code identities and zero observed swap growth.
Exact duplicate compiler, geometry, control and provenance records are stored
once in `common`/`worker_common`; each original raw report hash is retained.

| Group | Best one-request rank | Gross bytes saved per executed row | Extra factor/map bytes | Net at 150+8 |
| --- | ---: | ---: | ---: | ---: |
| Layer 0 QKV | 2 | 118 | 5,248 | 13,278 |
| Layer 12 QKV | 1 | 112 | 3,200 | 14,384 |
| Other ten sampled groups | 0 | 0 | 0 | 0 |

Both positive cases cross the maximum input packing width from 24 to 23 bits.
Smaller L1 improvements usually fail to cross integer bit-width thresholds. The
table selects the best measured rank, including unchanged rank zero; it is not
a preselected serving policy. Terminal-pruned final MLP rows are priced from the
compiler, not assigned a full 150-row prefill. Complete manifests, framing,
native factor ownership and publication are still additional costs. These are
one-fresh-request tensor projections, not savings subtracted from the historical
cold/exact-repeat pair. No extrapolation to the other 84 groups is made.

This screen does not support the initial 2–5% all-link target. The two small
threshold wins are insufficient to justify a new serving composition on current
evidence; other public factor constructions remain untested.

## Contract and novelty

For original symmetric per-row W8A8 stage, construct public integer factors
`W = A B + E` and compute `A(Bx)` locally. Delegate dense `E x`, reconstruct its
exact signed integer result in its **own** public bounded ring, then add local
integer result **before** applying original activation/weight scales and bias.
Never requantize factors into an approximate replacement, saturate an
intermediate, narrow E silently, or omit any residual entry.

This search uses residual-row sign prototypes, not earlier block means/moments
or exact output-row dictionaries. Network-algebra evidence rejected moments
after factor/snapshot costs and found no dictionary duplicates. Here objective
is public row-L1, exact residue widths and transmitted tensor payload, with
factor-delivery break-even checks. It is a bounded hypothesis screen, not a
claim that random checkpoint matrices have useful integer low rank.

## Bounded fitting and admission

`scripts/probe_residual_delegation.py` searches ranks zero through requested rank
(default two; maximum four). Each step considers at most two public residual
row seeds, prototype magnitudes 4/16 and coefficients -2/-1/0/1/2. All factors
are signed i8. Construction and comparison are integer-only. Row decisions
minimize residue width, then L1; outer-product candidates minimize total packed
input/output/correction payload, then L1. Rank-zero baseline remains reported.
This greedy search need not find globally optimal factors. It never fits private
activations, approximate outputs or floating SVD factors.

Default `native-i8` policy prohibits any row update outside [-128,127]. Optional
`reference-i16` retains wider E exactly, records native rejection and credits no
savings for rejected candidates. Native `CompiledMatrix` accepts only i8;
ordinary `clear` accepts only i8 activations, so it **cannot** execute A on widened
Bx. Local factors use explicit checked int64 research arithmetic. Native widened
factor kernels, factor snapshot ownership, source-bound residual admission,
provider artifact publication, session/digest binding and authenticated protocol
integration are unresolved production gates. Residual rank is unknown, not
assumed small; E stays dense and remote. Dense higher-precision E can only lose
native admission in current interfaces.

Public `127 * row_L1(E)` determines strict signed bounds. Provider ring is actual
`seeded_ring_profile` u16/u24/u32 choice; output widths agree with installed
`offset_row_bits`. Synthetic tests and workers exercise native wrap32 reduced
to that ring, stage-packed input projection, prepared row-residue corrections,
masked outputs and exact signed reconstruction. This is a clear algebra
reference with public synthetic inputs, **not private serving or a cryptographic
review**. Existing trust, non-collusion and one-use mask requirements are not
relaxed by subtracting public factors.

## Numeric checks

Each candidate retains complete E and checks the entire matrix identity.
Synthetic activation extremes and seeded stand-ins compare all integer outputs
against original native W8 clear stage and an int64 product; selected output
rows additionally use independent Python arbitrary-precision dot products over
every input coordinate. Original stage float32 dequantization must agree.
Only hashes, public dimensions/bounds and aggregate costs enter reports; no raw
private values are written.

The separate `--generated-only` worker creates seed23 two-layer Qwen2 hidden32 /
intermediate64, fits native-admitted rank-one factors, then compares every logit
and active KV bit after prefill and two generated decode steps through compiled
research callbacks. It checks executed per-phase client MAC budgets. That worker
passed all three compared states, with nonzero rank-one factors on all eight
body groups. Measured callback client-MAC fractions remain below 10% in both
phases. It is a generated clear-decoder check, separate from checkpoint quality
or protected protocol execution.

Tiny checks executed:

```sh
target/frontier-sdk-env/bin/python -m pytest -o pythonpath= -q \
  tests/test_residual_delegation_probe.py
```

Result: **9 passed**. These tiny tests are separate from the fresh-process
checkpoint and generated-decoder cohorts above.

## Coverage, bytes and resources

Pinned source: `Qwen/Qwen2.5-0.5B-Instruct`, revision
`7ae557604adf67be50417f59c2c2f167def9a775`, explicit symmetric per-row W8A8.
Compiler/semantic bindings derive ordered fused weight keys and actual terminal
prefill row counts; no hand-written dispatch by checkpoint name. Initial cohort
fits layer-zero QKV, attention-output, gate/up and down families; the later screen
adds middle and final layers. Unique coverage is **12/96 groups**, not all layers.
Compiler geometry is checked against frozen
network-algebra geometry; complete conserved control is checked for matching
body fingerprint. Whole-body fitted byte savings remain null.

Historical public geometry alone gives 357,826,560 body MACs per fully executed
row. Rank two across all 96 groups would add 970,752 client MACs (0.2713%),
970,752 raw factor bytes and 304,128 raw new width-map bytes. These are geometry
forecasts, **not fitted results or terminal-pruned admission**. Runner independently
prices actual compiled prefill and decode and reports both <=10% gates. Per-stage
gate is stricter still, so selected ranks cannot consume more than 10% in either
phase. E remains full dense provider computation; no original body stages move
entirely onto client.

Largest historical group is 9,728 x 896: 8,716,288 i8 weight bytes. Fitting uses
int16 residuals, bounded 128-row int64 trial chunks, a few full candidate buffers
and transient full int64 L1/identity scans. Conservative matrix cohort admission
is **768 MiB per serial worker**; generated callback admission **1 GiB** includes
Torch import. These are admission requests, not measured RSS. Rank-two raw local
factors add about 0.97 MB over entire body; current int64 conversion adds another
7.77 MB if all factors converted simultaneously, before widened Bx/output arrays.
Future native copies/snapshots must be charged separately, never presumed free.
Client memory remains a priority; current native factor snapshot is zero solely
because no native factor kernel exists.

Tensor accounting separately records factor delivery/storage, original scales,
new maps, residual storage/provider snapshots, unchanged original weights held
by probe, inputs, corrections and outputs. Horizon projections (1/4/16 new
150+8-token requests) subtract raw factor/map delivery once. Reported break-even
rows are **optimistic lower bounds**: actual manifests/framing/compressed
artifact distribution and unchanged bundle/control bodies still need charging.
These are not historical cold/exact-repeat-pair predictions; repeat-cache rows
must be priced from actual serving ledger. Reject candidates with no admitted
per-row savings, no attainable charged break-even, or increased client memory
beyond policy. Do not mix these projections with previous measured codec or
placement savings.

Fitting CPU includes bounded search and factor copying, separately from source
load/quantization and parity CPU. Identity scans and generated setup add further
CPU. All installed dependency Python/native files, interpreter, installed PLLM,
repo Python/Rust sources/manifests, probe/shared helper and control/geometry are
hashed around workers. Cached checkpoint/config bytes are checked before/after.
No performance benefit can be inferred from exact arithmetic alone.

## Parent-only serial replay

Do not overlap with other checkpoint/import/fitting/timing cohorts. Output paths
must not exist; failures retain partial evidence and are not rankable. Runner
uses shared `fresh_workers()`, exclusive `frontier-probe.lock`, host reserve,
watchdog/swap admission and offline local-only checkpoint resolution.

```sh
# Minimal cohort: four compiler families, layer zero, ranks 0/1/2, one seed.
target/frontier-sdk-env/bin/python scripts/probe_residual_delegation.py \
  --rank 2 --seeds 1 --layer 0 --role all \
  --output target/residual-delegation-layer0.json

# Optional arithmetic/state integration gate; separate fresh serial worker.
target/frontier-sdk-env/bin/python scripts/probe_residual_delegation.py \
  --generated-only --output target/residual-delegation-generated.json

# Wider completed screen: repeat separately for layers 0, 12 and 23.
target/frontier-sdk-env/bin/python scripts/probe_residual_delegation.py \
  --rank 4 --seeds 2 --layer 12 --role all \
  --output target/residual-delegation-layer12-rank4-replay.json

# Only if native-i8 result warrants wider-residual falsification.
target/frontier-sdk-env/bin/python scripts/probe_residual_delegation.py \
  --rank 2 --seeds 1 --layer 0 --role mlp_gate_up \
  --residual-policy reference-i16 \
  --output target/residual-delegation-wide-gate-up.json
```

Remaining gates: actual private serving admission,
all-link framed byte conservation, factor publication/storage, widened native
factor execution and serving CPU/RSS unimplemented/unmeasured. No promotion
until those costs and full numeric state gates pass.
