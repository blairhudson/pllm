# Preparation-owned subset correlation compiler

This independent bounded probe compiles a public signed-i8 matrix into subset
tables, then produces fresh exact ring32 correlations `(r, s, W*r-s)`. It tests
one concrete function-specific setup amortization, not a new PCG security result.
No checkpoint values, decoder, transport or production role handles are involved.

## Construction and ownership

Partition input coordinates into blocks of at most `b` coordinates. Store all
`2^b` subset sums of each block's output vectors. Signed-i16 is exact for `b<=12`
even for weights -128. Split a fresh uniform-looking ring32 mask into 32 bit
planes; one subset lookup per block and plane reconstructs `W*r mod 2^32`.
Two independent OS-generated 256-bit keys, separately domain/context-bound through
SHAKE256, expand `r` and `s`. Trusted Preparation computes the correction; client
would receive both keys; Inference receives **only** `W*r-s`. Neither key, mask nor
table index goes to Inference. Tables stay in Preparation, unlike client-owned
Maverick P/Q. Fresh issuance requires fresh keys and binding; cancellation and
malformed replies consume test-local material.

Privacy rests on the existing trusted-Preparation/non-collusion contract plus
SHAKE256 pseudorandomness, not ordinary seed compression magically creating
correlations. Preparation still performs matrix-dependent expansion for every
sample. Python test handles are clonable and cannot securely erase all copies;
this is explicitly **not production ownership**, malicious verification or an
authenticated distributed protocol. A modified reply is not guaranteed to reject.
No material is persisted in reports. Seed delivery is a priced proposed role
contract, not implemented transport: local tests directly hand expanded masks
to their client object. Issuance measures one local expansion; additional remote
client expansion is counted symbolically, with client CPU/RSS unknown.

## Exact costs and decisive boundary

For block widths `w_k`, tables occupy `2*m*sum(2^w_k)` bytes. Setup performs
`m*sum(2^w_k-1)` subset-sum lane additions. Every fresh sample performs
`32*m*number_of_blocks` lane accumulations, plus `32*n` index bit extractions,
mask expansion and output subtraction. Dense evaluation uses `m*n` MACs.
Measured controls use installed native SIMD `wrap32` and, when advertised by the
installed build, native exact `digit8(...,32)`. Native module import, worker-pool
creation and immutable matrix snapshot are charged in setup. Input/output
conversion remains inside each issuance. Synthetic fixture creation is separately
charged; NumPy and Python integers remain independent oracles, not timing controls.
Operations differ: these counts are **not CPU ratios**.

For bounded `b<=12`, subset accumulation count exceeds dense MAC count. Thus
there is no arithmetic-count amortization horizon under a one-operation-per-MAC
comparison. Timing could differ by implementation; serial measurements include
setup before deriving empirical horizons against each native control. A 100x gain cannot
come from replacing these counts with zero-cost seed expansion.

Raw per-sample delivery is 64 key bytes Preparation→client and `4*m` correction
bytes Preparation→Inference, before authenticated headers. Client expands
`4*(n+m)` bytes; Preparation expands the same, separately. Client online masking
and reconstruction add `n+m` ring operations, unchanged from dense issuance.
Cold checkpoint delivery, authenticated framing and complete response metrics
remain unknown, not zero. Tables have no delivery channel because ownership never
moves. There is no free client key expansion measurement in this probe.

## Admission and geometry

`--config PATH` derives all distinct remote shapes from ordinary compiler lowering
and schedule ownership. No shape is hardcoded into the probe. Omit it to resolve
the existing pinned Qwen fixture **offline**, config only. Gate/up and down shapes
are retained even on rejection. At its known dimensions, 8-bit blocks reject both
MLP shapes; 4-bit blocks admit their geometry under the 256 MiB estimate. 12-bit
blocks reject even the square body stage. Tests lock these regression fixtures,
not model-specific compiler logic.

Admission includes public i8 weights, tables, independent oracle's transient i64
weight copy, expansion/output/scratch buffers and 96 MiB interpreter/allocator
allowance. Native controls price snapshot/serialization and oracle copies separately;
digit8 also prices float32 conversion workspace.
This is a conservative estimate, not a peak RSS measurement or cache admission
for all stages simultaneously. Serial run handles one stage at a time; sum tables
times compiler multiplicity describes full-model retained setup storage. Setup
discard/rebuild and disk streaming are not implemented or charged as free reuse.

Prior results remain unchanged: Maverick's 22.90 GB P+Q gate and uncertified
BabyBear distance; Secret Replication's full-domain expansion; projected
polynomial coefficient compression without full-model numeric parity; seeded
masked-wire keys retaining explicit corrections. Public low-dimensional mask
bases and sparse shift refresh leak cross-input information. This method neither
uses those bases nor weakens fresh full-ring masking to reduce its table cost.

## Two-stage replay

Lightweight independent integer, all-ones/full-ring, partial-block, fresh/domain,
malformed/cancel/replay and table-size checks:

```sh
/Users/blairhudson/workspace/pllm/target/frontier-wheel-env/bin/python -m pytest --noconftest -o pythonpath='' -q tests/test_correlation_compiler_probe.py
```

After coordinator authorizes exclusive measurements:

```sh
/Users/blairhudson/workspace/pllm/target/frontier-wheel-env/bin/python scripts/probe_correlation_compiler.py --serial --processes 3 --samples 4 --output docs/evidence/research-correlation-compiler-serial.json
```

Uses existing installed wheel and NumPy, no Rust/build/install. At least three
fresh processes **per mode and retained geometry**; four queries per process are
not process replication. Two-mode cells alternate AB/BA order; three-mode cells
use balanced Latin rotations. Following cells reverse base order. Modes share
public synthetic weight identity, geometry, ring and sample count, not mask reuse.

Full-shape timing retains only 4-bit blocks: higher widths reduce query additions
but still fail the dense arithmetic-count target while increasing table storage
and setup exponentially. Their counts, admission and explicit tradeoff pruning
gates remain in the report; **no timing dominance is asserted**. Retaining minimal
table/setup widths bounds this negative-gate cohort to five geometry cells (four
full shapes plus small control).

Hashes freeze probe/helpers, tests, actual **imported** `coded_resources` and native
wrapper/support paths, and installed extension bytes before cohort. Workers check
before/after execution; parent checks after cohort. Installed BuildRef is separately
held stable by the shared helper. Native absence blocks measurement; unsupported
digit8 is explicitly gated, never silently replaced by NumPy.

Rejections occur before weights. Shared frontier lock, 512 MiB host headroom and
swap watchdog apply. Estimated child peak below 256 MiB; allow approximately 2–8
minutes total, 30–45 fresh processes. Timing is Preparation issuance/component
evidence, not seven-metric whole-response evidence. Eleven lightweight tests pass,
including both available native kernels. Authorized serial results follow below;
no whole-response improvement claimed.

## Authorized serial result: negative Preparation gate

Evidence: `docs/evidence/research-correlation-compiler-serial.json`. Existing
installed native wheel; Python 3.13, synthetic signed-i8 values at compiler-derived
dimensions, exact ring32. Source lock
`383f5ed6947ba56ceed6083a9503d6302f09559e1e57ee614409900b36084c29`
and config SHA-256
`18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45`
bind geometry, **not checkpoint-value measurements**. Both native control modes
were available and executed. Native SIMD uses four threads; native digit8 includes
its exact conversion/SGEMM/reconstruction path. Candidate is Python/NumPy subset
lookup, not an optimized native subset kernel.

**45 distinct fresh worker PIDs**, three per geometry/mode, execute five geometry
cells with balanced Latin rotations. Each process checks one fresh correlation
against independent NumPy masked-input evaluation and Python integer clear output,
then times four other fresh correlations. That is **45 independently checked
correlations / 114,624 output values**, plus **180 timed fresh correlations**;
timed samples are not counted as additional independent oracle checks. Full gate/up
`9728×896` contributes 87,552 checked values; full down `896×4864` contributes
8,064. The other three geometries contribute 19,008. Every checked value matches.
All 96 compiled body stages have an admitted 4-bit geometry; this is distinct-shape
coverage, not execution of all 96 checkpoint matrices or a decoder response.

Each timing below is median across three **process medians**, not twelve queries
treated as independent process replicates. Setup includes native import and
immutable snapshot, or table compilation. RSS is median process lifetime peak,
including independent oracle and source validation, not online-only memory.

| Output × input | Mode | Setup ms | Issuance wall ms | Issuance CPU ms | Peak RSS MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| 64 × 64 | Native SIMD wrap32 | 3.414 | 0.106 | 0.324 | 44.44 |
| 64 × 64 | Native digit8 | 3.660 | 0.031 | 0.032 | 44.34 |
| 64 × 64 | Subset, 4-bit | 0.329 | 0.871 | 0.871 | 43.00 |
| 896 × 896 | Native SIMD wrap32 | 4.039 | 0.174 | 0.524 | 47.67 |
| 896 × 896 | Native digit8 | 6.276 | 0.674 | 0.668 | 50.67 |
| 896 × 896 | Subset, 4-bit | 10.936 | 27.522 | 27.354 | 51.53 |
| 1152 × 896 | Native SIMD wrap32 | 3.875 | 0.184 | 0.483 | 55.44 |
| 1152 × 896 | Native digit8 | 4.596 | 0.394 | 0.394 | 55.41 |
| 1152 × 896 | Subset, 4-bit | 11.656 | 27.890 | 27.838 | 62.20 |
| 896 × 4864, down | Native SIMD wrap32 | 5.660 | 0.388 | 1.101 | 90.30 |
| 896 × 4864, down | Native digit8 | 9.085 | 3.307 | 3.278 | 93.89 |
| 896 × 4864, down | Subset, 4-bit | 60.845 | 137.108 | 136.629 | 118.48 |
| 9728 × 896, gate/up | Native SIMD wrap32 | 9.551 | 0.744 | 1.999 | 145.27 |
| 9728 × 896, gate/up | Native digit8 | 13.260 | 3.184 | 3.169 | 145.19 |
| 9728 × 896, gate/up | Subset, 4-bit | 94.793 | 126.182 | 120.849 | 202.88 |

Synthetic fixture construction, separately recorded in evidence, costs median
9.052/12.725/11.939/21.017/31.902 ms respectively for small/square/QKV/down/gate-up
shapes across all nine processes. It includes fixture RNG initialization, so must
not be presented as checkpoint import cost. Real checkpoint import/distribution,
remote client expansion, framing, peer transport and complete serving metrics
remain unknown and excluded from these component comparisons.

**No finite steady-state setup-amortization horizon against either native control**
at any measured cell: subset issuance is slower in every process-median comparison.
At full shapes it is **151.80–353.22× slower than native SIMD**, and 39.62–70.86×
slower than native digit8. Native SIMD wins every full-shape wall comparison;
digit8 wins small control, not these single-row full stages. The tiny candidate's
lower setup cost is not a sustained issuance improvement. At every full shape,
candidate setup is also higher and candidate median lifetime RSS rises by
3.86–57.61 MiB relative to SIMD. Maximum observed child RSS is **203.094 MiB**;
watchdog observes **zero swap growth**. No cell exceeded the declared 256 MiB
estimate ceiling. Source/extension hashes and installed BuildRef remained frozen;
no cohort execution failure occurred and no earlier report was replaced.

Full-model table projection is **2,862,612,480 bytes** at 4-bit blocks, before
weight/scratch/runtime ownership. Higher-width gates remain in the evidence:
8-bit tables project 22,900,899,840 bytes and reject both MLP geometries; admitted
square/QKV cells are pruned from timing on declared setup/storage cost grounds.
12-bit tables project 242,819,432,448 bytes and reject all full geometries.
Those pruned tradeoffs are not proven slower by timing. These sums charge each of
24 compiler occurrences per full shape; measuring only distinct shape tables does
not make the other model stages' setup or storage free.

### Next gate

**Do not promote this subset-table representation as a Preparation optimization.**
The negative result is scoped to this bounded exact-ring representation and Python
prototype, not every model-specific PCG. A further native implementation would
need a credible mechanism overcoming eight subset lane accumulations per dense
MAC and full-model table/cache cost, then the same native controls, independent
full-shape oracle and setup-inclusive gate. A true compact private correlation
compiler additionally needs reviewed role-separated key security, production
one-use ownership and expansion costs—not public bases, mask recurrence, or
secret seeds relabeled as free matrix-dependent correlations.
