# Exact FP64 Preparation corrections — investigation 5

## Result and scope

The subsequent [parent timing screen](tier-objectives-reference.md) checks all
outputs but measures 2.36–7.65× the installed integer CPU on its two bounded
geometries. This layout fails that cost gate; full-width stage costs remain
unmeasured. The source proof and original worker controls follow.

Bounded signed-i8 × unsigned ring16/24/32 products can use integer-exact FP64
**provided the backend uses ordinary binary64 multiply/add or FMA reductions**.
For full u32, a universal reduction block of 16,384 columns is sound. One DGEMM
replaces 2/3/4 digit streams within each admitted block/output tile. Wider source
blocks require an actual immutable-weight absolute-sum certificate. This is not
an unconditional certification of Apple's closed implementation, nor a claim
that DGEMM is faster than existing integer SIMD or Accelerate SGEMM.

Owned files only:

- `scripts/probe_exact_f64_corrections.py`
- `scripts/research_native/exact_f64_corrections/accelerate.py`
- `tests/test_exact_f64_corrections_probe.py`
- `docs/evidence/exact-f64-corrections-synthetic.json`
- this note

The initial worker performed no production registration, checkpoint runs, wheel
builds, installation, commits, further delegation or timing cohorts.

## Existing paths inspected

- `crates/pllm-core/src/digit_gemm.rs`: Apple Silicon/macOS capability gate;
  Accelerate framework, legacy LP64 `cblas_sgemm`; row-major A × B-transpose,
  alpha=1, beta=0; balanced base-256 digits with carry; 1,023-column reductions,
  1,024-output tiles. Four-Mi-element input/output cap, 64 MiB explicit workspace.
  Each magnitude <=128 product and arbitrary partial sum is strictly below
  2^24. Finite/integer checks precede i32 conversion and wrapping u32 shifted
  reconstruction. No retained float weights. The digit count is bits/8.
- `crates/pllm-core/src/kernels.rs`: immutable signed-i8 `Matrix` snapshot,
  borrowed weight slice; `wrap32` uses exact ring arithmetic with integer SIMD
  and output transposition. It does not need an FP64 magnitude bound.
- `crates/pllm-python/src/lib.rs`: digit8 binding rejects input over 16 MiB,
  decodes immutable u32 bytes, releases Python while executing native work,
  serializes a u32 result and copies into Python bytes. Wrap32 has analogous
  immutable-byte boundary.
- `python/pllm/runtime/correction_batch.py`: concatenated u32 masks, selected
  product kernel, **u32 subtraction of output mask, then ring mask**, followed
  by authenticated pack/accept. `benchmark_correction_stripes.py` agrees.
- `crates/pllm-core/src/codec.rs`: prepared input is signed-i8 x minus mask in
  wrapping ring; final prepared output adds output mask then centers the ring.
  FP64 replacement changes neither signs, mask reservation/lifecycle, nor trust
  boundaries. It only computes W r before the existing output-mask subtraction.

Standalone ctypes fixture calls Apple's real `cblas_dgemm`, no compiler or wheel
needed. Same flags/leading dimensions as SGEMM; all float64 operands contiguous,
dimensions positive and below signed-i32 limit, C disjoint, alpha=1/beta=0.
Fixture is deliberately gated to macOS arm64 like existing digit8 capability;
availability means symbol present, not numeric certification or speed.

## Whole-domain derivation

Let b be 16, 24, or 32, M=2^b−1, E=2^53, w_i ∈ [−128,127],
x_i ∈ [0,M]. Both operand conversions to binary64 are exact. **Do not reinterpret
u32 as i32**: 0xffffffff means 4,294,967,295, not −1. Each product is an integer
in [−128 M,127 M], magnitude <=128 M <=549,755,813,760 <E and exactly represented.

For a block of k columns and row j, define S_j=Σ_i |w_ji| from the immutable
source (widen before abs; int8 abs(−128) overflows). Any subset of its products
has absolute sum <=M S_j. If M S_j <=E, every leaf product, every internal sum
of disjoint subsets in any reduction order, lane accumulation, lane merge,
and final dot is an exactly represented integer. Induction also covers FMA:
its exact product-plus-partial is an integer in range, so rounding changes
nothing. Alpha=1 introduces no scaling error; beta=0 must not read uninitialized
C as an input. E itself is allowed, unlike the existing deliberately strict
SGEMM limit.

Universal certificate substitutes S_j<=128k:

| Ring | M | Universal k_max=floor(E/(128M)) | Source S_max=floor(E/M) |
| --- | ---: | ---: | ---: |
| 16 | 65,535 | 1,073,758,208 | 137,441,050,656 |
| 24 | 16,777,215 | 4,194,304 | 536,870,944 |
| 32 | 4,294,967,295 | 16,384 | 2,097,152 |

These are mathematical bounds, not allocation permissions. Probe also caps all
weight/input/output owners at 4 Mi elements, batch at 4,096, and modeled fixture
workspace at 64 MiB. Thus ring16/24 can use a whole admitted row numerically,
but geometry/workspace may still require tiles. Ring32 can use whole width
<=16,384; beyond that it splits by default, or explicitly certifies actual
source slabs. Zero rows admit any permitted width. Signed coefficient sums,
final dot magnitude, observed masks, or unverified caller bounds do NOT certify
all intermediate sums. Cancellation does not rescue an unsound width.

Only conventional reduction DAGs are covered. Arbitrary BLAS algorithms could
use transformed input/output sums (e.g. Strassen-like cross-output arithmetic),
lower precision, or compensated/scaled operations outside this proof. Apple's
symbol/API and successful tests do not document that DAG. Native promotion
requires a vendor/version numeric guarantee or an independently audited kernel
with this operation structure. NumPy BLAS carries the same conditional gate.
Finite/integer checks catch many violations but cannot catch an incorrect
integer-valued result. Parity is evidence of observed behavior, not universal
certification.

### Conversion, reconstruction, subtraction

1. Check each FP64 partial finite, integral, and within its certified bound <=E.
2. Convert to **signed i64**, whose range safely contains [−E,E]. Never convert
   directly to u32 or i32: negative products and large positive dots violate
   those conversion domains. Never exploit saturating Rust float casts.
3. Convert i64 to u32 modulo 2^32, wrapping-add block residues, then mask to b
   bits. Python fixture instead uses i64 `(previous_residue + partial) & M`.
   Its addition lies within [−E,E+M], strictly inside i64. Reduction after each
   block avoids a potentially huge unbounded raw integer total across blocks.
4. Subtract output mask in integer space before ring narrowing. Once product is
   a canonical residue, residue−s ∈ [−M,M], so i64 subtraction is safe; production
   wrapping u32 subtraction then `& M` is identical for all three rings. Reducing
   product earlier is algebraically identical to `(raw_dot−s) mod 2^b`.

If subtraction is fused into the FP64 dot instead, require M(S_j+1)<=E;
universal k <=floor((E−M)/(128M)). For ring32 that is **16,383**, not 16,384.
Tests show concrete failure:

```
dot = -128 * 16384 * (2**32 - 1) = -(2**53 - 2**21)  # exact
dot - (2**32 - 1)                                  # odd integer beyond -2**53
```

The float subtraction rounds, losing numeric identity. Separately, 16,384
extreme products plus final −1 × (2^21+1) at width 16,385 produce −(2^53+1),
which FP64 cannot represent. Both unsafe designs are explicitly rejected;
default split reconstruction passes independent big-integer oracle.

## Cost model: not a kernel-FLOP result

For R outputs, K columns, B batch, k reduction block, n output tile:

- Original i8 owner RK; input/output mask owners 4B(K+R). No persistent float
  weight copy in proposed path. A cached full FP64 owner would add **8RK**,
  coexist with original i8/page owners, and needs separate admission.
- Candidate Rust workspace: `4BR + 8(Bk + nk + Bn)`; includes ring accumulator,
  input float slab packed once per reduction block and reused across output
  tiles, temporary float weights and FP64 output. This loop order is part of
  proposal; copying input anew per output tile would multiply conversion cost.
- Fixture uses larger conservative bound:
  `max(32BR, 4BR + 8(B+n)k + 48Bn)`, plus `8nk` for on-call source scans.
  This charges float checks, int64 conversion, reconstruction/subtraction
  temporaries; prior slabs explicitly released before new allocations.
- Full weight conversion reads/writes 9RK bytes per issuance, versus SGEMM's
  5RK. Full mask conversion reads/writes 12BK bytes, versus digit8's
  `8*(b/8)*BK*ceil(R/1024)` because current digit8 repacks each digit per output
  tile. DGEMM partial checks/reconstruction visit `BR*ceil(K/k)` elements;
  baseline visits `(b/8)*BR*ceil(K/1023)`.
- Boundary allocations/copies include input serialization + Rust decode
  (8BK bytes written), output serialization + Python bytes (8BR), and initial
  weight serialization + native snapshot (2RK). Native ring output already in
  workspace; Python correction allocation adds 4BR. Mask concatenations,
  original immutable bytes, page overlaps, codec/MAC output and role ownership
  remain separate. Boundary figures are allocation/copy bytes, not all memory
  reads; conversion figures count operand reads plus converted writes.
- Source-certified mode in this fixture scans abs of packed weights each call,
  O(RK), with `8nk` extra scratch. Future cached certificates cost u64 row/block
  entries (`8R*ceil(K/k)`), initial source scan, authenticated source binding,
  and page-lifetime handling. An issuance scan cannot be omitted from timings.
- BLAS internal packing/scratch/thread stacks unknown, not zero. Existing
  digit8 role admission uses 224 MiB extra allowance; cannot assume that fits
  DGEMM or reduce it from a workspace formula. Runtime imports/worker stacks,
  cold weight setup, platform BLAS lazy initialization remain visible.
- Traffic/correction encoding unchanged: `(b/8)BR` payload bytes, existing
  headers/authentication; no claimed reduction in all-link bytes.

Synthetic `[R=256,K=4096,B=1,b=32]`: 1 DGEMM versus 20 SGEMM calls, but Rust
workspace 8,424,448 versus 1,053,692 bytes, and weight conversion 9,437,184
versus 5,242,880 read/write bytes. B=32 candidate workspace 9,535,488 versus
1,244,032 bytes. MAC/call counts are models only. Apple Silicon FP64 throughput,
packing cost and memory bandwidth may erase savings, especially decode B=1.
Candidate must beat **both** installed wrap32 and digit8 transfer-inclusive
paths, not merely a split-FLOP count.

## Tests and exact replay

Using installed `target/frontier-sdk-env`, macOS 26.5.2 arm64, NumPy 2.5.3:

- 65 tests passed, including randomized operands, −128/127/0, full unsigned
  extremes, carry boundaries, cancellation, split at width 16,385, output tiles,
  non-contiguous inputs, all rings, masks, checked conversion and rejection.
- 36 bounded parity cases: 18 each NumPy and actual Accelerate DGEMM; seed 5005,
  widths 1/7/1023/1024/16384/16385. Independent Python integer oracle uses no
  NumPy integer matmul or candidate arithmetic.
- 6 installed native controls: wrap32 and digit8 at each ring, width 1025,
  identical unsigned inputs and integer subtraction, compared to same oracle.
  Installed path/hash recorded; no source checkout shadow or fallback allowed.
- No performance, complete-response, tier, remote-MAC or SOTA qualification.

```
target/frontier-sdk-env/bin/python -m pytest -q tests/test_exact_f64_corrections_probe.py
target/frontier-sdk-env/bin/python scripts/probe_exact_f64_corrections.py \
  --output docs/evidence/exact-f64-corrections-synthetic.json
```

Evidence stores probe/test/fixture/source hashes, installed native binary hash,
platform, bounds and modeled costs. No evidence hash embedded into itself.

### Parent-only serialized timing replay

Under frontier lock and existing host admission:

```
target/frontier-sdk-env/bin/python scripts/probe_exact_f64_corrections.py \
  --timing --rows 16 --cols 1024 --batch 4 --bits 32 --repeats 3 \
  --output docs/evidence/exact-f64-corrections-timing.json
```

Four serial fresh subprocesses (NumPy, Accelerate, installed wrap32, installed
digit8), each timeout 90 s. No concurrent cohort. Each repetition validates
against independent oracle outside timed region. Records separate setup CPU/
wall time, transfer-inclusive issuance CPU/wall time and process max RSS.
First issuance includes lazy initialization; no warmup silently discarded.
Single-thread environment requests and integer executor threads=1 do not prove
vendor thread count; verify externally. Installed control requires actual native
extension, build=False and exact installed path.

Default case has 65,536 MACs per native multiply; three issuance repeats plus
oracle, four processes serial. Explicit candidate workspace ~161 KiB, original
i8 and masks ~32 KiB; Python/NumPy/SDK imports and vendor BLAS dominate RSS.
Reserve existing >=224 MiB platform allowance **plus measured environment and
host headroom**, not a tiny algebra-only admission. Timing code caps oracle at
2 million MACs, dimensions at R<=1024,K<=16384,B<=64, repetitions<=5. Larger
representative cohorts/checkpoint scans require separate parent admission.
Fixture's emulated byte boundary and Python checks are not a compiled Rust
implementation, so native SDK versus fixture timing is exploratory, not a fair
production speedup measurement. It excludes network/complete Preparation.

## Native registration plan and outstanding gates

Worth a gated prototype, not registration now:

1. Pin/audit Accelerate implementation's binary64 conventional reduction
   semantics, or provide a controlled binary64 kernel. Re-run adversarial parity
   for each OS/BLAS/CPU revision. Performance runs need chip/OS/BLAS identity,
   effective threads, peak RSS, serial matched fresh replicates, cold/setup and
   issuance, both prefill and decode widths/batches.
2. If warranted, add private Rust module following SGEMM FFI with f64 alpha/beta,
   checked dimensions, overflow-safe workspace admission, i64 checked conversion
   and wrapping ring reconstruction; no float-space subtraction. Keep immutable
   weight certificate internal, optional, and never trust user-supplied proof.
3. Expose distinct capability and opt-in kernel, never silently route digit8 or
   wrap32 to it. Python immutable-byte/GIL-detach binding, capability contracts,
   preparation selector/schema and paged execution all require parent-owned
   edits. Retain existing exact fallback/rejection and pricing. Source-certified
   page bounds must bind to authenticated resident snapshot, not unchecked files.
4. Compare compiled candidate transfer-inclusive to integer SIMD and digit8,
   include conversion/packing/workspace and lifetime overlaps. Only then admit
   representative stage/checkpoint runs and complete-client responses. Preserve
   >=90% remote body MACs independently for prefill/decode; qualify Low/Medium
   using maximum whole-client RSS over matched fresh replicates. Preparation
   CPU/RSS, client CPU/RSS, TTFT, N−1 decode rate and all-link traffic still gates.

Numerical negative result: universal >16,384 ring32 reductions and fused FP64
subtraction at width 16,384 are unsound. Positive result: conventional FP64 block
algebra and observed native synthetic parity. Hardware speed/RSS advantage and
closed-vendor universal soundness remain unestablished.
