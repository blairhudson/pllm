# Investigation 6: bounded limb-stationary corrections

## Finding and scope

**Resident, multi-tile opportunity only. Current paged and single-tile stages
already pack once.** The subsequent [parent C screen](tier-objectives-reference.md)
measures only 1.72–2.59% lower isolated CPU, with every output checked against
installed native integers. No whole-response or tier improvement is established.

Inspected actual implementation before writing reference:

- `crates/pllm-core/src/digit_gemm.rs:93-142`: native CPU Accelerate SGEMM;
  reduction block 1023, output tile 1024, loops `block -> tile -> digit`.
  Weight float packing is already reused across digits within one tile. Input
  balanced digits are packed anew for every output tile. Not resident-stationary.
- `crates/pllm-core/src/paged.rs:232,287,450-469,485-494`: constructors choose
  `page_rows = ((1 << 20) / cols.max(1)).clamp(1,512)`. Each decoded page gets
  its own call to the resident kernel; each such page fits one output tile.
  Thus each mask digit slab is **already packed once per current constructor
  page**. Proposed within-page schedule saves zero packing work. Externally
  supplied valid artifacts may have larger pages; only those with >1024 rows
  offer within-page savings. Artificial larger pages in tests are geometry
  controls, not a claim about ordinary paged execution.
- Reuse across pages would require retaining more mask digits, or changing
  `execute` to reread/decode pages for each reduction block. Neither is this
  proposal; both change ownership/I/O and require separate investigation.
- `crates/pllm-python/src/lib.rs:1443-1459` copies input bytes to a Rust u32
  vector and serializes returned vector to Python bytes. Installed binding has
  a 16 MiB input bound. Transfer/copy cost cannot disappear from comparison.
- `python/pllm/runtime/benchmark_memory.py:354-359` admits **224 MiB extra per
  active digit8 stage**, including 64 MiB native tiles, binding/page overlap and
  64 MiB platform BLAS allowance. This investigation never changes admission.

Only assigned probe, tests, note, evidence and unique standalone C fixture added.
No SDK, registration, shared build, checkpoint, install, commit or child agent.

## Schedule, identity, ownership and preflight

Candidate loops `block -> pack D digits -> tile -> digit`, D = bits/8. Each block
owns at most D float slabs of shape batch x min(cols,1023). One weight tile and
one product tile remain unchanged. No full-weight float copy, persistent mask
cache, all-column mask digit cache, new client weights or cross-request reuse.
Even when cols <=1023, the slab covers the input only because the entire input
fits the existing bounded reduction block. All owners retire on call exit.

Balance exactly matches Rust: for shift s, carry is accumulated
`sum(128 << previous_shift)`; digit is
`((((uint64(x)+carry)>>s)+128)&255)-128`. Carry remains u64 (including ring32
top-digit overflow). Signed weights remain i8. Products and reductions obey
`1023 * 128 * 128 = 16,760,832 < 2^24`; finite/integer/absolute-bound checks remain.
Each signed result becomes a wrapping u32 add shifted by s; apply ring16/24
mask only at end. No change to block bounds or SGEMM arithmetic/reduction order.

Let M=batch, R=full output rows, C=input columns, L=rows of current page (R for
resident), K=min(C,1023), N=min(L,1024), I=4MC and O=4MR. Candidate native owner
bound is

```
W = 4 * (M*L + D*M*K + N*K + M*N) <= 64 MiB
```

Baseline uses `M*K` instead of `D*M*K`. Conservative overlapping extra bound is

```
resident: W + 3I + 3O + 64 MiB + page_transient_bytes
paged:    W + 3I + 4O + 64 MiB + page_transient_bytes
```

Paged W includes page-local output; extra also charges full paged result and
binding/serializer owners. `page_transient_bytes` must come from actual pager
metadata (`stored + 2*decoded + 1` compressed, `2*decoded` raw), not a synthetic
uncompressed estimate. Resident immutable signed weights are separate base
ownership; pager metadata/file owner and all other runtime owners also stay in
existing whole-role admission. Conservative payload model is not a bound on
Python interpreter/allocator or whole-process RSS. Synthetic reference models
pages as views, **not authenticated disk pages**; its paged accounting remains
conservative for raw decode even though those synthetic views do not allocate.

Preflight checks integer geometry, 1..4096 rows per batch, ring16/24/32, domain,
input/output <=4,194,304 elements, W and extra <=224 MiB before allocating output.
One-use synthetic owner is claimed only after domain/admission checks, before
output allocation; any post-claim failure leaves it spent. No retry, mask reuse
between sessions or output-dependent branching added. Synthetic baseline and
candidate comparisons use separate test-only tickets, not production randomness.

Production integration would select unchanged baseline for <=1 output tile per
page, or if extra slabs fail either existing cap. Example: batch4096, rows1024,
cols1023, ring32 passes original native workspace but candidate fails 64 MiB;
do not relax admission. Entire-stage paged preflight would need all page metadata
before the outer result allocation; existing `execute` allocates that result
before per-page kernel validation. This reference does not silently fix shared
SDK behavior or claim that stronger production guard already exists.

## Deterministic work and likely bottleneck

For P pages and T=sum(ceil(page_rows/1024)) output tiles:

| Work | Baseline | Candidate |
| --- | ---: | ---: |
| Mask elements converted to float | D*M*C*T | D*M*C*P |
| Weight elements converted to float | R*C | R*C |
| SGEMM calls | D*ceil(C/1023)*T | unchanged |
| Float MAC products | D*M*R*C | unchanged |
| Reconstruction element updates | D*M*R*ceil(C/1023) | unchanged |
| Extra float slab bytes | 0 | 4*(D-1)*M*min(C,1023) |

Resident test geometry batch32, rows4096, cols1024, ring32:

- Mask packing 524,288 -> 131,072 elements (4x smaller), not 4x total runtime.
- Weight packing 4,194,304; 32 SGEMMs; 536,870,912 MACs; 1,048,576 reconstruction
  updates all unchanged.
- Native bytes 4,976,512 -> 5,369,344; delta 392,832 bytes.
- Candidate conservative extra 74,444,288 bytes; base signed weights 4,194,304.
- **Still admit 234,881,024 bytes (224 MiB) per active stage**, not 74,444,288.

For batch1 decode at same R,C, mask savings are only 12,288 conversions versus
4,194,304 weight conversions. Weight packing, small-GEMM overhead, reconstruction
and binding copies likely dominate decode. At larger batch, SGEMM dominates and
larger slab working set can harm cache behavior. Paged decode additionally keeps
page I/O/authentication/decompression with zero ordinary within-page savings.
Whether resident mask packing is significant remains a profiling/native gate.

## Executed small checks

Exact replay from repository root (installed validation interpreter; no SDK
imports or checkout shadowing):

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 target/frontier-sdk-env/bin/python -m unittest discover -s tests -p test_limb_stationary_corrections_probe.py -v
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 target/frontier-sdk-env/bin/python scripts/probe_limb_stationary_corrections.py --output docs/evidence/limb-stationary-corrections-synthetic.json
```

Final test output: **Ran 7 tests in 0.083s; OK**. Probe: **15 ring/geometry/page
cases plus 9 coherent adversarial cases passed**, both schedules independently
checked against signed Python integer oracle. All three rings, tile1025 tail,
block1024 tail, page tails, -128/127 weights, max masks, digit carry boundaries,
coherent near-bound reductions, random signed controls, deterministic work,
single-tile paged negative, one-use and spent-on-failure controls, rejection
before output allocation and unchanged caps included. Test elapsed time is not
a performance result. Evidence contains no native/complete-response timing.

## Serial parent-only native gate

First acquire existing frontier lock and host admission. Run on Apple Silicon
with existing compiler/Accelerate only. No package install or whole-wheel build.
Standalone fixture mirrors baseline/candidate schedule but is **C, not installed
Rust SDK**, resident only. Output allocation, input ndarray->bytes->native copy,
output bytes->ndarray copy are timed. Weight construction, synthetic mask
generation, dynamic-library import and independent oracle are excluded from
timing; separately account cold setup before complete-response qualification.
Native C does not enforce production mask reservation/authenticated pager.

```sh
mkdir -p target/limb_stationary_corrections
cc -O2 -Wall -Wextra -dynamiclib scripts/research_native/limb_stationary_corrections/fixture.c -framework Accelerate -o target/limb_stationary_corrections/fixture.dylib
```

Small full-oracle native gate, each command serial (all rings, both tail axes):

```sh
for bits in 16 24 32; do
  VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 target/frontier-sdk-env/bin/python scripts/probe_limb_stationary_corrections.py --native-library target/limb_stationary_corrections/fixture.dylib --batch 2 --rows 1025 --cols 3 --bits "$bits" --repeats 1 --output "target/limb_stationary_corrections/native-tail-rows-$bits.json"
  VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 target/frontier-sdk-env/bin/python scripts/probe_limb_stationary_corrections.py --native-library target/limb_stationary_corrections/fixture.dylib --batch 2 --rows 3 --cols 1024 --bits "$bits" --repeats 1 --output "target/limb_stationary_corrections/native-tail-cols-$bits.json"
done
```

Then bounded transfer-inclusive prefill-like and decode-like commands:

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 target/frontier-sdk-env/bin/python scripts/probe_limb_stationary_corrections.py --native-library target/limb_stationary_corrections/fixture.dylib --batch 32 --rows 4096 --cols 1024 --bits 32 --repeats 3 --output target/limb_stationary_corrections/native-prefill.json
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 target/frontier-sdk-env/bin/python scripts/probe_limb_stationary_corrections.py --native-library target/limb_stationary_corrections/fixture.dylib --batch 1 --rows 4096 --cols 1024 --bits 32 --repeats 3 --output target/limb_stationary_corrections/native-decode.json
```

Resource estimate: compiler one tiny translation unit, reserve 256 MiB host
headroom (estimate, not measured). Each default benchmark sequentially performs
six native calls totaling 3,221,225,472 float MACs; one thread requested. Reserve
224 MiB stage extra plus 4 MiB signed weights plus 128 MiB Python/NumPy/harness
headroom (~356 MiB process reservation), and independent OS/compiler headroom.
Fixture hard caps weights at 16 MiB, input/output at 16 MiB each, repeats at 5;
all overlaps/caps remain preflighted. Small native checks have <25K float MACs per
schedule call. No compilation or native timing performed by worker. Replicate
fresh processes for independent peaks; in-process alternation is only a screen.
Output includes source/library hashes, native baseline/candidate output digest
identity, outside-timing oracle scope, wall/process CPU and combined peak RSS
(macOS ru_maxrss bytes). Large-geometry oracle checks two batch rows x two output
columns; small gate checks every output independently. No network timing, actual
Rust speed claim, authenticated pager measurement or full-response result.

## Complete cost gates before promotion

1. Native fixture compile/full-oracle checks first; then transfer-inclusive
   timings, cold/warm separation and cache/regression profile. C speedup alone
   cannot establish Rust SIMD/Accelerate packing speedup.
2. Parent-authorized native Rust integration/admission audit, preserving exact
   carries, output wrap, existing fallback, BLAS checks and one-use failure
   lifecycle. Check resident and actual authenticated paged stage output identity;
   ordinary <=512-row pages should remain unchanged baseline, not inflated D slabs.
3. Matched complete responses: same weights/quantization, topology, masks' trust
   boundary, phase placement, threads, source/native hashes, prompt/output lengths,
   seed controls and cold/base/output-dependent work. Measure client CPU and
   whole-client RSS, Preparation CPU/RSS, TTFT, N-1 decode throughput and all-link
   bytes separately for prefill/decode. Include mask generation, codec/copies,
   weight read/auth/decode, scheduling/upload and correction application.
4. Preserve >=90% remote body MACs independently in prefill and decode. Mechanism
   changes no placement, body count or wire format; zero traffic reduction is
   expected, not measured. Extra remote digit GEMMs are not body-MAC placement.
5. Charge all simultaneous stage owners, BLAS/thread stacks, page transients,
   resident base weights, bindings, histories, concurrency and parent host
   headroom. Keep 224 MiB extra per active stage/stage-window reservation intact.
6. Fresh-process matched replicates: qualify using largest whole-client peak
   <=268,435,456 bytes low or <=2,147,483,648 medium. Conservative runtime admission
   and host headroom remain independent. Preparation-only speedup/bytes never
   qualify a whole-client tier, latency/traffic win or SOTA.

All six gates remain open except local synthetic identity/ownership/count screen.
No measured native bottleneck, cost reduction, traffic saving or tier/SOTA claim.

## Hashes

Evidence records full source/probe/test/fixture/native-source SHA256 values.
At final worker validation:

- Probe: `38752ba4869297dc15796169ac579338e4ada2eead2b03125aa95344d3f2be50`
- Tests: `5a3d7cc644a7300cb21618fe76fcdb1c6699a74f52346c930446e71b707f0336`
- Fixture: `ee9c01f1a6063e2a4bc1d56c5b228c42515af2decd23e91085b3b2f08695db6b`
- Evidence: `17d7d86be055f985d0c0290e4b205552c2f86de32430343327667e69da986f29`

Recompute including note itself (self-hash intentionally not embedded):

```sh
shasum -a 256 scripts/probe_limb_stationary_corrections.py tests/test_limb_stationary_corrections_probe.py scripts/research_native/limb_stationary_corrections/fixture.c docs/evidence/limb-stationary-corrections-synthetic.json docs/research/limb-stationary-corrections-reference.md
```
