# Certified digit8 reduction blocks: bounded reference

Investigation 1 of `docs/plans/tier-objectives-ten.md`. **Viable exactness
mechanism; isolated prefill improvement with setup/memory costs.** Parent completed
the compiled adversarial check and 24-process pinned-source cohort: 6,494,208
stage outputs agree, 150-row C issuance CPU falls 9.99%, and one-row work regresses.
Setup and process RSS rise. Results, frozen evidence and remaining SDK/tier gates
are in the [parent consolidation](tier-objectives-reference.md).
The initial worker and provisional scan records below retain their original scope.

## Actual source and numeric contract

Inspected `crates/pllm-core/src/digit_gemm.rs`, `kernels.rs`, `paged.rs` and
`python/pllm/runtime/transformer_engine.py:1663–1694` before implementation.

* Native digit8 uses fixed `BLOCK=1023`, output `TILE=1024`, float32 SGEMM with
  alpha=1/beta=0, signed balanced base-256 digits, and wrapping u32 reconstruction.
  Every output must be finite, integral and have magnitude **strictly below**
  2^24. Inputs/outputs each admit at most 4,194,304 elements; batch 1–4096;
  rings 16/24/32 bits; explicit kernel workspace at most 67,108,864 bytes.
* `Matrix` retains a global maximum weight magnitude, not per-row/per-block
  absolute sums. Digit8 does not even use that maximum. Paged digit8 decodes
  each page, invokes this same kernel, and copies each page result into a
  separately owned full output. **Current digit8 does not exploit certificates.**
* Preparation expands fresh input and output masks, computes the product, then
  subtracts output mask using uint32 arithmetic, masking narrower rings. Existing
  fingerprint, weight commitment, quantization and profile validation precedes
  this operation. No proposed certificate bypasses those checks.

### Proof, including all partial sums

For output j and contiguous reduction block B, let
`L(j,B) = Σ(i∈B) abs(int16(w[j,i]))`. Widen before abs: `abs(int8(-128))`
is not 128. Balanced digits satisfy `-128 ≤ d ≤ 127`, at **every** limb,
including the highest limb. Thus each product is an exactly representable
integer with `|w*d| ≤ 128*|w| ≤ 16384`.

Admit a block only if **every** output row in its tile has
`L(j,B) ≤ floor((2^24-1)/128) = 131071`. For any subset S of its terms,
including an arbitrary reordered/tree-reduction partial,

```
|Σ(i∈S) w[j,i]*d[i]| ≤ Σ(i∈S) |w[j,i]*d[i]|
                       ≤ 128*L(j,B) ≤ 16777088 < 16777216.
```

All these integers are exactly representable in binary32. Induction on any
ordinary dot-product addition/FMA reduction tree establishes exact products
and intermediates, not merely an accidentally integral final result. This uses
the same conventional SGEMM dot-product semantics as the current kernel; it
does not certify Strassen-like transforms or an undocumented low-precision
backend. Alpha=1/beta=0 adds no unbounded previous float output. Native postcheck
alone cannot detect all rounding mistakes; the source certificate is essential.

For constant magnitude a>0 the bound is `floor(131071/a)` columns. At a=128
it remains 1023; a=127 allows 1032; a=32 allows 4095. Although exactly 2^24 is
representable, 1024 terms of `(-128)*(-128)` violate the existing strict native
postcheck, so this investigation deliberately does **not** admit that endpoint.
Cancellation and average row magnitude are never substituted for individual-row
absolute sums. One extreme row constrains its entire output tile.

For limb shift s, existing carry is
`C_s = 128 * Σ(t=0..s/8-1) 256^t`, and

```
d_s = ((floor((x + C_s)/2^s) + 128) mod 256) - 128
Σ(s=0,8,..,b-8) d_s*2^s ≡ x (mod 2^b).
```

`x+C_s` uses u64; at s=24, carry is 8,421,504 and maximum addition
4,303,388,799, well below u64 overflow. A negative exact SGEMM result safely
fits i32, converts to its u32 residue, then shifts/adds with wrapping u32.
Reducing modulo 2^16 or 2^24 after u32 wrapping gives the required smaller-ring
residue. Reference shifts use uint64 (largest shifted residue below 2^56),
then mask. Independent oracle instead directly sums `int(w)*int(x)` using Python
unbounded integers, subtracts output mask, and reduces modulo 2^b.

Correction is `(Wx - output_mask) mod 2^b`, **not** subtraction between float
or prematurely centered outputs. Optional centered decoding maps residue r to
`r` below 2^(b-1), otherwise `r-2^b`; midpoint maps to `-2^(b-1)`.

## Initial Python mechanism and ownership

`scripts/probe_certified_digit_blocks.py` implements:

1. Public-source-only planner: greedy longest contiguous blocks per output tile,
   bounded by every row's absolute sum and a workspace width cap. No mask,
   exception weights, secret-dependent boundaries, approximate arithmetic or
   persistent float weight copy. Different output tiles may have different cuts.
2. Shape/ring validation, source-byte SHA256 binding, full deterministic
   recertification before execution, and rejection of altered bounds, coverage,
   costs or source. Immutable native ownership remains a future integration gate;
   this local Python array must not be mutated concurrently with execution.
3. Float32 NumPy digit-GEMM execution and an independent integer oracle. This is
   **not the proposed Accelerate kernel** and establishes no native timing.
4. Reproducible synthetic evidence and an optional source-locked scan command.

Planner bounds: contiguous signed-i8 output-by-input source, dimensions at most
16384, at most 16,777,216 source weights, native batch/input/output bounds above.
Planning is O(output_rows × input_columns) weight inspection plus hashing.
In native integration, store 24-byte block records, reserve worst-case one record
per column per output tile, and charge 32 bytes/tile-row planner scratch. The
width cap fits both metadata/scratch and kernel arrays inside 64 MiB. Explicit
prospective native workspace is:

```
4*(batch*output_rows + batch*max_block + tile_rows*max_block + batch*tile_rows)
+ 24*input_columns*number_of_tiles + 32*tile_rows.
```

This reservation is conservative; it is **not a bound on Python ndarray/dict
heap or BLAS-private workspace**. Binding input/output copies, expanded mask
owners, correction owner, source i8 owner, paged decoded/encoded overlap and
BLAS-private allocations are separate. No global client-memory credit claimed.
For zero weights, numerical certificates alone permit arbitrarily long blocks;
the workspace cap still splits them. Tests exercise this case at 1024×16384,
batch=256, without executing that large GEMM.

Fixed tile partition: greedy cuts minimize block count under these monotone
source/width constraints. They do not globally optimize output tiling or BLAS
performance. Reducing tile rows can free workspace but increase input digit
packing and calls. For actual paged execution, recertify each decoded page with
its true row boundary; whole-stage counts are not a measured paged schedule.

## Synthetic findings

`docs/evidence/certified-digit-blocks-synthetic.json` contains exact hashes,
seed=20261014, 90 independently checked modular correction outputs, five
4097/4098-column fixtures across all three rings, and larger **count-only**
fixtures. All-minus128, all-one, cancellation, sparse extremes and mixed full-i8
fixtures expose both opportunities and no-improvement cases.

For a resident/single-page 1024×4864 matrix, batch=150, ring32:

| Public synthetic weights | Block widths | SGEMM calls | Partial output reconstruction elements | Explicit candidate workspace |
| --- | --- | ---: | ---: | ---: |
| Native fixed control | 1023,1023,1023,1023,772 | 20 | 3,072,000 | 6,032,808 B |
| Constant -128 | same | 20 | 3,072,000 | 6,182,312 B |
| Constant 32 | 4095,769 | 8 | 1,228,800 | 20,608,424 B |
| Constant 1 or 0 | 4864 | 4 | 614,400 | 24,219,648 B |

The latter two reduce call/reconstruction counts 60%/80%, **not measured CPU**.
All candidates still execute 2,988,441,600 dense products, convert 4,980,736
weight elements, and pack 2,918,400 mask-digit elements. Weight conversion and
total mask-digit packing do not decrease at fixed output tiling: every element
still participates exactly once per corresponding loop. Fewer slabs/calls may
reduce fixed overhead, but larger slabs increase peak workspace and could lose
cache locality. All-minus128 adds metadata with no work reduction. Source scan
must establish real contiguous sums before performance speculation.

Unframed correction words remain 614,400 bytes for that ring32 example; input/output
fresh mask owners remain 3,532,800 bytes. Placement, dense body MAC distribution,
traffic, freshness, inventory lifetime and protocol messages are unchanged by
this algebra. Actual prefill/decode ≥90% remote-MAC shares and all-link traffic
still require compiled-topology/complete-response verification.

## Tests and exact replay

Run from repository root, using installed environment:

```sh
target/frontier-sdk-env/bin/python -m pytest -q tests/test_certified_digit_blocks_probe.py
target/frontier-sdk-env/bin/python scripts/probe_certified_digit_blocks.py --output docs/evidence/certified-digit-blocks-synthetic.json
```

Initial source-only replay: **20 passed in 0.69 s**, synthetic replay succeeded. Tests cover
strict boundary 1023 vs 1024 extreme products; final cancellation counterexample;
last admissible L1=131071; tile-constraining outlier row; all 65,536 low-two-digit
carry transitions in each ring; every subset partial of extreme products;
float32 execution versus integer oracle for carries, negative weights, wrap and
centered output subtraction; workspace-only splitting of zeros; tampered source,
bounds, coverage, cost and ring; malformed geometry/noncanonical masks; scan
denied without explicit admission. No checkpoint values were read.

Evidence includes hashes of probe, tests and inspected native/runtime sources.
Native builds, compiled numeric comparisons and timed response cohorts were
absent from this initial worker validation. The subsequent compiled stage checks
and published rescan are recorded in the parent consolidation; response evidence
remains absent.

## Source scan replay (parent only)

Supply an **independently retained** public source lock, not a digest manifest
manufactured from whatever checkpoint files currently exist. Lock JSON schema:
`{"revision":"PUBLIC_REVISION", "sha256":{"config.json":"64 lowercase hex",
"model.safetensors":"64 lowercase hex"}}`. Paths must be relative without `..`.
HF snapshot symlinks are allowed because their resolved bytes are authenticated.
Select explicit public tensors already known to be output-by-input; no inferred
orientation, model execution, network downloads, fallback quantizer or builds.

Runnable serial command (parent sets these three absolute paths/key):

```sh
PYTHONPATH= target/frontier-sdk-env/bin/python scripts/probe_certified_digit_blocks.py \
  --scan-checkpoint "$PUBLIC_CHECKPOINT" --source-lock "$PUBLIC_SOURCE_LOCK" \
  --tensor "$PUBLIC_OUTPUT_BY_INPUT_TENSOR" --batch 150 --bits 32 \
  --admit-public-source-scan \
  --output docs/evidence/certified-digit-blocks-public-scan.json
```

`fresh_workers` owns `~/.cache/pllm/frontier-probe.lock`, demands **3 GiB**
post-reserve host headroom, bounds runtime at 240 seconds, caps thread pools at
four, enables offline execution and guards host swap pressure/build identity.
Worker separately rejects locked source file bytes + **256 MiB** runtime/scratch
above **2 GiB**, before large-file hashing/tensor-value reads. Source dimensions
and native batch bounds checked from headers before values; 1–8 keys, at most
16 Mi weights per tensor, 32 Mi total. Quantize 32 complete rows at a time with
installed native W8 quantizer; retain one tensor's i8 weights at a time. Installed
SDK/extension paths must resolve under the environment prefix, rejecting checkout
shadowing. Source files, independently supplied lock and installed quantizer/
extension/probe sources rehashed after scan. Scanner records scales, i8 source
commitment, block certificates, counts and conservative admission estimate.

If source admission fails, narrow the independently locked shard selection or
defer; never weaken gates. Scanner yields **counts only**, not source equivalence
to a compiled stage or a performance result. Parent then needs:

* Match quantized/scales commitments, stage orientation, page boundaries and
  compiled role placement; price certificate creation/retention and invalidation.
* Standalone source-locked native candidate, exact Accelerate comparisons on
  extreme and real stage values; no shared build concurrently with workers.
* Measure Preparation CPU/RSS and cold/base certificate setup separately from
  output-dependent issuance; measure setup amortization, call/packing overhead,
  BLAS-private RSS and workspace/cache tradeoffs. Keep native 100/40 controls.
* Matched fresh-process complete responses: client CPU/RSS, Preparation CPU/RSS,
  TTFT, N−1 decode throughput, all-link traffic, largest whole-client peak. Low
  ceiling 268,435,456 B; Medium 2,147,483,648 B; conservative admission and host
  headroom remain independent. Exact identity and ≥90% remote body MACs checked
   independently for prefill/decode. **No tier, traffic, serving-speed or SOTA claim yet.**

## Initial parent source scan: superseded by frozen rescan

Parent independently passed original 20 tests/replayed synthetic evidence, then
ran admitted scan `target/tier-objectives-ten/certified-down-scan.json` (target
artifact). The worker inspected this **JSON result only**, not
checkpoint values. Revision `7ae557604adf67be50417f59c2c2f167def9a775`, lock
`docs/evidence/orthogonal-frontier-qwen25-source-lock.json`, lock SHA256
`be4a2b32dd596cc5a5ad096a356e69bb13f2324e049fa9c06609e5e6eb4b2d32`.

Qwen2.5-0.5B-Instruct layers 0/8/16/23 `mlp.down_proj.weight`, each [896,4864],
batch150/ring32, certify **20→8** SGEMM calls. First block widths respectively
**4255/4375/4419/4277**, remaining columns form a second block. Explicit candidate
workspace about **19.0–19.7 MB**, fixed kernel **5,355,432 B**; raw ring32 correction
payload remains **537,600 B**, dense products **2,614,886,400**. Observed scan swap
growth **0 B**, conservative source-scan admission estimate **1,256,533,939 B**.
These are source-weight counts/admission, **not runtime speed, compiled stage
equivalence or tier qualification**. Prior scan probe/test hashes differ from
the extended native-runner hashes intentionally; preserve its original identity.

## Standalone native fixture (parent compilation/execution only)

Added independently written
`scripts/research_native/certified_digit_blocks/fixture.c`; inspected the sibling
limb fixture only for ABI/Accelerate style, never edited it. No SDK/Rust
registration, shared builds or source model loading by this worker.

* `cdb_create` validates bounded source geometry/ring/workspace **before reads or
  large allocations**, copies i8 weights into owned immutable snapshot, derives
  fixed1023 or greedy certified records. Independent second traversal checks
  contiguous full coverage, dimensions, width and every output row's absolute
  sums against owned snapshot; stored maxima never trusted. No supplied unchecked
  record API. Stats/records interfaces return copies, not mutable owner pointers.
* `cdb_execute` validates canonical inputs and output masks; allocates float
  slabs per issuance; baseline visits block-major/output-tile-major exactly like
  native control, certified uses its tile-specific blocks. Same balanced carry,
  SGEMM alpha1/beta0, finite/integer/strict-2^24 checks before safe i32 conversion,
  wrapping u32 shifts/adds, integer output-mask subtraction then narrowing.
  Failure marks output invalid. Wrapper creates separate input/mask/output buffers
  and must not race destroy against execution; C caller must provide valid
  non-overlapping output buffers of declared size.
* C workspace cap 64 MiB includes output and conservative record/scratch
  reservation. C fixed fixture also allocates/reserves record metadata absent
  from production fixed kernel: **C control is not byte-identical Rust baseline**.
  Both C arms use same source transport, snapshot ownership, metadata reservation,
  per-issuance allocations and ABI transfer path. C fixed has no source-L1 scan;
  certified cold setup adds derivation plus independent validation.
* `NativeFixture` ctypes bridge snapshots source across ABI, serializes/copies
  input and output masks each invocation, allocates output and copies it back into
  NumPy. Certified records/workspace independently match Python source planner.
  Crosscheck is setup work, not timed issuance. C source-copy/create timing also
  reported separately from whole wrapper setup, so Python certificate checking
  is not mislabeled as native planner CPU.

### Parent commands, in order

Unique output below must not already exist; build refuses to overwrite an
existing dylib. Change directory suffix for a new build, retaining old manifest.
These commands were **not executed by this worker**.

```sh
PYTHONPATH= target/frontier-sdk-env/bin/python scripts/probe_certified_digit_blocks.py \
  --compile-native --admit-native-run \
  --native-library target/tier-objectives-ten/certified-digit-blocks-v1/fixture.dylib \
  --output target/tier-objectives-ten/certified-digit-blocks-v1/compile.json
```

Runner executes `/usr/bin/clang -O2 -std=c11 -Wall -Wextra -Werror -Wno-deprecated-declarations -dynamiclib
fixture.c -framework Accelerate -o fixture.dylib`; **no fast-math**. Records exact
command/compiler version, source/dylib SHA256, installed BuildRef, swap growth.
The deprecation exception retains the deliberately selected LP64 CBLAS ABI.
Compile subprocess timeout 60 s; fresh-worker timeout 240 s; common frontier lock
and 3 GiB post-reserve admission/watchdog. No implicit compilation on later runs.

```sh
PYTHONPATH= target/frontier-sdk-env/bin/python scripts/probe_certified_digit_blocks.py \
  --native-check --admit-native-run \
  --native-library target/tier-objectives-ten/certified-digit-blocks-v1/fixture.dylib \
  --native-build-record target/tier-objectives-ten/certified-digit-blocks-v1/compile.json \
  --output target/tier-objectives-ten/certified-digit-blocks-v1/adversarial.json
```

Parent check executes all outputs against Python unbounded-integer oracle for
all-minus128, constant32/4097-column tail, cancellation, full signed-i8 random,
1025-output-row tail tile; rings16/24/32; both C arms. Includes carry boundaries,
source mutation after handle construction (snapshot identity), malformed raw C
geometry and raw C input/output-mask rejection. Checks 12,444 outputs in 30
small cases. This check performs **no checkpoint loading or timing**.

Then one public tensor per serialized cohort (repeat separately for other layers):

```sh
PYTHONPATH= target/frontier-sdk-env/bin/python scripts/probe_certified_digit_blocks.py \
  --native-timing --admit-native-run --admit-public-source-scan \
  --native-library target/tier-objectives-ten/certified-digit-blocks-v1/fixture.dylib \
  --native-build-record target/tier-objectives-ten/certified-digit-blocks-v1/compile.json \
  --scan-checkpoint /Users/blairhudson/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 \
  --source-lock docs/evidence/orthogonal-frontier-qwen25-source-lock.json \
  --tensor model.layers.0.mlp.down_proj.weight --bits 32 \
  --repeats 2 --replicates 1 \
  --output target/tier-objectives-ten/certified-digit-blocks-v1/down0-cohort.json
```

Fresh processes: **one- and 150-row** masks, each with C fixed, C certified,
installed exact integer, installed native digit8 controls: 8 serial workers at
default 1 replicate. One untimed warm issuance, then 2 measured fresh-mask
issuances per process; caps 4 repeats/3 replicates. Deterministic public probe
masks reproduce identical input/output-mask sequences across arms, never an SDK
freshness change. Rotate arm ordering between replicates; small cohorts remain
exploratory, not sustained benchmarks or complete responses. Pooling fails on
different source/scales/lock/probe/build, mask commitments, output commitments,
missing/duplicate arms or divergent full-output counts.

**Every representative output**, including warm and measured outputs, compares
against installed exact integer `wrap32` plus uint32 mask subtraction, outside
issuance timing. Installed integer/digit8 unavailable or checkout-shadowed means
fail closed, not fallback or skipped oracle. Integer-control self-check is only
reproducibility; C/adversarial Python oracle is the independent algorithm check.
Extension, fixture, parent compiler manifest, probe/tests and source lock hashes
remain explicit and revalidated. Existing source-lock changes are rejected.

Report source setup (source hashing/imports/header checks/quantization/Python
plan), installed oracle matrix setup, whole arm snapshot/certificate setup,
C source-copy/create phase, then **transfer-inclusive issuance** wall/CPU.
Timed C issuance includes Python domain checks, serialization, ctypes copies,
C allocation/packing/SGEMM/reconstruction/subtraction and output-copy return.
Installed controls include their public binding copies/kernel/subtraction.
No model source load, certificate setup, random-mask generation, oracle or
network traffic belongs to timed issuance. It is not full PRG mask expansion,
protocol upload or Preparation server timing. C comparisons cannot establish
Rust integration speedups; installed arms are contextual controls only.

Native cohort additionally reserves locked source file bytes + **512 MiB**
runtime/scratch/oracle/fixture headroom, rejects estimates above **2 GiB** before
hashing/value reads, and still demands **3 GiB** post-reserve host headroom.
For this lock estimate is **1,524,969,395 B**. Common 4-thread pools and installed
integer executor=4; this bounded fixture does **not** replace maintained 100/40
complete-response controls. Record process high-water RSS before and through
post-source validation; includes source pages, both candidate/oracle weight
snapshots, ABI overlap and validation, not just kernel workspace. BLAS-private
RSS remains measured rather than proved by explicit workspace formula. No
Low/Medium, ≥90% remote MAC or traffic qualification from fixture timings.

### Worker validation of extension

Only small Python tests executed; **32 passed** (observed replay 1.37 s). Added
admission/CLI rejection before compilation/loading; mock-ABI ctypes transfer,
snapshot/copy/close lifecycle checks (not compiled C); matched cohort identity
and missing/duplicate control gates; synthetic replay retaining all previous
90 oracle outputs/counts with native fixture marked **uncompiled/unexecuted**.
These were worker-only results. Parent subsequently passed the compiled checks,
replayed the four-tensor scan under the final identity, and completed the native
cohort described in [the consolidation](tier-objectives-reference.md). SDK and
complete-response gates remain outstanding.
