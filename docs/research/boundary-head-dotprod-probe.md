# Exact signed-byte-limb SDOT head probe

Status: bounded, synthetic research fixture only. No production kernel, compiler,
runtime, serving, security, or architecture changes. Full-head synthetic v3
measurement retained below; no model/SDK timing or production integration.

Allocation-attribution follow-up (v3): separate full-head cohort, not pooled with v2. Prior
frontier's **37.22% lower head API CPU** compares preallocated candidate output
against allocating `Matrix::wrap32`; it is historical API-level evidence, not an
allocation-matched result or isolated SDOT instruction speedup. Archived v1/v2
records and summary/replay tests remain unchanged. New v3 records add a separate
allocating-output comparison; never relabel old records as v3.

## Retained v3 full-head result

Parent completed six serial same-build Apple M5 workers, each four balanced
repetitions at `151936 x 896`; all full-vector oracle comparisons passed.
For allocating-output APIs, including candidate input packing:

| Aggregation | Candidate CPU | Native API CPU | Lower CPU |
| --- | ---: | ---: | ---: |
| Median of six worker medians of four samples (prior frontier convention) | 4.228 ms/call | 6.54875 ms/call | 35.4380607% |
| Median of six worker four-call totals | 17.989 ms/four calls | 26.1945 ms/four calls | 31.3252782% |

These are distinct aggregation methods on the same cohort, not interchangeable
estimates or separate wins. Both are reported to expose sensitivity to slower
samples. Both APIs allocate output; native internal allocation/transposition still
differs. Synthetic weights, not isolated SDOT throughput or whole-client/SDK gain.
Maximum fresh-process RSS **281,673,728 bytes**; guard observed zero swap growth.
This process RSS is **not Low-client evidence** or a whole-client memory tier.

Exact raw JSONL, guard and build log were frozen with SHA256 and all eight
measurement-time source hashes **before this doc edit** in
[`../evidence/boundary-head-dotprod-v3-2026-10-06.json`](../evidence/boundary-head-dotprod-v3-2026-10-06.json).
Deterministic summary lives alongside as `.summary.json`. Filename date is the
requested archive label; actual embedded build timestamp is
`2026-10-15T01:30:15.853172+00:00`. Replay uses frozen source locks, not today's doc
hash; it rejects incomplete workers, different builds, unbalanced orders,
non-exact results, invalid/sum-inconsistent timings and failed resource guards.
Historical v1/v2 archives and frontier summary remain unchanged and uncombined.

Exact replay (no build, model or timing job; stdout matches retained summary):

```sh
python3 scripts/summarize_boundary_head_dotprod.py docs/evidence/boundary-head-dotprod-v3-2026-10-06.json
target/frontier-sdk-env/bin/python -m pytest -q tests/test_boundary_head_dotprod_summary.py tests/test_boundary_decode_frontier_summary.py
```

## Question and exactness

For a single ring32 input vector, can four balanced byte limbs amortize input
preparation across many signed-i8 output rows and beat the current native integer
head? The control is `Matrix::wrap32(Executor::new(1, true), x, 1)`, borrowing the
**same immutable Matrix weight allocation** as the candidate. No persistent
transposed, widened, digitized, or duplicate weight allocation is introduced.

For each unsigned input `x`, start `c = i64(x)` and repeat four times:

```
d[k] = ((c + 128) & 255) - 128
c = (c - d[k]) >> 8
```

Thus `x = sum(d[k] * 256**k) + c * 2**32`. Discarding final carry is exact
modulo `2**32`; reinterpreting each original byte as i8 without the carry is not.
The limb slab is four contiguous length-H i8 slices, built once per invocation.
Each weight vector load feeds four SDOT accumulators. Their signed horizontal
sums plus scalar tails reconstruct via wrapping u32 shifts/adds.

All dimensions are checked before payload allocation. `H <= 131071` ensures
`H * 128 * 128 <= i32::MAX`, including horizontal reduction and tail additions;
larger H is rejected, not silently overflowed or chunked. Other limits: nonzero
dimensions, at most 256 MiB weights, conservative live allocation bound 512 MiB,
1–100 repeats, at most 2 GiB weight-bytes times repeats, nonzero PRNG seed.
The work guard retains its historical definition; v3 measures two comparison
pairs per repeat, so it performs twice the timed head calls of v2.
Emitted payloads are capped at 65536 weights and 1024 rows. The target
`V=151936, H=896` fits these guards; each limb bound is 14,680,064.

The native scalar oracle independently multiplies signed weights and original
u32 inputs in i64, accumulates wrapping u64, then narrows. Python tests use
arbitrary-precision integer sums reduced modulo `2**32`, comparing complete
candidate and native-control output arrays, not result hashes.

## Stable target support and fallback

Observed host: **Apple M5**, aarch64, macOS 26.5.2. Rust **1.85.1**
(`4eb161250`, 2025-03-15). Runtime detection reports `dotprod=true`, native control
SIMD enabled, candidate selects `arm-sdot-inline-asm`.

On this compiler `vdotq_s32` fails with E0658, unstable library feature
`stdarch_neon_dotprod` (Rust issue 117224). The separately scoped example instead
uses stable `std::arch::asm!` for SDOT inside an aarch64-only
`#[target_feature(enable = "dotprod")]` function. No nightly features, C helper,
manifest edits, `RUSTFLAGS`, or globally forced target features. The function is
called only after runtime feature detection. Sixteen-byte loads occur only when
16 valid elements remain; scalar tails handle all other widths.

`scalar` mode forces the independent balanced-limb scalar candidate. Other
architectures and ARM machines without dotprod automatically select that path;
the unchanged native Matrix control retains its own normal SIMD detection.
Forced scalar dispatch was tested on this M5; an actual no-dotprod CPU and
non-aarch64 compilation were **not** tested here.

## Reproduction

Build and small correctness suite (no full-head workload):

```sh
target/frontier-sdk-env/bin/python scripts/probe_boundary_head_dotprod.py --build --rows 3 --cols 17 --repeats 2 --processes 2
target/frontier-sdk-env/bin/python -m unittest discover -s tests -p test_boundary_head_dotprod_probe.py -v
target/frontier-sdk-env/bin/python scripts/probe_boundary_head_dotprod.py --rows 97 --cols 896 --repeats 2 --processes 2
```

Tests cover widths 1, 3, 4, 15, 16, 17, 31, 32, 33, 127, 896; signed weights
including -128; u32 extrema; balanced carry edges; wrapping outputs; automatic and
forced scalar paths; largest accepted H; rejected geometries; fresh-worker RSS;
both starting orders, alternating repeat schedules and paired fresh-process balance.

v3 follow-up validation: six probe unittest cases passed; `97 x 896`, two repeats
and two fresh workers passed with `arm-sdot-inline-asm`, all historical/matched
pairs exact and native build/source locks accepted. Forced scalar and limb tails
passed within the suite. Existing archived frontier summary/replay tests remained
unchanged: `target/frontier-sdk-env/bin/python -m pytest -q
tests/test_boundary_decode_frontier_summary.py` passed all 11 cases. System
`python3` lacked pytest; the repository benchmark environment supplied it.
This is correctness/scheduling smoke, not an allocation-matched speedup cohort.

Parent-controlled serial measurement command (already completed; replay above
does not rerun it; future timing only after other benchmarking stops):

```sh
target/frontier-sdk-env/bin/python scripts/probe_boundary_head_dotprod.py --build --rows 151936 --cols 896 --repeats 4 --processes 6 --order-offset 0 > boundary-head-dotprod-qwen-v3.jsonl
```

Use a new output filename; do not overwrite archived v2/v3 JSONL.
Six workers run serially, never concurrently.
Each native process is launched by
a separate fresh Python parent. That parent reports `RUSAGE_CHILDREN.ru_maxrss`
normalized to bytes (macOS bytes; Linux KiB), plus native-process user/system CPU.
Build happens before these parents and does not contaminate child high-water
marks. JSON also records platform, dimensions, feature selection, validation,
setup, packing, candidate kernel, and baseline elapsed/process-CPU seconds.
Top-level phase timings are **totals across repeats**, excluding initial warmup
and oracle validation. `samples` contains each repeat's phase timings, their
packing-plus-kernel sum, actual native-call order and exactness result.
POSIX `clock()` CPU counters have microsecond-scale granularity;
zero packing CPU on tiny smoke cases is expected. Non-Unix phase CPU placeholders
are zero; the Python RSS driver targets Unix/macOS/Linux.

Both candidate forms and native API receive warmup and full-vector validation. `--order-offset 0`
starts candidate-first; `1` starts baseline-first. Native positional argument
`ORDER_OFFSET` exposes the same choice (omitting it defaults to 0). Subsequent
repeats alternate order; each fresh process alternates its starting offset.
Packing stays immediately before its candidate call in either order. Each
pair is validated after both calls, outside timing. v3 runs both historical and
allocation-matched pairs per repeat, alternating which pair runs first using the
same offset. `samples[].cohort_order` records this schedule. Even repeat counts balance
order within each process; even process counts balance starting order across
processes, including odd repeat counts. Defaults are four repeats and two
processes. JSON records effective offset, process index/count and actual sample
orders, so explicitly requested odd counts cannot silently masquerade as paired
starting balance. Warmup order is preallocated candidate, native baseline, then
allocating candidate; alternating
timed order mitigates rather than eliminates cache, thermal and order effects.
Historical candidate reuses its preallocated output while native Matrix
control retains its public API's allocations/transposition. New allocating
candidate calls `vec![0u32; rows]` inside the timed call, then invokes the exact
same research candidate function. Native comparator remains unchanged real
`Matrix::wrap32`, borrowing the same weights. Both APIs now allocate returned
output, but allocation **counts and internal work are not equal**: native API
also allocates an accumulator and transposes into its returned output. No public
preallocated-output equivalent exists here; no copied production SIMD comparator
was introduced. Output validation and deallocation remain outside timing for
both calls. This narrows output-ownership attribution only; neither pair isolates
SDOT instructions from loop structure, reduction, allocation, or transposition. Setup
contains synthetic weight generation, Matrix construction, and buffer allocation;
fresh-process CPU/RSS additionally includes warmup and scalar oracle work.

### Exact v3 timing keys

Historical top-level and `samples[]` keys keep their v2 meaning: `packing`,
`kernel`, `candidate_including_packing`, `baseline`. Existing `comparison` remains
the historical description. New `allocation_matched_comparison` states limits.

New keys, both top-level and under every `samples[].allocation_matched`:

- `allocation_matched.packing`: balanced-limb input preparation, separate from call.
- `allocation_matched.candidate_allocating`: output allocation/zeroing plus candidate.
- `allocation_matched.candidate_allocating_including_packing`: sum of preceding
  two phases; end-to-end candidate head API cost with reused limb scratch.
- `allocation_matched.baseline`: separately measured real allocating native API;
  never borrow historical baseline timing to manufacture a matched pair.

Each timing contains `elapsed_seconds` and `cpu_seconds`; top-level values sum
repeats. Including-packing totals are phase sums, not independently stopwatched
wall intervals; they exclude instrumentation gaps, setup, validation and freeing
returned outputs. Native API needs no limb packing; its baseline already includes
its internal input handling. `exact` accompanies each matched pair and total.
`samples[].allocation_matched.order` records candidate/native order.
`process_schedule.paired_cohort_start_balance` reports even-worker start balance.
Limb scratch stays preallocated in both candidate forms; this does not measure a
whole process, SDK request, or production integration.

The Python driver records SHA256 of the executed binary, all four owned source /
research files, native control source, Cargo manifests and lockfile. The binary
embeds SHA256 of the example and `kernels.rs` as compiled; mismatches with current
source are rejected. It also embeds build provenance supplied by `--build`:
actual `rustc -vV`, `cargo -vV`, compiler path, release build command, build UTC,
relevant build/target flag environment (not registry credentials), and discovered
repository/home Cargo config hashes. Reports retain that build-time provenance
rather than substituting whichever toolchain happens to be installed at timing
time. Direct Cargo builds without embedded provenance are rejected by the driver;
rebuild through `--build`. Runtime host/platform/CPU brand and native feature
detection accompany every report. Binary and source hashes are checked before
and after execution; concurrent changes invalidate the record. Keep JSONL with
its embedded binary hash even after rebuilding: measurements belong to that
artifact, never to a later binary at the same path. The build step is outside
fresh measurement parents. Comparison metadata explicitly labels the native
API allocation difference. These are reproducibility records, not a hermetic
build or artifact attestation.

Original v1 fixed-order smoke, `97 x 896`, one repeat, under concurrent development
(historical only, superseded by v2 balanced scheduling; no new timing cohort):

| Phase | Elapsed | Process CPU |
| --- | ---: | ---: |
| Setup | 154.334 us | 155 us |
| Input packing | 0.792 us | 0 us |
| Candidate kernel | 3.375 us | 4 us |
| Native Matrix control | 5.000 us | 5 us |

Fresh-process peak RSS: 6,291,456 bytes. Complete outputs matched. These
microsecond values are smoke evidence only, **not** a speedup estimate or timing
cohort. During that original v1 smoke, no Qwen-sized, full SDK, or end-to-end
measurements were launched; the separate synthetic full-head v3 cohort is above.

## Owned memory and interpretation

JSON explicitly lists each logical buffer. Steady weights are `V*H` bytes, input
`4H`, input limbs `4H`, candidate output `4V`, independent oracle output `4V`,
retained native-control output `4V`. Native control calls may temporarily own
another `8V` bytes (accumulator/output transpose); setup briefly holds raw weights
alongside Matrix's adopted-by-copy allocation, an additional `V*H` bytes. That
raw buffer is dropped before warmup/timing. No extra persistent full weight copy.
v3 additionally owns allocating candidate output `4V` during its matched pair
(`owned_buffers.matched_candidate_call_output_bytes`), dropped outside timing;
the conservative preflight bound is now `2*V*H + 24*V + 8*H` bytes. Retained
baseline, historical candidate, and oracle outputs remain available for validation.
At Qwen geometry: weights 136,134,656 bytes; limb slab 3,584 bytes; each output
607,744 bytes. Figures exclude Vec metadata, allocator overhead, stack and code;
fresh-process peak RSS is the separate whole-process measure, not their sum.

Recommendation: **keep research-only despite positive synthetic full-head evidence**. Require
`allocation_matched.candidate_allocating_including_packing` to beat its paired
`allocation_matched.baseline` at target geometry with repeatable results and
acceptable measured RSS. Retain historical arms as secondary diagnostics only.
A tiny-cache smoke
cannot establish a memory-streaming one-row head win; four limb streams and
horizontal reductions can lose. No float, fastmath, SGEMM, digit8/FP64 preparation
proxy, predicted end-to-end gain, or reduced-output claim is involved.

The coupled private boundary retains original prepared body, full logits, KV and
sampler; see `private-boundary-tiers-reference.md`. Prior Qwen150+8 confirmation
reported whole-client max 210.22 MiB, traffic 19.93% lower/request 20.26% faster,
but decode 1.9235 TPS versus maintained 3.794. Serial full-head replies remain a
separate network cost. This probe targets residual worker CPU only; improving
this kernel does not remove those replies or establish an overall speedup.
