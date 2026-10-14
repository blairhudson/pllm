# Ten optimization screens inside fixed client-memory tiers

These investigations target Preparation CPU, client CPU, TTFT, decode throughput
and all-link traffic while treating whole-client RSS as a constraint. They follow
the small, recipe-dependent [attention-tiling results](attention-tiling-reference.md),
rather than assuming another isolated memory reduction will improve serving.
The [frozen investigation contract](../plans/tier-objectives-ten.md) retains exact
numerics, current privacy boundaries and at least 90% remote body MACs in each
phase. Low and Medium remain 256 MiB and 2 GiB whole-client maxima.

## Decisions

Ten Sol 6.1 investigations produced bounded references and independent correctness
controls. Parent validation adds a pinned public-weight scan and serial operator
measurements. **No candidate establishes a new Low/Medium serving SOTA or fresh
all-link traffic reduction.** Operator-process peaks do not qualify a complete
client. The strongest remaining kernel lead is source-certified wider digit
blocks; its measured native gate is recorded separately below.

Source-hashed mechanism notes retain their initial worker-phase findings,
including statements that the worker did not run timing. The subsequent parent
measurements and current decisions are recorded here under their own artifacts.

| ID | Mechanism | Evidence and decision |
| --- | --- | --- |
| 1 | [Weight-certified digit blocks](certified-digit-blocks-reference.md) | Every output/block must satisfy `sum(abs(w)) <= 131071`. Four pinned Qwen down tensors permit 20→8 SGEMM calls. A 24-process native stage screen lowers 150-row C issuance CPU 9.99%, while setup and RSS rise; one-row execution regresses. Strongest follow-up is prefill-only native integration with cheaper certificate setup. |
| 2 | [Phase-split placement](phase-split-placement-reference.md) | Independent exhaustive control agrees with the constrained planner. The seven-group public replay finds no gain over fixed placement; a separate toy establishes that phase differences can matter. No compiled split executor or client cost result. |
| 3 | [One-slot mask runahead](mask-runahead-reference.md) | Native mask/ticket bytes and cancellation/burn controls agree. Immediate-wait executor calls cost more than synchronous expansion. Only actual overlap with an independent transport operation can repay this overhead. |
| 4 | [Preparation flowshop ordering](preparation-flowshop-reference.md) | A tick oracle checks 1,152 schedules. Simulated exhaustive optimum improves 150-row completion 1.84–2.72%, but the proposed heuristic regresses; 16-row jobs show no gain. No real transport improvement. |
| 5 | [Exact FP64 corrections](exact-f64-corrections-reference.md) | Bounded integer outputs agree, including separate installed controls. Transfer-inclusive DGEMM fixtures use 2.36×/7.65× the native integer CPU; defer this layout. Vendor reduction certification remains separate from observed parity. |
| 6 | [Stationary packed mask limbs](limb-stationary-corrections-reference.md) | Reuses a packed slab across resident output tiles. Complete native integer audits pass, but isolated C CPU falls only 2.59%/1.72%. Current paged slabs already pack once; no paged benefit follows. |
| 7 | [Resident head shards](resident-head-shards-reference.md) | Full logits, local lookup, eviction and authentication controls pass. The prototype duplicates derived snapshots because no native row-range interface exists. Defer until shared authenticated ownership can be priced and timed. |
| 8 | [Decode-attention lifetime fusion](fused-decode-attention-reference.md) | 120 small fixtures match the original reductions bitwise. Six fresh timing workers retain identical hashes; repeated same-fixture CPU falls 3.35%, with effectively unchanged process RSS. Small secondary lead, pending compiled complete-state execution and response evidence. |
| 9 | [Column-streamed projection](column-streaming-linear-reference.md) | Complete integer outputs and one-use ordered accumulation pass. The tiny fixture starts partial work before upload completes, but upload grows 425→789 bytes and kernel launches grow to five. Larger real-link latency and complete correction ownership remain gates. |
| 10 | [Cross-session masked microbatches](masked-decode-microbatch-reference.md) | Native matrix batching already exists. New bounded queue ownership passes policy/preflight/cancellation controls; fixed waits hurt low load. This is a possible scheduling policy, not a newly discovered matrix kernel or measured serving gain. |

These are project hypotheses, not literature-novelty claims. Negative and small
results remain useful: they prevent promoting a plausible operation count into a
whole-response claim.

## Source-certified native digit blocks

The independently retained public source lock pins Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`. A fresh four-tensor scan under the final
probe identity checks layers 0/8/16/23 down projections, each 896×4,864. Installed
W8 row quantization and immutable file hashes precede planning. At 150 rows,
every tensor admits two reduction blocks instead of five: eight rather than
twenty SGEMMs. Workspace rises from 5.36 MB to 19.02–19.71 MB. Raw ring32 correction
payload remains 537,600 bytes; dense products, masks and protocol work do not decrease.
The earlier preliminary scan remains a target artifact with its original hashes;
the published rescan uses the final source.

The standalone native fixture owns an immutable i8 snapshot, derives its records,
then independently verifies every row/block sum and complete coverage. Thirty
adversarial cases check 12,444 outputs against Python unbounded integers, including
extreme weights, carry/wrap boundaries, source mutation, tail tiles and raw-C
domain refusal. The compiled fixture and installed SDK are frozen separately.

The subsequent layer-zero cohort has three replicates, two batch sizes and four
arms: **24 fresh processes**. Each performs one charged warm issuance and three
timed issuances, checking **all 6,494,208 output positions** against the installed
exact integer kernel. Source, scale, input-mask, output-mask and result commitments
match across arms. Arm ordering rotates across replicates. Source loading/hashing,
quantization, Python planning, oracle setup and complete arm setup remain separate
from transfer-inclusive issuance. Native/BLAS thread requests are four; this is a
different cohort from the one-thread small-operator screen below.

| Batch | C fixed CPU | C certified CPU | Installed integer CPU | Installed digit8 CPU |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.574 ms | 2.001 ms | 0.571 ms | 1.288 ms |
| 150 | 14.732 ms | 13.260 ms | 102.422 ms | 13.821 ms |

The 150-row C pair reduces median issuance CPU **9.99%**, but median arm setup
increases **3.308→14.493 ms**, and median process high-water RSS through final
source validation increases **95.09→106.71 MB**. The explicit workspace increase
and complete oracle weights remain charged. The one-row pair increases issuance
CPU **27.13%**. Installed controls provide context; different bindings and owner
layouts prevent treating a C-versus-Rust difference as a Rust optimization result.
All 24 workers observe zero swap growth.

At steady-state medians, the extra 11.185 ms setup needs roughly eight 150-row
issuances to repay 1.472 ms per call, before any other costs. This arithmetic is
not measured cold amortization. Each of the four scanned tensors also admits
fixed public 4,096/768 cuts: first-block prefixes inherit the checked bound, and
the remaining 768 columns satisfy the worst-case bound. A contiguous row-major
certificate check for those two cuts may avoid the current greedy column scan;
its cost reduction is unmeasured. **Prefill-only integration and cheaper cold
setup are the next gates**, followed by compiled role/admission and matched
complete-response evidence. Current paged execution needs page-local certificates.

## Serial parent operator screen

`scripts/benchmark_tier_objectives.py` admits fixed cases under 384 MiB, holds the
shared frontier measurement lock, observes host reserve and swap growth, and
checks installed SDK and probe identities before and after each child. Three
replicates use alternating case order. Each operator has three in-process samples;
the table takes each process's median first, then the median across processes.
Requested BLAS/native thread count is one; vendor thread execution is unverified.
The standalone C fixture is compiled separately against Accelerate without
fast-math. The host is not an exclusive-host latency experiment.

### FP64 product and integer output-mask subtraction

Eighteen fresh workers compare the installed Rust wrap32 and digit8 kernels with
the standalone NumPy/Accelerate FP64 layout. All outputs pass an independent
Python integer oracle. Immutable-weight setup is separately recorded; timed
operations include the declared input/output boundaries and integer subtraction.
They exclude source loading, mask issuance, transport and decoder work.

| Batch × outputs × inputs | Native integer CPU | Installed digit8 CPU | FP64 CPU |
| --- | ---: | ---: | ---: |
| 32 × 16 × 2,048 | 0.180 ms | 0.293 ms | 0.425 ms |
| 1 × 256 × 4,864 | 0.155 ms | 0.475 ms | 1.186 ms |

The first shape is an oracle-bounded bulk fixture, not a complete checkpoint
stage. The second retains the wide reduction but only 256 output rows. None
supports a claim about full-width DGEMM throughput. The examined implementation
fails this small-stage cost gate.

### Packed-limb reuse

Six fresh workers compare two standalone C schedules with identical arithmetic,
input geometry and output bodies. Both retain all weight conversions and dense
products. Every timed result also matches the installed exact integer kernel:
2,433,024 checked output positions, with the extra audit charged outside operation
timing. Separate tail/ring workers cover output/input block boundaries.

| Batch × outputs × inputs | Original C schedule CPU | Stationary C schedule CPU | Change |
| --- | ---: | ---: | ---: |
| 32 × 4,096 × 1,024 | 4.556 ms | 4.438 ms | −2.59% |
| 1 × 4,096 × 1,024 | 1.165 ms | 1.145 ms | −1.72% |

These are projection timings, not full correction issuance or a comparison
against the installed Rust implementation. The combined-process peak includes
both C arms and the independent integer audit; it cannot supply per-arm RSS.
Mask packing falls fourfold in this geometry, yet total CPU changes little.

### Mask expansion and decode attention

Six fresh mask workers compare synchronous expansion with submitting and waiting
on a persistent one-slot executor. At four rows of 2,048→4,096 geometry, median
wall time rises 0.242→0.332 ms and CPU 0.241→0.356 ms. At 32 rows of 896→9,728,
wall time rises 3.082→3.141 ms and CPU 3.083→3.164 ms. This measures the local cost
to hide, **not achieved network overlap**. Worker/thread shutdown remains charged
to complete-worker CPU, outside the per-operation window.

The separate decode-attention driver uses three alternating baseline/candidate
pairs with key capacity 1,024, query position 287, two KV heads, seven query heads
per KV head and width 64 under `prefix_f32`. Each child charges one cold operation
and 256 repeated operations: 1,542 total operations and 795,820,032 products.
Every paired hash matches. Median cold CPU is 1.277→1.157 ms; repeated same-fixture
CPU is 0.783→0.756 ms per operation. Largest process peaks are 49.48/49.66 MiB.
This excludes complete decoder scheduling and is not autoregressive throughput.

All four parent timing cohorts observe zero swap growth. Their process peaks
include imports/setup and their own audits; no complete-client RAM tier follows.

## Evidence and replay

Each mechanism's linked note retains its synthetic fixture, source contracts,
negative controls and exact replay. The parent operator evidence is:

- [`tier-objectives-mask-runahead.json`](../evidence/tier-objectives-mask-runahead.json)
- [`tier-objectives-f64-corrections.json`](../evidence/tier-objectives-f64-corrections.json)
- [`tier-objectives-limb-stationary.json`](../evidence/tier-objectives-limb-stationary.json)
- [`tier-objectives-limb-tails.json`](../evidence/tier-objectives-limb-tails.json)
- [`tier-objectives-decode-attention.json`](../evidence/tier-objectives-decode-attention.json)
- [`tier-objectives-limb-build.json`](../evidence/tier-objectives-limb-build.json)
- [`certified-digit-blocks-public-scan.json`](../evidence/certified-digit-blocks-public-scan.json)
- [`certified-digit-blocks-native-build.json`](../evidence/certified-digit-blocks-native-build.json)
- [`certified-digit-blocks-native-adversarial.json`](../evidence/certified-digit-blocks-native-adversarial.json)
- [`certified-digit-blocks-native-down0.json`](../evidence/certified-digit-blocks-native-down0.json)

The [evidence manifest](../evidence/tier-objectives-manifest.json) binds all twenty
synthetic and parent artifacts. Parent regression validation passes **528 tests
and 33 subtests**; 680 retained source/build digest checks have zero mismatches.
Ruff, whitespace checks and the 435-page docs-content check pass. Native fixture
compilation and adversarial execution are separate from mock-ABI Python tests.

The certified-block note supplies the executed compile/check/source-timing
commands. The published source cohort uses `--repeats 3 --replicates 3`; the
four-tensor source scan selects layers 0, 8, 16 and 23 in one admitted invocation.

Use the installed validation environment, without importing `python/pllm` from
the checkout. Build the small limb fixture using the admitted parent compile
procedure in its [reference](limb-stationary-corrections-reference.md), then run
these commands **serially**:

```sh
PYTHONPATH='' target/frontier-sdk-env/bin/python scripts/benchmark_tier_objectives.py --case limb-tail --replicates 1 --output target/limb-tails.json
PYTHONPATH='' target/frontier-sdk-env/bin/python scripts/benchmark_tier_objectives.py --case f64 --replicates 3 --output target/f64-screen.json
PYTHONPATH='' target/frontier-sdk-env/bin/python scripts/benchmark_tier_objectives.py --case limb --replicates 3 --output target/limb-screen.json
PYTHONPATH='' target/frontier-sdk-env/bin/python scripts/benchmark_tier_objectives.py --case mask --replicates 3 --output target/mask-screen.json
PYTHONPATH='' target/frontier-sdk-env/bin/python scripts/probe_fused_decode_attention.py --measure --profile prefix_f32 --capacity 1024 --position 287 --width 64 --heads 2 --group 7 --iterations 256 --replicates 3 --output target/decode-attention-screen.json
```

Complete-response follow-up must compare each surviving candidate against its
matching installed incumbent at both memory tiers, preserve complete logits/KV
and one-use retirement, and charge loading, transport, state and cleanup. Low's
largest historical 288-token placement peak leaves only 24.94 MB below its
ceiling; that historical subtraction is not admission headroom for new buffers.
