# Architecture

PLLM is a mixed Python and Rust project with one Python distribution named
`pllm.run`. Maturin builds the PyO3 module `pllm._native` and packages it
alongside the `pllm` import package in `python/pllm`.

## Dependency direction

```text
Python SDK, CLI and runtime
            │
            ▼
pllm._native (`pllm-python`)
            │
            ├──▶ pllm-compiler ──▶ pllm-models ──▶ pllm-types
            │          │                 │
            │          ├──▶ pllm-core ───┤
            │          └──▶ pllm-garble ─┘
            ├──▶ pllm-bench ─────▶ pllm-compiler
            ├──▶ pllm-garble (bounded research probes only)
            └──▶ pllm-assurance ─▶ pllm-types

native providers ──▶ pllm-plugin-api (independent C ABI)
```

The Rust crates have no Python or web framework dependency. The binding translates
Python bytes into validated native inputs, releases the Python interpreter lock for
bounded execution, and returns typed handles or immutable bytes. Python retains model,
protocol, service, and application orchestration. SEAL/TenSEAL is a separate
cryptographic dependency.

## Crates

| Crate | Responsibility |
| --- | --- |
| `pllm-types` | Canonical records, strict digests, plan locks, privacy contracts, and shared evidence vocabulary |
| `pllm-core` | Bounded numeric primitives, immutable integer matrices, codecs, masking, SIMD dispatch, and reference kernels |
| `pllm-models` | Model-family configuration validation, semantic decoder IR, state contracts, and graph transformations |
| `pllm-garble` | Experimental one-use arithmetic-garbling material and evaluators |
| `pllm-compiler` | Model-neutral semantic scheduling, region lowering, fixed-scale research composites, and plan verification |
| `pllm-assurance` | Scoped assurance results and checked public fixtures |
| `pllm-bench` | Native and deployment measurement records tied to plan and environment digests |
| `pllm-plugin-api` | Independently versioned C-compatible native provider vtables, statuses, handles, buffers, header, and conformance fixtures |
| `pllm-python` | The PyO3 `pllm._native` boundary exposed through the `pllm.run` distribution |

`crates/pllm-core` contains the integer matrix executor, scalar reference paths,
runtime AVX2 and NEON selection, bounded coefficient arithmetic, codecs,
quantization, masking, output subtraction, and operating system random sampling.
NEON power-of-two modular products use exact ring32 accumulation with four
independent vector accumulators; odd moduli retain widened arithmetic. A
six-process synthetic Qwen3-shaped screen checks 2,138,112 output positions and
reduces modular process CPU 24–44%. Bounded bundle-frame decoding additionally
removes encoded-frame copies, saving about 1 MiB of Python-traced temporary
allocation. These isolated measurements do not establish complete Preparation,
request-latency or client-RSS improvements. A separate fresh-client Qwen2.5
150+8-token cold/repeat pair under native shared 100/40 caps reduces Preparation
run-window CPU 42.48%/43.87% and aggregate cold-first CPU 12.52%, while complete
request latency improves only 1.68%/2.62%. Outputs and covered application bodies
match with zero observed swap growth. Client lifetime peak RSS rises 248.87 to
261.82 MB; these single samples establish no client-memory improvement or
Qwen3-4B whole-response result. Replay and scope are in
`docs/research/prepared-kernel-reference.md`.
The prepared client additionally fuses quantize/mask/pack and residue
unpack/unmask/dequantize through bounded, interpreter-lock-released native calls.
Freivalds retains integer outputs until verification. Eighteen fresh synthetic
processes preserve 22,192,128 checked boundary values with 12–39% lower CPU and
up to 60.09% less Python-traced allocation. A separate matched Qwen2.5 150+8
cold/repeat SDK pair lowers client peak RSS 246.79 to 240.60 MB (2.51%), with
identical outputs and application bodies and zero observed swap growth. Cold
latency rises 0.56%, repeat latency falls 0.39%, and cold-first aggregate CPU
rises 1.12%; complete-request speed benefit remains inconclusive. Details:
`docs/research/prepared-boundary-reference.md`.
Preparation also subtracts output masks directly in uint32 before exact ring
narrowing, and prepared WebSockets decline deflate for arithmetic frames. A
separate matched Qwen2.5 150+8 cold/repeat pair improves decode TPS 19.41%/23.60%
and lowers Preparation peak RSS 218.87 to 200.59 MB. Cold request latency rises
7.99%, repeat latency falls 3.23%, client peak RSS rises 230.16 to 238.39 MB and
cold-first aggregate CPU rises 1.88%. Outputs and application bodies match with
zero observed swap growth; TCP-stream traffic rises slightly. A separately
costed public-artifact pre-encoding prototype lowers replay CPU but raises
publication-plus-first-read CPU about 5%; ordinary warm clients already reuse
their own object cache. It does not activate provider caching. Full evidence:
`docs/research/prepared-delivery-reference.md`.
The separate opt-in `pllm.native.MetalGEMM` MLX/Metal direct kernel accepts
bounded signed-i8 weights and i8/u32 inputs for exact clear and wrap32 outputs
on Apple Silicon. It owns a GPU weight snapshot and synchronizes output to
NumPy. The `AppleMetal` component can select it for client-owned or public
provider-owned integer stages at or above its digest-bound row threshold.
Prepared Inference and Preparation, the Freivalds-verified prepared placement,
and the two offset workers each bind their own GPU stage weights and use native
CPU for smaller work, including decode. Proprietary, FHE and incomplete
protected compositions cannot select it. Metal stage weights are preflighted
against a 2 GiB per-role bound before allocation. Transfer-inclusive
single-stage measurements must not be counted as whole-decoder or CPU-only
topology acceleration.
Prepared client-owned prefix layers and selected linear roles also compose with
Metal. Their client GPU snapshots are immutable and plan-bound; provider GPU
allocation excludes those body stages. CPU snapshots remain for small rows and
decode, and the benchmark counts both storage categories separately. A pinned
three-request Qwen cohort preserves outputs with almost equal CPU/Metal request
latency and another 44.04 MB of client GPU weights; compatibility alone does not
justify automatic device selection.
Core also contains a bounded, non-selectable coded-matvec verifier inspired by
Maverick's public preprocessing and sparse-check algebra. Its Walsh-linear code
proves half-distance over BabyBear for six to ten output rows; exponential code
length, no LPN input privacy, no batch projection delegation or full-width
resource admission prevent activation in a decoder or topology.
A separate bounded RAA numeric reference checks repeat/permute/weighted
accumulation, the sparse transpose, and offline mask-correction algebra against
field matrix products, without secure mask sampling, a code-distance certificate
or an executable verifier contract.
The separate Rust-only RAA delegation reference now composes those operations
at bounded full widths with fresh uniform fixed-weight OS-random sparse noise,
client-owned corrections, post-response coded challenges and non-cloneable
one-use query handles. A 256 MiB combined preprocessing/material payload cap and
eight-live-query limit precede issuance. It does not certify code distance or
dual-LPN security: the source's quoted finite-field distance-failure bound uses
`q > 2^31`, outside this reference's BabyBear field. Compiler-derived Qwen2.5
stage shapes have a synthetic-weight native reproduction, separate from
checkpoint values, decoder quality, provider transport and WAN evidence. The
22.90 GB whole-body `P+Q` lower bound remains a storage gate. Source analysis and
reproduction live in `docs/research/maverick-reference.md`.
The installed `pllm.protocols.MaverickDelegatedLinear` now exposes this bounded
algebra through ordinary operator benchmarks, with exact Python signed-integer
oracles and separate preprocessing, mask issuance, evaluation and verification
timings. Nine fresh processes check 50,688 outputs across a tiny control and two
Qwen-derived synthetic stage shapes. Client public payload remains 51.72/66.45 MB;
both MLP shapes reject before preprocessing. SDK admission also prices native
copies and scratch under 256 MiB and bounds product work. Code-distance and
dual-LPN certification, batch delegation and decoder transport remain absent.
Query payload, public tables and estimated field-boundary bytes remain separate;
there is no full-wire or process peak-memory result.
Matrices own their validated weights. Their dimensions and contents cannot be
mutated through the public Rust API. An executor owns a persistent Rayon pool.
The separate direct `pllm.native.PagedGEMM` API owns an authenticated private
file snapshot of raw or adaptive-zlib weight pages, performs bounded native
clear/modular/wrap32 operations and local row gathers, and retains no full weight
array in process memory between calls. Snapshot and decoded-weight hashes,
strict page/frame bounds and source-mutation isolation precede use. One pinned
head kernel's raw paging lowers isolated process peak RSS by about 10.9x while
increasing head CPU 2.42x; filesystem cache and total device memory are outside
that RSS measurement. Compressed pages save artifact bytes but regress repeated
online CPU substantially. Source artifacts and private snapshots occupy separate
disk storage. The explicit `ClientBundleTransport("artifacts", storage="paged")`
now streams weight objects into verified private files and binds native paged
token lookup, head and selected CPU body stages to the ordinary compiled SDK.
It verifies the original object/bundle/source identities before execution and
shares one native executor across snapshots. Public weights are fetched in full;
private token-indexed gathers remain local. Prepared, verified and offset tiny
prefill/decode, cache and gateway controls preserve exact logits and KV. Raw/zlib
artifact delivery and batching compose; client-owned paged body weights reject
Metal snapshot allocation. Admission prices bounded client page buffers and
separate cache/transient/snapshot disk owners. Whole-client real-checkpoint RSS
and matched full-response paging costs remain measurement gates.
Paged weight artifacts now stream into a bounded, authenticated immutable Rust
file shared by delivery and paged kernels. Actual-byte validation and original
source/object/bundle identities remain; no mutable path escapes. Independent
owners keep execution valid after delivery retirement, and the final owner closes
the snapshot. Admission charges one private weight payload separately from cache
and tokenizer disk. Twelve fresh synthetic import processes preserve 1,823,232
integer outputs and reduce import CPU 17–18%, while halving peak private-file
payload. A separate fresh-client Qwen2.5 150+8-token pair under native shared
100/40 caps preserves outputs and covered application bodies with zero observed
swap growth, but client peak RSS rises 222.54 to 237.13 MB and request time
115.041 to 116.738 seconds. Aggregate cold-first CPU rises 3.67%. The Qwen3
response pair rejects memory admission. These results establish no whole-client
memory or response-time improvement; import and logical disk savings retain
their isolated scope. Details: `docs/research/native-snapshot-reference.md`.
A separate Qwen3-4B Preparation-only raw-page probe matches all 144 stages'
71-row correction frames while reducing median process peak RSS from 4.61 GB to
739.09 MB (6.24x). Issuance CPU is nearly unchanged; loading plus issuance costs
12.41% more CPU and adds 3.63 GB of private snapshot disk. That historical
probe-only loader did not activate a Pipeline choice. The separately admitted
`ModelAwareCorrections(storage="paged")` now exposes CPU Preparation snapshots
through ordinary Experiments and prices their buffers and separate disk owners;
Metal and Freivalds Preparation reject this selection.
The core also owns the bounded `pllm.numeric.silu.quadratic_q7.v1` reference:
signed Q7 over `[-1, 1]`, deterministic ties-to-even rounding, and an encoded-domain
absolute SiLU error bound of `0.02285`.
It also owns exact reference primitives for bounded signed Q14-to-Q7 rescaling,
Q7 multiplication, and their gated-MLP composition with Q7 SiLU. Rescaling and
multiplication use deterministic ties-to-even division by 128 and reject inputs
outside their declared domains rather than saturating.
The Compact-inspired Q7 SiLU numeric reference fits at most eight Chebyshev
pieces from bounded public offline calibration counts, hashes the source,
calibration and fitted coefficients, and evaluates through plaintext interval
selection. Its planned nonlinear component remains non-executable until
protected selection and polynomial evaluation are composed with whole-model
evidence.
For LogRow research, core also defines a distinct public-range scaled SiLU
reference over `[-M, M]` for integer `M` from 1 to 16. Input and output use
Q7 scaling, ties-to-even rounding and strict domain rejection; its
conservative float32 absolute-error bound is `2.1 M / 256 + 0.000002`.
The range must be set from public offline information, never a private prompt.
`pllm-garble` also has a bounded, one-use, in-process half-gates reference that
selects a piece from a private Q7 input and returns opaque output labels to the
evaluator. Only the trusted client decodes the index. This selection circuit is
not yet composed with protected polynomial arithmetic or a compiler plan.
A separate one-use Boolean lookup oracle evaluates the committed Q7 profile on
hidden client input, leaving its output opaque to the evaluator. Its 257-row
table checks encoded fidelity, not Compact's polynomial execution cost; it
likewise has no transport, compiler binding, or whole-model coverage.
The half-gates Boolean builder now supports bounded signed division by a
public denominator with exact ties-to-even rounding. An isolated one-use
Compact circuit composes that primitive with private interval selection to
compute a Q20 normalized Chebyshev coordinate, but does not yet evaluate
coefficients or connect to a compiler plan.
A further bounded half-gates reference combines private selection, Q20
coordinate and public-coefficient Chebyshev arithmetic in one one-use circuit,
returning an opaque encoded Q7 result. It matches the fitted numeric profile
at checked intervals, but is an in-process research reference with no native
tensor schedule, transport, model-quality evidence or performance claim.
The compiler also binds one protected Q7 SiLU element to a validated semantic
decoder plan, phase, operation, tensor index, immutable Compact profile, code
digest, policy bound, and unique issuance. This experimental Rust-only API
checks context again before one-use evaluation; it does not activate a
Pipeline component, execute a tensor schedule, or transport material between
roles.
For public uniform calibration with one, four or eight requested pieces, its
half-gate ciphertext body exceeds the 257-row lookup oracle's body. The
polynomial reference has not earned tensor-scale resource admission or a
performance claim.

A separate bounded SIGMA/FuseFSS-inspired native reference compares unfused
truncate/reduce plus sign extension with mask-aware fused signed rescaling and
an original-input nonnegative predicate. Both layouts select the same comparison
backend: the initial quadratic arithmetic prefix-DPF, an independently
implemented linear-size DCF from FSS for Mixed-Mode Secure Computation, Figure 1,
or its Section 4.1/Figure 14 universal interval-key construction.
Floor and the separate PLLM ties-to-even extension pass independent
integer oracles. Party-local keys are non-cloneable, preflighted under a 64 MiB
combined allocation estimate and burned on completion, malformed frames, replay
or cancellation. The dealer remains in-process; context framing is not
authenticated transport or malicious-output verification. The initial twelve fresh-process
cases check 3,840 helper outputs. For 64 16-bit/seven-bit-shift ties-even lanes, fusion halves peer
bodies from 1,336 to 668 bytes but increases local online time from 29.98 to
49.32 ms and per-party/lane key payload from 5,888 to 9,428 bytes. A hypothetical
39+8-token Qwen2.5 SiLU-width placement projects 63.24/101.25 GB of keys for the
unfused/fused pair. A separate 24-process backend cohort checks 7,680 output pairs
and reduces fused per-party/lane keys from 9,428 to 1,718 bytes and matched local
online time from 96.32 to 11.13 ms, with unchanged 668-byte peer bodies. Compact
39+8 projections still require 15.31/18.45 GB of keys for unfused/fused helpers.
A further 24-process cohort shares one universal DCF per width within each
one-use lane/phase and checks 7,680 output pairs. Fused 16/7 keys fall from
1,718 to 758 bytes and issuance from 12.83 to 5.74 ms per 64 lanes. Seven
comparisons still execute, so online time barely changes (11.30 to 11.03 ms)
and peer bodies stay 668 bytes. Hypothetical unfused/fused 39+8 keys still
require 6.66/8.14 GB; phases, lanes and invocations never share one-use masks.
No Qwen numeric mapping, tensor schedule, distributed dealer or SDK component is
admitted; independent review and complete-operator cost reduction remain gates.
All three cohorts and exact source contracts are retained in
`docs/research/interval-rescale-reference.md` and its linked historical controls.
The installed `pllm.nonlinear.SharedRescale` component separately benchmarks these
fused/unfused helpers with 1–256 caller-owned or public signed-integer lanes,
2–32-bit widths, floor or ties-even rounding and all three comparison backends.
Both original-input sign and rescaled value pass an independent Python oracle;
each sample issues and burns fresh native party-local material with the
interpreter lock released. It reuses the immutable operator benchmark report,
including source citations, exact configuration replay, both-party material,
per-phase/direction frames and separate local issuance/online timings. The
in-process dealer, generic arithmetic-comparison contract and unknown wire/RSS
remain explicit. This exposes negative cost results without activating the
general FuseFSS compiler or a decoder.

The complete bounded shared Q7 gated block composes four exact rescalings and
two fresh Beaver multiplications in a 24-bit ring. Twelve fresh native cases
check 11,540 outputs, with all remaining block material burned on failure.
Interval keys with fusion need five peer rounds, 246,272 key bytes per party and
4,876 peer bytes per 64 outputs. Hypothetical Qwen 39+8 full-MLP geometry needs
41.33 GB of keys and 409.12 MB peer bodies, before other decoder operations.
On eight pinned public W8A8 prefill/decode traces, 36.07% of gate/up pairs exceed
the profile's public range; even the separate maximum scaled gate range has
418 violations. The reference neither clips inputs nor executes an unsupported
candidate decoder. Numeric mapping, cost, distributed issuance and independent
review still block admission. Details: `docs/research/nonlinear-block-reference.md`.

A separate public affine-SiLU reference binds Q8/Q9 precision, 16/64 mathematical
pieces, explicit zero/identity tails and wider gate/up ranges. Its complete
32-bit shared block compares scalar and vector coefficient lookup using universal
DCF, four exact rescalings and two fresh Beaver products. All 1,560 synthetic
shared outputs match independent integer oracles. On twelve fresh public Qwen
prompts, no input exceeds the selected ranges, but the best profile matches only
10/12 prefill and 35/36 same-token decode selections; no full logits or KV state
are exact. Vector lookup cuts matched complete key payload 11.51%, yet hypothetical
39+8 MLP material remains 60.10 GB plus 465.16 MB peer bodies across six rounds
per block. Rescalings occupy 86.42% of that material. This clear-checkpoint
numeric diagnostic and separate in-process protected block fail fidelity and
cost admission; secret float conversion, range enforcement, independent review
and role transport remain absent. Details: `docs/research/piecewise-gated-reference.md`.
The primary-paper cost audit confirms that raw payload arithmetic, including
51.94 GB for generic ties-even rescaling, but does not identify it with SIGMA or
FuseFSS costs. See `docs/research/fss-paper-cost-audit.md`.
The subsequent bounded masked-wire reference independently implements
early-terminated Boolean DPF comparisons, public-gap floor shifts, a 1,024-entry
Q12 residual SiLU lookup and masked multiplication. Public gate/up ranges prove
effective widths and every gap assumption. Non-cloneable party-local state burns
on completion, malformed frames, replay or cancellation; issuance preflights
at most 256 lanes under a 64 MiB combined allocation estimate. Twelve fresh
processes check 5,136 protected outputs. Complete-block payload is 5,162 bits
per party/element, projecting 6.93 GB combined material and 397.12 MB peer frames
at Qwen 39+8 geometry, before other operators and unimplemented protected float
conversion. This is 8.67 times less material than the numerically different
60.10 GB layout. The fresh twelve-prompt clear diagnostic matches 10/12 prefill
and 34/36 same-token decode selections; no full logits/KV are exact. It remains
cost/numeric-gated for recommended decoder execution, with in-process issuance
and no reviewed transport. The installed
`pllm.nonlinear.MaskedWireGatedMultiplyQ12` separately exposes this bounded operator
for user-driven research benchmarking, even when its quality/cost is poor. It
accepts public fixtures or caller-owned Q24 inputs, enforces the same range and
resource bounds, issues fresh material per sample and releases the interpreter
lock during native execution. Immutable reports retain replayable configuration,
source/profile digests, SIGMA/Grotto/Pika citations, independent integer parity,
separate float64 approximation errors, both-party material and phase/direction
peer frames. Client mask/boundary estimates and public tables stay separate;
full wire and peak memory remain unknown. No reusable secret handles escape.
SDK operator availability, paper coverage and whole-model execution are independent
claims. SIGMA GPU execution and FuseFSS general fusion remain unimplemented.
The separate `pllm.nonlinear.LogRowGatedMultiplyQ9` operator composes a private
4,096-row lookup with exact half-gates rescaling and multiplication through opaque
labels. Its Q18-to-Q9 numeric profile matches uncompressed Curl; the existing
Qwen diagnostic remains imperfect. Fresh one-use circuits retire serially under
a 16 MiB allocation estimate. Reports charge client encodings, evaluator lookup
and arithmetic material, constant labels, public metadata and boundary estimates
separately. No compiler tensor schedule or provider transport is admitted.
See `docs/research/logrow-product-reference.md` for the exact composition and costs.
The installed `pllm.nonlinear.CompactGatedMultiplyQ12` separately composes private
piece selection, polynomial SiLU, exact rescaling and signed multiplication.
Public density-weighted dyadic Chebyshev fitting, Q12 Horner rounding and explicit
zero/identity tails adapt Compact's source contract. Fresh one-use half-gates
circuits retire serially under a 64 MiB allocation estimate; the trusted client
alone decodes their final outputs. A fresh twelve-prompt W8A8 Qwen cohort has no
range violations, but its 8-piece cubic profile matches only 9/12 prefill,
35/36 same-token decode and 9/12 eight-token trajectories. No full logits or KV
are exact. Each frozen profile checks 131,444,736 independent integer outputs.
Nine fresh operator processes check 2,889 protected outputs; hypothetical 39+8
MLP geometry needs 1.97–3.42 TB of evaluator material plus 13.40 GB of client
encodings. Polynomial arithmetic dominates this Boolean adaptation's costs.
Operator availability, clear checkpoint numerics and protected decoder execution
remain separate; tensor scheduling, role transport and protected float conversion
are absent. Details: `docs/research/compact-product-reference.md`.
The optional `backend="mixed"` on the Curl, LogRow and Compact complete gated
operators composes party-local arithmetic products with explicit half-gates
conversion, selection and exact rounding. A local trusted dealer supplies fresh
random-OT correlations and Beaver triples; output-label hash commitments keep the
free-XOR delta with the garbler. Thirty-six fresh processes check 2,916 protected
outputs. LogRow material falls 24.9% and Compact 12.2–29.6%, while Curl regresses.
Whole-MLP projections remain 203 GB–3.02 TB plus peer traffic. A fresh twelve-prompt
Qwen diagnostic checks 128,876,544 independent integer outputs per frozen profile
but preserves no complete candidate logits/KV. Defaults and numeric identities
stay unchanged; distributed setup, transport and decoder scheduling remain absent.
See `docs/research/mixed-gated-reference.md` for ownership, full costs and replay.
The separate one-evaluator cost screen binds fresh installed LogRow and Compact
scalar reports to native-checked Qwen2.5 MLP geometry. At 39+8 tokens, evaluator
material alone requires 326.40 GB for LogRow and 1.82–2.92 TB for two default-public
Compact profiles, against a historical 116.84 MB covered-body control. Client
issuance keeps encoding material local; hypothetical offline Preparation issuance
adds its separate client delivery. Boundaries and public metadata stay separate
estimates. These changed numeric profiles and unimplemented role compositions
cannot become a prepared decoder through cost projection. Details and replay:
`docs/research/prepared-region-gate.md`.
The masked-wire component now also selects experimental CPU two-offset gated
regions. A native model-neutral pass binds each dense float32 SiLU/multiply pair,
the fixed numeric profile, floor float32-to-Q24 and Q12-to-float32 client boundaries,
256-element chunks and a per-chunk 64 MiB combined material-allocation ceiling. The trusted
client issues fresh party-local keys online and relays opaque peer frames between
authenticated workers; ordinary application counters charge both relay legs and
key delivery. Session/phase/region/index bindings and monotonic worker admission
burn malformed, replayed, cancelled and expired material. Whole-topology memory
admission prices one client chunk and eight possible chunks per worker. The opt-in
`dispatch="overlap"` binds one additional client-prefetched chunk and concurrent
worker requests within each phase. Failure and cancellation drain outstanding
calls and burn unused material. Client chunk/frame admission grows from 70 to
140 MiB before margin; each worker still owns one active chunk per session. Network
offers without nonlinear resource pricing reject. Generated Qwen2/Qwen3 whole
prefill/decode matches independent integer-oracle logits and KV in both layouts;
separate-process SDK and gateway requests complete. These functionality checks
do not establish real-checkpoint quality, independent operators or cryptographic
review. Clear-SiLU and changed nonlinear profiles have separate benchmark numeric
contracts even if output selections agree.
A twelve-case synthetic Qwen-width region cohort checks 175,104 exact outputs.
At shared native 100/40 caps, overlap lowers unfused/fused row latency from
3.55/6.34 to 2.16/3.77 seconds with unchanged key and traffic bytes and nearly
unchanged aggregate CPU. Worker cases observe zero swap growth. This bounded
region timing excludes whole-response execution and peak-memory comparison;
details and replay are in `docs/research/nonlinear-decoders.md`.
A separate fresh-client Qwen2.5 30+2-token SDK pair confirms 39.61% lower complete
request latency with unfused overlap: 2,705.809 to 1,634.109 seconds under native
100/40 caps. Client peak RSS rises 227.20 to 229.49 MB; aggregate CPU remains
nearly equal and both transfer 5.72 GB of TCP-stream bytes. Outputs match with
zero observed swap growth. This single-response pair excludes physical wire,
checkpoint distribution and independent operators; the Q12 quality gap remains.
An independent test-only native issuance/codec profile checks 1,926 outputs and
attributes most unfused key-generation time to three rescalings. Paired AES
expansion and allocation-free encoder counting subsequently preserve 11,556
checked outputs while reducing 256-lane issuance 28.22%/33.22% for unfused/fused
layouts, with unchanged key bytes. The separate opt-in
`MaskedWireGatedMultiplyQ12(material_encoding="seeded")` regenerates only each
party's independent random key fields from fresh role/context-bound seeds.
Client wire masks retain independent OS randomness and correlated corrections
stay explicit. Packet namespaces and compiler/session bindings reject encoding
substitution. Worker expansion retains full key allocation bounds and one-use
burn semantics. A 175,104-output synthetic Qwen-width cohort lowers unfused/fused
capped row latency 2.111/3.706 to 1.784/3.353 seconds and 256-lane key payload
21.02%/12.94%, while aggregate process CPU rises about 1.9%/1.5%. This local
encoding adaptation has no independent cryptographic review; row-region evidence
does not establish complete-response savings. A separately matched fresh-client
30+2-token whole-response pair on the optimized implementation measures 14.79%
lower latency (1,611.722 to 1,373.282 seconds) and 17.17% less TCP-stream traffic
(5.7166 to 4.7350 GB) with seeded packets. Its 981,603,840-byte key reduction
exactly accounts for the stream and application-body differences. Aggregate CPU
rises 2.73%; client peak RSS falls 228.84 to 225.72 MB. Outputs match with zero
observed swap growth. Both candidates retain trusted-client online issuance and
the same changed Q12 numeric profile; earlier implementation cohorts stay separate.
The optional `MaskedWireGatedMultiplyQ12(layout="fused")` now implements bounded
multi-view fusion on the complete block: rescaling, index reduction and predicates
derive from one masked gate input. Eighteen fresh processes check 5,778 exact outputs.
Rounds fall from eleven to eight; 64-lane peer frames fall from 6,020 to 5,552 bytes,
while both-party keys rise from 82,592 to 145,984 bytes and local online time from
7.31 to 17.51 ms. Wider direct-input comparisons cause the material increase.
The historical 39+8 geometry projects 6.93/12.25 GB unfused/fused keys and
397.12/387.30 MB peer frames. Both retain the same numeric profile and bounded
operator availability; compiler tensor binding and role transport remain separate.
Both operator methods use the ordinary immutable `NonlinearProtocol` component
contract, discovery and serialization. `pllm.metrics.benchmark_operator` owns
benchmark geometry and samples separately; `replay_operator_benchmark` also
migrates historical probe-v1 configurations without rewriting their evidence.
Details: `docs/research/masked-wire-reference.md`.

Compact, Curl and LogRow additionally have bounded installed operator bindings:
`pllm.nonlinear.CompactPiecewiseActivation`,
`pllm.nonlinear.CurlWaveletEncodedTable` and
`pllm.protocols.LogRowGarbledLookup`. They share public Q7 calibration, immutable
component discovery/serialization and `benchmark_operator`, with independent
integer oracles and separate float64 error. Compressed Haar tables change the
numeric identity. Reports identify garbler/client versus worker ownership,
charge constant labels and client encodings, retain public metadata/boundary
estimates and separate public fitting, issuance and evaluation timings. Serial
lanes burn material before the next issuance. These operator APIs do not yet
bind whole-decoder execution or transport. Historical ciphertext-only evidence
retains its original scope; see `docs/research/nonlinear-frontier.md`.

The separate installed `pllm.nonlinear.CurlWaveletGatedMultiplyQ9` composes Haar
lookup with protected index reduction, exact up/output rescaling and fresh Beaver
multiplication. It admits 1–256 Q18 lanes with gate/up bounds ±48/±128 and Q9
output, under a 64 MiB combined allocation estimate. Party-local material burns
on completion, malformed frames, replay or cancellation. Its exact interval-key
helpers replace Curl's probabilistic truncation. Nine fresh processes check 2,889
protected outputs. At hypothetical Qwen 39+8 geometry, levels 0/2/4 require
213.08/81.58/49.06 GB of combined keys and unchanged 971.27 MB peer frames;
four-level exact rescaling alone costs 37.91 GB. A fresh twelve-prompt compiled
W8A8 diagnostic fits the fixed ranges but matches only 10/7/8 prefill selections,
35/33/33 same-token decode selections and 10/5/4 eight-token trajectories. No
candidate logits/KV are exact. Each clear mapping matches 130,043,904 independent
integer outputs. This bounded operator keeps negative results benchmarkable;
complete cost, fidelity, protected float conversion, distributed issuance,
independent review and compiler/transport binding remain decoder gates. Details:
`docs/research/wavelet-gated-reference.md`.

A further clear joint-output reference uses Q12/Q16 inputs, 512/2,048 public
SiLU secants, explicit tails and one final rounding of the complete gated
numerator. Checked i128 arithmetic and an independent segmented-integer oracle
agree on 253,550,592 checkpoint elements. Q12/2,048 matches all 12 prefill and
36 same-token decode selections in a fresh cohort, but no logits/KV are exact
and worst logit error is 2.78; the other three frozen profiles miss selections.
Its fully specified dense two-input function shares require 8.80 TB of tables
per Q12 element. An uncompressed per-lane half-gates layout already exceeds the
116.84 MB historical whole-response control with a one-AND-per-element floor
of 171.84 MB at 39+8. Mask-wrap, sharewise-rounding and public-coefficient leakage
counterexamples prevent treating those operations as free. No protected material
is issued; compressed joint keys, conversions and complete costs remain absent.
Details: `docs/research/joint-gated-reference.md`.

`crates/pllm-python` contains only the Python binding. Maturin builds this crate
as `pllm._native`. It binds `pllm-core`, `pllm-models`, `pllm-compiler`,
`pllm-types`, `pllm-bench`, and `pllm-assurance` through PyO3. The stable Python
ABI is configured from Python 3.11. The application dependency matrix currently
limits Python to 3.11 through 3.13. `pllm-plugin-api` is a separate C boundary;
plugins do not link to PyO3 internals, and runtime loading is not yet exposed.

`crates/pllm-models` owns the model-family-neutral semantic decoder IR. Semantic
operators, layer identity and persistent-state kinds are explicit, so compiler and
method passes do not parse adapter-specific node names or weight paths. Implemented
lowering adapters are Qwen2 (`pllm.qwen2.v1`), dense Qwen3 (`pllm.qwen3.v1`),
bounded bias-free unscaled dense Llama-style configurations
(`pllm.dense_gated_decoder.v1`),
the exact Qwen3.5-4B text decoder (`pllm.qwen3_5_text.v1`), the exact
Phi-4-mini-instruct decoder (`pllm.phi4_mini.v1`), and the exact text decoders in
the official `google/gemma-4-E2B-it` and `google/gemma-4-E4B-it` outer
configurations (`pllm.gemma4_e2b_text.v1` and `pllm.gemma4_e4b_text.v1`, model
family `gemma4_text`). Other Qwen, Gemma, Nemotron, Kimi, GLM and future families
must lower into the same IR or extend its semantic vocabulary rather than
introduce family-specific compiler paths.
Source-format identity, exact supported variants and checkpoint artifact paths for
hybrid-text, fused-dense and shared-KV decoders live in
`crates/pllm-models/src/source_mappings.rs`; reusable graph construction lives in
`hybrid_text_decoder.rs`, `fused_dense_decoder.rs` and `shared_kv_decoder.rs`.
Their builders accept mapped weight roles and source-validated architectural
dimensions rather than hard-coded model names or weight paths. New source formats
must map into these contracts or add a genuinely reusable operator/state contract;
do not create model-family-named Rust graph files. The older Qwen2/Qwen3 and dense
Llama reader still has graph code in `lib.rs`; it is legacy code to generalize,
not a pattern for new families.
Capability modules transform this IR through generic component contracts and
record immutable, digest-bound transformation lineage on the resulting plan.
Substitutable implementations are grouped by capability; a new family is added
only when a method has a genuinely different contract or lifecycle.

Semantic adapter support, checkpoint import, runtime graph support, compiler
operator coverage, protected/private parity, generation quality, benchmark
evidence and deployment support are separate claims. A complete bounded semantic
plan does not establish any later claim, and coverage is composition-scoped.

For batch-one untransformed plans, `baseline.masked_linear_cpu` now builds one
model-neutral prefill/decode schedule from semantic operators, dependencies and
declared weight artifacts. Independent remote operators with the same semantic
input are grouped without parsing adapter-specific node names or weight paths; all
other implemented operators are placed locally in dependency order. The Python
binding resolves those weight groups against checkpoint stages, verifies exact
shapes, configuration, tokenizer, local tensors, quantized bytes, scales, modulus
policy and preparation commitments, and rejects any unresolved operation or stage.
The same compiled binding and execution path is covered for tiny Qwen2, dense
Qwen3 and bounded bias-free Llama-style checkpoints, including Qwen3 Q/K
normalization and an untied head. Bounded Llama 3 wavelength-transition RoPE
now also passes a generated tiny checkpoint through native scheduling,
prefill/decode, a PyTorch float32 reference (matching selected tokens; W8A8
logit error under 0.05), and client-only/prepared SDK and gateway requests.
The pinned public Llama 3.1 8B configuration mirror additionally lowers and
compiles for a bounded 8+2-token workload without loading weights.
Other scaling modes and real Llama checkpoint import, quality and deployment
remain unvalidated. A separate pinned SmolLM2-135M-Instruct checkpoint uses
the same generic bias-free dense adapter: it imports 120 compiled stages and
completes a 39+32-token prepared request through two local role children with
masked traffic and no plaintext prompt/token-ID bytes sent. A distinct
12-prompt local W8A8 prefill cohort matches its own float32 top token 11/12
times, with 3.8734 worst absolute logit error. Covered warm all-link bodies
are 88.40 MB and online bodies 55.68 MB; two isolated-client-bundle-cache
cold samples have 23.48 s median aggregate CPU. The smaller model has no
cross-model task-quality ranking, independent-provider privacy proof,
same-token W8A8 decode parity or full-wire measurement. All-layer rank-128
MLP output cuts fail held-out prefill/decode parity; they remain offline
float32 oracles, not a protected decoder. Pinned Phi-4-mini-instruct
additionally compiles for an original-context workload. Generated tiny Phi
fused-QKV/gate-up weights import into that schedule and pass typed-session
W8A8 prefill/decode against a PyTorch
float32 reference (matching selections; worst logit error below 0.05), plus
client-only/prepared SDK and gateway requests. Phi's original-context factor
vector stays fixed for the response; crossing into extended LongRoPE context
fails closed before execution because retained KV would need re-rotation.
The pinned official Phi BF16 checkpoint additionally imports 194 verified
artifacts into 130 compiled stages and completes a short original-context
local-clear prefill/decode. A two-prompt locked float32 reference-quality
comparison scores 0/2 W4A4 and 1/2 W8A8 top-token agreement (worst absolute
logit errors 40.67 and 29.90). A separate same-token diagnostic with float32
body projections and the W8 token/head boundary recovers both checked top
tokens, identifying compounded activation and weight quantization as the
immediate fidelity gate. This is functionality and a narrow quality measurement,
not validated generation quality or protected execution. A separately selectable
public offline per-channel equalized W8A8 component fits an immutable,
source-locked profile from public token IDs, binds scales into body/stage
digests, applies them at the trusted client and uses the same prepared role
protocol. The pinned Phi cohort scores 2/2 top-1 with 9.59 worst logit error
versus plain W8A8's 1/2 and 29.90; five other public prompts score 3/5 for
both W8A8 choices. This narrow improvement does not establish broad quality.
Matched real-checkpoint prepared requests completed through two separate
co-located child roles. Both sent zero plaintext prompt/token-ID bytes and
performed 128 masked remote stages. Equalization reduced covered online
application bodies by 827,392 upload and 802,816 download bytes, while its
2.23 MB public profile increased cold bundle distribution. Full wire costs,
full-response aggregate compute, operator independence and a real-checkpoint
gateway request remain unvalidated. The pinned
Qwen2.5-0.5B-Instruct checkpoint additionally passes a clear native-kernel
prefill-to-decode functionality test; a separate tiny test exercises the masked
stage protocol. Admitted compiled public decoders derive provider stage tables from ordered
weights and declared biases from that schedule; the client binds it and runs
semantic prefill/decode operations before reserving inventory. The inference
role re-lowers the bounded plan and checks its digest, body, stage and runtime
configuration commitments before accepting a compiler-bound session. This
covers `gateway --local`, bare public client requests and loopback benchmarks;
standalone experiment-backed provider roles receive the same numeric choices.
Gemma 4 E2B/E4B semantic plans pass the same bounded baseline prefill/decode
compiler scheduler. Pinned E2B additionally imports 213 compiled stages and
executes two complete local-clear W8A8 prefill-to-decode runs. Compared with
the same-token PyTorch BF16 reference, the two-prompt cohort records 2/2
top-1 agreement, 0.70 mean top-5 recall and 3.5625 worst absolute logit
error. A separate prepared HTTP request exercises the same E2B checkpoint
through distinct in-process inference and preparation service instances. Its
client audit records masked online traffic and zero plaintext prompt/token-ID
bytes sent; the 2,790,944,732-byte client bundle is delivered in bounded
chunks. Neither the two-prompt numeric cohort nor the single-request loopback
test establishes representative generation quality, independent providers,
or a two-child deployment. Generic local BF16 scale, tanh-GeLU, softcap,
permutation and slice primitives have isolated schedule contracts and a
finite-BF16 numeric oracle: on the checked PyTorch CPU implementation the
tanh-GeLU result differs by at most 0.0000305 over all 65,280 finite BF16
inputs, and softcap with cap 30 agrees exactly. The one-element checkpoint
scalar now has an exact-shape client-tensor binding; Gemma 4 RMSNorm uses
direct checkpoint weights, without a `+1` offset. BF16 output boundaries are
now declared on local and remote operators, checked end-to-end against the
semantic graph, and rounded by the client executor. This is an isolated
representation contract, not independent cryptographic privacy evidence.
The exact source configuration, sliding/full KV ownership, PLE token boundary,
and runtime stages are now bound for the E2B local-clear and prepared-loopback
diagnostics. The older runtime graph remains separate for existing tiny
transport tests. Sources without semantic lowering adapters still
retain their older inspection/runtime graph; an adapter with incomplete compiled
coverage cannot silently activate that graph, except for the separately labeled
Gemma 4 compatibility path. Verified and proprietary live runtimes likewise
remain outside this compiler-bound baseline until their executor and trust
contracts are implemented. No matched provider-cost or broad generation-quality
claim follows from one loopback response. The compiled decoder produces
logits; request-level temperature and top-p selection are still applied by the
client outside the plan's greedy reference operator, and are not included in its
execution digest.
The standalone semantic sliding-KV reference retains only the valid `W−1`
prefix, constructs query-relative windows on demand, rejects overlarge views
before state mutation, and preflights declared windowed state against a 2 GiB
client-memory ceiling before runtime construction. The compiler admits its
bounded operation schedule; the local-clear E2B run covers whole-decoder PLE
execution, and one prepared in-process two-service response checks provider
protocol connectivity. Multi-process and multi-host parity remain open.
Pinned default and proportional RoPE descriptors now bind sequence-before-heads
layout and BF16-stepwise arithmetic; checked positions match a PyTorch CPU
oracle, while real-checkpoint and full-context rotary fidelity remain unproven.
The pinned Gemma 4 E2B checkpoint's main token table and packed per-layer token
table are physically separate, share token IDs, and form one ordered public
boundary stage with two output slices. Before any stage quantization, semantic
import checks the shapes and floating-point dtypes of all 540 required text
decoder artifacts against the cached official checkpoint's headers. The separate
local-clear functionality test loads their values, while the separate prepared
loopback exercises the provider stage protocol. Neither establishes
representative generation quality or independent provider deployment.
An explicit in-process LogRow research override can evaluate all SiLU tensors
of a bounded tiny Qwen2/Qwen3 compiled decoder with one-use, session-admitted
material and a separate execution digest. The wider public-range profile
executes the ordinary generated tiny checkpoint. This override is not an
executable Pipeline/Experiment choice, does not expose a provider-protected
nonlinear path, and does not establish real-checkpoint quality. Retained matched
local cohorts show higher online CPU and offline issuance cost than clear SiLU.

There is no `research.single_evaluator` profile. The fixed-Q10 research path is an
ordinary component composition with executable regions for dense gated-decoder operators, graph-derived
Q14-to-Q10 edges, clear attention and layer composites, and bounded one-use Q7
SiLU/multiply material, but those protected and fixed-scale components are not yet
composed into a real-model whole decoder. A bounded native MPCache selection
oracle preserves its full-cache Qwen control but static retention matches only
34/36 same-token decode selections, combined selection 32/36 and layer-shared
selection 31/36.
It retains full backing KV and uses client-local plaintext gathers; selected
addresses depend on private queries. A separate native retained-row KV owner now
physically stores only selected float32 rows plus bounded decode capacity and
original token positions. Its 64 MiB per-owner payload limit, atomic append and
compaction, zeroizing retirement and absolute-position attention pass independent
storage/numeric controls. Caller conversions, exports and compaction overlap are
additional owners. The internal compiler-bound retained-state extension now
executes static client-local retention through the same semantic decoder, with
a separate source, policy, state and execution identity. It admits bounded
exclusive dense KV owners, prices retained/dense/transient/workspace payloads
together, preserves original causal positions and retires every response owner
on failure. Full-KV snapshots and continuation reject retained state. Generated
Qwen2/Qwen3/Llama full-retention prefill/decode preserves every logit and KV bit;
static retention matches an independent selected-view oracle and releases dense
backing arrays. A fresh twelve-prompt pinned Qwen cohort preserves all 48 logit
vectors and final KV at full retention, reducing median owned KV from 6.29 to
5.16 MB through capacity sizing. Keeping 75%/50% uses 3.90/2.62 MB but each matches
only 35/36 same-token decode selections and no complete decode logits or final KV.
Local-clear process peaks, including full body weights, remain 1.21–1.23 GB while
diagnostic CPU rises. Explicit `ClientLocalKv(retention=KvCacheEviction(...))`
now selects the static contract in ordinary client-only and prepared Experiments.
Generated SDK/gateway controls include separate prepared role children, completion
and cancellation retirement, and client-local message-history re-prefill. Retention
policy binds benchmark numeric identity; native placement and benchmark admission
price state/workspace owners. Two fresh-client prepared SDK cohorts compare dense,
full, 75% and 50% retention at 52/379 input tokens plus eight outputs under native
100/40 caps. Full retention preserves output text and lowers client peak RSS from
264.90/546.31 to 257.77/524.85 MB (2.69%/3.93%). Sparse retention does not monotonically
reduce peak RSS; the longer 75% response changes its text. Covered bodies remain
257.79/989.08 MB across policies and latency is nearly unchanged. Every individual
runtime check passes with zero observed swap growth, while distinct numeric
contracts correctly prevent a cross-policy ranking. These one-response local
samples do not establish broad quality or a 10× client-memory improvement. The
original longer workload's residue-frame rejection remains separate from memory
admission and the admitted rerun. Protected provider-held selection remains gated.
A separate Rust-only selection substrate
garbles stable top-k and complete-record movement through fixed bitonic networks.
The trusted client supplies private nonnegative float32 scores and complete KV;
the evaluator receives opaque input/output labels without a decoder. Fresh owner,
layer, revision and issuance bindings reject stale or substituted results, while
the client alone commits exact retained bits. One outstanding query per owner
and a 64 MiB combined allocation estimate precede issuance. The initial
16-row/16-feature cohort matches 474 independent-oracle output words in twelve
fresh processes.
Selecting eight of sixteen two-feature rows requires 1.10 MB of label/ciphertext
material plus 2.06 MB estimated public metadata to retire 160 bytes of state.
The 16×16 allocation gate and initial full-width feature gate reject larger cases.
Caller-declared plan context, local garbling and logical body counts do not
establish compiler-bound provider state, distributed scoring or transport;
independent review and cheaper complete selection remain required.
A second layout sorts compact private indices then performs a complete one-hot
oblivious gather. Thirty-six fresh processes check 4,041 exact output words,
including two/four-row 128-feature KV. Both layouts share exact geometry-only
resource counts and admission before allocation; owner/revision and one-use checks
also bind the layout. Compact records lower metadata estimates but the gather
increases logical label/ciphertext bodies in all five matched cases. At 256 rows
and 128 features, half retention projects 1.77/8.71 GB per layer for full/compact
layouts, before public metadata. All complete-state Qwen shapes reject before
issuance, so no provider transport is activated. Details:
`docs/research/mpcache-reference.md`.
The later mixed-mode selector replaces complete-record Boolean multiplexers with
fresh ring32 Beaver products; one-shot garbled predicates retain explicit
random-OT conversion costs. Thirty-three fresh processes check 171,303 complete
output words. Top-k admits 256 rows of 128-feature K/V plus absolute positions
under a 15.45 MB live-allocation estimate, but needs 295.26 MB of both-party keys,
78.59 MB peer frames and 19,200 serial rounds to retire 131,584 clear-state bytes.
Protected scoring, live owner/revision binding and provider transport remain
absent. The same substrate exposes `CipherPruneTokenPruning(backend="mixed")`
through ordinary operator benchmarks and replay. Partial pruning improves the
matched small half-gates control; empty and keep-all cases regress. Count and
degree disclosure remain explicit, and every possible count is preflighted
under 64 MiB live allocation, 512 MiB cumulative material and bounded product
work. Complete 39×896 synthetic pruning executes with 13.09 MB keys and 8.39 MB
peer frames; 150-row worst-case admission rejects. These are independent bounded
operator results, not decoder memory savings or reviewed role transport.
The explicit mixed `batch_size=4` or `16` coalesces independent primitive phases
under the same admission ceiling, retaining fresh material and context bindings
for every operation. A public dependency-preserving compaction wavefront hides
selected positions within the existing count disclosure. Thirty-six fresh
processes check 468,585 complete outputs. At 256×257 top-k, size 16 reduces
exchanges from 19,200 to 1,200 with unchanged 295.26 MB keys and nearly unchanged
local CPU. Outer framing raises peer bodies 78.59 to 78.77 MB and the conservative
live-allocation estimate rises 15.45 to 58.29 MB. Complete 39×896 pruning drops
688 to 78 exchanges, retaining 13.09 MB keys. These local round reductions do
not establish WAN latency, protected scoring or live provider-state integration.
Details: `docs/research/shared-selection-reference.md`.
The separate bounded-metadata top-k layout narrows only public-proven predicate
domains: nonnegative finite float32 score bits, stable reverse indices and
real/padded/recent priorities. Complete record words and movement products stay
ring32. Eighteen fresh processes check 239,010 complete words with exact
issuance/frame reconciliation. At 256×257 and batch size 16, keys fall from
295.26 to 164.74 MB and peer bodies from 78.77 to 50.35 MB; 1,200 exchanges and
the conservative 58.29 MB live-allocation estimate remain. No private score range
selects a width, and threshold pruning rejects this contract. The cost still
exceeds retiring 131,584 clear-state bytes by orders of magnitude; protected
scoring, live state binding and provider integration remain gated.
The separate common-profile Compact/LogRow/Curl
screen counts activation-only material and missing conversions explicitly;
four-level Curl still projects 1.40 GB material and 730.30 MB peer bodies at
39+8 before the gated product. NEXUS/THOR plaintext exponential references reject
eight Qwen prompts under their fixed public score range, producing no candidate
quality or ciphertext result. These are component/source gates, not admitted
decoder protocols. Transformed MPCache and compiler-bound
E4B checkpoint execution remain incomplete. The pinned Qwen3.5 text-only graph
now compiles through shared convolution, gated-delta recurrence, partial-MRoPE,
gated normalization and persistent-state contracts. Its generated four-layer
checkpoint binds the complete native stage schedule and passes client-only W8A8
prefill/decode against the upstream float32 Torch reference, with identical
selected tokens and worst checked logit error below 0.05. Separate prepared SDK
and gateway requests complete two generated tokens through two local role
children with masked stage traffic and no plaintext prompt/token-ID bytes sent.
The pinned official Qwen3.5-4B checkpoint additionally validates 426 required
text artifacts, imports 130 compiled stages and completes prefill/decode.
Against an independent upstream BF16 same-token reference, W8A8 matches 2/4
selected tokens with 3.66 worst absolute logit error; a float32-body
diagnostic with the same W8 token/head boundary matches 3/4 with 0.46 worst
error. A real-checkpoint SDK and gateway response each complete through two
local prepared-role children, with masked traffic and no plaintext prompt or
token-ID bytes sent. This narrow cohort does not establish representative
quality, independent operators or matched provider costs. The text rotary
contract still rejects multimodal position axes and prompt image/video tokens.
The pinned
Qwen3-0.6B checkpoint passes a local clear-kernel compiled prefill-to-decode
functionality check, but W4A4 and one tested short-prompt W8A8 case
diverge from the FP32 reference. Multi-process provider deployment, broad quality and matched-cost
evidence remain open. Existing runtime support for a checkpoint family is a
separate axis unless an exact schedule, binding and execution test say otherwise.

## Python package

The opt-in `OpenAI.conversation` and `AsyncOpenAI.conversation` handles manage one
response-owned continuation chain under an Experiment with bounded
`ClientPrefixReuse`. Native full-KV admission and compiled maximum state geometry
precede reservation. Each successful turn retires the previous response/history/KV;
failed turns preserve the prior owner. Async cancellation drains the worker and
retires the chain, including a concurrently committed turn. The selected 1-byte to
256-MiB bound prices one retained KV/logit snapshot; transient execution, snapshot
overlap, text and the separately bounded prefix cache remain additional owners.
Generated Qwen2/Qwen3 checks preserve every compared logit and KV bit against
explicit response continuation. This is bounded ownership, not measured whole-client
RSS or a new numeric/cryptographic protocol. Example: `examples/conversation.py`.
Continuation admission now validates its public source/composition/schedule once
into an immutable native handle retaining only three resource limits. Per-query
overflow, geometry and workspace checks remain unchanged; deserialization
revalidates the contract. Runtime binding reuses its already-checked schedule.
An isolated six-pair, 32-admission Qwen metadata probe lowers median CPU from
1.170 to 0.034 seconds, without measuring whole-client RSS or response latency.
Evidence and replay: `docs/research/sdk-continuation-reference.md`.

The installed `pllm.metrics.BenchmarkTarget` and `benchmark_suite` share bounded
fresh-process execution across runtime imports, native geometry, selected artifact
imports, prepared stages, operators, synthetic state, cost projections and ordinary
role responses. Immutable cases, sampling, build locks and metric budgets produce
`BenchmarkResult` v2 records in the existing evidence registry; v1 retains its
original semantics. Private inputs and a common private cohort salt pass over
stdin. Source/numeric/state, fixture, oracle, owner and measurement-window
commitments gate comparison; timeouts and cancellation retire owned processes.
`pllm.search.FrontierStudy` expands explicit variants, interactions and ablations,
prioritizes affected cases and reruns controls alongside missing candidates.
Exact-build coverage and separate confirmation preserve missing, unsupported and
blocked cells rather than propagating old timings. Response targets retain all
seven serving metrics through the ordinary benchmark driver. Selected-stage
synthetic fixtures, local sealed-state references and digit-lifted CPU GEMM are
separately scoped research targets, not serving components. The five tenfold
challenge screen completes 123 fresh processes with 30,832,128 independent
correction checks and 2,377,728 synthetic state/attention checks. Runtime floors
miss both tenfold RAM budgets. Eight-bit digit GEMM reduces one large-stage CPU
62.58% but increases memory, and four-row chunking reverses its CPU benefit.
Sealed state lowers scoped peak RSS 33.03% while multiplying attention CPU 7.07×.
Restart-state mechanics and explicit horizon estimates establish no resumed
decoder or tenfold serving result. Full evidence and API feedback live in
`docs/research/benchmark-targets-reference.md`.

The maintained prepared-serving study now derives its control from
`examples/prepared_sdk.py`, exports exact coverage and seven-metric snapshots,
and constructs independent finalist/control confirmation cohorts through
`FrontierStudy.confirm`. Runtime decomposition records independent import-child
footprints and module roots; ordinary text clients defer JSON-schema loading to
strict-tool requests. A separate opt-in
`ModelAwareCorrections(kernel="digit8", min_rows=64)` binds exact Rust-owned
balanced-byte lifting to CPU Accelerate on Apple Silicon. Bounded 1,023-product
float32 blocks and wrapping reconstruction preserve u16/u24/u32 corrections;
small batches retain native integer kernels. Resident and authenticated paged
weights share the implementation. Admission prices 224 MiB of extra workspace
per concurrent Preparation stage, and unpriced network offers reject. This
implementation has separate measurements from the historical NumPy research
kernel. A twelve-client Qwen2.5 150+8-token cold/repeat confirmation under native
shared 100/40 caps reduces cold-first Preparation CPU 25.03% with digit8. The
windowed cold-request median improves 1.94%, while sampled Preparation RSS rises
5.69%; serial issuance retains a different latency/memory tradeoff. Outputs and
covered application bodies match with zero observed swap growth. These bounded
three-replicate results do not establish a tenfold gain or automatic selection.
A separate six-client installed-build comparison with integer Preparation keeps
outputs and application bodies equal and lowers median client peak RSS 222.13 to
218.15 MB (1.79%). Cold/repeat latency remains nearly unchanged with zero observed
swap growth. Its small whole-client result stays separate from import-only and
digit8 measurements.
Contracts and replay: `docs/research/prepared-maintained-reference.md`.

`pllm.search.BeamSearch` complements grid and random search with bounded,
feedback-guided exploration of immutable `SearchSpace` component combinations.
The shared `pllm.benchmark_search` facade runs all three through ordinary SDK
role benchmarks, with per-trial memory admission and one private cohort salt.
Source, numeric body, workload, output, topology, verifier and metric semantics
gate ranking; rejected trials and unknown measurements remain explicit. Reports
retain exact Python factories for replay. Best measured neighborhoods are not a
global optimum certificate, and finalist confirmation remains separate from
single-response exploratory tuning. Client process-lifetime peaks and omitted
public-artifact distribution do not become comparable costs through search.

`pllm.search.optimization_space` generates a bounded ordinary `SearchSpace`
without changing model, numeric, verifier or topology identity. Client-owned
body weights and new cache reuse require explicit permission. The explicit
`pllm.compiler.plan_on_load` facade resolves a source once, checks its locked
configuration, lowers the semantic plan and invokes existing placement search.
It returns the same `PlanningResult` used by SDK, gateway and benchmark, without
loading tensor values, timing candidates or reserving roles. Objective ties
prefer fewer changes to an admitted incumbent; unknown required costs reject.
Cache capacity is charged on a miss as well as a hit. Per-candidate graph indexes
and admitted schedules are reused, while native legality is independently checked.
Persistent measured GPU/compression tuning remains separate from this bounded
geometry/artifact/state cost selection.

Canonical benchmark reports and `pllm.metrics.communication_per_token` normalize
covered application bodies by authoritative generated outputs in decimal MB.
They distinguish online, measured-run covered, setup-inclusive and cold-first
rates. Decode-only uses transport counters sampled inside execution after the
first output, an `N-1` output denominator, and phase-counter conservation.
Unknown measurements or zero denominators remain null; startup and warmups are
charged once rather than averaged into per-run ratios. No full-wire claim follows.

Network-aware planning reuses the existing Experiment, compiler, role adapters,
SDK, gateway and benchmark driver. `pllm.deployment` owns immutable network,
party, offer and snapshot records plus discovery and opaque execution leases;
`pllm.search` owns bounded placement policy and candidate generation;
`pllm.compiler.plan` invokes native role/resource validation and returns an
immutable `PlanningResult`. Its costs retain their estimate/measurement scope
and unknown fields. Existing local Experiment v2 documents preserve their
identities; explicit network deployment uses v3. Authenticated HTTP offers bind
instance epochs and source commitments. Reservation, arm, execution and release
use fresh bounded leases, with expiry, drain and cancellation enforcement.
The live host slice implements CPU two-offset and prepared Inference/Preparation
roles; snapshot/local planning also covers admitted client-only compositions. Registered operator
names and authenticated offers do not establish physical independence. The
ordinary gateway and benchmark can consume selected plans or planning requests;
two controlled loopback network changes selected different hosts with zero
measured latency regret against their matched controls. This is bounded search
over supported candidates, not a universal optimal scheduler.

`python/pllm` is the only installed namespace. Its public API consists of the
client classes, configuration, response types, semantic model planning,
application factories and native matrix interface. `pllm.lower_model` accepts
only model configuration plus workload bounds and returns an immutable
`ModelPlan`; it does not resolve or load weights, tokenizers, devices or runtime
state. `pllm.Model` is the shared source specification, while `pllm.load_model`
performs the separate resolver/import step and records actual source-file hashes in
a path-independent checkpoint lock. Generic immutable configuration stays in
`pllm.configuration`; concrete implementation classes live in capability families
(`protocols`, `preparation`, `correlation`, `quantization`, `kernels`, `roles`, `nonlinear`,
`schedulers`, `state`, `passes`, and `verification`), and `pllm.components`
derives descriptor discovery from those classes. `pllm.profiles` provides typed
slot contracts for the two-service-role public baseline and shipped one-provider
proprietary engines while generic serialized pipelines remain available. An
optional `pllm.roles.PreparedProviderRoles` topology component explicitly binds
the existing client, offline Preparation, and online Inference graph to a new
composition and execution digest; omitting it preserves old v2 digests. The
native scheduler and live provider admission accept only this implemented graph,
and unimplemented role topologies fail closed. `Experiment.resolve().role_graph`
reports its offline/online channels and operator-separation requirement, but
local inspection cannot establish non-collusion. `pllm.providers`
discovers static external manifests and package-confined resources without importing
provider code; factory import is a separate approved operation. `pllm.metrics` owns
typed metric semantics; `BenchmarkResult` and `EvidenceRegistry` preserve exact
evidence cohorts without implicit ranking. `pllm.search` generates validated
immutable candidates and applies explicit cohort-safe Pareto directions.

The bounded local shared-gate research references issue one-use party-local
material for a two-input table, a correlated-hidden table, or a quadratic
Q7 SiLU×up **numerator** in a 24-bit ring. The quadratic dealer shares five
shifted coefficients per gated output and two independent shares of one
hidden mask; each party exchanges only its own masked hidden share and retains
an opaque output share. A semantic Qwen2.5-shaped 39+1-token resource gate
projects 5.03 MB of online bodies for MLP gated products alone and 70.81 MB
of one-use material per party. Charging one independent attention-query
hidden-source opening raises the optimistic projection to 10.06 MB, above
the 6 MB tenfold target before attention arithmetic or protected rescaling.
The 24-bit source range, Qwen numeric parity, protected output truncation,
whole-layer schedule, distributed dealer and independent role transport are
unimplemented. These test-local references do not activate an Experiment.

The installed `pllm.correlation.SecretReplicationPcf` separately benchmarks
two-party line correlations and punctured GGM keys through a one-use full-field
scalar/vector consumer. Fields 2/17/257, 1–256 SDK lanes and a native 16 MiB
combined allocation estimate bound the local dealer and party states. Eighteen
fresh processes check 5,778 exact outputs. At field 257 and 64 lanes, combined
raw keys fall from 16,424 to 328 bytes, while expanded correlations remain 1,032
bytes and peer frames 676 bytes. Reports separate dealer issuance, worker
expansion, online exchange and client-boundary estimates. Full-field expansion
still performs 65,664 combined field-output PRFs; near-32-bit symbolic counts
exceed one trillion and are not executed. Exact restricted-domain and mask-reuse
counterexamples prevent treating this as ring32 elementwise triples or FSS-key
compression. Distributed setup, reviewed transport, measured peak memory and a
compiler-bound decoder consumer remain absent. Details:
`docs/research/secret-replication-reference.md`.

The separate `pllm.metrics.ProjectedPolynomialCostProbe` combines mask-derived
linear coefficients, offline projection of constant coefficients and seeded
one-party correlation shares for a modular quadratic-gated MLP numerator.
Rust `pllm-garble` owns issuance, AES counter expansion, codecs, coefficient
arithmetic and non-cloneable one-use party state; public projections reuse
`pllm-core` wrap32/wrap64 SIMD kernels. PyO3 releases the interpreter lock for
execution. Python owns immutable probe configuration, reports and an independent
integer oracle. Four native ablations preserve modular outputs at 24/32/64 bits;
the checked 8-row 32→128→32 case reduces 24-bit material from 32,338 to 7,090
bytes while peer openings remain 1,618 bytes. This is numerator-only local
research, with no intermediate rounding, full-model numeric admission,
independent cryptographic review or full-response performance claim.
An independent pinned float32 eight-prompt gate rejects the fixed Taylor
polynomial at 0/8 prefill and decode selections; more flexible public per-channel
quadratic fits reach only 6/8 and 5/8. Quartic reference-path residuals are sparse
on average but have no private correction or public capacity certificate. The
polynomial-only decoder construction remains closed.
`TokenNetworkBudgetProbe` separately derives per-generated-token, executed-row,
source-opening and material budgets from compiler dimensions. Its 100-fold
39+32 body target leaves 1.135 MB online and 1.790 MB all-link; the checked
24-bit resident layout could afford at most three full-width openings per row
if everything else were free. `PublicPolynomialShiftRegression` executes a
native leakage witness against exposed complete shifted coefficients, showing
why opaque coefficient shares cannot be replaced by a public masked program.
These are hypothesis tests, not admitted privacy or decoder protocols.

The bounded in-process `TwoOnlineOffsetReference` runs two separately loaded
native stage kernels under one compiled Qwen2/Qwen3 decoder plan. The client
makes fresh exact-ring additive shares for each linear stage, validates both
committed responses and reconstructs before local nonlinear work. Tiny
prefill/decode matches the existing masked-stage numeric path. It records both
logical client/worker serialized stage bodies, integer matrix operations and
provider-stage times. Both workers still share one process; these figures omit
HTTP/TLS, cross-host traffic, setup and aggregate client compute. This reference
is not itself the executable Experiment option or independent-party privacy evidence.
`TwoOnlineOffsetTransport` can additionally run the same compiled decoder on
two separately hosted loopback HTTP worker processes. Each worker re-lowers the
plan, verifies body, stage and graph commitments, requires a distinct bearer
credential, binds session and stage IDs, and burns replayed or cancelled
sessions. Three Qwen2 and three Qwen3 generated tiny-checkpoint runs each
matched client-only logits and selections after two-token prefill and one decode.
Each two-worker response used 8 remote stages, 55,296 body integer MACs,
17,272 packed stage-body bytes and 19,398 HTTP application-body bytes including
admission/completion. Co-located children do not establish non-collusion.
Authenticated per-worker process CPU samples cover the online run window; each
worker's lifetime peak RSS is reported separately. Headers/TLS, cold model
distribution, startup CPU, client peak memory, full wire bytes and a matched
prepared-role cold/full-response CPU cohort remain unmeasured. A second
three-run per-family cohort uses the *same W8A8 quantized body fingerprint*
and 21 client-rendered input tokens plus two generated tokens for client-only,
two-worker and prepared roles. Client-only performs 202,752 body integer MACs
and transmits no online provider bytes; two workers perform 405,504 body MACs
and exchange 67,442 HTTP application-body bytes per response, against 31,574
recorded online body bytes for the prepared path. Client-only and two-worker
logits and selections agree; the prepared path's generated selections were
not compared. The initial prepared inventory uses 72,842 (Qwen2) or 72,664
(Qwen3) additional covered body bytes, plus a 51,581-byte idle refill over
three responses. Thus the three-run covered totals are 219,145/218,967 bytes
for prepared versus 202,326 bytes for two workers. Prepared startup body
counts are now separate from online run counters; neither report measures
full wire traffic, both-worker checkpoint distribution or cold aggregate
compute. Both the local kernel and the child workers pin W8A8 explicitly:
`MaskedTransformerEngine()` alone defaults to W4A4, which has a different body
fingerprint and cannot enter this comparison. The benchmarkable
`TwoOnlineOffsetCpu` composition now selects the two-worker protocol through
the same native compiled schedule, topology role supervisor, SDK client,
gateway and `benchmark run` path as the prepared and client-only compositions.
Each worker re-lowers the selected composition and the client binds its
token-boundary bundle before opening two authenticated one-use sessions.
Run-window application bodies and authenticated CPU samples appear in the
ordinary topology ledger. Co-located workers remain ineligible for an
independent-provider privacy claim; full-response compute-cap validation
remains open.
`pllm.metrics.LatentResponseCostProbe` additionally binds non-streaming research
costs to the semantic decoder plan and checks that its live token-selection and
feedback steps remain client-local. Matched 39+8 and 39+32 pinned Qwen2.5 W8A8
prepared controls use 73.83/113.55 MB covered online bodies and 116.84/178.97 MB
covered warm all-link bodies. A hypothetical one-round protected MLP cut has
6.92/10.54 MB of narrow online input/output bodies but 167.03/254.18 MB of
one-use keys for both parties. A resident two-source opening already needs
11.87/18.06 MB online before attention. The existing small-field selection
reference would require far more peer openings at full vocabulary and reveals
the selected index to the client; it cannot implement private token feedback.
These are explicitly non-executable, partial-body cost gates, not deployed
privacy, quality, or complete-wire evidence.
`pllm.metrics.EncryptedQuadraticShareCostProbe` additionally executes a
bounded, in-process one-use BFV quadratic over a 32-value additive-share
bottleneck: one party encrypts its own share; the other adds its share,
evaluates the exact `x²+3x+7` field polynomial, subtracts a fresh uniform
mask, and returns ciphertext for the first party's opaque output share.
Its public evaluation context excludes the secret key. One 39-row island
uses about 177 kB of two-way serialized bodies and a 411 kB context; eight
and 32 sequential one-row islands use about 1.42 and 5.68 MB before decoder
work, exceeding the matched Qwen 100-fold covered-body targets. Neither
Qwen nonlinear fidelity, compressed correlation issuance, independent roles,
malicious behavior, nor a whole-response execution path is established.
An independent rank-interface gate hooks the pinned Qwen2.5 float32 MLP outputs
after full nonlinear evaluation, fits bounded rank-16/32 bases on public
calibration text, and compares twelve held-out prefill/decode trajectories. All
24 rank-32 projections match 3/12 prefill and 1/12 same-token decode selections;
even six projections match only 6/12 and 8/12. No retrospective margin witness
passes; observed errors are not an a priori certificate. This optimistic
oracle rules out claiming that Qwen can simply be retrofitted with such cuts;
it does not test a jointly trained architecture.
A second optimistic gate fits a public per-layer affine map and projects only
its nonlinear residual. With six rank-32 cuts it matches 8/12 prefill and
7/12 same-token decode selections, versus 6/12 and 8/12 for projecting the
complete MLP output. All 24 affine-residual cuts match 3/12 and 0/12. The
full original MLP is still evaluated first, and duplicating just its public
float32 bypass maps across two workers would distribute another 154.14 MB.
This narrow mixed result does not enable a share-executable Qwen decoder.
An additional source-preserving extraction calibrates a public affine
linearization for the original bias-free gated MLP and keeps selected original
SiLU×up channels exact. This local float32 implementation needs no gradient
training; selecting all 4,864 channels reconstructs checked original MLP
outputs within 0.000142. With only layers 0, 12 and 23 extracted, selecting
128 per layer matches 5/12 pinned Qwen prefill and 9/12 same-token decode
selections. Selecting 2,048 matches 11/12 each, but projecting that width
over all 24 layers costs 165.49 MB of two-worker cut bodies. For the pinned
39+32-token workload, the 10× covered-body target is 17.90 MB and the 10×
online target 11.35 MB; 128 channels per layer project 10.66 MB online before
input, head, attention, offline distribution or complete wire. The optimistic
byte interface and the incomplete numeric fidelity cannot authorize a live
source-preserving topology. Public calibration does not use private prompts.
An optional closed-form per-channel public affine calibration keeps the
pretrained weights and online cut width fixed. For one final-layer replacement,
omitting all channels improves checked prefill agreement from 9/12 with the
mean Taylor expansion to 12/12, but decode remains 9/12. For layers 0, 12 and
23 at 128 exact channels, public least-squares calibration still matches only
5/12 prefill and 9/12 decode, whether or not one public decode row per prompt
is included. Fitting only 40 decision/decode rows overfits: 1/12 prefill and
0/12 decode. These exploratory calibrations do not establish whole-decoder
quality or a 10× all-link improvement. One full-width layer alone projects
16.36 MB of client/worker cut bodies against the 17.90 MB 10× covered-body
target, leaving almost no room for the other layers. A new method needs fresh
held-out data.
The separate `pllm.runtime.share_linear_feedback_reference` is a bounded
in-process two-share field-state toy. Each worker holds only its additive state
share and applies public affine maps; the trusted client reconstructs a
rank-width cut, calculates a quadratic, freshly shares its result and selects
the next code. Tiny prefill/decode matches an independent clear recurrence,
and typed frames bind one session, worker, row and layer and burn malformed or
replayed sessions. Its 39+32-token **hypothetical** 24-layer/896-wide/64-code
projection uses 0.80/1.24 MB of header-inclusive bodies at rank 16/32 with
four-byte words, but that numeric backend, trained model and output code do
not exist. The executable toy uses a different 16-bit field and 8-code head;
it has no Qwen fixed-point or generation-quality parity, independent operator
transport, full-wire/cold distribution, or compute-cap claim. Sequential
client cuts remain a latency gate: 1,680 cuts in the direct schedule, versus
an unimplemented 806-round lower bound if causal prefill were wavefront-batched.
The ordinary four-topology benchmark now separately samples each local role's
cumulative CPU from process birth and client/dashboard CPU from benchmark startup
through the first cold response. A pinned Qwen2.5-0.5B W8A8 30+1-token cohort
measured 11.71 s client-only, 34.66 s prepared, 28.65 s two-worker offset and
142.83 s Freivalds-verified, of which 109.87 s belonged to Preparation. The
verified path misses the measured CPU comparator by 4.98x on this one response.
This supplements, rather than rewrites, historical run-window cohorts; incomplete
full-wire metering, cold checkpoint distribution, representative measurements and
operator independence still prevent full policy admission.

The same bounded diagnostic uses client-only clear W8A8 execution as its
single-party comparator: one body matrix operation per stage, no online
provider bytes, and explicitly counted local checkpoint artifacts and quantized
weight snapshots. Cold checkpoint transfer and peak memory remain unmeasured;
the role graph has one client and no channels. The same placement is now an
explicit `ClientOnlyCpu` Experiment: native semantic scheduling binds its
linear stages to client-owned integer kernels, loads the checkpoint locally,
and executes through the shared SDK, gateway and benchmark driver without
inference/preparation children. The benchmark's zero provider-body count
excludes model download and application HTTP; full-response resource comparison
and representative model quality remain open.

`pllm.deployment.RoleDeployment` separately validates a digest-bound, immutable
role-to-operator and HTTP(S) origin declaration; optional TEE policy pins the
technology, code measurement, verifier root and TCB policy. The CLI can inspect
it against the installed or research graph, including co-located endpoints and
declared non-collusion, but neither an operator label nor a digest verifies
physical independence or a hardware quote. The document is never substituted
for the live Experiment deployment, and TEE execution remains unavailable.

`pllm.research` keeps attribution, quarantined upstream artifacts, clean-room methods,
and promotion gates static and non-executable. Public objects are imported on demand.
The docs research scorecard validates canonical matched cohorts from
`docs/data/research/benchmark-cohorts.json` and binds scored Python factories plus
shared configuration sources by SHA-256. It ranks each metric only within one
model, source, numeric, workload, lifecycle and output identity; cumulative client
RSS and other unknown measurements remain unranked. Paper implementation coverage
is independently recorded in `docs/data/research/implementations.json`. The pinned
Qwen2.5 150+8-token sample reduces covered bodies from 603.45 to 476.95 MB with
the exact prepared stack, or 266.91 MB when reusing 96 explicitly public prefix
tokens. Baseline request latency remains lower. Indexed tokenizer and capsule
artifacts add 9.29 or 12.25 MB separately; their distribution is unmeasured.
An independent Curl-inspired Rust reference implements bounded one-use shared
lookup and public Haar compression from the published equations. Each online
party owns one share; the local research dealer supplies the correlated one-hot
vector. Four Haar levels reduce dealer-share payload from 4,100 to 260 bytes per
lookup, with unchanged 136-byte header-inclusive peer openings and up to 0.12888
encoded SiLU error. Input is already truncated. This reference has no protected
truncation, distributed dealer, reviewed transport or executable decoder binding.
`pllm.assurance.PublicSubspaceMaskRegression` independently constructs a bounded
mod-2 leakage witness for public mask bases over exact u16/u24/u32 rings. It
follows the Carnival construction and Maverick Appendix E attack but is neither
an inference component nor a privacy proof when no witness is found.
`pllm.runtime` holds the separate
runtime model graph, HE preparation, transport, protocol, scheduling and
importers. Applications should not depend on internal module locations.

`python -m pllm` and the installed `pllm` command use the same entry point.
A source checkout does not shadow an installed wheel through a second package
at the repository root.

`pllm.runtime.build_roles` constructs the selected loopback role topology used by
`gateway --local`, the development dashboard, and the benchmark driver. The
client-only graph keeps model weights and computation in the client without child
providers; the two-worker offset graph starts one process per worker; the
prepared graph starts Inference and Preparation as separate
child processes, keeps generated credentials
in environment variables rather than argv or status records, binds health-checked
URLs to client/gateway factories, and owns process-group shutdown. The benchmark
native supervisor signals all provider groups before waiting, with one five-second
grace period and one two-second forced-retirement budget for the graph. Concurrent
pressure, startup-rollback and foreground cleanup share one owner. Repeated
SIGINT/SIGTERM during main-thread cleanup are deferred until retirement, then
delivered through the original handler; benchmark sampling errors still retire
the dashboard and provider roles.
The benchmark
dashboard driver now runs in-process, so a diagnostic run has two role children
rather than a dashboard child that creates another process tree. This topology is
for local development and measurement; it does not establish non-colluding
operators or production deployment.
`pllm gateway --local --experiment TARGET` resolves the same typed `Experiment`
documents and trusted Python targets as the benchmark CLI, then passes the complete
experiment to `build_roles`. Every shipped runtime-backed profile therefore uses
the same gateway command and fails through profile resolution rather than a
gateway-specific profile branch.
Public masked-linear experiments may select `pllm.quantization.SymmetricPerRow`
in their immutable Pipeline for W4/W8 weights and A4/A8 activations. The numeric
choice binds local-role preparation and inference settings as part of the
composition identity. Without it, existing Pipeline digests and the local-role
W8A8 default are preserved. A conflicting out-of-band bit override fails
closed; neither selection establishes real-model generation quality.

## Hot operations

Live network execution now admits CPU prepared Inference/Preparation roles as
well as two-offset workers through the same authenticated reserve/arm/release
lifecycle. Each prepared lease owns an isolated runtime app and inventories;
client credentials and selected-peer push credentials are distinct. Source,
composition, epoch, capacity and workload bounds are checked before material.
Release, expiry and cancellation burn the attempt's state. Declared operator
separation remains a placement statement, not evidence of physical independence.

The placement planner can bind `pllm.search.ArtifactCostEvidence` to actual
public bundle manifests and declared resident object keys. A bounded
`PlanningPolicy.reuse_horizon` prices missing public objects once and manifests,
fresh preparation and online arithmetic bodies per workload. Unknown required
costs still reject candidates; residency declarations remain estimates and
execution revalidates cache contents.

Client-local `ClientStateCostEvidence` can separately price a completed-prefill
reuse assumption under matching configuration, plan, source and client ownership.
It contains no tokens, cache keys or transferable KV; canonical numeric cache
admission remains required. Memory capacity still prices a miss and full fallback.
Configuration switching can charge one transition body cost and a primary-objective
improvement margin, but only a feasible incumbent can be retained. These are
estimates, not live-state migration or measured whole-response compute.

`SymmetricPerRow(causal_reduction="prefix_f32")` is a separately digested
float32 full-causal numeric option. Scores, softmax and weighted values reduce
only over each query's valid prefix. Pinned Qwen partition and teacher-forced
decode checks match every logit and KV value. This permits growing-width
completed-prefill reuse; default numeric mode retains the same-width gate and
generated snapshots remain explicitly response-owned by default. The optional
`ClientPrefixReuse(generated_prefixes=True)` additionally requires the native
full-KV `prefix_f32` continuation contract. It qualifies completed, already
executed generated tokens under a distinct canonical-incremental basis, excluding
pending sampled tokens, cancellation and `store=False`. Twenty-four pinned
Qwen checkpoints match fresh logits and every KV value. A separate three-request
conversation benchmark reduces covered setup-inclusive bodies from 402.05 to
366.63 MB over the lean prepared/reuse/compressed-artifact control. The common
241.72 MB first response already exceeds that cohort's tenfold target.

`CompiledRuntimeModel.transfer_snapshot` separately checks numeric-state
equivalence across admitted placement and delivery choices, preserves verifier
strength and lineage, validates native full-KV bounds and returns independent
client-local state. Response-owned snapshots are rejected. Automatic planner
cache migration and matched live switching costs remain separate. Its seal is
trusted-client provenance, not authentication of caller-mutated tensor contents.

Additional bounded SDK probes keep fresh-traffic candidates cost-gated. The
native masked-output relay halves client downloads but replaces them with peer
traffic and increases client reconstruction CPU about 5.3-fold. A token-local
projection memo preserves checked logits/KV and reduces its selected-kernel CPU,
but projects only 0.41% arithmetic-body savings while adding 2.06 MB of client
weights/snapshots. The progressive signed-i8 head oracle certifies all 16 checked
greedy queries at each tested precision, but retains the full head; six-bit
refinement projects 40.8 worker CPU seconds using measured one-row PIR, without
a public padding capacity. These three probes do not activate runtime choices.

`pllm benchmark run --docker` uses the existing role supervisor and benchmark
driver with a minimal public CPU runtime image and one Linux container per
provider role. Model/checkpoint and selected shared-Hub blobs mount read-only;
credentials enter environment variables. Reports retain exact composition and
body identities, cgroup CPU/memory and per-interface counters. The client stays
on the host. Docker-local roles are co-located; interface samples include control
and telemetry and do not establish complete all-link wire accounting.

Local role benchmarks now preflight whole-topology allocation estimates before
model-value loading or provider launch. Retained weight owners, loading and
bundle copies, masks/corrections, state/cache and Metal snapshots share a host
budget with an OS/application reserve; swap is excluded. Docker admission also
prices the existing complete Desktop VM and other containers, never enlarges it,
and gives providers hard no-additional-swap limits. A host-pressure monitor aborts
owned work on reserve loss or new swap growth. Estimates are conservative, not
measured peaks or an OOM proof; unpriced runtime/crypto paths reject.
`--backend native` enables existing local CPU/Metal processes; `--backend auto`
can fall back from unsafe Docker CPU placement while preserving the Pipeline.
Enforced WAN/link shaping cannot fall back to an unthrottled measurement.
`--preflight-only` and `pllm.metrics.benchmark_memory` expose the same admission.
The attempted Qwen3-4B cohort exhausted a 32 GiB host after Docker was raised to
20 GiB; no successful result was produced. Single-snapshot stage ownership,
bounded float loading, segmented bundle delivery and role-specific weight retention
reduced its then-current lean native allocation estimate from 34.78 to 19.75 GiB,
above available headroom. That historical figure included 5.98 GiB for Inference,
5.52 GiB for Preparation and 8.25 GiB for the client, with a 25% margin in each,
including conservative bundle-copy, 71-row mask and decoder-workspace bounds;
the original 7.49 GiB BF16 checkpoint is not a whole-topology memory estimate.
Docker retains the legacy allocation
upper bounds for potentially older images. A tiny native
CPU/Metal control passes functionality and cleanup, with equal outputs and zero
observed swap growth; it does not establish large-model capacity.

Client admission now prices live allocation roots, view aliases, numeric scratch
and geometric full-KV/snapshot overlap; unknown layouts retain the unreleased
output bound. The runtime already retired last-use graph entries, so this is estimate
correction rather than a memory optimization. Lazy masks and paged artifacts
plus that pricing put the bounded Qwen3-4B client estimate at 1.07 GiB, above the
initial 0.83 GiB target. The benchmark retires completed response-owned state while
keeping its separately qualified prefix cache. Ordinary reports expose current
and process-lifetime client/dashboard RSS; candidate peaks require fresh processes.
A matched Qwen2.5 37+8 complete-response cohort measures 665.49 to 306.43 MB peak
RSS (2.17x) for eager/resident versus lazy/paged allocation, with matching outputs
and zero observed swap growth. A separate tokenizer ablation measures another
5.53% reduction, 308.0 to 291.0 MB. The SDK shares its request-owned tokenizer and
encoded input; token-ID execution does not load a decoder vocabulary, while direct
string helpers construct one lazily. It adds no persistent tokenizer cache. These
different cohorts are not multiplied. The separate 9.44x Qwen3-4B storage/import
result excludes body/KV work. A subsequent native Qwen3-4B paged-client 16+8-token
capped response completes with all runtime/audit checks passing, 421.31 MB
client/dashboard process-lifetime peak RSS and zero observed new swap. Separate
Preparation and Inference peaks are 4.03 and 4.50 GB. This is one cached-checkpoint
functionality/resource sample without a matched resident 4B control or a
generation-quality comparison; 10x whole-client memory remains unestablished.
A later matched 16+8-token, fresh-process tokenizer-lifetime probe lowers
client/dashboard peak RSS from 390.32 to 350.45 MB (10.21%) with identical output
text and zero observed new swap. It reuses one tokenizer across benchmark input
admission, preparation sizing and response execution instead of constructing
three. Retaining that object through process exit increases end-of-run RSS from
337.54 to 350.19 MB. This probe-only allocation change adds no SDK cache. It did
not implement request-lifetime cleanup; a matched resident 4B control remains open.

The SDK now exposes bounded `tokenizer_scope()` ownership across request sizing,
preparation and execution, applied automatically by the benchmark. It retains at
most one admitted bundle's original tokenizer, replaces it on bundle changes,
and releases scope ownership on exit or failure; async worker calls inherit the
request context. Prompt-dependent work remains local. The explicit
`pllm.tokenization.IndexedTokenizer` now admits trusted offline byte-level BPE
indexes under source and artifact digests, bounded local queries and
request-owned private SQLite snapshots. Unsupported contracts reject. A fresh-client
Qwen3-4B ordinary SDK pair now measures the promoted tokenizer at 16+8 tokens
under native shared 100/40 caps: client peak RSS falls 360.89 to 262.73 MB (27.20%),
with identical outputs and covered application bodies and zero observed swap growth.
Request latency stays about 166 seconds; sampled aggregate RSS rises 5.57 to
5.82 GB. Both choices use paged client/Preparation weights, demand issuance and
packed/residue codecs. The 9.29 MB public index and 0.73-second compiler CPU are
additional; checkpoint distribution, physical wire, quality and a resident 4B
control remain unmeasured. See `docs/research/qwen3-sdk-artifacts.md`.
A subsequent matched Qwen3-4B 16+8-token pair isolates resident versus paged client
weights within that indexed-tokenizer stack, using fresh native clients and shared
100/40 caps. Client peak RSS falls from 1,020.46 to 261.05 MB (3.91x), while request
time rises from 142.665 to 155.460 seconds. Outputs and covered application bodies
match with zero observed new swap. Sampled simultaneous client/provider RSS falls
6.38 to 5.80 GB; aggregate cold-first CPU is nearly equal. The resident control
includes the shared native weight-view ownership correction, whose separate
synthetic probe halves retained matrix payload but leaves import peak unchanged.
This closes the matched resident-control measurement gap for one short workload,
without establishing 10x savings, representative quality, filesystem-cache or
whole-device memory savings. Evidence: `docs/research/qwen3-client-memory.md`.
The historical twelve-prompt FP32 quality attempt rejects before tensor loading:
16.09 GB of reference weights alone exceeds 13.84 GB of post-reserve headroom.
The ordinary quality diagnostic now optionally streams upstream eager FP32
parameters module by module under separate allocation and host-pressure admission.
Generated Qwen2/Qwen3/Llama prefill and cached-decode logits match the resident
upstream reference exactly; weights retire after every operator and failure.
A subsequent pinned Qwen3-4B twelve-prompt cohort matches 11/12 W8A8 prefill
selections and 35/36 same-token decode selections, with worst logit errors of
5.43/10.81. Peak active reference weights are 1.56 GB, not a process-RSS result;
all 19,152 parameter loads preserve the bound with zero observed swap growth.
The optional decode diagnostic freezes the first candidate's greedy trajectory
and retains separate phase scores and token-cohort commitments. This closes the
bounded held-out numeric measurement gap, not representative generation quality,
provider performance or cryptographic review. The original rejected cohort stays
separate; replay and evidence are in `docs/research/qwen3-client-memory.md`.
Compiled unscaled float32 RoPE now also shares exact coefficients
within a phase under a 16-table/1 MiB bound, priced by admission and cleared on
completion or failure. It preserves the original vector shapes and arithmetic
order. The tokenizer pair retains these shared SDK choices in both candidates;
it does not isolate their individual effects.

A separate five-candidate client-offload screen moves public BPE vocabulary and
merge compilation into a trusted offline artifact while keeping every
prompt-dependent lookup local. Its bounded SQLite reference cuts isolated
tokenizer-process peak RSS from 169.57 to 36.18 MB and matches 16 checked public
cases, but warm tokenization is slower and the derived artifact costs another
9.29 MB. The other four screens are cost-gated: immutable encrypted KV storage
adds 48.11 MB of projected bodies to move only 6.78 MB of short-response KV;
encrypted mask tapes cut sampled expansion CPU but add 329.04 MB all-link at
16+8 tokens; shallow HE attention and SiLU add large ciphertext bodies and fail
the existing float32 numeric identity. The encrypted screens are synthetic
in-process references, not runtime choices. Offloaded public compilation does
not authorize remote token-text queries, and mask tapes retain Preparation's
existing erasure and non-collusion assumptions.
A follow-up fresh-process, 16+8-token prepared Qwen3-4B pair reduces whole
client/dashboard peak RSS from 347.00 to 225.28 MB (35.08%) against the shared
original-tokenizer control. Source, Pipeline, numeric body, output text and
covered transport bodies match, with zero observed swap growth. The 9.29 MB
public index is pre-positioned; its distribution and separate compiler cost
are outside the ordinary response counters. This historical probe-only response-owned
adapter keeps private queries local and does not activate a Pipeline choice,
establish broad numeric parity or demonstrate a 10x whole-client improvement.

A second five-candidate client-offload screen uses pinned Qwen3-4B weights and
16+8-token geometry with synthetic private-value stand-ins. Public RoPE tables
preserve checked operator bits and save 1.84 ms of response-shaped rotary CPU,
at 25,093 additional public artifact bytes. Exact private square-root lookups
project 49.34 MB of bodies and 12.85 worker CPU hours; garbled activation absmax
projects 7.79 GB of online labels and 62.31 GB of one-use ciphertexts, before
remaining normalization/quantization work. Both layouts fail the cost gate.
Two-worker private tied-embedding/head queries preserve checked integer and
float32 bits and could move a 388.96 MB paged client weight artifact, but add
10.03 MB of projected online bodies and increase aggregate kernel compute;
client RSS and whole-response benefits remain unmeasured. Public norm-gamma
folding changes the W8A8 input codes, weights and outputs and cannot inherit
the original numeric identity. These isolated references retain non-collusion
or trusted-issuance assumptions and activate no Pipeline or provider protocol.

Bulk stage and bundle HTTP responses close their real transport and detach
HTTPX's response/stream reference cycle after consumption or failure. Recursive
artifact walkers likewise retire temporary readers and loaded-object owners.
These explicit lifetimes preserve returned bytes, bundle identities and response
metadata without relying on forced global garbage collection; GC-disabled
regressions cover success, failure and interrupted-stream cleanup.
A separate fresh-client Qwen3-4B 16+8 pair lowers paged-client peak RSS from
253.94 to 226.98 MB (10.61%), with matching outputs and covered application bodies
and nearly equal 166-second request time. Sampled simultaneous topology RSS rises
2.80%; the scoped run observes 524,288 bytes of host swap growth without tripping
the guard. This one-response allocation ablation does not establish zero swap,
whole-topology savings or held-out quality. Evidence and replay are in
`docs/research/qwen3-client-memory.md`.

Semantic linear-step temporaries now have per-step lifetimes, preventing
loop-local aliases from retaining consumed grouped gate/up tensors into the next
remote exchange. Qwen2/Qwen3/Llama prefill/decode ownership regressions fail on
the previous wheel and pass with scoped owners. An alternating four-client
Qwen2.5 150+8 cold/repeat ablation preserves outputs and application bodies, with
zero observed swap growth, but median client peak RSS remains effectively
unchanged at 230.59/231.01 MB. This deterministic ownership correction establishes
no whole-client memory reduction. A separate alternating four-client cohort
combines the correction with existing sixteen-row duplex prefill. Online TTFT
falls 18.28%/17.91% cold/repeat, client lifetime peak RSS 238.47 to 225.85 MB and
client cold-first CPU 4.19%. Decode medians regress 1.88%/3.73%; two-response
TCP-stream traffic rises 0.17%, with nearly unchanged aggregate cold-first CPU
and Preparation elapsed time. Outputs match with zero observed swap growth.
Qwen3 remains admission-blocked, and these scoped combined-stack results do not
establish a tenfold client-memory improvement. All seven priority metrics and
replay are in `docs/research/prepared-prefill-reference.md`.

Dense full-KV growth is capped at the compiled response's maximum key sequence,
so short responses no longer allocate the historical 64-row minimum. Restored
state grows under the receiving workload bound; overflow rejects before mutation.
Admission prices the same ceiling with its existing six-owner overlap allowance.
Generated Qwen2/Qwen3/Llama prefill, decode and restored-state controls preserve
every logit and active KV bit. Windowed and retained-state owners have separate
contracts. A fresh Qwen3-4B 16+8 pair reduces the KV-array payload from 18.87 to
6.78 MB but leaves client peak RSS nearly unchanged, 235.24 to 234.98 MB (0.11%);
end-of-run RSS rises. Outputs and covered bodies match with zero observed new
swap. This bounded allocation change does not establish a client-memory or
causal latency improvement. Details and matched resource evidence:
`docs/research/qwen3-client-memory.md`.

Paged artifact delivery additionally keeps the original tokenizer source in an
authenticated private file under a 64 MiB bound. Binding and indexed-tokenizer
verification stream its actual contents without retaining the serialized blob;
ordinary tokenizer libraries retain an explicit bytes boundary. Admission prices
transient and snapshot disk owners separately, and client shutdown closes the
snapshot. A corrected fresh-client Qwen3-4B 16+8 allocation pair lowers client
peak RSS from 231.29 to 207.70 MB (10.20%), preserving outputs and application
bodies with nearly equal 169-second requests and zero observed swap growth.
Sampled topology RSS rises 5.29% and aggregate cold-first CPU rises 1.30%.
Historical-control parsing owners retire before both responses; a preliminary
AST-contaminated pair is excluded. These scoped client savings do not establish
whole-device memory savings or representative quality.

A matrix is copied into Rust once at compilation, then reused for later calls.
`CompiledMatrix.weight_view()` exposes a read-only NumPy alias that retains that
immutable owner. Provider stage loading retires the original i8 allocation or
mapping; metadata, preparation and GPU import use the shared CPU snapshot.
Resident client bundles likewise rebind all stages sharing an imported matrix to
its immutable native view after first compilation, retiring the duplicate owner.
Tied token lookup/head and untied matrices preserve their original weight and
compiled binding identities. First-import overlap remains, and callers retaining
the original array still own its storage; retained-payload reduction alone is not
a peak-memory improvement. Details: `docs/research/qwen3-client-memory.md`.
Floating-point stage import and weight validation use bounded row chunks.
Fresh public role launchers retain checked stage metadata after retiring weights
unused by that role, without allocating a native snapshot for those stages.
Preparation retains only provider-executed stages. Inference and offset workers
retain delivery weights, but drop a tied main-token orientation already supplied
by the head. Auxiliary token tables keep their separate weights. Body and stage
commitments remain unchanged; retired stages reject execution. Direct all-stage
engines retain their existing ownership, and externally supplied engines are not
pruned by a role adapter.
Compiled disk-cache entries remain protected while their stages are loaded.
Direct callers retaining their original arrays still pay for those arrays;
client bundle, native execution and GPU snapshots retain separate accounting.
Input/output conversions and the initial native import still copy. An isolated
four-stage synthetic control reduces measured process peak RSS from 522.08 to
213.06 MB with identical weight hashes and integer outputs. This is loader/kernel
evidence, not a whole-decoder peak measurement or 4B capacity result.

Moving arithmetic to Rust does not remove dependent client-to-inference stage
exchanges, offline preparation work, or client attention state. No speed claim
follows merely from the choice of implementation language.

Hugging Face sources use the standard shared Hub snapshot cache for both online
and offline resolution, with immutable snapshot commits recorded in source locks.
Streamed integer stage weights use a separate regenerable compiled cache whose
default 6 GiB LRU limit evicts completed inactive entries after model load/unload.
An active checkpoint may temporarily exceed that limit; client-bundle cache
payloads remain separately disposable.

The legacy packed-BFV correlation backend remains available to research and
confidential-weight protocol code but is not selected by public inference.
Direct-FHE and blinded execution retain their separate transport.

Public-weight deployments use offline seeded inventory preparation. Before chat,
the client asks inference to create an inventory with immutable model, body, stage,
quantization, and attempt-budget commitments. It authorizes that inventory through
trusted preparation, then sends preparation one root seed and a batch size for each
remote stage. Batches default to 64 rows and are configurable with
`PLLM_PREPARED_INVENTORY_ROWS`. Domain-separated expansion binds each row to the
inventory, model, body, stage, weight, shape, quantization, ring, modulus, and wire
width and produces one-time input mask `r`, output mask `s`, and ticket.
The native client retains SHAKE256 cursors and a one-use row ledger rather than
all stages' expanded masks. It expands only claimed stage rows, handles disjoint
out-of-order leases through bounded XOF skipping, and burns cancelled ranges
without expansion. Domains and emitted mask/ticket bytes are unchanged. An
isolated Qwen3-4B-shaped 71-row mask probe reduces peak RSS from 628.16 to
89.60 MB and retained masks from 507.76 to 0.224 MB, with matching bytes and
zero observed swap growth. This measures mask handling rather than whole-client
memory; expansion CPU moves into row claims. Python retains lease orchestration;
Rust owns bounded expansion and the burn ledger under a GIL-released binding.
Preparation computes each batch of `W·r-s`, pushes it to inference's fixed endpoint,
and waits for inference's acceptance acknowledgement. After every stage is loaded, the
client asks inference to seal the inventory; only then does inference report `READY`.
The push WebSocket URL is derived from the validated inference HTTP(S) origin, and
only the provider push credential authenticates its upgrade.

At chat start, the client reserves inventory rows for the execution. Each online
stage message contains only the one-time ticket and `x-r`. Inference atomically
consumes the matching preloaded row and returns `W·x-s`; the client adds `s` and
center-decodes. Prefill batches this as one ticket vector and one packed matrix
per stage rather than one serialized envelope per row; decode retains one ticket
per stage over its persistent connection. Online chat is therefore
client-to-inference only. Preparation is
idle online, and refill occurs only while the client is idle between executions.
Inventory is in memory and may be reused across chats, but restart or idle expiry
discards it. Reservation is a burn boundary: cancellation, early end of stream, or
failure also burns every unused row reserved for that execution.
Stages use the smallest exact `u16`, `u24`, or `u32` ring selected from the exact
signed output bound. The preparation role is public-weight-only, holds the same
model body, and must follow the protocol, erase masks, and not collude with the
inference provider. Self-hosting keeps that trust inside the client boundary.
The online public path does not use BFV or contact preparation.

The optional `pllm.protocols.MaskedLinear(output_encoding="row_residues")` uses
public per-output weight-row bounds to encode both corrections and online outputs
in exact smaller power-of-two residues. Full-width input masks, one-use tickets,
numeric dequantization and the non-collusion contract are preserved. Native
admission, stage/weight-bound layouts and separate frame namespaces reject codec
mismatches; Freivalds checks restored integer outputs. Bounded compressed layouts
stay inline in public bundle manifests. Rust owns packing and mask reconstruction.
A pinned three-request Qwen W8A8 cohort reduces the lean prepared/reuse/artifact
stack from 507.46 to 490.41 MB setup-inclusive bodies, with identical outputs and
no client body weights added. Client cold CPU is nearly equal in the single pair;
full wire, peak memory and a full-response compute-cap result remain unknown.

`pllm.deployment.WanConditions` and `PartyAccess` expose immutable per-party
consumer access profiles, defaulting to 100 Mbps download and 40 Mbps upload.
`pllm.metrics.wan_readiness` and ordinary benchmark reports conserve directed
bodies, share access across a party's peers, retain intra-party traffic separately
and bind live role placements. They report analytic bandwidth floors and
bottlenecks separately from execution measurements. `benchmark run --wan` and
explicit rate/profile flags additionally enforce shared upload/download caps
through the existing Docker supervisor. Each party has an endpoint namespace
and a routed access namespace: WAN egress limits upload, private-LAN egress limits
download. This covers the client and all concurrent peers without requiring IFB.
Role grouping shares one access link; intra-party traffic stays local. Management
and telemetry bypass the rate queues, while routed service traffic includes
control, ACKs and retransmissions. Helpers have separate CPU/memory accounting;
startup, failure and cancellation retire owned namespaces and queues. Matching
kernel readbacks gate measured end-to-end, online and N−1 decode throughput.
`--wan-estimate` preserves analytical-only execution. A pinned 39+8 Qwen cohort
has identical outputs and decode rates of 2.203 uncapped, 2.081 at 100/40 Mbps and
0.775 at 20/8 Mbps. These are single-sample local rate-emulation measurements
with no added RTT, not measured Internet latency. The reference screens and
prepared codec are recorded in `docs/evidence/wan-methods-2026-10-04.md`;
the enforced cohort is in `docs/evidence/wan-emulation-2026-10-04.md`.

Explicit `benchmark run --backend native --wan` additionally enforces the same
shared per-party access capacities through bounded opaque TCP relays, supporting
native CPU and Metal without Docker. Startup, client/provider and provider/peer
service streams share upload/download budgets with a 16 KiB burst; intra-party
traffic is retained separately. Conserved counters include HTTP/WebSocket framing
but exclude kernel headers, ACKs, retransmissions, telemetry and checkpoint
distribution. The relay and sampler belong to measured client CPU/RSS. Kernel
shaping and native stream pacing retain distinct evidence scopes; automatic
fallback cannot silently substitute one for the other.
Native relay reads now use 8 KiB under the same 16 KiB shared burst, preserving
late timer refill without enlarging access allowances. Eighteen fresh TCP
processes check 251,658,240 payload bytes: a saturated 40 Mbps link rises from
about 31.91 to 40.03 Mbps, with increased relay CPU. This is emulator calibration;
historical rate-capped decoder cohorts retain their original pacing scope.
A matched Qwen2.5 150+8 cold/repeat pair preserves outputs and application bodies
with zero observed new swap. Calibrated request latency falls 17.45%/17.52%, but
client peak RSS rises 229.82 to 251.18 MB and decode throughput remains mixed.
The separate fresh-client public-artifact HTTP screen amortizes pre-encoding by
the second client; four clients reduce publication-inclusive provider CPU from
1.28 to 0.39 seconds while adding 10.15 MB of disk and barely changing client
CPU. It covers two public objects, not an SDK provider cache or complete bundle.
The separately selectable `ClientBundleTransport("artifacts", compression="zlib",
provider_cache_bytes=...)` now admits a 1 MiB–2 GiB app-owned encoded-file cache
on Inference and offset workers. Publication, staging and pinned readers share
the byte ceiling; source/object/codec commitments, verified reads, bounded LRU,
corruption repair and cancellation/shutdown retirement preserve original frames.
Local memory admission adds 48 MiB workspace per serving role and prices its
disk separately; network offers without disk pricing reject. First-use publication
remains provider work, exposed with hit/miss and storage metrics. Generated
SDK/gateway checks do not turn the historical two-object screen into a full-bundle
performance result. `examples/prepared_sdk.py` composes paging, demand/windowed
issuance, sixteen-row duplex prefill, exact packed/residue frames, prefix reuse and
offline indexed-tokenizer construction through public APIs.
The combined-stack allocation profile localizes Qwen2.5 growth to loading and
prefill, while the corresponding Qwen3 profile remains admission-blocked.
Evidence and replay: `docs/research/prepared-serving-reference.md`.
`--isolate-candidates` starts a fresh native client for each Experiment under
one private parent-owned cohort salt. Native memory reports additionally sample
the live client/provider RSS sum every 50 ms; they never sum independent lifetime
peaks. Samples can miss shorter peaks or double-count shared pages, and do not
price uncharged GPU allocations or OS cache. The idle launcher is outside the
measured role graph. Research's parallel-coordinates chart normalizes each
selected metric with best-at-top, preserves unknown gaps and raw units, identifies
client-only/naive-offset controls, and links paper attribution to chronology cards.
Its priority axes are client RSS/CPU, Preparation CPU/RSS, time to first token,
N−1 decode throughput, and all-link traffic, with their device, trust-control
and access-link rationale visible beside the chart. Missing Preparation roles
remain unknown rather than scoring as free work.
Generated research-data tables preserve every registered cohort and all exploratory
trials with digest-checked Python configurations, without ranking across cohorts.
A fresh-client 17-configuration Qwen2.5 150+8-token native 100/40 cohort matches
outputs with zero observed swap growth. Client-only takes 4.26 s, naive offset
174.48 s and prepared baseline 104.17 s per request. The public-prefix stack
reduces measured TCP-stream traffic from 604.30 to 267.76 MB, client lifetime
peak RSS from 605.00 to 222.71 MB and sampled total RSS from 2,272.41 to
1,173.62 MB, reusing 96 public tokens and excluding delivery of 12.25 MB of
public artifacts. Native bandwidth-limited results do not preserve the earlier
uncapped Metal latency lead. These are one-response local samples; full physical
wire, GPU compute, checkpoint distribution and Internet RTT remain outside scope.

The opt-in `research.verified_masked_linear_cpu` profile adds a trusted-client
Freivalds check to that prepared path. Preparation returns authenticated,
per-row projections over the client channel; Inference never receives the root
seed, challenges, or projections. The client verifies bounded integer stage
outputs before dequantization and burns material on use, cancellation, or
failure. This is a Slalom-derived engineering adaptation, not a TEE reproduction
or an actively malicious preparation guarantee. It does not reduce online
traffic.

Paper-specific Qwen2.5 factories now measure that verifier and public channel
equalization through the ordinary SDK. A native shared-100/40, fresh-client,
50+8-token Slalom pair preserves outputs; exact codec/delivery composition
reduces request latency from 140.95 to 130.69 s and cold TCP-stream traffic from
374.00 to 334.21 MB, while aggregate CPU increases from 76.61 to 78.95 s.
Only matching verifier contracts rank. Native memory admission now queries the
actual Freivalds check count and prices client/Preparation projection owners;
Inference has no verifier payload. Historical Docker upper bounds remain.
This corrects accounting rather than weakening verification or optimizing RAM.
Fixed SmoothQuant-style Qwen calibration uses eight public sequences and a
separate twelve-prompt prefill diagnostic: top-1 agreement falls from 11/12 to
10/12 despite a worst-logit-error improvement from 3.6207 to 2.5092. The public
profile is 729,527 bytes and has its own source and numeric identities. Provider
CLI startup preserves the original Hub repo/revision when resolving offline,
shares the selected cache directory and revalidates its pre-launch source lock
before loading stages. Replacing that source with a local-path spec would
invalidate the profile. These scoped adaptations are not full paper-system
reproductions or representative quality results.
Within the fixed equalized body, one native 100/40, 50+8-token pair preserves
outputs while exact codec/delivery composition reduces request time from 78.26
to 70.24 s, TCP-stream traffic from 289.41 to 251.25 MB and client peak RSS from
540.85 to 262.96 MB. Aggregate CPU increases from 34.47 to 37.28 s. Plain and
equalized bodies rank separately; profile publication and distribution remain
outside response counters. All paper controls and the held-out numeric report
are retained in the Research data with rerunnable, digest-checked Python files.

A subsequent frozen four-choice Qwen2.5 quality screen uses separate public
calibration and twelve fresh held-out prompts with three same-token decode steps.
W8A8 matches 10/12 prefill and 35/36 decode selections; equalization improves
prefill to 11/12 but regresses decode to 32/36. W4A8 scores 1/12 and 21/36;
W8A4 scores zero. Public profile compilation costs 21.78 CPU seconds and 729,527
bytes. No default changes or same-quality cost ranking follow. Qwen3 calibration
rejects its 9.51 GB allocation estimate against 8.12 GB headroom. Exact source,
numeric and cohort identities remain separate from earlier quality samples;
details are in `docs/research/prepared-precision-reference.md`.

The installed `pllm.protocols.PrivateEmbeddingLookup` separately benchmarks
FastQuery-inspired one-hot BFV coefficient packing and signed packed-share
conversion. Native bounded packing, uniform whole-polynomial masks and one-use
half-gates conversion compose with separate client/provider SEAL contexts. Nine
fresh processes check 1,215 exact integer outputs under a 128 MiB allocation
estimate. That historical control sends full ciphertext replies, and the local
conversion issuer sees both shares rather than executing distributed SiRNN.
The mixed 4/3/3 hypothetical Qwen 39+32 geometry needs 1.643 GB of HE frames and
293.35 MB of conversion keys, before boundary labels, setup and remaining
operators. A separate twelve-prompt public W8A8-source requantization screen
checks 1,007,104 elements per profile; mixed 4/3/3 matches 10/12 prefill, 34/36
same-token decode and 6/12 eight-token trajectories, without exact logits/KV.
The tied client head retains the same 136.13 MB integer matrix, so embedding-only
relocation releases zero weight bytes. These operator and clear numeric checks
do not establish paper fine-tuning, full-width HE execution, provider transport
or whole-client memory savings. Details: `docs/research/packed-embedding-reference.md`.
The same component now optionally consolidates outputs with native extracted-LWE
packing or BumbleBee-derived direct interleaving. Separate one-use party handles
convert privately held packed coefficients using locally issued random-OT
correlations, correcting carries and returning only independent ring32 shares.
Bounded ciphertext serialization, RNS extraction/embedding and complete
trace/packing preserve signed integer outputs. Both context/Galois-key copies,
setup delivery, conversion phases and extraction buffers precede 128 MiB admission.
Eighteen fresh processes check 3,456 outputs. At 4,096×64 mixed precision, online
HE/conversion frames fall 1,642,896 to 180,624 bytes while setup material and local
compute rise. The unadmitted Qwen 39+32 projection still needs 234.32 MB online,
409.73 MB conversion keys and 1.76 MB setup per context; refreshing 70 contexts
costs 123.33 MB setup. Full-width noise, distributed OT setup, SiRNN and decoder
transport remain absent. Tied-head ownership still releases zero weight bytes.

The installed `pllm.passes.CipherPruneTokenPruning` separately benchmarks exact
Q16 attention-score thresholds, public retained counts, stable complete-record
compaction and disclosed compacted-order degree decisions. Up to two fresh
one-use half-gates circuits adapt CipherPrune's OT primitives; a dedicated marker
preserves all payload bits. Empty/all-retained outputs skip compaction, and
all-retained outputs preserve their original shares without degree disclosure.
Geometry-only worst-case pricing checks every possible retained count against a
64 MiB combined allocation estimate before issuance. Fifteen fresh processes
check 381 exact outputs. Keeping 19/39 hypothetical Qwen-width token records
projects 631 MB evaluator material before client encodings and public metadata
to retire 143 kB of two-party record payload. All full-width shapes reject
admission. Public counts and degree decisions disclose information even when
original positions are not separately opened. Learned thresholds, selected
polynomials, protected float conversion, causal KV mapping, distributed input
encoding and decoder transport remain absent; checkpoint quality, physical wire
and measured memory benefits are unknown. Details:
`docs/research/token-pruning-reference.md`.

Public model bundles also carry the quantized token-lookup and output-head
matrices. The client evaluates token lookup locally and applies the output
head only to the final prefill row. This removes vocabulary-sized HE work from
the online public-weight path. Transformer-body matrices remain provider-owned;
proprietary bundles never include either boundary matrix.

Bundle schema 2 stores quantized matrices once and lets stages reference them.
Public provider HTTP delivery snapshots bundle metadata and streams immutable
native-backed binary segments with the same canonical MessagePack identity.
Artifact export shares these segments; artifact import verifies every object and
the canonical raw digest before entering the existing client validator, without
constructing a second contiguous raw bundle. Raw downloads enforce declared
lengths during receipt. The direct bytes API still materializes a bundle.
A bounded 64 MiB matrix serialization control reduces isolated process peak RSS
from 441.94 to 244.83 MB; it does not measure whole-decoder or client import peak.
An opt-in bundle transport sends independent bounded zlib frames, then checks
the original uncompressed bundle digest before client import; only the raw
bundle is cached. The pinned Qwen2.5 cold-pair diagnostic saves 21.08 MB of
covered bodies at the cost of higher CPU and latency, with no warm transfer
savings or measured disk reduction.
For tied embeddings, one vocabulary-by-hidden matrix uses per-token row scales;
token lookup gathers its rows and `lm_head` multiplies by the same array. This is
not numerically identical to schema 1 token lookup, which independently quantized
the transposed matrix with per-hidden-feature scales. Output-head quantization is
unchanged. Any per-layer token table remains a separate auxiliary object using
its existing orientation.

An opt-in, byte-bounded in-memory trusted-client cache can retain exact-token
prefill state and logits under a compiled decoder and bundle fingerprint. Only
complete identical prefills in ordinary prepared public execution are reused;
verified placement, recurrent/windowed/shared state, `store=False`, continuations
and changed token sequences do not activate reuse. Decode still executes normally
and all reserved rows remain one-use. A 39+1-token Qwen2.5 repeated-prompt
diagnostic reduced covered second-response bodies from 98,723,710 to 986,876
and online masked bodies from 62,248,704 to zero. This is a repeat-workload
optimization with visible access-pattern differences, not a general per-prompt
network reduction or an independently operated privacy proof.
`pllm.state.ClientPrefixReuse` additionally binds a fixed public compiler input
bound and a byte-bounded in-memory KV cache to a prepared baseline Experiment.
It replays only the uncached suffix, and rejects cache hits whose per-stage
row/body estimate would lose to batched prefill. Cache identity includes the
compiled plan, source bundle and complete token prefix; changed prompts cannot
reuse the wrong state. A pinned Qwen2.5 two-candidate loopback cohort reduced
covered online bodies from 279.29 MB to 11.58 MB for a 175-token prompt with a
168-token shared prefix and seven fresh suffix rows. This is an opt-in
repeat-workload result, not a saving for unrelated new prompts or full wire.
That historical cohort predates the stricter numeric state gate: pinned W8A8
prefill-only KV reused at a different attention-reduction extent differed by
1.8366 logits. Cache v2 keys therefore include original actual full-input width.
Only sealed completed-prefill state with matching width may enter ordinary
proper-prefix reuse. Immutable eight-row blocks share storage across qualified
checkpoints, while returned snapshots remain independent; eviction erases blocks
only after their last reference. The separately digested native continuation
schedule admits batched uncached suffixes and must receive an exact provider
acknowledgement before material reservation. Historical-mode generated KV differs
from fresh prefill and remains response-owned. Fresh-prefix promotion requires
the separately selected canonical generated-prefix contract described above.

`pllm.roles.OutputHeadAtInference` separately moves an untied public output
head into the prepared Inference stage schedule. It binds provider stage,
bundle, source and composition commitments before material reservation, leaves
the token table at the client, and rejects tied heads before checkpoint import.
One remote head application per generated token increases online and offline
traffic even as it removes the head matrix from cold client delivery. A
generated untied 512-wide/65,536-vocabulary 1/8/32-token cohort measured a
33.82 MB smaller client bundle and positive covered cold-byte savings across
all three lengths, but online-only bytes increased; no official-checkpoint
quality or long-output crossover has been established.
`pllm.roles.ClientPrefixLayers` is another optional prepared placement:
complete first semantic decoder layers execute on the trusted-client CPU, with
remaining layers admitted as masked remote stages. Their W4/W8 i8 weights and
scales are added to the client bundle under a 512 MiB extra-weight bound; the
provider still holds the public full checkpoint. One pinned Qwen2.5 40+1-token
loopback cohort with two client layers reduced covered online stage bodies from
63.84 to 58.52 MB but raised covered cold all-link bodies from 247.23 to
268.71 MB. The short-prompt cohort therefore does not make this a cold-transfer
or disk optimization; full-wire, independent operators and matched generation
quality remain separate gates.

`pllm.roles.ClientLinearRoles` selects grouped semantic projection roles and can
union them with complete prefixes using `prefix_layers`. Native scheduling binds
matching groups to trusted-client CPU or selected Metal kernels; checkpoint import,
live bundle admission and provider stage tables
share that ownership declaration. Added matrices/scales retain the 512 MiB bound
and both provider roles retain their public full checkpoint. Pinned Qwen2.5
attention ownership passes bit-identical W8A8 prefill/decode logits on three
prompts. A separate 39+32 warm response uses 148.30 MB covered bodies versus
178.97 MB baseline and 93.22 MB online versus 113.55 MB. It retains 87.69% of body
linear MACs remotely, adds 44.04 MB i8 weights, 0.20 MB scales and a 44.04 MB native
weight snapshot at the client; these storage counts are not peak-memory samples.
The raw cold bundle grows; representative quality, full wire and independent
deployment remain separate claims.

`pllm.preparation.PreparedInventory` and `pllm.protocols.ClientBundleTransport`
bind startup/row-floor and bundle-encoding choices to ordinary prepared Pipelines.
Omitting them preserves previous digests and defaults. Explicit conflicting SDK
or benchmark overrides fail closed. Request sizing changes offline inventory
work, not one-use reservations or ordinary online stage bodies; zlib negotiates
bounded frames but retains raw cache bytes. The ordinary benchmark accepts a
bounded ordered JSON context sequence on one live client, records only salted
cohort digests, and matches ordered counts/caps/warm states before summing costs.
Setup-through-first-response and whole-benchmark accounted-body totals include
recorded startup/warmups once, alongside historical per-run medians. On one fixed
113/130/141-input-token, eight-output-token-per-request Qwen cohort, attention plus
prefix reuse and zlib reduces online bodies from 647.60 to 240.68 MB and covered
setup-inclusive bodies from 1,171.96 to 547.41 MB. These co-located protocol-body
counts exclude wire framing and checkpoint distribution; compression increases
cold CPU in this cohort.
These growing-context measurements are historical, before the matching-width
cache gate, and do not establish current exact cross-width reuse savings.

`ClientBundleTransport("artifacts")` adds public immutable object locality to
compiled prepared bundles. Every import fetches an authenticated manifest, then
reuses or fetches bounded content-addressed objects under source, shape, dtype,
orientation and numeric commitments. It reconstructs and verifies the exact raw
bundle before ordinary native admission. The byte/count-bounded cache verifies
reads, repairs corruption and uses atomic writes; its default payload ceiling is
2 GiB. It contains no private KV, masks or one-use inventory. A cached pinned
Qwen SDK cohort fetched 173 cold objects (145.28 MB), then only 48 new objects
(44.04 MB) when switching to client attention. Returning to baseline fetched zero
objects but still paid for a fresh 195 kB manifest. Raw, zlib and artifact controls
have bit-identical checked W8A8 prefill/decode logits. Ordinary SDK admission and
the canonical benchmark path passed without temporary option whitelists.
Object storage, reconstructed NumPy payloads and native snapshots are distinct
ownership categories, not peak-memory measurements. The real-shaped shallow
encrypted-linear feasibility screen remains non-selectable: its sampled exact
4,864-to-896 projection costs roughly 40 times the prepared stage bodies.

Compressed content-addressed delivery composes with prepared, verified and
two-offset execution. `ClientBundleTransport("artifacts", compression="zlib")`
frames missing objects, validates encoding/raw-size acknowledgements and retains
the same verified raw cache identities. Offset clients require both authenticated
workers to admit the exact compiled continuation; cumulative stage-row budgets
and cancellation bind the complete attempt. Verified caches additionally seal
verifier strength and client-minted inventory incarnations. The explicit cache
composition issues material at requested failure bits plus 12 and admits at most
4,096 inventory transitions, preserving the requested union bound; targets above
68 fail against the backend's 80-bit ceiling. Every remote stage must be checked
before sealing state. Generated KV remains response-owned unless the explicit
canonical generated-prefix contract qualifies the completed executed prefix.

One co-located Qwen2.5 W8A8 39/39/45-input, eight-output-per-request cohort matched
all 15 outputs. Prepared reuse, on-demand issuance and compressed artifacts used
310.96 MB covered setup-inclusive bodies against 616.34 MB control, without new
client body weights. Adding an attention/first-layer placement union reduced
online bodies from 231.07 to 92.85 MB but raised the combined total to 323.94 MB
and added 114.47 MB of client weights/scales/native snapshots. The seeded/packed
offset control used 496.80 MB covered total; adding reuse and compressed artifacts
used 303.94 MB. These costs do not include full wire or checkpoint distribution;
client peak memory remains unknown. The real verified control timed out during
inventory startup at 180 seconds and contributes no successful cost result.
Bounded planning prices union ownership once and can bind schema-v2 artifact
transfer lengths separately from raw residency. It preserves unknown CPU/GPU
costs and cannot infer a benefit from compatibility alone.

Optional WAN throughput choices retain the same compiled numeric execution.
`TwoOnlineOffsetLinear(dispatch="seed_first")` requires seeded input and overlaps
the independent worker request with client share construction and the other
worker exchange. `ClientBundleTransport(batch_objects=64)` coalesces missing
public artifacts under a 1 MiB raw-group bound while checking every original
object digest. `PreparedInventory(stage_window=4)` overlaps independent stage
issuance under a default 16 MiB active-work bound. The explicit
`stage_window_bytes` option binds a 1–64 MiB window when concurrent stages are
selected; native compilation checks it and whole-topology admission still prices
every concurrent owner. It seals only after all pushes are acknowledged. Failure and cancellation burn
the attempt before workers are joined. These choices add no client body weights.
A matched fresh-client Qwen2.5 150+8-token cold/repeat sequence measures the explicit
four-stage/64 MiB window with paged client and Preparation weights. Cold preparation
falls 42.04 to 38.18 seconds; complete cold/repeat requests improve 2.67%/3.67%,
with unchanged covered bodies and 2.02% more cold-first aggregate CPU. Windowed
paging uses 281.85 MB client peak RSS versus 564.77 MB with resident client weights.
Explicit prefix reuse lowers the exact-repeat request 103.45 to 11.67 seconds and
the two-request TCP-stream total 830.67 to 495.07 MB, while its first response
still takes 139.11 seconds. All eight outputs match with zero observed swap growth.
These are one cold/repeat pair per configuration, with pre-positioned tokenizer
artifacts and no checkpoint-distribution, physical-wire or independent-operator
measurement. Details and replay: `docs/research/prepared-frontier.md`.
The explicit `MaskedLinear(prefill_chunk_rows=4)` splits prefill into disjoint
one-use row chunks on one duplex WebSocket. It bounds frame groups, does not
retry accepted work through HTTP, validates ordered session replies, and verifies
the complete integer stage before dequantization when Freivalds is selected.
Tiny full logits/KV and pinned functionality pass; its incremental measured TPS
gain remains inconclusive. Native public-byte rANS is exposed only through the
artifact entropy probe: sampled Qwen delivery grows versus zlib and costs more
client decode CPU. Neither choice is inferred beneficial by the planner.
The matched single-salt Qwen 39+8 WAN cohort uses shared 100/40 Mbps party caps
and 20 ms added egress delay per party. Prepared batching/windowing improves
request TPS 17.3%; seed-first offset improves decode TPS 1.94-fold with unchanged
application bodies. Outputs match; peak client memory, independent operators,
full physical wire and a tenfold network improvement remain unestablished.

`MaskedLinear(prefill_pruning="terminal")` uses native backward row demand from
logits and persistent state to prune only whole row-independent linear groups.
It preserves state roots and local tensor layouts; batched continuation retains
all rows. Default uniform inventory still issues and burns the full reservation,
so isolated arithmetic projections overstate its live offline savings.
`MaskedLinear(request_encoding="compact")` additionally uses a separately
negotiated single-row frame namespace with fresh nonce/ticket and authenticated
session/stage context. Both choices compose with paged/compressed artifacts,
row residues, prefix reuse and Freivalds verification without new client body
weights. Defaults preserve previous identities. A matched 39+8-token SDK cohort
reduces covered setup-inclusive bodies from 236.56 to 234.07 MB (1.05%) and
online bodies by 3.50%, with identical outputs and zero observed new swap.
Client CPU is slightly lower, aggregate CPU nearly equal and request latency
slightly higher; independent peak-memory and WAN speed benefits are unmeasured.
A distinct public-prefix capsule
oracle preserves checked Qwen logits/KV and reduces arithmetic plus capsule
delivery 17–21× for a fixed 256-token public prefix; adding cold client and two
provider checkpoint artifacts reduces the ratio to about 1.30×. Its portable
authenticated importer was subsequently implemented as `PublicPrefixCapsule`
described below. Exact integer anchors and output lifting fail the
client-cost/byte gate. Five separately scoped 100× constructions include a tiny
finite-task lookup, explicit basis-leakage and rounding counterexamples, and two
unimplemented architecture hypotheses. These are recorded in
`docs/research/aggregate-network.md`; no general 10× or 100× improvement follows.

Four incremental choices now compose through ordinary prepared SDK Experiments
and benchmarks: the indexed tokenizer, paged Preparation weights, compiler-demand
`PreparedInventory(allocation="demand", refill="on-demand")`, and
`PublicPrefixCapsule` under an exact source/numeric/full-KV `prefix_f32` contract.
Demand allocation binds per-stage counts to authorization and reserves each
inventory once; cancellation and early completion burn unused material. The
capsule imports only explicitly public, trusted-publisher state, never discovers
or exports a prefix from private prompts. Artifact commitments and local paths
bind the Pipeline, with local read bytes separated from network accounting.
A matched six-Experiment Qwen2.5 150+8-token cohort preserves output digests
with zero observed swap. Demand alone removes 447 unused correction rows;
paged Preparation lowers its child peak from 581.21 to 217.86 MB. Reusing 96
public prefix tokens with all four reduces covered bodies from 481.92 to
266.91 MB; pricing 12.25 MB of pre-positioned public artifacts once gives
279.17 MB. Publisher CPU is separately recorded. Single-sample local results,
sequential client lifetime peaks and covered bodies do not establish independent
client-memory, whole-wire or WAN improvements. Reproduction and full costs are
in `docs/research/incremental-sdk.md`.

`MaskedLinear(request_encoding="stage_packed", output_encoding="row_residues")`
further packs masked inputs to each stage's maximum public output-residue width.
Omitted high-bit contributions vanish in every output ring; original masks,
corrections and one-use inventory domains remain intact. Native composition
admission, source-bound layouts, explicit session acknowledgement and a bounded
distinct frame namespace reject forged widths and downgrades. The option composes
with pruning, paging, compression, prefix reuse, duplex execution and Freivalds
checks without client body weights. A matched 39+8 pair reduces covered bodies
from 234.07 to 229.98 MB (1.75%) and online bodies by 5.95%, with matching output
digests and zero observed new swap. Aggregate CPU is nearly equal and request
latency slightly higher; full wire and independent peak-memory benefits are
unmeasured. Odd-radix packing, block-moment bypasses and signed row dictionaries
fail the byte/client-cost gate; sparse shift-mask reuse exposes private relations.
Five separate whole-decoder 100× hypotheses have bounded entropy, label-linkage,
polynomial-cost, tensor-rank and state-equivalence vetoes. Their scope and
reproduction live in `docs/research/network-algebra.md`; none activates a new
protected decoder or establishes a general 10×/100× result.

A further first-principles screen tests five 10× and five 100× hypotheses in
`docs/research/network-first-principles.md`. Its two-syndrome output codec gives
a 16× symbol reduction on a publicly promised one-sparse toy domain, but no
checked Qwen output block qualifies and the syndrome cannot certify its domain.
Exact activation-span reuse has at most 0.124% arithmetic savings on the checked
trajectory. A fixed continued-fraction SiLU profile with explicit tails matches
12/12 checked W8A8 token selections, but no complete logits or KV state; worst
absolute logit error is about 2.39. Finite-field inversion is not IEEE division.
An exact finite-field observable lift runs 256 nonlinear toy transitions through
local additive-share linear updates. Its Qwen feature width, private encoding
cost and token feedback are unknown. A Hermite-jet counterexample exposes the
secret when one worker receives a degree-one mask's value and derivative.
Concrete ramp-sharing, branching-program, log-domain and encrypted bitheap
layouts retain explicit traffic/compute gates. These independently constructed
references activate no SDK option or protected decoder.

The installed `pllm.protocols.PackedEncryptedLinear` separately benchmarks bounded
1–32-dimensional public diagonal matrices on private CKKS vectors. Rust owns
public layouts and degree/depth/resource admission; separate SEAL client/provider
contexts execute the encrypted operations. Only the client holds the secret key
and decodes final results. Eighteen fresh processes check 1,224 outputs, with
maximum independent float64 error below `9.37e-7`. At 32×32, one sequential
rotation key reduces setup frames from 25.16 to 1.20 MB; four-query online frames
remain 1.606 MB. Reports separately charge key setup, public layouts, padded
context-bound query/reply frames, encrypted operation counts and remaining depth.
Generic admission checks up to eight finite levels under a 512 MiB allocation
estimate. Circuit privacy, malicious-result verification, physical transport,
checkpoint quality and autoregressive execution remain absent. NEXUS is an
all-CKKS system, not an HE–garbled-circuit boundary; its bootstrap prerequisite
is unimplemented in SEAL. The diagonal reference cites only its background
primitives. MOAI's inaccessible full source prevents a paper-specific
implementation claim. Details: `docs/research/encrypted-linear-reference.md`.

The installed `pllm.nonlinear.PackedEncryptedSoftmax` independently composes
NEXUS repeated-squaring or THOR polynomial/square-normalize primitives with
encrypted reciprocal and final normalization. Fixed public scores in [-1/8,1/8]
and power-of-two widths 1..32 bind separate numeric profiles; unrelaxed reciprocal
iterations are PLLM adaptations. Both schedules consume fourteen levels at
degree 32,768 under SEAL TC128 and a separate 2 GiB allocation estimate. Only the
client decrypts final outputs; independent Decimal profile and softmax oracles
separate CKKS error, approximation error and probability-mass error. Eighteen
fresh processes check 738 outputs with worst softmax error below `3.21e-5`.
At width 32, both methods charge 260.19 MB setup and 25.19 MB online for three
rows; local online medians are 3.63/8.56 seconds. Completed sample key owners
retire before the next issuance. Physical wire, measured peak memory, attention
QK/value products, masking, bootstrap, decoder state and private token feedback
remain absent. A separate fresh twelve-prompt pinned W8A8 Qwen audit rejects all
48 prefill/decode positions under the fixed score range; the unchanged clear
decoder executes without an encrypted candidate, clipping or private range fitting.
This narrow-domain operator establishes no checkpoint quality or paper-system
reproduction. Details: `docs/research/encrypted-softmax-reference.md`.

## Application boundary

The client owns plaintext input, inventory root seeds and masks, private
activation scales, model state, output decoding and public token-boundary matrices.
The optional `pllm gateway` process runs inside that client boundary and exposes
Responses API and Chat Completions API on loopback. Compatible applications send
ordinary API requests to the gateway; the gateway terminates those requests and
uses the same private PLLM client path underneath. Inference and preparation
services are not application-facing OpenAI-compatible endpoints.
The trusted preparation service and untrusted inference provider both hold the
public transformer body. Preparation receives batched stage root seeds before
chat; inference receives the resulting corrections and later the masked integer
tensors, never the seeds. The client never receives the correction. Client-to-inference,
client-to-preparation, and preparation-to-inference credentials are distinct.
The local Responses API gateway is inside the client boundary.
An ordinary provider endpoint cannot be made private by changing its URL in an
unmodified SDK.

The public protocol assumes both roles follow the computation and do not
collude. The arithmetic simulator, guarded controls and transport authentication do
not establish security against arbitrary malicious participants. These limits
do not change with the packaging layout.

## Repository boundaries

Fumadocs lives in `docs` and has its own npm manifest. It is deployed as a static
site and is not bundled into the Python wheel. `paper/manuscript.md` and
`paper/whitepaper.md` are the canonical Pandoc Markdown paper sources. Builders
live in `docs/scripts/` and run as part of the docs build. PDFs, source archives,
figure outputs, rendered publication MDX and download copies are generated and
ignored by Git; intermediate paper assets live in `docs/build/papers/`. Pandoc
generates both website articles and the technical PDF, using Tectonic locally or
pdfLaTeX in CI for the latter. The two-page whitepaper PDF uses a dedicated Pandoc HTML/CSS
print layout rendered by Chrome or Chromium; only technical-paper layout details
remain in TeX. The technical build also emits a standalone, citeproc-resolved TeX
archive for arXiv without a Pandoc build dependency. Historical measurements are kept
under `docs/evidence` and are not rewritten as native Rust results.

The loopback benchmark dashboard runs the real client, preparation, and inference
roles and receives child-role OTLP metrics and traces directly. Protocol byte counts
are custom OTEL metrics; prompts and activation payloads are never telemetry
attributes. The generated tiny checkpoint validates transport behavior only.
The benchmark report additionally groups existing client audit counters into a
directed, phase-labeled **serialized protocol-body** ledger for the prepared
graph, without adding overlapping upload/server counters twice. It reports
client and all-link accounted bodies plus available run-window CPU per role.
Bounded client-local and child-role OTLP stage counters now reconcile five
directed links against the audit before the compiler's semantic roles receive
attribution. A pinned Qwen2.5 W8A8 39+32-token response accounts for 178.97 MB
all-link: 148.30 MB in MLP stages and 30.67 MB elsewhere. Removing all MLP
stage bodies for free still cannot meet the 17.90 MB tenfold target. Two
same-body Qwen cold samples measure 41.03 s median aggregate CPU in the
prepared graph against 38.01 s for two offset workers. One co-located cohort
does not establish a compute-cap admission. Total wire bytes, independent
operators, upstream checkpoint distribution and setup remain unmeasured;
macOS per-process `nettop` snapshots undercount even reconciled online bodies
and cannot close that gap.

Interactive dashboard runs default to Qwen2.5-0.5B-Instruct; random tiny weights
require the explicit `--tiny` transport-smoke option.
Completed and failed benchmark summaries are appended to a schema-versioned local
SQLite archive. Its matrix groups only completed runs by model ID, immutable body
fingerprint, cold/warm mode, and exact input-token count. The archive never stores
prompts, generated text, token IDs, activation payloads, seeds, masks, credentials,
or protocol spans.
The headless benchmark can execute multiple Experiment configurations sequentially
and records both configuration and Pipeline digests. It ranks latency and throughput
only when measured model fingerprint, input and output token counts, output cap, and
warm state match exactly. An explicitly requested winner export writes the
lowest-median-full-latency Experiment as canonical JSON for later reruns.
With an explicit output path, the CLI atomically checkpoints candidate identity,
status and each completed report before proceeding. Incomplete documents use a
distinct progress schema with no rankings; timeout and cancellation retain earlier
reports without persisting exception text, private prompts or the cohort salt.
The ordinary final report replaces the checkpoint only after the comparison is
built. Checkpoints do not resume one-use sessions or merge separate cohorts.
The separate opt-in `pllm benchmark quality` path scores the same prefill token
cohort against a pinned local float32 checkpoint for up to eight immutable
Experiments, using their selected W4/W8 and A4/A8 settings. It records exact
checkpoint, dataset, token-cohort, configuration and environment digests plus
aggregate top-1, top-k and worst-logit-error scores, without archiving prompt
text, token IDs or logits. Source safetensors are capped at 12 GiB and the
bounded in-memory logit working set at 512 MiB; candidate kernels are released
before loading the float32 reference. These bounds do not measure peak memory.
This is a compiled local clear-kernel numeric
diagnostic; reference loading, provider traffic, privacy and whole-generation
quality are outside its measured scope.

## Build and release boundaries

The Cargo workspace version is inherited by all Rust crates. The Python source
version and citation version are checked against it by `scripts/release.py`.
Release changes use that script rather than independent manual edits.

The root `cargo test` runs the Python-independent default Rust workspace. The
wheel matrix compiles the binding for each platform and runs Python tests against
the installed wheel.
The publisher receives only validated artifacts and OIDC credentials; it does
not compile source with publishing credentials in scope.

References: Maturin project layout and configuration documentation;
PyO3 0.26 parallelism documentation. These links are listed in the development
page of the Fumadocs site.
