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
disk storage. Compiler-bound streamed bundle integration remains pending, so this
does not yet reduce the ordinary SDK decoder's retained bundle memory.
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
composed into a real-model whole decoder. Transformed MPCache and compiler-bound
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
reduce its lean native allocation estimate from 34.78 to 19.75 GiB, still above
available headroom. That figure includes 5.98 GiB for Inference, 5.52 GiB for
Preparation and 8.25 GiB for the client, with a 25% margin in each. The client
still charges conservative bundle-copy, 71-row mask and decoder-workspace bounds;
the original 7.49 GiB BF16 checkpoint is not a whole-topology memory estimate.
Docker retains the legacy allocation
upper bounds for potentially older images. A tiny native
CPU/Metal control passes functionality and cleanup, with equal outputs and zero
observed swap growth; it does not establish large-model capacity.

A matrix is copied into Rust once at compilation, then reused for later calls.
`CompiledMatrix.weight_view()` exposes a read-only NumPy alias that retains that
immutable owner. Provider stage loading retires the original i8 allocation or
mapping; metadata, preparation and GPU import use the shared CPU snapshot.
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

The opt-in `research.verified_masked_linear_cpu` profile adds a trusted-client
Freivalds check to that prepared path. Preparation returns authenticated,
per-row projections over the client channel; Inference never receives the root
seed, challenges, or projections. The client verifies bounded integer stage
outputs before dequantization and burns material on use, cancellation, or
failure. This is a Slalom-derived engineering adaptation, not a TEE reproduction
or an actively malicious preparation guarantee. It does not reduce online
traffic.

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
issuance under a conservative 16 MiB active-work bound, admits results in order,
and seals only after all pushes are acknowledged. Failure and cancellation burn
the attempt before workers are joined. These choices add no client body weights.
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
`paper/whitepaper.md` are the canonical Pandoc Markdown paper sources. Pandoc
generates both website articles and the technical PDF, using Tectonic locally or
pdfLaTeX in CI for the latter. The whitepaper PDF uses a dedicated Pandoc HTML/CSS
print layout rendered by Chrome or Chromium; only technical-paper layout details
remain in TeX. Historical measurements are kept
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
