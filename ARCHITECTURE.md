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
NumPy; it is not a compiled-plan placement or provider backend. Transfer-inclusive
single-stage measurements must not be counted as whole-decoder or CPU-only
topology acceleration.
Matrices own their validated weights. Their dimensions and contents cannot be
mutated through the public Rust API. An executor owns a persistent Rayon pool.
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
remain unvalidated. Pinned Phi-4-mini-instruct additionally compiles for an
original-context workload. Generated tiny Phi fused-QKV/gate-up weights import
into that schedule and pass typed-session W8A8 prefill/decode against a PyTorch
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

A matrix is copied into Rust once at compilation, then reused for later calls.
Input and output conversions are explicit. This is not a zero copy interface.
The present snapshot consumes one extra signed byte per weight while the source
matrix is retained for metadata and preparation-service loading. Benchmarks must
count that storage and buffer conversion work.

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
For tied embeddings, one vocabulary-by-hidden matrix uses per-token row scales;
token lookup gathers its rows and `lm_head` multiplies by the same array. This is
not numerically identical to schema 1 token lookup, which independently quantized
the transposed matrix with per-hidden-feature scales. Output-head quantization is
unchanged. Any per-layer token table remains a separate auxiliary object using
its existing orientation.

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
roles and receives their OTLP metrics and traces directly. Protocol byte counts
are custom OTEL metrics; prompts and activation payloads are never telemetry
attributes. The generated tiny checkpoint validates transport behavior only.
The benchmark report additionally groups existing client audit counters into a
directed, phase-labeled **serialized protocol-body** ledger for the prepared
graph, without adding overlapping upload/server counters twice. It reports
client and all-link accounted bodies plus available run-window CPU per role,
but leaves total wire bytes and full-response compute-cap admission unset:
HTTP/TLS/control traffic, correction acknowledgements, pre-run bundle and
checkpoint distribution, and upstream setup are not completely metered.

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
