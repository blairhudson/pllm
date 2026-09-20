# Architecture

PLLM is a mixed Python and Rust project with one Python distribution. Maturin
builds the PyO3 module `pllm._native` and packages it alongside `python/pllm`.

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
| `pllm-compiler` | Region lowering, complete model-aware Qwen2 baseline scheduling, fixed-scale research composites, and plan verification |
| `pllm-assurance` | Scoped assurance results and checked public fixtures |
| `pllm-bench` | Native and deployment measurement records tied to plan and environment digests |
| `pllm-plugin-api` | Independently versioned C-compatible native provider vtables, statuses, handles, buffers, header, and conformance fixtures |
| `pllm-python` | The PyO3 `pllm._native` boundary exposed through the single Python distribution |

`crates/pllm-core` contains the integer matrix executor, scalar reference paths,
runtime AVX2 and NEON selection, bounded coefficient arithmetic, codecs,
quantization, masking, output subtraction, and operating system random sampling.
Matrices own their validated weights. Their dimensions and contents cannot be
mutated through the public Rust API. An executor owns a persistent Rayon pool.
The core also owns the bounded `pllm.numeric.silu.quadratic_q7.v1` reference:
signed Q7 over `[-1, 1]`, deterministic ties-to-even rounding, and an encoded-domain
absolute SiLU error bound of `0.02285`.
It also owns exact reference primitives for bounded signed Q14-to-Q7 rescaling,
Q7 multiplication, and their gated-MLP composition with Q7 SiLU. Rescaling and
multiplication use deterministic ties-to-even division by 128 and reject inputs
outside their declared domains rather than saturating.

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
the exact Qwen3.5-4B text decoder (`pllm.qwen3_5_text.v1`), the exact
Phi-4-mini-instruct decoder (`pllm.phi4_mini.v1`), and the exact text decoders in
the official `google/gemma-4-E2B-it` and `google/gemma-4-E4B-it` outer
configurations (`pllm.gemma4_e2b_text.v1` and `pllm.gemma4_e4b_text.v1`, model
family `gemma4_text`). Other Qwen, Gemma, Nemotron, Kimi, GLM and future families
must lower into the same IR or extend its semantic vocabulary rather than
introduce family-specific compiler paths.
Capability modules transform this IR through generic component contracts and
record immutable, digest-bound transformation lineage on the resulting plan.
Substitutable implementations are grouped by capability; a new family is added
only when a method has a genuinely different contract or lifecycle.

Semantic adapter support, checkpoint import, runtime graph support, compiler
operator coverage, protected/private parity, generation quality, benchmark
evidence and deployment support are separate claims. A complete bounded semantic
plan does not establish any later claim, and coverage is profile-scoped.

For batch-one, untransformed Qwen2, `baseline.masked_linear_cpu` now has a complete
model-aware prefill/decode schedule. The compiler groups q/k/v and gate/up stages by
graph topology, schedules every layer, KV transition, final norm, last-token
selection, output head, greedy selection and feedback, and emits a canonical digest.
The Python binding then verifies that schedule against the model configuration,
tokenizer, local norm tensors, quantized stage bytes, per-row scales, modulus policy,
preparation commitments and shared boundary weights. A plan-bound session enforces
workload limits and remote-stage output contracts, poisons partially advanced state,
and erases logits and KV state when execution ends. The pinned
Qwen2.5-0.5B-Instruct checkpoint passes a clear native-kernel prefill-to-decode
functionality test through this path; a separate tiny test exercises the masked
stage protocol.

This baseline does not promote `research.single_evaluator`. The fixed-Q10 research
path has executable regions for all dense-Qwen semantic operators, graph-derived
Q14-to-Q10 edges, clear attention and layer composites, and bounded one-use Q7
SiLU/multiply material, but those protected and fixed-scale components are not yet
composed into a real-model whole decoder. Qwen3, transformed MPCache execution,
Qwen3.5, Phi and Gemma likewise remain incomplete for whole-model compiler
execution. Existing runtime support for those checkpoint families is a separate
axis unless an exact schedule and binding say otherwise.

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
(`protocols`, `preparation`, `correlation`, `kernels`, `roles`, `nonlinear`,
`schedulers`, `state`, `passes`, and `verification`), and `pllm.components`
derives descriptor discovery from those classes. `pllm.profiles` provides typed
slot contracts for the two-role public baseline and shipped one-role proprietary
engines while generic serialized pipelines remain available. `pllm.providers`
discovers static external manifests and package-confined resources without importing
provider code; factory import is a separate approved operation. `pllm.metrics` owns
typed metric semantics; `BenchmarkResult` and `EvidenceRegistry` preserve exact
evidence cohorts without implicit ranking. `pllm.search` generates validated
immutable candidates and applies explicit cohort-safe Pareto directions.
`pllm.research` keeps attribution, quarantined upstream artifacts, clean-room methods,
and promotion gates static and non-executable. Public objects are imported on demand.
`pllm.runtime` holds the separate
runtime model graph, HE preparation, transport, protocol, scheduling and
importers. Applications should not depend on internal module locations.

`python -m pllm` and the installed `pllm` command use the same entry point.
A source checkout does not shadow an installed wheel through a second package
at the repository root.

`pllm.runtime.build_roles` constructs the single loopback role topology used by
`gateway --local`, the development dashboard, and the benchmark driver. It starts
inference and preparation as separate child processes, keeps generated credentials
in environment variables rather than argv or status records, binds health-checked
URLs to client/gateway factories, and owns process-group shutdown. The benchmark
dashboard driver now runs in-process, so a diagnostic run has two role children
rather than a dashboard child that creates another process tree. This topology is
for local development and measurement; it does not establish non-colluding
operators or production deployment.

## Hot operations

A matrix is copied into Rust once at compilation, then reused for later calls.
Input and output conversions are explicit. This is not a zero copy interface.
The present snapshot consumes one extra signed byte per weight while the source
matrix is retained for metadata and preparation-service loading. Benchmarks must
count that storage and buffer conversion work.

Moving arithmetic to Rust does not remove dependent client-to-inference stage
exchanges, offline preparation work, or client attention state. No speed claim
follows merely from the choice of implementation language.

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
generates each website article and PDF, using Tectonic locally or pdfLaTeX in CI;
only shared PDF layout details remain in TeX. Historical measurements are kept
under `docs/evidence` and are not rewritten as native Rust results.

The loopback benchmark dashboard runs the real client, preparation, and inference
roles and receives their OTLP metrics and traces directly. Protocol byte counts
are custom OTEL metrics; prompts and activation payloads are never telemetry
attributes. The generated tiny checkpoint validates transport behavior only.
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

## Build and release boundaries

The Cargo workspace version is inherited by all Rust crates. The Python source
version and citation version are checked against it by `scripts/release.py`.
Release changes use that script rather than independent manual edits.

The root `cargo test --workspace` runs the Python-independent Rust workspace. The
wheel matrix compiles the binding for each platform and runs Python tests against
the installed wheel.
The publisher receives only validated artifacts and OIDC credentials; it does
not compile source with publishing credentials in scope.

References: Maturin project layout and configuration documentation;
PyO3 0.26 parallelism documentation. These links are listed in the development
page of the Fumadocs site.
