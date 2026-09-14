# PLLM: a privacy-focused multi-party inference runtime

**Architecture decision and implementation specification — 14 September 2026**  
**Status:** proposed design, not a release, benchmark result, or security certification.

## 1. Product definition

PLLM compiles public-weight models into explicit multi-party execution plans, executes those plans, and measures their privacy assumptions, numerical fidelity, online performance and full preparation cost. It is a research and inference runtime, not the name of one masking protocol.

The stable public abstraction is:

> Model + numerical semantics + privacy policy + topology + implementation choices + workload + hardware → validated, versioned execution plan.

The project should make published methods reproducible, experimental ideas easy to insert, and invalid compositions difficult to execute. It should provide a familiar Python interface, declarative configuration, a high-performance Rust execution layer, and evidence-backed documentation.

Retain the `pllm-inference` distribution name, `pllm` Python import and CLI, `uv` development workflow, PyO3/maturin packaging, and a Responses-compatible **client-side** gateway. Treat older CLI forms as migration aliases with explicit warnings. An API-compatible gateway is part of the trusted client boundary; the untrusted inference service never receives ordinary plaintext Responses requests.

### Preferred deployment

- **Client:** owns inputs and outputs; ideally only tokenizes, encodes, decodes and manages requests; holds no model weights.
- **Preparation:** may hold the complete public model; creates input-independent, one-time execution material before the corresponding online execution; does not receive the matching online transcript.
- **Inference:** one online model-evaluating worker, with the public weights and prepared material; no plaintext prompts, outputs or intermediate semantics under the selected protocol's assumptions.
- **Coordinator:** launches jobs and aggregates non-sensitive measurements; not an additional cryptographic party and not a recipient of all party secrets.

No HE, no online Preparation dependency, and no second online model worker are hard constraints for this preferred profile. Alternative research profiles may explicitly relax them for faithful comparisons. A comparison track is not a recommendation to deploy a different topology.

“Multi-party” describes trust roles. Several threads or GPUs owned by one Inference administrator are not several independent parties. Conversely, separate local subprocesses demonstrate message boundaries, not administrative non-collusion.

## 2. Evidence reset and defaults

The source manuscript reports nine full Qwen2.5-0.5B-Instruct loopback executions of **masked linear outsourcing with substantial client computation**. The latest garbling reports describe synthetic operators and small graphs, not a complete model, production security, or near-native TPS. Their custom weighted transducer has a useful component trade-off but is not an established fastest secure backend. [I01–I04]

Therefore ship two explicitly named starting points:

### `baseline.masked_linear_cpu`: default for the first local full-model benchmark

Reproduce the archived numerical graph and partition before changing either. Use one Client, one offline Preparation and one Inference subprocess; persistent binary transport; exact Rust modular kernels; packed prefill; one-use inventory; no HE. Preserve the original local embeddings, output head, attention and nonlinearities. Report that the client is **not thin**.

The archive's exact numeric manifest and model snapshot must be recovered and hashed. Do not invent missing scales or silently substitute W8A8. If the archive cannot be recovered, create a new named numerical profile, establish its reference outputs, and label it a new baseline—not a reproduction of revision `277d19f`.

This default is an engineering starting point supported by a reported end-to-end experiment, not a present SOTA claim or an audited production profile. A baseline reproduction must pass again in the integrated runtime.

### `research.single_evaluator`: preferred target configuration

Model-aware offline arithmetic garbling, one online evaluator, no client model, exact integer semantics, reviewed conversion/gate constructions wherever implemented, and model-derived representation bounds. Start with public matrix kernels over arithmetic labels and exact Boolean/CRT helpers. Candidate nonlinear backends are Boolean half-gates, full arithmetic LUTs, reviewed compact LUTs, and the experimental weighted transducer.

Do not globally select weighted paths: the source experiment shows Boolean synthesis winning on prepared bytes, flat lookup winning on online latency, and weighted paths occupying a third trade-off. [I04]

Only implementations with the required operator coverage and accepted assurance level are eligible. The profile must fail with a useful coverage/cost report when full Qwen support is missing. It must not fall back to client nonlinear computation, a second evaluator, a weaker privacy model, or plaintext execution.

### Measured auto-selection

Eventually `auto` selects from a signed local or published evidence catalogue for a key containing:

`model digest × numerical graph digest × privacy contract × client constraints × topology × device × network × workload bucket × implementation versions`.

There is no universal “fastest per model” record. Context length, batch, prefill/decode, hardware, preparation capacity and client budget all matter. No eligible measured plan means `NO_VALIDATED_PLAN`, not a speculative winner. The catalogue never contains secret preprocessing.

## 3. Architecture and repository

```text
Python SDK / CLI / trusted local Responses gateway
                         |
                 Configuration + policy
                         |
Model adapter → Semantic IR → Numeric IR → Protected IR → Distributed plan
                                  |             |              |
                         reference oracle   method registry   kernel registry
                                  |             |              |
                              correctness / costs / evidence
                                                |
                role-specific artifacts and execution DAGs
                  /               |                  \
               Client         Preparation          Inference
                  \_______________ protocol ______________/
                                  |
                      benchmark records and method cards
```

Proposed monorepo:

```text
python/pllm/
  api/                 # Runtime, Client, Session; typed user API
  config/              # models, JSON Schema, migration and lockfiles
  models/              # adapters, tensor mapping, formats, reference execution
  ir/                  # semantic/numeric/protection plan schemas
  compiler/            # analysis, region selection, cost models, search
  methods/             # protocol descriptions and Python reference evaluators
  bench/               # launchers, suites, metrics, comparisons, reports
  research/            # source/provenance registry and reproduction recipes
  gateway/             # trusted local HTTP/SSE compatibility surface
  plugins/             # discovery and compatibility checks
crates/
  pllm-types/          # types, manifests, canonical encoding and IDs
  pllm-arithmetic/     # exact rings, fields, CRT, quantized packing
  pllm-crypto/         # reviewed primitive wrappers, secret ownership
  pllm-kernels/        # scalar, SIMD and optional accelerator dispatch
  pllm-protocols/      # role-specific masking/garbling/MPC implementations
  pllm-inventory/      # one-use allocation and durable recovery
  pllm-transport/      # authenticated persistent binary channels
  pllm-runtime/        # executor, scheduling, buffers, protected state
  pllm-python/         # thin PyO3 boundary, exports pllm._native
accelerators/          # optional CUDA/CUTLASS/Metal bindings and kernels
research/              # paper records, pinned external reproduction recipes
benchmarks/            # public workloads, numerical profiles, frozen suites
schemas/               # plan, method, config, result, evidence versions
security/              # contracts, composition analyses, negative controls
```

Do not force every dependency to be Rust. Reviewed native cryptographic or GPU libraries may sit behind a narrow FFI. Record their versions, licenses and security assumptions. Rust is the native orchestration and ownership boundary, not an excuse to rewrite established cryptography.

Maturin supports a mixed Python/Rust project and PyO3 exposes a deliberate way to detach from Python for native work. [S20–S21] Python should not dispatch per scalar, gate, residue, or token-stage packet. Compile region schedules once and execute them in native code. Keep OS entropy, masks, label offsets and prepared secrets in native opaque objects rather than ordinary Python arrays or serializable models.

## 4. Component contracts

### 4.1 Model resolver and `ModelAdapter`

**Input:** pinned repository revision or explicit local source; format; adapter version; allowed architecture features.  
**Output:** `ModelManifest`, immutable tensor store, `SemanticGraph`, tokenizer artifact and reference runner.

Interface responsibilities:

- `inspect(source) -> ModelDescriptor`: read metadata without executing checkpoint Python.
- `validate_tensors(descriptor, store) -> ValidationReport`: names, shapes, aliases, dtypes, shard coverage, quantization groups and bias conventions.
- `lower(descriptor, store) -> SemanticGraph`: declare operations, learned parameters, attention semantics and state updates.
- `reference(graph, public_fixture) -> ReferenceResult`: official-framework comparison in a trusted test harness.
- `conformance_cases() -> fixtures`: prefill, decode, positions, cache growth and attention-mask edge cases.

Resolve mutable names once, then lock file hashes. `trust_remote_code` is false by default. A user-installed adapter is trusted executable code, not made safe by a manifest. Safetensors is the initial tensor container; using that format does not authenticate a model or prove its architecture. [S22–S24]

The first real-checkpoint target is Qwen2.5-0.5B-Instruct. Its current official config lists 24 layers, hidden width 896, 14 query heads, two KV heads, intermediate width 4,864, SiLU, RMSNorm, tied embeddings and vocabulary 151,936. [S25] Lock the actual snapshot before tests; current `main` metadata does not verify an archived revision.

### 4.2 Semantic operator library

Operations describe mathematical behavior, not a cryptographic protocol. Initial set:

`embedding`, `linear`, `bias_add`, `reshape`, `transpose`, `split`, `concat`, `residual_add`, `rms_norm`, `layer_norm`, `rope`, `qk_matmul`, `causal_mask`, `softmax`, `attention_value_matmul`, `silu`, `gelu`, `gate_multiply`, `kv_append`, `kv_read`, `lm_head`, `argmax`, `sample`, `eos_update`.

Retain fused high-level regions as well as legal decompositions. An attention primitive is not universally replaceable with a dense matmul. Every operation defines deterministic semantics or an explicit randomness distribution, shape rules and state effects.

Future operators include private expert routing, gather/scatter, recurrent/state-space updates, convolution, hybrid attention and multimodal encoders. Unknown features are explicit coverage gaps. No rule should infer that all models with approximately similar tensor names are interchangeable.

### 4.3 Numerical specification

`NumericType` records signedness, value modulus/domain, accumulator width, scale representation, rounding, saturation, clipping, quantization grouping and public range certificate. Storage bit width is separate from semantic modulus and wire encoding.

Provide three separate semantic identities:

1. Source floating-point graph and official-framework outputs.
2. Frozen declared quantized/approximated plaintext graph.
3. Protected implementation of that frozen graph.

Private-vs-quantized-reference equality proves (3) implements (2); it does not prove (2) preserves (1). A new approximation creates a new numerical graph digest and requires quality evaluation.

Bounds may be interval, norm/ellipsoid or another reviewed sound abstraction. Include learned scale factors, biases, residual correlations and approximation error. A measured activation maximum is calibration evidence, not a worst-case certificate. Exact rescaling, sign and CRT conversion are explicit operations—not hidden casts.

### 4.4 Protocol method registry

`MethodSpec` contains:

- immutable method ID/version and source references;
- semantic operators/regions supported;
- accepted input/output protections and numerical domains;
- participants and phase-specific message graph;
- trust and non-collusion assumptions;
- integrity scope, leakage, state and fresh-resource requirements;
- preprocessing provider and schema;
- kernel alternatives and conversion requirements;
- implementation/evidence/security-review status;
- estimate function and reproduction recipe.

Each method also declares algebraic preconditions such as invertibility, valid public constants and permitted linear relations between labels; label arithmetic that reconstructs correctly is not by itself a privacy argument. A method is an algorithm/protocol; a kernel is an implementation of a particular step. `masked.linear.v1` is not the same type of thing as `cpu.avx2.ring24.gemm.v1`.

Registration namespaces: `pllm.models`, `pllm.methods`, `pllm.conversions`, `pllm.preparation`, `pllm.kernels`, `pllm.metrics`, `pllm.launchers`. Registry discovery must not enable an experimental plugin by default. Allow plugin metadata inspection without importing its executable code where feasible; imports still require trust.

### 4.5 Native kernel contract

A kernel exposes `supports(request)`, `prepare_public(constants)`, `workspace_bytes(shape)`, `execute(ctx, inputs, outputs)` and `measure(public_fixture)`.

The capability signature includes exact arithmetic domain, shapes, strides, value layout, weight encoding, alignment, device, required instruction sets, workspace, overflow conditions, constant-time claim and implementation hash. Correctness dispatch must precede performance dispatch.

Kernels must explicitly implement modular overflow semantics and accumulator reductions. Never process uniform wide random shares by converting them to an inexact native floating-point matmul. Preflight precision bounds may permit an exact float-backed integer fixture; that is a narrowly typed implementation, not a general permission.

CPU first: portable exact reference; Rayon with bounded thread pools; AVX2/AVX-512 where available; ARM NEON. GPU track: retain packed public weights where mathematically valid; unpack a tile once; reuse it across compatible label components; include transfer, reduction and label traffic in timings. No CUDA support claim until built and tested on actual devices. Metal is a separate backend, not a CUDA compatibility promise.

Use native buffers with explicit ownership and immutable views. A zero-copy view must not allow the caller to mutate an input that has already been bound to one-time material. Snapshot/freeze at binding or hold exclusive immutable ownership.

### 4.6 Conversions

Conversions are first-class protected operators. Their registry records source/destination representation, parties, randomness, online messages, work, numeric error and security assumptions.

Examples: arithmetic-label → bit-label; bit-label → arithmetic-label; additive-share → Boolean-share; signed lift; exact ring narrowing; CRT base extension; authenticated output verification.

The compiler cannot connect two fast methods because their plaintext tensors have the same shape. It needs an implemented, eligible conversion with a compatible proof boundary. Duty-Free Bits has a particular conversion direction and assumptions; it is not a free bidirectional bridge. [S05]

### 4.7 Preparation providers

Providers produce typed material, not generic byte blobs. Initial schemas include `masked_linear_record`, `arithmetic_garbling_instance`, `boolean_garbling_instance`, `lookup_instance`, `verification_epoch`, and—in explicitly different profiles—`beaver_triples`, `fss_keys`, `he_keys/ciphertexts`.

Classify every resource as:

- reusable public compilation/cache;
- reusable private state with an explicit bounded-use argument;
- one-use material per input/activation/wire instance;
- immutable-operand material whose reuse has a protocol-specific justification.

A public model copy allows local label-base multiplication, but not reuse of garbled instances across changing inputs. Ordinary seed compression is not pseudorandom correlation generation. A PCG must supply the exact correlation type and current reviewed parameters; scalar triples do not automatically implement correlated matrix or spectral records. [S15–S16]

An offline bundle contains separate party slices, public schema metadata and a preparation receipt. Secrets are encrypted to the intended party; the coordinator receives only the public receipt. `ready` means the recipient validated and installed the required material, not that a server assertion cryptographically proves its future computation.

### 4.8 Inventory and crash consistency

Core invariant:

> Each one-time item is assigned to at most one distinct semantic input/activation across retries, cancellation, concurrent writers and supported recovery.

States: `prepared → available → bound → possibly_exposed → accepted → retired`; any nonfinal state may retire; no return from bound/exposed to available. Bind and freeze request identity before any bytes can leave the sender. An identical-message retry is distinct from assigning the mask to a new input.

For garbling, allocate an execution epoch and unique wire/material instances. Sharing the same immutable wire as graph fan-out is not re-encoding a second value on it. KV reuse, speculative rollback and beam branches require explicit lifetime rules. Restoring model state never restores consumed randomness.

Use durable allocation journaling, idempotent execution IDs, crash injection, process-safe allocation and context-bound key derivation. Plain SQLite/WAL or an append journal does not prevent snapshot rollback. Either exclude arbitrary snapshots/cloning or use a separately specified monotonic coordinator. Non-colluding parties cannot rely on the malicious evaluator to enforce client freshness.

### 4.9 Party executor and transport

Each role receives only its executable schedule, public inputs and permitted secret slice. Native runtime execution is an asynchronous dependency DAG with compute, transfer, inventory and verification edges. No secret-dependent stage selection or observable shortcut unless explicitly allowed by the leakage contract.

Initial transport: persistent authenticated binary WebSocket over TLS, preserving the existing path as a reference. Local benchmark agents generate ephemeral test credentials into separate role directories; remote identities are user-provisioned. Later transports implement the same abstract protocol and independent counters.

Every frame binds protocol/schema version, session, execution epoch, plan digest, stage, material allocation, sequence number, logical step, declared length and message kind. Enforce limits before allocation; reject mismatched plans and unknown fields/versions. TLS protects links, not inference correctness or administrative independence.

Measure bytes at the serialized protocol layer and at the transport/NIC layer when available. A message seen at both endpoints is one transfer, not two. Report framing/TLS/retransmission separately rather than mixing them with logical tensor bytes.

### 4.10 Protected generation state

The model adapter declares state slots; the selected protocol declares their encoding and lifetime. Initial slots: per-layer K/V arrays, logical position, attention mask, sampling state, EOS state and request output state.

The single-evaluator target needs protected embedding access, attention products, normalization, sampling/argmax and token-feedback labels. A client-side head or private sampler requiring a per-token client decision is a different profile. Identify it rather than concealing it under “local processing”.

Use bounded public execution budgets. For garbling, compile a bounded unrolled request or an explicitly composable per-step schedule with fresh step instances. Reusing one decode circuit for different tokens is prohibited without a reusable-garbling construction. Dummy/padded work, EOS handling, garbled RAM/oblivious access and cache access patterns must have an explicit policy. Public context length leakage does not automatically authorize secret expert indices or token-dependent table paths.

### 4.11 Trusted client API

`Runtime.from_config`, `Runtime.compile`, `Runtime.prepare`, `Runtime.session`, `Session.generate`, `Runtime.benchmark`, `Runtime.citations` form the Python API. Public configuration objects are immutable and cloneable, with `get_params(deep=True)` and `with_params(...)` for search ergonomics, without mimicking training semantics unnecessarily. The named-component/search approach follows the usability pattern of scikit-learn pipelines. [S26]

`pllm serve` starts the trusted client-side gateway by default on loopback. `/v1/responses` supports an explicitly tested subset: text input, streaming output, model listing, cancellation and typed errors initially. Unsupported tools/response formats fail explicitly. The remote inference endpoint accepts encoded protocol messages only. API compatibility does not make a plaintext remote API private. [S30]

## 5. Four intermediate representations

| IR | Contains | Explicitly excludes |
|---|---|---|
| Semantic IR | model operations, parameter refs, tensor shapes, state transitions, generation semantics | masks, protocol opcodes, device decisions |
| Numeric IR | quantized operators, scales, bounds, clipping, rounding, exact LUT identities | implicit approximation, unspecified overflows |
| Protected IR | value protection, permitted views, role placement, conversions, freshness effects | hidden plaintext fallback, untyped crypto transitions |
| Executable plan | per-role DAGs, kernels, buffers, messages, preparation budget, lifecycle | mutable model refs and unresolved `auto` choices |

Protected values have at least:

`TensorId`, `NumericType`, `ProtectionType`, `Owner/ViewSet`, `EpochId`, `StateKind`, `Layout`, `ShapeBucket`.

Protection variants include `Public`, `ClientPlaintext`, `MaskedRing`, `AdditiveShare`, `ArithmeticLabel`, `BooleanLabel`, `Ciphertext`, and `AttestedPlaintext`. These are incompatible types unless an eligible conversion explicitly connects them. The last two are available only in separately authorized profiles.

Compiler stages: resolve → semantic conformance → choose/freeze numeric graph → prove/check bounds → enumerate protected regional alternatives → insert compatible conversions → validate policies → estimate total costs → place/schedule → produce role slices and a signed public plan → execute calibration → lock selected plan.

The validator checks **declared necessary compatibility conditions**. It is not a proof that an arbitrary third-party protocol is secure. Never derive a “proved secure” badge from passing type checks.

## 6. Search and compiler optimization

Do not expose an unconstrained Cartesian product of all knobs. Search a conditional grammar of valid regional plans.

Legal candidate dimensions include protocol family permitted by the profile, exact kernels, SIMD/device variants, bound analyses, residue choices that satisfy the bound, LUT variable orders, nonlinear implementations, conversion boundaries, prefill packing, prepared block size and public schedule.

Security level, accepted corruption model, role independence, allowed leakage and application numerical-quality limits are **constraints**, not rewards that the optimizer can trade away. Research relaxations create distinct benchmark cohorts.

Begin with enumerated baselines and random search; add constrained Optuna multi-objective search when the candidate space is stable. [S27] Keep a Pareto set over latency, total resource cost, preparation bytes and client work. Avoid a single arbitrary weighted score.

Use regional dynamic programming/beam search with conversion costs and Pareto pruning. Model state, critical-path dependencies, memory limits and globally coupled representation choices prevent independent per-op minimization from being generally optimal. A later e-graph/ILP formulation is optional, not a prerequisite.

Evaluate in stages:

1. static policy, coverage, bound and storage validation;
2. exact arithmetic/property tests;
3. operator/region benchmarks;
4. full-block measurements and cost-model correction;
5. full-model numerical and quality checks;
6. multi-process request and steady-supply benchmark;
7. held-out workload confirmation and published-plan promotion.

Candidate pruning may use an optimistic cost bound, but must record pruning reasons and model uncertainty. Never promote a cost-model prediction as a measurement. Keep final test prompts/workloads separate from tuning data. Tie-breaking and randomness are recorded. Fresh cryptographic material is used for distinct executions even with a reproducible public workload seed.

Cache compiled **public** artifacts by complete digest. Never cache one-time preparation as an ordinary reusable benchmark result. Search resume must not resurrect consumed material.

## 7. Benchmarking specification

### Launchers and isolation

`inprocess` exists only for arithmetic/debug tests. `local_processes` launches separate role processes with distinct directories, credentials, thread budgets and resource counters. `ssh` runs preinstalled role agents on existing hosts; a launcher must not copy all party secrets onto one remote coordinator. Containers and Kubernetes can be later adapters.

A benchmark configuration is public and identical across roles; the compiler derives role-specific bundles. All roles report their loaded plan digest and binary/backend versions over authenticated channels; this is a software receipt, not hardware attestation or proof of execution. An online-only test explicitly shuts down or disconnects Preparation after readiness and records any attempted contact as a violation.

Two supply regimes are separate experiments:

- **Frozen inventory:** prepare before the request, disconnect Preparation, consume a known pool.
- **Sustainable supply analysis/test:** quantify replenishment for future independent requests and count every Preparation resource. No current request obtains input-dependent assistance; a stall is not hidden from end-to-end timing. An explicitly strict offline-only deployment can use finite-horizon supply accounting instead of concurrent fleet replenishment.

### Workload ladder

1. Operators: matched rings, gates, tables, conversions and protected access.
2. Regions: MLP including all rescaling; attention including private products/normalization; vocabulary feedback.
3. Real full Qwen2.5-0.5B model, pinned snapshot and tokenizer.
4. Larger supported dense checkpoints.
5. Other architectures only after independent conformance gates.

Local smoke: 30-token prompt, 16-token generation cap, concurrency one, one warmup and three measured runs. This is a smoke test, not publication-grade latency statistics. Add archival 63/255-token cases only when the old numeric profile is recovered.

Search suite: public prompt lengths 32/128/512, generation 16/64, concurrency 1 first, several content fixtures per bucket. Publication suite: add longer contexts, realistic EOS and fixed-length decode modes, multiple machines/networks and independently repeated requests. Start with at least 30 requests per selected cell; sample counts and confidence intervals—not a magic repetition count—determine which tail claims are supported.

A fixed-length run must actually advance the model and protected state at every token. Repeating one stage or forcing the same logits is not autoregressive TPS. Report padded/dummy tokens separately from useful output tokens. Speculative decoding counts accepted tokens, rejected draft work and consumed/burned preparation.

### Timing definitions

Use client monotonic time for end-to-end clocks; use per-device events for local kernel durations. Do not subtract unsynchronized timestamps from different machines.

- `request_ttft`: request accepted at trusted gateway → first verified/releasable token.
- `ready_ttft`: inventory ready and request released → first token; separate from request TTFT.
- `decode_tps`: `(G-1)/(t_last_token - t_first_token)` for `G >= 2`; undefined otherwise.
- `request_output_tps`: `G/(t_completed - t_accepted)`.
- `aggregate_output_tps`: useful emitted tokens divided by workload wall time.
- `tpot`: all observed inter-token intervals; report within-request and across-request aggregation separately.

Also record import, compile, public setup, one-time setup, preparation compute, preparation transfer, readiness waiting, queueing, prefill, decode, verification, finalization and cancellation. For GPUs, synchronize the measured region and include backend events; distinguish device execution from host enqueue time.

### Required resource and privacy measurements

Every result reports useful/padded tokens, role CPU-seconds, role accelerator active time and allocated device-seconds, peak RAM/VRAM, public model bytes, client model/secret bytes, disk/HBM preparation footprint, logical bytes by party pair/direction/phase, transport bytes when observable, online dependency rounds, prepared/consumed/retired records, wasted work and failures.

Record a labelled privacy contract, not a scalar “privacy score”: adversary model, allowed collusion sets, hardware assumptions, plaintext locations, shape/timing/access leakage, integrity coverage, parameterization, proof references and implementation-review status. No-plaintext telemetry is a hygiene check, not a confidentiality proof.

Numerical metrics: stage exactness against the declared graph, max error, mismatched logits, greedy agreement, perplexity/NLL delta, task metrics, long-context behaviour and stochastic-distribution tests when relevant. Comparisons using changed numerical graphs belong in accuracy/performance cohorts.

DeepEval-like metric objects are useful for extensibility, but default evaluation must not call a remote judge with private prompts or outputs. [S28] Public-quality suites or locally executed metrics are the default.

### Economics and sustainable throughput

For material type `j`, let `r_j` be production rate and `u_j` useful consumption per token including expected waste. A necessary steady-state bound is:

`TPS_sustainable <= min(TPS_online, min_j(r_j/u_j))`.

This is a capacity bound, not a full queueing model. Include fixed build cost divided by an explicitly stated request horizon, all one-use costs, transfer/egress, storage and all parties' compute. Sharing W at Preparation is an allowed replica whose capacity cost remains visible. Do not divide one-use garbling by unlimited requests.

### Baselines and comparisons

Always keep (a) same declared plaintext numeric graph, (b) same partition without cryptographic protection where meaningful, and (c) optimized native inference with its own exact checkpoint/quantization disclosure. Add the original PLLM path and faithful published-family implementations. Show comparisons within compatible privacy/quality cohorts; use a separate table for changed trust models or party counts.

External paper numbers are context, not rows in a matched measured speedup table. Record `author_reported`, `external_artifact_run`, `pllm_reimplementation`, and `analytic_estimate` as distinct evidence origins.

Store raw samples, failures and timeouts. Compute confidence intervals with request-level clustering; token events from one request are not independent samples. Tail estimates need adequate sample counts. Never silently delete outliers or omit failed configurations.

## 8. Research registry, documentation and reproduction

The initial `research_registry.json` distinguishes primary-source access from PLLM implementation status. Import previous bundles as **reference experiments** with original reports and hashes, not as proven methods or integrated reproductions.

Each method page should contain:

1. Plain-language purpose and a small worked example.
2. Original paper, version/venue, canonical link and relevant section.
3. What PLLM implements, what it omits, and differences from the paper.
4. Privacy/trust/role diagram in prose or a maintained figure.
5. Numeric domain, required ranges, supported models/shapes and conversions.
6. Configuration/API parameters and one executable example.
7. Offline/online resource costs, actual benchmark artifacts and hardware.
8. Correctness, proof and audit status as separate fields.
9. Failure modes, known attacks and rejected variants.
10. Reproduction command and generated citation list.

Do not use a single “reproduced” Boolean. Use axes:

- **Source:** primary full text accessed / abstract only / metadata only / unresolved.
- **Scope:** algebra / operator / region / full model / deployed multi-process benchmark.
- **Fidelity:** original artifact run / faithful reimplementation / adapted method / novel composition.
- **Validation:** reported / locally rerun / independently confirmed.
- **Assurance:** proof applicability, composition review and implementation audit, individually.

A published proof is not inherited automatically by altered arithmetic, hash instantiation or cross-protocol conversion. Weighted transducers and masked spectral gates remain `experimental` even after every functional test passes. Rejected constructions remain in attack regressions, never in ordinary auto-selection.

Docs pages, CLI `methods show`, API parameter docs, method citations and result provenance should all derive from the same registry. Handwritten explainers may be concise, but performance tables must be generated from evidence records. Public benchmark inputs may be bundled; raw secret seeds, active labels, masks, credentials and private payloads must not be published.

Use Markdown/MDX-compatible source and retain the existing documentation framework unless there is a separate reason to migrate; the research registry and Python API reference should not depend on the site framework. Retain the ability to publish Markdown and machine-readable metadata for agents and offline use.

## 9. Model evolution

Support is a matrix, not a model-name list:

`format × architecture feature set × numeric profile × protocol family × device × workload mode`.

Adding a model that uses existing operators should require an adapter and conformance fixtures, not protocol rewrites. Adding a genuinely new operation requires a semantic definition, plaintext oracle, numeric lowering, protected implementation(s), cost model and regression fixtures. An unsupported method/operator cell is normal and explicit.

First adapter: `qwen2_dense`. Follow with other dense text families after official-checkpoint tests. Do not claim real checkpoint support from synthetic shape-compatible fixtures. Previous portfolio reports explicitly distinguish format parity, quantized-graph parity and original-model quality. [I05]

For MoE, expert routing and token counts per expert are secret-dependent; public weights do not make routing public. For hybrid recurrent architectures, state updates need explicit protected lifetime rules. For multimodal models, encoders and preprocessing need their own privacy boundary. Dynamic Python execution, unknown quantization formats and unsupported RoPE/cache variants fail closed.

Use a versioned adapter contract rather than freezing the runtime to a particular Transformers release. PyTorch export or ONNX can be optional frontends; neither should define PLLM's protected IR or silently erase state semantics.

## 10. Tests and release gates

Mandatory suites:

- exact scalar vs SIMD/GPU arithmetic, boundary values, overflow and round-trip codecs;
- semantic/model differential tests at intermediate layers and full generation;
- conversion interoperability and compile-time rejection tests;
- one-use inventory with timeout, crash, concurrent allocation, rollback assumptions and cancellation;
- party-view and network-boundary tests;
- benchmark-counter reconciliation and phase-clock checks;
- poisoned model/plan/version/shape/label negative cases;
- preserved attacks: mask reuse, exposed offset, sparse point-and-permute leakage, shared Fourier coefficients, local truncation carry omission, visible fallback, weighted-path early exit, evaluator-derived node masks;
- fresh-wheel tests on supported platforms and deterministic public build artifacts.

Security assertions are tested where mechanically observable and accompanied by assumptions elsewhere. A passing test suite is not a cryptographic proof. The benchmark harness never claims physical isolation on a single-host process test.

Promotion gates: reference operator → native operator parity → complete region including conversions → public checkpoint correctness/quality → separate-party benchmark → independent reproduction → eligible default. Security review progresses on a separate track and can block promotion at any stage.

## 11. Paper direction

Suggested title:

**PLLM: Reproducible, Cost-Aware Compilation for Private Language-Model Inference**

Primary claim to test:

> Joint selection of protected representations, conversion boundaries, model-derived numeric bounds and one-time-material schedules improves the measured privacy-compatible performance frontier of autoregressive inference.

This is not the first multi-protocol runtime or mixed-protocol compiler. MP-SPDZ, HyCC and CrypTFlow/EzPC are direct systems/compiler prior art and must appear in related work. [S31–S34] The contribution must be specific to the implemented generative-inference workload, constraints, cost accounting and measured improvements.

Research questions:

- Can regional selection beat the best uniform backend at matched privacy and numerical semantics?
- How much do preparation, conversions and client work change apparent rankings?
- When do model-aware bounds remove whole protected arithmetic channels without changing outputs?
- How much do chosen plans transfer across context lengths, devices and architectures?
- Does any plan meet the preferred single-evaluator constraints with competitive full-model latency and total compute cost?

Ablations: fixed vs searched plan; kernel-only vs conversion-aware cost; interval vs geometric bounds; separate vs fused gates; flat vs Boolean vs weighted vs reviewed LUT; frozen vs sustainable inventory; old client-heavy vs single-evaluator topology in separate labelled cohorts.

Report Pareto frontiers with confidence intervals, absolute numbers, accepted quality budgets, party resources and failure rates. Publish exact sources/commits/configuration and reproduce original systems where feasible. Do not advertise a theoretical wire count, Python scalar speedup or rejected unsafe optimization as a full-model SOTA result.

## 12. Implementation order

**Milestone A — evidence and contracts.** Import old reports/results with hashes; establish method status axes, schemas, numeric/role contracts and a tested configuration validator. Freeze what was measured and stop promoting component claims by inference.

**Milestone B — benchmark spine and baseline.** Wrap the existing full-model masked-linear path in the common executor/launcher, recover the archived numeric graph, add raw metric records and same-graph plaintext comparison. Run Qwen in three processes with Preparation demonstrably absent online.

**Milestone C — native operator library.** Port exact rings/CRT, half-gates, conversions and LUTs through the same Rust contracts. Keep Python references. Implement reviewed alternatives before optimizing custom constructions. Import weighted-path experiments only as explicit research backends.

**Milestone D — complete single-evaluator block.** Include input conversion, nonlinearities, scaling, attention, vocabulary feedback and protected state. Measure preparation estimates before allocating material. A block that requires impractical storage should fail budget preflight.

**Milestone E — compiler search.** Start with legal enumerations, measured regional cost tables and Pareto beam search; add black-box search on selected public knobs. Freeze plans for held-out verification.

**Milestone F — full-model study and publication.** Broader model/device/network experiments, reviewed threat-model composition, externally reproducible artifacts and a paper grounded in measured comparisons.

The durable asset is the runtime and evidence system. Individual research techniques should be replaceable without discarding the model adapter, benchmarks, state handling or the paper's provenance.

## Source key

Internal reports are user-supplied experimental evidence, not independent validations. Exact local hashes are in `source_inventory.json`.

- **I01:** Original PLLM manuscript, 11 September 2026, `Pasted text.txt`.
- **I02:** Revised paper, `pllm_revision/PLLM_revised.md`.
- **I03:** Model-aware experiments, `pllm_model_aware_lab/REPORT.md`.
- **I04:** Weighted-transducer experiments, `pllm_tenx_lab/REPORT.md`.
- **I05:** Earlier `PLLM-Research-Portfolio-Round14.pdf`, Library excerpts on architecture/format/quality distinctions; not copied into this bundle.

Primary research and software links, access scope and implementation status are listed in `research_registry.json`. The URLs identify source work; they do not certify a PLLM reproduction.
