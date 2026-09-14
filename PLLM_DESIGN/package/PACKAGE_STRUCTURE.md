# PLLM package structure

**Proposed public API and repository organization — 14 September 2026**

This is an additive design for the existing PLLM uplift, not an implemented release or a replacement repository. Package names, classes and routes below are proposed contracts. A branch is created when its implementation exists; its presence in this design is not a support claim. Retain the `pllm-inference` distribution, `pllm` import and CLI, and the current migration/parity gates.

## 1. Organizing principle

Organize imports by **what a component does**, not by the paper it reproduces, its provider, its current maturity, or which party executes it. Paper attribution and maturity belong in the component registry. Party placement belongs in the compiled plan.

There are three surfaces:

1. Public Python modules: small, discoverable configuration objects, builders and result views.
2. The component contract: static descriptions and category-specific extension interfaces used equally by built-ins and community providers.
3. The Rust execution system: model-data processing, compiler passes over tensor data, numerical and cryptographic kernels, material allocation, scheduling, transport, benchmark clocks and view-restricted adversary execution.

Use sklearn's named composition, parameter inspection and conformance-check ergonomics, and DeepEval's domain-oriented imports and category-specific extension points. Do not imitate training with a meaningless `fit()` method, and do not treat the execution DAG as a linear sklearn Pipeline. Sources S1–S2 in `SOURCES.md`.

## 2. Canonical Python tree

`__init__.py` files are omitted except where important. Public imports use the package facade; implementation files have a leading underscore. `py.typed` and generated native stubs are shipped. This is a public navigation map, not a requirement to create dozens of empty modules.

```text
python/pllm/
├── __init__.py                  # Model, Pipeline, Experiment, Runtime, Benchmark, Study
├── py.typed
├── config/                      # Immutable public configuration; loading and migrations
│   ├── profiles.py              # Named baseline / single-evaluator / comparison profiles
│   └── serialization.py         # Public specifications only; never live execution secrets
├── types/                       # Small shared vocabulary; no heavyweight dependencies
│   ├── numeric.py              # Domain, scale, rounding, ranges and certificates
│   ├── protection.py           # Representation identity, ownership and permitted views
│   ├── topology.py             # Parties, trust domains and phase constraints
│   ├── state.py                # Epochs, effects, lifetimes and state transitions
│   ├── plans.py                # Versioned IR/plan descriptors and opaque plan handles
│   ├── evidence.py             # Source, correctness, assurance and measurement records
│   └── errors.py               # Stable typed diagnostics
├── components/                 # Minimal shared extension mechanism
│   ├── base.py                 # BaseComponent, ComponentRef, immutable configuration
│   ├── specs.py                # ComponentSpec, ProviderSpec, CategorySpec
│   ├── registry.py             # Explicit, scoped and freezable registries
│   ├── discovery.py            # Metadata-only inspection, then approved provider loading
│   ├── compatibility.py        # Contract and feature negotiation
│   └── native.py               # Public native-plugin descriptor interface, not raw secrets
├── models/                     # ModelAdapter, Model and immutable model descriptions
│   ├── sources/                # Local/checkpoint repository resolution and locks
│   ├── formats/                # Safetensors first; other formats require tested readers
│   ├── architectures/          # Architecture descriptors; no protocol-specific models
│   │   ├── qwen2.py            # First real-checkpoint target
│   │   └── blocks.py           # Reusable attention, MLP and state-block descriptions
│   └── reference.py            # Trusted public-fixture reference interfaces
├── operators/                  # Semantic operations, not their private implementation
│   ├── linear.py
│   ├── attention.py
│   ├── normalization.py
│   ├── activations.py
│   ├── indexing.py
│   └── generation.py           # Embedding, sampling, EOS and token-feedback semantics
├── numerics/                   # Frozen numerical semantics and policies
│   ├── quantization.py
│   ├── bounds.py
│   ├── rounding.py
│   └── approximation.py        # New approximation => new numeric graph identity
├── representations/            # Typed protected encodings; distinct from numeric dtype
│   ├── plaintext.py
│   ├── masked.py
│   ├── shared.py
│   ├── garbled.py
│   └── encrypted.py            # Comparison profiles only unless explicitly selected
├── protocols/                  # ProtocolMethod facades and capability declarations
│   ├── masked_linear/
│   ├── garbling/               # Arithmetic, half-gates, LUTs and weighted-path candidates
│   ├── secret_sharing/
│   ├── function_sharing/
│   └── homomorphic/            # Optional research comparison; not the preferred profile
├── conversions/                # Explicit protected representation changes
│   ├── arithmetic_boolean.py
│   ├── residues.py             # CRT, base extension and signed interpretation
│   ├── rescaling.py
│   └── codecs.py               # Representation-specific transport encodings
├── preparation/                # PreparationProvider and typed material requests
│   ├── correlations.py
│   ├── garbling.py
│   └── material.py             # Reusable / one-use / immutable-operand declarations
├── kernels/                    # KernelBackend selectors; all computation is native
│   ├── cpu.py
│   ├── cuda.py                 # Optional; advertised only for tested builds
│   └── metal.py                # Independent optional capability, not implied by CUDA
├── compiler/                   # Compiler API; native lowering/analysis/search primitives
│   ├── ir/                    # Semantic, numeric, protected and executable plan views
│   ├── passes/
│   ├── constraints.py
│   └── cost.py
├── pipeline/                   # Pipeline, named regions, selection rules and Experiment
│   ├── composition.py
│   └── policies.py
├── runtime/                    # Runtime, Client, Session and opaque execution handles
│   ├── state.py                # Protected KV, sampler and recurrent-state handles
│   ├── inventory.py            # Native allocation/retirement interfaces
│   └── scheduling.py           # Public policies; no per-token Python dispatch
├── deployment/                 # Placement and launching, separate from protocol choice
│   ├── topology.py
│   ├── transports/
│   └── launchers/              # Local processes first; remote agents later
├── benchmarks/                 # Benchmark, suites, workloads, reports and runner facades
│   ├── workloads/
│   ├── suites/
│   └── reports.py
├── metrics/                    # BenchmarkMetric; units and cohorts are explicit
│   ├── latency.py
│   ├── throughput.py
│   ├── traffic.py
│   ├── resources.py
│   └── quality.py
├── search/                     # Study and immutable public search spaces
│   ├── grid.py
│   └── pareto.py
├── assurance/                  # PrivacyContract, scoped findings; not a privacy score
│   ├── contracts.py
│   ├── attacks/
│   ├── proofs/
│   └── findings.py
├── research/                   # Sources, recipes, provenance and reproduction evidence
│   ├── sources.py
│   ├── reproductions/
│   └── artifacts.py
├── integrations/               # Optional adapters to external ecosystems
│   ├── responses/              # Trusted client-side API gateway
│   ├── opentelemetry/
│   └── deepeval/               # Public result adapter; no implicit prompt export
├── testing/                    # Public community conformance toolkit
│   ├── component_checks.py
│   ├── composition_checks.py
│   ├── model_checks.py
│   ├── native_checks.py
│   └── docs_checks.py
├── _cli/                       # Private implementation of `pllm` commands
├── _internal/                  # Private bootstrap, migration adapters, no plugin imports
└── _native.*                   # PyO3 extension; not the external plugin API
```

Canonical public imports are from domain facades, for example `pllm.models.ModelAdapter`, `pllm.protocols.ProtocolMethod`, `pllm.kernels.KernelBackend`, `pllm.metrics.BenchmarkMetric`, and `pllm.testing.check_component`. `pllm.components` is for extension contracts, not the only place where users find every algorithm.

A paper may contribute several components in different packages. Dash-related arithmetic belongs in `protocols.garbling`; its CPU implementation belongs in `kernels`; a reproduction recipe belongs in `research.reproductions`. An experimental weighted path remains in `protocols.garbling` with an experimental status, rather than changing import paths when promoted.

## 3. Dependency rules

```text
types
  ↑
components + config
  ↑
model / operator / numeric / representation / protocol / conversion descriptors
  ↑
compiler + pipeline
  ↑
runtime + deployment
  ↑
benchmarks + search + integrations
```

This illustrates permitted layers, not every edge. Compiler/runtime operate on registry specifications and contracts, not imports of named protocol implementations. Models describe semantics and do not import protocol families. Kernels do not know model names. The registry must not import benchmarks or every installed provider during `import pllm`.

`assurance` and `research` attach contracts and provenance through shared types; the runtime does not need network research retrieval to execute a locked plan. `metrics` consume typed results and native event outputs. Heavy metric reductions can also be native; Python must not own timed execution.

No generic `utils` package, global mutable singleton registry, giant root export list, global category enum, implicit remote code execution, or import-time device/model discovery.

## 4. User object lifecycle

Separate three things that otherwise invite accidental secret reuse:

- **Component configuration:** immutable, inspectable, cloneable, JSON-safe; `get_params(deep=True)`, `with_params(...)`, `to_spec()`.
- **Compiled public plan:** immutable and content-addressed; all auto choices resolved and all provider versions locked.
- **Execution session/material:** opaque native handles, fresh, non-cloneable and non-pickleable; explicit close/retire semantics.

Nested selection uses named regions and paths such as `mlp__activation__backend`. Clone copies configuration only. It must never duplicate randomness, material, buffers, outstanding requests or active state. `with_params` returns a new configuration and invalidates incompatible plans; do not mutate an active session with sklearn-style `set_params`.

Python constructors do not download weights, start servers, compile cryptographic material or initialize a GPU. Cheap configuration validation is permitted; expensive validation belongs at resolution/compile time. The public sequence is **configure → resolve → validate → compile → prepare → execute → measure**.

Pipeline is a configuration/composition object for a graph. It is not a guarantee that arbitrary sequential components compose. The native compiler resolves operations, representations, placement and state effects into the per-party DAG.

## 5. Native repository map

```text
crates/
  pllm-types/          # Canonical shared representations and schema models
  pllm-components/     # Native contracts, category handlers and compatibility checks
  pllm-models/         # Tensor stores, parsers, validation, quantization data processing
  pllm-numerics/       # Exact rings/fields, CRT and numeric certificates
  pllm-compiler/       # Typed IRs, analyses, lowering and regional plan selection
  pllm-protocols/      # Protocol families and explicit conversions
  pllm-preparation/    # Providers and fresh-material generation
  pllm-kernels/        # Portable/SIMD/device execution; FFI to reviewed native libraries
  pllm-runtime/        # Party executor, inventory, state, scheduler and cancellation
  pllm-transport/      # Authenticated channels, framing and transport counters
  pllm-bench/          # Native clocks, launch supervision, measurements and reductions
  pllm-assurance/      # View-restricted attacks, trace/model checks and evidence hooks
  pllm-plugin-api/     # Versioned C ABI headers/vtables and Rust safe wrappers
  pllm-python/         # Thin PyO3 extension: pllm._native
```

These domains need not become separate crates immediately. Preserve the current `pllm-core` boundary until splitting is justified. Python module granularity and Rust crate granularity need not be identical.

The native host receives an entire compiled region or execution plan. Python must not provide per-gate/per-label/per-packet callbacks on optimized paths. Approved native providers are loaded into the relevant role process only. Maturin packages the Python facade and native artifact; it does not define the plugin ABI. See S6–S8.

## 6. Adding model architectures

Architecture is orthogonal to source format and private protocol:

1. Add or reuse a source resolver and a native checkpoint reader.
2. Define tensor mappings, aliases, shape constraints, positional conventions, normalization, attention and generation state in a `ModelAdapter`.
3. Emit semantic operations using reusable blocks, not special cases inside a protocol backend.
4. If a genuinely new operation/state effect exists, register its semantics and native execution/lowering support separately. Until then, report unsupported coverage.
5. Add architecture conformance fixtures for prefill, incremental decode, cache growth, positions, masks, tied parameters, numeric reference agreement and protected execution.
6. Lock the checkpoint, tokenizer, adapter version and semantic/numeric graph digests.

Small public metadata can be inspected in Python. Weight scanning, format decoding, transformations and numerical work belong in Rust. Architecture declaration callbacks run only during configuration/compilation and are trusted plugin code. `trust_remote_code` remains off by default.

Support is a matrix of format × architecture features × numeric semantics × protocol × device × workload. A hypothetical future recurrent or expert-routed model does not become supported simply by registering its name. Routing indices and data-dependent state access require a corresponding privacy protocol.

## 7. Migration from the uplift package

Keep the existing `pllm` public imports as re-exporting facades; do not shadow them with a second installed runtime. Merge the temporary `pllm_uplift` work through parity-tested native modules. Map prior `methods` identifiers to protocol/category descriptors; retain old locked identifiers with explicit aliases and warnings. Preserve the recorded meaning of published artifacts.

Move Python weight-processing and benchmark-clock work into native implementations before labeling a path optimized. Preserve the original masked-linear baseline as a named profile and the single-evaluator architecture as a separate coverage-gated profile. Protocol scope, client workload and evidence must not change during a package rename.

## 8. Acceptance criteria

Importing a domain must not initialize unrelated optional dependencies. Every public export has types and documentation. Every builtin is represented through the same category contracts as a community provider. Every active plan locks concrete component artifacts. New category descriptors are discoverable without changing a global enum. Unsupported required categories, operations and conversions fail before preparation. No tests, conformance results or documentation badges imply a stronger privacy claim than their recorded scope.
