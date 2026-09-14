# PLLM Component Compatibility Standard — PCS 1 (draft)

**Status: proposed interoperability contract, not a security certification or implemented native ABI.** PCS versioning is independent of the PLLM Python distribution's release number.

## 1. Scope

A separately published package should be able to contribute a model adapter, numerical policy, protocol, conversion, kernel, preparation provider, compiler pass, metric, adversary or new category. The host must be able to discover, configure, validate, compile, benchmark and document that contribution without copying its code into PLLM.

Interoperability has distinct axes: Python/configuration API; category contract; semantic/numeric meaning; protected representation; party/security policy; native ABI; peer wire protocol; model feature coverage; and evidence scope. A pass on one axis is not a pass on the others.

The words MUST, MUST NOT and SHOULD below express proposed requirements for a PCS-1-conforming release.

## 2. Identity and packaging

Core keeps ownership of the ordinary `pllm` Python package. Community packages MUST use their own distribution and import names, for example distribution `pllm-acme-lut`, import `pllm_acme_lut`. They MUST NOT install files inside `pllm`, monkey-patch its private modules, or claim first-party status through a name. This avoids shared-package collisions; PyPA documents both metadata discovery and the risks of main-package namespace extension [S3–S4].

A component has distinct identities:

| Identity | Example | Meaning |
|---|---|---|
| Provider | `org.example.lab` | Publisher namespace, verified separately from this string |
| Distribution | `pllm-acme-lut` | Python wheel installation unit |
| Component | `org.example.lab/weighted-path` | Stable implementation identity |
| Component version | `0.1.0` | Version of that implementation |
| Category | `pllm.protocol_method.v1` | Versioned interface and capability contract |
| Method | `pllm.method.weighted_path.v1` | Algorithm/semantics being implemented, not proof of faithful reproduction |
| Python symbol | `pllm_acme_lut.WeightedPathGate` | User-facing import |
| Artifact digest | SHA-256 in a resolved lock | Exact installed implementation being executed |

Namespace ownership MUST be established by an approved catalogue or explicit user policy. A reverse-domain-looking string, wheel hash or signature alone proves neither expertise nor correctness.

Use ONE versioned provider entry-point group:

```toml
[project.entry-points."pllm_inference.providers.v1"]
acme = "pllm_acme_lut.plugin:get_provider"
```

The group uses the normalized distribution name. The entry point is a provider, which may describe multiple categories and components. Do not create a required new entry-point group for every future category. The usual PyPA entry-point mechanism is sufficient; PCS defines the object it returns [S3–S5].

For PCS 1, each provider distribution has one ordinary top-level Python package and includes `<top_level_package>/pllm-plugin.json`. The top-level package is obtained from the entry-point module reference, without importing it. Wheel resources, docs, schemas and native artifacts are listed by distribution metadata. Paths must be normalized, package-relative and confined to the distribution; reject traversal and symlink escapes. An editable installation without verifiable file metadata requires an explicit development manifest and cannot silently count as release conformance.

Production distributions MUST declare supported host package versions in Python metadata, supported PCS/category versions in the manifest and tested runtime/native build combinations in conformance records. Do not invent an untested runtime-version range. The example manifest in this bundle is a design specimen, not a releasable wheel.

## 3. Discovery is not loading

Discovery reads entry-point metadata and the static JSON manifest. It MUST NOT call `EntryPoint.load()`, import the provider, run a model loader, evaluate a config expression or load a native library merely to list components. `importlib.metadata` exposes both entry-point data and distribution file metadata [S5].

The lifecycle is:

```text
inspect static metadata
  → validate schema, dependencies and required features
  → approve provider and execution roles
  → explicitly load its configuration factory
  → validate requested composition
  → resolve and verify exact native artifact
  → freeze the role-specific plan
  → execute inside the authorized role
```

Calling the approved factory or loading a library executes trusted code. A manifest is NOT a sandbox. In-process native/Python plugins become part of that role's trusted computing base. Isolation is a separate deployment property. Do not load every provider into every party or give the coordinator all role secrets.

Duplicate component ID/version claims fail unless an explicit authority-approved alias is selected. No import-order winner, implicit override, network auto-install or silent replacement of built-ins. Discovery may show unavailable components and their unmet dependencies; execution fails when required support is absent.

## 4. Small base contract, category-specific interfaces

`BaseComponent` is a convenience facade over a structural contract, not a mandatory inheritance hierarchy. Its configuration surface is deliberately small:

| Method | Contract |
|---|---|
| `describe()` | Return a static, JSON-safe `ComponentSpec`; no device or secret state |
| `get_params(deep=True)` | Return immutable configuration parameters and nested named parameters |
| `with_params(**changes)` | Return a NEW configuration; reject unknown parameters |
| `to_spec()` | Return canonical public component reference plus parameters |

Category interfaces define actual behavior; a metric does not implement garbling, and a model adapter does not pretend to have a native matmul method.

| Category interface | Required responsibilities |
|---|---|
| `ModelAdapter` | Inspect public config, validate tensor schema, lower semantic operations, describe state and fixtures |
| `ProtocolMethod` | State preconditions/assumptions; validate region; provide native lowering and preparation references |
| `Conversion` | Define source/destination encodings, arithmetic relation, parties, errors and lifetime changes |
| `KernelBackend` | Describe domain/layout/device support; expose native implementation and workspace requirements |
| `PreparationProvider` | Specify typed fresh-material requests, parties, consumption and receipts |
| `CompilerPass` | Declare IR version, analyses needed, preserved properties and native pass implementation |
| `BenchmarkMetric` | Declare input event schema, units, aggregation, comparison cohort and evidence |
| `Adversary` | Declare allowed corrupted view, goal, resource budget and native/test implementation |
| `ProofChecker` | Identify formal model, assumptions, checker and implementation-refinement boundary |

Python factories may assemble declarations and call native entry points at coarse boundaries. Performance-critical tensor processing, protocol preparation/execution, live trace collection and benchmark timing MUST NOT use Python callbacks. Pure-Python numerical references are allowed only in explicitly labelled test/research paths, never as a hidden optimized fallback.

A component package need not supply every layer. A protocol can select compatible host kernels; a kernel can accelerate multiple protocols. Dependencies refer to capabilities and specific contracts, not imports of another provider's private class.

## 5. Extensible categories

A `CategorySpec` contains a namespaced `id`, its own contract version, public parameter/manifest schemas, lifecycle phase, interface identifier, host feature requirements, compatibility rules, conformance-suite ID and documentation section.

For example `org.example.lab.cache_predictor.v1` could be registered under `runtime.state` without adding a new top-level Python package. The host can index and document its metadata generically.

Execution is NOT magical. A new category is executable only if its approved handler lowers to supported IR operations/native capabilities, or the host explicitly implements its new semantics. Unknown REQUIRED category contracts, state effects, encodings or features fail closed. Unknown optional descriptive annotations may be retained without execution effects. Arbitrary predicates, class paths or remote JSON-schema references in unapproved metadata MUST NOT execute.

A plugin may register several cooperating categories through the one provider. Categories are not a global enum and are not inferred from the directory tree. The `category_catalog.json` file defines the proposed initial core categories; it is data, not a closed universe.

## 6. Composition contract

Before preparing any material, the compiler checks all of:

1. Operation identity and semantic/numerical graph definition: quantization, rounding, accumulator range, clipping, approximation and source bounds.
2. Exact input/output protection contracts, including modulus, layout, label scheme, authentication scheme, owners, garbling epoch and permitted algebraic relations.
3. Explicit eligible conversion whenever representations differ. Equal shape or equal dtype is insufficient.
4. Party roles, administrative trust domains, online/offline phase behavior and permitted collusions.
5. View/leakage policy including traffic, lengths, routing, timing and output release.
6. Active vs semi-honest assumptions, model binding and applicable proof/composition obligations.
7. One-use, bounded-use or justified immutable-operand lifetimes; cloning, retry, cancellation and state effects.
8. Native kernel availability, hardware features, workspace and total preparation budgets.
9. Implementation coverage and assurance evidence required by the selected deployment profile.

Required contracts are structural, versioned schemas with host-defined semantics, not a single `compatible=True` tag. For example, matching `arithmetic_label` strings does not establish compatible label offsets, garbling construction, field, epoch or proof assumptions.

The host produces a compatibility report with accepted constraints, failed constraints and unresolved obligations. Passing the report does not prove a protocol secure. An assumption accepted by deployment policy remains an assumption.

Unknown required fields and versions fail. Optional descriptive extension fields are namespaced, inert and preserved. Security-critical requirements cannot be smuggled into ignorable annotations.

## 7. Native provider ABI

Separately built community Rust libraries MUST NOT exchange Rust trait objects, `Vec`, `String`, allocator-owned containers or Rust exceptions/panics across a dynamic-library boundary. Rust's default ABI does not provide stability guarantees [S8]. PyO3's Python ABI compatibility does not establish a PLLM plugin ABI [S7].

Define an independently versioned **PLLM Native Plugin ABI 1** in `pllm-plugin-api`:

- A versioned C entry symbol (`pllm_plugin_v1`) returns a length/version-tagged C vtable.
- Descriptors and errors use canonical owned byte representations; scalar FFI fields use explicit-width integers.
- The host validates semantic/native feature requirements before any execution.
- Operations use opaque generational buffer/material handles, with role, domain, device, shape, owner and immutable/mutable borrow metadata checked by the host.
- The host owns allocation or an explicit paired allocation/free interface. No cross-library deallocation assumptions.
- Async operations return completion/event handles; stream ordering, cancellation, buffer lifetime and device synchronization are explicit.
- No unwinding crosses the C boundary. Native failures return typed statuses; crash isolation requires a process boundary.
- Wire framing and protocol versions are separate from plugin ABI versions.
- `shutdown` and cancellation retire material safely and release approved handles; plugin lifetime must cover outstanding callbacks/events.

The optimized executor loads and calls the native vtable directly for whole regions or batched operations. Python is used only for discovery/configuration and result presentation. Do not route each native operation through Python.

Built-in source-linked Rust implementations can use Rust traits inside one build. Separately compiled dynamic plugins use the C boundary. A community wheel may include both its Python facade and native artifact via maturin, but the artifact path and hash must be resolved in the lock. Reviewed native dependencies and accelerator FFI are permitted; Rust is the ownership/execution boundary, not a prohibition on CUDA or existing cryptographic code.

A native plugin in the same process can violate its declared permissions if malicious. Capability checks are useful invariants, NOT memory isolation. Only explicitly trusted code is loaded into a party with secrets. A restricted out-of-process provider is a different performance/deployment mode; it does not change which parties are non-colluding.

## 8. Versions, parameters and artifacts

Version independently: host Python API, PCS manifest, category contract, semantic operation, protection representation, native ABI, wire protocol, evidence schema and docs schema. Component package releases declare tested intersections rather than relying on one giant version number.

Configuration clones contain no secrets. Fresh state is absent from public manifests and cache keys. A resolved plan records component IDs/versions, source/native artifact hashes, public parameters, model/tokenizer hashes, numeric graph, role mapping, required features, assurance requirements and a docs build identifier. Plans never silently resolve to newer community components during a replay.

Use explicit migrations. Semantic changes, including rounding, failure leakage or one-use behavior, cannot be disguised as a compatible kernel update. Patch security fixes may intentionally reject formerly accepted unsafe inputs; release notes and affected-evidence records explain the change.

## 9. Conformance and assurance

`pllm.testing` supplies category-driven checks analogous in role to sklearn's estimator checks [S1]. Proposed checks cover import independence, immutable parameter cloning, schema round trips, wrong-field rejection, conversion/role compatibility, numerical parity, cancellation, plan locking, native ABI failures, docs completeness and source attribution.

A protocol needs view-restricted attack fixtures and an explicit proof boundary. A model needs prefill/decode and state conformance. A kernel needs exactness and overflow/property tests across its advertised layouts. A metric needs units and cohort tests. A native provider needs hardware/build-matrix tests.

Record results separately: `interface_conformant`, tested composition scope, numerical correctness scope, implementation coverage, performance evidence and assurance finding. Findings retain `proved_in_model`, `refuted_in_scope`, `not_refuted`, `inconclusive`, `outside_contract`, `unchecked`. There is no single certified-private badge. Failed/unavailable checks are not silently skipped into a pass.

Community code is not auto-eligible for production because it passes interface tests. Eligibility is a deployment policy based on complete coverage and applicable evidence. Built-ins obey the same rule.

## 10. Upstreaming

A community implementation may keep its public import while delegating to an upstreamed implementation. Preserve attribution and paper-reproduction distinctions. Add an explicit alias/migration, not a silent change of algorithm behind an existing ID.

Previously locked artifacts keep resolving to their recorded versions and hashes. Equivalent semantics can share a method ID; different implementations still retain distinct component identities. Evidence is not inherited across a port unless the recorded validation/refinement justifies it.

Acceptance into core requires maintainers, licensing/dependency review, category tests, semantic and privacy review appropriate to scope, supported native builds, documentation and a migration story. Popularity or a faster microbenchmark is insufficient.
