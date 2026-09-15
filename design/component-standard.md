# Component standard

Status: PLLM Component Standard 1 design contract. Conformance is not security certification.

This standard covers built-in and separately distributed components. Packaging and source placement
are defined by the [package structure](package-structure.md); documentation and evidence display are
defined by the [documentation standard](documentation-standard.md).

## Identity and discovery

A component has separate provider, distribution, component, version, category, method, artifact,
and source identities. Stable component IDs use an authority-owned namespace such as
`org.example/component`; strings and signatures do not establish publisher trust by themselves.
Method identity describes semantics; component identity describes one implementation. Equivalent
method semantics do not merge implementation identity, artifacts, evidence, or trust decisions.

Community distributions MUST use their own import package, declare one versioned provider entry
point, and expose a static manifest. Discovery reads package metadata and inert JSON only. It MUST
NOT import provider code, load native libraries, fetch models, auto-install dependencies, or run
configuration expressions. Duplicate identities fail unless explicit policy selects an alias.

Loading a Python factory or native library executes trusted code. Approved providers are loaded
only into authorized roles. In-process plugins join that role's trusted computing base; manifests
and capability checks are not sandboxes. Unknown required contracts and versions fail closed.

For PCS 1, provider entry-point group is `pllm.providers.v1`; one provider MAY contribute many
categories and components. Its package-relative `pllm-plugin.json` lists descriptors, schemas, docs,
native artifacts, supported host/category versions, and tested build combinations. Paths MUST be
normalized and confined to the distribution. Editable installs require explicit development policy
and MUST NOT silently count as release-conformant artifacts. Discovery never auto-installs or
fetches provider dependencies.

## Configuration contract

Every component exposes immutable JSON-safe public parameters and supports these semantics:

- `describe()` returns a static descriptor without device or secret state.
- `get_params(deep=True)` returns named public parameters.
- `with_params(**changes)` returns a new configuration and rejects unknown paths.
- `to_spec()` emits a canonical public reference.

Configuration clones contain no secrets or fresh state. `with_params()` MUST NOT mutate an active
plan or session and MUST reject unknown, overlapping, or ill-typed paths. A descriptor contains its
identities, category contract, lifecycle phase, public parameter schema, capabilities, required host
features, role eligibility, artifacts, and evidence pointers. Descriptors advertise support; a
compiler check establishes whether that support applies to one plan.

Categories define behavior. Initial categories include model adapter, protocol method, conversion,
kernel backend, preparation provider, compiler pass, benchmark metric, adversary, and proof
checker. Categories are versioned namespaced contracts, not a closed global enum. New executable
semantics require an approved host handler or lowering to supported IR.

A category specification owns its parameter and manifest schemas, lifecycle phase, interface ID,
compatibility rules, host capabilities, conformance suite, and documentation section. Unknown
optional descriptive annotations MAY be retained as inert namespaced data. Security-critical or
state-changing behavior MUST NOT be hidden in optional annotations.

Category interfaces remain narrow:

- Model adapters validate model configuration and tensor schemas, emit semantic operators and state,
  and identify conformance fixtures.
- Protocol methods state equations and assumptions and provide typed lowering and preparation needs.
- Conversions define exact source/destination representations, parties, arithmetic relation, error,
  leakage, epoch, and lifetime changes.
- Kernels declare domain, layout, device, exactness, workspace, and native implementation support.
- Preparation providers produce typed fresh material, receipts, role slices, and consumption rules.
- Compiler passes declare required IR, analyses, effects, and preserved properties.
- Metrics declare event schema, units, aggregation, and comparison cohort.
- Adversaries declare corrupted view, goal, resource budget, and test implementation.
- Proof checkers identify formal model, assumptions, checker, and refinement boundary.

Python references are permitted in explicitly labelled test and research paths. Optimized execution,
preparation, tracing, and benchmark timing MUST NOT depend on Python callbacks.

## Composition

Before preparation, compiler MUST validate operation and numeric semantics; input/output
representations; explicit conversions; roles and corruption sets; leakage; active or semi-honest
assumptions; state/material lifetime; kernel/device/resource support; implementation coverage; and
profile evidence requirements. Equal shape, dtype, or label name is insufficient compatibility.

Compatibility includes independent API/category versions; exact rounding, scale, range, clipping,
and approximation; representation modulus, layout, authentication, holders, and epoch; role and
online/offline behavior; traffic and output leakage; retry, cancellation, rollback, and one-use
lifetime; native ABI and wire versions; model-feature coverage; and required evidence. Unknown
required fields fail. A provider MUST NOT use a generic `compatible` Boolean in place of these
contracts.

Compatibility reports separate accepted constraints, failures, assumptions, and unresolved proof
obligations. Passing necessary structural checks is not proof of secure composition.

## Native plugin ABI

PLLM Native Plugin ABI 1 is independently versioned and C-compatible:

- One versioned entry symbol returns a size/version-tagged vtable.
- Scalars use explicit-width integers; descriptors and errors use canonical byte records.
- Buffers, material, events, and sessions use opaque generational handles with checked metadata.
- Allocation/free ownership, borrowing, stream ordering, cancellation, and lifetime are explicit.
- Panics or exceptions never cross ABI; failures return typed status.
- Wire protocol versions remain independent from plugin ABI versions.

Rust `Vec`, `String`, trait objects, allocator-owned containers, and unwinding MUST NOT cross a
dynamic-library boundary. Optimized calls operate on whole regions or batches. Process isolation
is required when deployment policy does not trust plugin code with role secrets.

The host validates role, domain, device, shape, owner, mutability, generation, and lifetime metadata
on opaque handles. Async completion and device synchronization are explicit. Shutdown and
cancellation retire material before releasing handles. A native artifact's path and digest are
frozen in the plan lock; PyO3 ABI compatibility does not establish plugin ABI compatibility.

## Versioning and artifacts

Host Python API, component standard, category contract, method semantics, representation, plugin
ABI, wire protocol, evidence schema, and docs schema version independently. Providers declare tested
intersections rather than one broad compatibility number. Semantic changes to rounding, leakage,
failure behavior, or material reuse require a new relevant identity and migration.

Plans lock component and method identities, versions, public parameters, source and native artifact
digests, required features, model/tokenizer and numeric graph identities, role mapping, and evidence
requirements. Replay MUST NOT float to a newer provider. Upstreaming preserves old identities,
attribution, explicit aliases, and prior artifact resolution; evidence transfers only with recorded
validation or refinement.

## Conformance and promotion

Conformance records independently report interface behavior, numerical scope, implementation
coverage, performance evidence, assurance findings, tested build matrix, and source fidelity.
Allowed assurance outcomes are `proved_in_model`, `refuted_in_scope`, `not_refuted`,
`inconclusive`, `outside_contract`, and `unchecked`. No scalar privacy score or secure boolean is
valid. Core promotion additionally requires ownership, licensing, native support, applicable
review, docs, and explicit migration; evidence never transfers automatically between ports.

Category suites additionally cover schema round trips, immutable cloning, wrong-field rejection,
conversion and role failures, numerical parity, cancellation and rollback, lock resolution, ABI
failure handling, public imports, docs completeness, and source attribution. Tests unavailable in a
declared scope remain unavailable, not skipped into a pass. Built-ins receive no conformance or
eligibility exemption.
