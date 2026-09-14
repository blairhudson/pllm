**Organize PLLM by what components do—not by paper, provider, party or implementation language.** Keep a small shared component contract underneath those packages, and mirror the public structure into the documentation.

That preserves the existing Python/PyO3 foundation while making Rust the execution layer and Python the developer-facing interface. Pasted text.txtTXT START_HERE.mdMD

**PLLM_package_ecosystem_design.zipZIP**

The bundle contains the full package tree, proposed community compatibility standard, documentation contract, category catalogue and example manifests. These are proposed interfaces, not claims that the modules already exist.

## 1. Recommended Python package structure

The useful patterns to borrow are sklearn’s named components, parameter inspection and shared conformance checks, plus DeepEval’s straightforward domain-specific imports and extension interfaces. We should not copy training-oriented methods such as `fit()` into an inference compiler. [scikit-learn](https://scikit-learn.org/stable/developers/develop.html)

```
pllm/
├── __init__.py                 # Model, Pipeline, Experiment, Runtime, Benchmark, Study
├── py.typed
│
├── config/                     # Immutable configuration, profiles and migrations
│   ├── profiles.py
│   └── serialization.py
│
├── types/                      # Small, stable vocabulary shared across components
│   ├── numeric.py              # Domains, scales, rounding and range certificates
│   ├── protection.py           # Encodings, ownership and permitted views
│   ├── topology.py             # Parties, trust domains and phase constraints
│   ├── state.py                # Epochs, effects and resource lifetimes
│   ├── plans.py                # IR/plan descriptors and opaque handles
│   ├── evidence.py             # Correctness, assurance and measurement records
│   └── errors.py
│
├── components/                 # Common extension and discovery mechanism
│   ├── base.py                 # BaseComponent, ComponentRef
│   ├── specs.py                # ComponentSpec, ProviderSpec, CategorySpec
│   ├── registry.py
│   ├── discovery.py
│   ├── compatibility.py
│   └── native.py               # Public native-plugin interface
│
├── models/                     # Model, ModelAdapter
│   ├── sources/                # Local/repository resolution and checkpoint locks
│   ├── formats/                # Safetensors and other explicitly supported formats
│   ├── architectures/
│   │   ├── qwen2.py            # First real-checkpoint target
│   │   └── blocks.py           # Reusable model-block descriptions
│   └── reference.py
│
├── operators/                  # Mathematical operations, independent of privacy method
│   ├── linear.py
│   ├── attention.py
│   ├── normalization.py
│   ├── activations.py
│   ├── indexing.py
│   └── generation.py
│
├── numerics/                   # Definition of the numerical graph
│   ├── quantization.py
│   ├── bounds.py
│   ├── rounding.py
│   └── approximation.py
│
├── representations/            # Protection is separate from numerical dtype
│   ├── plaintext.py
│   ├── masked.py
│   ├── shared.py
│   ├── garbled.py
│   └── encrypted.py
│
├── protocols/                  # Protocol methods, grouped by mathematical family
│   ├── masked_linear/
│   ├── garbling/               # Arithmetic, half-gates, LUTs, weighted paths
│   ├── secret_sharing/
│   ├── function_sharing/
│   └── homomorphic/
│
├── conversions/                # Explicit changes between protected representations
│   ├── arithmetic_boolean.py
│   ├── residues.py
│   ├── rescaling.py
│   └── codecs.py
│
├── preparation/                # Offline providers and typed material requirements
│   ├── correlations.py
│   ├── garbling.py
│   └── material.py
│
├── kernels/                    # Public selectors; computation happens in Rust/native code
│   ├── cpu.py
│   ├── cuda.py
│   └── metal.py
│
├── compiler/                   # Compiler API and native-pass configuration
│   ├── ir/
│   ├── passes/
│   ├── constraints.py
│   └── cost.py
│
├── pipeline/                   # Composition, named regions and experiments
│   ├── composition.py
│   └── policies.py
│
├── runtime/                    # Runtime, Client, Session
│   ├── state.py                # Protected KV, sampler and recurrent-state handles
│   ├── inventory.py
│   └── scheduling.py
│
├── deployment/                 # Placement and launching—not protocol selection
│   ├── topology.py
│   ├── transports/
│   └── launchers/
│
├── benchmarks/                 # Benchmark orchestration and result interfaces
│   ├── workloads/
│   ├── suites/
│   └── reports.py
│
├── metrics/
│   ├── latency.py
│   ├── throughput.py
│   ├── traffic.py
│   ├── resources.py
│   └── quality.py
│
├── search/                     # Study, search spaces and constrained optimization
│   ├── grid.py
│   └── pareto.py
│
├── assurance/                  # Privacy contracts and scoped findings
│   ├── contracts.py
│   ├── attacks/
│   ├── proofs/
│   └── findings.py
│
├── research/                   # Sources and reproducibility—not algorithm implementations
│   ├── sources.py
│   ├── reproductions/
│   └── artifacts.py
│
├── integrations/               # Optional external integrations
│   ├── responses/              # Trusted client-side gateway
│   ├── opentelemetry/
│   └── deepeval/
│
├── testing/                    # Public toolkit for core and community conformance
│   ├── component_checks.py
│   ├── composition_checks.py
│   ├── model_checks.py
│   ├── native_checks.py
│   └── docs_checks.py
│
├── _cli/                       # Private CLI implementation
├── _internal/                  # Private bootstrap and migration adapters
└── _native.*                   # Private PyO3 extension
```

This is the intended structure, not a requirement to create empty packages immediately. Public exports should come from package facades; private implementation files can change without breaking imports.

### Important boundaries


| Distinction                           | Why it matters                                                                         |
| ------------------------------------- | -------------------------------------------------------------------------------------- |
| **Operator vs protocol**              | SiLU’s mathematical definition is not its garbled implementation.                      |
| **Numerics vs representation**        | An integer’s scale and rounding are not its masking or label scheme.                   |
| **Protocol vs kernel**                | One protocol can use several CPU/GPU implementations.                                  |
| **Preparation vs runtime inventory**  | Generating material and enforcing its safe consumption are different responsibilities. |
| **Metric vs assurance finding**       | A measured latency is a number; “privacy not refuted” is a scoped finding.             |
| **Research record vs implementation** | A paper can contribute several components across different packages.                   |


For example, Dash-related arithmetic belongs under `protocols.garbling`, its native kernels under `kernels`, and its reproduction recipe under `research.reproductions`.

**Do not move a component between import paths when it stops being experimental.** Keep its semantic home and change its maturity/evidence metadata.

## 2. Familiar configuration objects; separate live execution state

The public object lifecycle should be:

```
configure → resolve → validate → compile → prepare → execute → measure
```

Three distinct objects prevent accidental reuse:


| Object                      | Properties                                                          |
| --------------------------- | ------------------------------------------------------------------- |
| **Component configuration** | Immutable, inspectable, cloneable and JSON-safe                     |
| **Compiled plan**           | Immutable, content-addressed and locked to concrete implementations |
| **Session/material**        | Opaque native state; non-cloneable and non-pickleable               |


Configuration objects should support:

```
describe()
get_params(deep=True)
with_params(**changes)
to_spec()
```

Use named parameter paths such as:

```
mlp__activation__backend
attention__scores__kernel
preparation__batch_size
```

`with_params()` returns a new configuration. It must not mutate an active session or copy its randomness.

**Keep** `Pipeline` **as the friendly composition object, but compile it into a graph.** Do not force branching, attention, state updates and multi-party messaging into a sequential sklearn-style execution model.

## 3. Community compatibility: one provider mechanism, many categories

I would formalize **PLLM Component Compatibility Standard — PCS 1**.

### Separate package names; shared contracts

A community contribution should look like:

```
Distribution:  pllm-acme-lut
Python import: pllm_acme_lut
Component ID:  org.example.lab/weighted-path
Category:      pllm.protocol_method.v1
```

It should **not** install files into `pllm/` or monkey-patch private modules.

Use Python’s standard entry-point metadata for discovery. PyPA explicitly supports this mechanism and warns about extending the main package namespace for plugins. [Python Packaging](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/)

```
[project.entry-points."pllm_inference.providers.v1"]
acme = "pllm_acme_lut.plugin:get_provider"
```

**One provider can contribute multiple components and new categories.** Avoid a hard-coded entry-point group for every category that might ever exist.

### Discover metadata before executing code

Each provider ships a static `pllm-plugin.json` containing its components, requirements, schemas, documentation and native-artifact declarations.

Discovery reads that file through distribution metadata. It does not import the plugin just to list it; Python exposes entry-point and installed-file metadata separately from loading the referenced object. [Python documentation](https://docs.python.org/3/library/importlib.metadata.html)

The sequence is:

```
inspect → check requirements → approve provider → load factory
        → validate composition → lock native artifact → execute
```

Loading a plugin executes trusted code. A manifest, signature or compatibility badge is **not a sandbox**.

### Small common interface, category-specific behavior

Do not make every extension inherit an enormous `BaseComponent` with irrelevant methods.


| Category             | Specialized contract                                                    |
| -------------------- | ----------------------------------------------------------------------- |
| Model adapter        | Validate checkpoint structure, emit semantic operations and state       |
| Protocol method      | Declare assumptions, validate a region, provide lowering/preparation    |
| Conversion           | Specify the exact relationship between source and destination encodings |
| Kernel               | Advertise numerical/layout/device support and a native implementation   |
| Preparation provider | Produce typed material with explicit lifetimes                          |
| Compiler pass        | Declare required IR, preserved properties and implementation            |
| Metric               | Declare input events, units, aggregation and comparison cohort          |
| Adversary            | Declare the corrupted view, attack goal and resource budget             |
| Proof checker        | Identify the formal model, assumptions and implementation boundary      |


Built-ins and community components should use **the same contracts and checks**.

### New categories without redesigning the runtime

A `CategorySpec` registers:

```
category ID and contract version
parameter/manifest schemas
lifecycle phase
interface identifier
required host capabilities
compatibility rules
conformance suite
documentation section
```

A new cache-prediction category could live under `runtime.state`; it does not require another top-level package.

The distinction is crucial:

**The host can discover and document an unknown category generically. It cannot execute unknown semantics generically.**

Execution requires an approved handler that lowers to supported IR/native operations, or an explicit host extension. Unsupported required behavior fails before preparation.

## 4. Compatibility must go beyond “the Python object loads”

PCS should check these independently:


| Compatibility axis    | Required checks                                                         |
| --------------------- | ----------------------------------------------------------------------- |
| **API**               | Supported host/category versions and configuration schemas              |
| **Numerical**         | Operation, domain, scale, rounding, range and approximation             |
| **Protection**        | Encoding scheme, modulus, labels, authentication and epoch              |
| **Composition**       | Explicit conversions between incompatible representations               |
| **Deployment**        | Parties, trust domains, online/offline participation                    |
| **Privacy/integrity** | Permitted views, leakage, corruption model and proof obligations        |
| **State**             | One-use material, immutable fan-out, retries, cancellation and rollback |
| **Native/wire**       | ABI, device capabilities, message versions and buffer ownership         |
| **Evidence**          | Implemented coverage, conformance scope and applicable assurance        |


Matching `arithmetic_label` names is insufficient: two implementations may use incompatible offsets, fields, garbling constructions or epochs.

The compiler should return **accepted constraints, rejected constraints and unresolved obligations**, not one unexplained Boolean.

The previous design already requires explicit protected conversions and separates configuration authentication from execution correctness; this standard makes those requirements part of the community interface. INTERFACES.mdMD

### Native plugins need their own ABI

A community Rust package must not exchange Rust trait objects or allocator-owned containers directly with an independently built host. Rust’s default calling convention and layouts do not provide that stability guarantee. [Rust Project Goals](https://goals.rust-lang.org/2026/open-enums.html)

Define a versioned C ABI in `pllm-plugin-api`, with opaque handles, explicit-width fields, owned buffers, typed errors and completion events.

```
Community Python facade
          │ startup/configuration only
          ▼
Approved native provider
          │ versioned C ABI
          ▼
PLLM Rust executor
```

**No Python callback per gate, label, tensor tile or protocol message.** The Rust executor calls the native provider directly.

Maturin packages the facade and native library; PyO3’s Python ABI compatibility is separate from this PLLM plugin ABI. [Maturin](https://www.maturin.rs/project_layout.html)

### Upstreaming should preserve identity and evidence

An upstreamed component can gain a core import while the original package becomes a compatibility facade. Keep attribution, explicit aliases and migrations.

Previously locked experiments must still resolve their recorded artifact hashes. Do not silently replace a community implementation with a different core implementation and inherit its old benchmark or security results.

## 5. Supporting new model architectures

Keep these independent:

```
checkpoint source
    → file format
    → architecture adapter
    → semantic graph
    → numerical graph
    → protected implementations
```

A new architecture using existing operations should mostly require a tensor mapping, architecture descriptor and conformance fixtures—not changes to every protocol.

The adapter describes attention, positional encoding, normalization, parameter aliases, state transitions and generation behavior. Heavy checkpoint processing remains native.

A genuinely new operation—such as a new recurrent update or private expert-routing mechanism—requires a semantic definition and supported protected implementation. Registering the model name alone must not make it “supported”.

The conformance matrix should cover:

```
format × architecture features × numeric profile
       × protocol × device × workload
```

Prefill, incremental decode, cache growth, positional behavior and tied weights need separate fixtures. Model architecture should never be inferred from approximate tensor-name similarity.

## 6. Documentation: mirror the packages, add learning paths

The main component guide should follow the public packages. Add guided learning, cross-cutting recipes and generated API reference rather than making newcomers navigate implementation details.

```
docs/
├── start/
│   ├── installation
│   ├── first-private-request
│   └── first-local-benchmark
├── learn/
│   ├── privacy-and-threat-models
│   ├── parties-and-offline-work
│   ├── masked-linear-inference
│   ├── arithmetic-and-boolean-garbling
│   ├── numeric-semantics-and-model-quality
│   └── reading-research-and-evidence
│
├── models/
├── operators/
├── numerics/
├── representations/
├── protocols/
│   ├── masked-linear/
│   └── garbling/
│       ├── arithmetic
│       ├── half-gates
│       ├── lookup-tables
│       └── weighted-path
├── conversions/
├── preparation/
├── kernels/
├── compiler/
├── pipeline/
├── runtime/
├── deployment/
├── benchmarks/
├── metrics/
├── search/
├── assurance/
│
├── research/
│   ├── papers/
│   ├── reproductions/
│   └── experiments/
├── recipes/
├── reference/
│   ├── python/pllm/
│   ├── native/
│   ├── schemas/
│   └── cli/
├── contribute/
│   ├── component-standard
│   ├── publish-a-provider
│   ├── add-a-category
│   ├── native-plugin-abi
│   └── upstreaming
└── agents/
    ├── implementation-map
    ├── research-map
    └── reproduction-checklist
```

**Every directory should be a real overview page with child links.** Neither humans nor agents should depend on a JavaScript sidebar to discover the hierarchy.

### One document, two equivalent representations

Use this exact contract:

```
Human:
/docs/dev/protocols/garbling/weighted-path

Markdown:
/docs/dev/protocols/garbling/weighted-path.md
```

Fragments remain valid:

```
/docs/dev/models/architectures/qwen2.md#state
```

Use immutable release paths for reproducibility, with `stable` as an alias. Pages expose the resolved release and documentation build.

**Fumadocs is a good fit:** it supports processed Markdown, `.md` rewrites and hierarchical `llms.txt` generation. Custom MDX components need explicit Markdown rendering, otherwise important content can remain as JSX rather than readable text. [Fumadocs](https://www.fumadocs.dev/docs/integrations/llms)

Generate both views from:

```
Authored explanation
+ component/category metadata
+ generated API signatures
+ research and evidence records
              ↓
       One content graph
              ↓
 HTML + Markdown + JSON indexes
```

Parameters, caveats, citations, status and examples must agree. Do not maintain a separate agent-oriented rewrite that can drift from the human documentation.

### Each component page should answer the same questions

Use a consistent sequence: purpose and example; party responsibilities; mathematics; compatible inputs/outputs; parameters and conversions; material lifetimes; performance evidence; privacy assumptions; original-paper relationship; reproduction commands and source/test locations.

Keep **API stability, implementation coverage, reproduction fidelity and assurance** as separate fields.

### Give agents structured navigation—not just a huge text dump

Publish versioned indexes containing:

```
document ID, title and summary
HTML path and Markdown path
parent, children and prerequisites
Python modules and symbols
component IDs and category
source/test locations
release, build and content hash
```

Provide `llms.txt`, section indexes and a complete `index.json`. A full-text export is useful, but should not be the only discovery mechanism.

The intended navigation is:

```
Goal → recipe → component contract → source/tests → evidence
```

For example, “improve SiLU performance” should lead to the nonlinear-backend comparison recipe, its compatible encodings, the native source locations, benchmark commands and known counterexamples.

## 7. What to freeze first

**Stabilize the shared types, component specification, provider discovery, native boundary and documentation metadata before expanding the algorithm catalogue.**

Everything else can evolve behind those contracts. That gives community contributors a stable integration target, lets the compiler combine compatible work, and keeps research evidence attached to the implementation that produced it.

The downloadable design includes **24 initial component categories**, a **64-node example documentation graph**, and **17 passing structural checks** for discovery records, navigation consistency and `.md` route handling. Those checks validate the design specimens—not an installed plugin, native ABI or deployed documentation site.

The three detailed specifications are , and .
