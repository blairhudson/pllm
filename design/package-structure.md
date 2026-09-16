# Package and repository structure

## Status and organizing rule

This document is the canonical placement and dependency contract. Trees describe destinations when
implementation exists; they do not require empty packages and do not claim support from a path's
presence.

Code is organized by semantic responsibility, not paper, provider, maturity, role placement, or
implementation language. A method may contribute model, protocol, conversion, kernel, and research
records in different semantic homes. Promotion changes status and evidence metadata, not import
paths or component identity.

## Repository boundaries

```text
Cargo.toml, Cargo.lock            Rust workspace and reproducible dependency lock
pyproject.toml, uv.lock           `pllm` distribution and Python dependency lock
python/pllm/                      Python public facades and orchestration
crates/                           Rust-owned compiler, execution, benchmark, and assurance code
schemas/                          Canonical public serialized contracts and fixtures
design/                           Normative product and engineering contracts
docs/                             Rendered user/developer documentation source and outputs
research/                         Sources, methods, recipes, assurance models, and evidence
examples/                         Tested examples; never normative by themselves
tests/                            Python, integration, packaging, and contract tests
scripts/                          Repository, release, docs, and validation tooling
deploy/, infra/                   Deployment templates and infrastructure policy
verification/                    Implementation verification assets when present
paper/                            Manuscript source; claims remain evidence-scoped
```

`design/` governs intended invariants. `schemas/` governs accepted machine shapes for its version.
`docs/` explains supported use. `research/` records sources, reproduction work, and evidence. None of
these directories may silently substitute for another: a research recipe is not runtime code, an
example is not a schema, and a schema's presence is not implementation evidence.

Production code MUST NOT depend on documentation rendering, manuscript building, mutable research
downloads, or historical evidence. Research tools MAY consume public package APIs and schemas but
MUST NOT be imported to execute a locked production plan.

## Distribution boundary

PLLM ships one Python distribution named `pllm`, one import package named `pllm`, and one CLI named
`pllm`. Maturin builds private PyO3 module `pllm._native` from `crates/pllm-python`. `pllm._native` is
an internal Python boundary, not the independently versioned native plugin ABI.

Wheel contents MUST contain only runtime-required Python, typed stubs, `py.typed`, native artifacts,
licenses, and deliberately shipped data. Source distributions include enough pinned Rust and Python
source to rebuild. Tests, benchmarks, research, docs, and scripts MAY be included in source archives
without becoming runtime dependencies. Build trees, caches, local databases, keys, downloaded
models, generated credentials, and prepared state MUST NOT enter either distribution.

Optional dependencies name actual capabilities. BFV/FHE dependencies belong in an explicit extra
and MUST NOT be implied by generic protected runtime terminology. Platform and accelerator support
is advertised only for tested wheel combinations.

## Python package boundary

Public APIs are exported from `pllm` or stable domain facades. Current canonical facades include:

```text
pllm                     configuration, ModelPlan, compile, benchmark, assure, client APIs
pllm.components          component configuration contracts
pllm.config              configuration facade
pllm.pipeline            Pipeline and Experiment declarations
pllm.protocols           protocol component declarations
pllm.kernels             kernel selectors
pllm.preparation         preparation declarations
pllm.deployment          placement declarations
pllm.runtime             intentionally public runtime/client contracts only
```

Future domain facades MAY add models, operators, numerics, representations, conversions, metrics,
search, assurance, integrations, or testing when stable public objects exist. They MUST NOT be
created empty merely to match a conceptual tree.

Implementation modules use a leading underscore or live behind a facade. `_native`, `_cli`, and
`_internal` are private. Public documentation and examples MUST import from facades, not a private
module or generated extension. Every public export requires a type stub or typed source,
documentation, and a compatibility test. Root exports remain curated; do not turn `pllm.__init__`
into an export of every implementation class.

`pllm.components.list_components()` and `pllm.components.get_component()` are the only built-in
descriptor discovery APIs. Discovery reads the static descriptor registry and MUST NOT import
runtime, provider, or native implementation modules. Paper metadata and implementation status are
documentation data, not a Python runtime API.

Configuration is immutable, JSON-safe, and side-effect-free to construct. Model-data scans,
compilation, protected computation, material allocation, scheduling, clocks, and trace capture use
coarse Rust calls. Python callbacks MUST NOT occur per scalar, gate, label, tensor row, protocol
frame, or timed benchmark event.

## Rust workspace boundary

Current crates have these canonical responsibilities:

| Crate | Responsibility |
| --- | --- |
| `pllm-types` | Lowest-level serializable plan and evidence records |
| `pllm-core` | Exact integer execution, codecs, and masking primitives |
| `pllm-models` | Model adapters, semantic decoder plans, and model-state transformations such as cache policies |
| `pllm-compiler` | Deterministic validation, analysis, lowering, and plan compilation |
| `pllm-bench` | Native region timing and canonical measurement records |
| `pllm-assurance` | Deterministic scoped assurance fixtures and reports |
| `pllm-garble` | Explicit research implementation of arithmetic garbling |
| `pllm-python` | Thin PyO3 bindings and conversion to public Python objects |

New crates are justified by ownership, dependency, build, or reuse boundaries, not by mirroring
every Python module. Shared domains such as components, numerics, protocols, preparation, kernels,
runtime, transport, or plugin API MAY remain in an existing crate until a split improves those
boundaries.

Dependency direction is inward:

```text
serializable types + exact primitives
  -> models and semantic components
  -> compiler and plan validation
  -> execution, benchmark, and assurance services
  -> pllm-python
```

More specifically:

- `pllm-types` MUST NOT depend on compiler, execution, Python, docs, or research code.
- Model adapters emit shared semantic records and MUST NOT import a named protocol family.
- Kernels implement typed operations and MUST NOT branch on model names or paper identities.
- Compiler and runtime consume component contracts, not a provider's private implementation type.
- Benchmark and assurance code consume locked plans and typed events; they do not redefine runtime
  semantics.
- `pllm-python` may depend inward on native services. Native domain crates MUST NOT depend on PyO3.
- Cycles, global mutable registries, and catch-all `utils` crates or packages are prohibited.

## Components and plugins

Built-ins live in their semantic Python facade and owning Rust crate while satisfying the same
descriptor and compatibility contracts as external providers. A model-graph transformation belongs
with model or compiler semantics. Like-for-like implementations share a capability-named module;
paper names appear only on concrete implementations and attribution records. A new paper MUST NOT
create a top-level crate or Python package merely because its implementation is new.
The capability taxonomy is not closed: a new semantic role or lifecycle may introduce a sensibly
named family, and a crowded family may later split into a crate without changing what its typed
component contract means.

Community providers use a separate distribution and top-level import package, for example:

```text
distribution:  pllm-acme-lut
import:        pllm_acme_lut
component:     org.example/weighted-path
entry point:   [project.entry-points."pllm.providers.v1"]
manifest:      pllm_acme_lut/pllm-plugin.json
```

They MUST NOT install into `python/pllm`, monkey-patch private modules, or win by import order.
Static manifests, schemas, docs, and native artifacts remain package-relative and are enumerated by
distribution metadata. Executable plugin code stays in its provider package. Native dynamic
providers use the component standard's C ABI; they do not link to `pllm._native` internals.

## Schemas, docs, and research

Canonical JSON Schema draft 2020-12 files live only in `schemas/`. Filenames use
`<record>.schema.json`; `$id` and in-document `schema_version` identities are stable and versioned.
Fixtures live under `schemas/fixtures/`. Generated copies MAY be shipped to documentation or package
resources only when CI verifies byte or semantic parity and identifies `schemas/` as source.

Normative engineering contracts live in `design/`. User-facing source lives in `docs/` and follows
the documentation standard. Generated search indexes, Markdown mirrors, and static output remain
under docs-owned build paths and are never imported by runtime code.

`docs/data/research/` stores the source-aware paper catalog, notes, source locks, and bibliography;
`docs/evidence/` stores retained historical evidence. Public paper pages and the reimplementation
backlog are generated or authored under `docs/content/`. Algorithm implementations do not live in
these documentation paths or under paper-named runtime directories. Source claims, reproduction
status, implementation coverage, measurements, and assurance findings remain separate records.

## Artifacts, state, and caches

Four storage classes MUST remain distinct:

| Class | Examples | Rules |
| --- | --- | --- |
| Public build artifacts | resolved config, model lock, `plan.lock.json`, role plans, compatibility report | Immutable, content-addressed, exportable |
| Role-local runtime state | identities, session state, reservations, prepared material | Owner-only, non-cloneable authorization, never exported as config |
| Cache | downloaded public models, compiled kernels, docs/build caches | Recomputable; cache hit never grants authorization or freshness |
| Evidence output | scrubbed benchmark and assurance reports | Versioned, cohort-bound, no secret payloads |

A compiled artifact directory has public structure, not a combined role-state archive:

```text
build/<name>/
  experiment.resolved.json
  model.lock.json
  plan.lock.json
  compatibility.json
  roles/client.plan.json
  roles/preparation.plan.json
  roles/inference.plan.json
```

Role-local directories contain identities, sealed material, reservations, and state owned only by
that role. Public receipts MAY be exposed through a separate redacted status area. No ordinary build
artifact contains combined secret slices.

Repository examples SHOULD use `build/<name>/` for public plan artifacts and `runs/<run-id>/` for
scrubbed evidence. Installed defaults follow platform directories: configuration under
`$XDG_CONFIG_HOME/pllm`, mutable state under `$XDG_STATE_HOME/pllm`, and recomputable data under
`$XDG_CACHE_HOME/pllm`, with platform-equivalent locations where XDG variables do not apply.

Copying a directory, restoring a snapshot, or finding a cache entry MUST NOT authorize reuse of
consumed material. Public and secret role slices are delivered separately. Secret state, prompts,
outputs, credentials, and private model material MUST NOT enter benchmark exports, release archives,
crash reports, or telemetry by default.

## Releases

Release rules:

1. Python, Cargo, citation, and tag versions agree after documented normalization.
2. `uv.lock`, `Cargo.lock`, and docs lockfiles required by release policy are committed.
3. Release archives come from a reviewed committed tree, not untracked workspace state.
4. Wheels and source distributions are inspected for forbidden files and required licenses/stubs.
5. Schema, public import, native binding, example, and documentation parity checks pass.
6. Checksums are generated after artifacts and are themselves reviewed.
7. Release notes distinguish API support, implementation coverage, evidence, and security changes.

A source archive proves content identity only. A checksum, signature, successful build, or included
test report does not prove algorithm fidelity, runtime correctness, performance, or privacy.

## Naming and extension rules

- Python modules, parameters, and functions use `snake_case`; public classes use `CapWords`.
- Rust crates use `pllm-<domain>`; Rust modules use `snake_case`.
- CLI commands and options use lowercase kebab case; environment variables use `PLLM_` plus upper
  snake case.
- Component IDs use an authority-owned namespace and lowercase kebab-case name:
  `org.example/component-name`.
- Category, method, schema, ABI, and wire identities carry independent explicit versions. One package
  version MUST NOT stand in for all of them.
- Files called `utils`, `misc`, `common`, `new`, `experimental`, or paper acronyms require a narrower
  semantic home. Experimental status is metadata, not a namespace.
- Generic homomorphic-encryption abbreviations are not runtime, package, component, or profile
  labels. Use precise representation or method names. `BFV`, `FHE`, or homomorphic encryption
  terminology is valid only for code and contracts that actually use those constructions.
