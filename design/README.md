# PLLM design

This directory is the canonical design contract for PLLM 0.1. It describes intended
interfaces and acceptance gates; it does not claim that every interface or research method is
implemented. `MUST`, `MUST NOT`, `SHOULD`, and `MAY` are normative.

- [Architecture](architecture.md): ownership, lifecycle, plans, profiles, and trust boundaries.
- [Python API](python-api.md): configuration, model lowering, compilation, and evidence interfaces.
- [CLI](cli.md): command grammar, behavior, roles, outputs, security, and implementation status.
- [Component standard](component-standard.md): identity, discovery, composition, and native ABI.
- [Package structure](package-structure.md): repository, distribution, dependency, and artifact boundaries.
- [Documentation standard](documentation-standard.md): one documentation source and evidence-safe claims.
- [Roadmap](roadmap.md): ordered delivery and promotion gates.

Canonical machine contracts live in [`../schemas/`](../schemas/). Paper records and implementation
status live in [`../docs/data/research/`](../docs/data/research/), and implementation requirements
live in the public research backlog. Historical measurements remain under
[`../docs/evidence/`](../docs/evidence/) and are not implied to be current runs.

## Authority

Apply canonical contracts in this order:

1. Fail-closed security, freshness, trust, and role-separation rules in
   [architecture](architecture.md), the [component standard](component-standard.md), and repository
   `SECURITY.md`; when rules differ, the stricter constraint wins until resolved.
2. Topic-specific design pages govern intended semantics and interfaces; this README is only an
   index and summary.
3. Versioned schemas govern serialized record shape. Prose governs meaning not expressible in JSON
   Schema. A conflict is an error, not permission to choose the weaker interpretation.
4. Shipped code, stubs, CLI help, tests, and release docs establish current implementation status.
   They do not turn an unimplemented normative interface into an implemented one or erase a target
   invariant.
5. Roadmap ordering, examples, research recipes, source statements, and historical evidence do not
   override normative contracts. Examples illustrate; tests show only their recorded scope.

Each design page separates normative target from implementation status. Claims about source
literature, implementation coverage, measurement, assurance, and deployment assumptions remain
distinct even when one page links them.

## Principles

1. One immutable configuration model serves Python, YAML, and CLI authoring.
2. Compilation is `Config -> LogicalPlan -> ExecutionPlan -> PlanLock`.
3. `Runtime`, `Session`, and `PreparedMaterial` are opaque live resources, never configuration.
4. Named components and explicit representation conversions replace implicit composition.
5. Rust owns hot-path work; Python crosses coarse FFI boundaries for configuration and results.
6. Strict profiles fail closed when coverage, conversion, trust, parameter, or evidence gates fail.
7. Claims, source statements, measurements, and assurance findings remain separate records.
