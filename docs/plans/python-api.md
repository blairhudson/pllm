# Python API

## Status and scope

This document preserves the canonical Python contract that the CLI mirrors. Normative requirements
describe target behavior; the final section records current implementation separately. Public
symbol presence is not evidence of complete model coverage, deployment support, or security.

## Lifecycle and ownership

```text
Experiment -> resolve profile
model config -> lower_model(...) -> ModelPlan -> apply components
compile request -> compile(...) -> immutable compiled plan
compiled plan -> benchmark(...) -> EvidenceReport
assurance fixtures -> assure() -> EvidenceReport
```

This surface is Rust-first and PyO3-backed. Python owns immutable declarations, strict data loading,
coarse service invocation, and result presentation. Rust owns semantic lowering, plan validation,
model-data scans, hot-path computation, benchmark clocks, and assurance execution where those
features are implemented.

Configuration and `ModelPlan` values are immutable, inspectable, cloneable as public data, and
JSON-safe. Compiled plans are immutable opaque native objects with canonical public records. Live
runtime, session, buffer, and prepared-material handles are non-cloneable and non-pickleable. No
public clone operation duplicates secrets, randomness, reservations, or active state.

Constructors MUST NOT download models, discover devices, load providers, compile plans, prepare
material, start processes, or open network connections. Each side effect begins at an explicit
resolution, compilation, preparation, execution, benchmark, or assurance call.

## Configuration and components

`Model`, `ComponentRef`, `Pipeline`, `Deployment`, `ExecutionBudget`, and `Experiment` are immutable
public declarations. They support strict `to_spec()` serialization. Configuration objects support
`get_params(deep=True)` and immutable `with_params(**changes)` using `__`-separated paths.

`ComponentRef` identifies a registry component and public parameters; it is not an import path and
does not imply provider trust, availability, compatibility, or coverage. Concrete classes live in
their capability families, and each class's `describe()` method is the source for its registered
descriptor. `pllm.components.get(identity)` resolves a class; descriptor discovery is derived from
that registry. Named entries in `Pipeline.components` bind slots in a semantic graph. They are not
sequential transformers.

A built-in profile is a `Pipeline` subclass whose constructor signature is its slot contract.
`MaskedLinearCpu(model, *, linear, preparation, inference, kernels)` is the complete public-weight
baseline. `ProprietaryGuarded`, `ProprietaryBlinded`, and `DirectFHEProfile` bind the shipped
one-role proprietary engines and reject another implementation identity in the same slot. Category
ABCs reject a component in the wrong slot. Generic serialized pipelines remain available through
`Pipeline.from_spec()`, which promotes exact supported shapes back to typed classes. Whole-slot and
nested `with_params()` changes remain immutable.

JSON and safe YAML loading MUST reject duplicate keys, unknown required fields, non-finite values,
unsupported types, oversized documents, and executable tags. Python objects, JSON, YAML, and CLI
overrides MUST converge on identical canonical bytes and configuration digests.

## Model lowering

```python
plan = lower_model(
    model_config,
    batch=1,
    max_input_tokens=128,
    max_new_tokens=32,
)
```

`lower_model(...)` accepts supported public model configuration as a mapping, UTF-8 JSON bytes, or
JSON string and returns `ModelPlan`. It lowers bounded prefill and decode semantics without loading
weights. Bounds are part of plan identity; changing batch or token limits creates a different plan.

`ModelPlan` exposes canonical bytes, a digest, immutable prefill/decode views, and `to_dict()`.
`ModelPlan.apply(component)` returns a new plan after applying one typed model-graph component. It
MUST reject components at the wrong lifecycle phase or with incompatible semantics. Application
order and resulting component identities are captured by the plan digest.

`ModelPlan.coverage(profile)` returns a `DecoderCoverageReport` separating primitive availability
from executable operator coverage. `complete` means complete only for the named profile, bounded
plan, and report version. It is not a security, fidelity, quality, or deployment claim.

## Compilation

```python
compiled = compile(request)
```

`compile(...)` accepts a strict compile-request mapping, canonical UTF-8 JSON bytes, or JSON string.
It invokes one native compiler and returns an immutable compiled plan. Invalid input and unsupported
contracts raise `CompilationError`; the diagnostic MUST identify stage and unmet requirement.

Compilation freezes model and tokenizer identities, semantic and numeric graphs, components and
artifact digests, representation conversions, roles, workload, and target capabilities. It MUST
resolve all `auto` choices and fail closed rather than substitute plaintext, Python, another
protocol, a different role topology, or weaker evidence requirements. Compilation allocates no
secret prepared material.

`Experiment.resolve()` and `compile(...)` are distinct: profile resolution canonicalizes supported
public intent; compilation validates a concrete compile request and creates an executable native
plan. A future convenience accepting `Experiment` directly MUST lower through these same contracts,
not implement a parallel path.

## Benchmark and assurance

```python
report = benchmark(
    compiled,
    weights=weights,
    input=input_bytes,
    id="case-1",
    privacy_cohort="cohort-a",
    numeric_cohort="numeric-a",
    environment=environment,
)
assurance_report = assure()
```

`benchmark(...)` accepts only a plan returned by `pllm.compile`, immutable input bytes, explicit
cohort identities, and a structured environment. It invokes native timed execution and returns an
immutable `EvidenceReport`. Warmups, repetitions, thread count, SIMD choice, failures, and oracle
comparison remain report fields. Different privacy or numeric cohorts MUST NOT be compared as one
performance population.

`deployment_benchmark(request)` validates supplied deployment observations into a canonical report;
it does not itself prove that observations came from a trusted deployment.

`BenchmarkResult` is the canonical full-result envelope for new benchmark workflows. It validates
metric parameters and units, exact profile/component/model/plan/configuration/environment identity,
cohorts, warmup and measurement samples, unavailable reasons, failures, and limitations.
`EvidenceRegistry` preserves completed, failed, and unavailable results and queries exact cohorts; it
does not aggregate or rank unlike records. Existing `EvidenceReport` schemas remain unchanged.

`assure()` runs deterministic, bounded native assurance fixtures, including negative controls, and
returns `EvidenceReport`. Success means a report was produced. Findings retain their model, view,
assumptions, budget, and exact outcome; no report implies universal privacy or implementation
refinement.

`EvidenceReport` provides canonical bytes, schema version, immutable data view, and `to_dict()`.
Source claims, implementation coverage, measurements, assurance findings, and deployment
assumptions MUST remain separate records even when presented together.

## Search

`SearchSpace` binds discrete public parameter paths and typed constraints to one immutable base
`Experiment`. `GridSearch` enumerates every compatible combination; `RandomSearch` samples without
replacement under an explicit seed. Candidate configuration digests and trial IDs bind evaluator
results. Invalid individual or combined substitutions fail before evaluation.

`ParetoFrontier` requires exact scope, model, workload, environment, privacy cohort, numeric cohort,
warmup/repetition count, and objective semantics. Each objective declares `min` or `max`; no implicit
scalarization is provided. Failed and unavailable records remain available but are excluded from the
frontier. Bayesian search and CLI orchestration remain unavailable.

## Research ingestion

`SourceRecord`, `ArtifactLock`, and `MethodRecord` keep publication attribution, upstream artifact
quarantine, PLLM adaptation, and lifecycle evidence separate. `ResearchRegistry` resolves all
references and computes fail-closed promotion blockers. Restricted upstream code stays
external-process/oracle-only, and reproduction verification requires pinned artifact and evidence
digests. The registry performs no acquisition, import, build, subprocess, network, or execution.

Promotion requires a native operator, region, or full model; source and artifact locks; fidelity and
benchmark passes; protected-execution and assurance passes or explicit non-applicability; an explicit
promotion pass; and at least one eligible profile. Planned, reference-only, and structural adaptations
remain in the registry without becoming runtime methods.

## Errors and compatibility

Public validation uses typed exceptions rooted in `ValueError` or `TypeError` as documented. Error
messages MAY improve, but stable machine diagnostics belong in versioned report or exception data,
not string parsing. Unknown schema versions and required fields fail closed. Optional namespaced,
inert annotations MAY round-trip only where their owning schema permits them.

Public facades, signatures, type stubs, docs, and tests MUST change together. Imports from
`pllm._native`, private runtime modules, or a provider's implementation module are not stable API.

## Implementation status

As inspected on 19 September 2026:

- Immutable generic configuration, strict JSON/YAML loading, canonical digests, nested
  `with_params()`, and local `Deployment` declarations are implemented in
  `python/pllm/configuration.py`; concrete component classes live in their public capability
  families and feed the class-derived registry.
- `lower_model`, immutable `ModelPlan`, component application, and coverage reports are implemented
  through PyO3 in `python/pllm/modeling.py`.
- `compile` accepts mapping/bytes/string compile requests and returns native `CompiledPlan` in
  `python/pllm/compiler.py`.
- `benchmark`, `deployment_benchmark`, `assure`, immutable `EvidenceReport`, canonical
  `BenchmarkResult`, and `EvidenceRegistry` are implemented in `python/pllm/evidence.py` for their
  documented scopes.
- Typed exhaustive/seeded search, constraints, evaluator binding, and explicit Pareto filtering are
  implemented in `python/pllm/search`; Bayesian and CLI search orchestration are unavailable.
- Static research attribution, upstream quarantine, lifecycle separation, and promotion decisions are
  implemented in `python/pllm/research`; artifact acquisition and execution stay outside this API.
- Public capability families and the class-derived registry expose the installed protocol, kernel,
  preparation, role, pass, scheduler, state, verification, correlation, and benchmark-metric
  components; approved external provider descriptors can extend the same registry explicitly.
- Remote `Deployment`, general `Runtime`/`Session` composition, full locked-plan execution, broad
  model coverage, and the target CLI mirror are not established by these APIs.
