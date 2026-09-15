# Research and method standard

Status: normative target contract for adding, reproducing, composing, measuring, and promoting
research-derived methods. It defines acceptance gates and placement; it does not claim that every
record type, command, API, method, or gate is implemented.

## Scope and organizing rule

PLLM maintains an open-ended method registry, not a completion quota. Registry length, `Rxx`
numbering, and priority order are curatorial metadata, not product scope, implementation status, or
eligibility. A relevant source MAY be registered whenever its identity, access state, citation, and
license review are recorded. Each method then advances independently, may remain blocked
indefinitely, and may stop with a negative result.

Research records describe papers and experiments. Runtime code is organized by semantic component
responsibility under the [package structure](package-structure.md), not by paper. A paper may motivate
several method semantics or components, and one method may cite several papers. Paper identity,
method identity, component identity, and implementation identity MUST NOT be collapsed.

## Method lifecycle

No stage may be inferred from a later artifact's existence. Passing one stage creates evidence only
for its recorded scope and does not waive another stage.
Source acquisition and reproduction-workflow status MUST be recorded separately. Acquiring a paper,
repository, or commit does not mean that an artifact built, ran, matched, or reproduced its source.

1. **Source lock, license, and citation.** Record primary publication metadata, stable URL or DOI,
   authors, publication version, access date, full-text digest when available, approved citation,
   and source-access limitation. For an upstream artifact, also lock repository URL, commit,
   submodules, dependencies, build recipe, artifact digest, and license disposition. Missing full
   text, an unresolved license, or an unavailable required artifact is a blocking result, not
   permission to reconstruct omitted details.
2. **Clean-room PLLM reference.** Write a small, inspectable reference from the locked publication's
   specification, equations, and public test material without copying, linking, importing,
   translating, or transpiling upstream implementation code. The reference states exact numeric
   semantics, roles, setup, inputs, outputs, failures, and assumptions. Python is permitted only in
   an explicitly labelled research or test reference path; its presence does not establish native
   support.
3. **Fidelity tests.** Test the clean-room reference against published vectors, equations, original
   workloads, and, when lawful and available, outputs from the locked upstream artifact. Record
   matches, mismatches, unavailable cases, adaptations, and coverage. Fidelity to an adaptation is
   not reproduction of the original method.
4. **Native implementation.** Implement reusable hot-path behavior in its semantic Rust owner and
   expose it through ordinary component contracts. Differential tests bind the native artifact to
   the clean-room reference over declared domains, boundaries, failures, and randomized cases. A
   passing reference test is not native parity, and kernel parity is not complete-method coverage.
5. **Typed composition.** Bind method semantics, representations, conversions, numeric policy,
   roles, corruption sets, leakage, state and material lifetimes, workload, and implementation
   coverage into `LogicalPlan`, `ExecutionPlan`, and `PlanLock` identities. Shape or dtype agreement
   alone is insufficient. Unknown or unsupported contracts fail closed without plaintext, Python,
   topology, or protocol substitution.
6. **Scoped assurance.** Exercise applicable proof models, adversaries, negative controls, and
   implementation checks. Every result retains its formal model or observed view, assumptions,
   resource bound, implementation/refinement boundary, and exact outcome. Allowed outcomes remain
   `proved_in_model`, `refuted_in_scope`, `not_refuted`, `inconclusive`, `outside_contract`, and
   `unchecked`; no scalar security score or universal secure Boolean is valid.
7. **Matched multi-objective benchmark.** Compare complete, locked candidates only inside an
   immutable evidence cohort. Include offline preparation, all online roles, conversions, warmups,
   failures, and resource use. Report the full objective vector and uncertainty; no weighted summary
   score may hide a regression.
8. **Documentation and promotion.** Publish exact source relationships, citations, lifecycle state,
   supported scope, composition contract, assurance outcomes, benchmark cohort, negative results,
   artifacts, and limitations. Promotion to a built-in or default follows component and profile
   gates; documentation alone does not promote a method.

A changed source version, method semantics, implementation artifact, plan, workload, or evidence
cohort starts new immutable lineage nodes and reruns affected downstream stages. It MUST NOT rewrite
an earlier result into applicability.

## Provenance and lineage

Every accepted record is immutable, versioned, canonically serialized, and content-addressed where
its schema supports a digest. Human-readable names and moving aliases are discovery aids only. A
correction creates a new record that explicitly supersedes or retracts the old record; published
claims continue to resolve their original identities.

Registry labels such as `R23` are aliases, not publication-source, upstream-artifact-lock, recipe, or
method identities. Relationships MUST retain distinct record IDs rather than substitute an alias.

Minimum distinct identities are:

- source record and exact publication digest;
- upstream repository commit, dependency/build lock, and executable artifact digest;
- method semantic ID and version;
- clean-room reference revision and artifact digest;
- fidelity-test definition and result bundle;
- component/provider version and native artifact digest;
- model, tokenizer, numeric graph, privacy contract, role topology, workload, and `PlanLock`;
- assurance model, adversary, fixture, checker, scope, and result;
- evidence cohort, benchmark protocol, environment, run, and evidence bundle;
- claim, document build, and manuscript version.

Lineage edges MUST name the relation, such as `derived_from_specification`, `tested_against`,
`implements`, `composes`, `measured_by`, `supports_claim`, `supersedes`, or `retracts`. A normal
lineage is:

```text
source + upstream lock
  -> method semantics
  -> clean-room reference
  -> fidelity result
  -> native component artifact
  -> locked composed plan
  -> assurance and benchmark evidence
  -> scoped documentation or claim
```

Aliases such as `latest`, branch names, package ranges, unpinned model names, and mutable URLs MUST
NOT appear as sole provenance for evidence or publication. Evidence does not transfer to a port,
fork, optimization, changed parameter set, or upstreamed component without an explicit validation or
refinement edge.

## Citation obligations

Every method card, reference, implementation page, benchmark comparison, promoted component, and
manuscript that derives semantics or motivation from prior work MUST cite the original research and
link its source record. Citation includes title, authors, publication venue or archive, year, stable
identifier, consulted version, and access state. Source-reported numbers also identify their table,
figure, section, workload, and assumptions when available.

PLLM adaptations MUST cite every source whose equations, construction, parameters, workload, or
artifact informed the adaptation and MUST state the semantic differences. A transitive citation or
an upstream repository name does not replace citation of original research. Missing full text,
missing artifacts, conflicting versions, and inaccessible supplementary material remain visible in
the citation/source record and limit claims.

Copied license text and required notices remain with distributions that contain licensed material.
Citation is not license permission, and license permission is not fidelity evidence.

## Upstream artifact boundary

An upstream artifact is an oracle only. Subject to its license and access terms, an isolated
reproduction process MAY build and execute the exact locked artifact and emit scrubbed test vectors,
outputs, traces allowed by policy, and measurements. Those outputs become typed evidence tied to the
upstream artifact digest; they do not become runtime dependencies or PLLM implementation source.

Upstream source, binaries, containers, generated code, patches, and dependencies MUST NOT be
vendored into the PLLM repository, wheel, source distribution, native crate, or provider package.
PLLM production code MUST NOT import, link, invoke, or fetch them. External artifacts live in an
operator-controlled content-addressed cache or isolated workspace outside the repository and are
resolved only by an explicit research recipe. If execution or output redistribution is not allowed,
the oracle case remains unavailable.

Reference implementers use the publication specification and approved public vectors, not upstream
implementation source. Oracle operators expose only recorded inputs, outputs, build identity, and
diagnostics needed by the fidelity contract. Similar output does not prove identical internals,
security, or complete fidelity.

## Negative results

Negative and unavailable results are first-class, append-only records. They include source or
license blocks, build failures, fidelity mismatches, unsupported operators, numeric divergence,
quality loss, assurance counterexamples, resource exhaustion, incomparable cohorts, benchmark
regressions, and failure to advance the Pareto frontier.

Failed candidates MUST remain addressable from the method and cohort records. They MUST NOT be
deleted from the denominator, silently retuned after observation, relabelled as skipped, or averaged
only across successful runs. A subsequent fix creates a new candidate and evidence bundle linked to
the prior result. Security-sensitive counterexamples follow coordinated disclosure policy, but their
existence and eventual disposition remain part of lineage.

`not_refuted` means only that a named attack failed within its recorded budget. `not_available`,
`unsupported`, `failed`, `inconclusive`, and `not_applicable` remain distinct. Negative evidence may
block promotion; it is still a valid research outcome.

## Evidence cohorts and Pareto comparison

An evidence cohort is an immutable canonical record and digest of comparison invariants. It freezes:

- semantic task, accepted output, model/tokenizer revisions, graph, and workload distribution;
- numeric domains, scales, rounding, approximation, error limits, and quality acceptance criteria;
- privacy/integrity contract, roles, corruption and trust assumptions, leakage, output policy, and
  deployment topology;
- hardware allocation, OS, toolchain, dependencies, devices, threads, process placement, and
  resource limits;
- preparation/freshness policy, cache state, warmup, repetitions, stopping rule, seeds or input
  sampling policy, and failure accounting;
- metric definitions, units, aggregation, percentiles, uncertainty method, and benchmark harness
  identity.

Candidate records bind their method, component/native artifact, parameters, plan lock, environment,
and raw run IDs to that cohort. Changing a cohort invariant creates another cohort. Comparison across
cohorts is descriptive and separated; it MUST NOT be presented as a matched speedup. Search may vary
only predeclared candidate dimensions and may not weaken numeric, quality, privacy, topology,
workload, freshness, or evidence contracts.

Every matched benchmark reports these Pareto objectives without collapsing them into one score:

| Objective | Direction and minimum accounting |
| --- | --- |
| TPS | Maximize sustained accepted output tokens per second; state concurrency and denominator. |
| Latency | Minimize end-to-end and declared TTFT/inter-token percentiles, not one favorable sample. |
| Peak memory | Minimize per-role and per-device/host peaks; also report simultaneous system peak. |
| CPU | Minimize core-seconds for every role and phase; utilization alone is insufficient. |
| Network | Minimize bytes and rounds per link and phase, including setup and retries. |
| Disk | Minimize persistent/temporary footprint and bytes read/written, including prepared data. |
| Preparation | Minimize wall time and complete offline work; report storage, traffic, and amortization separately. |
| Quality | Maximize predeclared task quality or minimize loss under the frozen acceptance method. |
| Privacy | Prefer only an explicitly justified stronger contract in the partial order of assumptions, leakage, and assurance; never assign a scalar privacy score. |

A claim that candidate `A` Pareto-improves baseline `B` requires `A` to be no worse on every
comparable objective and strictly better on at least one, with predeclared uncertainty treatment.
Different privacy contracts are incomparable unless a reviewed relation proves one at least as
strong under the same relevant assumptions. A candidate that trades objectives may be a new
non-dominated frontier point, but all regressions remain in the claim.

## Publication and promotion gate

Publication language MUST classify work before drafting claims:

| Class | Required evidence | Permitted claim |
| --- | --- | --- |
| Reproduction | Source lock, clean-room reference, and fidelity tests against original scope | Faithful or partial reproduction in exactly recorded scope; no PLLM novelty claim |
| Engineering improvement | Reproduction lineage plus native parity, typed composition, scoped assurance, and matched benchmark | Portability, integration, reliability, or measured engineering improvement; no scientific-contribution claim |
| Scientific contribution | Distinct method-semantic identity, documented prior-art search and novelty argument, all lifecycle gates, and a genuine matched Pareto-frontier advance | New scientific contribution scoped to its evidence |

A new PLLM research manuscript MUST NOT be opened, submitted, or promoted as a scientific paper for
reproduction alone, code cleanup, a native rewrite, unmatched speed, a weakened privacy/numeric/
quality contract, or a benchmark win on one hidden-cost metric. It requires both a genuinely novel
method or composition and statistically supported improvement in at least one declared Pareto
objective while remaining non-dominated by the strongest eligible matched baselines. A strict
`Pareto-improves` claim additionally requires dominance as defined above.

If novelty or frontier advance is absent, record the reproduction, engineering result, or negative
result and publish it only under that label in documentation/evidence. Existing historical
manuscripts remain immutable scoped records; they do not waive this gate for future work. Product
promotion is separate: even a scientific contribution cannot become a default until component,
coverage, assurance, lifecycle, quality, and release gates pass.

## Placement and public surfaces

Canonical homes are:

| Material | Canonical location and boundary |
| --- | --- |
| Record schemas and fixtures | `schemas/`; presence defines shape, not implementation |
| Source/method records and citations | `research/methods/`, including `sources/`, `source-locks/`, `cards/`, registry, and `references.bib` |
| Reproduction workflows | `research/recipes/`; commands and deliverables are planned until executed evidence says otherwise |
| Clean-room reference code and fidelity fixtures | `research/reference/<method-id>/` when present; test/research only, never imported by production runtime |
| Native and Python code | Semantic owners in `crates/` and stable facades in `python/pllm/`; never a paper-named runtime API |
| Composition/conformance tests | `tests/` and owning crate tests, linked by method/component/plan identities |
| Scoped assurance | `research/assurance/` for models/fixtures and evidence bundles for outcomes |
| Accepted research evidence | `research/evidence/`; raw/transient scrubbed run output may begin under `runs/` |
| User and research documentation | `docs/content/` in the shared documentation graph; generated outputs remain docs-owned |
| PLLM manuscript source and bibliography | `paper/`; each manuscript claim links exact evidence and lineage identities |
| Upstream artifacts | External content-addressed cache or isolated workspace only; never vendored or installed |

Paths describe ownership when material exists and do not claim that an empty or absent destination is
implemented. Historical evidence retains its existing path and identity rather than being rewritten
to match a newer layout.

CLI and Python expose generic typed metadata through `pllm.research`: immutable source, method, and
recipe records plus deterministic list/get operations. Registry aliases resolve only when unique.
The CLI uses these same public operations, so checkout and installed-wheel fallback resolve identical
records. Metadata inspection is inert: recipe records describe workflows but neither API executes a
recipe, imports upstream code, nor passes work to runtime services.

Canonical JSON under `research/methods/` and `research/recipes/` remains source of truth. A readable,
generated wheel catalog carries exact records, relevant schemas, relationship IDs, generated-file
notice, catalog version, and SHA-256 digest of canonical JSON payload bytes. The sole generator is
`scripts/generate_research_catalog.py`; CI and release checks SHOULD run it with `--check`. Both checkout and
fallback paths reject duplicate fields, non-finite numbers, schema violations, duplicate or ambiguous
identities, missing required relationships, excessive nesting, record counts, strings, containers,
or bytes. Public failures disclose validation class only, never raw record values.

Runtime selection remains through `ComponentRef`, typed composition, profiles, and locked plans. It
MUST NOT add commands or classes named after papers, special-case an `Rxx` identifier in compiler or
runtime logic, or expose APIs such as `DashRuntime` or `run_mpcache()`. Research tooling MAY accept a
research record ID to locate its recipe; the recipe lowers to ordinary components and plans before
execution. Metadata inspection MUST NOT import providers or execute upstream artifacts.

These metadata surfaces do not imply executable lifecycle orchestration. Current registry files,
recipes, schemas, and historical evidence MUST NOT be described as executable merely because they
exist.
