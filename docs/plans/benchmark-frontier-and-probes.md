# Incremental benchmark frontiers and scoped SDK probes

Status: **design proposal with an implemented bounded first slice**, 6 October 2026.
The original sketches below record the broader proposal. The installed API uses
`BenchmarkTarget`, `BenchmarkCase`, `BenchmarkSuite`, `Sampling`, `BuildRef`,
`FrontierStudy` and `Variant`; see [SDK usage](../content/docs/sdk/evaluate/benchmark-targets.mdx)
and [the implementation screen](../research/benchmark-targets-reference.md).
Existing measurements retain their original scope and implementation identities.

The [five tenfold challenges](frontier-tenfold-challenges.md) exercise this design
with concrete allocation, Preparation-kernel and persistent-state hypotheses.
They add explicit acceptance obligations for target vetoes, native backend identity,
restart ownership and composition interactions. Their bounded screens are separate
from complete serving implementations or tenfold results.

## Outcome

Make each optimization produce three things through the SDK: a small reproducible
probe, an explicit set of affected benchmark cases, and evidence for the complete
compositions that actually benefit. A frontier answer must identify the measured
build, composition, workload and lifecycle, plus missing coverage.

Two additions share one measurement engine:

1. **A versioned frontier study** maintains candidates, dependencies, matched
   benchmark cases and immutable frontier snapshots as the implementation changes.
2. **Scoped probes** execute a selected artifact, operator, prepared stage or
   decoder phase through the same implementation used by normal SDK requests.

Propagation means updating candidate configurations and required measurements.
Measured CPU, RSS, latency and traffic never propagate by adding improvements or
copying a result onto a different composition.

## Findings from recent work

| Inspected work | What happened | Missing reusable contract |
| --- | --- | --- |
| `benchmarks/research/sota.py`, `prepared_prefill.py`, `prepared_frontier.py`, `examples/prepared_sdk.py` | The named SOTA factory omits sixteen-row prefill and preparation windowing; the combined SDK example selects both, provider caching and prefix reuse. The repeated prefill cohort measures a different composition. | One explicit candidate/study registry with exact configuration coverage. |
| `scripts/probe_prepared_path.py` | Reimplements installed-build hashing, alternating subprocesses, cache isolation, private cohort salts, admission, checkpoints and shutdown. | A shared isolated cohort runner, including implementation-only A/B comparisons. |
| `scripts/probe_client_response_memory.py` | Restores historical functions through Git/AST manipulation. Retained parsing owners contaminated an initial tokenizer-memory pair. | Installed-build controls and measurement ownership that keep setup/reporting out of the measured process where possible. |
| `scripts/profile_client_frontier.py` | Wraps private methods to observe bundle loading, binding, preparation, prefill, decode and allocation roots. | Stable metadata-only phase observations and bounded probe entry points. |
| `scripts/probe_public_snapshot.py` | Twelve isolated imports improved CPU 17–18% and halved peak private-file payload. The separate response pair increased RSS and latency. | Explicit links between local evidence and full-response confirmation; no automatic promotion of a local improvement. |
| `scripts/decoder_probe_support.py`, `probe_batched_greedy.py` | Rebuild source resolution, W8A8 binding, local stage callbacks, counters and logits/KV oracles. A real drafter accepted no tokens while a perfect-draft oracle looked favorable. | Model-neutral decoder fixtures, execution counters and separate real/oracle controls. |
| `scripts/probe_prepared_region_gate.py` | Already uses `benchmark_operator`, then implements its own compiler geometry, role-delivery accounting and cost veto. | A typed projection adapter with complete ownership, assumptions and unpriced work. |

Existing infrastructure is substantial:

- `Experiment`, `with_params`, native schedules and resource admission already
  define composition and legality.
- `benchmark_operator` already separates component configuration from samples,
  supplies integer oracles and retires one-use material.
- `BenchmarkResult`, `EvidenceRegistry`, `SearchSpace`, grid/random/beam search
  and `ParetoFrontier` already own evidence and candidate semantics.
- The ordinary CLI supports isolated clients and ordered request sequences;
  several scripts independently recreate parts of that orchestration.
- `benchmark_search` currently runs one cold response per trial in a shared
  client process, and exposes only full latency, request throughput, covered
  bodies and online bodies as objectives. It cannot rank isolated client peaks.
- `benchmark_result.v1` has no CPU metric and only coarse offline/online/full
  phases. The scorecard separately extracts the seven serving metrics in JS.
- The docs registry hashes historical factories and their shared configuration
  sources. Editing a shared factory to mean “current best” would invalidate that
  historical replay contract.

Sources: [Python evidence API](../../python/pllm/evidence.py),
[search adapter](../../python/pllm/_search_benchmark.py),
[operator runner](../../python/pllm/metrics/operator_benchmarks.py),
[scorecard](../scripts/research-scorecard.mjs),
[prefill evidence](../research/prepared-prefill-reference.md),
[snapshot evidence](../research/native-snapshot-reference.md),
[greedy gate](../research/batched-greedy-reference.md).

## 1. One public measurement vocabulary

Keep responsibilities in existing capability families:

| Owner | Proposed surface | Responsibility |
| --- | --- | --- |
| `pllm.metrics` | `BenchmarkSuite`, `Scenario`, `Sampling`, `BuildRef` | Immutable workload, lifecycle, observations, isolation and build declarations. |
| `pllm.metrics` | `Probe`, `ProbeInputs`, `benchmark_probe`, `benchmark_suite` | Bounded targets and explicit execution. |
| `pllm.evidence` | `BenchmarkResult` v2, registry bulk registration | Canonical measurements, provenance, raw-report references and queryable coverage. |
| `pllm.search` | `Variant`, `FrontierStudy` | Pure candidate generation, impact/retest planning and frontier selection using existing search. |
| Native compiler/runtime | Validated probe execution contracts | Semantic selection, representation/state bounds, stage/resource admission and existing kernels/protocols. |

Constructors remain inert. Execution and source resolution happen only at an
explicit runner call. Probe and study declarations support canonical serialization
and immutable parameters; live inputs, sessions and material do not.
The benchmark policy is separate from the inference Pipeline and its digest.

`benchmark_probe` returns a canonical `BenchmarkResult`; `benchmark_suite`
returns a tuple of results plus writes an optional atomic run manifest when an
output directory is requested. Existing detailed operator/loopback reports remain
immutable evidence artifacts referenced by digest. They are not rewritten into a
different measurement scope.

`BuildRef` identifies an installed execution environment. The worker supplies its
observed SDK, native-library and dependency hashes; a caller-provided label cannot
assert them. Comparing builds is an explicit treatment axis. Ordinary composition
studies hold build identity fixed. Unsupported historical runner protocols return
an explicit unsupported result; they do not install historical functions into a
current runtime.

## 2. Frontier propagation

### Study and case identity

A `FrontierStudy` binds a suite, immutable base Experiments, allowed variations,
comparison policy and a finite trial budget. Repository-specific studies name
Qwen workloads; the SDK and compiler remain model-neutral.

A case has these independent axes:

- source lock, semantic/numeric/state contract and verifier/trust contract;
- exact Experiment, plan/binding and installed implementation;
- model geometry, input/output bounds, sampling and observed output identity;
- lifecycle and cache state, execution placement, hardware and network treatment;
- target scope, instrumentation and metric definitions.

Candidate identity is distinct from comparison identity. Explicitly varied
configuration or build fields may differ between candidates. Source, numerics,
workload, output and all other locked comparison fields must still match.
Numeric equivalence comes from compiled contracts and observed parity, not a
manually curated list of parameter names to ignore.

The initial serving suite should declare:

1. Cold artifact-cache request, cached checkpoint, no reusable prefix.
2. Same-client exact repeat, with startup charged once.
3. Growing conversation with bounded state and exact continuation semantics.
4. Multiple fresh clients sharing one serving instance, for publication/cache payoff.
5. Explicitly public-prefix workload, separate from fresh private input.

“Cold” names each owner: checkpoint cache, compiled cache, client object cache,
provider encoded cache, tokenizer artifacts and prefix state. A reset operation
removes only runner-owned data. Uncontrolled OS cache stays declared.

A repeat scenario contains its cold first request and the subsequent requests in
one client lifetime. It yields per-request and sequence observations, with setup
charged once. Independent scenarios start independent client lifetimes unless an
explicit shared-provider scenario owns that relationship.

### Variants and dependencies

A `Variant` is a named immutable `with_params` patch, applicability requirements,
and a hypothesis about affected capabilities/metrics. It can cite supporting and
negative probe evidence. Installation, legality, parity, cost and end-to-end
benefit are separate coverage fields, not one “implemented” flag.

Derive execution dependencies from the compiled plan and serving contracts:
artifact publication/import, client binding, preparation, stage codecs, kernels,
prefill, decode and retained state. Repository code ownership maps connect changed
modules to these capabilities. Unmapped changes conservatively affect the whole
study. Dependency declarations prioritize work; they cannot certify that old
timings remain valid on a new build.

For example:

| Change | Cheap affected checks | Confirmation cases |
| --- | --- | --- |
| Shared native snapshots | Artifact import, validation, binding, owner retirement | Cold load and complete cold response |
| Sixteen-row duplex prefill | Prepared stage/chunk framing and prefill parity | Cold/repeat TTFT, all seven metrics |
| Preparation window | Concurrent issuance, push acknowledgement and live buffers | Cold/repeat preparation and full request |
| Provider artifact cache | Publication plus 1/2/4 fresh readers, repair and cancellation | Fresh-client cohort with publication charged once |
| Conversation ownership | Continuation admission, state retirement and exact KV | Growing conversation; no fresh-request speed inference |
| Nonlinear method | Operator parity, range and role/material projection | Numeric-quality gate, then only an admitted topology |

### Retest algorithm

1. Freeze source, candidate specs, build locks, workload and measurement policy.
2. Resolve native legality and resource admission before allocating values.
3. Query existing records by exact case identity. Missing observations, changed
   builds, failed admission and incomplete confirmation become explicit gaps.
4. Run the cheapest applicable parity/resource probe. A complete material lower
   bound can veto a candidate; an optimistic estimate cannot promote it.
5. Generate bounded neighbors of the incumbent and existing nondominated
   candidates through `SearchSpace`/grid/random/beam machinery. Include curated
   interactions sharing a resource, such as chunk rows × preparation window or
   paging × cache capacity. Include declared prerequisite bundles. Keep an explicit
   exploration allowance for combinations whose isolated effects are inconclusive.
6. Measure candidate and incumbent together in a new matched cohort. Screen first;
   freeze finalists before separate, repeated confirmation. Include selected
   leave-one-change-out ablations of the combined finalist.
7. Produce an immutable frontier snapshot: measured nondominated candidates,
   inconclusive comparisons, excluded/blocked cases and untested combinations.

The snapshot records coverage counts and the exact candidate budget. “Best tested
under this suite” is meaningful; an incomplete search is not a global optimum.
No multiplicative combination of separate RSS or latency reductions is allowed.

Fresh runs mint a new private cohort salt. After interruption, retain completed
records but issue a new cohort, re-supply inputs and rerun the control with pending
candidates. Do not merge independently salted prompt digests or resume one-use
sessions. Historical anchors track trends without making cross-run rankings.

An implementation change preserves all historical measurements and marks the new
build's coverage incomplete. Cheap exact facts may be recomputed or reused only
under their own unchanged, checked dependencies. Wall time, CPU, RSS and network
observations always retain the build and run where they were measured.

### Proposed study usage

This sketch uses a caller-owned `prepared` Experiment, input text and registry;
the classes/functions introduced here are not executable in the current SDK.

```python
from pllm.deployment import WanConditions
from pllm.metrics import BenchmarkSuite, Scenario, Sampling, benchmark_suite
from pllm.search import FrontierStudy, Variant

suite = BenchmarkSuite(
    base=prepared,
    scenarios=(
        Scenario.repeat("request", count=2, max_output_tokens=8),
    ),
    metrics="serving",
    sampling=Sampling(processes=3, order="balanced"),
    backend="native", wan=WanConditions(), emulate_wan=True,
)
study = FrontierStudy(suite, variants=(
    Variant("duplex16", {"pipeline__linear__prefill_chunk_rows": 16}),
    Variant("window64", {
        "pipeline__inventory__stage_window": 4,
        "pipeline__inventory__stage_window_bytes": 64 << 20,
    }),
))
work = study.plan(registry, max_trials=12)  # inspection only; includes retest gaps
results = benchmark_suite(work, inputs={"request": prompt})
registry = registry.register_all(results)  # returns a new immutable registry
frontier = study.frontier(registry)
```

One trial is one isolated candidate/scenario replicate, including its declared
role processes; budgets count these executions, not just configurations.
Confirmation is a separately budgeted plan. The study never edits the application's
Experiment or selects a new runtime default as a side effect of measurement.

## 3. Scoped probes

### Targets

Each target declares prerequisites, outputs, lifecycle, oracle and exact supported
metrics. Unsupported targets fail explicitly. Scope-specific admission charges
only the required owners, plus the controller and instrumentation; a preparation
probe need not reserve the whole decoder topology.

| Target | Executes | Avoids / evidence scope |
| --- | --- | --- |
| `Probe.geometry` | Lowering, schedule, shapes, row demand, allocations and body-cost calculations | No tensor-value loading, material or roles; estimates only. |
| `Probe.artifacts` | Selected publication, serving, import or binding path | No decoder; source/object authentication and public-file ownership still execute. |
| `Probe.operator` | Existing `benchmark_operator` executor and oracle | No model import or provider transport; local operator evidence. |
| `Probe.preparation` | Selected stages' fresh correction issuance; optional actual push/ack path | No tokenization, attention or generation. Issuance-only and pushed material have distinct scopes. |
| `Probe.stage` | One compiled linear stage or bounded row group: quantize, mask, encode, evaluate, decode and verify | No complete decoder. Clear-local, protocol-local and actual role-transport modes are distinct. |
| `Probe.decoder` | Prefill, teacher-forced decode, continuation or a bounded admitted region | No application response loop unless selected; prerequisites and state setup stay charged separately. |

`Probe.project` connects an existing operator report to native-checked plan
geometry and an explicit hypothetical issuer/role layout. It returns estimates or
construction-specific lower bounds, records missing operations and never creates
an executable protected decoder. Client labels, evaluator material, each delivery
edge, public artifacts and frame estimates remain separate. It generalizes the
prepared-region gate without reusing two-worker accounting for one evaluator.

### Semantic selection and loading

Select stages by semantic role, layer, phase and public bounds, never source weight
paths or family-specific node names. Resolve selectors into immutable native
descriptors with dimensions, representation, source/weight identities, state
requirements and resource limits.

The stage loader must read and quantize only selected checkpoint artifacts through
the existing source mappings and bounded loaders. A partial binding records its
partial scope; it cannot masquerade as a validated complete model bundle.
Checkpoint-backed, generated-checkpoint and synthetic-weight fixtures remain
different evidence identities even when their shapes match.

Local clear execution uses the same numeric kernels. A protocol-local probe uses
the ordinary one-use arithmetic path with in-process roles. A transport probe uses
the ordinary authenticated role adapters and messages. An arbitrary callback is
not a substitute for a protected or verifier-bound stage.

### Proposed probe usage

```python
from pllm.metrics import Probe, ProbeInputs, Sampling, benchmark_probe

report = benchmark_probe(
    Probe.stage(
        prepared,
        role="mlp_gate_up", layer=0, phase="prefill", rows=16,
        weights="synthetic", execution="protocol_local",
    ),
    inputs=ProbeInputs.synthetic(seed=7),
    sampling=Sampling(processes=3),
)
```

Here the model supplies public geometry, not checkpoint-weight fidelity. The seed
controls only public synthetic fixtures. Masks, tickets, challenges, triples and
garbled material still receive fresh protocol randomness on every execution,
including warmups and retries.

The same runner accepts `Probe.artifacts(prepared, operation="import")` with an
authenticated artifact fixture, or `Probe.preparation(...)` with selected stage
weights. These must become short installed-SDK examples without private imports,
manual subprocess code, custom clocks or raw JSON accounting.

### Inputs, state and observations

`ProbeInputs` supplies public fixtures, bounded caller-owned values or a recipe
for constructing state. Private values remain outside serialized benchmark specs
and reports; transport to an isolated local worker uses bounded private IPC or a
private temporary snapshot, with both owners admitted and retired.

Decode/continuation probes have explicit state provenance. Real prefill-derived
state must match the source, binding, numeric contract and position; synthetic KV
can measure bounded mechanics but cannot establish checkpoint fidelity. The worker
constructs prerequisites before the timed phase, reporting their costs separately.
Its lifetime RSS still includes those prerequisites. No subtraction of unrelated
peaks manufactures a decode-only RSS measurement.

A context-managed internal probe session owns the binding, state and material.
It can reuse immutable public weights within an explicitly warm exploratory run;
fresh-process confirmation starts independent owners. Prefix snapshots remain
trusted-client state. Sessions/material are non-cloneable and non-pickleable;
failure, cancellation and early exit retain ordinary consume/burn/retire behavior.

Add bounded, opt-in metadata observations at existing load/bind/prepare/prefill/
decode and stage boundaries. Reuse existing OTEL counters and ownership tracking.
Observers receive IDs, dimensions, elapsed counters and allocation categories,
never tensors, prompts, masks or references that keep arrays alive. Detailed
allocation tracing is a separate diagnostic mode with its own identity and cost.
Nested elapsed/CPU spans do not sum unless their intervals are disjoint.

Independent oracles run outside the timed window, preferably in a separate
validation process. Capture resource samples before report serialization and
reference allocation. The report identifies kernel-only, boundary-inclusive and
transport-inclusive costs instead of using one ambiguous “online” duration.

## 4. Shared runner and evidence contract

### Process and lifecycle ownership

Extract common machinery from the CLI and installed-build probes:

- Fresh process per comparable client-peak sample; balanced candidate/control
  ordering, isolated owned caches and explicit same-client sequences.
- Separate counts for independent processes, in-process repetitions, requests,
  warmups and statistical blocks. A repeat request is not another cold replicate.
- Existing native/Docker/WAN supervisors, whole-owner admission and pressure
  monitor; controllers, role children, relays and samplers are priced explicitly.
- One cancellation tree and deadline covering setup, execution and cleanup.
  Checkpoint each completed case atomically; failed/blocked work keeps typed reasons.
- Provenance collection outside measured startup where possible. If a worker must
  inspect itself, record that interval and avoid retaining source/AST owners.
- Fresh one-use issuance per warmup/sample/attempt, even for identical inputs.
  A timeout retry begins a new attempt and charges all abandoned work.

The SDK, CLI, search and probes should delegate to this runner. Wrapping a
full `run_loopback_benchmark` for every target does not satisfy scoped execution.
Artifact and operator probes must actually avoid loading/launching unused stages.

### BenchmarkResult v2

Extend the existing canonical envelope with versioned fields for:

- observed build/runner identity, exact target and fixture provenance;
- scenario, phase-window, cache state, process isolation and measurement method;
- oracle type/outcome, implementation coverage and source/numeric/state bindings;
- metric role, unit, window, denominator, aggregation, origin and sample ownership;
- measured values, deterministic counts, estimates and lower bounds as distinct
  origins; missing values remain null with a reason;
- trial lineage, cohort membership, raw sample/report digests, failure/admission
  outcomes and explicit projection dependencies.

Add a CPU metric and typed timing windows. Preserve `benchmark_result.v1` bytes
and historical reports. A reader adapter may expose known fields but must not
invent missing build, scope or isolation information. `EvidenceRegistry` remains
immutable; bulk registration is a convenience, not implicit ranking.

The seven-metric serving preset is centralized in Python and exported for docs:

| Metric | Required semantics |
| --- | --- |
| Client RAM | Fresh client process-lifetime peak, declared included harness/relay owners. |
| Client CPU | Benchmark startup through cold first response; process CPU seconds. |
| Preparation CPU | Role process birth through cold first response; separate from online/run-window CPU. |
| Preparation RAM | Sampled request-window peak plus sampler interval; lifetime peaks remain another metric. |
| TTFT | Existing online-to-first-token window; separately expose request-start TTFT including setup. |
| Decode throughput | Authoritative outputs after the first, `N−1`; null for fewer than two outputs. |
| All-link traffic | Conserved directed counters with explicit application-body, TCP-stream or physical-wire layer and lifecycle. |

Cold, repeat, sequence-total and multi-client metrics are separate observations.
Publication/setup is charged once to the lifecycle that incurred it. Missing
Preparation roles do not score as zero work. Raw measurements retain bytes;
presentation uses decimal MB. Public artifact/checkpoint distribution and any
unmeasured GPU or filesystem-cache owners remain explicitly priced or unknown.

### Comparison and promotion

Reuse one versioned comparison policy in search, CLI and docs. Require exact
source/numeric/state/verifier/workload/output compatibility and matched measurement
semantics. Build changes are allowed only in declared implementation comparisons;
host/runtime conditions remain controlled. Keep topology/quality experiments in
their own explicitly declared comparison contracts.

Maintain an exploratory observed Pareto set and a separately confirmed set.
Confirmation fixes candidates, independent replicate count, practical regression
margins and uncertainty rule before executing. Small or noisy results can remain
inconclusive; repeated sampling must not stop opportunistically on a winning result.
One fast sample cannot remove an incumbent. Unavailable required metrics prevent
a complete seven-metric frontier claim, but scoped results remain useful.

## 5. Publication and propagation workflow

```text
new component/build or changed study
    -> immutable variants + dependency/coverage diff
    -> admission and scoped parity/cost probes
    -> bounded composition/interaction screen
    -> fresh matched finalist + incumbent confirmation
    -> registry + immutable frontier snapshot
    -> generated replay configurations, SDK recipes and Research tables
```

Freeze measured Experiment JSON, public artifact manifests and build locks in the
snapshot. A generated Python factory reconstructs that exact Experiment instead
of importing a mutable `sota.py`. Current recommendation aliases point to a named
snapshot and case; historical aliases and evidence never change meaning.

CI validates recipe/result digests, coverage, replay serialization and publication
consistency. A changed implementation creates visible pending measurements; CI
does not silently launch expensive real-checkpoint benchmarks. Explicit bounded
study execution performs that work. New canonical exports should drive both
`benchmark-cohorts.json`/Research rendering and SDK examples, with separate labels
for runnable candidate recipes and measured recommendations.

An explanation query must answer: “Which additions does this result include,
which relevant additions remain unmeasured, and what is the cheapest valid next
comparison?” This is the missing link behind the current frontier ambiguity.

## 6. Delivery slices and acceptance

### A. Shared runner with two real probes

Deliver `BenchmarkResult` v2 and the runner with **artifact import** and a
**compiler-selected prepared stage** in the same slice. Adapt `benchmark_operator`
reports. Keep public examples around 10–20 lines; no empty framework-only release.

Acceptance: installed-wheel examples; exact integer/digest oracles; selected-stage
loading only; separate synthetic/checkpoint scopes; fresh-process RSS ownership;
actual-byte validation; memory rejection, timeout and cancellation cleanup; no
one-use reuse across samples. Differential checks must reconcile with the existing
snapshot and prepared-boundary probes on identical fixtures.

### B. Frontier studies using the same runner

Deliver suites, variants, coverage/retest planning, all seven metrics and generated
snapshot exports. Route ordinary benchmark/search through the shared machinery.
Freeze existing historical factories; introduce new versioned recipes.

First study: current Qwen2.5 150+8 cold/repeat candidates combining sixteen-row
prefill and admitted preparation windows. Provider-cache payoff receives a
separate fresh-client scenario; conversation and public-prefix reuse have their
own cases. Qwen3-4B 16+8 is a separate cell and remains blocked whenever its actual
admission fails. Cheap probes do not fill that whole-response gap.

Acceptance: changing chunk/window options creates the right missing cases;
code-only changes create new-build gaps; composition interactions are measured;
old reports remain readable; unknown metrics and failed comparisons cannot win;
independently salted runs cannot merge; generated recommendations name exactly
the measured build/composition. A deliberately locally-positive, globally-negative
fixture must remain outside automatic promotion.

### C. Phase/state and projection adapters

Add decoder-phase fixtures and stable observations, then geometry/operator cost
projection. Migrate the profiler, greedy block gate and prepared-region screen
to short SDK scripts. Preserve the real-drafter/oracle distinction, complete
rejected-work costs, output/KV checks and one-evaluator delivery accounting.

Acceptance: generated Qwen2/Qwen3/Llama exact state/parity controls; prerequisite
setup charged separately; no retained observer arrays; no full model-value load
for geometry; no real checkpoint claim from synthetic state; no measured network
or whole-response metric from a local cost projection.

## Decision

Prioritize **one runner, exact coverage identities and two useful probe targets**.
Then use them to keep measured compositions current. This addresses both repeated
probe scaffolding and stale frontier reporting without replacing the existing
compiler, role protocols, search algorithms or evidence history.
