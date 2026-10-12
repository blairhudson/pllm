# Scorecard format adapters

`docs/scripts/research-scorecard-adapters.mjs` reads original reports without
turning them into synthetic legacy comparisons. Register new evidence under
`adapted_cohorts` in `docs/data/research/benchmark-cohorts.json`:

```json
{
  "id": "unique-study-id",
  "title": "Study title",
  "evidenceTier": "screen",
  "adapter": "benchmark-suite-v1",
  "report": "docs/evidence/study.json",
  "sha256": "SHA-256 of original report bytes",
  "plan": "docs/evidence/study.plan.json",
  "plan_sha256": "SHA-256 of original plan bytes",
  "replay_file": "scripts/replay_study.py",
  "replay_sha256": "SHA-256 of replay source bytes",
  "scope": "Exact workload, lifecycle, resource windows and exclusions"
}
```

For a new ordinary `BenchmarkSuite` report, its plan should contain `suite` with
the exact serialized measured suite. Existing frontier-study plans instead expand
their frozen base cases, nominated variants and declared interactions. The two
omitted `PreparedInventory` defaults retain their public serialization semantics.
At least three fresh processes per arm, one measurement repetition, zero warmups,
matching installed builds, fresh client caches, successful admission/oracles and
complete original cells are required. No independently salted invocations pool.

Comparison commitments partition source, numeric/state/privacy, workload, output
and measurement scope. Raw authoritative outputs, source locks, private prompt
commitments, cold/repeat order, row conservation, directed traffic conservation,
metric units/owners/windows and exact plan cases are checked again. Parameters,
target/Experiment/Pipeline digests, build locks and available frozen source hashes
remain attached to each row. Runtime source snapshots are historical build
provenance, not assertions that today's runtime matches an older installed wheel.

Cold and repeat requests receive separate views. Cold CPU runs through the first
response; repeat CPU is unknown. Client lifetime and aggregate sampled RSS cover
the entire cold/repeat scenario. Provider RSS uses the selected request window.
Cold TCP totals include startup; repeat TCP uses consecutive counter deltas.
Means of cold and repeat timings are never substituted for either request.

The `epoch-prefix-v1` reader requires the completed 36-cell/60-final-state cohort,
three original replicates per configuration, exact final-state/output commitments,
natural EOS/full-cap use, actual pending joins, epoch transitions, private owner
retirement and versioned lifecycle body coverage. It creates four workload views.
Group first-text/text-delta values take the within-cell response median, then the
three-cell median. They remain separate `scopedTimings`, **not** ordinary `ttft`
or `decodeTps` axes; those ordinary axes stay null for every epoch workload.
No group latency or summed response throughput is invented. WAN conditions,
role mapping, body-counter coverage and measurement-window identities must match
across all cells in each workload, not merely validate independently.
Complete cleanup CPU and sampled-after-startup provider RSS retain their distinct
windows. Missing simultaneous aggregate RSS is not a sum of independent peaks.

## Explicit exclusions

* MLP confirmation A/B and the screen never pool with non-overlapping C.
  Unmeasured 16-row confirmation nominees have no C score; original exploratory
  screen observations remain in separate historical views. Historical `client_mlp` research
  selectors stay separate metadata, explicitly **not Pipeline components**.
* Earlier incomplete epoch attempts cannot supply completed confirmation medians.
* Different numeric/workload/build/scope groups never compete jointly; an
  incomplete or singleton comparison group rejects rather than inventing leaders.
* Local tokenizer read counters do not establish unique artifact publication size.
  Artifact footprint remains unknown without source-bound publisher evidence.
* Physical wire, energy and representative generation quality remain unknown.

## Attribution extension

`applicationBodyAttribution` reads the canonical
`pllm.benchmark_traffic_decomposition.v1` from either report location. Duplicate
copies must agree. It checks raw audit/ledger attribution, v3 coverage, integer
byte conservation, actual output denominators and selected request windows.
Cold-first includes startup exactly once; repeat windows exclude it.

`applyTrafficAttribution` supplies base/per-token headline values only for a
complete validated partition. Current prepared reports retain mixed overhead and
therefore have partial partitions with null headline values. Their expandable
table shows each replicate's known base, known output-dependent, residual and
total application bytes, in decimal MB. Column medians are not added.

The optional `trafficAttributionConsumer` hook can agree with canonical complete
evidence but cannot replace it. Historical reports remain unattributed.
Covered-body and TCP-stream metrics carry separate `accountingPlane` metadata;
the measured TCP total is never substituted into the application-body split.
Version 3 body ledgers require complete lifecycle controls and correction ACK
coverage reconciled with accounted correction pushes. They never retrofit v1/v2
historical counters.

## Historical replay is not a current-build claim

`benchmarks/research/historical_evidence.py` verifies original report bytes and
exposes `recorded_contract(source, case)` without importing or running the SDK.
Ordinary `case(source, case)` reconstructs the exact archived `BenchmarkCase`,
including its original `BuildRef`, target, internal selector and budgets.
`target()` alone does not retain build requirements; use `case()` for replay.
Epoch `experiment()` alone is a recipe: the recorded contract also binds original
build, frozen measurement-file hashes, workload/variant, concurrency, response
statuses and lifecycle scope. The original driver is required for those windows.

Rows show hash-bound recorded cases/Experiments and original build/source locks;
their frozen loader is itself hash-bound in the manifest. Original mutable driver
links are provenance, not proof that today's example or wheel recreates the
historical configuration, build or private workload. Fresh replay always creates
new evidence and a new private cohort; historical BuildRef environments and
private prompts may no longer be available.

## Evidence labels

The registry declares `evidenceTier` separately from comparison eligibility.
Exploratory leaders display **Screen best**; independent confirmation leaders
display **Scoped SOTA**. Observations and unclassified evidence display
**Observed best**. Every label remains confined to its matched comparison group.
Confirmation identifies a follow-up measurement lifecycle; sample counts remain
visible and do not imply statistical certainty.

Cold+repeat scenario views display summed input, generated-output and allowance
counts for both requests. Their first-request identity remains a scoped detail;
it does not describe the scenario's total traffic or process-lifetime RSS.

Lightweight verification:

```sh
node docs/scripts/research-scorecard.mjs
node --test docs/tests/research-scorecard-adapters.test.mjs docs/tests/research-scorecard.test.mjs
```
