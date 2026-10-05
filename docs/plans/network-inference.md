# Network-aware inference: CLI and SDK design

Status: **bounded foundation implemented; broader network design remains a roadmap**.
Date: 2026-10-01.

This document retains the original design below. Proposed signatures are not
automatically supported: the implementation ledger identifies the tested subset.
Existing network-reduction evidence is linked in
[network-io-10x.md](../research/network-io-10x.md).

## Implementation ledger

| Slice | Implemented and checked | Boundary |
| --- | --- | --- |
| A: snapshot planning | Immutable network/request/result records, bounded deterministic search, strict native role/resource admission, replayable decisions, selected local SDK execution | Installed client-only, prepared and two-offset compositions; unknown required costs reject |
| B: live parties | Explicit network Experiment v3, authenticated offers and directory membership, instance epochs, capacity reservation/arm/release, expiry, drain and cancellation | Live CPU two-offset and prepared Inference/Preparation roles; isolated one-use inventories and selected-peer push credentials; declared operators do not prove independence |
| C: shared application paths | Ordinary gateway and benchmark accept selected plans or planning requests; controlled east-fast/west-fast cohorts select different hosts | Selected latency regret was zero in two three-repetition loopback scenarios; full-wire and whole-response aggregate CPU remain unknown |
| E1: private state locality | Immutable shared KV blocks, compiler-bound batched suffix schedule, explicit canonical valid-prefix float32 reductions, real provider acknowledgement before inventory reservation | Canonical numeric choice admits growing-width completed-prefill reuse; historical numeric mode still requires matching full extent; generated KV remains explicitly response-owned |
| E2: public artifact locality | `ClientBundleTransport("artifacts", compression="zlib")`, authenticated bounded object frames, shared raw content-addressed cache, exact reconstruction before native import | Public prepared, verified and offset bundles; private state and one-use inventory never become public artifacts |
| E3: shallow HE | Real-shaped exact encrypted linear-region feasibility screen | Tested projection costs roughly 40 times the prepared stage bodies; non-selectable |
| F: locality-aware planning | Source-bound `ArtifactCostEvidence`, declared public residency, `reuse_horizon` and horizon body objectives | Public misses charged once; manifests, one-use material and online work charged each workload; declarations are estimates |
| G: Linux benchmark backend | Ordinary `benchmark run --docker`, slim public CPU image, one container per provider role, cgroup and interface samples | Co-located Docker development topology; host client; interface counters include control/telemetry and do not establish all-link full wire |
| H: load-time choices | `search.optimization_space` and `compiler.plan_on_load` resolve and check source identity, propose bounded existing components and select through native-admitted placement | Preserves numerics/privacy; explicit client ownership/cache permission; prices geometry and supplied evidence, not speculative GPU/compression speedups |
| I: composition and per-token measurement | Metal with client-owned attention/prefix; canonical online/setup/decode MB per generated token | Extra GPU snapshots counted; pinned combined cohort has near-equal CPU/Metal latency; application bodies, not full wire |
| J: combined contracts | Authenticated offset continuation, prefix/role placement unions, compressed artifact reuse and sealed verified-cache lineage | Exact tiny logit/KV parity; verified cache reserves twelve failure bits for 4,096 inventory transitions; real verified startup timed out and has no measured cost win |

Reproducible controls and current APIs:

- [`examples/benchmarks/network_planning.py`](../../examples/benchmarks/network_planning.py)
- [`examples/networks/README.md`](../../examples/networks/README.md)
- [`network-planner-benchmark-2026-10-01.md`](../evidence/network-planner-benchmark-2026-10-01.md)
- [`batched-continuation-2026-10-01.md`](../evidence/batched-continuation-2026-10-01.md)
- [`artifact-locality-2026-10-01.md`](../evidence/artifact-locality-2026-10-01.md)
- [`exact-linear-he-review-2026-10-01.md`](../evidence/exact-linear-he-review-2026-10-01.md)
- [`metal-placement-qwen25-2026-10-03.md`](../evidence/metal-placement-qwen25-2026-10-03.md)
- [`examples/benchmarks/automatic_planning.py`](../../examples/benchmarks/automatic_planning.py)
- [`combined-compatibility-qwen25-2026-10-03.md`](../evidence/combined-compatibility-qwen25-2026-10-03.md)

### Selection and tuning policy

Do not enable every compatible optimization. Native contracts decide legality;
explicit workload/client limits and scoped cost evidence decide benefit. Retain
the incumbent on unpriced ties. New cost records should bind source, numeric
contract, input/decode shape, reuse horizon, device, runtime implementation and
network identity; invalidate observations when those identities change. Keep
private cache state and one-use material outside persistent tuning data. Current
load-time planning prices geometry, artifact locality, reusable-state assumptions
and switching. Schema-v2 artifact cost records additionally price checked encoded
object lengths independently from raw resident storage. Union placement counts
overlapping stages once; offset estimates price seeded input and conservative
numeric packing bounds. A measured GPU/compression CPU autotuner remains follow-on
work. The current combined cohort finds the lean prepared stack cheaper in total
than moving further layers client-side, despite the union's lower online traffic.

### Numeric state gate

Pinned W8A8 execution exposed 1.9434-logit drift between generated incremental
state and canonical one-shot prefill, and 1.8366 drift when prefill-only state was
reused under another attention-reduction extent. Token equality is insufficient
to authorize exact fresh-cache reuse. Cache keys now include original prefill
extent under the historical numeric choice and require a sealed completed-prefill
execution basis. The opt-in `SymmetricPerRow(causal_reduction="prefix_f32")`
binds valid-prefix score, softmax and weighted-value reductions. Checked pinned
Qwen splits and teacher-forced decode match every logit and KV value, allowing
growing-width completed-prefill reuse in that composition. Generated state
remains private to matching `previous_response_id` continuation. Broad generated
fresh-promotion is pending a compatible numerical contract, not enabled by this
implementation.

## 1. Product contract

PLLM compiles a locked model and workload onto available parties, subject to
explicit privacy, numerical, resource and participation constraints. A party
advertises capabilities; an admitted execution assigns its logical roles.

The first planner chooses among supported compositions and placements. It does
not synthesize arbitrary cryptographic protocols, silently substitute checkpoints,
or equate a registered operator name with verified physical independence.
"Best" means best feasible candidate under a named cost model and bounded search;
reports state whether search was exhaustive, truncated, or inconclusive.

The trusted client owns planning involving private state, plaintext requests,
sampling, masks and decryption keys. An optional directory handles membership and
public offers. It receives neither prompts nor private cache-prefix identifiers.
The gateway remains inside the client boundary; no remote directory becomes an
application-facing plaintext inference endpoint.

## 2. Reuse the existing structure

| Existing surface | Change |
| --- | --- |
| `Model`, `Pipeline`, `Experiment`, `ExecutionBudget` | Keep source, exact composition and workload authority. Introduce versioned network deployment intent, not an `Auto` component that overrides the Pipeline. |
| `pllm.roles.RoleGraph` and topology components | Continue to define logical roles, channels and separation requirements. Available parties are a different axis from graph roles. |
| `pllm.deployment.RoleDeployment` / `RolePlacement` | Reuse assignment vocabulary. Current assessment is inspect-only; native placement validation and live role admission are required before a binding authorizes execution. |
| `pllm.providers` / `pllm.components` | Reuse installed method identities and provider discovery. A party advertisement references descriptors; it cannot install or activate code. |
| `pllm.search.SearchSpace`, constraints and Pareto machinery | Add bounded network placement search and typed planning policy. Preserve existing parameter search. |
| `pllm.compiler.compile` / `CompiledPlan` | Preserve the strict native compile API and opaque plan. Add a higher-level planning facade that uses it and validates deployment bindings. |
| `pllm.schedulers` | Keep protected tensor scheduling here. Network discovery and placement search do not belong in this module. |
| `pllm.metrics` / `pllm.evidence` | Extend typed costs, observations, selection outcomes and cohort matching. Preserve unknown and unmeasured values. |
| `runtime.build_roles`, SDK, gateway, benchmark | Add a network deployment launcher using the same role adapters, session protocols and semantic executor. Retain the local launcher as the test/development backend. |

### Versioning

Today `Deployment` permits only `kind="local"`; existing experiment serialization
must remain byte-identical. Introduce network intent in an explicit experiment
v3 schema, with `Deployment.network(network_id=..., network_spec_digest=...)`.
Existing v2 local documents continue to load and preserve their digests. A new
writer must not silently rewrite a v2 document as v3.

Model architecture and execution legality remain Rust/compiler-owned. Python
collects observations, generates bounded candidate assignments and invokes native
validation; it must not recreate a family-specific scheduler. New native records
need a versioned strict request/envelope, not extra unchecked keys in
`pllm.compile_request.v2`.

### Proposed code organization

Add responsibilities to existing packages, with thin CLI adapters:

```text
python/pllm/deployment/network.py       specs, offers, snapshots, discovery client
python/pllm/deployment/execution.py     binding, reservation client, opaque lease
python/pllm/deployment/role_placement.py versioned role-to-party assignments
python/pllm/search/placement.py         policy, request, bounded candidate search
python/pllm/compiler.py                 high-level plan facade; strict compile stays
python/pllm/plan.py                     immutable decision/plan inspection
python/pllm/runtime/party.py            party host and existing role adapters
python/pllm/runtime/directory.py        authenticated membership/offer service
python/pllm/runtime/servers.py          local/network deployment launcher interface
python/pllm/_cli/network.py             network command handlers
python/pllm/_cli/plan.py                plan command handlers
python/pllm/_cli/app.py                 parser/dispatch, existing commands extended
crates/pllm-types                      canonical binding/requirement records
crates/pllm-compiler                   assignment and execution legality checks
crates/pllm-python                     opaque native validation boundary
```

Paths for new modules are proposed. Export their public objects through existing
package facades; applications should not import `runtime` modules. Metrics and
evidence extensions remain in their current packages. Extract role-host lifecycle
code only when both local and network launchers exercise the shared implementation.
Do not create empty parallel service abstractions ahead of those callers.

## 3. Proposed SDK records and ownership

These are generic records, not new selectable research components.

| Proposed symbol | Home | Contract |
| --- | --- | --- |
| `NetworkSpec` | `pllm.deployment` | Immutable network identity, directory/static discovery endpoints, trust roots, allowed operators and local credential-reference names. No secret values. Static discovery works without a directory. |
| `PartySpec` | `pllm.deployment` | Local operator configuration: allowed role/component IDs, device/resource ceilings, concurrency, availability, artifact policy, permitted workloads and peer restrictions. |
| `PartyOffer` | `pllm.deployment` | Authenticated, expiring capability/resource offer from one party instance. Includes instance epoch and descriptor digests; declared capabilities are distinguished from checked capabilities. |
| `NetworkSnapshot` | `pllm.deployment` | Immutable offers plus directed link observations and artifact-residency claims, with source, timestamp, expiry, uncertainty and measurement scope. No credentials or one-use tickets. |
| `PlanningPolicy` | `pllm.search` | Hard compatibility/privacy/resource requirements; ordered objectives; allowed evidence levels; planning limits and freshness policy. |
| `PlanningRequest` | `pllm.search` | Explicit candidate Experiments and/or existing bounded SearchSpaces, source/semantic locks, workload and policy. Deduplicate candidates by canonical identity. |
| `PlanningResult` | `pllm.plan` | Selected Experiment, native compiled plan, role placement, cost vector, rejected alternatives, search coverage and all input digests. Serializable explanation; not a live reservation. |
| `ExecutionBinding` | `pllm.deployment` | Immutable commitment joining a selected plan, assignment, capability epochs and snapshot to one execution attempt. Revalidated by each participating role. |
| `ExecutionLease` | `pllm.deployment` | Opaque, non-cloneable, expiring live handle for reserved capacity and private credentials. Not part of JSON configuration, plan archives or ordinary telemetry. |

Extend `RolePlacement` through a versioned record with `party_id` and instance
epoch; a URL is not a party identity. `operator` identifies the assessed trust
domain, not a substitute for authentication. Offer identity, host identity and
operator identity remain distinct.

Party preferences include allowed preparation versus online roles, memory/disk
and egress ceilings, CPU/device limits, concurrency, model allowlists, willingness
to fetch artifacts, and maximum reservation duration. They are admission limits,
not suggestions a cost optimizer may override. Prices are optional observations
with currency, unit and validity; monetary ranking is unavailable when incomplete.

### Artifact and state classes

The planner must distinguish:

1. **Public immutable model artifacts:** source-locked tensors, quantized weights,
   scales, bundles and compiled representations; reusable by exact content identity.
2. **Client-private reusable state:** KV blocks, logits and exact prefix matches;
   retained locally and never advertised as public content hashes.
3. **One-use protected material:** prepared corrections, roots, verifier rows and
   correlation keys; bound to roles and inventories, never a transferable cache hit.
4. **Live capacity:** concurrent sessions, CPU/device allocation and buffers;
   reserved under expiring leases, not implied by free memory in an old snapshot.

Only inventory owner/client exchanges may expose the existing protocol's private
readiness information. The directory sees coarse capacity, not roots, tickets or
prompt-linked inventory identifiers. A restart changes party epoch and invalidates
claims about volatile state.

## 4. SDK lifecycle

Proposed signatures, for design review only:

```text
pllm.deployment.discover(spec: NetworkSpec) -> NetworkSnapshot
pllm.deployment.probe(spec: NetworkSpec, *, budget) -> NetworkSnapshot
pllm.compiler.plan(request: PlanningRequest, *, snapshot: NetworkSnapshot) -> PlanningResult
pllm.deployment.open_execution(result: PlanningResult, *, network: NetworkSpec,
                               credentials) -> context manager[ExecutionLease]
```

`discover` authenticates offers and reads existing observations. `probe` performs
explicitly budgeted active measurements. `plan` is deterministic and offline:
source metadata/semantic locks and required estimates must already be available.
Missing metadata produces an unresolved requirement; it does not download model
weights, prepare inventory, probe machines or reserve resources implicitly.

`PlanningResult.status` is `feasible`, `infeasible` or `inconclusive`; only a
feasible result with a native-validated selection can reach `open_execution`.
A truncated search with a valid incumbent reports `feasible` and incomplete
search coverage; it makes no optimality claim and requires policy permission to
execute an incomplete search result. Without an incumbent, truncated search is
`inconclusive`, not `infeasible`.
Exported files contain strict versioned records; loading revalidates digests and
native commitments rather than deserializing an opaque handle from arbitrary bytes.
Async discovery/admission and `AsyncOpenAI` use matching asynchronous context
management so cancellation releases partial leases and burns reserved material.

Proposed consumer flow, **not executable today**:

```text
network = NetworkSpec.from_file("network.json")
request = PlanningRequest.from_file("request.json")
snapshot = discover(network)
decision = pllm.compiler.plan(request, snapshot=snapshot)

with open_execution(decision, network=network, credentials=local_credentials) as lease:
    with pllm.OpenAI(execution=lease) as client:
        response = client.responses.create(
            model=decision.experiment.pipeline.model.model_id,
            input=private_prompt,
            max_output_tokens=32,
            temperature=0.0,
        )
```

`OpenAI(execution=...)` and the matching async client are proposed additions.
Existing explicit-endpoint construction continues to work. `execution` conflicts
with manual provider URL, precision or placement overrides. Responses API and
Chat Completions behavior stays shared with the existing client/gateway path.

One lease represents a bounded response attempt in the initial slice. Public
model caches and separately owned unreserved inventory may outlive it. A lease
does not allow reuse of response-reserved correction rows. On context exit,
capacity is released and unused response reservations burn as today.

### Policy example: structure rather than magic defaults

A request contains explicitly allowed Experiments/SearchSpaces and a policy with:

- Exact checkpoint/source lock and permitted numerical contracts. Default: fixed
  model and precision. Approximation/model substitution requires explicit candidates
  and matching quality evidence.
- Allowed privacy contract IDs, accepted operator assessments and mandatory
  separation relations. No scalar "privacy score" traded against milliseconds.
- Client retained-weight, cache-payload and working-memory budgets as separate
  fields; a payload bound is not a measured peak-RAM bound.
- A remote-work minimum with an explicit denominator, e.g. body-linear MACs or
  all-linear MACs. Whole-response CPU requirements need matching CPU evidence.
- Objective order, e.g. full-response latency, then covered all-link bodies.
  Each objective specifies phase, scope and evidence type. Predicted full-wire
  cost and measured protocol-body cost are not interchangeable.
- Cold/steady/session horizon and bounded request counts. Weight transfer cannot
  be amortized across an unspecified number of hypothetical future requests.
- `max_candidates`, `max_assignments`, deterministic search seed and observation
  freshness. Search exhaustion is distinct from exhaustive infeasibility.

Unknown quantities cannot satisfy hard measured-cost bounds. Soft optimization
may use explicitly admitted estimates, with uncertainty and missing terms shown.
Default comparisons are lexicographic or Pareto; arbitrary weighted sums of unlike
units are not the initial policy interface.

## 5. CLI design

Keep inference in `gateway`, party execution in `serve`, comparisons in
`benchmark`. Add only `network` and `plan` top-level command families.

**Every row below is proposed syntax, not a current runnable command.**

| Grammar | Behavior |
| --- | --- |
| `network inspect NETWORK` | Validate discovery/trust configuration offline. |
| `network parties NETWORK` | Retrieve authenticated offers, eligibility, status and expiry; no active performance probes. |
| `network snapshot NETWORK --output SNAPSHOT` | Save bounded canonical offers/observations for replay. |
| `network probe NETWORK --budget PROBE_POLICY --output SNAPSHOT` | Perform approved bounded link/kernel measurements and save their provenance. |
| `network drain NETWORK --party ID` | Authenticated administrative request: reject new reservations, finish admitted work. |
| `network leave NETWORK --party ID` | Drain then withdraw; does not move or reuse in-flight material. |
| `serve directory --network NETWORK` | Optional authenticated offer directory; no model/prompt processing. |
| `serve party --party PARTY_SPEC --network NETWORK` | Validate installed capabilities, host existing role adapters, join/renew offers and handle bounded reservations. |
| `plan create REQUEST --snapshot SNAPSHOT --output PLAN` | Offline search and native validation; record choice and rejections. |
| `plan inspect PLAN` | Show model, roles, parties, channels, commitments, costs, unknowns and coverage. |
| `plan explain PLAN` | Show constraints, alternatives and why they lost or failed. |
| `plan validate PLAN --snapshot SNAPSHOT` | Check compatibility/freshness against supplied snapshot; no capacity promise or execution. |
| `gateway --plan PLAN --network NETWORK` | Execute the fixed selected deployment after fresh admission; stale bindings fail explicitly. |
| `gateway --request REQUEST --network NETWORK` | Plan and acquire per-response bindings under one immutable policy; bounded replanning before execution only. |
| `benchmark run --request REQUEST --network NETWORK --compare-feasible` | Run scheduler and bounded feasible controls through the existing benchmark driver. |
| `benchmark run --plan PLAN --network NETWORK` | Fixed-plan measurement on registered parties. |

Existing `serve inference`, `serve preparation`, `gateway --local --experiment`,
`benchmark run --experiment`, `benchmark quality` and `topology inspect` stay.
`topology inspect` remains about the logical role graph; `plan inspect` adds the
specific physical assignment. No new `network-benchmark` or alternate chat client.

`--plan`, `--request` and existing `--experiment` are mutually exclusive selection
modes. `--local` cannot silently override a network assignment. Existing explicit
provider URL mode conflicts with network planning. Credentials come from configured
local stores/environment references, never tokens pasted into exported offers or
plan JSON.

Reuse global `--format human|json|jsonl`, `--quiet`, `--no-input`, `--no-color`,
`--dry-run` and output overwrite conventions. Dry-run does not reserve, prepare,
probe or start services; network calls intended by a discovery command are made
explicit, with offline file inspection available separately. Human and JSON errors
use the existing CLI failure classes/exit codes, adding reason codes such as
`NO_FEASIBLE_PLACEMENT`, `SEARCH_LIMIT`, `STALE_OFFER`, `UNKNOWN_REQUIRED_COST`,
`CAPABILITY_MISMATCH` and `RESERVATION_CONFLICT`.

The existing benchmark's optional SDK temperature control should gain a CLI
`--temperature` flag in the same slice. Preserve omitted-value behavior; archive
effective sampling, output caps and ordered cohort identity. No automatic change
from default stochastic sampling to greedy when a planner is enabled.

## 6. Planning, admission and failure semantics

1. **Validate inputs.** Resolve locked semantic operators, state, precision and
   explicit candidate component coverage. Reject pending components.
2. **Build feasible assignments.** Match role requirements, party preferences,
   assessed trust domains, resource bounds and installed native capabilities.
3. **Cost the schedule.** Charge artifact misses, conversion/snapshots, setup,
   preparation, online directed links, dependency rounds, queue/compute and
   retained state. Model critical paths and overlap; do not sum all link RTTs
   or assume all work overlaps. Prefix information is used client-locally.
4. **Select and explain.** Preserve a cost vector and rejected alternatives.
   Stable tie-break by canonical identity. Native code validates final legality.
5. **Reserve.** Acquire provisional role leases in a deterministic order with
   bounded TTLs. On partial failure release all acquired capacity; offers alone
   never guarantee availability. Each role validates plan and artifact bindings.
6. **Prepare and arm.** Ensure exact artifacts and one-use inventory are admitted.
   Commit all roles before online execution; preparation traffic and abandoned
   issuance are counted. Commit ambiguity aborts the attempt rather than allowing
   a subset to execute. Readiness requires all acknowledgements.
7. **Execute.** Existing authenticated, session-bound protocols run the selected
   schedule. No new role gets plaintext merely because its hardware is faster.
8. **Finish/abort.** Burn response-reserved material, release capacity and record
   sanitized scoped evidence. Retry has a new attempt and fresh material.

Initial adaptation is between responses, not arbitrary migration in a decoder
step. Automatic retry after streaming has begun must not duplicate delivered
tokens; report failure or use an explicitly defined resumable protocol.
Operator-separation checks include material history: an operator previously given
a root/correction/share must not receive its complementary material through a
role reassignment. Initial retry rules conservatively prohibit such reuse.

The saved plan binds source, composition, policy, cost-model and snapshot digests.
Live attempts additionally bind instance epochs and leases. Credentials and masks
never enter reusable compile keys. Public snapshot expiry does not invalidate an
otherwise reusable semantic compilation, but does invalidate its scheduling claim.

## 7. First implementation slices and acceptance

### A. Offline planner with a real selected execution

Add strict planning/snapshot records, static offers, native placement checks and
bounded enumeration using existing prepared, two-offset and client-only graphs.
Implement `plan create/inspect/explain/validate`. Replay fixtures must select
different placements under different client/link budgets. Compare native results
with a small exhaustive assignment oracle. Execute one selected local plan through
the existing SDK and benchmark; this milestone cannot end at inspect-only records.

### B. Authenticated network execution

Add experiment v3 network intent, party service, optional directory, signed or
authenticated expiring offers, reservations and shared local/network role adapter
interfaces. Keep remote execution fail-closed until live admission exists.
Test stale epochs, overcommit, forged descriptors, partial commit, drain, loss and
forbidden operator reassignment. Multi-process functional tests do not prove
independent operation; include separately operated hosts for deployment evidence.

### C. Measure planner quality in ordinary benchmarks

Add network/request selection to `benchmark run`; retain exact same sampling and
workload for feasible controls. Use explicit, finite scenario policies for slow
client links, fast worker links, cold/resident artifacts, queue load and prefixes.
Randomize/repeat run order and isolate or charge cache/preparation carry-over.
Record cost prediction error, selected-plan regret, rejected candidates and
reservation/retry work. Regret compares matched repetitions, not unrelated runs.

Measure actual per-role peak memory, compute and Linux wire traffic where
available. macOS `nettop` was rejected for undercounting audited bodies. A
full-wire requirement stays unmet until a reconciled meter exists. Never rank
an unknown setup or CPU term as free.

### D. Exact state and artifact locality

Implement batched continuation and then exact generated-state promotion, with
native state/budget contracts. Add artifact-granular distribution only after
transfer manifests and authority checks exist. These optimizations feed the same
planner rather than a separate serving path.

Initially assign complete supported logical roles, not arbitrary layer shards to
every joining machine. More parties provide placement choice and request capacity.
Per-region sharding comes only with executable state, material and transport
contracts; additional machines do not automatically accelerate one response.

## 8. Next experimentation direction

Research should unlock a useful rejected deployment or improve an existing Pareto
point. Preserve fixed checkpoints/no training. All proposed thresholds below are
future preregistered gates, not retrospective claims about saved evidence.

### E1. Squeeze repeated work and unnecessary issuance first

Current combined three-context cohort already reduced online covered bodies
647.60 -> 240.68 MB and setup-inclusive bodies 1,171.96 -> 547.41 MB. These are
workload-specific measured bodies, not a fresh-prompt or full-wire improvement.

The block-shared cache improved retained payloads, but one tiny branch makes 736
stage calls versus 32 fresh batched calls. Next experiment compares fresh prefill,
current sequential suffix and compiler-bound batched suffix at identical logits,
cache budgets and private prefix visibility. Track causal masks, positions,
sampled-but-not-evaluated tokens, cache pressure, calls, bytes and latency.
State representation/shape changes are new compiler contracts, not a callback hack.

In the same residual ledger, distinguish unreserved inventory retained for later
use from reserved rows burned on EOS/cancellation. Compare multi-request horizons
before reducing prewarm floors. Shorter reservations may reduce waste but may not
initiate preparation during active execution or reuse burned rows. Preserve
correctness first; promote only measured reductions with no hidden setup costs.

### E2. Weight boundary: exact locality before lossy extraction

Use the 23-placement frontier to measure baseline, QKV-only, output-only and
attention on real registered parties. Attention forecasts/measurements retain
87.69% body-linear work remotely but only 74.70% all-linear; policy chooses its
denominator explicitly. Measure both cold single response and fixed reuse horizons.

The next reusable optimization is **artifact-level content-addressed delivery**:
switching placement should fetch only missing public weight/scales objects, not
another monolithic client bundle. Bind canonical manifests, tensor orientation,
quantization, shape and source digests; verify local hashes before trusting residency.
Measure raw/compressed transfer, decoding CPU, source and native copies, eviction
and peak memory. Keep graph/numeric identity checks even when payload bytes match.
Neither private KV state nor one-use materials enter this public artifact cache.

Head placement must account for tied embeddings: moving a tied head remote does
not remove its client token table. Untied-head placement remains a workload/cold
delivery tradeoff, and new serving claims need checkpoint and request evidence.

Lossy low-rank extraction stays parked after its held-out failures. New exact
weight representations need a different mechanism, not another rank sweep:
prove arithmetic equality and measure weight/scale/index/decoder overhead before
checkpoint trials. Public weights are compressible artifacts; fresh masks and
corrections are not interchangeable with them.

### E3. HE boundary: reconsider a shallow linear region, not an entire decoder

The previous TenSEAL token-boundary screen failed depth/refresh: 2 or 4 sampled
serial products versus an optimistic 96 per forward, before norms and softmax.
Its 39+32 illustrative boundary traffic of 17.44/46.31 MB also missed the old
11.35 MB online tenfold gate. **Missing depth remains fatal even after relaxing
the tenfold goal.** Those boundary figures cannot be called a working HE decoder.

A narrower, testable question fits the new placement direction:

> With client attention and client nonlinear work retained, can an optimized
> exact encrypted public-linear gate/up or down stage replace its prepared
> transfer/preprocessing cost on a bandwidth-constrained network?

Specify client-key BFV/BGV-style integer input -> public-weight linear evaluation
-> encrypted output -> client decryption. Use actual Qwen shapes (896 -> 9,728
grouped gate/up; 4,864 -> 896 down), actual W8A8 public bounds, and both batched
prefill and one-row decode. This is a proposed new protocol/placement contract,
not a switch that an existing prepared stage may silently activate.

Packing and the plaintext modulus must preserve the entire integer dot-product
range or reproduce the exact ring semantics under a proved conversion. No
unproved modular lift or CKKS rounding assumption. The provider has no secret key;
decryption remains client-local and no unauthorized decryption oracle is exposed.
The first threat model is the declared honest-but-curious contract; malicious
output guarantees require a separate enforcing mechanism.

Account for rotation/Galois and relinearization keys actually needed, encoded
weight expansion, ciphertext serialization, key distribution, encryption/decryption,
CPU/device work, conversions, memory, fresh randomness and all input/output traffic.
Charge per-client key setup under explicit 1/10/100-response horizons. Same-client
batching may be a separate workload; do not combine unrelated clients under one key.
Run one actual packed projection before extrapolating; previous 32x64 BFV and
per-gate CKKS screens do not establish real-stage costs for this layout.

Use a first **25% targeted-region covered-body reduction** screen against the
same prepared stages. Also compute whole-response effects; 25% of a region is
not 25% of the decoder. Require exact integer parity on boundary/adversarial inputs
and pinned prefill/decode, an explicit client-work cap, and a matched aggregate
compute comparison against both prepared and two-offset alternatives. Unknown
full cost, unacceptable key/weight expansion or failed compute cap vetoes selection.
This is a bounded feasibility experiment; existing HE results give reason for
skepticism, not permission to project unmeasured wins.

Only if shallow HE is viable should a later study reconsider encrypted MLP or
token-boundary execution. That study needs a concrete refresh-capable backend,
complete nonlinear/state/feedback semantics and independently implemented method
logic in PLLM. Upstream research code remains specification/oracle material;
standard cryptographic dependencies remain separately identified. More parties
or threshold keys do not remove nonlinear depth or conversion costs for free.

### E4. Stop conditions and evidence integration

Keep the conventional complete-MLP garbling construction parked: known payload
alone was 131.39 GB for 39+32. Keep tested Smol fallback parked: 0/16 successes
before fallback and 1.56-1.58x projected cost per success after fallback. Neither
should enter network search as an eligible optimization.

Record each new experiment in existing evidence/metric contracts with source,
numeric, workload, sampling, network, implementation and environment identities.
Promote only an end-to-end region/topology with admission and lifecycle tests.
A planner may explain a research coverage gap; it may not select a cheap estimate
whose missing operations were priced as zero.

## 9. Immediate work order

1. Preserve and validate current cache/sampling changes as the working baseline.
2. Deliver slice A: deterministic snapshot planning plus one executed selected plan.
3. Deliver slice B and its multi-party lifecycle regressions through existing APIs.
4. Deliver slice C; start E1/E2 with fixed workloads and resource budgets.
5. Run E3 as a bounded independent feasibility gate, not a dependency of the network
   MVP. Spend further HE implementation effort only after real-stage costs pass.

Success is a demonstrably better choice on a changed network, not more command
names, parties, planned-component counts, or another isolated arithmetic primitive.
