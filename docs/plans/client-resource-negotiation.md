# Client-led resource negotiation

Status: **implementation in progress**. Configured ordinary SDK and gateway
requests now own discovery, native planning, one-response admission and lease
retirement. Sync/async, interrupted streams, cancellation and real CLI gateways
use the same owner. CPU `MlpPrefillTiling`, exact private role admission, bounded
installed execution choices, native allocation-owner pricing, fresh process/host/
container observations and concurrent local reservations are implemented. Separate
HTTP-role controls exercise automatic full/tiled/full reselection and infeasible
budgets. Per-client native matrix worker limits bind resident/paged imports and
cache reads. Compatible choices now span client delivery/paging, codecs, inventory,
scheduling and separately installed kernels/Preparation variants. Bounded explicit
component axes preserve interactions. Arithmetic estimates price selected public
ring/residue bounds, terminal rows and demand corrections with correct seed/correction
directions. Default ranking uses exchanges, online bodies, preprocessing bodies and
client weights; a zero hysteresis margin permits improvements in later objectives.
Dependent-exchange and directed-delay terms are available. A bounded process-local
cache now reuses immutable native admission and allocation facts; every decision
still checks current resources and offers. Optional `PlanningPrior` tables compile
matched network benchmarks into build/source/host/link-bound TTFT/decode forecasts
for configured SDK/gateway requests. Incomplete or mismatched coverage falls back
to geometry. Cross-workload phase models and measured WAN confirmation remain
later slices.
The end-state requirements below distinguish those remaining contracts.

## 1. Product contract

Ordinary `OpenAI` / `AsyncOpenAI` requests own negotiation automatically. The SDK
resolves the configured deployment, observes resources, derives bounded workload
and candidate choices, selects a plan, acquires leases and retires them. Callers
can override limits or selection without taking over that lifecycle.

Parties advertise installed choices and available capacity. The client compiler
selects an exact feasible plan; every participating party independently admits
its own role before any response material is issued. Explicit discovery,
`PlanningRequest` construction and `open_execution` remain advanced controls.

Memory is a feasibility constraint, not merely another score. A tiled plan with
slower time to first token (TTFT) can be the right choice when the full plan exceeds
the client's budget. Among feasible plans, optimize the caller's objective:

- Minimize request TTFT subject to a minimum decode tokens/second (TPS).
- Or maximize decode TPS subject to a request-TTFT ceiling.

Source, numeric identity, verifier strength, role trust requirements and required
remote-compute fraction are hard constraints. The SDK derives a bounded candidate
set from the configured deployment's admitted choices; callers can restrict it.
Each decision records its resolved defaults, search coverage and missing evidence.

The first vertical slice negotiates **ordinary full MLP versus explicitly tiled
MLP prefill**, on the existing Client / Preparation / Inference graph. The native
semantic compiler remains the authority for eligibility and execution.

## 2. What exists, what changes

| Existing foundation | Required extension |
| --- | --- |
| `NetworkSpec`, authenticated discovery, expiring `LivePartyOffer` | Bounded installed configuration/workload offers with per-choice resource and execution descriptors |
| `PlanningPolicy`, `PlanningRequest`, `compiler.plan` / `plan_on_load` | Fresh client-resource observation, complete allocation pricing, phase-aware performance objectives and evidence |
| `PlanningResult`, native placement validation | Explain full versus tiled choice; retain local evidence and rejected alternatives |
| `open_execution`, reserve/arm/release, `OpenAI(execution=...)` | Minimal role admission records, exact descriptor acknowledgements, fresh local admission checks |
| `OpenAI`, `AsyncOpenAI`, client settings and request lifecycle | Automatic request-scoped planning/lease ownership; optional policy overrides and request-scoped diagnostics |
| Internal compiler-bound MLP tiling | Ordinary immutable Pipeline selection, provider/session binding and complete resource admission |
| Gateway and benchmark planning inputs | Same selection and lease path, request-boundary refresh and matched validation |

Current limitations matter to the implementation:

- `PartySpec` installs one exact workload and up to 32 compatible Pipelines.
  Shared-engine alternatives can vary delivery, scheduling, request codecs and
  inventory; offset adapters also select their own input/output codecs. Kernel,
  prepared output-layout and Preparation-owner changes need separately installed
  compatible engines. Automatic choices preserve source, precision, topology,
  verifier, placement, reuse permissions and trusted artifacts.
  Its HTTP host currently admits
  at most 64 input and 32 output tokens. Larger buckets need explicit admission
  work; the 150/288-token research results cannot simply be selected on that host.
- `PlanningPolicy` checks peak-memory limits with the shared native allocation
  model and current process/headroom observations. Whole-request CPU costs remain
  unknown; required unknown costs reject.
- Existing `link_transfer_ms` prices arithmetic bytes / directed bandwidth. It
  explicitly excludes latency and overlap. Separate dependent-exchange/directed-delay
  terms now expose tile round trips. Matching whole-attempt priors can rank TTFT/TPS;
  unseen workloads and generalized phase forecasts remain unmeasured.
- Reserve v2 sends only exact execution/role contracts. Full planning explanations,
  observations, budgets and alternatives remain client-local.
- MLP tiling is an ordinary immutable scheduler component. Initial-prefill memory
  pricing reduces proven intermediates; full-stage continuation and residue-codec
  limits remain conservative.

## 3. Declarations and ownership

| Owner | Declaration | Visibility |
| --- | --- | --- |
| Client | Total client-process RAM budget, CPU worker limit, optional request CPU-work budget, private snapshot/cache limits, local disk limits | Client-local policy |
| Client | Current process RSS, post-reserve host headroom, CPU/build identity and concurrency, cache/state ownership | Client-local observation and admission |
| Client | Locked source/numeric contract, public input/output bounds, allowed Experiments, performance objective | Selected execution requirements go to relevant parties |
| Party | Installed source and component identities, exact supported workload/configuration choices, CPU/device support, memory/disk/session ceilings, peer restrictions | Authenticated expiring offer |
| Party | Current admission availability and queue estimate, instance epoch, offer expiry | Authenticated expiring observation; reservation remains authoritative |
| Links | Directed bandwidth and delay, shared party upload/download limits, uncertainty, observation age | Public network-cost inputs |

Keep requirements separate from observations. A caller's 256 MiB budget is not a
claim that 256 MiB is free. Two available threads are not a measured CPU throughput
or two CPU-seconds. CPU worker limits bind actual native execution; whole-request
CPU-work estimates require matching calibration/evidence.

Use the existing `PlanningPolicy.max_client_peak_memory_bytes` and
`max_client_cpu_ns` rather than introducing duplicate budget fields. Add a
versioned client-resource observation and explicit worker/disk/performance policy
fields. Dynamic client headroom remains outside the network's static trust roots.

Start provider choice negotiation with an explicit bounded allowlist of installed
configuration/workload descriptors. Compile and acknowledge the chosen entry
exactly. General parameter ranges can follow only with generic native legality and
resource contracts. An offer never authorizes installing arbitrary client code.

## 4. Between-party protocol

```text
Client SDK / local gateway       Inference host          Preparation host
          |                           |                         |
          |---- discover ------------>|                         |
          |---- discover -------------------------------------->|
          |<--- authenticated installed offers + capacity ------|
          |
          |  observe local resources; compile and rank locally
          |  select exact Pipeline, workload, role map and schedule
          |
          |---- reserve role contract ----->|                         |
          |---- reserve role contract --------------------------->|
          |<--- exact acknowledgements + expiring leases --------|
          |---- arm matched attempt ---------------------------->|
          |---- arm matched attempt ---------->|
          |<--- all roles armed ---------------------------------|
          |
          |---- authorize inventory ------->|
          |---- fresh preparation seeds ------------------------>|
          |                           |<---- corrections ---------|
          |                           |----- push ACK ----------->|
          |---- seal inventory ----------->|
          |<--- READY ---------------------|
          |
          |<=== online masked stages =====>|  Preparation idle
          |  local attention/state, nonlinear work and sampling
          |
          |---- complete/release ---------->|                         |
          |---- complete/release -------------------------------->|
```

Prepared seeds never reach Inference; correction bodies go directly from
Preparation to Inference. Preparation retains
its existing trusted, erasing, non-colluding role. Both provider roles and their
client/peer credentials remain distinct.

An optional directory returns approved host membership/offers. Discovery also
works directly from configured endpoints. Prompt text, token IDs, private KV,
cache-prefix identifiers and client resource-policy details stay client-local.
Selected model, workload bounds, tile layout and protocol traffic retain their
explicit public visibility.

### Exact admission contract

Introduce a versioned compiler-derived role admission record containing:

- Selected source, numeric, composition, semantic/execution schedule and role-graph
  commitments, with sufficient public inputs for independent native validation.
- Role assignment, approved peer identities/origins, host epochs and installed
  descriptor digests.
- Public workload/row/call/frame bounds; exact codec and tile layout; this role's
  memory, disk, CPU concurrency and material requirements.
- Fresh attempt ID, expiry and one common execution-binding digest.

Each host validates its own installed source, executor, peers and live capacity.
Its acknowledgement binds the same execution digest and its exact role contract.
The client checks all acknowledgements before starting preparation. Keep the full
`PlanningResult`, alternative hosts, private reuse assumptions and resource
observations in the client explanation. A selected-plan commitment must not be a
hash of private prompt contents or private cache identifiers.

Reserve capacity with short expiries, then arm only after every reservation
succeeds. This is a coordinated admission barrier, not atomic distributed commit.
If reserve/arm fails, release all acquired leases; lost acknowledgements are
handled through attempt-bound status/release and expiry. No material before the
all-armed barrier. Allow bounded fresh discovery/replanning before execution;
each retry uses a fresh attempt and retains the original constraints.

Requested lease duration must cover the admitted setup/response window within
each host's maximum. A longer predicted request rejects until a separately bounded
renewal contract exists; refreshing an offer does not extend an execution lease.

After issuance or online execution begins, failure burns the attempt's one-use
material. A retry requires a newly admitted attempt and fresh material; account
for wasted preparation and traffic. Cancellation, stream close, expiry and host
restart retire leases and outstanding work. A host restart changes its epoch.

## 5. Resource and performance selection

### Client memory

Share one model-neutral allocation contract across compiler planning, SDK
admission and benchmark admission. Price distinct owners and their overlaps:

- Runtime/import floor and existing process allocations.
- Tokenizer and bundle loading, public-file/page buffers, immutable matrix views.
- Preparation masks, tickets and transport frames.
- MLP tiles plus complete hidden outputs and attention workspace.
- KV, logits, retained continuation/prefix state and snapshot overlap.
- Concurrent requests, native scratch and allocator/runtime margin.

Use current RSS plus conservative **incremental** live-allocation envelopes;
deduplicate aliases and already resident owners. Check loading, preparation,
prefill and decode peaks, not just the MLP tensor payload. Check host headroom
separately and recheck immediately before allocation/reservation. An in-process
SDK must charge its host application's baseline; overlapping requests share a
local capacity owner rather than each spending the same free memory.

Median benchmark RSS is supporting evidence, not a ceiling. Admission estimates
and measured RSS remain separately labeled. Pressure monitoring can cancel owned
work; it is not an OS-enforced peak-RSS guarantee. A policy requiring that stronger
guarantee must reject backends without an enforcing execution boundary.

Tile-aware pricing may reduce proven intermediate owners only. Existing full-stage
codec checks and unsupported composition rejection remain active until separately
implemented and validated.

### CPU and WAN

Create scoped cost evidence bound to source/numeric/schedule, workload bucket,
runtime build, CPU/device, thread limit, lifecycle and network observations.
Retain origin, sample window, age, uncertainty and missing terms.

Compute a schedule dependency graph for preparation, first output and subsequent
decode steps. Price:

- Client and provider compute, queue/load uncertainty and material issuance.
- All directed setup, artifact, preparation, request, response and control bodies.
- Shared per-party uplink/downlink contention, not one full access link per peer.
- Dependent request/reply delays and acknowledged push rounds.
- Only overlap actually supported by the selected executor.

Do not add full stage CPU and transfer time twice when the schedule overlaps them;
do not assume independent tiles remove their gate/up-to-down dependency. A smaller
tile saves memory but adds exchanges. High-RTT WAN evidence must therefore be
separate from historical loopback/native-pacing cohorts.

Default TTFT objective covers request start through the first delivered token,
including required preparation and artifact misses. Report online TTFT separately.
Decode TPS uses the existing post-first-output `N-1` window; fewer than two outputs
cannot supply that measured metric. Predicted TPS/TTFT remain forecasts, not network
service guarantees. Unknown required resource costs or explicit performance bounds
reject. With default best-effort performance preferences and missing comparative
timings, retain an otherwise resource-admitted incumbent/default and record that
performance is unranked. Optional budgeted calibration is explicit work, charged
to the client, never an implicit expensive trial run.

### Decision and refresh

1. Filter by exact source/numeric/privacy/verifier/state compatibility.
2. Reject uninstalled, stale, over-budget or unpriced required choices.
3. Rank feasible choices by the resolved TTFT/TPS objective and deterministic ties;
   default best-effort selection may retain a feasible unranked incumbent when
   comparative timing evidence is missing.
4. Emit selected settings, conservative costs, evidence identity, rejected reasons
   and search coverage.
5. Recheck and reserve; refresh at request boundaries when observations expire,
   workload changes, capacity fails or client limits change.

Retain a feasible incumbent unless improvement clears switching cost and
uncertainty. Never retain an infeasible incumbent. A request's plan is fixed once
issued. Between-request switching reuses private state only through the existing
numeric/state/ownership checks; otherwise price a full re-prefill. Public artifact
reuse and one-use material issuance remain separate.

## 6. CLI experience

The default client workflow uses ordinary configuration and a gateway launch:

```sh
pllm gateway
```

Configured model/deployment/trust and credential references are resolved through
the existing client settings path (`PLLM_CONFIG` or the standard PLLM TOML file).
Add shared network and optional planning settings there. The user configures the
deployment once; the SDK inside the gateway negotiates lazily for each request.
Explicit `--config client.toml` selects a settings file. Existing `--network` and
`--model` flags can override their corresponding configured defaults.

Automatic negotiation, network settings, peak-memory policy, per-executor native
worker limits and configured prior-performance preferences are implemented.
The performance ceilings below remain proposed end-state semantics.
Files below are operator-created inputs, not fixtures supplied by this plan.

Provider operator, on each host:

```sh
pllm serve party --network network.json --party inference.json
pllm serve party --network network.json --party preparation.json
```

Each party file declares its installed choices, capacity and permitted peers.
Credentials resolve through local references; secret values are absent from argv
and exported JSON. Bind host/port to that operator's configured service origin.

Optional policy overrides belong in the same client configuration, with matching
SDK fields. This **proposed settings excerpt** requests a 256 MiB client-process
ceiling, two native workers and fastest first output among candidates forecast to
sustain at least two decode tokens/second:

```toml
[client.planning]
max_client_peak_memory_bytes = 268435456
max_client_native_threads = 2
min_decode_tps = 2.0
objectives = ["request_ttft_ms"]
```

Omitted fields inherit resolved defaults; the ordinary user needs no planning
document. Memory and native-worker fields and the settings table are implemented;
TPS bounds and TTFT objectives require further configuration. An inverse
objective minimizes `decode_token_ms` under `max_request_ttft_ms`.

Advanced client workflow, inspect or reproduce a decision:

```sh
pllm network snapshot network.json --output snapshot.json
pllm plan create request.json --snapshot snapshot.json --output plan.json
pllm plan explain plan.json
```

`request.json` explicitly constrains candidates, source/workload locks, limits and
objective. It is an advanced replacement for SDK-derived planning inputs.

Proposed `plan create` captures a bounded local
resource observation through the same public SDK record, then calls the pure
planner. The exported decision retains this observation for replay. It grants no
capacity; live admission refreshes it. Snapshot replay never silently incorporates
new observations into the old explanation.

The planning CLI's own RSS is not the gateway/SDK runtime floor. Bind forecasts to
the intended client runtime/build and its floor evidence; execution observes the
actual hosting process again. A saved plan never transfers another process's free
memory or cache ownership.

Advanced callers can supply that planning request to the same gateway:

```sh
pllm gateway --request request.json --network network.json
```

The intended `--request` behavior uses the ordinary automatic lifecycle with the
caller-supplied constraints. Existing `--plan plan.json` pins an exact choice and
rejects when fresh admission fails. Both use the same SDK path.
`plan explain` shows resource scope, objective, selected hosts/layout, binding
digests, predicted phase costs and rejected alternatives with reason codes.

## 7. SDK experience

### Default: ordinary inference

**Implemented default behavior using the existing inference call shape:**

```python
from pllm import OpenAI

with OpenAI() as client:
    response = client.responses.create(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        input=private_prompt,
        max_output_tokens=8,
    )
```

Deployment membership and credentials come from the user's existing configured
connection/trust context, extended with network settings. Supplying the model per
request is optional when configured. Construction validates settings; discovery,
source binding and capacity reservation happen lazily when needed. The application
does not construct a `PlanningRequest`, resource observation or execution lease.

The intended complete response lifecycle combines the implemented ownership and
memory admission with later workload-bucket, worker-limit and phase-cost work:

1. Resolves the chosen model, authorized deployment and immutable numeric/trust
   contract; observes local resources and fresh/cached authenticated offers.
2. Performs bounded local input sizing, chooses an admitted workload bucket and
   generates supported candidates through the existing bounded optimization path.
3. Derives conservative memory/CPU limits from process/container limits and host
   headroom after reserves, intersected with any caller limits. Uses existing
   native worker bounds and charges concurrent requests against one local owner.
4. Applies the default best-effort request-TTFT preference with decode-time
   tie-breaking, or the caller's objective and hard performance constraints.
5. Compiles, revalidates and admits the plan; reserves/arms all roles; issues fresh
   preparation material; streams or returns the ordinary response.
6. Releases response-owned leases on completion, cancellation, stream close or
   failure. Client shutdown drains owned attempts. Async follows the same rules.

Bound and cache source metadata, offers, compilation and evidence under their
identities/expiries. Fresh capacity and resource admission still happen per
attempt; cached decisions do not keep execution leases alive. Replan only on
relevant changes or admission failure, rather than rerunning the full search for
every token/request. The ordinary timeout includes negotiation and preparation.

Default search stays within the configured source/numeric/privacy contract and
permitted placement. New client-owned body weights or private-state reuse retain
their existing explicit permissions. The configured Experiment seeds the installed
full/tiled schedule choices. `candidate_selection="pinned"` fixes its Pipeline;
placement/resource management remains automatic around that choice.

### Optional: override policy, keep automatic ownership

**Proposed policy-default/API extension:**

```python
from pllm import OpenAI
from pllm.search import PlanningPolicy

client = OpenAI(
    planning=PlanningPolicy(
        max_client_peak_memory_bytes=256 << 20,
        client_worker_threads=2,
        min_decode_tps=2.0,
    ),
)
```

The caller specifies only changed preferences. Version the policy interface so
the SDK supplies client identity, evaluation time and bounded search defaults
during resolution; these are currently mandatory low-level constructor fields.
Export the fully resolved canonical policy/request for reproduction. Preserve
existing v1 document replay rather than changing historical identities.

Resolve defaults in order: explicit SDK/CLI policy fields, explicitly configured
environment overrides, client configuration, versioned automatic defaults. Only
specified fields override inherited values. Resource availability and immutable
Experiment/plan constraints remain hard intersections; contradictory pins fail.

Opt-in request-scoped diagnostics under `client.runtime` expose the ordinary
immutable `PlanningResult`: effective limits and their origins, selected plan,
estimated costs, evidence and rejection reasons. Keep diagnostics tied to the
response/attempt, bounded and prompt-free, including concurrent requests and
failures. Default requests emit neither a negotiation transcript nor a stream of
planning progress messages.

### Advanced: explicit discovery, planning and lease ownership

Retain the lower-level APIs for custom planners, deterministic replay and external
orchestration. `ClientResources.observe()` is explicit for these pure-planning
callers. The ordinary SDK captures the same observation automatically:

```python
from pllm import OpenAI
from pllm.compiler import plan
from dataclasses import replace
from pllm.deployment import (
    ClientResources, NetworkSpec, discover, open_execution,
)
from pllm.search import PlanningRequest

network = NetworkSpec.from_file("network.json")
request = PlanningRequest.from_file("request.json")
resources = ClientResources.observe()
request = replace(request, client_resources=resources, policy=replace(
    request.policy, allocation_model="native_shared_v1",
    evaluated_at_ms=resources.observed_at_ms,
))

decision = plan(
    request,
    snapshot=discover(network),
)

with open_execution(decision, network=network) as lease:
    with OpenAI(execution=lease) as client:
        response = client.responses.create(
            input=private_prompt,
            max_output_tokens=8,
        )
```

`private_prompt` is application-owned and never passed to discovery. Local sizing
checks the request against its declared bucket before material reservation.
`open_execution` validates the exact selected plan, acquires provider leases and
owns their release. Non-feasible decisions cannot open a lease. The ordinary
automatic SDK path also refreshes local resource observations and reserves process
headroom; callers using raw leases currently own their local allocation policy.
Bringing that same local reservation owner into the raw lease API remains part of
the end-state contract. Async discovery/open/client paths retain cancellation cleanup.

`plan_on_load` accepts the same resource input and policy. Its bounded,
source-preserving candidate generator supplies the default SDK path as well as
explicit planning. CLI, gateway and benchmark share defaults, policy resolution
and lifecycle ownership with the SDK.

## 8. Implementation order and acceptance gates

| Slice | Deliverable | Required proof |
| --- | --- | --- |
| 1. Automatic SDK lifecycle | Ordinary responses own discovery, existing prepared planning and leases; shared settings/default resolution; optional explicit controls | Configured `OpenAI()` and `AsyncOpenAI()` match manually leased controls; laziness, stream close, cancellation, failure, concurrent admission and client shutdown retire correctly; gateway follows the same path |
| 2. Executable choice | Promote compiler-bound MLP tiling into an ordinary immutable component contract; full/16/64-row candidates; exact session binding | Generated Qwen2/Qwen3/Llama full logits and KV parity, separate-role SDK/gateway requests, failure burns material; unsupported graphs reject |
| 3. Resource admission | Shared allocation-owner model, automatic client observations, optional limit overrides, CPU worker binding and concurrency admission | Tight budget admits a tiled choice and rejects full; roomy budget admits both; insufficient budget rejects before client tensor loading/material; override precedence and explicit Experiment pins hold |
| 4. Negotiated hosts | Versioned bounded installed choices/workload buckets, minimal role admission and matching acknowledgements | Stale epochs, mismatched schedules, exhausted capacity, partial reserve/arm, lost ACK, restart and cancellation retire correctly; concurrent attempts cannot overspend |
| 5. WAN objective | Phase cost evidence and dependency/contended-link model; default best-effort selection and optional hard performance bounds | Low/high RTT and bandwidth scenarios select the best eligible control within stated uncertainty; absent required evidence rejects; missing optional timing keeps a feasible unranked incumbent with accurate diagnostics |
| 6. End-to-end confirmation | Fresh-client matched full/tiled cohorts through ordinary inference, automatic/overridden budgets, WAN conditions and request-boundary refresh | All seven serving metrics, exact output/body identities, predicted versus observed costs, selection regret, cleanup and no observed new swap; retain separate cohorts |

The first provider slice can enumerate one small admitted workload bucket. Broader
150/288-token confirmation requires explicit larger-bucket and tile-call-budget
admission before launch; raising a JSON limit alone is insufficient. Price loaded
variant descriptors, shared immutable weights and per-lease allocations together.

Use synthetic fixtures to verify decision boundaries; they do not certify physical
RSS or WAN performance. Real measured comparisons use fresh isolated clients,
matched controls and independent confirmation. A 16% memory improvement remains
valuable when it makes a resource-feasible plan; promotion does not require a
universal latency win or tenfold gain.

Implementation homes: `pllm.deployment` for offers/resources/leases,
`pllm.search` for policy and bounded selection, `pllm.compiler` and
`crates/pllm-compiler` for semantic/native admission, existing capability families
for the selectable working-set component, and the shared runtime/CLI/benchmark
paths for execution and evidence. Version new records explicitly; existing
configuration and evidence identities retain their original meaning.

References: [network foundation](network-inference.md),
[working-set frontier](working-set-frontier.md),
[MLP evidence](../research/mlp-working-set-reference.md).
