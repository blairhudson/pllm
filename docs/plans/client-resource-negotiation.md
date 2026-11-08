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

#### Bounded observer lifecycle (design only)

The bounded observer design is SELECTED prospectively, not implemented or authoring-ready.
Callback allowance R is finite, prepaid and NONRENEWABLE; exhaustion permanently refuses
observations/admissions, with NO weaker fallback. Existing thresholds, caps and HOLD stay unchanged.
Numeric R, rates, API/owner types, source/test authoring and execution remain unapproved.

The selected embedding responsibility requires an authentic ALREADY FUNDED external entry and
retained return destination, covering bootstrap/query/result controls through CONSUMPTION and actual
retirement; original ledgers and backing remain charged. Narrow checked-publication exclusions are
NOT selected. Establishing the first external issuer is a trust boundary, not recursive self-funding.

| Category | Existing contribution or missing semantic join |
| --- | --- |
| Existing native primitives | [original_source.rs](../../crates/pllm-compiler/src/original_source.rs):108–145, 836–848 provides actual Context/config custody, distinct from controller-transition funding. Prepay (751–786) uses the same budget capacity and original CUMULATIVE LOGICAL W with failure burn, NOT CPU time. Commit Drop/install concerns its own source ledger, not Python F/P. In [shared_allocation.rs](../../crates/pllm-garble/src/shared_allocation.rs), `OwnedExternalCharge` preserves original counter identity; `OwnedCharged` orders direct payload destruction before charge destruction; this is not an all-alias backing-retirement witness. Charge alone is not backing custody. These are STRUCTURAL CONTRIBUTIONS, not a Host-purpose grant. |
| Existing Python capacity retention | `_CleanupSlot` supplies a process-rooted charge but observes BEFORE admission and binds payload LATER. `_MemoryReservation`'s explicit `retain_failed`/`drain_failed` protocol retains original token/custody and seals its backend attempt against retry after uncertain failure; not pre-entry Host funding or a guarantee for ordinary `__exit__`. Partial-publication/release hazards remain; no standalone keeper/registry or slot self-bootstrap follows. |
| Unrepresented semantic join | The authentic EXTERNAL original owner must authorize Host-observer PURPOSE, distinct original work W/backing-byte B extents and ledger identities, retained destination/witness, and the specific calling process's monitor identity/epoch and exhaustion. Token lookup, parser, budget pointer, capacity or disk coordinates are not authority substitutes. Current API binding is ABSENT; no concrete original-issuer binding or complete funding is proved. |

For entire `LocalTopology` constructor/startup coverage, entry must already exist before construction
(servers.py:308–348): helpers, startup callbacks, imports and preparation precede sampling; line 1092
DEFINES observation, while 1110 CALLS it. Qualification benchmarking also samples before model
inspection. Earlier caller preflight needs a prior entry too; no universal outermost-caller boundary
is proved. Later qualification/graph reservations are not retroactive payment.

Conditionally, when an authentic OUTER entry ALREADY exists, preserve the actual original root,
W/B extents, destination and witness BEFORE in-scope construction/preparation, validate observer
purpose/instance INSIDE that funded region, then initialize or qualifiedly reuse the observer,
or deliver a FUNDED refusal in the ORIGINAL destination.
Without ANY authentic outer entry, controlled-contract embedding MUST NOT CALL: library validation
or refusal cannot fund itself.

**SELECTED DESIGN ONLY: host provider and reserve-only issuer**
- Proposed native owner: `crates/pllm-python/src/host_entry.rs` (NEW PROPOSED file, not existing/adopted);
  integrate with existing `crates/pllm-bench/src/host_memory.rs`. Reuse/extract method-neutral cumulative W ledger/checks
  from compiler `original_source.rs`, WITHOUT source-purpose authority; shared-allocation counters/charged payloads prove neither all-alias backing nor payment.
- Four PRIVATE opaque roles express SELECTED ownership semantics; concrete names/types and bank representation remain PROPOSED UNADOPTED: `HostOwner` is the actual retained native identity-bearing original host-bank allocation,
  not reconstructible from labels/counter snapshots, with distinct B accounting/genuine custody, process/epoch and original outcome cells; `HostEntry` records original-bank debit, operation and destination.
  `HostObserver` immutably binds original lifecycle owner/purpose/process/epoch and actual lifecycle W-debit witness, nonrenewable R, backing/terminal D, monotone callback use and sticky exhaustion; query/failure/partial initialization cannot reset owner/epoch/bank or replenish R.
  `HostOutcome` retains success/refusal/unresolved resources through consumption/qualified cleanup; the original `HostOwner` itself retains outcomes, not a standalone registry. Preallocated cells are not authenticated/live destinations until actually bound.
  No public numeric self-authorization or standalone keeper. Host-purpose genesis/entry semantic DESIGN is SELECTED—not implemented authority or proof of actual funding; private roles do not guarantee protection from hostile native code.
  Native-authoritative WHOLE publication with nonauthoritative Python projections is additionally SELECTED DESIGN ONLY; the inline-M profile remains UNADOPTED, and no concrete primitive/storage layout or actual funding is proved.
- Ordinary Python route: NEW PROPOSED `python/pllm_host_entry/` bootstrap package alongside `pllm`, using normal Maturin
  `python-packages`; optional normal console alias `pllm-controlled -> pllm_host_entry:main`. Names are NOT adopted.
  The pre-import obligation is SELECTED; the unadopted concrete route locates sibling `pllm/_native` by installed layout WITHOUT importing/querying `pllm`, with ctypes loading restricted exports BEFORE `PyInit__native`/`pllm` import.
  Single SUPPORTED NATIVE LIBRARY IMAGE/PROCESS identity and funded dispatch are SELECTED obligations: ctypes and later Python import must use the same pinned image. Dispatch current CLI/permit SDK imports only after actual first-entry debit and authenticated retained-outcome binding.
  Importing `pllm` first is unsupported as a CONTROLLED route; legacy stays unchanged. No new build system or bespoke attestation/qualification driver.
  SELECTED child fail-closed obligation: parent-controlled dispatch does not qualify child processes or existing `-m pllm` launches. Each child requires authentic funded entry, original-owner/destination binding
  and qualified native-image identity BEFORE covered imports; otherwise refuse before unsupported launch. Parent frames, charges, banks and observer state imply no inherited authority.
  Child issuance/IPC/grant transport remain unadopted.
- SELECTED narrow FIRST-ISSUER GENESIS direction places minimal bootstrap/stdlib loader, native-image load and restricted original-owner
  establishment at a TRUST BOUNDARY, not already-paid runtime. Arbitrary initializers, framework packages, native Python registration,
  config and benchmark preparation are not silently free. This precise boundary requires concrete loader/image/initializer audit; ABI/implementation remain unadopted. Creating a bank proves no payment, and genesis failure/reentry must not create a replacement bank.
  Covered imports/preparation/query/result controls THROUGH consumption follow real W debit; B charge alone is not custody.
- SELECTED DESIGN ONLY reserve-only issuer requires authentic unused ALREADY-PAID E_control, bounded native Request, authentic BodyAuthority and live original D_control/D_body; missing/exhausted parent slice means NO CALL.
  Consume one finite operation-specific issuance slice BEFORE inspection/quote/authority/metadata/reservation/publication/refusal/consumption; it must already fund issuance, refusal, consumption and metadata-retirement tail. Parent W pays validation/quoting.
  Control-entry authority alone does not confer lifecycle creation authority. Only first-root establishment is genesis; later issuance is not. Each entry preserves ITS OWN bank, not a universal payer.
  Retain actual acquired handles and partial outcome state, not just charges; partial W/B commit remains in original custody, with no invented rollback/substitute bank.
- SELECTED issuer-core/parent-metadata design: allocate/stage metadata under an EXPLICITLY AUTHORIZED ORIGINAL PARENT B context with an owning guard and authentic body-owner references, NOT body-budget storage.
  Metadata remains under its actual original parent owner/root; D_body holds an authenticated loan/reference retaining parent guard/counter identity, not transferred credit/custody or a new keeper. Contract requires the prepaid parent metadata-retirement tail to outlive issuance-outcome consumption until actual retirement.
  Under the SAME actual body-ledger guard, parent-paid checks establish usability and checked-add/ceiling BEFORE commit: SELECTED healthy ordinary overflow/over-limit refusal yields QuotaRefused with NO FURTHER body W debit or health change; prior parent fees stay spent, and poison is never cleared.
  SELECTED success semantics irreversibly debit the original body bank ONCE; later body-B refusal cannot refund W. Poison/uncertain commit retains originals and sticky unresolved state, never reset; its uncertainty failure domain remains UNSELECTED.
  This narrows selected Host quota-refusal design, NOT existing PrimitiveAttempt::debit (original_source.rs:162–171 burn-before-check stays unchanged). Reuse/extract guarded arithmetic, not a recursive lock-taking adapter.
  Reserve original body B and a prearmed PENDING resource holder only: no future body backing is claimed owned. Publish authority/holder in D_body, retaining the D_control issuance outcome through consumption/uncertainty; staged ownership survives failure.
  Arbitrary body builders, foreign construction and initialization run ONLY LATER under E_body, outside this core; attach actual backing owners before publication.
  Issuer-core/body separation and the parent-tail requirement are SELECTED design obligations requiring proof, not existing implementation or adopted APIs.
- SELECTED Host-only logical-helper boundary: scope is ENUMERATED issuer-core PREPARATION/ADMISSION metadata-B and body-B reservations ONLY, not all Host operations.
  Budget.reserve_bytes (shared_allocation.rs:112–120) used.fetch_update counts ONE semantic atomic invocation, including ONLY unchanged reserve_update (28–33) captured bytes/limit, checked-add/ceiling Some/None and implementation reinvocations; changed callbacks require review.
  Preparation/project arithmetic, branches, copies, calls, map_err/resource_error and previous+bytes remain separate; admitted_peak.fetch_max (120–124) and release fetch_sub (129–132) are separate atomics.
  Count identified HostMutex lock invocations and EACH actual guard retirement, including poison, separately; wrappers/moves/unlock do not double-count. Identified STANDARD allocation invocations abstract ONLY private standard internals; requested size, actual capacity/backing, admission, initialization, copying and cleanup remain separate.
  OwnedExternalCharge.try_build (669–685) does not collapse an arbitrary builder; body builders stay outside the core. Iterator/Eq vocabulary applies only to explicitly identified Host operations.
  Finite outer-call enumeration COULD support finite LOGICAL demand, not actual CAS-retry, CPU/RSS, wait, allocator-work, progress or termination bounds; no PUMA/Maverick transfer follows.
  Counts, rates, complete W/B extents, authentic actual prepayment, custody and publication proofs remain prerequisites for complete fees or source authoring.
  Concrete bank representation, API/types/ABI/names remain UNADOPTED; uncertainty failure domain, numeric R, rates and endpoint remain UNSELECTED. No source/test authoring, execution, cap or HOLD grant follows.
- E/L may default to the same owner; distinct E_query requires an explicit authentic Host-query(E, L, epoch) grant and fee-owner map: no credit/custody transfer, new R or callback double debit;
  current native same-budget source checks are NOT a foreign-ledger adapter. R = 0 refuses before acquisition; refusal cannot fund/own
  an old unqualified source. Preserve genuinely existing owners/destinations only; no new ordinary legacy `__exit__` retry policy.
- Before source authoring, prove finite issuer/bootstrap/import/child/result/refusal quotes/rates and backing overlap; authentic retained
  original destination/debit and partial-failure publication; identical native image/state across ctypes/Python import, pin/unload tail;
  and complete F/P mutations. Endpoint/destination liveness remains UNSELECTED; a process-local root does not survive exit.
  Actual genesis/image qualification is not proved. No numeric R, rates, API/type/endpoint, source/test authoring, code/execution, cap or HOLD adoption follows; independent review remains required.

**UNADOPTED inline-M issuer-only profile**

- C is actual exclusive FIXED-CAPACITY storage containing ALL inline M metadata, Gq pending-guard storage and authenticated native C/Dctrl/Dbody cells. Original P guard Gc covers WHOLE actual C backing.
  Gc, prepaid P-tail permit and reference-release controls survive OUTSIDE C in EXISTING funded P-rooted control storage: no new keeper/registry/backedge/cycle or reading/dropping through destroyed C.
- An EARLIER authentic ALREADY-PAID P entry E_parent funds COMPLETE C construction/admission/initialization, failures/partials/unreturned C and eventual tail BEFORE construction starts; gap-free actual custody handoff is required.
  C cannot self-fund; later provisioning is NOT genesis. Capacity covers ALL admitted fixed native Requests; a mismatch after funded A/Q causes P-paid refusal, never resize/fallback/another P charge.
  Post-F C preparation may ONLY suballocate authentically COMPLETED PRE-F original P capacity whose purpose/remaining extent covers simultaneous C/temporary/Hparent/guard peak WITHOUT new/increased capacity publication. E_parent W prepayment/Gc/native-used counters are NOT capacity authority; no metadata/cleanup exemption permits fresh observations/body admission, R renewal or bank reset.
  Pre-import qualification does NOT require a Python `_capacity` token: external original admission requires actual owner/extent/purpose/process/epoch/retained destination AND an actual ordering witness binding completed admission to the SAME controlled F domain, not unrelated timestamps/epochs/bank birth/counters/labels. Covered fresh/increased external OR Python admission after F refuses at ordered actual joint publication; RLock/snapshots alone are insufficient.
  Outside Python `_capacity`, an actual original reservation/subledger relationship must encumber the WHOLE outstanding native peak; disjoint-pool labels/observed bytes/fresh budgets do not prove coverage or prevent double spending. A preallocated slab PLUS default-Global C allocation is not one backing/suballocation; no new allocator is adopted. Unknown coverage/publication retains original owner/witness/charges/backing/funded resolution tail, without inferred admission/rollback/new root/replay; external-owner/admission → native-P/resource-accounting/F binding remains uninstantiated and a proof gate.
  UNADOPTED Host.prepare-holder purpose requires actual external owner, real original P work ledger/B counter, retained destination and ALREADY-FUNDED existing P-owned outcome cell Hparent; metadata/control purpose alone grants no observer-lifecycle creation authority.
  Conditionally under the selected narrow genesis direction, initial establishment binds trusted external owner, actual original-P ledger and live root controls BEFORE committing E_parent; Hparent failure custody/authority already exist, with establishment/failure cost covered independently of the NOT-YET-COMMITTED E_parent. Failed/reentered genesis preserves original-bank identity, spent work and applicable failure state: no replacement bank/reset/later-credit bootstrap. Subsequent preparation is ordinary original-P entry, not an actually qualified root/image/ABI proof.
  Conditional flow: paid E_parent before quote → Hparent receives ONE original P charge_external with complete C payload ONLY (including inline Gq storage); the helper ITSELF adds OwnedExternalCharge wrapper overhead for Gc ONCE (or authentically reuse the original guard already covering WHOLE C, without duplication/reparenting) → ONE standard C allocation/capacity check/initialization under gap-free Hparent custody, including failed/unreturned C → Gc/tail controls outside C → unused funded E_control BEFORE core A/inspection. Actual Gc/external-control physical storage is separately funded OUTSIDE C; logical wrapper charge is not another physical backing, and no body backing is claimed owned now.
  Before core entry, actual authenticated coverage binds the specific C/original Gc/Host-issuer purpose/outstanding obligations and actual operation/extent/lifetime/capacity to Hparent's EXISTING retained original tail permit; E_control BORROWS coverage without consumption/transfer/duplication/refund. Shared P or equal amounts prove nothing.
  Hparent/permit outlive parent issuance/consumption and ALL C/unreturned-child/outcome/retirement obligations; ambiguous publication retains originals without debit replay. Any Gq-to-Gc reuse requires explicit original-authority/capacity/purpose/lifetime/fee joins: ONE release, ONE fee, not relabeling.
  Actual constructor/handoff/capacity/helper scope must independently qualify; the enumerated issuer-helper boundary does not blanket-approve preparation allocation or prove actual funding/native APIs/rates/authoring readiness.
- Authority binds actual current caller/process/image/epoch and noninheritable original E_control roots/paid extents through actual-use witness or trusted lifecycle through use, not PID/address/digest/fork copies; fork confers no child-bank/reset.
  Current OS checks require paid qualified slots, not assumed qualification. External noncyclic bindings keep ALL borrowed actual P/Q owners, ledgers, HostOwner/destinations alive through ALL loans/tail; Gq counter retention alone is insufficient.
  Missing lifetime bindings make this profile INAPPLICABLE; additional reference retains are stage-local work, not bounded by the wrapper Arc count.
- Fixed native Request has operation/purpose/process-epoch, owner/destination references and bounded W/B/capacity fields; no callbacks/strings/arbitrary traits/body code.
  Pipeline: A consumes unused already-paid slice/arms existing C → Q pure grant/identity/bounds quote → M bounded IN-PLACE request writes → W same-Q-work-guard healthy checked preflight → B one Q chargeExternal → P single NONCLONE authenticated C loan/bounded authenticated borrow drain → Dctrl consumption → qualified tail.
  Ordinary quota refusal changes no used W/health; success commits W once irreversibly, including after B refusal. Early initialization AND distinct later request writes count if performed, never twice for the SAME event.
  No core metadata allocation/P-B reservation/try_clone/try_build/resize/reentry or recursive lock-taking debit. Actual future builders/backing remain outside the E_control core under E_body.
- `crates/pllm-garble/src/shared_allocation.rs:425–433`: payload.checked_add(size_of::<OwnedExternalCharge>()) → reserve → successful Arc clone. Overflow means zero reserve/clone; zero payload STILL charges guard metadata.
  Wrapper/project arithmetic/refcounts remain separate. Granted counter TOTAL includes metadata; inventory whole C physically under P once. Q reservation is NOT future body backing, additive physical accounting or RSS proof.
- CONDITIONAL CORE call slots: chargeExternal/fetchUpdate/successful-WRAPPER-ArcClone ≤ 1 each; Q lock/actual guard retirement/publication/consumption ≤ 1 each; successful peakFetchMax and actual Gq-release fetchSub count separately.
  No CORE metadata allocation/parent-B reserve does NOT mean zero WHOLE-CALLER allocations/reservations. Outer P-paid C provisioning/admission/staging/retirement/Gc/reference/drop/root vectors remain separate and uninstantiated, not a free prefix.
- Tail requires original Q release authority if Gq exists, ACTUAL body backing retired, ALL borrows drained and all C-owned report/outcome obligations qualified complete.
  Then Gq releases ONCE → actual inline M/C destruction/deallocation → original Gc release OUTSIDE C → surviving outer bookkeeping → P-tail/root last. Dctrl consumption alone proves neither drain nor backing retirement.
  Ambiguous branches retain actual originals without retry or a newly selected failure domain. Proposed P-paid Gq reservation/release fee map remains UNADOPTED, requiring Q authorization and a body quote EXCLUDING release; body/lifecycle pays actual future builders/backing retirement.
- UNADOPTED disjoint event sets: V_pre[P] (C provisioning/initialization/admission/failed handoff), V_issue[P] (reached core writes/work-debit machinery/guards/reservation/publication/consumption), V_tail[P] (authorized Gq release/actual C retirement/Gc release/surviving controls), V_body[Q/lifecycle] (future builders/backing retirement, excluded from issuer); prior P coverage must contain V_pre ⊎ V_issue ⊎ V_tail, with separately authorized Q/lifecycle coverage of V_body before execution.
  E_parent prepays V_pre and V_tail PLUS its DISTINCT issuance/refusal/consumption obligations; E_control's NEW debit covers ONLY V_issue. E_parent funds its own issuance machinery; later Q body W funds none of this prefix.
  Demand includes reached prefixes plus qualified tail only; uncertainty retains outstanding obligations, without refund/replay. P pays debit MACHINERY, not body work credit W: the original Q debit remains irreversible. Count actual inline-M C backing once; separately inventory actual temporaries and funded external controls. Q payload-plus-guard-metadata reservation is not physical backing. Only initial owner birth is genesis; C preparation is not free. No coefficients/rates/complete-fee claim follows.
- Proposed reuse/extraction: shared_allocation.rs:945–957 is allocation-only (budget.allocation(), Vec::new/one try_reserve_exact/error mapping), with NO logical charge/Arc clone. Existing exact_capacity consumes Vec and drops mismatches: do NOT reuse unchanged for retained uncertainty; use a SHARED borrowed/staged check with the ACTUAL Vec already Hparent-rooted before validation, not a parallel charged container/duplicate P guard. Hook 134–145 is production Ok/test-only failure injection, not physical evidence.
  rust-toolchain.toml:2 pins 1.85.1; proposed source profile is default-Global Vec<C>, empty length/capacity zero, non-ZST C, isize-fitting layout, additional one: try_reserve_exact → grow_exact/finish_grow allocation, NOT reallocation → reported capacity one/requested Layout C. Own requested buffer bytes size_of(C) BEFORE calling; API-only equality after allocation cannot fund an oversized prefix. See pinned [raw_vec.rs](https://raw.githubusercontent.com/rust-lang/rust/1.85.1/library/alloc/src/raw_vec.rs) and [alloc.rs](https://raw.githubusercontent.com/rust-lang/rust/1.85.1/library/alloc/src/alloc.rs).
  Under allocator-contract/no-unwind assumptions, allocation success → Result → pointer/capacity update has no PROJECT callback/fallible operation; this is not crash atomicity or OS/runtime proof. Private allocator block/cache/RSS and transitive deployed allocator/image remain unproved; the repository toolchain pin is not actual build proof, and typed C/layout/adoption remain unqualified.
  Proposed initial C fields are ONLY explicit fixed native tags/counters/arrays and genuine plain borrowed P/destination references; optional Q/Gq/body backing/loans are ABSENT. No Default/conversion/trait/clone/string/heap/callback/custom constructor-destructor behavior; C_temp acquires NO owning body resource. These constraints adopt no Rust type/layout or infallible C::empty.
  E_parent EXCLUSIVE initialization authority spans len-zero/cap-one/non-ZST-layout/live-external-owner prechecks → accounted metadata C_temp construction/staging/move → ONE pinned [Vec::push](https://github.com/rust-lang/rust/blob/1.85.1/library/alloc/src/vec/mod.rs#L2416-L2428), growth skipped → ptr::write(value) → len one. Actual Vec/Gc remain Hparent-rooted, tail controls outside C; no reentry/competing access. Expose unused authentic E_control ONLY after full initialization/length publication.
  Nonowning metadata moves add no owning-resource gap, but write-before-length is NOT crash atomic: unknown initialization retains original obligations, without retry/reinitialization/blind drop; cleanup must know ACTUAL initialized fields. Initial None neither removes generated C drop glue nor qualifies later Some(Gq) destruction. Heap size_of(C) excludes temporaries/control storage/RSS; qualify E_parent temporary extents separately. Earlier zeroing/initialization and later request writes are distinct paid events, never double-billed; qualify move/write accounting without inventing tariffs.
  Concrete fields/destructors/temporary bounds, actual constructor/Hparent handoff, allocator image/no-unwind assumptions, helper scope/fees/rates and actual funding remain proof gates; no native C constructor exists here or is adopted/automatically approved for standard-allocation private-internal collapse, and no native grant follows.
- Profile/layout/API/types/fee map remain UNADOPTED. Source census, complete fees/prepayment, native provenance/custody, failure prefixes, layout, tail and F/P proofs remain outstanding.
  No physical CPU/RSS/progress/termination or source/test authoring, execution, cap or HOLD claim/grant follows; all existing gates remain unchanged.

Current `host_memory()` has no owner argument; native OnceLock stores only initialization result.
The source has raw self-context, not an original funder. Epoch reuse requires the same original authority
and exhaustion history; it never refills R. The creation paragraph below excludes retrofunding.

The minimal future extension is shared OBSERVER-PURPOSE/entry binding expressing these joins,
NOT a concrete API/type, implemented authority, endpoint, new funder/keeper or tariff/readiness grant.

CURRENT LEGACY publication sites are [resources.py:139–245](../../python/pllm/deployment/resources.py#L139):
cleanup writes the owner charge at 150, constructs the slot at 152, then publishes its root at 153;
its exception handler ATTEMPTS charge rollback at 155. Other owner publications occur at 204 and 245.
These Python owner/cleanup paired writes are source evidence, NOT the selected future controlled P.
In that legacy audit, whole cleanup publication requires both ledgers and an authoritative original
outcome; charge alone or the last store alone cannot establish completion or backdate publication.

SELECTED DESIGN ONLY: opt-in native AUTHORITATIVE WHOLE capacity/cleanup publication roots the actual
original owner, charge, cleanup-custody root and authoritative original outcome, ordered with observer F.
A native permit/shadow counter alone is NOT whole publication. Python maps are NONAUTHORITATIVE views
outside the gate: absent/failed/stale/deleted projections cannot grant capacity, refund or prove retirement.
Unknown projection retains the original native authoritative owners/charges. ALL controlled admissions,
observations, cleanup and release must use native authority or fail closed, without legacy bypass;
UNCONTROLLED legacy mode remains unchanged. No implementation or actual native-root/funding proof follows.
F and actual P share ordering, including observation/read admission, with no F between admission mutations: before-P partials or F-first mean NO admission; retain only genuine grants/resources, original witness/custody/charges and funded failure tail. P-first qualifies ONLY FULL native owner/charge/cleanup-root/authoritative-original-outcome publication, never inferred completion or rollback.
Authenticate/deliver the ORIGINAL result/witness BEFORE body activation; uncertain delivery/consumption permits no replay/regrant. Projection failure leaves native P/charges intact, not renewed authority; failed/unknown retirement retains originals without release/retry or selecting an uncertainty domain.
Qualified success releases EACH original BODY charge only after ITS backing/borrow/alias obligations retire; metadata/root/tail/reference-release owners and charges survive until THEIR OWN obligations retire, even with coincident banks. C/M destruction precedes Gc release; Gc-release references/P-tail controls survive OUTSIDE C; successful W stays burned. Projection deletion proves none of these transitions.
Actual retirement remains outside the gate under qualified ORIGINAL contracts; release must remain
possible after F. External reservation/subledger/F binding, original-root lifetimes/funding, temporaries,
construction, noninheritance and uncertain-publication proofs remain prerequisites, not established facts.

One UNSELECTED candidate orders capacity lock → short native gate, never the reverse.
Selection, imports, telemetry, Python/public callbacks, allocation AND construction remain outside that gate,
under authentic already-funded preparation; current constructor-between-writes prevents simply gating the whole method.
No concrete API, transaction, rollback guarantee or existing native commit primitive is selected or proved.
Observations precede ledger installation; Python RLock does not govern native callbacks. Watcher/probe, reentry,
retained-reference, interruption and partial-failure proofs remain required. Python SetItem/insertion-allocation and paired-write atomicity analysis belongs to legacy/alternative commits; selected native P does NOT require a Python-map transaction.
Projection allocation/work/failure custody needs separate funded coverage OUTSIDE the gate and confers no authority; earlier deletion proof cannot transfer to SetItem.

The entire callback prefix, including gate entry, must be prefunded BEFORE activation; ordinal
spending and F occur inside the gate, not before handler entry. R = 0 means no activation.
Same-source nonreentrancy and cancellation INSIDE the Rth callback BEFORE any return must exclude
R+1/coalesced late callbacks. No early unwind may bypass this; nonreturn retains custody, not drain.
These are conceptual requirements, not existing code.

A durable original parent must precede source creation; record the returned handle without an
allocation gap and fully root a stable context before installation/activation. Current
[host_memory.rs:167–205](../../crates/pllm-bench/src/host_memory.rs#L167) stores only a result in
OnceLock AFTER resume: callbacks may precede publication, and late publication must NOT reopen F.
Prove no duplicate initialization after panic; old unqualified monitors cannot be retrofunded.
Inactive last-reference release is undefined, so ordinary Drop is not qualified partial cleanup.

The owner holds the creation reference to the source with its raw context association; callbacks
and readers borrow context. A nonowning source pointer in context avoids a retain cycle. Couple
reader fencing/drain. Cancel-handler quiescence, source release, context finalization and actual
physical backing retirement are DISTINCT; release/finalizer ENTRY proves no backing retirement.
The original funded tail retains charges until actual retirement; its exact endpoint is UNPROVED,
not an invented acknowledgement.

E_init + R * E_callback + E_exhaust + E_cancel + E_sync + E_final + E_tail is a CONDITIONAL component
envelope only. Unknown expansions, backing, concurrent readers or terminal storage preclude fee,
admission and CPU/RSS claims. Samples and repeated refusals need EACH authentic new entry's funding;
R is not a process-work bound, and new entries do not refill the callback quota.

The controlled profile must apply PROCESS-WIDE across observer/admission routes, including startup,
direct selection and benchmark watchdogs; unopted legacy behavior stays unchanged. Existing callback
sysctl-error fallback uses dispatch flags, not automatic sticky failure for all errors; fresh-sample
errors already propagate. Exhaustion has no fallback. Retirement/release MUST remain possible:
exhaustion fences NEW observations/admissions, not reclamation.

A prearmed, protected original witness must bind the actual original owner, monitor identity/epoch,
distinct W/B funding extents, retained destination, ledger/token/amount, payload and operation.
Epoch labels/booleans authorize neither reset nor ownership. Distinguish F, P, public return,
quiescence and retirement; wrapper exceptions consult the original witness, NOT token absence.
Interrupted partial commit retains original custody as required above.

The documentary source pin is Swiftlang/libdispatch 2078b9d4a2853b8b07a61e3d460e9dfd9d3c3454:
[source.h](https://raw.githubusercontent.com/swiftlang/swift-corelibs-libdispatch/2078b9d4a2853b8b07a61e3d460e9dfd9d3c3454/dispatch/source.h) and
[object.h](https://raw.githubusercontent.com/swiftlang/swift-corelibs-libdispatch/2078b9d4a2853b8b07a61e3d460e9dfd9d3c3454/dispatch/object.h).
Same-source handlers are nonreentrant; cancellation prevents future event handlers but running ones
continue. The cancel handler follows event-handler return and underlying handle-reference release.
Release is asynchronous; finalization requires all references. event_f captures context when set;
cancel_f reads it when called. No additional blanket finalizer/cancel ordering, OS/deployment
qualification or dependency adoption follows from this source pin.

UNAUTHORED prose oracles cover races/concurrency, coalesced Rth events, zero/partial/unknown cancel,
delayed finalization/lost parents, old unqualified monitors/refill attempts, repeated refusals and
gate reentry. APIs, mechanisms, R, rates and error presentation remain UNSELECTED. Finite coverage,
commit, cancellation, reader drain and backing-retirement proofs are prerequisites before authoring.

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

Controlled-profile observer exhaustion cannot refresh or refill its callback allowance.
A cached observation cannot serve as a weaker-monitoring fallback after exhaustion.
Request-boundary refresh does not override the bounded observer lifecycle above.

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
