# Common additive-share to common-mask consuming contract

Date: 27 October 2026. Contract-only proposal, Astra request 67599.
Source inspection: root `23d691c6f0d353d988bcb405305bf4a9a1674479`.
D59–D64 remain frozen approvals, not execution evidence or completed methods.
No implementation, D43 matrix activation, transport authentication, or security
qualification is conferred here. Completed Astra privacy/custody reviews leave
joint-security, custody, funding and distributed-publication obligations OPEN;
their conditional findings do not authorize Rust implementation or bounded tests.
Section 8 adds a conditional D63 contract proposal only: pre-issuance separation
and authentic consuming handoff remain unimplemented, without composed-security clearance.
USER71483 authorizes the detailed SIGMA bridge DESIGN tranche only. The issuer,
types, construction, authority and security functionality below are concrete
PROPOSALS, not approved interfaces or protocols. No code, numeric-profile change,
tariff adoption, implementation or execution is authorized.

## 1. Fixed functionality and trust boundary

Input is the authentic D43 terminal output: four additive lanes
`Y0 + Y1 = Y` in `R = Z/(2^50)`, Q12, in original lane order.
Output is two role-bound common-mask owners for the same four-lane wire and
an opaque, consuming dealer-mask owner. No clear `Y` reconstruction, lane change,
rounding, rescaling, or new range certificate is permitted. Ring correctness
does not establish a signed no-wrap interval for a later matrix operation.

Match `paper/methods/shared-computation-contracts.md:117–127`: two online
evaluators (model/input owners), static semi-honest corruption of either one,
trusted input-independent preprocessing dealer; no dealer/evaluator collusion.
This is a restricted common-masked opening, not malicious or production security.
Public-weight outsourcing, PUMA PRSS, and FuseFSS have different topology and/or
numeric contracts; none substitutes for this protocol or its review.
No silent clear fallback, privacy weakening, or unreviewed activation is allowed.

## 2. Protocol and complete role views

For each lane the dealer independently samples uniform `rho, a` in `R`:
`r0 = a`, `r1 = rho - a (mod 2^50)`.
Dealer privately preprocesses `ri` to role `Pi`; `rho` stays inside an opaque
`DealerMaskFor<C, Ring>` destined for its unique admitted downstream consumer.
Each `Pi` consumes its authentic share owner and mask-share owner, computes
`mi = Yi + ri (mod 2^50)`, and issues one bound peer frame containing four `mi`.
After validation, each computes `Yhat = m0 + m1 = Y + rho (mod 2^50)`.
Neither evaluator receives `rho`, the peer's share, or the peer's mask share.
The dealer receives no online frame, `Yhat`, or online completion payload.

| Role | Knowledge included in review, not merely the new output |
| --- | --- |
| P0 | Its original private inputs, full prior D43 view/transcript and randomness, `Y0`, preprocessing `r0`, local `m0`, received peer frame `m1` including metadata, `Yhat`, subsequent gate keys/messages and local state. |
| P1 | Symmetric full prior D43 view, `Y1`, `r1`, `m1`, received peer frame `m0`, `Yhat`, downstream keys/messages and local state. |
| Dealer | Public plan/bindings, independent `rho,a`, both preprocessing shares, retained `rho` and downstream correlated-key preprocessing; no online frames or `Yhat`, and no collusion. |

Fresh uniform `rho` gives a marginal pad argument only. It does NOT prove joint
security with prior D43 transcripts, received peer frames, the evaluator's own
`ri`, or downstream matrix correlated keys. Independent review must give each
allowed corruption's joint simulator and specify permitted leakage, including
shape, binding metadata, scheduling/abort observations and preprocessing views.
Dealer independence and key correlations must be checked across the whole graph.
Delivered frames cannot be erased or recalled; local cleanup cannot repair a
privacy leak. Arithmetic identities and passing negative tests are not proofs.

## 3. Authenticity, lineage, and reuse anchors

Paths below are relative to `crates/pllm-garble/src/masked_wire_reference/`
except the explicitly named Python paths.

1. `typed/gapars_terminal.rs::ShareFor<C>` retains `TerminalLocalFor<C>`, beta,
   material and lease. `values()` returns a borrow, not output identity or
   authority. No public array constructor, equal-digest adoption, caller labels,
   or copied share slice can create the bridge's authenticated input owner.
2. `typed/layernorm4/parent/runtime.rs::Role::finish` requires both-ready and
   returns a coordinator-backed `ShareHandle`; its loan still borrows state.
   A future sealed move must consume the original D43 parent/role output owner,
   retaining coordinator/source lineage and original root ledger. No independent
   coordinator may mint authority from `ShareFor::values()` or a `ShareLoan`.
3. Reuse `typed::{CommonMaskedFor, DealerMaskFor}` and `matrix_with_ticket`.
   Their native backing owners and leases move intact. Existing variants contain
   no bridge backing: extending this common contract needs reviewed ownership
   and layout, not copying into an unrelated public root. Existing matrix checks
   for private weights, dimensions, ring/scale, distinct wires and range remain.
4. `python/pllm/runtime/shared_mpc.py::ValueOpeningFrame` and
   `shared_session.py::SharedPeerConnection` supply framing/bounded-exchange
   patterns only. Python frames use MessagePack plus copied arrays; connection
   relies on externally authenticated transport. Neither authenticates native
   D43 output identity, supplies a native header bound, or proves native custody.
5. PUMA PRSS / FuseFSS are not substitute issuers, mask generators, numerical
   profiles, or security arguments for this D43-to-common-mask boundary.

Bind actual producer root and source identity, numeric profile, plan, wire,
lane order, four-lane shape, ring 50, scale 12, session, role, operation, phase,
and unique consumer use. Preserve original model/source validation. Compare
actual source/coordinator/budget ownership as well as metadata; equal digests
are insufficient. Reserve the bridge and dependent uses in the same original
root ledger, including dealer custody and downstream mask consumption.

## 4. Linear lifecycle and erasure

Proposed consuming state: `Admitted -> Issued -> Begun -> Validated -> Prepared
-> Committed -> Consumed`; precommit failures become `Burned`. Admit original
resources/uses before construction or entropy. `begin` spends the original use
before any frame escapes; its spent continuation retains its own backing/cover.
No cloning, fork, reissuance or second use of `rho`/`ri` is permitted.
Production coordination may retain only nonsecret phase/use/decision metadata,
never secret role state or a backlink to it; see §8.
Proposed opening payload is exactly 32 bytes: four little-endian u64 residues,
each with its top 14 bits zero. Reject malformed ingress; never reduce it into
validity. Retain actual frame binding/digest, role, phase, count and canonical
checks. `finish` requires an authentic committed-use capability, not matching
labels or ready messages. This is a proposal, not an existing native frame/API.
Wrong root/role/use/shape, replay, out-of-order phase, malformed/noncanonical
residue or frame, missing peer, and premature finish refuse before publication.
Before commit, failure/drop/cancel/timeout/unwind burns locally unpublished
dependent uses and REQUESTS an authoritative abort; local cleanup does not prove
global abort. An unknown shared decision is **IN-DOUBT**: keep outputs unpublished,
retain required custody/cover, and prohibit reissue. `Burned` is terminal for the
spent local attempt; it is not evidence that an unobserved COMMIT does not exist.

**Proposed separate decision authority (not supplied by existing owners).**
Require authenticated participant enrollment, genuine issuer-bound issuance and
original-use registration, exclusive COMMIT-or-abort decisions, non-equivocation,
durable crash persistence and consumer-verifiable decision authentication. COMMIT
binds BOTH role preparations and their exact ordered opening transcript and
original downstream use. Preparation acceptance is not joint publication. Labels,
ACKs, local `require_both_ready` and `CoordinationFor::abort` are not this authority;
no existing native or Python owner provides it. Authority construction, enrollment,
authentication, persistence and recovery interfaces remain unselected and OPEN.

### Proposal counterexamples: authorization is not freshness

These are counterexamples to incomplete proposed contracts, not demonstrated
exploits of an implemented bridge (none exists). Snapshot authentic pre-use owners,
original ledger and consumer acceptance state; consume once, then restore those
states and replay the original authorizations. If no non-rollbackable state outside
that snapshot rejects the replay, a second acceptance needs no forgery. Rust moves,
matching bindings and crash-durable SQLite storage alone do not prevent restoring
an older valid state; durability is not rollback resistance. The proposed authority
must cover this boundary too, not merely authenticate a COMMIT.

Section 2 gives a distinct confidentiality counterexample: if an actual changed
input is processed with reused `rho`, then `H=Y+rho`, `H'=Y'+rho` imply
`H-H'=Y-Y'` lane-wise modulo `2^50`. Replaying identical bytes alone does not
establish changed-input leakage; this example assumes genuine changed-input reuse.
An authorized transcript likewise cannot certify independent matrix preprocessing;
see §7's `A0=rho` counterexample.

One-sided delivery loss is tolerable only with permanent abandonment/burn of the
uncertain instance: no reissue, refund or recovery of that instance. This does not
assert global abort, revoke COMMIT or allow early release of still-required custody.
D66 selects a research-only NONRESUMABLE lifecycle for the next security review:
exclude rollback/cloning of issuer, evaluator, authority and consumer state;
permanently abandon uncertain instances/uses, with no reissue or refund. The
separately authenticated, durable, non-equivocating authority remains required;
its construction/API and an authenticated-two-role alternative remain UNSELECTED.
D63 independent review, role isolation, full-view simulation, original funding,
custody, admission, masks, arithmetic and HOLDs remain; no production antirollback,
security qualification or implementation/execution grant follows.

This functionality could enforce **no publication without joint authorization**,
not simultaneous delivery or global abort after commit. After commit, lost final
delivery/crash preserves COMMIT: no fresh issuance or reconstruction retry. A local
cancel cannot revoke a peer's authorization or disclosure; more ACKs cannot repair
asynchronous final-message loss. Without approved authority, remain blocked or
obtain an explicitly weakened contract. No remote erasure or crash cleanup is promised.

Keep original D43 backing, shares, dealer/evaluator masks, scratch, unpublished
send/receive frames and outputs under retained owning leases until exact owned
erasure completes. Keep downstream dealer masks and evaluator backing charged
through their final consuming operation. Borrow guards and partial transitions
must not release backing or refund admission early. Specify erasure order and
bound cleanup on success, cancellation, poison and unwind before implementation.
No promise covers caller copies, network buffers, peer memory or delivered data.

**Proposed required owner — committed-but-not-accepted output custody.** An
issuer-bound pending-output owner must retain original cover until acceptance
atomically moves backing AND cover into the unique consumer. Cancellation retires
only backing still locally owned; old frames retain cover until their last covered
backing retires. Spent uses/work never refund; a standalone budget guard cannot
substitute issuer authority. This owner is not supplied by current typed runtime.

## 5. Concrete resource/work inventory and admission blockers

Use the SAME original root budget, aggregate ceiling `64 MiB = 67,108,864 B`,
and all tighter existing caps. Checked arithmetic must reject overflow. Admission
must price the full simultaneous live set and cumulative work before construction
or the first entropy byte, not just an incremental payload or apparent stack size.
For target/context `C`, derive actual type size AND alignment, enum padding,
Arc/mutex/control allocation layout, vector capacities, allocator accounting,
serialization copies, retained backing and failure overlap. No guessed constants.

Existing anchors: terminal `resources<C>()` includes `MaterialFor`, `ShareFor`,
`Controls`, terminal moves and `4*C::TRANSIENT_LINK_BYTES`; parent issuer
`resources<C>()` includes root, source/table, keeper, two role slots, coordination,
wrappers, native children, retained frames and runtime owners. Retain its full
live reservation; do not subtract D43 owners merely because shares are borrowed.
Typed charges use actual `size_of::<CommonMaskedFor<C, Ring>>()` and
`size_of::<DealerMaskFor<C, Ring>>()`; existing vector accounting adds
`ALLOCATOR = 64`. That is an existing accounting convention, not proof of new
allocator overhead, target layout, serialization bounds or stack containment.

| Item | Concrete known subtotal / required complete bound |
| --- | --- |
| Original D43 | Retain parent issuer's `Resources::total_bytes` and all outstanding source/table/native/share/frame/control owners; reconcile shared allocations once by original lease identity, never twice-credit or prematurely refund. Actual transition liveness must be enumerated. |
| Entropy | Eight independent u64 draws (four `rho`, four `a`), low 50 bits: exactly 64 entropy bytes; eight draw reductions. No entropy before successful full admission. |
| Dealer preprocessing | Four `r0` and four `r1`: 64 payload bytes total, excluding headers, private delivery, capacities, owner/lease/control bodies and copies. Retained `rho` adds 32 payload bytes; `a`/draw scratch overlap must be enumerated. |
| Evaluator masks/shares | Two four-u64 mask-share payloads: 64 B, plus authentic D43 retained shares/backing, concrete role-owner layouts, leases and guards. Moving preprocessing ownership does not justify unpriced copies. |
| Online send frames | Two canonical 32-byte opening payloads: 64 B aggregate exchange payload only. Add bound metadata, alignment, owner, capacity and serialization/delivery buffers; no complete transport/header layout is frozen. |
| Receive/serialization | Two received payloads alone are 64 B if both retained. Enumerate bounded ingress, validation, parsed arrays, send serialization and queued/partial-send copies at maximum overlap; Python `pack`/`unpack` is not a native allocation bound. |
| Outputs/downstream | Two `Yhat` payloads: 64 B, plus actual common-owner/backing capacities and leases; retain opaque `rho` owner and dependent matrix/key/consumer reservations until final consumption/erasure. |
| Controls/failure | Price coordinator, role slots, root/use ledger extensions, reservations/results/options, locks/Arcs, move temporaries, bindings, cancellation and timeout state; enumerate normal and partial-failure peak plus bounded erasure workspace/work. |
| Ring work | Dealer: four subtractions/four reductions for `r1`; evaluators: eight additions/eight reductions for `mi`; reconstruction: eight additions/eight reductions total across both roles. Together with draws: 20 add/sub operations and 28 reductions; prior D43/downstream work remains additional. |
| Non-ring work | Exact bounded frame encode/decode/validation/copy counts and byte visits, root/use transitions, entropy calls, delivery operations and cleanup visits must be priced from the final layout, including failure paths. Two evaluator frames, eight opened masked lane values total; also charge private dealer deliveries and their framing. |

**Funding/admission BLOCKER:** no sealed consuming bridge owner/layout, complete
bound framing/transport layout, capacity/copy schedule, dependent-consumer retained-live set, or
bounded cleanup/work quote exists for this transition. Thus no exact combined
byte total or complete work allowance can honestly be asserted under 64 MiB.
An implementation proposal must fill every bound above using actual target types
and preconstruction capacity limits, reconcile the original ledger, and prove
`peak(root retained + bridge + downstream + failure overlap) <= root allowance`.
No construction/entropy is allowed while any term is unknown. Charging work is
an operational admission obligation, never a cryptographic security proof.

### Proposed bridge phase peaks and work extension (not a full quote)

Extend the original issuer `Resources::total_bytes`, `Work`, `IssuanceWork` and
`internal_use::Ledger`; do not create fresh inventory or credit removed fields.
Preserve original source/table/native/share/frame cover and original nonrefundable
attempt/use charges. The following logical arrays refine, rather than add a
second copy of, the inventory above; `v4` means one four-residue owning payload:

- Issuance peak: retained D43 + `rho4 + r04 + r14` + draw scratch + subtraction
  scratch + private delivery owners. Draw `a` directly into `r0` to avoid a
  separate `a4` allocation; this is a proposed construction schedule, not an
  existing implementation or permission to omit actual scratch.
- Prepared peak: retained D43 + `rho4 + r04 + r14` + two send `mi4` + two received
  peer `m4` + two unpublished `H4` + bounded codec/ingress copies + downstream
  material. Retain any still-live earlier delivery/scratch owner in the actual
  phase peak and every failure overlap; this list does not authorize early erasure.
- Controls: two role continuations, two pending common owners, dealer continuation,
  provenance/cover, bindings, preparation/decision records, locks and guards.
  Extend existing `IssuanceControls`/`TransitionControls` to price containing enums,
  `Option`/`Result`, arguments, return/move slots and error overlaps. Embedded
  arrays are charged once inside their containing owner; every distinct owning
  allocation or live move slot remains priced. Cell clones are not new backings.
- Dealer work: four lanes, two independent draws per lane, split subtraction and
  private role-local delivery. Per role: four opening additions, four canonical
  ingress lane checks, four `H` additions, one send, one preparation and one
  acceptance. These refine §5's arithmetic subtotal, not a second ring-work charge.
  Codec, metadata/authentication validation and cleanup need finite bounds from
  frozen lengths/capacities and actual live owners; cleanup work is cumulative.

Header/authenticator sizes, queues, persistent/recovery storage and failure
overlaps remain **UNKNOWN**; no full byte/work quote follows. Finite entropy
callback/retry, transport-wait and recovery bounds are not adopted. IN-DOUBT has
no duration bound; finite owner capacity does not imply finite custody duration.
Before entropy, checked admission must cover original retained + incremental +
downstream + failure peaks AND cumulative work including cleanup. Maverick's
one-atomic-call choice is not a SIGMA tariff or a substitute for this inventory.

## 6. Conditional regression definitions and remaining gates

Future tests only after independent Astra joint-simulation/custody review:

- Independent integer ring oracle for four lanes, wrap boundaries and all roles:
  exact `Yhat = (Y0 + Y1 + rho) mod 2^50`, unchanged lane order and Q12 metadata.
- Authentic lineage success; equal-digest/different-root, caller-array, wrong
  source/profile/role/use/session/wire/shape, replay and phase-order refusal.
- Exact-storage/exact-work admission succeeds; each one-unit-short bound refuses
  with zero entropy before admission and no partially usable capabilities.
- Partial send, cancel, timeout, drop and unwind at each transition burn siblings
  and dependent local uses before commit and request authoritative abort; unknown
  decisions stay IN-DOUBT without publication or reissue. Finish requires authentic joint authorization.
  Postcommit crash preserves the decision; no simultaneous-delivery/global-abort assertion.
- Retained-backing/lease tests keep D43 and downstream owners funded through
  exact erasure; no lease refund while borrowed, serialized or failure-live.
- Negative-oracle fixtures deliberately reuse masks, use one-bit masks, leak a
  peer share or dealer mask, mix sessions, or publish early; detect the forbidden
  behavior/leakage, not merely algebraic failure. Passing is not security proof.

Remaining gates: complete resource/cleanup funding; authentic consuming parent
handoff; independent full-view joint simulation with downstream correlated keys;
dealer separation and custody review; then conditional private Rust child
`share_to_common_mask.rs` and tests. No D43 matrix activation, native run, or
production claim is authorized by this document. Missing source/security
obligations return for review rather than becoming invented defaults.

## 7. Independent privacy review (activation held)

### Bridge-only conditional arithmetic simulator

Fix a genuinely separated prior evaluator view `Vi` and its legitimate terminal
share `Yi`. Require `rho,a` uniform in `R^4`, mutually independent and independent
of the ENTIRE prior execution (not merely its public bindings). With
`r0=a`, `r1=rho-a`, `H=Y+rho`, each role's pair `(ri,H)` is independent uniform
in `R^4 x R^4` conditional on that prior view/share. A bridge-only simulator
samples uniform independent `ri,H`, sets `mi=Yi+ri`, `mpeer=H-mi`, and exactly
simulates arithmetic view `(Vi,Yi,ri,mi,mpeer,H)` in `R`; no reconstruction of clear
`Y` is needed. This is a conditional extension of a fixed prior view, not a D43 proof.

Extending to the ACTUAL view requires specified authenticated private preprocessing
and peer/transcript channels, the §4 authority functionality and an explicit leakage
kernel: original bindings, canonical bytes, authenticated sender/channel identity,
ordered transcript commitments, preparation/decision/delivery observations,
lengths, timing and aborts. A simulator cannot forge credentials, silently assume
private-dependent failures away or infer channel functionality from Rust ownership.
These channel/authority/leakage specifications remain OPEN. Prior D43 and weight-mask
history, downstream matrix `Material` coefficients `a,b,c`, fresh masks and dealer
view require a JOINT proof; D17 freshness alone is insufficient. The conditional
matrix extension below records additional hypotheses, not composed-security clearance.

#### Joint-mask condition and abstract adversarial-coin negative control

**D63 derivation/review only.** Write `q=2^50`, `R=Z/qZ`, `d=4`. Fix the
ENTIRE legitimate role prior `(Vi,Yi,ell)` (including allowed leakage), then any
compatible `Y`. Fresh `a,rho` must be independent uniform vectors, independent of
the entire prior execution. The maps `(a,rho) -> (ri,H)` have inverses
`a=r0, rho=H-Y` for role 0 and `rho=H-Y, a=H-Y-r1` for role 1.
Thus every full four-lane pair `(ri,H)` has probability `q^-8`; restriction to
any two lanes gives each pair probability `q^-4`. Averaging over any `Y`
compatible with the fixed prior preserves these same laws. The complete
arithmetic opening view follows by `mi=Yi+ri`, `mpeer=H-mi`; this does not
simulate authentication, leakage kernels or the prior protocol itself.

**EXPLICIT abstract adversarial-coin fixture, NOT an actual same-prior D43
attack or security clearance.** On two lanes instead take `rho=(t,t)`, uniform
`t in R`, with independent uniform `a in R^2`. Each scalar mask is uniform and
reconstruction remains correct, but `H[1]-H[0]=Y[1]-Y[0]`. This refutes scalar
marginals as a substitute for the joint condition. For fixed `Y`, role 1's
`r1=(t,t)-a` is STILL uniform and independent of `t` and `H`; this fixture does
not break role 1's own-mask independence. Embed these two bad lanes in four by
giving the other two lanes independent fresh coins.

For a finite abstract illustration only, use `Z/4Z`, fixed role-0 share `(0,0)`,
and candidate totals `Y=(0,0)` versus `(0,1)`. Good independent masks give all
16 values of `H` probability `1/16` in both cases. Bad masks give four diagonal
values versus four shifted-diagonal values, each probability `1/4`. Event
`H[1]-H[0]=0` has good probability `1/4` in both cases, bad probability `1`
versus `0`. These abstract candidate totals do not establish feasible D43 priors.

An actual witness requires TWO feasible executions under ONE fixed AUTHENTIC
model/source binding, with the SAME ENTIRE legitimate role prior, share and
allowed leakage, but different selected output-lane difference. Equal `ell`
alone is insufficient. Varying beta may change model/source bindings/commitments; varying beta
alone does not establish such a witness. Carry opening preceding beta in `arithmetic.rs:523–533`
and `typed/gapars_terminal.rs:342–344,298–306` (paths relative to §3) does not
prove prior equality. No such feasible pair or joint protocol security is
established here. No new tests, protocol adoption or activation are authorized;
§8's proposed issuance still requires the full independence and custody gates.

Actual D43 terminal arithmetic is **RING50/Q12**, residues stored in `u64`,
not Ring64. For the `[1,4] x [4,1]` contraction, let `R=Z/(2^50)`,
`rho,s in R^4`, output mask `t in R`. Source `matrix/core.rs::{deal,evaluate}`
gives `A0+A1=-rho`, `B0+B1=-s`, `C0+C1=rho*s+t`, where products are
row/column contractions. Inputs `H=Y+rho`, `K=W+s` yield
`qi=Ai*K+H*Bi+Ci+[i=1]*H*K`; their sum is `Y*W+t`.

**Conditional extension only — per-role SIMULATION CUT.** Fix the entire
legitimate prior `Vi`, authentic own share `Yi` and allowed leakage for role
`i`. Retain every prior observation, including terminal `hi`, keys, material,
randomness, frames and failures. The cut precedes exposure of the fresh bridge
and matrix split pads below; fresh samples never overwrite existing material.
The prior D43 simulation, separation and channel hypotheses remain prerequisites.

| Role-local knowledge at the cut | Conditional arithmetic extension only |
| --- | --- |
| Fresh hidden weight mask: no observed downstream keys, weight mask `s` or output mask `t` | Sample independent uniform `ri,H,K,Ai,Bi,Ci,O`, dimensions `4,4,4,4,4,1,1` (22 ring coins). |
| Legitimate operand owner already knows exact `W,s,K=W+s`; no exposed bridge/output/split randomness | RETAIN `W,s,K`; sample independent uniform `ri,H,Ai,Bi,Ci,O`, dimensions `4,4,4,4,1,1` (18 ring coins). |
| Preissued keys, other mask history, known output mask, or correlated prior observations | Neither shortcut applies. Preserve ALL observations and derive the actual conditional law; OPEN. |

In either applicable row, set `mi=Yi+ri`, `mpeer=H-mi`,
`qi=Ai*K+H*Bi+Ci+[i=1]*H*K`, `qpeer=O-qi`. For the first row,
require fresh independent uniform `rho,a,s,t` and nine role-0 matrix split pads,
jointly independent of the prior and private operands. For fixed compatible
`Y,W`, the equations of `matrix/core.rs::{deal,evaluate}` give the inverse
`rho=H-Y`, `s=K-W`, `t=O-Y*W`, with `a=ri` for role 0 or
`a=rho-ri` for role 1. Role 0 recovers the nine pads directly from
`Ai,Bi,Ci`; role 1 recovers `A0=-rho-Ai`, `B0=-s-Bi`,
`C0=rho*s+t-Ci`. This is a bijection on 22 ring coins, without division
or a field assumption. Mixing over compatible `Y,W` conditional on the fixed
prior preserves this law only under those independence hypotheses.

For the second row, condition on the retained exact `W,s,K`: `s` is fixed,
NOT independent of that prior. The same inverse, omitting `s=K-W` as a fresh
coin, gives an 18-coin bijection when `rho,a,t` and all nine split pads remain
fresh independent uniform, unexposed and independent of the conditioned prior
and private operands. Resampling independent `K` instead would give event
`K=W+s` probability `2^-200`, versus real probability 1. This is a
simulator-scope counterexample, NOT protocol leakage. Preissued material or
output-mask knowledge requires the third row's separate conditional derivation.

Retain actual frame binding/digest, role, phase, count and canonical fields,
authentication and ORDERED decision/abort leakage consistently with the retained
prior. Uniform samples neither forge authentication nor prove a channel,
prior-D43 simulation or secret-dependent abort kernel. These are conditional
arithmetic extensions, not cryptographic or composed-security clearance.

#### D66 conditional authenticated-view extension

Primary derivation: Astra29 (83596); independent review: Astra21 PASS (83604).
Let `V_i` include ALL retained prior and transitively reachable owner, dealer
and credential state. Assume **BaseSim** JOINTLY simulates `(V_i, full arithmetic
view, already-authorized auxiliary leakage)` from legitimate ideal inputs/outputs
and approved leakage. Fresh-coin independence and the matrix/D43 hypotheses
above must hold against the WHOLE prior; these remain unproved for actual D43.

Applying the SAME conditional postprocessing kernel to this joint tuple, with
independent coins and the modeled schedule, preserves equality or statistical
closeness. Computational indistinguishability extends only through an efficient
(PPT) kernel. Causality, adaptive scheduling and abort coverage must belong to
the base claim wherever claimed; this is not a full UC result. The kernel must
not access additional real private `Y` or other state outside that joint tuple.

Genuine ideal authentication and enrolled issuer/OriginalUse capabilities, or a
legitimate signing oracle, are explicit ASSUMPTIONS, not labels, hashes or forged
native capabilities. The authority's exclusive COMMIT binds BOTH ORDERED
preparations; its decision is durable and delivery loss cannot revoke COMMIT.
Under D66, unknown decisions remain unpublished and uncertain instances/uses
are permanently abandoned, with no replay, reissue or refund. Authority construction/API
and the authenticated-two-role alternative remain unselected.

Metadata, openings, `H`, signature IDs, lengths, timing and decisions are leakage
ACCOUNTING OBLIGATIONS, not permissions for new disclosure. Existing approved
leakage must suffice to simulate them; otherwise return to the user.
If `rho` is in `V_i`, uniform `H` alone fails joint privacy: `Y=H-rho`.
This is an abstract counterexample, NOT an established same-prior D43 attack.

Postprocessing does NOT repair the base simulation. Authority ingress/view,
signing and abort timing, actual D43 prior simulation and transitive isolation
still require proof. Full security, funding and admission remain separate gates;
no construction/security qualification, code/execution grant or HOLD clearance.

Uniform marginals do not suffice. Input-independent preprocessing alone also
fails: valid splits with `A0=rho`, `A1=-2*rho` expose `Y=H-A0` to P0,
despite uniform individual marginals of `rho` and P0's `A0`. This proposed bad
split satisfies `A0+A1=-rho`; with the remaining B/C identities above, arithmetic
still works. Matching authorization therefore does not prove independent
preprocessing. This is a counterexample to weakened distribution requirements,
not a claim that authentic `matrix/core.rs::deal` selects this split. Metered `fill` callbacks prove neither independent
entropy nor freedom from private captures. Typed matrix checks/output-mask draws
do not discharge these hypotheses. No composed-security clearance or D16 adoption.

Source-backed blockers remain (paths relative to §3's source root):

- `typed/layernorm4/parent/issuer.rs::issue_reserved_with` retains a
  `RowMaterial` keeper containing masks and the clear local dealer;
  `runtime::Role` retains coordination reaching both role slots. This is not a
  separated evaluator-memory-corruption deployment.
- `runtime::ShareHandle::loan` provides a borrowed `ShareLoan`, not an authentic
  consuming loan-to-owner handoff. No such bridge handoff exists.
- Terminal shape `[4]` does not satisfy matrix input shape `[rows,inner]` in
  `matrix_with_ticket`; an authenticated consuming adapter remains required,
  not a copied slice or metadata-only reshape.
- `typed/role_coordination.rs::CoordinationFor::operation` handles its existing
  local failure domain; no cross-boundary sibling/descendant burn and
  authenticated-delivery lifecycle is supplied for this bridge and consumer.
- No combined preconstruction/backing quote closes original D43, bridge,
  downstream consumer and failure overlap under the same retained root.

**No implementation-ready reusable unit; D63 activation remains held.** Existing
proposal and obligations above remain proposals, not executable design or an
independent full-view/joint-security approval.

## 8. Pre-issuance separation and authentic consuming handoff (proposal)

**Conditional contract only; D63 activation remains held.** Transitive separation
must precede private issuance and cover D43 execution, not merely terminal outputs. Existing
`LocalDealerFor` composition is reference/test-only for this boundary. Process
separation and authenticated delivery are unimplemented; neither Rust privacy nor
`Arc` ownership provides process isolation. Moving outputs cannot undo exposure.

### Source-backed separation boundary

Paths follow §3. Before issuance, remove evaluator reachability through
`PartyInitializer`/`Role`/`FrameHandle`/`ShareHandle.coordination` to
`CoordinationFor.slots[0..2]`, and through `InitializerState.keeper` to
`RowMaterial` masks, modular/coordinate material, terminal mask and dealer.
Public control may retain original source/session/profile/wire/use plus
phase/decision metadata only: no secret slots, RNG captures or secret backlinks.
Also remove transitive `ModularParty`/`ModularFrame`/`ModularValue.creation` and
`CoordinateParty.modular` reachability to `ModularCreation` native product,
energy and interval secrets, INCLUDING nested inputs. Preserve genuine nonsecret
provenance and original `Arc` identity by the proposed shared custody extraction
below; deleting checks or relabeling a digest is not separation. Bare coordinate
begin/advance frames and Index output must retain actual role-local backing and
cover. Cancel leaves inputs/TR, and Initializer retirement leaves frames: neither
is complete retirement. All such owners must be separated before issuance;
terminal-handle drop cannot undo earlier observations or external strong references.
Concrete complete-view counterexample: while P0 owns `Y0`, its
`ShareHandle.coordination -> slots[1].state.share` still reaches `Y1`
(`typed/layernorm4/parent/issuer.rs:169–196,350–353`,
`typed/role_coordination.rs:46–50`; `parent/` below means `typed/layernorm4/parent/`);
jointly these reconstruct `Y`. The ordinary loan selects only its own role
(`parent/runtime.rs:105–123`), so this is a reachable-memory custody finding,
not a public cross-role loan API. Removing top coordination alone misses
`InitializerState.keeper -> RowMaterial`; removing that keeper too still misses
`ModularParty`/`ModularFrame`/`ModularValue.creation -> ModularCreation` and
nested `inputs -> ModularInputs` (`parent/coordinate.rs:56–116`). Review the
whole retained graph, not just terminal handles. Original backing and covers stay
funded until owned erasure before any storage refund; spent work/use charges never
refund. A proposed provenance cell must retain no secret parents, RNG state or
callback paths; stripping these edges creates no early cover credit.
`runtime::Role` and `issuer::ShareHandle` retain
`Arc<CoordinationFor<C, InitializerState<C>>>`; `role_coordination::CoordinationFor`
reaches both role slots. `issuer::InitializerState` retains role values, gate
material, frames, terminal share and an `Arc<RowMaterial<C>>` keeper in both
states. `RowMaterial` retains original masks, modular/coordinate masks, terminal
dealer mask and `LocalDealerFor`. `gapars_terminal::ShareFor` retains terminal
share, beta, material and leases; its `values()` borrow is not a consuming handoff.
`typed.rs::LocalDealerFor` retains source/plan/session/namespace/counter/budget
and lease, **not a stored RNG seed**. Randomness and material separation are still
required. `layernorm4::SourceFor` has descriptor/use/table provenance, no mask or
cancellation backlink; reuse requires review of table disclosure, not automatic
approval of the whole reachable source graph.

Production custody must enforce these complete-view boundaries:

- Each `Pi`: only its own input/share, D43 keys/randomness/transcript, `ri`,
  local/received masked frames and its own matrix keys. No sibling secrets,
  dealer masks, dealer/peer RNG state or secret-bearing coordination backlink.
- Dealer: fresh `rho,a`, mask shares and correlated matrix preprocessing; no
  online shares, openings or completion payload. Public evaluator coordination
  contains only phase/use/cancellation metadata, not the current role slots.
- Trusted end client: authorized own input/output and public control, not
  implicitly peer/dealer secrets or authority to collude with the dealer.

Preserving these boundaries requires no changed user trust choice. A coupled-memory
provider must not silently gain a separated-provider privacy claim.

### Conditional protection-domain and full-prefix acceptance addendum

**PROPOSED, not adopted topology.** Four protection domains would be P0, P1,
DEALER and a separate authenticated decision authority. The authenticated two-role
alternative remains unselected. USER71483 remains DESIGN ONLY; D66 remains
rollback-excluded, research-only and NONRESUMABLE. No implementation is authorized.

| Proposed domain | Exclusive secret custody / permitted sharing |
| --- | --- |
| DEALER | `RowMaterial`/`ModularCreation` masks, dealer and dealer secrets remain here; extract authentic immutable declaration/provenance/accounting from secret backing, not copied public labels. |
| P0 | Only role-0 `InitializerState`, `ModularParty`/`CoordinateParty`/`ModularInputs`, native children, frames/TR outputs and shares; no role-1 or dealer secrets. |
| P1 | Symmetric role-1-only owners and secrets; no role-0 or dealer secrets. |
| Decision authority | Exclusive authenticated decisions, but NO operand/share/mask/dealer secrets; private credentials and authority state do not become public by being called “nonsecret.” |

Authentic nonsecret provenance/accounting may be shared only under proved isolation.
Forbid evaluator-transitive `Arc`/`Weak`, callback, context and transport-buffer
backlinks to dealer or sibling secret owners. Rust visibility is not adversarial isolation.
Actual local D43 reachability of both shares fails this separation premise; it is
not an attack on the hypothetical separated protocol.

Source anchors below are under `crates/pllm-garble/src/masked_wire_reference/`;
these clarify earlier line references, not another owner census. Split custody in
`typed/layernorm4/parent/issuer.rs::issue_reserved_with` (806), after reservation
validation (813–815) but BEFORE clear-copy (817) and root issuance:
`prepare_root_reserved_with` (875–881) precedes `root.encode` (882).
Nested `parent/coordinate.rs::issue_modular` (443–467) clones secret-bearing
`ModularCreation`; `issue_coordinate` (635–654) repeats that reachability.
`source_lookup/interval50.rs::DeclarationHandleFor` (62–68) already exists as a
layout-only handle: no constructor/accessor/Clone/escape; attachment is unimplemented.
`transport.rs::export_party` consumes its owner, but `import_party_with_material`
imports BORROWED bytes, not a consuming authenticated import. `typed/material.rs`
`MatrixHandoff`/`MatrixImportPermit`/`MatrixMaterialPacket` supply a local consuming
pattern only, not distributed authorization. Reuse existing shared framing patterns.
Original root/source/session/profile/operation/use/native-execution identity,
authentic consuming handoff and failure-burn/uncertain custody remain governed by
§§3–4 and §8's consuming transition; no replacement definitions or fresh authority.

**Full-prefix acceptance obligation (extends §7, not a security proof).** Require
ONE causal simulator `S_i` per static corrupted role, uniformly for all permitted
schedulers `σ` and stopping times `τ`, with matched ideal scheduling/stopping rule:
`Law(View_real(i,σ)^{≤τ}) ≈_* Law(View_ideal(S_i,σ)^{≤τ})`.
Here `≈_*` is the required indistinguishability strength, TO BE SPECIFIED, not proved.
The full transitively reachable prior plus arithmetic/lifecycle observations must
arise from authorized ideal inputs, legitimate outputs and ALREADY approved causal
leakage. Include every prefix, success, refusal, abandonment, IN-DOUBT and postcommit
loss, not only success-conditioned views; no ad-hoc leakage expansion. The source
prior needs its own simulator and fresh conditional joint laws. All code, security,
funding, HOLD4657, matrix activation and native-execution gates remain unchanged.

### Preprocessing independence and authenticated contributions

**Source observations, not separation proof.** Paths follow §3; `issuer` below
means `typed/layernorm4/parent/issuer.rs`. `issuer::LocalClearRootSpec4::new`
(50–68) borrows and range-checks clear X, gamma and beta together.
`issuer::issue_reserved_with` copies all three into a zeroizing array (687–688),
then calls `root.encode(&clear[i])` (745–769). `typed.rs::RootInputFor::encode`
(608–629) reads clear values and full masks, writing value-plus-mask into both
common owners. Thus the current local issuer observes clear, mask and masked
value; dropping the clear array or retiring owners cannot undo that observation.
`issuer::RowMaterial` construction (956–965) retains masks, dealer and source.

Bounded preparation-dataflow finding: `typed.rs::LocalDealerFor::prepare_root_reserved_with` (515–563) creates masks and two zeroed common destinations.
Within `issuer::issue_reserved_with` (804–946), local mean/center/energy maps, arithmetic mean/floor, products square/normalize/affine,
interval50, coordinate, norm13 and terminal preparation consume contexts, public PROFILE/table, masks, RNG and leases—not
`ProducedValueFor`, `ValueLoan` or the clear-copy array. norm13 table values are public fixed D43 data;
`typed/gapars_terminal.rs::issue_retained_with` (177–239) consumes the beta **mask**, not private beta.
Retained source provenance or masks do not themselves imply private-operand reads. For equal approved public bindings and equal
preparation-entry resource/use state, with independent RNG/context behavior, the inspected preparation dataflow is mask-only.
This bounded functional statement is not a complete-callgraph, cryptographic, simulation or error-independence proof.
The current root loop interleaves `root.encode(&clear[i])` (`issuer::issue_reserved_with`, 753) before later factories;
clear validation can prevent factories, private-dependent `FnMut` captures or budget history can change error/success, and the clear array remains live.
The whole current issuer is therefore **not input-independent**. Public-binding declaration and session/range/provenance classification,
joint simulation and pre-issuance custody remain OPEN. Next review must examine independently admitted, fixed public/resource state
before any private contribution, with independent RNG/context behavior; this finding grants no settled protocol, API or reassignment approval.

`typed/layernorm4/table.rs::{ENTRIES,NUMERATOR,construct}` fixes the public
8192-entry table and numerator `2^57`, not a checkpoint/custom table.
`issuer::metadata` (81–85) labels the fixed local constructor, not authenticated
trained-source provenance. The `FnMut` RNG wrapper (700–710) meters fill bytes;
it proves neither entropy quality nor independence, including closure captures.

**Required contract; obligations remain OPEN.** Separate input-independent
preprocessing from authenticated operand contribution. Reuse or extend the
model-neutral root prepare/encode boundary; do **not** move `RootInputFor`
wholesale to an evaluator: it contains the full mask and both common owners.

The selected SIGMA private-model scenario ([source pin](../../paper/methods/source-lock.v1.json), PoPETs 2024(4), pp. 61–79)
protects client input and proprietary model (Introduction, p. 61); gamma/beta are model-owner parameters (§3.1.6, Eq. 1, pp. 64–65),
not end-client inputs assigned for convenience. Dealer preprocessing is independent of both parties' inputs (§2.2, p. 63);
an input-wire owner receives its mask and contributes `xhat = x + r` to its peer, **not the dealer** (§§2.4.1–2.4.2).
LayerNorm consumes `xhat`, `gammahat`, `betahat` (H.3, Fig. 12, p. 78); interior X is a protected predecessor output
(§§2.4.2, 3.2.3/Fig. 13), not a generic client clear value. PLLM's authenticated source/contribution capabilities and consuming ingress remain OPEN.
Model-server knowledge of weights does not authorize full preprocessing-dealer state; public checkpoint availability does not authorize switching scenarios.
Distinguish public bindings from private operands without hiding operands from their legitimate owner; no silent dealer/client authority expansion.
Apply §7's per-role SIMULATION CUT and mask/key-knowledge table to these
contributions and the consuming transition below: retain legitimate owner
knowledge of `W,s,K`; preissued material/history needs its own conditional law.
The paper does not supply D43's exact `[-4096,4095]` range-ingress proof: local contract/range checks and the paper's static semi-honest,
noncolluding model are not malicious authenticated-input or range proofs.

For fixed public bindings, the **entire preprocessing distribution**, including
errors and randomness, must be independent of private X/gamma/beta. Review must
then jointly simulate each approved owner's authenticated contribution, full
D43 view and bridge/matrix composition. Local range checks are not remotely
authenticated range proofs. RNG closure captures, entropy/independence and child
factory proofs remain OPEN; apparent call-site separation does not discharge them.

D63 authorizes derivation/review only. Ring50/Q12, `[4] -> [1,4]`, original root
ledgers, unique-use/burn, admission and all existing gates, including native
headroom, remain unchanged. Independent source-contract review does not discharge the separation and joint-simulation obligations or grant protocol, implementation, security or native-execution approval.

### Authentic consuming transition

Before issuance, precommit the original terminal wire and unique matrix consumer
under the same source/session/profile and original root ledger. Permit only
row-major `[4] -> [1,4]`, lane `j -> (0,j)`, preserving Ring50/Q12; no numeric
reinterpretation or rescaling. Require matrix `inner = 4` plus independent weight,
numeric and range checks; this reshape creates no signed no-wrap certificate.

**Proposed authentic issuer/type chain.** Role-local authenticated predecessor or
contribution → original D43 native owners → original `ShareFor<C>` → sealed bridge
continuation → pending `CommonMaskedFor<C,Ring>` → unique matrix operand. Consume
both genuine output handles role-locally, without outstanding loans; no issuer
collects both online shares. Preserve original strong source/declaration/creation
provenance and `Arc` identity plus original leases/root ledger, not digest adoption,
copied arrays, caller tags or a replacement coordinator. Protected predecessor X
ingress and authentic model-owner gamma/beta contributions remain prerequisites;
`RootInputFor` holding full mask and both destinations stays OFF evaluators.

**Proposed construction.** Bind original terminal and downstream use BEFORE
entropy; admit all retained/incremental/downstream/failure cover and cumulative
work first. Independently preprocess `rho,a in R^4`, `r0=a`, `r1=rho-a`, delivering
each `ri` privately only to its role. Retain `rho` in a unique opaque
`DealerMaskFor` continuation precommitted to that consumer. Dealer receives no
online shares, openings or completion payload. This reuses the common native
contracts with proposed sealed backing alternatives, not parallel SIGMA types.

| Proposed state/owner | Consuming obligation |
| --- | --- |
| Admitted | Original terminal/use and downstream consumer fixed; original ledger covers the full quote before entropy. |
| Issued | Each role receives only its authentic `ShareFor` continuation and private `ri`; dealer retains its independent mask continuation. |
| Begun | Consume original share and `ri` use BEFORE frame escape; spent continuation retains actual backing and cover. Open only `mi=Yi+ri mod 2^50`. |
| Validated | Authenticate peer/channel/transcript and check all original producer/source/session/profile/wire/role/operation/phase/lane-order/consumer-use bindings. Four LE u64 payload lanes have top 14 bits zero; REJECT, never normalize. |
| Prepared | Own unpublished `H=mi+mpeer` and its cover. Preparation binds exact ORDERED opening transcript and original use. Acceptance of preparation is not publication. |
| Committed | Verify separate authority's durable non-equivocating COMMIT binding both preparations (§4). Joint authorization, not simultaneous delivery. |
| Consumed | Move pending common backing AND cover ONCE into the unique matrix operand; dealer continuation is consumable only by its precommitted consumer. |
| Burned / IN-DOUBT | Precommit local terminal burn requests abort; unknown shared decision remains unpublished, covered and non-reissuable. No local cleanup claim proves global abort. |

Postcommit loss preserves COMMIT with no fresh issuance/reconstruction retry;
local cancel cannot revoke peer disclosure. Retire locally owned secrets and
frames BEFORE cover; spent work/uses never refund. Delivered frames, prior
observations and external strong references cannot be retroactively erased by
dropping the terminal handle. No remote-erasure/crash-cleanup guarantee follows.
Section 4 governs pending-output commit→accept custody. Exact sealed layouts,
authority/channel construction, complete funding and joint security remain OPEN;
this detailed proposal grants no API/protocol implementation or activation approval.

### Existing public quote versus unpriced phase-transition owner

`issuer::resources::<C>()` (384) is already independent of clear codes; Phase A must freeze approved public profile, width four,
range, metadata/table geometry, context/session and resource/use state, with independent RNG/context behavior.
`issuer::LocalClearRootSpec4` (50–68) contains only private-code references, no quantizer field: Phase A must neither construct/read
that spec nor create clear copies; `typed.rs::RootInputFor::encode` (608–629) reads private codes and cannot belong to Phase A.
`typed.rs::LocalDealerFor::from_reserved_namespace` (1482–1498) directly calls `getrandom::fill`, not the supplied callback;
preserve its existing 32-byte entropy charge. `typed.rs::root_payload_bytes(4)` (584–599) already pays mask, both destinations
and wrapper for each of three roots: this is not missing payload capacity or permission to reuse those credits.
`issuer::IssuanceControls` (309–343) has one optional root/encoded transient, not a phase-return owner retaining three pending roots
and destinations alongside completed native party-key initializers/`RowMaterial`; `q.keeper` (394–400) prices current layout/link allowance only.
Freeze that concrete pending owner, party/dealer separation and return/`Err`, contribution, abort and delivery overlap before quoting
incremental owner/move costs against original leases. These costs remain **UNPRICED**; removed spec/clear slots are not automatic credits.
Preserve original reservation splits (`issuer::issue_reserved_with`, 687, 712–740), `IssuanceWork` charges before callback (701–710)
and per-role `issuer::ledger` (638–648) through failure, without fresh budget or reset. `_original_run: None` (1054–1059) supplies no authentic predecessor lineage.
Role arrays confer no registry/delivery authority; contributions must not reach the dealer, nor the whole mask package an operand owner.
API/protocol adaptation still requires user approval and independent security review; no new Rust signature/type is authorized here.
All existing D63 derivation-only, Ring50/Q12, `[4] -> [1,4]`, headroom/memory gates and OPEN ingress/joint-simulation obligations remain unchanged.

**Additional source blocker — transitive custody remains unresolved.**
Removing `InitializerState.keeper` and outer `CoordinationFor` alone cannot separate evaluator custody.
`typed/layernorm4/parent/coordinate.rs:56–116` retains `Arc<ModularCreation>` through
`ModularParty.creation`, `CoordinateParty.modular`, `ModularFrame.creation` and `ModularValue.creation`;
that creation owns native product, energy and interval masks, source and lease.
Online checks still require the original energy binding/source and `interval.is_for`
(`coordinate.rs:347–360`), creation pointer identity and interval/source provenance (`533–568`).
`source_lookup/interval50.rs:119–120` compares role and original declaration pointer in
`ValuesFor::is_for`: its `MasksFor` argument does not mean this check reads mask values.
These dependencies require separate nonsecret ORIGINAL provenance and lifetime custody,
not digest labels, a fresh caller-supplied source, or deletion of authentic identity checks.
`parent/runtime.rs:133–161,256–260,309–313` reads keeper bindings/source, not mask values;
nevertheless `parent/issuer.rs:147–159,194–196,953–965` retains dependency funding and drop order.
Deleting that `Arc` therefore requires explicit replacement accounting retaining existing leases
through every dependent native owner, frame, output and failure path; it creates no refund credit.
The revised target is ONLY a proposal: transitive dealer-secret unlinking plus authentic nonsecret
provenance/lifetime extraction across native and coordinate owners, with a shared model-neutral
`RootInputFor` prepare/contribution extension, not a paper-specific duplicate or approved redesign.
No noninterference, numeric equivalence, security or funding is established. Three-root payload funding
remains retained; new pending/`Err`/provenance/delivery overlap remains **UNPRICED**. Model-owner gamma/beta,
protected-predecessor X ingress/conversion **OPEN**, no dealer receipt of masked contributions,
distributed abort and joint simulation **OPEN** remain unchanged. This proposal does not authorize
Rust/API/protocol implementation; all existing review and activation gates remain held.

**Proposal-only custody amendment (not implementation approval).**

- Source-supported clean ownership closure is `SourceFor` metadata plus fixed
  `CertifiedTableFor` (8192 constant/index entries), `DeclarationFor`'s original
  source and CURRENT SEALED `LeaseFor<&Budget>` / `LeaseFor<OwnedBudgetContext>`
  counter-only implementations (`shared_allocation.rs:210–263`): no dealer parent,
  mask or key; this is not a future generic guarantee
  (`typed/layernorm4.rs:29–41,87–93`, `SourceFor`/`ProducedValueFor`). Cleaner
  backlinks are **not** a noninterference/security proof. Propose one issuer-created,
  immutable 1:1 nonsecret provenance cell per creation: original source and interval
  declaration strong handles, COMPLETE energy/output bindings, original creation
  cover plus incremental lease; never mask/key/value/native frame/callback/secret parent.
- Dealer creation also holds that original cell. Replace creation pairing only
  with the same original cell pointer, retaining original source/declaration,
  role/phase/use checks; equal metadata/digest is not authority. Existing declaration
  identity is explicit in `source_lookup/interval50.rs:52–60,119–120`
  (`DeclarationFor`, `ValuesFor::is_for`). No already-issued identity migration;
  no new authentication for `finish(peer)` or bare-byte paths lacking creation
  pairing. Party-native coefficients/local values/lanes/GPU keys remain **SENSITIVE**.
- Actual escape holes remain in `typed/layernorm4/parent/coordinate.rs:543–585`
  (`CoordinateParty::advance`): bare arithmetic `FrameFor` owns bytes+lease only,
  and Index's `ProducedValueFor` lacks creation retention. Propose a shared sealed
  private ROLE-LOCAL retained-owner pattern, following
  `source_lookup/norm13.rs:33–44` (`FrameFor`, `ValueFor`), not numeric duplication:
  sensitive backing + strong original cell + incremental control lease. Every
  party/frame/output/intermediate and future derived owner retains custody until
  its backing retires; borrowed bytes only. Owned moves atomically transfer all
  retention before publishing backing; failure overlap retains both live owners.
  Drop backing BEFORE leases; original creation charge survives the last dependent,
  INCLUDING dealer masks. Copies/serialization require explicit bounded funded
  owners; consuming transport retention handoff remains **OPEN**.
- Cell/Arc/wrapper layouts, strong handles, construction, argument/result moves
  and failure propagation remain **UNPRICED positive incremental** costs, with no
  removed-field credit or early refund. Retain existing root payload, entropy,
  work/uses, all keeper/`RowMaterial` funding and original dependencies. Preparation/
  private-contribution split, authentic X ingress/model-owner gamma/beta,
  distributed handoff/abort and joint security remain **OPEN**. This is a contract
  proposal only, not approved Rust types/APIs/protocol/redesign/authoring/execution.

#### Proposed construction-custody handoff

Direction approved for detailed funding review only; implementation remains unauthorized. Independently **SOURCE-CLEARED conditional finding**, not protocol adoption, security approval or approved Rust/API changes; paths below are under `crates/pllm-garble/src/masked_wire_reference/`.
`typed/layernorm4/parent/coordinate.rs:56–65,250–268` keeps the original creation `LeaseFor<C>` and splits `q.product_dealer` only after caller-held energy/interval secrets exist (`parent/issuer.rs:850–863`, same typed prefix).
Moving that cover into a late cell alone is insufficient: on error the last cell could drop before caller-held backing.
Authentic `DeclarationFor` identity arises only from `EnergyMaskLoan` in `source_lookup/interval50.rs:294–323`; fixed entries are public metadata (`typed/layernorm4/parent.rs:156–159`), not an early original identity.
Thus the complete cell cannot precede those secrets without changing the authenticated dependency.
Conditionally propose, in outer `issuer::issue_reserved_with`, after authentic `reservation.bytes == q.total_bytes` validation (`issuer.rs:683–686`) but BEFORE clear-copy/dealer/mask locals:
exact early flow is split incremental cover → incremental `Arc<LeaseFor<C>>` anchor → split original modular aggregate (`q.modular = structural.product_bytes`, `issuer.rs:569`) → split `product_dealer` → original-cover `Arc<LeaseFor<C>>` anchor.
Declare incremental-cover anchor FIRST, original-cover anchor SECOND, covered secret locals AFTER; both outer strong anchors remain UNMOVED through every callee, error and unwind.
Late cell shares BOTH anchors: a direct incremental lease held only in that cell would not cover the outer original-cover anchor after cell failure.
Independent review conditionally clears only this two-anchor custody under private cell/anchor handles with no escaping Weak, raw ownership escape or independent cover extraction; otherwise Weak can retain Arc backing after lease refund. Terminal-drop accounting is logical only: the incremental lease drops BEFORE its own Arc allocation is physically deallocated, consistent with the existing logical tariff, not physical-through-deallocation coverage or a global RSS bound.
Final creation secret fields precede its cell handle; proposed cell field drop order is original source Arc, original interval-declaration Arc, COMPLETE energy binding, COMPLETE output binding, original-cover Arc, incremental-cover Arc.
Source/declaration handles are authentic original clones only; this authorizes no extra Rust types or visibility changes.
No secret parent, key, mask or value belongs in that cell. Revised modular handoff must account for the ALREADY-EXTRACTED dealer slice:
the old `lease.bytes == q.product_bytes` check and second `q.product_dealer` split (`coordinate.rs:250,268`) cannot remain unchanged; remaining modular lease continues existing issuance without duplicate splitting.
No refund, re-reserve, same-budget adoption or removed-field credit. Caller-borrowed `spec.codes` and RNG captures are NOT covered by this modular creation lease.
`Arc::new` allocation-failure abort is not a recoverable-`Err` cleanup guarantee; conditional error/unwind custody must not imply abort recovery.
Earlier allocation/splits change FAILURE/attempt ordering; successful entropy sequence MUST remain unchanged, but this has not been validated.
Nonsecret closure relies only on CURRENT sealed `&Budget`/`OwnedBudgetContext` (`crates/pllm-garble/src/shared_allocation.rs:210–263`), not a generic future guarantee.
Positive base increment includes ONE cell Arc allocation plus TWO lease Arc allocations, using existing `issuer.rs:39–43` `arc<T>` helper/allowance; embedded fields are not duplicate backing charges, and old fields supply no credits.
Proposed internal attachment counts are conditionally known below; containing/transient layouts, argument/result storage, failure overlap, early allocation and literal W/D remain **UNPRICED**: no numerical full quote or whole-funding closure follows from this source finding.
Existing `resources.total_bytes` must include that positive increment, admitted in the original reservation BEFORE entry; no spare `q` credit or self-funded quote work.
Propagation into frames, indices and consuming transport remains **OPEN**; this conditional construction finding does not close those custody obligations.
Escaped-owner census (same source prefix): `typed/layernorm4/parent/coordinate.rs:293–318,336–407,479–505,523–586` retains creation in masks/two parties and modular frame/value plus inputs, but coordinate begin/advance returns bare arithmetic frames and an index retaining only source provenance.
Per completed role the current edges yield one modular frame, one modular value, two coordinate frames and one index; dealer `take_index` yields one index per creation (`coordinate.rs:123–126`, `parent/issuer.rs:878`), while `parent/runtime.rs:308` stores the role index in values[13]. These are existing edges, not new Arc-slot prices.
`parent/issuer.rs:184,215–241,345–353` shows FrameHandle/ShareHandle retaining whole InitializerState through CoordinationFor; retirement clears values but leaves stored frames, so finish/cancel is not backing retirement (all paths here retain the `typed/layernorm4/` prefix).
Extend/reuse shared `arithmetic.rs:282–305` FrameFor/MaskedOutputFor/StepFor and `typed/layernorm4.rs:81–92` ProducedMaskFor/ProducedValueFor contracts, following `source_lookup/norm13.rs:34–43` retained FrameFor/ValueFor ownership, not parallel SIGMA frame types.
Each proposed role-local owner needs actual sensitive backing, a strong cell handle and funded lease, with backing dropping first; the cell alone cannot replace storage, ModularInputs, locking or retirement.
Full source/declaration/binding/role/phase/use checks remain mandatory; this census supplies neither numeric costs nor implementation/API approval.
Input-independent preprocessing, role isolation and joint security are NOT established.
Existing HOLD4657, matrix activation, authoring and native-execution gates remain unchanged; no implementation authorized.

### Funding and review gate

Checked peak admission must include retained D43 backing, separated role controls,
bridge frames/material/scratch, matrix material and move-failure overlap. Preserve
original nonrefundable work/attempt charges, including issuance, opening,
validation and cleanup. Concrete numeric ABI/layout totals remain **UNKNOWN**;
§5's known payload arithmetic is not a complete quote or permission to invent prices.
Extend original `Resources`, `IssuanceWork` and `internal_use::Ledger` before
entropy. Count unique allocations and distinct inline live slots, including
moves, Options/Results, locks, transport and cleanup. Arc cell clones are not
allocations; cover anchors retain original identity, never a fresh ledger. The
64 entropy bytes, 20 add/sub operations, 28 reductions and two 32-byte sends are
only a bridge partial subtotal, not full funding or a 64 MiB fit proof.

Independent joint simulation must include full prior D43 views, bridge
preprocessing/openings, correlated matrix keys, abort leakage, fresh splits and
fresh output masks. Section 7's conditional extension does not discharge those
premises or existing weight-mask history. Next
artifact is an independently reviewed **pre-issuance D43 role-separation and
consuming-handoff contract**, not executable design. Implementation, security,
native execution and D43 matrix activation gates remain held.

#### Proposed base-cell layout (funding review only)

Independent conditional source clearance covers layout only;
Rust/interface/visibility authoring needs explicit user approval.
Proposed coordinate `LateCell<C: AllocationContext>` has this field DROP order:

| Field | Proposed type |
| --- | --- |
| source | `Arc<SourceFor<C>>` |
| declaration | `interval50::DeclarationHandleFor<C>` (PROPOSED) |
| energy, then output | `Binding`, then `Binding` (complete bindings) |
| original_cover | `Arc<LeaseFor<C>>` |
| incremental_cover | `Arc<LeaseFor<C>>` |

Under the masked-wire prefix, `source_lookup/interval50.rs:56–60` has private `DeclarationFor`:
public intervals, original source and lease; nonsecret only for CURRENT sealed contexts.
Proposed handle privately holds `original: Arc<DeclarationFor<C>>`, authentically cloned from
`MasksFor`; `ValuesFor` comparison preserves role and `Arc::ptr_eq` (`118–130`). Cross-branch
type/accessors need `pub(in crate::masked_wire_reference)`, not `pub(super)`.
No public constructor, raw/Weak escape or cover extraction; reuse `source_is`,
`is_for` and `output` where owners remain. The cell must not retain secret `MasksFor`.
Fresh 1:1 cell creation follows authentic declaration (`typed/layernorm4/parent/issuer.rs:861–870`).
Cell identity replaces no source/declaration/Binding/role/phase/use checks
(`coordinate.rs:245–264,347–360,423–440,533–535,568–571`, same parent prefix).
`D_base = checked_sum([A(LateCell<C>), A(LeaseFor<C>), A(LeaseFor<C>)])`, where A
uses existing `issuer.rs:39–43` padded `arc<T>` allowance plus 64: exactly THREE
new backings, no fourth for the embedded handle, no recreated source/declaration
allocation or old-field credit. Containing/shared-owner layouts require a separate positive checked increment, not blind `n * sizeof(Arc)` accounting. Copies/transients/escaped W/D remain UNPRICED.
Both earliest anchors, drop order/privacy and logical-only terminal accounting remain mandatory.
No physical-through-deallocation proof, entropy, security or input-independent preprocessing
approval follows. All existing gates remain held.

##### Conditional attachment census (funding review only)

Two independent Astra source audits conditionally clear this proposed census, not
implemented custody or funding proof. Paths are relative to
`crates/pllm-garble/src/masked_wire_reference/`; `P = typed/layernorm4/parent`.
Both roles can retain these strong cell attachments past coordinate output and
before backing retirement (`P/coordinate.rs:293–318,365–407,479–505,523–586`):

| Retained destination | Attachments |
| --- | ---: |
| Modular and coordinate dealers | 2 |
| Modular and coordinate parties, both roles | 4 |
| Two `ModularInputs` backing allocations | 2 |
| Modular frames and values, both roles | 2 + 2 |
| Coordinate begin/advance frames, both roles | 4 |
| Retained `TRMaskedOutput`s, both roles | 2 |
| Dealer index and role indices | 1 + 2 |
| **Total retained strong cell attachments** | **21** |

Dealer index moves at `P/issuer.rs:878`; role indices reside at `P/runtime.rs:308`.
These are attachment totals, **not 21 new allocations**; the base remains exactly
three backings. Issuance has seven attachments. Under the frozen successful-path
policy, MOVE the initial cell handle into dealer `ModularOwner` and directly clone
into the other 20 destinations: **20 new cell clones**, with no scratch clones.
Current `issue_modular` (`P/coordinate.rs:293–318`) instead clones its construction
root into all three owners and drops the root; copying that pattern adds a clone
and provisional reference outside this bound. Partial construction/errors may
retain fewer attachments, but the clone-work bound assumes direct attachment and
no retry; it is not a current-source guarantee. Moves and `Result`/`Option` do not
inherently clone, but their transient storage slots still require pricing.

Each anchor backing peaks at two strong references (unmoved outer anchor + cell),
then one: two anchor clones into the cell. One additional original declaration
clone enters the cell, with no separate declaration backing allocation. Declaration
retention totals ten: mask 1 + parties 2 + paired frames 4 + values 2 + cell 1
(`source_lookup/interval50.rs:71–115,475–514`). Each paired wrapper holds one
declaration reference despite two bodies; the nine existing owners are separately
funded, not a new ten-owner increment.

`ModularInputs` has two actual backing allocations, with two references per role:
party + frame, then frame + value. The value moves into coordinate input
(`P/coordinate.rs:537–538`), adding no third reference. Preserve actual native/local
input owners and leases: a nonsecret cell cannot replace custody or isolation.
Cancellation/Initializer retirement does not retire all backing
(`P/coordinate.rs:332–335,519–522`; `P/issuer.rs:215–241`): frames, inputs and TR
can survive, and failure can retain TR while dropping the index.

The escaped coordinator graph has 15 `FrameHandle`s per role despite 17 reserved
storage slots per role: 30 frames + 2 `ShareHandle`s. `CoordinationFor` strong peak
is at most 34 including two finishing receivers; nested 21 cell owners are NOT
multiplied. The separate EXISTING cumulative successful-path Coordination clone
census is 35: issuer 1 + initializer-to-role 2 + frames 30 + shares 2
(`P/issuer.rs:1055–1071`; `P/runtime.rs:22,449,476`). Existing quote ownership does
not automatically make this a new 35-clone increment; literal work coverage remains OPEN.

Dispatch/finish APIs are private; `FrameHandle`/`ShareHandle` have no `Clone`, and
loans borrow. No reachable unbounded internal clone/retry was found
(`P/runtime.rs:19–124,431–487`; `role_coordination.rs:109–159`). Caller Arc-wrapping
of an opaque handle creates caller-owned storage outside the closed callee quote,
not approval for another API/consuming API, a global retry cap or infinite internal
reserve. Future actual copies, serialization and transport retention still need
funding; the existing consuming-transport gate remains unchanged.

Checked containing/transient enum, `Option`, `Result`, argument, control and
error-overlap layouts remain OPEN, as do exact cell/anchor/declaration clone,
drop/attempt work and the original-admission increment. No authoring, native,
security, protected-X, role-isolation or entropy claim follows. HOLD4657 and all
existing gates remain held; this is funding review only, not implementation proof.

##### Reusable retained-wrapper funding direction (proposal only)

Independent Astra 14 review supports detailed funding review of a private shared
inline `Retained<B,C,R>`, not completed D/custody or interface-authoring approval.
It OWNS actual backing B, closed authentic strong LateCell retention R and a control
`LeaseFor<C>`; completed-struct field drop order is B → R → control. No Weak, raw
escape, extraction, mutable owner, `DerefMut`, callback or public adoption is proposed.
Current public/shared arithmetic `FrameFor`/`MaskedOutputFor` remain unchanged,
with no SIGMA branch; private containing alternatives would enlarge actual enums,
`Option`s and `Result`s. The native → source → lease order in
`source_lookup/norm13.rs:34–43` is a precedent, not a generic erasure guarantee.
If `B::drop` panics, remaining fields drop during ordinary unwind: this does not
promise indefinite cover or completed erasure. Double panic/abort has no full-cleanup
guarantee; the concrete backing audit is conditionally complete as scoped below, not a generic B proof. Partial construction
needs a separate local/argument/field-initialization unwind record; both early outer
anchors stay unmoved and are declared before covered secret locals.

The reviewed closure covers `FrameFor`/`MaskedOutputFor`/`OutputMasksFor`: arithmetic Zeroizing Vec u8/u64 (`arithmetic.rs:270–295`), product Zeroizing boxed slices (`elementwise_product.rs:75–86`), and source-lookup frames/outputs through lookup/selection scalar-zeroizing backing (`source_lookup.rs:380–403`).
It includes norm13 native → source Arc → lease (`source_lookup/norm13.rs:20–43`) and Produced Local/Arithmetic/Inverse/Original actual native/vector owners (`typed/layernorm4.rs:47–92`; `typed.rs:239–262,300–306,341–345`).
TR is arithmetic output, indices own Local vectors, and inverse owns norm13 (`P/coordinate.rs:575–585`; `P/runtime.rs:303–330`).
Sealed `LeaseFor<&Budget>`/`LeaseFor<OwnedBudgetContext>` only refund atomic counters/drop Arc Budget, not payload (`shared_allocation.rs:116–120,210–283`). Enumerated backing destructors have no user callbacks, poison or locks; Coordination retirement guards are separate.
Locked zeroize 1.9.0 `Zeroizing::drop` calls zeroize; Vec retirement performs volatile writes, clear and spare-capacity wiping (`lib.rs:270–291,434–448,520–537`). Its checked-multiply expect and isize assertion remain present: valid Vec u8/u64 allocation invariants exclude their failure, not their existence.
Boxed scalar slices zeroize before Box drop. The normal valid-owner semantic path is nonpanicking; final allocator/deallocation boundaries are not promised abort-free or failure-free.
Only normal completion establishes backing retirement before R/control release. Unwind dropping remaining fields does not prove B completed; double panic/abort still has no full-cleanup guarantee.
Actual zeroizing writes imply no wiped-RSS, historical-copy or declaration-erasure claim. Concrete private wrapper alternatives remain unfrozen; arbitrary generic B is unauthorized.
Retirement work must include actual zeroizing work under initialized/capacity bounds unless existing funding demonstrably covers it; the JSON C tariff is not reusable here.

Candidate checked `ΔD = D_base + Σ n·L(new containing layout) + Σ m·S(new transient control)`
is added ONTO existing funding with no old-field credit; `D_base` has three distinct
backings. Count nested fields once per containing storage and pointee allocations
separately. Conservative overcharge is not a completeness proof. There are exactly
two RoleSlot Arcs and two Coordination handles, not two inline InitializerStates;
21 attachments / 20 cell clones imply neither 21 allocations nor graph multiplication.
Anchor and declaration clone work remains positive and UNPRICED.

Independent Astra storage/liveness review 20 is complete with this conditional census,
not completed custody or funding. Here P is `typed/layernorm4/parent`; C/I/R mean
`P/coordinate.rs`, `P/issuer.rs`, `P/runtime.rs`; K=`typed/role_coordination.rs`,
L=`typed/layernorm4.rs`, N=`source_lookup/norm13.rs`. Base, nested and transient storage stay distinct:

| Storage group | Multiplicity, inclusion and source anchors |
| --- | --- |
| NEW base backings | Cell Arc 1 + anchor Arcs 2; source/declaration/bindings/covers are cell fields, not extra allocations. |
| Existing changed ModularCreation / RowMaterial Arcs | 1 each (C:56–65,293–318; I:150–159,956–966). |
| Existing RoleSlot / Coordination Arcs | 2 slots, each with Mutex + inline InitializerState (K:18–21; I:169–197,990–1020); 1 coordinator contains two slot HANDLES (K:46–50; I:1054–1073). |
| Existing ModularInputs Arcs | 2 allocations; two strong references per role, party/frame → frame/value (C:66–71,365–383,398–404). |
| Inline parties, frame slots and coordinate inputs | 4 parties inside RoleSlots; FrameOwner Options 17/role = 34 slots, widened even when None or another variant (I:162–184; R:441–452); two ModularValues move into coordinate input, not new Arcs. |
| Nested TR and indices | TR 2 in Options (C:116,556–585); dealer index 1 in masks[13] (L:47–61,81–86; C:479–488; I:878); role indices 2 in values[13] (L:65–92; C:577–585; R:303–317). Dealer index's native vector is an actual owning wrapper, not guard-only. |
| Existing escaped handles | 30 emitted frame + 2 share handles within the existing 34-handle allowance; no coordinator-graph multiplication; caller Arc-wrapping is caller-funded. |

Control records below are not necessarily instantiated aggregates or coexistent branches:

| Control inventory | Multiplicity and source anchors |
| --- | --- |
| ProductIssue / CoordinateIssue | 1 each, five split Results each (C:129–140,187–192 / 155–165,201–206). |
| Outer Issuance | 1, 24 lease Results, arrays/inline partials and child-result slots (I:309–343,394–400). |
| ProductEvaluator / CoordinateEvaluator | 2 each (C:142–154,193–200 / 167–175,207–213). |
| Transition | 2 (I:360–379,401–411; R:431–457). |
| Existing norm13 issuance / evaluators | 1 / 2, funding unchanged (N:52–65,103–109 / 67–81,95–102,198–246). |

Child controls can overlap outer controls; begin/finish are different calls, sibling
issue calls are sequential, and Result/enum alternatives do not all coexist. Pricing
still needs case-specific distinct storage or explicit conservative overcharge, never implicit ABI reuse.
Attachments reconcile as dealers 2 + parties 4 + inputs 2 + frames 6 + values 2 + TR 2 +
indices 3 = 21; initial MOVE + 20 clones remains conditional, not 21 allocations or approved clone-W prices.
TR is stored at C:575 before the fallible split at C:585; R:303–317 stores index BEFORE
inverse begin. Frames/TR/role slots/inverse controls can coexist. Cancel does not retire
inputs/TR (C:332–335,519–522); retirement clears values, not frames (I:215–241); inverse Err is not retirement.
OPEN: the two extra outer-anchor locals, cell construction/Result, declaration-clone
temporary and new split/return slots have unfrozen exact multiplicity/overlap. Concrete
private alternatives/containing layouts, target size/alignment, partial-construction
order, Err/unwind cover, allocator boundaries and final-wrapper destructor proof remain open beyond the conditional concrete closure above. Original-reservation coverage
must include the already-extracted dealer slice; both early outer anchors stay unmoved.
Exact split/allocation/clone/drop/attempt W remains UNPRICED. Full original funding stays,
with no old-field credit; symbolic D additions are not numerical ABI or a full quote.
The proposed generic private interface is NOT authoring-approved. HOLD4657 and all
security, role-isolation, ingress, input-independence, transport, entropy,
full-funding and native-execution gates remain unchanged.
