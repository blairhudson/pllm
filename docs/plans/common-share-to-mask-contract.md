# Common additive-share to common-mask consuming contract

Date: 27 October 2026. Contract-only proposal, Astra request 67599.
Source inspection: root `23d691c6f0d353d988bcb405305bf4a9a1674479`.
D59–D64 remain frozen approvals, not execution evidence or completed methods.
No implementation, D43 matrix activation, transport authentication, or security
qualification is conferred here. Independent Astra joint-simulation and custody
review must precede any conditional Rust implementation or bounded tests.

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

State: `reserved -> issued -> share-consumed/frame-issued -> peer-validated
-> both-ready -> published`, each role progressing within the same coordinator.
Publication happens once, only after both roles validate and become ready.
There is no cloning, fork, retry, reissuance, or second use of `rho`/`ri`.
Reserve before construction or entropy; issuance irreversibly spends its use.
Wrong root/role/use/shape, replay, out-of-order phase, malformed/noncanonical
residue or frame, missing peer, and premature finish refuse before publication.
Failure/drop/cancel/timeout/unwind burns both roles and all dependent uses;
no sibling may publish after an asymmetric or partial-send failure.

Keep original D43 backing, shares, dealer/evaluator masks, scratch, unpublished
send/receive frames and outputs under retained owning leases until exact owned
erasure completes. Keep downstream dealer masks and evaluator backing charged
through their final consuming operation. Borrow guards and partial transitions
must not release backing or refund admission early. Specify erasure order and
bound cleanup on success, cancellation, poison and unwind before implementation.
No promise covers caller copies, network buffers, peer memory or delivered data.

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
| Online send frames | Two four-u64 payloads: 64 B aggregate exchange payload only. Add exact native header, alignment, owner, capacity and serialization/delivery buffers for each frame. No native header is frozen. |
| Receive/serialization | Two received payloads alone are 64 B if both retained. Enumerate bounded ingress, validation, parsed arrays, send serialization and queued/partial-send copies at maximum overlap; Python `pack`/`unpack` is not a native allocation bound. |
| Outputs/downstream | Two `Yhat` payloads: 64 B, plus actual common-owner/backing capacities and leases; retain opaque `rho` owner and dependent matrix/key/consumer reservations until final consumption/erasure. |
| Controls/failure | Price coordinator, role slots, root/use ledger extensions, reservations/results/options, locks/Arcs, move temporaries, bindings, cancellation and timeout state; enumerate normal and partial-failure peak plus bounded erasure workspace/work. |
| Ring work | Dealer: four subtractions/four reductions for `r1`; evaluators: eight additions/eight reductions for `mi`; reconstruction: eight additions/eight reductions total across both roles. Together with draws: 20 add/sub operations and 28 reductions; prior D43/downstream work remains additional. |
| Non-ring work | Exact bounded frame encode/decode/validation/copy counts and byte visits, root/use transitions, entropy calls, delivery operations and cleanup visits must be priced from the final layout, including failure paths. Two evaluator frames, eight opened masked lane values total; also charge private dealer deliveries and their framing. |

**Funding/admission BLOCKER:** no sealed consuming bridge owner/layout, native
header/encoding, capacity/copy schedule, dependent-consumer retained-live set, or
bounded cleanup/work quote exists for this transition. Thus no exact combined
byte total or complete work allowance can honestly be asserted under 64 MiB.
An implementation proposal must fill every bound above using actual target types
and preconstruction capacity limits, reconcile the original ledger, and prove
`peak(root retained + bridge + downstream + failure overlap) <= root allowance`.
No construction/entropy is allowed while any term is unknown. Charging work is
an operational admission obligation, never a cryptographic security proof.

## 6. Conditional regression definitions and remaining gates

Future tests only after independent Astra joint-simulation/custody review:

- Independent integer ring oracle for four lanes, wrap boundaries and all roles:
  exact `Yhat = (Y0 + Y1 + rho) mod 2^50`, unchanged lane order and Q12 metadata.
- Authentic lineage success; equal-digest/different-root, caller-array, wrong
  source/profile/role/use/session/wire/shape, replay and phase-order refusal.
- Exact-storage/exact-work admission succeeds; each one-unit-short bound refuses
  with zero entropy before admission and no partially usable capabilities.
- Partial send, cancel, timeout, drop and unwind at each transition burn siblings
  and dependent uses; both-ready publication occurs once, never prematurely.
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

## 7. Independent review outcome — Astra28 (activation held)

Actual D43 terminal arithmetic is **RING50/Q12**, residues stored in `u64`,
not Ring64 arithmetic. Conditional bridge simulation: given an authentic prior
evaluator view and its `Yi`, sample independent uniform `ri,H` in `Z/(2^50)`
per lane, derive `mi=Yi+ri` and the peer message `H-mi`, with `H=Yhat`.
This accounts for the proposed bridge view only, contingent on fresh independent
preprocessing and the stated leakage/bindings; it is **not** a full D43 or joint
composition proof.

Source `matrix/core.rs::{deal,evaluate}` splits the matrix keys
`A=-rho`, `B=-rw`, `C=rho*rw+rz`. With common-masked inputs `H=Y+rho` and
`What=W+rw`, role `i` evaluates
`Zi=Ci+Ai*What+H*Bi+[i=1]*H*What` in the declared ring and matrix layout;
the sum is `Y*W+rz`. Any conditional simulation of these correlated keys depends
on fresh independent splits and a fresh independent output mask `rz`, not the
marginal bridge pad alone. `typed.rs::LocalDealerFor::matrix_with_ticket`
checks typed dimensions/numerics and draws the output mask; this is not joint
security clearance.

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
