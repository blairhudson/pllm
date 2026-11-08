# Consuming publication: source obligations and conditional certificate proof

**DESIGN ONLY / CONDITIONAL / UNSELECTED.** This note records reusable scientific
protocol obligations, finite counterexamples and a bounded theorem, not a receipt
collector, new API or protocol selection. Native Rust, Python and guards remain
frozen. No implementation, execution, funding, security or activation follows.
Existing authority requirements, admission, custody, caps and HOLDs are unchanged.

## Source boundary

The governing SIGMA proposal is [common-share-to-mask §4](../plans/common-share-to-mask-contract.md#4-linear-lifecycle-and-erasure),
including pending-output custody and the separate exclusive decision authority;
[§7 authenticated views](../plans/common-share-to-mask-contract.md#d66-conditional-authenticated-view-extension)
and [§8](../plans/common-share-to-mask-contract.md#8-pre-issuance-separation-and-authentic-consuming-handoff-proposal)
retain full-view, pre-issuance separation and authentic consuming-handoff gates.
The BumbleBee boundary is [OLT §18](../plans/bumblebee-encrypted-olt-closure.md#18-common-candidate-bound-consuming-decode-admission--design-only)
and [§18.1](../plans/bumblebee-encrypted-olt-closure.md#181-selected-logical-output-mapping).
[D63](../../paper/methods/DECISIONS.md#d63-conditional-sigma-exact-share-to-common-mask-consuming-contract),
[D66](../../paper/methods/DECISIONS.md#d66-research-only-nonresumable-sigma-bridge-lifecycle) and
[D67](../../paper/methods/DECISIONS.md#d67-bumblebee-wider-full-u-exact-down-design-target)
bound their respective design choices; none supplies an implemented issuer.

Two inspected accounting anchors must not be promoted into protocol authority.
[`shared_allocation.rs:538–550`](../../crates/pllm-garble/src/shared_allocation.rs#L538-L550)
requires an externally retained slot until associated backing retires, but transfers
capacity only, not custody or whole-payload retention. At
[`689–705`](../../crates/pllm-garble/src/shared_allocation.rs#L689-L705), charge Drop
releases bytes and field order destroys the direct payload before its charge.
Neither fact proves last-allocation destruction, wiping or distributed ownership.
[`row_fold.rs:27–59`](../../crates/pllm-core/src/row_fold.rs#L27-L59) records cumulative
nonrefundable work: successful debits subtract; failed checks subtract nothing.

## A. Six-case comparison

For both methods, genuine issuer identity and original-use binding must precede
the affected transition. Caller labels, matching hashes and logical role names
cannot mint authority. Original attempt/use and successfully spent work remain
permanent history. Capacity instead follows actual retained LOCAL backing, counted
once at full capacities and parent tails, including escaping aliases and overlap.
Proven retirement may release capacity independently of nonrefundable use/work.
Bindings and charges survive until actual retirement or proved atomic transfer.

**1. Unauthenticated refusal.** SIGMA refuses before an unauthenticated claim can
select an original use, begin it or publish. BumbleBee uses independently funded
ingress for public authentication before consuming genuine authority or reading
secrets. In both, preauthentication labels or failed lookups cannot burn unrelated
siblings; already incurred, properly funded ingress work is not retroactively free.

**2. Authenticated consumption.** SIGMA admits original retained, incremental,
downstream and failure resources before construction or entropy; `begin` spends
the authentic share/mask use before ANY opening frame escapes. BumbleBee first
authenticates and consumes its original one-use authority, then performs fallible
covered admission before allocation, initialization or secret read. These distinct
boundaries do NOT impose universal consume-before-ALL-preprocessing ordering.

**3. Admission failure.** SIGMA refusal before `begin` authorizes neither entropy
nor frame escape; it does not erase original issuance/attempt history. Failure
after beginning cannot restore that spent use. BumbleBee covered-admission failure
after authenticated consumption burns only the genuinely bound inventory and
returns no usable permit. For both, a failed debit spends nothing itself, but
cannot refund prior successful debits, consumed authority or incurred work.
Retire partial backing before releasing its capacity; do not invent fresh ledgers.

**4. Committed but unaccepted output.** SIGMA publication requires authenticated
joint COMMIT binding the exact ordered BOTH preparations, exact ordered opening
transcript and original downstream use. Local readiness is insufficient.
BumbleBee's immutable logical map fixes cells, directions and share roles, NOT
principals or release entitlement. Commit release disposition before possible
exposure; prefix, atomic and ACK policy remain UNSELECTED. For either method,
pending output needs stable externally retained disposition AND genuine backing
plus cover custody. Unique acceptance atomically transfers backing and cover
before effect; an ordinary network ACK does not establish that transfer.

**5. Partial or lost delivery.** SIGMA's D66 unknown decision stays unpublished
IN-DOUBT with permanent abandonment: no resume, reissue, refund or reconstruction
retry. An existing COMMIT survives loss and local burn cannot revoke it.
BumbleBee preserves committed entitlement and any disclosed-prefix disposition;
serialization failure, timeout or lost delivery never resets authority. Neither
method infers global abort, simultaneous delivery or remote erasure from cleanup.

**6. Unknown retirement.** Burn or logical last use is not physical retirement.
Both methods retain authentic binding, owner inventory, backing and charges until
retirement is proved. Borrowed input's original owner retains its wipe duty.
Peer, caller and kernel memory are not automatically charged to a local root;
ownership/accounting partitions need proof. Every distributed copy or root needs
its own custody, cover and transfer proof, without double-counting shared backing.

## B. Finite counterexamples to insufficient local evidence

**Indistinguishable local histories.** Fix one local view containing admission,
begin, preparation, a peer-ready ACK and then timeout. In history C, both valid
preparations reached the separate authority, joint COMMIT issued, but final delivery
to this role was lost. In history N, the peer preparation failed authority
validation and no COMMIT issued. The ready ACK is not authenticated joint-decision
evidence, so these histories can have identical listed local observations.
Local consumption plus ACK cannot distinguish C from N: publishing on that view
violates authorization in N, while local burn does not establish global abort in C.
Adding another ready ACK does not remove this finite last-message ambiguity.

**Escaping backing after Err.** Consider a proposed shared API that clones an Arc
to secret-bearing backing into a send queue, then returns Err and drops its stack
charge. The queue keeps the allocation live while charge Drop releases capacity:
live backing is uncovered. This is a counterexample to that proposed API, not a
claim that the cited implementation contains such a path. Payload-before-charge
Drop on one Arc destroys one reference, not necessarily the last allocation;
even last-reference destruction alone is not a wipe proof. Capacity-only splitting
cannot repair missing backing custody or cover retention across this escape.

## C. Bounded two-role certificate theorem — CONDITIONAL, UNSELECTED

This theorem studies COMMIT uniqueness and single acceptance only. It is NOT a
selected replacement for the current authority. Fix these explicit hypotheses.

An authentic issuer fixes unique ORIGINAL use `u` across aliases and epochs,
source/root/session/protocol/profile, ORDERED role verification keys and one fixed
acceptance-state domain, destination and downstream use. Relabeling an epoch or
creating another alias cannot create another original use or acceptance domain.
Enrollment, key validation and issuer construction remain UNSELECTED.

A canonical domain-separated COMMIT message binds that entire context, BOTH
authentic PUBLIC preparations and the exact ORDERED already-authorized opening
transcript. No public hash of secret masks or private backing is introduced.
Encoding and any digest construction must bind content unambiguously; their
concrete choices remain UNSELECTED, as do all primitives and state mechanisms.

Before irrevocably signing at most ONE COMMIT digest for `u`, each honest role
checks its own frozen preparation, authenticated peer preparation and full context.
Its sign-once state changes BEFORE any signature can escape. At least one fixed
designated role remains honest, uncompromised and non-equivocating throughout the
use lifetime; alternating honesty or later key compromise is not this hypothesis.

A verifier individually checks BOTH designated signatures on the identical message,
using scheme-valid keys and malicious-peer-key-resistant enrollment/verification.
Assume EUF-CMA security and collision-resistant binding if hashing is used.
Arbitrary peer-chosen key relations must not defeat the designated honest-key check.
No aggregate-signature shortcut or specific enrollment scheme is selected here.

The authentic consumer's ORIGINAL state atomically accepts `u` at most once,
including every replica and alias. Its genuine pending owner atomically transfers
backing AND cover before any acceptance effect. Signatures do not supply this
consumer atomicity or secret custody. D66 excludes cloning, rollback and resume
of issuer, evaluator, authority and consumer state; uncertainty means permanent
abandonment, with no reissue, refund or reconstruction retry.

**Conditional proof.** Any accepted certificate contains a valid signature under
the fixed honest designated key, hence an authorized honest signature except for
forgery. Two conflicting COMMIT contents for the same original `u` require forgery,
a binding collision, or that honest role signing twice, contrary to hypotheses.
The consumer state rejects replay by original `u`, not signature-byte deduplication.
Another valid signature encoding on the same message therefore grants no second
acceptance; strong unforgeability is unnecessary. A corrupt signer's asserted
truth and private computation are not proved by its signature. The claim is
qualitative for finite PPT executions, except forgery/binding failure; it adopts
no 40-bit target, concrete probability, numerical qualification or parameter set.

**Critical nonrefinement.** Two-signature COMMIT authentication can establish unique
COMMIT and single acceptance under these hypotheses, NOT the current authority's
exclusive COMMIT-or-ABORT functionality. Both roles can block progress. Loss of
the last signature does not establish global abort or revoke a peer's COMMIT.
Replacement needs explicit weaker-functionality selection OR a separate proved
cross-decision locking/abort construction; both remain UNSELECTED. No simultaneous
delivery, liveness, privacy simulation, secret custody, erasure, numerical or
funding proof follows. Actual D63 full-view metadata, timing, credentials and role
isolation remain open: dealer/evaluators stay distinct; dealer gains no online access.

## Finite unresolved construction list

1. Genuine issuer/enrollment and atomic original-use authentication/consumption.
2. Authority exclusive cross-decision locking/abort proof, or explicit weaker selection.
3. Canonical public certificate binding, primitives, keys and sign-once state.
4. Original consumer replica/alias atomicity and genuine pending backing-cover transfer.
5. Entitled destinations and actual atomic/prefix/ACK release boundary.
6. Full local/remote ownership partition, escaping aliases, retirement and wipe proofs.
7. Complete original admission/failure/cleanup funding without new tariffs or resets.
8. D63 joint full-prefix simulation, credentials, timing and pre-issuance role isolation.
