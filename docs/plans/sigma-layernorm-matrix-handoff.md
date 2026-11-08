# SIGMA private four-lane LayerNorm-to-matrix design

Status: **held; historical local design, not implementation approval**. Completed
Astra privacy/custody reviews supersede the former frozen/approval-ready wording.
The constructor, keeper, driver and accounting sections below preserve useful
history for the old coupled-memory proposal only; none qualifies a transitively
separated distributed scheme. No implementation, composed-security clearance,
D16 adoption or D43 matrix activation is authorized.

## Current review boundary (supersedes historical schedule claims)

USER71483 approves detailed bridge DESIGN only. Concrete issuer/type/construction,
authority and security proposals live in the shared contract's
[§8 Authentic consuming transition](common-share-to-mask-contract.md#authentic-consuming-transition),
with [§4 failures/authority](common-share-to-mask-contract.md#4-linear-lifecycle-and-erasure),
[§5 funding](common-share-to-mask-contract.md#5-concrete-resourcework-inventory-and-admission-blockers)
and [§7 privacy](common-share-to-mask-contract.md#7-independent-privacy-review-activation-held).
Those integrated requirements govern this boundary; historical local schedules
below are not a second bridge plan. No interface/protocol adoption, code, numeric
change, tariff, implementation, tests or execution is authorized by this tranche.

Separation must happen **transitively before private issuance**, not at terminal
handoff. Online `Role`/`Frame`/`Share -> CoordinationFor.slots ->
InitializerState.keeper` currently reaches dealer secrets. Coordinate
`ModularParty`/`ModularFrame`/`ModularValue`/`CoordinateParty` also retain
`ModularCreation`; removing the outer keeper alone does not separate custody.
This includes `PartyInitializer`/`Role`/`FrameHandle`/`ShareHandle.coordination`
through `slots[0..2]`, every creation backlink and nested input owner. Bare coordinate
begin/advance frames and Index output retain actual backing plus cover; cancel
leaves inputs/TR and Initializer retirement leaves frames. Terminal-handle drop
cannot undo prior observations or external strong references. Public control keeps
only original source/session/profile/wire/use and phase/decision metadata, no
secret slots or RNG backlinks. Authentic protected-X predecessor ingress and
model-owner gamma/beta contributions remain OPEN prerequisites.
Keep dealer `RootInputFor` (rho and both destinations) off evaluators. Online
matrix owners retain only their own coefficients, never dealer or peer secrets.

Proposed shared authentic nonsecret provenance cell retains original `SourceFor`,
original declaration identity, full energy/output `Binding`s, and original plus
incremental cover anchors. It has no secret parent, callback, native mask or peer
key. Escaped bare coordinate frames and Index output must retain cover too.
Reuse shared owners/provenance/admission machinery, not parallel SIGMA types.
These are unimplemented contract requirements, not approved interfaces.

Bind genuine terminal `Yi` and unique matrix use BEFORE entropy. The proposed
role-local chain is authenticated predecessor/contribution → D43 native owners →
original `ShareFor` → bridge continuation → pending `CommonMaskedFor` → unique
matrix operand. Preserve original strong provenance/`Arc` identity and root ledger,
not digest adoption. Exact reshape is `[4] -> [1,4]`, `j -> (0,j)`, Ring50/Q12,
no truncation/rescaling/new range certificate. Independent private preprocessing
delivers `r0=a`, `r1=rho-a` for uniform independent `rho,a in R^4`; opaque retained
`rho` is a unique `DealerMaskFor` for the precommitted consumer only. Dealer sees
no online shares, openings or completion payload.

Proposed consuming lifecycle: `Admitted -> Issued -> Begun -> Validated ->
Prepared -> Committed -> Consumed`; precommit failures become `Burned`.
`begin` spends the original use before frame escape; its spent continuation keeps
its own backing/cover. Bridge payload: exactly 32 bytes, four LE u64 residues
with top 14 bits zero. Reject malformed ingress without reduction; preserve
binding/digest, role, phase, count and canonical checks. `finish` needs an
authentic committed-use capability, not matching labels or ready messages.
Prepared owns unpublished `H` with preparation bound to exact ORDERED opening
transcript and original use; accepting preparation is not publication. Move
pending backing AND cover once into the unique operand only after verified COMMIT.

Proposed SEPARATE durable non-equivocating authority requires authenticated
enrollment, genuine issuance, exclusive COMMIT-or-abort, crash persistence and
consumer-verifiable authentication; COMMIT binds both preparations. It could ensure
**no publication without joint authorization**, not simultaneous delivery or global
abort after commit. Labels, ACKs and `require_both_ready` are not authority; no
existing native/Python owner supplies it. Precommit cancel REQUESTS abort, not
proof of global abort; unknown decision stays IN-DOUBT, unpublished and without
reissue. Postcommit loss preserves COMMIT, with no fresh issuance/reconstruction
retry; local cancel cannot revoke peer authorization/disclosure. Retire owned
secrets/frames before cover; spent work/uses never refund. No remote-erasure or
crash-cleanup promise. Authority/channel construction remains OPEN: without
approval, stay blocked or obtain an explicitly weakened contract.

Bounded counterexamples live in shared
[§4 authorization/freshness](common-share-to-mask-contract.md#proposal-counterexamples-authorization-is-not-freshness),
[§7 matrix distributions](common-share-to-mask-contract.md#7-independent-privacy-review-activation-held)
and [§8 custody paths](common-share-to-mask-contract.md#source-backed-separation-boundary):
P0's retained coordination reaches `slots[1].state.share`/`Y1` despite owning `Y0`;
removing that edge misses keeper/creation/nested-input paths. Proposed
`A0=rho,A1=-2*rho` preserves the correlation identity but reveals `Y=H-A0`;
authorization is not independence. Restoring authentic pre-use owners, ledger and
acceptance state can replay authorization without forgery; SQLite crash durability
alone is not rollback safety. Actual changed-input `rho` reuse gives
`H-H'=Y-Y' mod 2^50`, not merely replay of identical bytes. Proposal examples are
not current-bridge exploits. One-sided delivery loss requires permanent burn of
the uncertain instance, no reissue/refund/recovery, and no premature cover release.
D66 selects research-only NONRESUMABLE lifecycle for the next security review:
exclude issuer/evaluator/authority/consumer rollback/cloning; permanently abandon
uncertain instances/uses, no reissue/refund. Separately authenticated, durable,
non-equivocating authority remains required; construction/API and authenticated-
two-role alternative remain UNSELECTED. Retained backing erases before storage-cover
refund; spent work/uses never refund, provenance retains no secret parents/RNG/
callback paths. D63 independent review and original funding/custody/admission/masks/
arithmetic/HOLDs remain; role isolation and complete simulation/funding stay OPEN.
No production antirollback, security qualification or implementation/execution grant.

Funding remains unresolved. Price actual-layout peak over unique allocations
and distinct inline live slots, including moves, Options/Results, locks,
transport and cleanup. Cell Arc clones are not new allocations; original cover
anchors preserve original identity, not a fresh ledger. Extend original
`Resources`, `IssuanceWork` and `internal_use::Ledger` before entropy. Known
bridge partial subtotal is 64 entropy bytes, 20 add/sub operations, 28 reductions
and two 32-byte sends: not a complete quote or 64 MiB proof. Old fixed-driver
numbers below apply only to the old proposed local design. Full separated
layout/work/admission, joint security and publication authority remain OPEN.
The shared §5 phase peaks retain original D43 cover plus issuance masks/scratch/
private delivery, and Prepared's both mask shares, two sends, two received peers,
two unpublished `H` owners, bounded codec copies and downstream material. Extend
original `Work` and existing issuance/transition controls too; no fresh inventory
or removed-field credits. Embedded arrays are not double-charged, but each distinct
owner/move/error slot is priced. Headers/auth/queues/persistence/recovery/failure
overlaps remain UNKNOWN; finite entropy-retry/wait/recovery bounds are not adopted,
and IN-DOUBT has no duration bound. Check retained + incremental + downstream +
failure peaks and cumulative cleanup work BEFORE entropy. Maverick's atomic-call
choice is not a SIGMA tariff. Execution and memory-headroom gates stay held.

Shared §7 proves only conditional bridge arithmetic: for genuinely separated
fixed `Vi,Yi`, fresh `rho,a` independent of ENTIRE prior execution give independent
uniform `(ri,H)`; simulate `mi=Yi+ri`, `mpeer=H-mi`. Actual-view extension needs
specified authenticated channels, authority and leakage kernel covering bindings,
canonical bytes, sender identity, ordered commitments, preparation/decision/delivery,
lengths/timing/aborts. No forged credentials or hidden private-failure assumption.
Prior D43/weight history, downstream matrix `Material a,b,c`, fresh masks and dealer
view still require JOINT proof; D17 alone is insufficient. All unknowns and HOLDs
are preserved. Freeze here: no implementation or execution continuation.

## Provenance and scope

Audited committed baseline `78d65ae3198f6c00c63182290cb164a27792af09` against
historical audited snapshot `1af1dbbcc308c224063b99d1393084f934193023`. Between
those snapshots there are no
committed differences under `crates/pllm-garble/src/masked_wire_reference`,
`crates/pllm-garble/src/shared_allocation.rs`, or
`crates/pllm-core/src/edge_contract.rs`. Dirty working copies are not evidence.
Existing plans inspected: `sigma-activation-ownership.md` and
`ten-method-execution-decisions.md`. This is a new bounded endpoint, not an
extension of their already-approved activation scopes.

D17/D39/D43 in `paper/methods/DECISIONS.md` constrain existing approved scopes;
D39/D43 confer no adoption authority for this new SIGMA endpoint. Approval of the
COMMON common-mask primitive and separate approval of the composed transcript
and its implementation are prerequisites, not consequences of this design.
D17 requires fresh uniform full-output-ring arithmetic masks; literal one-bit
arithmetic-output masks remain prohibited. Boolean masks remain one bit.
Source provenance remains `paper/methods/source-lock.v1.json`. Governing cached paper:
`papers/r13-sigma.pdf`, previously locally SHA-256 verified as
`f911b715b0e0523fa4d77f21c28c6a0440c5be781e67b5b30419f20e505a6ebf`.
Fig.12 returns shares of a masked value; masked-output reconstruction and
composition are described in the paper. This does not establish security of
the complete D17/D43 adapted transcript.

Only LocalClearRoot4, original D43 integer map, reconstructed common-masked
terminal, and one private `[1,4] x [4,1]` matrix. No inherited LN inputs, wider
rows, transport, compiler route, SDK, trained execution, new numerical profile,
or changes to existing D39/D43 endpoints.

Ring M=2^50. X/gamma/beta retain D43's Q12 domain [-4096,4095]. Let F be its
final affine floor and Y=F+beta. Reconstructed native terminal gives
Fhat=F+rF. Both roles calculate Yhat=Fhat+betahat; dealer calculates
rY=rF+rBeta, all modulo M. No clear Y or additive-share endpoint is constructed.
W ranges over every signed ring50 Q12 code. Output is
sum(Y[i]*W[i], i=0..3) modulo M, Q24; no rescaling/no-wrap interpretation.

## Historical local constructor schedule (not separated-design clearance)

Names below are proposed private fields/calls, not claims about committed code.
Reuse `issuer`, `runtime`, `coordinate`, `internal_use::Ledger`,
`role_coordination`, and native issuers. Extract fixed prefix construction and
dispatch; do not duplicate the D43 implementation.

1. Validate only fixed public dimensions/identity metadata; calculate checked
   byte/work vectors. Reserve the entire aggregate once on the original counter.
   No private copy, table/hash factory, namespace entropy or root draw precedes
   reservation. After admission, scan/copy the four fixed-length clear roots;
   any failure consumes the request. Acquire the existing native matrix
   `IssuanceTicket` after reservation and before the first namespace/root entropy;
   move that ticket into borrowed matrix issuance, never reacquire it there.
2. Split aggregate into exactly six slices: prefix Q43, terminal A4, W root W4,
   matrix M4, affine payload Y288, and new containing metadata L. Prefix uses its
   original sublease schedule, but does not construct D39 terminal or share
   owners. Retain its terminal allowance as slack, never early refund it.
3. Split L exactly nine times: keeper, role0, role1, coordination, controls,
   typed F-mask wrapper, typed Y-mask wrapper, typed Y0 wrapper, typed Y1 wrapper.
   Each named slice is sized from its actual containing type. Parent L retains
   any rounding/slack until final destruction.
4. Issue original prefix through affine product. Construct one original
   namespace, three LN roots, certified table and unchanged prefix factories.
   Issue reconstructed native four-lane GapARS under A4; wrap original rF.
5. Prepare/encode W under W4: existing root helper splits exactly twice. Split
   Y288 exactly three times into capacity-four mask/value payload owners (each
   32+64 bytes). Allocate/initialize each once; compute rY once. Preallocate
   Yhat0/Yhat1; fill only at driver step16.
6. Issue matrix material under M4 using genuine retained rY/W mask owners.
   Reuse/extract the existing `matrix_with_ticket` borrowed issuance body;
   preserve its validation, binding, RNG order and native calls. Its current
   consuming adapter drops input masks: it cannot itself retain them in keeper.
   The extracted helper borrows originals and constructs the same material;
   public consuming adapter semantics remain unchanged. Exactly three existing
   matrix wrapper splits: two party charges and output-mask charge.
7. Construct keeper, two state slots and coordination. Driver owns coordination
   and keeper. States own original native parties and one keeper Arc each.
   Fill no frame slot during issuance. Construct two one-use reshape permits.
8. Run the closed driver. Matrix finish splits one result-wrapper lease/role,
   twice total. Transfer both results and driver's keeper Arc into one private
   result envelope. No fresh allocation, namespace, binding or lease split here.

Thus additional lease splits relative to retained prefix/native interiors are
`6+9+2+3+3+2 = 25`. There is one original aggregate reservation, not six reserves.
These are successful-path splits, not a bound on error-path metadata-retention
splits (see appendix). Each successful split creates one lease; every created lease releases once,
including zero-byte parents. The new layer has 26 eventual lease-drop events
including aggregate root; add prefix/native-created leases separately. Failure
uses only the successfully constructed prefix of this event list.

No constructor may opportunistically create extra bindings. Prefix keeps its
18 binding issuances. New bindings are exactly F, Y, W, C: four counter
increments/mask-ID digests. F uses terminal `[4]`; Y uses explicit `[1,4]`;
W `[4,1]`; C `[1,1]`. W and C are issued by existing root/matrix wrapper paths,
not issued again by the join. Existing descriptor values are stored/copied, not
rehash-derived from private values. Three new descriptors Y/W/C are constructed
once; terminal F uses the original fixed output descriptor.

## Field ownership and moves

`JoinKeeper<C>` owns the retained RowMaterial Arc (which owns F), Y mask, W mask, C mask,
fixed `JoinBinding`, keeper metadata lease, residual L lease, and the
dealer/prefix ownership link. The other eight L children are owned by their
named destination: two state slots, coordination, controls, F mask, Y mask and two
Y values. Y/W do not move out during matrix issuance.
Only masks enter dealer custody; no online Fhat/Yhat/frame is borrowed by dealer.

Each `JoinState<C>` owns the original InitializerState by value, native reconstructed
terminal, optional Fhat, preallocated typed Yhat, typed W value, typed matrix
party, optional matrix result, optional reshape permit, `[Option<JoinFrame>;19]`,
phase/write-index/retired flag and keeper Arc. The enclosing RoleSlot owns the
single role lease; JoinState has no duplicate metadata lease. Typed Y wrapper owns
its metadata lease and its affine payload lease; no raw unleased Vec adoption.
The original state's terminal/share Options and seventeen frame slots stay
empty; only the join's nineteen slots receive frames. Original Fhat remains
owned after Yhat calculation. W/Y common owners are
consumed into matrix begin exactly once; native matrix material and retained
dealer inputs remain independently owned.

Each permit owns exactly one keeper Arc, full producer/destination binding,
role, operand/use ordinal and consumed state. Its storage is inline in state;
no individual lease or allocation. Consume before checking. Validate pointer
identity, original counter, source/session/invocation, exact descriptors,
ring/scale/role/destination/use. Element count is not reshape authority.

New Arc backing allocations: keeper, two state slots, coordination = four.
New strong-reference clones: two state keeper links and two permit keeper links
= four. Driver initially owns keeper and coordination constructions. A result
envelope receives driver keeper by move; no clone. Prefix/native Arc activity
stays in their inventories. There is no separate escaping role/frame handle.

`IssuanceControls` owns four clear `[i64;4]` copies, constructor Options/Results,
all move temporaries and controls lease. `TransitionControls` owns borrowed
frame-parts arrays, native step Result, one moved permit, one moved matrix input
pair and result Option. Their actual containing sizes belong to L. Normal Rust
field moves are logical move events, not promises about compiler stack copies.

## Finite driver and cleanup

19 scheduled entries: begin; advances1..13; affine completion/terminal begin14;
terminal exchange15; terminal output/affine/reshape/matrix begin16; matrix
finish/mark-ready17; both-ready/result move18. Per role:17 frame-owner groups,
19 wire parts; retained array capacity19. Peer bytes are borrowed in-place.

Successful upper counts:19 cancellation atomic loads,19 coordinator operations,
38 role locks,34 peer-group arity checks,38 peer-part borrows,34 slot stores,
two permit consumptions, two ready marks, one both-ready check. Public loan,
parts, callback and retry invocations: zero. Matrix begin performs at most eight
typed input comparisons total; finish performs two splits already counted.

The new state's retirement method sets `retired` first. Subsequent calls return
immediately. First entry cancels each native party once, clears frame/value
Options once, and wipes live wrapper scratch once. Coordinate and modular
parties retain their existing downstream cancellation behavior. Caller cannot
invoke native cancel or access independent role handles. Success does not call
retire; destruction drops owned state after moving results.

Keep conservative coordinator bound of four abort invocations/eight state
retirement entries on error/poison/unwind, but only two first entries. Therefore
external cancel parameter **e=1 maximum per interval/norm13 role party**, not
eight. Evaluate successful destruction with e=0; cancellation path with e=1.
Partial-construction failure drops only actual constructed owners; no fake
cancel call on an unconstructed Option. No callback runs under locks.

Each new owner is constructed and disposed once: one keeper, two states, one
coordination, two permits, three affine Vecs, two terminal output owners, two
matrix party wrappers, two matrix result owners and one result envelope.
Options move, not clone. Wrapper-level explicit wipe applies only to affine
values still owned locally and clear scratch; consuming Y into matrix transfers
its wipe responsibility rather than adding a second owner. Clear scratch is
zeroized before controls destruction; destructor's second traversal is counted.
Retained backing delays original lease release until its last owner drops.

## Checked byte and work formula

Use the previously accepted deliberately conservative byte quote unchanged:

`B = checked_sum(Q43, A4, W4, M4, 3*(32+64), L)`.

Q43 is committed parent total, A4 reconstructed GapARS native quote, W4 existing
four-lane root quote, M4 existing matrix quote for (1,4,1). L sums actual
containing `sizeof`/Arc-layout charges for named keeper/slots/coordination,
issuance/transition controls, permit/frame/result wrappers and original
allowances; it includes step/write-index/retired/driver-armed fields. No new
dynamic scratch or early subtraction of old D39 reservation. Overflow or
original 64 MiB ceiling failure refuses before factories. ABI resolution is a
future compilation check, not an unspecified algorithmic bound.

Work is a checked vector sum, never a wall-clock scalar:

`W = I_prefix + I_driver + I_new + N_prefix + N_terminal + N_matrix + E(path)`.

I_prefix uses actual issuer/runtime/coordinate/table/root helper inventory;
N terms are independent native inventories (no duplicate wrapper/root RNG).
I_new has the fixed 25 splits, four binding issuances (including W/C), four Arc
allocations/four clones, three affine exact-capacity allocations, 12 zero-init
word stores, 12 affine add/reduce pairs, two permit checks/consumptions and
matrix typed wrapper operations. W-root and matrix-output allocation/fill work
belong to their existing wrapper inventory, not I_new again. When importing
those helper vectors, remove their binding and split coordinates already
listed in I_new; do not subtract their validation, hashing of native context,
allocation, fill or scan coordinates. This is accounting partition, not a
runtime refund. Prefix/native vectors likewise exclude the six outer splits.

Root/namespace RNG boundary: LN masks96 + W masks32 + matrix output8 + original
namespace32 = **168 bytes in18 calls** (12+4+1+1). There is no additional join
namespace. Add reconstructed terminal320 bytes and native matrix104 bytes;
native prefix remains separately counted. Output-mask8 is not in native104.

Hash expansion: typed length-prefixed digest input is sum(8+part length);
SHA256 blocks are checked ceil((input bytes+9)/64). A binding issuance hashes
one edge plus `len(mask_id_domain)+104` bytes. New four issuances therefore add
exactly four of those mask-ID messages with descriptor ranks1,2,2,2; do not add
W/C again in the root/matrix inventory. Nested binding/edge hashes in matrix
context remain charged at their actual existing call sites. No new join hash
is needed: full fixed binding records are compared directly.

Allocation/context/counter methods are abstract calls. `split` counts one
checked subtraction and one context-share invocation; every actual lease drop
counts one release invocation. Allocation counts exact-capacity checks and
allocator calls, not allocator internals. No CPU, OS entropy-latency, mutex
waiting, scheduler, physical erasure or RSS bound follows.

Wrapper scratch: four clear copies write128 bytes; range scans visit16 values;
explicit scratch wipe plus destruction visits256 bytes. Three affine buffers
initialize96 bytes and write96 bytes. At admitted exact capacities, their final
Vec destruction visits192 bytes (initialized length plus full capacity).
An explicit pre-drop wipe clears length: its additional lifetime cost is at
most96 capacity bytes, not a second 192-byte destruction. No extra wipe is
assigned to a transferred payload. Prefix buffers retain their existing explicit
retire plus destructor visits; table hashes65536 bytes/1025 SHA blocks and
eventually wipes131072 bytes (length plus capacity). No zeroization claim for unspecified
compiler temporaries.

The independently recovered native inventory is expanded in the appendix.
Logical wipe traversals do not prove physical erasure of compiler copies,
registers, or backend temporaries.

## Historical review packet (approval wording superseded/held)

Review must check proposed constructors against this event list: no extra
binding/namespace, no duplicate root/native entropy, no copy or uncharged
allocation from borrowing matrix inputs, exactly one retire gate per state,
correct e, and no erased owner's premature credit. Joint-transcript security
qualification remains separate; conditional algebra is not that qualification.

Former suggested implementation-approval wording is superseded and held. The
shared local keeper is not a qualified separated custody boundary; resolving
compiler layouts alone cannot close transitive custody, joint simulation,
original-ledger funding or distributed joint-authorization obligations above.

Future Sol ownership: private `typed/layernorm4/parent/matrix_handoff.rs` and
focused tests; minimal shared-prefix extraction in parent issuer/runtime;
minimal borrowed matrix issuance extraction in `typed.rs`. No source changes
are authorized by this document. Later exclusive-slot gates: independent full
four-value integer contraction oracle, wrong producer/use/shape/role and replay,
exact/one-short memory/work, zero pre-admission entropy/factories, injected
constructor failures, unwind/cancel and retained-owner drop accounting.

## Reconciled native-work and cleanup appendix (K)

### Accounting boundary and corrections

Source authority is the previously audited `78d65ae3` snapshot, not an assertion
that a working-file read proves a later commit identity. Reconciliation read the
whole proposed schedule and the relevant helper bodies; no source modification,
build, execution, entropy, network, or new commit was used. Paths below are under
`crates/pllm-garble/src/`, except explicitly named core/dependency files.

Use checked nonnegative vector sums. A component owns its allocations, original
payload initialization, RNG requests, direct hash inputs, and arithmetic;
`E(path)` alone owns explicit erasure and final destruction. Calling a child does
not charge that child's vector again. Source-level allocator/context/hash/AES
calls are events; their internal CPU instructions, allocator behavior, and
atomic retry loops are not bounded by this contract.

The proposed 25 successful additional splits and four new bindings check out:
`6+9+2+3+3+2=25`; W/C binding and root/matrix wrapper split events occur only in
`I_new`, even when their implementation is inside an existing helper. The original
namespace occurs once. Matrix input masks must be borrowed originals; the
consuming public adapter is not the retaining helper. Native matrix quotes
explicitly exclude borrowed operands; Q43/W4/Y288 keep those owners covered.

Two additional details must be preserved in the frozen implementation schedule:

* `matrix::party::IssuanceTicket::acquire` occurs before entropy. It owns one
  `Live::acquire` and one issuance `fetch_update`; no retry/reissue at join level.
* The 25 splits exclude failure-only metadata retention. Native matrix
  `PartyFor::cancel` calls `retain_metadata`, which can split once on the failing
  native operation. Typed `MatrixPartyFor::cancel` calls
  `retain_wrapper_metadata` on the failing wrapper and on each state's one
  external retirement: at most three wrapper retention splits, including zero
  byte splits. Charge at most four additional split/release events across this
  error path, not a second successful construction schedule.

### Disjoint numerical/native partitions

Let G(L) mean native reconstructed GapARS50/12, T(1) native TR50/42, P(L)
elementwise product. The actual native call list, excluding typed/root wrappers:

| Partition | Native factories | RNG bytes/calls | Issue AES blocks | Eval AES blocks, both roles | Heap factories |
| --- | --- | ---: | ---: | ---: | ---: |
| N_prefix | G(1), G(4), T(1), 3P(4), P(1), paired interval, inverse13 | 1528 / 131 | 436 | 4718 | 128 |
| N_terminal | G(4) | 320 / 25 | 80 | 80 | 11 |
| N_matrix | native (1,4,1), ring50 | 104 / 10 | 0 | 0 | 11 |
| Native sum | 11 outer factories, including nested children below | 1952 / 166 | 516 | 4798 | 150 |
| I_prefix + I_new root/namespace only | namespace; four 4-word roots; one C mask | 168 / 18 | 0 | 0 | 13 payload factories |
| Total RNG/AES | | **2120 / 184** | **516** | **4798** | native/root subtotal163 |

The certified table is one further payload factory in I_prefix. Prefix local
vectors, structural wrappers, and enclosing Arcs are not included in 163.
There are four arithmetic issuers, four product issuers, three Select issuers,
three correction builders, one inverse scanner, one paired scanner and one
matrix issuer; nested factories are not eleven more calls of each kind.

Derivations: G(L) consumes `(32+72L,1+6L)` RNG bytes/calls; T(L)
`(32+64L,1+5L)`; P(L) `(32+32L,1+4L)`; Select(L)
`(32+48L,1+6L)`. Paired interval including its two correction/Select children
is `(288,24)`; inverse13 including its child is `(176,13)`. Matrix native is
`(104,10)`; the additional 8-byte C-mask request is in the root row only.
`arithmetic::resources_for_context`, `elementwise_product::resources_for`,
`source_lookup::interval50::Resources::work_counts`, `norm13::resources`, and
`matrix::party::resources_for_context` supply the corresponding byte/AES terms.

Native successful lease splits are 32 arithmetic, 24 product, 18 Select,
27 correction (including reconstructed frame/output wrappers), 5 inverse
scanner, 9 norm13 wrappers, 17 paired wrappers, and 6 matrix: **138**.
Their direct allocation-context shares, excluding shares inside those splits,
are 4+4+3+3+1+1 = **16**. Each actual split contributes its own context share
and eventual lease release; the matrix failure-only split is additional.

### Complete GPU/AES event vector and temporary erasure

`compact_dpf/gpu127.rs` events across all native partitions:

| Event | Count |
| --- | ---: |
| Paired GPU issuance / sum of depths / ordinary issuance levels | 12 / 129 / 117 |
| LESS / descend / terminal_block / correction | 122 / 4424 / 122 / 4548 |
| walk nodes / branches / internal branches / leaf branches | 126 / 252 / 124 / 128 |
| child / children / cipher | 4546 / 384 / 4930 |
| AES single-block calls / two-block calls / total blocks | 4546 / 384 / 5314 |
| Retained GPU keys | 24 |

Depth is `bits-7`; paired issuance uses 4d AES blocks and 2d cipher schedules;
LESS uses d of each. The inverse has 63 walk nodes/126 branches and 8192 emitted
bits per role. The paired interval performs 51 iterations per role, reusing the
first prefix at the final endpoint: 102 LESS, 102 memberships/count updates,
204 weighted updates. G(4) terminal alone contributes 4 pair issuances, depth
sum20, 16 ordinary levels, 16 LESS, 64 descents/corrections, 16 terminals,
80 child, 40 children, 120 cipher and 160 AES blocks; subtract exactly this
from the table for N_prefix. Matrix contributes none.

For E(path), successful GPU temporary wrapper byte visits are the following
disjoint sum. Return wrappers moved to callers are counted at the consuming
site, not again as temporaries in the callee:

```
issue_with   = 12*152 + 117*50 + 129*64
child/cipher = 4546*64
children/cipher/ExpansionBlocks = 384*80
less         = 122*49 + 4424*16
descend      = 4424*48
terminal     = 122*32
correction   = 4548*16
walk         = 126*48 + 124*32 + 128*16
```

Each retained key initializes 983 secret-field bytes. Its explicit `Drop`
followed by `Zeroizing` field destruction visits `2*(16+55*16+55+32)=1966`
bytes, hence **24*1966** once, separate from these temporary terms. Unused
correction slots are still initialized/wiped; lane-vector padding is not.

Pinned aes0.8.4 has zeroize enabled. Committed `.cargo/config.toml` enables
`aes_armv8` on aarch64 macOS/Linux, with runtime software fallback. Full Aes128
hardware schedule construction/drop covers encryption and decryption schedules,
352 bytes/key; 64-bit software uses `[u64;88]`, 704 bytes/key. Charge **4930
constructors, 4930 drops, at most 4930*704 schedule-wipe bytes**, without assuming
which CPU branch runs. These are backend calls, not instruction counts.

Additional successful temporary wipes, excluding roots, retained buffers, and
the GPU/AES terms above:

| Source symbols | Bytes |
| --- | ---: |
| arithmetic::{issue_reserved_with,draw,split}, all four factories | 2136 = 10*128 + 49*8 + 29*16 |
| elementwise_product::issue_reserved | 1272 = 4*32 + 13*(32+32+24) |
| selection::{issue_reserved_with,draw}, three children | 456 |
| source_lookup::correction::push_index and lookup draw | 192 |
| source_lookup::begin_inner sum/count wrappers | 32 |
| interval50 nonce plus gpu127::interval50::{Scratch,ScanResult} | 234 = 32+2*(77+24) |
| matrix::core::deal / matrix::split | 288 |
| typed::fill_masks_with, four roots plus C mask, I terms only | 136 = 17*8 |

Arithmetic temporary terminal contribution is `4*128+20*8+12*16=864`;
N_prefix owns the remaining1272 arithmetic bytes. Root draw-buffer wipes belong
to I_prefix/I_new's E attribution, not native RNG/scalar rows again.

### Native arithmetic, validation, hashes, initialization and copies

These fixed visits supplement the event vector, not replace it:

* `typed/layernorm4/table::{construct,entry,certify}`: 8192 certificates,
  **212992** binary-search iterations; each entry has denominator construction,
  division, 26 midpoint/square comparisons and four certificate multiplications.
  Table construction/hash belong to I_prefix, not inverse native construction.
* Inverse scanner: 16384 emitted table multiply-adds, count additions and shifted
  indices across both roles; role1 negates sum/count once. Paired scanner counts
  are in the GPU table. Correction/Select loops each have L=1, three children.
* Product L-sum13: dealer13 multiplications/13 additions/26 negations/39 share
  subtractions; roles104 multiplications/78 additions, reconstruction26
  additions;182 reductions. Arithmetic L-sum10:20 finish-carry executions,
  each6 multiplications/4 additions/3 subtractions/2 masks, plus20 begin and20
  final add/mask pairs. Matrix24 MACs,8 negations,9 split subtractions,18 split
  masks,2 output masks,2 finish add/mask pairs.
* Select issuance per lane:4 multiplies,1 addition,1 negation,5 split
  subtractions,11 masks. Each evaluator:2 multiplies/2 subtractions; zero-bit
  branch adds1 multiply/2 additions; role1 adds1 multiply/1 addition. Count both
  roles for each of three children; no data-dependent loop count.
* Arithmetic canonical visits70 u64; product104; each Select8; inverse table8192
  plus one mask/two masked-index visits; each correction first peer two words
  per role; interval two scalar input visits; matrix54 u64. Peer headers:
  arithmetic16*76, product8*76, Select6*76, correction6*80, matrix2*76 bytes.
  Re-reading validated payload for arithmetic is additional to validation.
* Arithmetic per L: five zeroed word vectors40L bytes, two lane vectors,
  input copies16L, four serialized frames4*(76+8L). Product per L: zero
  initialization72L+2*(76+8L), then complete frame overwrite. Select per L:
  zero initialization104L, frame writes2*(76+8L). Correction per child: two
  16-byte populated mask vectors, five zeroed8-byte vectors, two96-byte frames.
* Inverse native: two65536-byte copied tables (131072 zero-init and131072 copy
  bytes), two8192-byte expansions,32 zeroed sum/count bytes. Paired native:
  64 populated sum/count bytes and51*4 u64 declaration entries. Matrix:
  168 zero-init bytes,24 copied bytes,168 frame-write bytes. Four root helpers:
  384 zero-init bytes,16 mask writes and32 masked-value add/mask writes; C:
  8 zero-init bytes and one mask write. Affine adds12 add/mask pairs separately.

There are **52 native SHA256 messages**, plus the one certified-table hash in
I_prefix. Set H(m)=checked ceil((m+9)/64). Literal lengths below are the actual
source byte literals, not an ABI-dependent parameter:

| Native source family | Exact message multiset |
| --- | --- |
| arithmetic, four factories | four bindings `len(DOMAIN)+len(Profile::DOMAIN)+211`; eight input messages `len(input-domain)+8L` at L=1,4,1,4 twice each |
| product, four factories | four bindings `len(DOMAIN)+290`; eight inputs `len(INPUT_DOMAIN)+160+16L` at L=4,4,4,1 twice each |
| paired interval | declaration `len(interval-domain)+51*4*8`; two invocation messages; two Select contexts; two Select bindings; four energy messages `len(energy-domain)+40`; four Select inputs |
| inverse13 | canonical-table `len(table-domain)+34+65536`; invocation; inverse extension `len(inverse13-domain)+32`; Select context/binding; two index messages `len(indices-domain)+40`; two Select inputs |
| matrix | base binding `len(base-domain)+258`; party binding `len(party-domain)+32`; two operand messages `len(operands-domain)+80` |

Lookup invocation length is
`len(DOMAIN)+len(MAGIC)+len(Profile::DOMAIN)+len(EzPC-literal)+371`.
Select context/binding/input lengths are respectively `len(context-domain)+32`,
`len(selection::DOMAIN)+241`, `len(input-domain)+8+16L`, here L=1.
Typed length-prefixed hashes are separate: for each actual call sum `(8+len(part))`
before H; include nested EdgeDescriptor/ConsumerUse/Binding digests, not merely
their32-byte results. Four terminal messages and four matrix messages partition
out of52; N_prefix has44. Certified-table hash is65536 bytes/1025 blocks.

Matrix private cancellation predicates: one pre-nonce call,24 calls from six
`pllm_core::streamed_matrix::accumulate_resident` traversals,9 coefficient-split
calls,2 evaluator-final calls = **36**, separate from19 driver atomic loads.
One failing resident traversal can additionally wipe its one8-byte output slice;
the subsequently dropped Vec still has its original length/capacity.

### E(path): capacities, failure and retirement

Use each admitted allocation's recorded len/capacity, not an assumed ABI or
allocator capacity. For zeroize1.9.0 numeric Vec, one call visits
`len_bytes+capacity_bytes`, clears len, then wipes the entire capacity. A later
call on that retained cleared vector visits capacity again. Box slices visit
their initialized length. Fields without a source wipe contribute zero.

Successful exact-capacity backing erasures (exclude temporaries above):
arithmetic G/T per L `4124L+608` including retained keys; product per L
`88L+152`; Select per L `240L+304`; correction child per L `240L+320` including
explicit begin-mask wipe; native matrix672; each four-word root192; typed C
mask16; certified table131072. **Do not also add24*1966 keys when using the
arithmetic formula.** Inverse copied tables/expansions and paired/scanner buffers
use the same Vec rule at their listed capacities; their owners are distinct
from the certified source table. Prefer per-owner len/capacity sums on failures.

For a failed `correction::PushGuard`, each of its seven build vectors gets one
explicit wipe and one final destructor: `initial_len_bytes+2*capacity_bytes`.
For a correction mask retained through native cancellation use the complete
lifetime upper bound, **replacing**, not adding to, its normal mask term:

```
norm13:   initial_len_bytes + (6+e)*capacity_bytes
interval: initial_len_bytes + (4+e)*capacity_bytes
0 <= e <= 1 per native role owner
```

These allow initial begin wipe, four/two internal cancel traversals, the one
external retirement, and final drop. Successful e=0 with no failure uses actual
begin/drop events, not the failure maximum. Guard failure and returned live-owner
cancellation are alternatives for an allocation, never summed on the same
owner. Remaining Option payloads drop once; no second whole prefix is issued.
Four coordinator abort calls/eight retire entries cause only two first-entry
retire bodies. Prefix broadcast has one ordinary zero-store traversal per first
retirement; local buffers use their explicit Vec wipe followed by final drop.
Matrix wrapper metadata retention can repeat without re-erasing consumed native
state; its split/release events were counted above.

This defines finite E(path) from the fixed allocation list and failure position:
sum the live owners' specified wipe events, GPU/scalar events reached before
failure, backend schedule drops, and at most one8-byte resident-cancel wipe.
Admission can use the complete successful temporary vector plus per-owner
maximum permitted failure cleanup; this overbounds a prefix without replaying
constructors. No RNG rejection loop was found; entropy requests are fallible
fixed-size calls. OS entropy/allocator/atomic internals remain abstract.

**Hygiene boundary:** plain rescale/product scalar intermediates and AES
key-expansion/encryption scratch have no corresponding explicit wipe in these
sources. Their additional wipe work is0, not infinity and not an invented
future wipe. Schedule/key drops do not prove erasure of registers/compiler
copies. No wholesale zeroization change is approved.

### Aggregate definitions and exact residual decision

The N and E terms above are closed finite native inventories. I_driver is the
19-entry schedule and its19 loads/19 coordinator operations/38 role locks/
34 arity checks/38 part borrows/34 stores/two ready marks/one both-ready check;
permit checks belong to I_new once. I_new is25 successful splits, four binding
issuances, four Arc factories/four clones, three affine factories,12 zero-word
stores/12 affine add-mask pairs, two consumed permits, W-root/C-mask helper work
minus their already-counted bindings/splits, and at most three error-retention
splits. Four new descriptor ranks are1,2,2,2, with only three new descriptors.

I_prefix must retain the exact source numeric call schedule: one `metadata`
(20 edge constructions including the array initializer and output,24 ConsumerUse
records), one namespace,18 bindings, three root prepare/encode calls, one table
construct, three dealer LocalMaps and six evaluator LocalMaps,72 Ledger burns
(24 dealer,24 each role), two arithmetic contexts and three product contexts,
one each coordinate modular/TR context, one EnergyMaskLoan/two EnergyValueLoans,
one Norm13MaskLoan/two Norm13ValueLoans. Its local allocation sites are three
local masks, one dealer broadcast, two role broadcasts, six local runtime
buffers; coordinate adds three one-word buffers, one ModularCreation Arc and
two ModularInputs Arcs. Runtime installs nine ProducedValues per role and moves
three local payloads per role. All nested edge/use/approved-implementation hashes,
source Arc clones, original lease splits and final releases belong to this term,
not to N or I_new. New-wrapper four-root clear-copy work must not also retain a
second copy/scan of the original three clear roots during extraction.

The constructor-tail ambiguity identified by K128 is resolved in the frozen
tail below. Apply its explicit retained/replaced constructor events to I_prefix;
do not execute and charge both old and new completing tails. Native K counts
are unchanged. The original25 additional successful splits are preserved; the
already specified error-retention events remain separate.

Byte quote identities are exact: Q43=`issuer::resources::<C>().total_bytes`;
A4=`arithmetic::resources_for_context::<C>(GapArs50/12,4).combined_reservation_bytes`;
W4=`typed::root_payload_bytes::<C>(4)`; M4=
`typed::matrix_reservation_bytes_for::<C>(ring50,1,4,1)` (not the smaller native
matrix quote); affine=288. The compiler-resolved size requirement is the checked
L sum for actual JoinKeeper, two RoleSlot<JoinState>, Coordination, issuance and
simultaneously live transition controls, F/Y masks, two Y values, result envelope,
and any extracted-prefix slack/return/move containers. Embedded permits and19
frame slots must appear in their containing types; existing original allowances
are retained. A4 covers native F payload, M4 native/typed C/material/output,
W4 original W owners; none covers unlisted enclosing join metadata.
Even with the parametric tail/layout below, **no numeric B or64 MiB fit claim** follows from
Q43 alone. Require checked B<=67108864 before all factories/entropy, then the
source-level work vector under the same reservation. No early refund, assumed
zero metadata, physical allocator/RSS or wall-time claim is permitted.

### Historical local constructor-tail decision (not current custody approval)

**Retain** the original `Arc<RowMaterial<C>>` and original `InitializerState<C>`
values. **Replace** the two old RoleSlot Arcs, old Coordination Arc and escaping
handles with the join's four Arc constructions. Extract construction through
affine issuance, stopping before `gapars_terminal::issue_retained_with`; do not
construct and then discard a D39 terminal. The old endpoint uses its original
completion tail. The following are proposed private types/signatures, not code
or compiler-verified layouts. `S(T)` means `size_of::<T>()`.

```
PrefixRoleParts<C> {
  values: [Option<ProducedValueFor<C>>;18], ledger: Ledger<24>,
  mean: arithmetic::PartyFor<C>, square: product::PartyFor<C>,
  interval: paired::PartyFor<C>, modular: coordinate::ModularParty<C>,
  coordinate: coordinate::CoordinateParty<C>, inverse: norm13::PartyFor<C>,
  normalize: product::PartyFor<C>, floor: arithmetic::PartyFor<C>,
  affine: product::PartyFor<C>, broadcast: Zeroizing<Vec<u64>>,
  local_buffers: [Zeroizing<Vec<u64>>;3], runtime_lease: LeaseFor<C>, work: Work,
}
PrefixParts<C> {
  roles: [PrefixRoleParts<C>;2],
  source: Arc<SourceFor<C>>, masks: [Option<ProducedMaskFor<C>>;18],
  modular: coordinate::ModularMask<C>, coordinate: coordinate::CoordinateMask<C>,
  bindings: [Binding;18],
  row_leases: [LeaseFor<C>;4], tail_slack: [LeaseFor<C>;5],
}
PrefixTail<C> {
  material: Arc<RowMaterial<C>>, states: [InitializerState<C>;2],
  tail_slack: [LeaseFor<C>;5],
}
```

`PrefixParts` owns the actual prefix payloads/native parties by move; broadcasts
have length/capacity4, role local buffers1/4/1. Its constructor borrows the one
dealer and admitted clear scratch. After F/W/C/matrix issuance, consume it with
`finish_prefix(parts, dealer, original_F_mask, completed_work) -> PrefixTail`.
Construct one RowMaterial Arc: move source (no source clone), masks, structural
owners, bindings and four original row leases. Put original F in the existing
`terminal_mask: DealerMaskFor<C,Ring>` field, with its L_F metadata lease and
native A4 backing. That owning type fits; no old D39 use of that field is invoked
on this endpoint. JoinKeeper does not also own F. This requires private shared
construction extraction, not a change to D39/D43 endpoint behavior.

Construct the two original InitializerStates by value. Move all role parts;
set terminal=None, share=None, frames=[None;17], phase=0, local_payloads=None;
clone RowMaterial exactly once/state. New dispatch uses the numeric native
prefix methods and returns frames directly to JoinState, never old Role,
PartyInitializer, FrameHandle, ShareHandle or loan factories. The original empty
slots/unused terminal/share storage remain charged inline, not zero-sized.

| Old tail event | Frozen disposition |
| --- | --- |
| RowMaterial Arc | one retained allocation, moved into JoinKeeper |
| RowMaterial source clone | replaced by move of PrefixParts.source |
| state keeper clones | exactly two retained, to RowMaterial |
| old two RoleSlot/one Coordination Arcs | zero constructions; new join owners replace them |
| old coordination return clone/initializers/role-frame-share handles | zero constructions/clones |
| original byte allowances for these omitted containers | retained slack, never subtracted/refunded |

Earlier source/table/ModularCreation/ModularInputs links remain at their original
call sites. They still have actual strong-reference releases; moving a later
variable does not remove them. Four **new** join clones remain separate from the
two retained original state keeper clones.

`tail_slack[5]` holds, in order: old clear-copy guard; old left party lease; old
right party lease; old coordination lease; original prefix parent remainder
containing the unused terminal allowance. First four are existing prefix splits;
the remainder is a moved parent, **not another split**. Allocate original role
broadcasts under the party leases before moving them here. Runtime leases stay
inside original states. RowMaterial retains old keeper/wrappers/scratch/table
residual leases in its existing four-element array. Replace the old zero-parent-
remainder assertion with the exact retained terminal allowance check. No old
terminal sublease is issued and no unused allowance is credited back.

### Concrete join records and conservative coexistence formula

The new records are fixed as follows; names from the old module retain their
actual existing field layouts. All following structs are design pseudotypes.

```
JoinBinding { parent: Binding, f: Binding, y: Binding, w: Binding, c: Binding,
              consumer: [u8;32], operand_use: [u32;2] }
ReshapePermit<C> { keeper: Arc<JoinKeeper<C>>, binding: JoinBinding,
                  role: u8, operand: u16, use_ordinal: u32, consumed: bool }
AffineMask<C> { native: DealerMaskFor<C,Ring>, payload: LeaseFor<C> }
AffineValue<C> { native: CommonMaskedFor<C,Ring>, payload: LeaseFor<C> }
JoinKeeper<C> { prefix: Arc<RowMaterial<C>>, y: AffineMask<C>,
  w: DealerMaskFor<C,Ring>, c: DealerMaskFor<C,Ring>, binding: JoinBinding,
  tail_slack: [LeaseFor<C>;5], residual_L: LeaseFor<C>,
  affine_residual: LeaseFor<C>, metadata: LeaseFor<C> }
JoinFrame<C> = enum { Prefix(FrameOwner<C>), Matrix(matrix::party::FrameFor<C>) }
JoinState<C> { prefix: InitializerState<C>, terminal: arithmetic::PartyFor<C>,
  fhat: Option<arithmetic::MaskedOutputFor<C>>, y: Option<AffineValue<C>>,
  spent_y_payload: Option<LeaseFor<C>>, w: Option<CommonMaskedFor<C,Ring>>,
  matrix: MatrixPartyFor<C>, result: Option<CommonMaskedFor<C,Ring>>,
  permit: Option<ReshapePermit<C>>, frames: [Option<JoinFrame<C>>;19],
  phase: u8, write_index: u8, retired: bool,
  keeper: Arc<JoinKeeper<C>> }
JoinResult<C> { values: [CommonMaskedFor<C,Ring>;2], keeper: Arc<JoinKeeper<C>>,
                control_lease: LeaseFor<C> }
```

JoinBinding.parent is the genuine original affine-product binding at slot17;
f/y/w/c are the four newly issued bindings, not regenerated identities.
Binding embeds source/session/namespace/issuance/descriptor/numeric/consumer
records; authority additionally checks the original keeper Arc and counter
identity, never public metadata alone. AffineMask/Value native fields hold their
L_Y/L_Yrole metadata slices; payload fields hold the three Y288 children.
At matrix begin move native Y into the consuming helper and move its payload
lease into spent_y_payload, retained until state destruction. No split/clone is
needed. Native Y drops on helper return/error before payload cover can retire.
Its metadata lease may release there; the still-live JoinState containing charge
covers the vacated AffineValue storage. Pre-consumption failure drops native
before payload. W uses normal root ownership; original Fhat stays retained.

Control records explicitly price logical return/move overlap:

```
TailMoves<C> { parts: Option<PrefixParts<C>>, returned: Result<PrefixTail<C>,Error>,
  moved_states: [Option<InitializerState<C>>;2],
  dealer: Option<LocalDealerFor<C>>, work: Option<IssuanceWork>,
  f: Option<DealerMaskFor<C,Ring>> }
PendingRole<C> { state: Option<JoinState<C>>, lease: LeaseFor<C> }
NewLeaseBank<C> { aggregate: Option<LeaseFor<C>>,
  partitions: [Option<LeaseFor<C>>;6], metadata: [Option<LeaseFor<C>>;9],
  affine: [Option<LeaseFor<C>>;3] }
IssueMoves<C> { root: Option<RootInputFor<C>>,
  encoded: Option<EncodedRootFor<C>>,
  terminal: Result<(arithmetic::OutputMasksFor<C>,[arithmetic::PartyFor<C>;2]),Error>,
  matrix: Result<(DealerMaskFor<C,Ring>,[MatrixPartyFor<C>;2]),Error>,
  keeper: Option<JoinKeeper<C>>, states: [Option<PendingRole<C>>;2],
  keeper_arc: Option<Arc<JoinKeeper<C>>>,
  slots: [Option<Arc<RoleSlotFor<C,JoinState<C>>>>;2],
  coordination: Option<Arc<CoordinationFor<C,JoinState<C>>>>,
  ticket: Option<matrix::party::IssuanceTicket>, tail: TailMoves<C> }
StepMoves<'a,C> { peer_parts: [&'a [u8];2],
  prefix_frame: Result<Option<FrameOwner<C>>,Error>,
  terminal_step: Result<arithmetic::StepFor<C>,Error>,
  matrix_frame: Result<matrix::party::FrameFor<C>,Error>,
  permit: Option<ReshapePermit<C>>, y: Option<AffineValue<C>>,
  w: Option<CommonMaskedFor<C,Ring>>, result: Option<CommonMaskedFor<C,Ring>> }
JoinControls<'a,C> { clear: Zeroizing<[[i64;4];4]>, issue: IssueMoves<C>,
  transitions: [StepMoves<'a,C>;2],
  driver_coordination: Option<Arc<CoordinationFor<C,JoinState<C>>>>,
  driver_keeper: Option<Arc<JoinKeeper<C>>>, driver_armed: bool,
  lease_bank: NewLeaseBank<C>,
  lease: Option<LeaseFor<C>>, result: Option<JoinResult<C>> }
```

Successful return transfers the EXISTING complete L_control lease with
`controls.lease.take()` into `JoinResult.control_lease`: no split, clone,
replacement reservation or refund. Initialize the driver's lease as Some;
the returned result's lease field is last, after values and keeper. The driver
must not run a separate control-charge refund when its lease is None.
The controls' result slot is last so unwind drops all remaining control-owned
scratch/owners before that slot can release its covering lease. Stage the result
there while completing all scratch wipes and driver/transition retirement;
only then take the result as the final nonfallible return move. On an earlier
error the lease stays in controls, or in its staged last-field result, until
dependent owners are retired. Do not move the staged result out before cleanup.
On final result destruction, values retire before keeper, then control_lease
releases the retained envelope/control charge. No custom Drop may invert this.

L_control uses the updated `size_of::<JoinControls>()`, including its optional
driver lease and the result's added lease field. This is finite containment:
JoinControls contains Option<JoinResult>; JoinResult contains LeaseFor and an Arc
to JoinKeeper, NOT JoinControls. Keep the entire original control allowance
retained in the result; no new child lease or subtraction is needed. Recompute
the checked preflight with actual compiled layouts; this correction asserts no
64 MiB fit. It preserves source-only fixture scope and all split/protocol/RNG
counts; it changes only metadata-cover ownership/lifetime.

These slots are Options/Results, not a requirement to initialize duplicate
secret owners. Charge their storage simultaneously anyway. Original prefix
locals/metadata/return controls and existing native-helper scratch remain
covered by Q43 and native quotes; new tail/move coexistence is additional here.
Use `A(T)=role_coordination::arc_bytes::<T>()+64`, checked, and fix the nine L
children to the following quantities:

```
L_keeper = A(JoinKeeper<C>)
L_role   = A(RoleSlotFor<C,JoinState<C>>)                 // twice
L_coord  = A(CoordinationFor<C,JoinState<C>>)
L_control = S(JoinControls<'a,C>)
          + 2*S(OperationFor<'a,C,JoinState<C>>)
          + 2*S(MutexGuard<'a,JoinState<C>>)
          + 26*C::TRANSIENT_LINK_BYTES
L_F      = typed::mask_charge_for::<C>()
L_Y      = S(AffineMask<C>) + typed::mask_charge_for::<C>()
L_Yrole  = S(AffineValue<C>) + typed::common_charge_for::<C>() // twice
L = L_keeper + 2*L_role + L_coord + L_control + L_F + L_Y + 2*L_Yrole
B = Q43 + A4 + W4 + M4 + 288 + L
```

The existing `OperationFor`/MutexGuard storage is charged twice even though the
driver can borrow serially. Twenty-six transient link slots cover the new parent
and25 successful child leases; existing prefix/native helper quotes cover their
own transfers and error-retention temporaries. L_control is one slice: no extra
guard/controls split. Embedded permits/empty prefix slots/new frame slots are
included in containing types. Conservative duplication of metadata size in Arc
and move-slot charges is intentional; it does not authorize duplicate payloads.
Old Q43 slack is not deducted. These are allocation-free checked layout formulas,
not measured bytes or a proof of physical compiled-stack use. Reject overflow or
B>67108864 before ticket/entropy/factories. The residual byte-fit check is exactly
compiler resolution of these named records and quote APIs, not an unknown owner.

### Tail lifetime and work deltas

#### One-owner lease conservation and slot lifetime clarification

Source check: `typed/role_coordination.rs` defines RoleSlotFor as `state: Mutex<S>`
followed by `_lease: LeaseFor<B>` (18..21), with no custom slot Drop. Therefore
last-Arc destruction drops the complete state before its one covering lease.
CoordinationFor (46..51) owns both slot Arcs before `_original_run` and
`_additional`. Its `operation`/OperationFor borrow the coordinator; `abort_locked`
locks and calls `Retire::retire(&mut S)` in place. Neither extracts S. Poison
recovery also retires in place. During a panic the role guard unwinds before
OperationFor acquires the slot locks for retirement (133..159). None of these
mechanisms clones/detaches a lease or automatically calls Retire on final Arc
drop; ordinary state-field destruction must still own all cleanup resources.

Assign the ONE original L_role child to `RoleSlotFor._lease`, not to JoinState.
Before publication it belongs to PendingRole.lease, whose state Option is first
in field order. Perform fallible allocation/admission checks while PendingRole
still owns that lease and its state/initialization controls. After those checks,
publish by a nonfallible move of the state into Mutex::new and the SAME lease
into RoleSlotFor._lease, then Arc::new; no intervening fallible callback or
lease drop. Unwind before publication drops any initialized state before lease;
after publication the slot's field order supplies the same guarantee. A move
between named control slots does not create a second live owner or split.

After publication the fixed driver only borrows `&mut JoinState` through its
slot's MutexGuard. It must not use Arc::try_unwrap, Mutex::into_inner, replace the
whole state, or return the whole state from `operation`. Only named Option-owned
operands/frames/results/permits move into the charged StepMoves controls. Keep
the coordinator/slot borrows and guards live for each transition; peer slices
cannot escape their guards. Any final value escaping slot destruction owns its
existing native/typed lease, while its result envelope owns transferred L_control.
Thus no whole JoinState outlives its slot after publication. If implementation
needs such an escape, it must stop: that is not this proposed ownership schedule.
Construction's temporary whole-state moves are explicitly covered by PendingRole
and the quoted controls, rather than justified by an unsupported RAII assertion.

All nine L children and original parents have exactly these destinations:

| Existing charge | Sole destination / transfer |
| --- | --- |
| L_keeper | JoinKeeper.metadata, LAST field after all payload/slack fields |
| L_role0, L_role1 | PendingRole.lease -> respective RoleSlotFor._lease; no JoinState.metadata |
| L_coord | CoordinationFor._additional, after its slot Arcs |
| L_control | controls.lease -> staged/final JoinResult.control_lease, per K145; no simultaneous second owner |
| L_F | RowMaterial.terminal_mask._lease (genuine original native F wrapper) |
| L_Y | AffineMask.native._lease, not an extra lease on AffineMask |
| L_Y0, L_Y1 | corresponding AffineValue.native._lease; helper consumes native value once |
| three Y288 payload children | AffineMask.payload and two AffineValue.payload; role payloads move to spent_y_payload on consumption |
| Y288 parent remainder | JoinKeeper.affine_residual, even when zero; NOT a new split |
| L parent remainder | JoinKeeper.residual_L, even when zero |
| aggregate parent remainder | CoordinationFor._original_run=Some(original aggregate lease), even when zero; NOT a second reservation |
| prefix parent/old clear/party/coord allowances | the five fixed tail_slack entries, by move |
| original four RowMaterial/runtime/native/root/matrix allowances | original owning fields/helpers unchanged; never supply a second new-wrapper lease |

NewLeaseBank is only inline construction storage for these existing parents and
children: all its Options start empty, receive each actual reservation/split once,
and are emptied by take into the destinations above. Its arrays name the six
partitions, nine metadata children and three affine children, not extra splits.
JoinControls drops dependency-owning issue/transition/driver fields before the
bank, then its optional control lease, then its last-field staged result.
For pre-transfer failures the bank therefore retains unassigned cover through
dependent cleanup; after transfer its corresponding Option is None. Never refill
an empty Option with a fresh reservation to disguise a missing owner.

Move both newly constructed slot Arcs into CoordinationFor.slots; the driver
borrows them through coordination rather than retaining cloned slot Arcs. Set
_original_run from the original aggregate parent and _additional from L_coord.
Consequently this resolves the coordinator's TWO lease fields using two already
existing distinct owners; it does not demand another metadata child. Keeper
metadata is last to avoid releasing its containing charge before residual-lease
fields. Its added affine_residual names the previously implicit zero-byte Y288
parent rather than creating a new allocation/lease.

K145 remains mandatory: finish all scratch wipes and driver retirement with the
result still staged last, then perform the final result move. On error/unwind,
cover stays in controls or the staged result; on success the driver has no control
lease to refund. Final result drop is values, keeper, control lease. Native
internal cancellation bounds and e<=1 are unchanged; do not add unconditional
extra native cancel calls merely to destroy the enclosing slot.

Re-evaluate the SAME checked L/B formula with the updated JoinState (no metadata
field), RoleSlotFor (one actual lease), PendingRole, NewLeaseBank, JoinKeeper's
affine residual, and JoinControls/JoinResult fields. This charges prepublication
and transfer storage as well as the live slot. No compiler-resolved size or64 MiB
fit is asserted here. Counts remain25 new splits,138 native successful splits,
at most four recorded matrix error-retention splits; source-only scope and all
native arithmetic/material/entropy/security premises remain unchanged.

JoinControls owns the **only** four-root clear copy. Extracted prefix borrows
its first three rows instead of running its old three-root `Zeroizing::new`
copy. Retain old guard as slack. Domain/encode scans still run; existing mask
copy/mu scratch remains at its original sites. Explicit clear wipe and final
destruction remain256 bytes, with no second96-byte secret clear copy.

Use one original `IssuanceWork` instance, initialized for the admitted2120-byte
schedule: account the namespace once, lend one fill closure across all factories,
test remaining==0 only after matrix issuance, then move it into RowMaterial.
Do not create a second prefix allowance or apply the old full-D43 entropy equality.
Original states' immutable Work describes only the retained prefix; full checked
work is the join's admission vector, not another allowance. Native K inventory
and previously recorded cleanup counts are unchanged.

After publication RowMaterial has three tail-attributable strong references:
JoinKeeper1 plus original states2. JoinKeeper has five: driver1, new states2,
permits2. Consuming permits drops their two links; result creation moves the
driver link into JoinResult without cloning. State destruction releases both
its join link and its original RowMaterial link. Result values drop before the
result keeper link, leaving all dealer/masks/slack covered until final disposal.
Internal native/source links retained by original frames/inputs remain genuine
owners and release at their actual drops, not at numeric phase completion.

Early failure drops only initialized Parts/Options. Secret/native fields precede
their covering leases; tail slack drops after dependent parts. Before publication
there are no role/permit backlinks; each published state already owns its keeper
link. One-shot retirement cancels original prefix parties, new terminal and
matrix once each, clears values/frames once, then retains keeper/slack until final
drop. Original prefix terminal=None prevents duplicate new-terminal cancellation.
Keep e<=1 and the existing native internal/error-retention bounds. Empty/zero-byte
leases still have actual release events. Field move is not an early refund.

This adds only metadata/empty inline storage, not another clear root, mask, key,
correlation, Fhat or frame payload. The old dealer/masks remain in a shared local
keeper: this is logical custody, not evaluator/dealer process isolation.
The25 new successful splits/four bindings/one namespace/four new Arc allocations/
four new keeper clones/19 steps/17 groups/19 parts are therefore preserved.
Original tail retains exactly one Arc allocation/two state keeper clones;
removed old slots/coordination/handles and one source clone are not executed.
Their original byte allowances remain reserved. The constructor-tail decision
is now closed parametrically; explicit implementation approval and numerical
layout/admission verification remain separate gates.

### Conditional privacy extension; joint-transcript gates remain held

For the additive-share bridge, `R=Z/(2^50)`, `rho,s in R^4`, scalar output
mask `t in R`. Source `matrix/core.rs::{deal,evaluate}` gives
`A0+A1=-rho`, `B0+B1=-s`, `C0+C1=rho*s+t`. With `H=Y+rho`, `K=W+s`,
`qi=Ai*K+H*Bi+Ci+[i=1]*H*K`, so `q0+q1=Y*W+t` (row/column contractions).

Only given a simulatable fixed prior D43 view `Vi` and its own legitimate `Yi`,
the extension simulator samples independent uniform `ri,H,K,Ai,Bi,Ci,O` with
dimensions `4,4,4,4,4,1,1`, sets `mi=Yi+ri`, `mpeer=H-mi`, computes `qi`
above and sets `qpeer=O-qi`. Keep actual frame binding/digest, role, phase,
count and canonical fields. `Vi` includes historically observed terminal `hi`,
keys/material/randomness/frames/failures; dropping backing does not erase history.
This does not simulate D43 or its secret-dependent abort kernel.

Required hypothesis: fresh `rho,a,s,t` and all nine matrix split pads are mutually
independent uniform AND independent of the joint prior execution/private operands.
Existing weight-mask history requires a separate joint proof, not uniform
marginals. Input-independent preprocessing and correct splits alone fail:
`A0=rho`, `A1=-2*rho` are valid, with uniform individual marginals of `rho` and P0's `A0`, but reveal `Y=H-A0`.
The `fill` callback guarantees neither independent entropy nor independent
captures. No conditional extension applies without its joint hypotheses.

For the old reconstructed-terminal design, `Yhat=Fhat+betahat` and
`rY=rF+rBeta` are deterministic identities, not a fresh-independent-rho proof.
Historical O129/local-driver arguments do not prove safe reconstruction of Fhat
or qualify the full prefix/weight/key/carry/share/failure view. Dealer access to
Yhat together with rY breaks privacy. Shared keeper access is not process
isolation; hashes/driver checks are not authenticated transport or active security.
COMMON primitive, composed-transcript, implementation, complete funding and
independent composed-security gates remain held. No D16 adoption, security
qualification, SIGMA activation or D39/D43 change is authorized.

### Astra49 conditional process-boundary counterexample (Astra23 source-only review complete)

**Independent Astra23 review confirms only this conditional, source-level
counterexample.** This is source-clear for that scope only, not deployed-attack
confirmation, role-isolation approval, security change or runtime qualification;
runtime/API remain held and the independent joint proof gate remains open.
This records a conditional obstruction to reusing current LOCAL clear/dealer
harness role objects as production evaluator-process memory. It is not a
deployed attack or a violation of the existing clear-reference contract. The
actual `LocalClearRoot` dealer sees all clear roots; that trusted dealer must
not be reinterpreted as an honest evaluator with private operand ingress.

The current constructor accepts all three clear roots. Roles 0/1 are share
indices, not an X-versus-gamma/beta owner assignment; see root issuance in
[`parent/issuer.rs:745–768`](../../crates/pllm-garble/src/masked_wire_reference/typed/layernorm4/parent/issuer.rs#L745-L768).
Any future experiment must explicitly stipulate a different private-ingress
role model: the corrupted evaluator knows gamma=beta=0, an honest owner supplies
X through trusted private encoding, and the ideal functionality hides X and S
from that evaluator. Its view includes reachable process memory, not merely
public Rust methods or selected wire messages. If the corrupted principal
includes the CURRENT clear-root caller/dealer, its simulator legitimately knows
X and the counterexample premise fails. The existing logical-custody
clear-reference contract is not violated.

Under that conditional model, the current private-field graph is not an
isolation boundary. A Role owns an Arc to Coordination, which owns BOTH slots,
although its own slot suffices for the projection below; see
[`parent/runtime.rs:5–8`](../../crates/pllm-garble/src/masked_wire_reference/typed/layernorm4/parent/runtime.rs#L5-L8)
and [`role_coordination.rs:46–50`](../../crates/pllm-garble/src/masked_wire_reference/typed/role_coordination.rs#L46-L50).

* `role -> coordination.slots[i].state` retains both `values[0]` and
  `keeper.masks[0]`; their backing values subtract to X modulo `2^50`,
  immediately after issuance, BEFORE the first online contribution.
  Root encoding establishes the masked-value identity in
  [`typed.rs:618–626`](../../crates/pllm-garble/src/masked_wire_reference/typed.rs#L618-L626);
  the retained mask array and state keeper link are in
  [`parent/issuer.rs:150–159`](../../crates/pllm-garble/src/masked_wire_reference/typed/layernorm4/parent/issuer.rs#L150-L159)
  and [`parent/issuer.rs:991–994`](../../crates/pllm-garble/src/masked_wire_reference/typed/layernorm4/parent/issuer.rs#L991-L994).
* Removing only the outer keeper still leaves a retained `ModularFrame` with
  `inputs.energy.backing` and `creation.energy.backing`; their difference
  modulo `2^50` recovers S. This frame projection exists ONLY after interval
  evaluation and modular-product begin, not before online execution. See
  [`parent/runtime.rs:278–285`](../../crates/pllm-garble/src/masked_wire_reference/typed/layernorm4/parent/runtime.rs#L278-L285)
  and frame retention at
  [`parent/runtime.rs:441–447`](../../crates/pllm-garble/src/masked_wire_reference/typed/layernorm4/parent/runtime.rs#L441-L447).
  The transitive owners are explicit in
  [`parent/coordinate.rs:54–93`](../../crates/pllm-garble/src/masked_wire_reference/typed/layernorm4/parent/coordinate.rs#L54-L93),
  and input/frame retention in
  [`parent/coordinate.rs:367–382`](../../crates/pllm-garble/src/masked_wire_reference/typed/layernorm4/parent/coordinate.rs#L367-L382).

These are private Rust fields, not public getters; field privacy and mutexes
do not provide process privacy. The counterexample concerns
the proposed evaluator-process memory boundary, not an assertion that an
external API caller can read those fields.

For that explicitly stipulated experiment, compare X=(0,0,0,0) with
X=(1,-1,0,0) as INTEGER Q12 CODES, not real-valued units. X/gamma/beta domains
are [-4096,4095]. With mu=floor(sum(X)/4), both means are zero and
S=1+sum((X[i]-mu)^2) is respectively 1 and 3: the energy constant is ONE Q24
code, NOT `2^24`. See local arithmetic in
[`parent.rs:42–75`](../../crates/pllm-garble/src/masked_wire_reference/typed/layernorm4/parent.rs#L42-L75),
mean shift in
[`parent/issuer.rs:32–37`](../../crates/pllm-garble/src/masked_wire_reference/typed/layernorm4/parent/issuer.rs#L32-L37)
and mean issuance in
[`parent/issuer.rs:815–820`](../../crates/pllm-garble/src/masked_wire_reference/typed/layernorm4/parent/issuer.rs#L815-L820).

With gamma=beta=0, true LayerNorm output is zero in both cases. Conditional on
fresh INDEPENDENT uniform fill, terminal h0/h1 give uniform individual output
shares and the same joint output distribution (U,-U) in both cases, not
identical realized shares across runs. The mere `FnMut` fill signature does not
guarantee this independence or uniformity. See terminal splitting and subtraction
in [`gapars_terminal.rs:229–238`](../../crates/pllm-garble/src/masked_wire_reference/typed/gapars_terminal.rs#L229-L238)
and [`gapars_terminal.rs:298–306`](../../crates/pllm-garble/src/masked_wire_reference/typed/gapars_terminal.rs#L298-L306),
with fill/draw/split in
[`arithmetic.rs:192–205`](../../crates/pllm-garble/src/masked_wire_reference/arithmetic.rs#L192-L205).
Thus legitimate evaluator input/output marginals coincide while the
deterministic memory projection recovering X (or just S) distinguishes the two
cases. Extra fields in the complete view cannot repair that projection. This
conditional memory-view obstruction neither proves the ideal restricted
transcript insecure nor establishes its simulatability.

Required boundary remains **pre-issuance isolation of all transitive secrets,
private operand ingress, and an observable-abort contract before the first
private contribution**. Endpoint cleanup cannot erase secrets already observed.
The prior conditional share-to-matrix extension lemma remains conditional on
its stated joint independence and simulatable-prior-view hypotheses; this
counterexample does not discharge or replace the future independent joint proof
gate, including failures. No new PRF, entropy or security assumption, weakened
revealed functionality, implementation approval, or role/API/Rust change is
introduced. Astra23's completed source-only review does not discharge the future
independent composed-security review; runtime/API remain held.
