# Raw CHECK prepayment contract

## Status and scope

The CHECK prepaid-grant and Early SIGMA-cover directions are approved for detailed
funding review only. Implementation, Rust types/visibility, APIs, security and
activation remain unauthorized. This is a reusable proposed contract, not a receipt.
It establishes no runtime proof, numerical correctness or whole-Q funding closure.

Ordinary `RawRegistration`, `RawHost`, `RawReader`, `Publisher`, native quotes,
layouts and modes remain unchanged, including ordinary `RawFirstNorm` execution.
Ordinary `NativePublisher.take` continues to deny RAW consumption
(`crates/pllm-python/src/replicated_transport.rs:68–92`). No registration Option
extension is proposed: that direction is superseded. No free padding, old-field
credit, replacement caller ledger, additional global cap or guard lowering applies.

## Private ownership and construction

The proposed opt-in native class has `shell: Publisher`, `custody: CheckCustody`,
and `pid: u32`, under the private name `NativeCheckPublisher`.
`CheckCustody` retains `reader: Option<RawReader>`, `host: RawHost`,
`grant: CommittedPrepayment` and `phase: CheckPhase`. Owning drop order is reader,
host, then grant; phase is control-only and must precede the grant in an eventual
physical declaration so the grant remains the last actual field. Layout is OPEN.
`CommittedPrepayment` privately controls an `OwnedExternalCharge` committed from
an authentic original `PrimitiveAttempt`; an existing original-root clone is allowed.
There is no Clone, adoption, detach or public constructor for this proposed grant.

The actual host moves privately from genuine `Publisher.raw_host` only in the
opt-in constructor mode, leaving the shell without a second host owner.
`crates/pllm-compiler/src/replicated_decoder/source.rs:86–112` supplies the existing
field ownership; it does not implement that transfer. `RawHost` and `RawReader`
are restricted to `replicated_decoder` (`verified_source/raw_owner.rs:419–432`,
under the same compiler source directory). A cross-crate class cannot directly
use these fields today; private placement/bridge and full layouts remain OPEN,
not implicit permission to expose types or invent a new public constructor.

Quote increments and upstream bootstrap must be admitted before the first new
work, while the header is mutable, before `Arc::new` freezes it
(`raw_owner.rs:265–280`). Constructor allocation and fallible publication
preparation finish before grant commit; postcommit installation/publication must
be infallible. This requirement is not established by the existing PyO3 wrapper.
The original attempt's commit requires that preparation first
(`original_source.rs:194–223`); no spare quote credit or self-funded quote work applies.

## One-shot claim and CHECK

The proposed phase path is Pending → Claiming → Claimed → Running → Checked.
Any claim/CHECK error or unwind makes it Spent; retirement reaches Released only
after actual lease exit. Spent never becomes active, and Checked cannot CHECK again.
Consumption means phase eligibility is consumed before internal work, not proof
that a by-value owner can be recovered. Private `&mut self` claim/run operations
require the actual persistently retained Python adopter's `_busy` gate.
Host and grant must never leave persistent custody through fallible code.
In-place borrowed internals are a proposed alternative, not implemented behavior.
If a by-value core owner is used, a recovery guard must restore Spent custody
before error/unwind; do not `Option.take` the owning bundle and then run fallible code.
Its proposed results are `Result<CheckOwner, (Error, SpentCheckHost)>` and
`Result<CheckedOwner, (Error, SpentCheckHost)>`, never an active failure owner.
An external adopter label or `BudgetGuard` does not authorize this transition.

Unsealed `RawHost.claim_reader` refusal does not set `reader_claimed` or
`reader_failed` (`raw_owner.rs:1014–1019`); grant one-shot phase is separate.
Weight lookup clears its failure latch on success (`raw_owner.rs:688–708`), so
CHECK failure before or after that clear must independently forbid retry.
Borrowed windows, tensors and guards drop before the reader. Reader Drop clears
only `reader_live`, never the failure latch or host charges (`712–718`).
Mark Spent and restore persistent native/adopter custody before constructing a
Python exception. Exception publication itself needs funded failure overlap.

## Original debit and retained cleanup

Abandoning an admitted, uncommitted attempt burns the original root; committing
does not clear a sibling burn (`original_source.rs:194–223`). Counter/capacity or
overflow refusal before attempt entry does not burn; admitted debit overflow or
exhaustion does (`623–635,692–720`). Dropping an unused committed grant does not
burn or refund WORK. Its resident charge releases only after successful retirement
and actual lease exit, not merely when numerical CHECK finishes.

A private Python adopter reuses genuine `_RawPublisherOwner` cleanup-slot, actual
CM/lease and gate semantics, while validating the exact opt-in native type.
Ordinary accepted native class remains unchanged
(`python/pllm/runtime/replicated_source.py:98–119,140–159`); no label replaces custody.
Reader/window custody retires first, then `begin_retire` → Ready → `retire` →
actual CM exit → `release_final_charges`. Native hooks currently borrow the host
(`crates/pllm-python/src/replicated_transport.rs:180–214`); host preparation and
retirement rules are in `raw_owner.rs:881–892,1038–1088`.
Cleanup failure retains the lease, native owner, charges and sticky cleanup failure
for retirement-only handling, never numerical retry or automatic cleanup retry.
Mutation-free Busy may be retried only after readers/operations stop; ambiguous
CM exit forbids retry/release (`replicated_source.py:189–235`).

## Approved construction-accounting review direction

Detailed funding review may expand the opt-in first-original-source CONSTRUCTION
boundary: an internally fixed positive entry allowance is checked against the
UNCHANGED cap before covered work. Authentic pre-root SPENT work must enter the
FIRST original ledger alongside existing JSON/config import spent, preserving
every slice. Root creation, initial debit and refusal cleanup need an explicit
construction convention, not a claimed ledger predebit before the ledger exists.
Once the original root exists, future grant work must debit before execution.
`json_value_input.rs:27–46,70–75,124–140` reports JSON quotation/construction/numeric/
census only; `replicated_decoder/source_entry.rs:426–465` copies storage, not caller
work. `original_source.rs:1176–1182,1193–1194` imports only config-owner work via
`JsonPayment::LegacyReported`; `688–707` assumes the entry fee in source-work quotation.
These compiler-source anchors supply no free credit: opt-in provisioning must
explicitly extend the original caller contract. `Resources.prepaid_work` is already
debited work, not remaining allowance (`original_source.rs:1152–1159`). Consuming
config/controller handoff, local spent accounting and refusal cleanup remain UNPRICED; no allowance
is funded now and no W/D quote, cap raise, Rust or activation is approved. Ordinary
paths remain unchanged. Post-Spent API refusal needs separate caller-entry funding:
one grant cannot fund unbounded refusals.
The additional private entry is PROPOSED, not implemented: genuine `OwnedCharged` config and working controller are consumed once per entry, not minted through a new public borrowed-bytes API.
Existing `replicated_decoder/source_entry_ingress.rs:190–229` imports public config before the first root (`original_source.rs:1165–1182`): a separately funded import/first root is a new concrete owner, not an old-ledger reset merely because public JSON is equal.
No extra persistent provisioning phase owner or global retry restriction is implied solely by equivalent metadata reimports; the ordinary immutable factory may remain reusable. Existing grants/roots must never renew/reset; post-Spent refusals and independent new admissions need separate funding, with all post-host custody/cleanup and claim/CHECK one-shot guards unchanged.

## Proposed local entry schedule (unpriced)

Independent conditional source clearance covers this schedule only, not numeric funding or API authoring approval.
The PROPOSED private entry consumes genuine `OwnedCharged<ImportedValue>`, `OwnedExternalCharge`, `DecoderWorkload` and `Limits`; it first checks `check_counter` identity.
Convert `resources().work.total` from `u128` to `u64` with checked conversion;
the successful first seed `initial` is checked `reported JSON + ENTRY + CLEANUP`, once
within unchanged `limits.max_work`. Internally derived `ENTRY > 0` and the separate
cached-count CLEANUP direction are approved for detailed pricing/design review
only; both remain UNPRICED, not funded or authorized for implementation.
Cached source counts/resource scalars and bootstrap/control helpers do not traverse JSON (`original_source.rs:1089–1106`); ENTRY must cover all repeated callee invocations, control and admission work, not merely top-level calls.
Construct `Ledger { used: initial, limit: cap, burned: false }` and the root once, with `cap = limits.max_work`; this is the construction convention, not a predebit into a nonexistent ledger.
`enter_attempt` with `AttemptDebit::ExistingSource` adds zero and preserves the seed (`original_source.rs:688–720`).
Hand off to `quote_on_root` with existing `JsonPayment::Prepaid`, avoiding double `LegacyReported` import debit; bootstrap quotation and remaining prepared-work fees still debit normally (`479,1193–1206`).
This private entry is not implemented; ordinary `quote` remains untouched and current ingress still calls it (`replicated_decoder/source_entry_ingress.rs:211–217`).
`new_context` calls `try_build`, reserving inline `OwnedCharged<Context>` storage before Arc allocation; it can fail even after bootstrap capacity checks (`original_source.rs:320–329`; `crates/pllm-garble/src/shared_allocation.rs:524–540`). Pricing and refusal cleanup must cover that failure.
Refusals before attempt entry have no persistent burn; the caller-funded construction/refusal convention is required and is not established by the seed alone.
Caller-side construction funding must cover pre-root refusals BEFORE ownership
handoff; a successful seed does not pay for rejected construction. Proposed CLEANUP
uses authentic cached `spans`, `array_children`, `map_entries` and `objects`
(`plan_input.rs:165–168`); coefficients, caller slots and callee paths remain UNPRICED.
Unused prepaid cleanup is a NONREFUNDABLE spent-debit retirement obligation, not
respendable credit. Retain payload controllers until actual retirement and do not
debit that same obligation again later. Existing import terminal `Value` cleanup
is not established paid: `json_value_input.rs:269–282` prices only quotation,
construction, numeric and census; span retirement is not `Value` retirement.
Variable JSON/source destruction has an unadopted conditional terminal candidate below; complete cleanup adoption and funding remain open;
no numeric funding, authoring, activation or change to ordinary `quote` follows.
Existing post-attempt failure burn, retained retirement custody and all ownership corrections above remain unchanged.
No actual W/D, funded increment or full-Q clearance follows; implementation and execution gates remain held.

### Reviewed symbolic ENTRY inventory (unpriced)

Astra reviewer 17 (session e7eb87) completed the literal ENTRY ledger, not numeric approval.
Here `OS = crates/pllm-compiler/src/original_source.rs`, `SA = crates/pllm-garble/src/shared_allocation.rs`, `E = ENTRY`, and `C = CLEANUP`.
The frozen proposed schedule is: `check_counter` once (OS:1070–1079); cached resource/count reads and checked `u128 -> u64` reported-work and N/A/M/O conversions;
evaluate `C = k0 + kv*N + ka*A + km*M + kc*O` with four checked multiplications and four checked additions;
then checked `reported + E + C` (two further additions), with the unchanged cap comparison.
Call `bootstrap_bytes` once (OS:1089–1106); repeated cached reads/helpers are additional work.
Split root, use `new_context` to construct `Ledger { used: initial, .. }`, split attempt, and enter `ExistingSource` with its checked zero-add;
then invoke `quote_on_root(Prepaid)` with one caller Call. All caller CONSTRUCTION is prefunded BEFORE handoff; rejected entries cannot use a successful seed.

Through first attempt entry, excluding work inside `quote_on_root`, reviewed helper multiplicities are:

| Helper/event | Multiplicity |
| --- | ---: |
| `context_control_bytes`, `primitive_attempt_control_bytes`, `control_bytes`, `arc_header_bytes` | 4, 4, 1, 8 |
| `split_owned` (SA:490–501), `new_context` (OS:320–329), `try_build` (SA:524–540) | 2, 1, 1 |
| `resize_payload`, conditional grow/reserve (SA:475–485,101–114) | 1, conditional |
| Builder `OnceLock::new`, `Mutex::new`, `Arc::new` | 1 each |
| Budget-link Arc clones from splits; attempt root-link clone | 2; 1 |

Context layout (OS:290–318) has 22 slot expressions and a bounded fold; attempt BASE (OS:570–616) is a compile-time constant, not a runtime inventory walk.
Caller Call and expanded nested helpers remain additive under the existing proposed grammar; callbacks cannot disappear inside opaque helpers.
Keep positive named primitive components `Rreserve`, `Anew`, `Aclone`, `Llock`, `Gguard`, `Oonce`, `Mmutex`, `Llayout`; the destructor table does not price them at one. Rreserve, Oonce and Mmutex have conditional UNADOPTED component candidates below; neither those candidates nor the other component reviews adopt fees or fund E.
CRITICAL: `reserve_bytes` uses a `fetch_update` closure/retries plus `fetch_max` (SA:101–114). One atomic event supplies no finite expanded retry count.
Finite E requires an explicit whole-invocation logical primitive contract or bounded semantics; neither is approved here. Lock/poison/guard paths need separate fees.
Allocation OOM abort is not a recoverable-Err or unwind-cleanup guarantee.

`E = max_branch[expanded fixed entry events + named primitive fees + additional control-owner retirement] + Qprefix`.
`Qprefix` covers uncovered fixed controls THROUGH the first quote debit (OS:1189–1204), including the debit helper itself and second `check_counter`.
Do not assume all of `quote_on_root` is paid: exclude ONLY demonstrably debit/prepaid-covered quotation bodies, without duplicating caller Call or import fees.
Counter/conversion/cap/bootstrap refusals retire the imported owner under C plus remaining working control; a successful root-child split adds charge retirement.
Growth failure retires that child with its old reservation; builder, Context and Arc do not yet exist.
After root creation, second-split/attempt refusal additionally retires root Arc, `OwnedCharged<Context>`, Context, empty OnceLock and ledger Mutex.
Where exactly applicable, additional charge retirement can reuse the conditional BODY 162 plus caller/glue events, but not the two imported charges already in C.
Root Arc retirement needs a Context callback, not a Budget callback. Attempt-abandon burn (OS:212–226) is separate.
Controllers persist through actual payload/import retirement; C remains spent, nonrefundable and never re-debited. No extra owner or global retry restriction follows.
Detailed review only: numeric ENTRY, C coefficients/adoption, full funding, Rust, activation and native execution remain held.
Ordinary `quote` is unchanged; a newly funded owner of identical JSON does not reset an old ledger.

Further conditional review records a reservation candidate and remaining UNPRICED boundaries, not logical-proposal adoption or a full E:
1. Treating whole `fetch_update` as `u=1` is a NEW UNADOPTED logical proposal, not an inherited atomic fee.
   Its SA:104–106 checked-add/filter callback is normalized ONCE, including capture and filter-closure construction even on None; retries are excluded, not bounded.
   The independently reviewed UNADOPTED `Rreserve = max(76,59,68) = 76` external, or 75 if ENTRY already includes its caller Call, is now a HISTORICAL conditional candidate under earlier helper fees, NOT K-combine-ready.
   Historical branch ledger (success / overflow / cap refusal): base 12 each; callback 36/27/36; map_err 4/7/7; Try 5/12/12; success tail 18/0/0; Return 1 each → 76/59/68.
   These are SPECIFIC NEW abstract pricing choices pending normalization/adoption, not Rust ABI or inherited cleanup fees:
   `Kouter=4` is two explicit borrow/place operations plus two field captures; `Kinner=2` is captured-limit reference read plus transfer, even on None. Scalar reads stay positive.
   Ordinary if-test + branch costs two. Enum dispatch and enum-tag construction each cost one; construction includes Ordering arguments. No silent change to the terminal C grammar follows.
   Whole `fetch_update` BODY is 3 (u=1 + result tag 1 + Return 1), C=4; callback is normalized ONCE at terminal evaluation, with retries EXCLUDED, not bounded.
    Historical selected `fetch_max` intrinsic BODY=2/C=3; wrappers and ordering dispatch gave fetch_max 13, superseded for current K by the reconciliation below.
   Pinned checked_add directly uses `add_with_overflow`, then `unlikely` + `unchecked_add` INTRINSIC, not an overflowing_add wrapper: prior BODY 15 is WITHDRAWN under the clarified terminal rule; reviewed BODY 11 on both branches is conditional on that rule, not aggregate Rreserve repricing. Earlier filter Some=12/None=3 and predicate=5 remain dependent fees requiring reconciliation.
   Rust 1.85.1 anchors: [atomic.rs:2901](https://github.com/rust-lang/rust/blob/1.85.1/library/core/src/sync/atomic.rs#L2901), [uint_macros.rs:486](https://github.com/rust-lang/rust/blob/1.85.1/library/core/src/num/uint_macros.rs#L486), [option.rs:1483](https://github.com/rust-lang/rust/blob/1.85.1/library/core/src/option.rs#L1483).
   Apply Rreserve once on the growth path only; resize, caller propagation, retirement and full E are excluded. Closure-capture/tag/intrinsic rules remain unadopted; Qprefix, Arc, layout and lock funding remain OPEN.
2. Positive Qprefix covers the entire bootstrap closure (OS:1189–1204; `crates/pllm-models/src/source_work.rs:181–265`), not demonstrably prepaid by future `quote_work`.
   The caller `quote_on_root` Call is already in ENTRY and must not be counted again.
   Include `restrict_limit`, the second ledger read and first debit: success has three locks and three guard drops;
   late failure may add attempt-abandon's fourth lock/guard, with poison retirement separate.
   The old 527/seven-usize allowance is not matched for reuse. Literal resource closure has two `Traversal` resources-for-native calls;
   complete expansion/reconciliation remains open, not a dead fee. The earlier Fprefix 4848 candidate is WITHDRAWN as unproved; array control is only a partial component, not a complete prefix.
   Array inventory is 22 + 44 = 66 entries, with up to 68 actual next calls; no new numeric prefix subtotal is frozen here.
   OS:1055 uses ImportedValue `resources().work.census`, a u128 (`json_value_input.rs:33`); `input_limits` performs bounded u128 → usize conversion with positive conversion/refusal work, distinct from unbounded 64-bit usize ↔ u64 helpers. No numeric bounded-conversion total is cleared here.
   Previous bootstrap 3276/3290 is WITHDRAWN for omitted field-place work. NEW review-only candidate, external Call INCLUDED: `Bsuccess = 3414 + Pnative + Ppost + 32Pu + 16Pa + 7Pm + 4Pnumber`; conservative all-branch ceiling is `3428 +` the same BODY maxima.
   With Pu=18, Pa=18, Pm=23 success/24 maximum and Pnumber=9, independent row audit conditionally confirms `4475 + Pnative + Ppost` on success or `4496 + Pnative_max + Ppost_max` across branches; +14 covers Try-failure switches. These remain unfunded review candidates, not adopted tariffs or physical CPU bounds.
   Pnative/Ppost are resources-for-native/census BODYs with conditionally calibrated layout parameters below; post includes native once, so whole B includes TWO native invocations. Pu is direct byte-sum callback BODY (not function u); Pa/Pm/Pnumber are source_work add/mul/number BODYs.
   Independently audited resource candidates, agreed by their author: native/post successful complete C are `1971 + Ls` / `4023 + Ls`; thus `Pnative = 1970 + Ls`, `Ppost = 4022 + Ls`.
   All-path BODY bounds are `Pnative_max = max(1974 + Ls, 38 + Lf)` and `Ppost_max = max(4025 + Ls, 60 + Lf)`. Ls/Lf are complete C for Layout::array::<NativeFrame> success/refusal, not BODY; their conditional conservative calibration follows.
   Resource closure (`crates/pllm-types/src/bounded_canonical/borrowed.rs:645–738`; `pllm-models/src/lib.rs:951–992`) pays actual mutable assignments/destinations without double-counting SSA; these candidates establish no full-prefix or ENTRY funding.
   Six folds perform 44 callbacks and 50 next calls, with no constant iterator normalization. Candidates assume pinned Rust 1.85.1 helper rules, instrumentation OFF, not debug fees, CPU work or runtime bounds.
   Old 6455 partial array census is INSIDE Ccontrol, not a separate prefix addend. No numeric E/full-prefix claim or bootstrap self-funding follows; these are not cleanup C fees, and successful quote_work cannot retrofund the prefix.
   Composition review counts five K/U/I/T/V BODY Calls once at the caller, complete bootstrap B C once, normal/poison retirement as alternatives, guard Calls in methods (not inside g), and caller retirement as complete C.
   Independent control review conditionally checks `Ccontrol_success = 6109 + 2H`, `Ccontrol_max = 6634 + 2Hbar` under q=0, X=14, S=8, D=51; H/Hbar are success/all-branch complete C for the SAME instantiated `arc_header_bytes::<OwnedCharged<Context>>`, invoked exactly twice inside context/primitive helpers, conditionally calibrated below.
   Source-array literals/folds/retirement are checked: explicit no-drop BODY 2 gives Cdrop=3 under the documented convention, not recursive Rust placeholders or real MIR; as_mut_slice/manual Drop still runs after exhaustion, with no scalar-usize child N walk. The +525 failure allowance covers nine source-Try and 66 fold premiums, enclosing Returns ONCE. Component calculation is conditionally closed as scoped below, but aggregate integration/funding, complete E/Qprefix, source-proof and metric/adoption clearance remain OPEN.
   Independent Rust 1.85.1 Layout calibration: actual invoked array subtree C=31 success/21 refusal (zero success 20); T::LAYOUT associated-const initialization has runtime cost zero, consistent with BASE. Conservative Ls=50/Lf=40 (zero success 39) adds explicit positive SLACK 19, not a const-initializer fee or metric change.
   Complete Layout::new C=19, extend C=50 success/46 refusal and pad C=26 are independently reviewed; anchors: `core/alloc/layout.rs:148–175,334–343,431–445,494–521`, OS:943–961.
   H=Hbar=193 conditionally composes; Hproject fixed success 79/extend-refusal 22 includes three/one receiver-formation PREMIUMS, consistent with native (autoderef is not autoref), not a newly adopted universal implicit-formation rule.
   Conditional native/post complete C=2021/4073, success BODY=2020/4072 and all-path BODY=2024/4075 yield bootstrap Bsuccess=10567/Bmax=10595 and Ccontrol_success=6495/Ccontrol_max=7020, each B/control envelope including its caller Call once. Post includes native once; B includes TWO total.
   These are unfunded candidate ceilings, not full ENTRY, public tariff or physical bounds; terminal C stays unchanged under ITS own assumptions, with no retroactive cleanup repricing, cap change or future-quote funding.
3. `Anew`, lock/guard, OnceLock/Mutex, backend retirement/Rfailure and other layout/integration fees stay positive named UNPRICED costs under pinned Rust 1.85.1 helpers; `Aclone` has a conditional normal Global-clone C=42 candidate below, not whole-Arc closure. Local Layout calibration does not adopt a blanket Llayout tariff.
   Pinned Darwin Rust 1.85.1's pthread Mutex backend has lazily allocated pinned heap state: a positive whole-lock logical fee cannot establish D funding, and inline Mutex size/Context Arc header do not prove backend-backing coverage.
   Separately inventory/reserve actual backend backing and construction overlap BEFORE first lock; retain coverage through actual backend retirement.
   Allocator, panic, poison and leak branches require checked custody; no physical/RSS bound follows.
   New whole-backend lock/unlock/new/retirement and TLS-address invocation fees are pricing proposals UNDER REVIEW, not adopted fees or full E clearance; numeric reconciliation remains open.
   Independent review confirms a Darwin Rust 1.85.1/libc 0.2.169 64-bit layout candidate: pthread logical pointee 64 B/alignment 8, attribute-construction temporary 16 B/alignment 8; neither compiled-layout nor RSS proof.
   Conditional read-only clearance requires the UNIMPLEMENTED opt-in to preserve fresh-root OS:1176–1182 order, reserve backend + attribute/transient overlap BEFORE first lock, and allow no publication, reentrancy or callback before that lock.
   `split_owned` clones Budget, not Context; the first Context clone is AFTER the lock. Successful private first entry therefore has one initializer/backing (64 B/alignment 8), no CAS contender; earlier refusal has zero.
   This is not a generic concurrent-initialization bound: concurrent initializers may each allocate, so one retained backing alone is not a peak proof.
   A new whole-Rust-backend lock boundary would absorb the initializer callback and is NOT cleared under the existing handwritten-helper/callback expansion grammar: reprice actual Rust backend + initializer before the extern-primitive frontier.
    Context callback must include public Mutex, Option and OnceLock field glue; the earlier empty-Context 98 estimate is withdrawn for outer-glue undercount, and the later Context 104 estimate is likewise NOT certified or reusable.
   Reviewed OS locks use scalar/private lexical guards; consumed poison/`into_inner` moves retire before Context, with no forget or guard-return path found. Existing payload-before-charge order (SA:550–555) can retain coverage without a new owner.
   Clearance excludes generic shared lower layers/returned SourceAdmission; safe `mem::forget` can still leak a generically locked backend. No universal permanent owner is proposed.
   Backend lock/initializer panic before a guard, AttrGuard-destructor panic, double panic and OOM remain uncovered. No pre-reserved extension exists yet, so this is not funded D or a physical/RSS proof.
   Context-root retirement is not Budget BODY 133; do not double-count the two imported charges already in C.
Independent review clarifies the strongest existing wording for NEXT DETAILED REVIEW, not metric adoption: terminal compiler intrinsics cost one event, without invented nested Call/Return;
actual Rust wrappers add Call/Return once, while explicit named atomic/acquire/opaque-release BODY 2 exceptions remain.
`mem::size_of` is a real wrapper (C=3); `mem::transmute` is a re-export, costing one inside its actual caller, consistent with the existing conditional Charge BODY 162 helper boundary.
Affected Arc/Box/array component ledgers still require reconciliation before combining with terminal C; no C coefficient changes or broader fee adoption follow.
Independent canonical Rust 1.85.1 review under that terminal rule gives checked_add C=12/BODY=11 on success/overflow, unchecked_add/sub C=3, checked_sub C=6 success/5 refusal, and checked_mul C=17 success/18 refusal.
Corresponding source_work add/mul/number candidates are C=19, 24/25 and 10; unbounded 64-bit usize ↔ u64 is C=4 with TryFromIntError, not Infallible. Prior checked_add C=16/BODY=15 invented two intrinsic Call/Return pairs (add_with_overflow plus unchecked_add or overflow cold_path).
These are review-only helper calibrations, not coefficient/tariff adoption: all dependent components require rechecking before aggregation; historical Rreserve 76/75 and the bootstrap candidates above are not combine-ready or paid totals.
Official Rust 1.85.1 shared-K calibration is review-only, not physical work or general metric adoption: Option::as_ref Some C=5/None C=4; `match *self` adds no separate copied-read/deref event.
Captureless closure construction costs zero, invoked error mapper C=3, and boolean literals cost zero; runtime expressions, places and writes remain positive.
Detailed-review shared K charges actual plain-record field writes/places/borrows, but no extra plain-record assembly or bare SSA-operand transfer (`q=0`, `A=3`); source_work:259–264 has four field writes + Ok = 5. Explicit capture/whole-array-transfer exceptions stay local, not generalized.
OS:1189 initial Quoting has three writes (A=3); prior caller 242/249 and record-assembly-inflated input 132 are NOT certified. Caller partial candidates are `236 + A` success and `250 + A − 6` terminating maximum, with helpers, reachable maxima and Rfailure extra, not whole-prefix bounds or a cleared replacement input price.
Try 5/12 excludes the enclosing function Return, counted ONCE on exit and already included in full BODY/C envelopes: OS:1204 success is mid-function with no Return, while terminating caller error needs one. Independently verified method candidates 120/133/152 already follow this rule; do not decrement them.
Calibrated helper candidates: NonNull::as_ptr C=3/as_ref C=9, Arc::inner C=12/deref C=16; OwnedCharged::payload C=4, external-charge budget C=20, owned budget C=23; UnsafeCell::get C=5 and guard Deref/DerefMut C=10.
The successful OS:1204 prefix boundary adds zero beyond actual debit propagation, not an extra quote-entry charge. Composition structure is conditionally reviewed, but component source crosschecks remain pending; no aggregate ENTRY, new tariff or full-Q funding follows, and all implementation/execution holds remain.
The conditional UNADOPTED external terminal candidate `C = 462 + 128N + 8A + 452M + 390O` is unchanged under ITS assumed helper rules, not a broader fee approval.
Import + E + cleanup stays once-prefunded before handoff under the unchanged cap; ordinary `quote`, adoption and native-execution holds remain unchanged.

### Current-K reservation reconciliation (unadopted)

Independent review conditionally reconciles SA:101–114 under CURRENT q=0/K;
historical Rreserve 76/75 and its branch table remain historical, not unchanged paid fees.

| External Rreserve branch | Base | Callback C | map_err | Try | Success tail | Enclosing Return | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Success | 12 | 31 | 4 | 5 | 18 | 1 | 71 |
| Overflow | 12 | 22 | 7 | 12 | 0 | 1 | 54 |
| Cap refusal | 12 | 31 | 7 | 12 | 0 | 1 | 63 |

Thus external `RreserveK = max(71,54,63) = 71`, or BODY maximum 70 if ENTRY
already owns caller Call. Callback C=31/22/31 is Call 1 + captured-bytes reads 2
plus checked_add C=12 + Kinner 2 + filter C=13/4/13 + Return 1.
Predicate BODY=5 and filter BODY Some=12/None=3 are reconfirmed. Old callback
36/27/36 falls by five: checked_add C=16 → 12 and removal of a free used-SSA operand fee.
Public fetch_max C=14 is Call/Return 2 + value place 1 + UnsafeCell::get 5
plus atomic_umax C=6, not historical 13; tail stays 18 because one unsupported SSA read is also removed.
Whole fetch_update C=4, callback normalization ONCE even on None, Kouter=4 and
Kinner=2 remain UNADOPTED local choices: callback is not literal whole fetch_update;
actual retry work is EXCLUDED, not bounded. Apply once on growth only; no resize,
caller propagation or retirement is included. Pinned Rust 1.85.1 anchors:
[option.rs:1483](https://github.com/rust-lang/rust/blob/1.85.1/library/core/src/option.rs#L1483),
[atomic.rs:2955,3573](https://github.com/rust-lang/rust/blob/1.85.1/library/core/src/sync/atomic.rs#L2955),
[UnsafeCell:2166](https://github.com/rust-lang/rust/blob/1.85.1/library/core/src/cell.rs#L2166).
This conditional reconciliation adopts no actual Rust primitive frontier, numeric fee,
funding or CPU bound; terminal C=462 stays unchanged under its own assumptions.

### Provisional Box normal-completion component (unadopted)

Astra independently certified this NEW component-only source-event candidate under
pinned Rust 1.85.1, shared q=0/K and instrumentation OFF. It proposes a logical
frontier, not a physical allocator bound: `Aalloc > 0` is the complete returning
`__rust_alloc` invocation, still UNPRICED; `Halloc_error` is the unknown nonreturning
extern invocation. `Aalloc` is not Quoting assembly `A=3` or cleanup array-edge A.
Complete C includes caller Call and, only on returning paths, the callee Return.

| Representative complete C | Normal nonnull | Null allocation |
| --- | ---: | ---: |
| `alloc` | `16 + Aalloc` | `16 + Aalloc` |
| Global `alloc_impl` | `76 + Aalloc` | `64 + Aalloc` |
| `allocate` | `78 + Aalloc` | `66 + Aalloc` |
| `exchange_malloc` | `100 + Aalloc` | Handler boundary: `73 + Aalloc` |

Here `Eexchange_nonzero = 100 + Aalloc`; the zero-size returning branch is
`Eexchange_zero = 61`. Representative thin-pointer helpers have complete C:
const `cast=3`, `addr=6`, const `is_null=14`, mut `is_null=19`,
`NonNull::new=28/24` (nonnull/null), `NonNull::slice_from_raw_parts=14` and
`as_mut_ptr=14`. The real runtime `is_null` helper, argument tuple and
`const_eval_select` intrinsic event (one) are counted; no intrinsic Call/Return
pairs are invented.

The null exchange boundary `73 + Aalloc` includes exchange's own Call, but NO
exchange Return and NO handler Call yet. The actual handler path adds 14,
including handler Call and runtime-helper Call before the extern invocation:
`87 + Aalloc + Halloc_error`. No fictitious Return, unwind or abort-cleanup
guarantee applies to this nonreturning branch.

Pinned compiler lowering (`rustc_mir_build/src/builder/expr/as_rvalue.rs:183–190`)
charges literal `ShallowInitBox=1`, bare local moves/bindings zero, and distinct
payload place/write `Wpayload=2`. Conservative `Sshallow <= 2` adds explicit
SLACK 1, not another source event. Conditional `CBoxnew <= Eexchange + 7` counts outer Call/Return 2 + generated SizeOf/AlignOf conservative allowance 2 + shallow 1 + payload place/write 2; K does not establish literal source pricing for the generated size/alignment operations.
`100 + Aalloc` belongs to exchange_malloc and ALREADY includes exchange Call/Return; add no further allocation Call/Return 2. Nonzero Box::new retains ceiling `107 + Aalloc` (optional independent SLACK 1), not a literal-source equality.
Pin wrappers add 11 (four Call/Return pairs plus Pin-constructor C=3), giving
`CBoxpin <= Eexchange + 18`, or `CBoxpin <= Eexchange + 19` with SLACK 1.
For nonzero size these are `118 + Aalloc` / conservative `119 + Aalloc`.
No `Aalloc=3` or numeric B=122 clearance is asserted. This Box-pin component
is distinct from bootstrap `Bsuccess=10567` / `Bmax=10595` above.

Official Rust 1.85.1 anchors: [alloc/alloc.rs:90–97,187–195,253–255,347–352,395–409](https://github.com/rust-lang/rust/blob/1.85.1/library/alloc/src/alloc.rs#L90),
[core pointer helpers](https://github.com/rust-lang/rust/blob/1.85.1/library/core/src/ptr/const_ptr.rs#L39)
(`const_ptr.rs:39–67`, `mut_ptr.rs:38–40`, `nonnull.rs:229–235`, `ptr/metadata.rs:124`),
[intrinsics.rs:3855–3869](https://github.com/rust-lang/rust/blob/1.85.1/library/core/src/intrinsics/mod.rs#L3855),
[Box wrappers](https://github.com/rust-lang/rust/blob/1.85.1/library/alloc/src/boxed.rs#L252)
(`boxed.rs:252,318,1679`, `boxed/convert.rs:43`, `pin.rs:1357`), and
[compiler lowering:129–166,183–190](https://github.com/rust-lang/rust/blob/1.85.1/compiler/rustc_mir_build/src/builder/expr/as_rvalue.rs#L129-L190).

This remains UNADOPTED: no ENTRY, Qprefix, funding, logical-frontier adoption,
physical/RSS, security or implementation authority follows. Arc, Mutex, OnceLock
and backend retirement remain unresolved; Rfailure's corrected normal-refusal
inventory follows below, still UNPRICED. Terminal external C=462 and its variable
coefficients remain unchanged under their own assumptions; no aggregate clearance
or change to ordinary paths, caps or execution holds follows.

### Conditional standard Arc Global components (unadopted)

Independent pinned Rust 1.85.1 review under current q=0/K and instrumentation OFF
gives NORMAL returning `Arc<T, Global>::clone` complete C=42:
Call/Return 2 + inner 12 + strong place 1 + Relaxed tag 1 + fetch_add 14
plus comparison/branch 2 + pointer read 1 + allocator place 1 + Global::clone C=3
plus from_inner_in C=5. Atomic-add C=6 includes ordering dispatch and the selected
named-RMW complete-C=3 exception; UnsafeCell::new C=3 and AtomicUsize::new C=6.
The increment precedes the overflow/abort check. Overflow/nonreturning boundary
remains UNPRICED, without a fictitious Return; 42 is not a whole-Arc fee or a
price for unimplemented SIGMA wrapper methods.

Historical `Arc::new` expressions `168 + Aalloc + Pbox`, `169 + Aalloc + Pbox` and ceilings `169 + Aalloc` / `170 + Aalloc` remain WITHDRAWN; C36 independently clears replacement conditional ceilings `173 + Aalloc` / `174 + Aalloc` with independent SLACK 1, NOT ADOPTED.
Their raw_with_allocator → into_raw → leak chain `21 + Pbox` → `26 + Pbox` → `29 + Pbox` (earlier 22/27/30 with Pbox=1) is likewise historical and WITHDRAWN. Pbox represented the additional Box-place projection in `&raw mut **b` (boxed.rs:1420); its proposed per-place allowance does not repair missing receiver/coercion events.
C36 reconciles Box::into_raw_with_allocator C=25, Box::into_raw C=30 and Box::leak C=33 (Pbox ALREADY embedded). NonNull::from_mut C=5 counts Call/Return 2 + explicit reference-to-mut cast 1 + mut-to-const coercion PREMIUM 1 + pointer write 1; From/Into C=7/9 and from_inner C=7.
Arc::new composes as `2 + 2*6 + 3 + (107 + Aalloc) + 33 + 9 + 7`; calculation-only `Aalloc=101` gives 274/275. Do not automatically replace other borrow fees or adopt these rates; Arc::clone C=42 is unchanged.
NonNull::as_ref C=9 and other addressing dereferences retain their existing
borrow-subsumption assumption; neither C=9 nor terminal C=462 is reopened.
`Aalloc` remains the UNPRICED returning __rust_alloc frontier, distinct from
Quoting A=3 and cleanup array-edge A. Anchors: [sync.rs:263–281,385–394](https://github.com/rust-lang/rust/blob/1.85.1/library/alloc/src/sync.rs#L263),
[clone:2158–2192](https://github.com/rust-lang/rust/blob/1.85.1/library/alloc/src/sync.rs#L2158),
[boxed.rs:1413–1423](https://github.com/rust-lang/rust/blob/1.85.1/library/alloc/src/boxed.rs#L1413).
No Context/backend/Arc Drop coverage, full E, fee adoption, funding, implementation
or execution authority follows from either component.

### Default allocator frontier qualification (unadopted)

Independent pinned Rust 1.85.1 source audit finds `__rust_alloc` is a compiler-generated forwarding shim → default `__rdl_alloc` → `Layout::from_size_align_unchecked` → `System::alloc`, NOT true external native allocation; the compiler generator itself is not runtime work.
For valid NONZERO layouts on default 64-bit Darwin, MIN_ALIGN=16: `align <= 16 && align <= size` calls native malloc; otherwise Apple `align > 2^31` returns null with NO native call; else Rust `aligned_malloc` calls native posix_memalign with effective alignment `max(align,8)`.
All actual shim/Rust wrapper/helper/control costs before the native frontier must be expanded; positive `M = Cmalloc` / `P = Cposix` remain symbolic/unadopted outside the calculation-only proposal below, not literal source-derived 3.
A custom `#[global_allocator]` in the downstream crate/dependencies can replace the entire default chain with arbitrary Rust; test allocators do not prove production selection.
Conditional finite-wrapper review requires pinned compiler/backend/ABI, selected System, Darwin64 and instrumentation OFF; linkage/interposition qualification remains OPEN.
A generic opaque `__rust_alloc` frontier absorbing Rust helpers is an explicit UNADOPTED exception, not justified by an extern declaration.
Proposed NEXT default-path `Aalloc = max_branch(expanded shim/Rust events + applicable positive native fee)` has conditionally reviewed components below; Fshim/native fees remain symbolic/unadopted outside the calculation-only proposal below and adoption OPEN, not a qualified allocator total.
Arc ceilings `173 + Aalloc` / `174 + Aalloc` and Box::pin `118 + Aalloc` / slack `119 + Aalloc` remain CONDITIONAL, NOT ADOPTED on that frontier (no additional +7); old Arc 169/170 ceilings are historical and withdrawn. No automatic substitution, aggregate E or C=462 change follows, nor full Box closure, backend-refusal coverage, physical/RSS, OOM/unwind/cleanup or security authority.
Official Rust 1.85.1 anchors: [compiler/rustc_codegen_llvm/src/allocator.rs:34–61,157–169](https://github.com/rust-lang/rust/blob/1.85.1/compiler/rustc_codegen_llvm/src/allocator.rs#L34),
[std/alloc.rs:395–403](https://github.com/rust-lang/rust/blob/1.85.1/library/std/src/alloc.rs#L395),
[std/sys/alloc/unix.rs:8–29,75–86](https://github.com/rust-lang/rust/blob/1.85.1/library/std/src/sys/alloc/unix.rs#L8), [sys/alloc/mod.rs:31–43](https://github.com/rust-lang/rust/blob/1.85.1/library/std/src/sys/alloc/mod.rs#L31).

### Current conditional default-allocator ledger (unadopted)

Independent review conditionally clears these components under Rust 1.85.1, default
System, 64-bit Darwin, valid NONZERO layout, normal null/nonnull return and instrumentation OFF.
`M = CnativeMalloc > 0` / `P = CnativePosixMemalign > 0` remain symbolic/unadopted complete invocations outside the calculation-only proposal below.
Layout::from_size_align_unchecked C=5 is Call/Return 2 + two field writes + terminal
transmute 1; THIS constructor calls neither an Alignment constructor nor as_usize.
`__rdl_alloc` literal C=`7 + S`, plus named receiver PREMIUM 1, gives `8 + S`, with
`S = CSystem` including its Call. Positive generated-forwarding fee `Fshim`, symbolic/unadopted outside the calculation-only proposal below,
has proposal 2 (external shim Call + logical Return); __rdl_alloc Call is already in C.
K does NOT price generated LLVM forwarding: calculation-only acceptance below is separate from tariff adoption, not a
CPU/runtime-instruction claim or compiler-generator execution fee. General `Aalloc = 8 + S + Fshim` uses receiver PREMIUM 1, not a literal numeric whole fee.
System uses receiver PREMIUM r=1 PER Layout accessor, a conservative allowance,
not an adopted autoref rule. Helpers: Layout::align C=7 / size C=3, Alignment::as_usize C=4,
null_mut C=8, usize Ord::max C=12; aligned_malloc C=44+P success / 50+P error includes common 42+P.

| System complete-C branch | Success | Error/null |
| --- | ---: | ---: |
| `align <= 16 && align <= size`: malloc | `31 + M` | `31 + M` |
| Otherwise Apple `align > 2^31`: no native call | — | `32` |
| `16 < align <= 2^31`: aligned | `69 + P` | `75 + P` |
| `align <= 16 && size < align`: aligned | `82 + P` | `88 + P` |

First-false `&&` still pays enclosing if-branch 1; earlier undercounted rows are WITHDRAWN.
`Sbar = max(31 + M, 88 + P)`; `Aallocbar = 8 + Fshim + Sbar`. Conditional
use of Fshim proposal 2 gives `Aallocbar <= max(41 + M, 98 + P)`; M/P remain symbolic/unadopted outside the calculation-only proposal below.
Additional Rust 1.85.1 anchors: [core/alloc/layout.rs:129–140](https://github.com/rust-lang/rust/blob/1.85.1/library/core/src/alloc/layout.rs#L129), [ptr/alignment.rs:90–92](https://github.com/rust-lang/rust/blob/1.85.1/library/core/src/ptr/alignment.rs#L90), [ptr/mod.rs:577–579,641–648](https://github.com/rust-lang/rust/blob/1.85.1/library/core/src/ptr/mod.rs#L577), [cmp.rs:1512–1516,1730–1734](https://github.com/rust-lang/rust/blob/1.85.1/library/core/src/cmp.rs#L1512); allocator links above apply.
No const-fn/free-folding exemption, automatic Arc/Box/ENTRY/C=462 substitution or ordinary-quote change follows. Metric/CPU, funding, production-selection/custom-allocator/linkage, implementation and execution holds remain; conditional Oonce/Mmutex constructor review follows below.

**Calculation-only authorization (not tariff adoption).** The user approved `M=3`, `P=3`
as FFI caller Call 1 + opaque WHOLE native-invocation event 1 + normal Return 1,
including native output effects; pure forwarding `Fshim=2` is Call 1 + logical Return 1, with NO opaque event.
System malloc base 31 / aligned common base 42 EXCLUDE complete M/P, so no Call/Return is doubled; __rdl_alloc Call is already inside `8 + S`.
Thus `Sbar = max(31 + 3, 88 + 3) = 91`; `Aallocbar = 8 + 2 + 91 = 101` for this calculation ONLY.
Rust __rdl_alloc/Layout/System/aligned_malloc helpers remain expanded; native internal work lies outside the NEW logical unit, not a physical-work conservatism claim or inherited K/C=462/free-native fee.
Default System, Darwin64, pinned compiler/backend/ABI and instrumentation OFF are required; production selection and linkage/interposition qualification remain OPEN, through standard build qualification, not a new runtime API/build wrapper.
No metric/tariff adoption, funding, implementation/execution or physical CPU/wall/RSS guarantee follows. Arc/Box, first-lock, Rfailure and full-E holds remain; no automatic substitution into qualified totals or funding.

### Conditional OnceLock/Mutex constructors (unadopted)

Independent source review conditionally certifies constructor candidates for Rust 1.85.1,
Darwin64, instrumentation OFF, q=0/K: actual Context types are `OnceLock<OwnedCharged<Configured>>`
and `Mutex<Ledger>`. Complete C includes caller Call and Return once; no actual compiled backend/cfg is inferred.

| Helper / constructor | Conditional complete C |
| --- | ---: |
| UnsafeCell::new / MaybeUninit::uninit; thin null_mut | 3 / 3; 8 |
| AtomicPtr::new / AtomicBool::new | 6 / 7 (Bool includes cast 1) |
| Queue Once / public Once; OnceBox / pthread sys Mutex | 12 / 15; 17 / 20 |
| Poison Flag | 10 unwind / 2 abort |
| OnceLock::new (`Oonce`) / Mutex::new (`Mmutex`) | 26 / 38 unwind or 30 abort |

`Oonce = 2 + 3 field writes + 15 + 3 + 3 = 26`; `Mmutex = 2 + 3 field writes + 20 + (10 or 2) + 3 = 38/30`, cross-cfg ceiling 38.
Combined complete C=64 unwind / 56 abort, or BODY=62/54 if enclosing caller already counts both Calls.
Runtime const calls are NOT free constant folding: explicit ZST-marker/poison field writes are paid; no generic SSA-operand transfer, extra assembly or receiver premium is added.
Ledger/Context construction and Option::map/Some remain separate caller work. Neither constructor constructs Configured nor allocates pthread backing; first lock/init/native allocation/drop/TLS remain EXCLUDED and unpriced under their own review.
Rust 1.85.1 anchors: [once_lock.rs:134–139](https://github.com/rust-lang/rust/blob/1.85.1/library/std/src/sync/once_lock.rs#L134), [poison/mutex.rs:273–274](https://github.com/rust-lang/rust/blob/1.85.1/library/std/src/sync/poison/mutex.rs#L273), [once/queue.rs:119–120](https://github.com/rust-lang/rust/blob/1.85.1/library/std/src/sys/sync/once/queue.rs#L119),
[once_box.rs:20–21](https://github.com/rust-lang/rust/blob/1.85.1/library/std/src/sys/sync/once_box.rs#L20), [mutex/pthread.rs:14–15](https://github.com/rust-lang/rust/blob/1.85.1/library/std/src/sys/sync/mutex/pthread.rs#L14), [poison.rs:107–112](https://github.com/rust-lang/rust/blob/1.85.1/library/std/src/sync/poison.rs#L107), [core atomic.rs:428,1223](https://github.com/rust-lang/rust/blob/1.85.1/library/core/src/sync/atomic.rs#L428).
These proposals remain UNADOPTED/unfunded: no C=462 change, full ENTRY, authoring or execution clearance, or proposed entry-owner layout/ownership adoption follows.

### Normal prefix-refusal retirement inventory (Rfailure unpriced)

Independent source review, including SourceAdmission manual Drop at OS:374–385, establishes this inventory for NORMAL refusal at OS:1189–1204 only; it is not numeric Rfailure closure.
A live caller/helper guard retires FIRST, already counted in its caller/method ledger;
after the explicit OS:1200 guard drop, none remains. Quoting fields retire in declaration
order source → working → attempt (OS:489–493). Source retires exactly ONCE under frozen imported-cleanup candidate `C = 462 + 128N + 8A + 452M + 390O`, scoped to its HISTORICAL normalization, not certified current helper K; do not repeat source/raw payload or its two embedded charges/Budget-Arc tails.
Context callback, outer Context-root Arc/root charge, working/controller charges and abandonment/caller guards are DISTINCT ownership cuts, not included by assumption. Working-charge retirement remains a separate positive complete cleanup obligation; aggregate Rfailure requires normalization compatibility and nonoverlapping cuts proved first.

Uncommitted `PrimitiveAttempt::drop` (OS:212–226) burns via one NEW lock, poison recovery and guard retirement. After manual Drop, fields (OS:141–145) retire root Arc FIRST
(nonlast while caller holds root), controller charge SECOND. Source payload has already
retired, but root Context/root-charge custody remains through the burn cycles.
CURRENT Prepaid `SourceAdmission::configure` (OS:362–370) returns prefix Err before `construct` is reached. Its admission manual Drop (OS:374–385) checks `configured.get` /
`is_none`, accesses ledger, and performs a SECOND lock/poison-recovery/burn/guard cycle;
admission fields (OS:236–240) then drop root Arc, last absent other owners. These two abandon cycles are distinct from already-counted caller/method guards.
Last-root retirement runs `OwnedCharged<Context>` payload → Context manual Drop taking
empty configured → OnceLock/ledger-Mutex field retirement → root charge, with actual Arc
strong/implicit-Weak/backing branches. Retained root readers change LAST-strong choice,
not the obligation to charge actual strong drops. Decompose Rfailure OR use a complete component, never both that component and its subcalls.

Normal Context retirement clarification (OS:122–129), pinned official Rust 1.85.1:
[`OnceLock::take`](https://github.com/rust-lang/rust/blob/1.85.1/library/std/src/sync/once_lock.rs#L484-L498) directly tests `is_initialized`, without `get_mut` or Try on the empty branch. Even an empty configured cell has TWO completion checks: Context's take and subsequent [`OnceLock` manual Drop](https://github.com/rust-lang/rust/blob/1.85.1/library/std/src/sync/once_lock.rs#L668-L677).
Initialized [`sys::Mutex::drop`](https://github.com/rust-lang/rust/blob/1.85.1/library/std/src/sys/sync/mutex/pthread.rs#L52-L65) takes the backend, then performs `try_lock` and `unlock` on the normal unlocked-success path BEFORE actual `Pin<Box<_>>`/PAL retirement; see [PAL methods and Drop](https://github.com/rust-lang/rust/blob/1.85.1/library/std/src/sys/pal/unix/sync/mutex.rs#L89-L128).
[`OnceBox::take`](https://github.com/rust-lang/rust/blob/1.85.1/library/std/src/sys/sync/once_box.rs#L56-L58) replaces its pointer with NULL; subsequent [`OnceBox` field manual Drop](https://github.com/rust-lang/rust/blob/1.85.1/library/std/src/sys/sync/once_box.rs#L79-L82) sees NULL and does not destroy the backend again.
Conditional component counts below are independently reconciled, not adopted rates: complete atomic/field-glue coverage, adoption and retirement residuals remain unfunded; actual Box/PAL/free wrappers remain separately reviewed. Root charge covers the Context callback through completion (SA:552–555), not outer Arc or charge cleanup; imported C is unchanged.
C31 independently reconciles Box::into_raw C=30, not prior 27 (into_raw_with_allocator C=25): two ManuallyDrop receiver borrows and reference-to-raw coercion add three events not established as subsumed.
Conditional Rust 1.85.1 Darwin64, instrumentation OFF, normal allocation/native success, unpublished single caller/no CAS contender: OnceBox::initialize `<= 71 + F`, retaining original initializer `F <= 241 + Aalloc + Ninit`; this F is local to first-lock pricing, not retirement Box::from_raw F below.
Fresh complete backend `<= 398 + Aalloc + Ninit + Nlock`; calculation-only `Aalloc=101` gives `<= 499 + Ninit + Nlock`. Cached complete backend `<= 90 + Nlock`.
`Ninit` sums four positive native complete invocations (attr_init, attr_settype, mutex_init, attr_destroy). User separately approved CALCULATION-ONLY complete C=3 (Call 1 + opaque native 1 + Return 1) for each returning pthread attr_init/settype/destroy, mutex_init/lock/trylock/unlock/destroy and free; all Rust wrappers remain separate, with no tariff adoption or backing-coverage clearance.
Thus calculation-only `Ninit=12`, `Nlock=3`, fresh backend `<= 514` and cached backend `<= 93`, retaining calculation-only Aalloc=101; malloc/posix or allocation-shim approval alone did not authorize these native values.
These backend candidates exclude public-Mutex guard/poison/TLS, OOM/unwind and whole E; no adoption follows.
A9's recovered A8/C140 ledger clears conditional complete public Mutex::lock C as complete child backend `B + 27` (abort), `B + 73 + Sonce` (unwind Ok), or `B + 80 + Sonce` (cached poison Err), AFTER receiver premiums; old 75/82 offsets are WITHDRAWN. Public outer Call/Return and B's own Call/Return are already included; BODY with public Call prepaid is `B + 26` / `B + 72 + Sonce` / `B + 79 + Sonce`. Here public C=`B + 4 + GuardNew`, GuardNew Ok/Err=`16/20 + FlagGuard`, FlagGuard Ok/Err=`26/29 + Tpanic`, Tpanic=`27 + Sonce`; Flag::get C=20 is outer 2 + field 1 + receiver 1 + Ordering 1 + Bool C15. Sonce applies once only on actual slow paths, not an all-path/unwinding proof or adoption/funding/backing/physical/execution clearance.
Conditional native-TLS `Cell<(usize,bool)>` const/no-drop accessor `&VAL`: `Sonce <= 60 + FtlsGenerated`; callback 14, try_with `52 + FtlsGenerated`, with `57 + FtlsGenerated`. Cell::get literal 9 plus named receiver PREMIUM/SLACK 1 gives 10; reference-to-raw PREMIUM 1 is explicit, not globally adopted.
User approved CALCULATION-ONLY `FtlsGenerated=3`: logical entry 1 + whole generated episode 1 + completion 1, NOT Rust Call/Return. A144 scopes this exactly once from address request to address returned, including resolver/TLV initialization, not an invented pthread_getspecific call; Rust wrappers remain separate.
Then `Sonce <= 63` is added ONCE only on actual slow paths; fresh unpoisoned does NOT imply global panic count zero or no TLS access. Const/no-drop Rust TLS does NOT exclude OS/loader initialization, hidden allocation or backing; selected Apple has_thread_local default is not compiled-std proof. Profile/path, backing, funding, physical-bound, adoption and execution holds remain.
Conditional existing-local-K reconciliation (C135/A11), not a new adoption: `N = 3` is Call + Dispatch + Return for INVOKED Option-None glue whose payload requires drop.
`G = 23 = Context 5 + OnceLock 3 + Some-Mutex Option 4 + public Mutex 4 + sys Mutex 4 + OnceBox 3` includes five field transfers and glue Call/Return/dispatch, but excludes child/manual C.
`a = 3` uses ONLY the existing named Acquire BODY 2 + Call 1 exception, not the ordinary terminal-intrinsic rule or a new native price. Type-specific scalar Ledger, poison Atomic and MaybeUninit/marker fields have no drop-requiring children; this is not a generic free-UnsafeCell rule.
The pinned [pthread sys-Mutex Drop](https://github.com/rust-lang/rust/blob/1.85.1/library/std/src/sys/sync/mutex/pthread.rs#L52-L65) `let Some(pal) = self.pal.take() else { return }` introduces a THIRD Option temporary: None on null, sole payload moved on initialized. Standalone residual glue, if invoked, is at most 3, but compiler partial-move/drop-flag control is NOT bounded by existing enum-dispatch normalization.
Literal-source bounds for `rnull`/`rinit` remain OPEN; any earlier source-based residual `<= 3` claim is withdrawn. Complete residual cleanup includes flag setup/tests and empty cleanup, excludes already-priced pattern dispatch and moved-payload retirement. There is no fourth None from OnceLock manual Drop; the second OnceBox take sees NULL.
User approved CALCULATING WITH new logical unit `R=3`: entry 1 + whole control 1 + completion 1 for ONLY the third Option temporary in that sys-Mutex `let Some(pal)` episode. A25 independently clears that site scope without overlap with N/G. This includes its complete residual control, not expanded Rust/MIR/CPU work; it grants no free residuals or permission at other R3 sites.
C151 resolves the child cut: original sys-Mutex manual-Drop fixed 58 ALREADY paid explicit mem::drop(pal) Call/Return 2 (own Call/Return 2 + pal field 1 + null-take baseline 48 + nonnull Pin::new_unchecked 3 + pattern 1 + try branch 1 + mem-drop 2). Missing Some-payload destination WRITE 1 makes fixed 59; Some tag replaces the baseline None tag, and Pin's field write is already inside C=3. Context initialized base is therefore `212 + F + 2P + T + U + D_pin + rinit + Rcaller`; approved R3 gives `215 + F + 2P + T + U + D_pin + Rcaller`. Null `205 + rnull` → 208 is unchanged; these are conditional calculations, not old-K source bounds.
`F` is Box::from_raw complete C; `P` is Pin<Box<_>>::as_ref complete C; `T`/`U` are actual PAL try_lock/unlock complete C; `D_pin` is the complete Pin-glue invocation including V (manual PAL destroy) and Box deallocation, EXCLUDING outer mem::drop Call/Return 2 already paid by the parent.
A147 independently reviews conditional `F=14`: NonNull 4 → Unique 8 → Box::from_raw_in 12 → Box::from_raw 14, with positive explicit field writes, Global evaluation zero and no trust/assume event under UB checks OFF.
Conditional `P <= 12` ([Pin:1370–1372](https://github.com/rust-lang/rust/blob/1.85.1/library/core/src/pin.rs#L1370-L1372), [Box:1970–1971](https://github.com/rust-lang/rust/blob/1.85.1/library/alloc/src/boxed.rs#L1970-L1971)) already includes internal Deref receiver-borrow and Box-projection premiums, not the two EXTERNAL `pal.as_ref()` receiver formations at pthread:57–58.
Keep positive `Rcaller` for those formations, distinct from compiler-residual R3: named one-per-call receiver premiums give `Rcaller=2` under the same conditional borrow convention, not global adoption. PAL try_lock/unlock receive Pin by VALUE; add no further external borrow for those calls or duplicate internal P premiums.
F=14/P<=12 gives initialized ceiling `250 + T + U + D_pin + rinit + Rcaller`, then `252 + T + U + D_pin + rinit` with Rcaller=2, then calculation-only R3 `255 + T + U + D_pin`. Prior 249/251/254 and `327 + D` combinations are WITHDRAWN for the missing Some write and mismatched child cut.
A25 and C45 independently clear conditional `T <= 34 + Ptry`, `U <= 33 + Punlock`, `V <= 13 + Pdestroy`, sum `<= 80 + Ptry + Punlock + Pdestroy`; old PAL 65 remains WITHDRAWN. Pin Deref literal 16 + explicit SLACK 4 gives 20, raw C=9; K retains the false DragonFly cfg branch event 1 with debug assertions OFF.
Approved native calculation-only C=3 gives T/U/V ceilings 37/36/16, sum 89; initialized Context ceiling is `328 + D_pin`. V stays INSIDE D_pin, never added twice; null remains 208, with no full E or adoption.
C45/A11's conditional boundary under pinned ABI/default Global→System, normal valid unlocked sole-owner retirement, UB checks OFF and qualified profile has NonNull::cast C/BODY=8/7, Unique::cast=13/12 and Box manual Drop C=`83 + Sdealloc + Nfree` before receiver premiums ρ=6 (one borrow each: Box Drop:1700 layout.size, Box:1701 allocator &self.1, Global:266 layout.size, alloc::dealloc:116 layout.size and layout.align, __rdl_dealloc:409 System). All six are outside 83; its allocator-field PLACE is distinct from the borrow. ptr.as_ptr/cast are by VALUE, for_value_raw is associated with no receiver borrow, and both mut-to-const coercions are already in 83. Whole mem-drop C=`96 + V_BODY + Sdealloc + Nfree + ρ + Gret` gives calculation-only `120 + Sdealloc + Gret` (V_C=16/V_BODY=15, Nfree=3); D_pin C=`118 + Sdealloc + Gret` excludes outer mem-drop 2. BODY subtracts one Call ONLY at the corresponding cut. Reconciled Context composition is conditionally `446 + Sdealloc + Gret`, NOT numeric closure: both terms remain OPEN.
Gret remains OPEN: compiler shim:363–389 has static field drops, NO drop flags or flag clearing. PAL glue BODY=`borrow 1 + V_C + Return 1` and Box glue BODY=`transfer 1 + PAL_glue_C + borrow 1 + Box_manual_C + Return 1` ALREADY pay two generated destructor-receiver borrows, distinct from ρ; Gret=6 is WITHDRAWN for duplicating them. Three Box→Unique→NonNull→raw-pointer generated field components plus pointee-address formation make ONE composite place (elaborate_drops:425–435); whether the existing Box→PAL transfer 1 covers the whole composite or already covers its address remains unresolved. Compiler-builder operations are not automatically runtime events; normal Goto still needs mapping to the existing dispatch convention. No Gret=4, 3 or 0 is proved. R3 remains ONLY the single Option let-Some episode, distinct from Context G=23; Sdealloc/__rust_dealloc shim remains OPEN. A9's READ-ONLY proposal is external Sdealloc=2 (shim Call 1 + logical completion 1), or BODY=1 when Call is prepaid, excluding the already-paid complete __rdl_dealloc child; separate calculation authorization is required before substitution, and allocation Fshim=2 authorization does NOT transfer. Native free=3 and TLS=3 remain calculation-only; no D closure follows.
A25/A9 correct guard-path Relaxed load C=14: outer 2 + field 1 + receiver 1 + UnsafeCell::get 5 + mut-to-const 1 + atomic_load 4; prior 16 misapplied the Acquire exception, not an exact Relaxed rate. Guard Drop uses one backend Relaxed load and one panic Relaxed load only for unwind; Acquire a=3/Context atomics stay unchanged. Public-lock correction above is resolved: global-count and poison-flag loads each net −1 (old intrinsic 3→1 gives −2, missing mut-to-const coercion adds 1), without recounting receiver premiums; this is NOT merely Tpanic 29→27 or global repricing of helpers/C462.
Pinned guard-path cuts (`std/src/sys/sync/mutex/pthread.rs` unlock and `std/src/sys/sync/once_box.rs` get_unchecked): sys cached unlock C=63, get_unchecked C=23; manual Guard Drop C=75 abort / `106 + Sonce` unwind; implicit glue C=78 / `109 + Sonce` (slow <=172); explicit mem-drop C=80 / `111 + Sonce` (slow <=174). BODY subtracts one Call; do NOT sum implicit and explicit cuts. Conditional captured-panicking=false/current-thread-not-unwinding path, no poison store, borrowed Mutex rather than owned retirement; choose the actual callsite cut before Rfailure composition. Sonce<=63 remains calculation-only; no adoption/funding/backing/physical/execution clearance follows.
These conditional formulas exclude root Arc/charge cleanup and establish no full Rfailure/E bound or funding. Imported `C = 462 + 128N + 8A + 452M + 390O` and allocator 101 remain unchanged.
This clarification grants no full Rfailure/E/ENTRY funding, ownership/layout adoption, unwind/OOM/leak, physical/security or implementation/execution clearance.

Legacy `quote` (OS:1176–1182) retains a BARE lexical root Arc: one attempt burn, then outer root drop, NO admission manual Drop. The proposal below chooses a bare-root owner for pricing only;
the consuming `quote_opt_in` caller is neither implemented nor selected, and need not use SourceAdmission or two burns.
`new_context` (OS:325–327), not `enter_attempt`, constructs `ledger.map(Mutex::new)`; current fresh constructors supply Some ledger; any proposed constructor relying on this must preserve that invariant. This inventory does not infer backend lazy initialization.
`PrivatePreparedConfiguration` (OS:284–288) has NO manual Drop; source/attempt move at 369, leaving trivial workload. Configure checks (365–367) and pre-root/admission refusals are OUTSIDE this prefix-cleanup scope and require separate funding.

Positive UNPRICED Rfailure components remain outer glue, working/controller charge drops,
attempt abandonment, applicable SourceAdmission abandonment, actual root drops and
Context/root-charge/Arc/backend retirement. No C-coefficient change, new ledger, refund,
unwind/OOM/leak guarantee from normal Drop, complete ENTRY/Qprefix funding, adoption,
authoring or execution clearance follows.

### Proposed bare-root entry ownership candidate (unadopted)

Two independent source reviews conditionally clear a NEW pricing candidate, not layout adoption, API authoring or implementation: retain the frozen ENTRY schedule above with a bare lexical caller Arc root and existing PrimitiveAttempt, distinct from implemented legacy quote and SourceAdmission.
Following the OS:1176–1189 pattern, borrow root for enter_attempt, then retain it lexically during `quote_on_root(source, working, attempt)`; root is NOT passed to that helper.
ExistingSource's zero-add preserves the seed (OS:707); Prepaid avoids LegacyReported import (OS:1193–1195).
Failure before attempt creation at OS:712 has zero attempt-abandon cycles; the direct debit burn write at 704 is distinct.
Attempt creation has ONE Context-root clone (OS:712–715); the two SA:490–501 Budget clones have different owners.
Helper Err retires Quoting source → working → attempt (OS:489–493); attempt manual Drop has one burn lock/guard, then root Arc NONLAST → controller charge (OS:141–145,212–226), then outer lexical root LAST absent other owners.
Success Quote (OS:1213–1222) retains source/working/attempt/original ledger; outer root drops NONLAST. No SourceAdmission or custom manual-Drop wrapper, hence no second admission burn, is proposed in this candidate.
Rfailure phase inventory: pre-root retires reached child/residual controls; post-root/pre-attempt adds final Context-root retirement; post-attempt adds one abandonment, attempt fields and outer last-root retirement.
Imported cleanup remains separate ONCE; this is not numeric Rfailure closure. Source-before-working pre-entry order is CONSERVATIVE ONLY, not inherently required for payload custody: source's own charge covers payload through destruction (SA:552–555); no unnecessary wrapper/order is required.
Actual new-entry lexical retention, genuine split amounts, seed/cap, Prepaid and rejected-entry prefunding remain UNIMPLEMENTED/UNADOPTED choices, under normal valid-owner assumptions.
Caller funding remains required BEFORE handoff under unchanged caps; a new owner is not an old-ledger reset. No extra persistent owner or global retry rule follows.
No OOM/unwind, physical/security, full ENTRY/funding, authoring or execution clearance follows; implementation and ownership-selection holds remain.

## Proposed terminal-cleanup tariff (unadopted)

Detailed review only: `crates/pllm-models/src/source_work.rs:3–7` permits structural
visits and named opaque invocations, but supplies no inherited destructor fee.
Official Rust 1.85.1 source anchors: [Vec drop](https://github.com/rust-lang/rust/blob/1.85.1/library/alloc/src/vec/mod.rs#L3773),
[BTreeMap iteration/drop](https://github.com/rust-lang/rust/blob/1.85.1/library/alloc/src/collections/btree/map.rs#L1701),
[Arc drop](https://github.com/rust-lang/rust/blob/1.85.1/library/alloc/src/sync.rs#L2545),
and [deallocation](https://github.com/rust-lang/rust/blob/1.85.1/library/alloc/src/alloc.rs#L114).
Vec drops its slice/RawVec, not capacity elements; BTreeMap IntoIter drop uses
`dying_next`, navigation and node release; Arc has strong/implicit-Weak final tails.
For locked serde_json 1.0.151/default Number/BTreeMap (no IndexMap in the locked
graph), propose `CLEANUP = k0 + kv*N + ka*A + km*M + kc*O`, with authentic cached
N=`spans` (values AND object-key strings, including duplicates/empty containers),
A=`array_children`, M=`map_entries`, O=`objects`. Final retained population is at
most these counts; earlier duplicate-replacement drops belong to construction.
Proposed bounds: nodes b ≤ M+O; `dying_next` ≤ M+O; successful entry retirements ≤ M;
ascent/node-release events ≤ M+O; key/value drop calls ≤ 2M; array-child retirements
≤ A; Value dispatch ≤ N; string/key and array backing owners ≤ N, plus one raw backing.
Container field glue/setup/termination is separate from children: array containers
≤ N and object containers O; `[]` with A=0 and `{}` with M=0 still incur cleanup.
Backing-owner dispatch applies even at capacity zero; deallocation requires backing.
`first_leaf_edge`/`next_leaf_edge` downward-edge visits ≤ b ≤ M+O are separate from
ascent/node release. This node bound includes an allocated empty root: last-node
release through `deallocating_end` remains possible when M=0.
Include two charge releases and two strong drops, with possible last-strong,
implicit-Weak and backing-release branches; external Weak operations need separate
funding. Feature-union/backend changes require fresh review; no physical custody claim.
HISTORICAL IMPORTED-CLEANUP SCOPE — UNADOPTED PRICING PROPOSAL ONLY: one event per variant dispatch, field-retirement transfer,
call/return and scalar operation/branch; handwritten helpers are expanded, not bundled.
Nested callee BODY includes its own Return; caller Call is separate, not a second Return.
Trivial byte-slice drop has fixed dispatch/return, not a per-byte walk; nontrivial
slice drop charges per initialized-child transfer/control. Atomic RMW/acquire and
opaque `__rust_dealloc` each cost one proposed event, plus wrapper/layout/branch fees.
This is an explicit abstract contract, not an LLVM/CPU/RSS/allocator-cost bound.
Independent historical review conditionally cleared this convention for its NUMERIC
PROPOSAL within detailed funding review, not recalibration or certification under current helper K.
Two UNADOPTED candidate normalizations remain separate. In this historical imported-cleanup normalization, `C(f) = 1 + BODY(f)`: caller Call is separate;
BODY includes its own Return once. Enum dispatch includes tag selection, without
an extra tag load/path branch; ordinary `if` counts comparison and branch.
SSA aliases/argument binding cost zero; consumed field reads/writes, explicit
place/borrow formation, casts and scalar operations cost one each; autoderef adds
no fee. Expand handwritten helpers, including pointer accessors. Historical intrinsic
pricing is one plus actual wrapper Call/Return, with atomic/acquire/opaque-release BODY 2 / C 3 where assumed; frozen numbers below are not recalibrated or certified under current helper K.
Current helper review instead prices ordinary Relaxed terminal intrinsics at one event, with actual wrappers, coercions and receiver premiums separate; named Acquire BODY 2 / C 3 remains exceptional, not a blanket atomic rate.
Checked operations and overflow detection are separate, as are result-tag,
dispatch and control events. Glue has Dispatch plus Return; each drop-requiring
field adds transfer, Call and BODY. No-drop scalar/record/ZST fields add no fee.
Manual Drop Call/method BODY comes first; an explicitly invoked no-drop BODY is two.
This is not physical instruction accounting. The assumed compiled closure is normal non-unwinding
cleanup with instrumentation OFF, including omitted `debug_assert!` AND UB-precondition checks, not merely non-TSan; Rust 1.85.1, serde_json 1.0.151 default
Number/BTreeMap and Global. This is conditional, not a claim about an actual unrun build; feature/backend changes require fresh review.
The UNADOPTED fixed-owner component is `BODY = 342 + V + R`, external caller
`343 + V + R`: V is root Value BODY, R original raw Vec BODY. Their Call slots
are already in the fixed ledger; do not add callback Calls again. Compact ledger:
two OwnedCharged glues at 6, ImportedValue glue 6, two charge glues at 162 = 342.
Each `C(manualDrop BODY 24)` is caller Call 1 + BODY 24 = 25;
`charge glue = 1 + (1 + 24) + 2 + 133 + 1 = 162`
(Dispatch, manual Drop, Arc-field transfer/Call, Arc BODY, Return). Resources/Counts are scalar-only,
no-drop zero (`json_value_input.rs:29–47,70–75`; `plan_input.rs:164–176`;
`crates/pllm-garble/src/shared_allocation.rs:429–432,544–555`).
Exactly two embedded charges/strong drops have complete final-tail BOUND
alternatives, not two destructions of the same Budget. Caller root/sibling
handles and external Weak-operation multiplicity are outside this component.
Arc backing release is fixed here; all Vec/RawVec allocation releases remain inside
their respective V/R callback bodies, never this fixed component.
The old 296 proposal is SUPERSEDED by incompatible field-glue/helper fees, not
padding. 342 is neither ENTRY nor whole CLEANUP. Independent conditional SOURCE
review finds no arithmetic defect in fixed-owner 342 and Value/Vec 119+128N+8A
under this HISTORICAL imported-cleanup normalization only; these TWO components remain UNADOPTED, not certified current-K components.
Raw Vec BODY is 119, array-edge overhead 8, key String BODY 123; ValueArray BODY
is `128 + 8*len + sum(child BODY)`. The former 130N candidate becomes 128N because
default scalar-only Number has no-drop BODY 2, not 6 (serde_json 1.0.151
[`number.rs:20–34`](https://github.com/serde-rs/json/blob/v1.0.151/src/number.rs#L20),
[`value/mod.rs:116–175`](https://github.com/serde-rs/json/blob/v1.0.151/src/value/mod.rs#L116);
Rust Vec drop linked above). No additional root Value/raw transfer Calls join 342.
For counts v=Value nodes, s=String/key-string owners, q=array containers and
e=array edges, `119 + 4v + 123s + 124q + 8e ≤ 119 + 128N + 8A`, using v≤N,
s+q≤N and e≤A; cached N counts values AND key strings, A counts array edges.
The candidate partial BODY bound is `461 + 128N + 8A + sum(MapLocal)`;
external caller adds one: `462 + 128N + 8A + sum(MapLocal)`.
MapLocal excludes key/value callback BODYs counted globally, but INCLUDES their
field transfers and Calls: recursive Map BODY is
`MapLocal + sum(key String BODY) + sum(value BODY)`.
Independent BTree Astra 11 (session e7eb985adfffe) conditionally clears the UNADOPTED
`sum(MapLocal) ≤ 452M + 390O` candidate under that assumed closure, not funding or adoption.
Local fees include Call + Return and exclude callback BODYs: successful entry 214,
node exhaustion 169, downward traversal 69, object-fixed work 152. For retained
entries m, nodes b, downward edges d and objects o:
`MapLocal ≤ 214m + 169b + 69d + 152o ≤ 452m + 390o`, using `d ≤ b ≤ m+o`.
Node release 118 is ALREADY inside 169; add no allocator/node fee or new counters.
Each object makes m successful `dying_next` calls plus one terminal call, including
an allocated empty root, in one forward traversal. Normal Dropper retires the value;
guards are forgotten normally. Cached M/O dominate retained populations, including
duplicate inputs; earlier duplicate-replacement construction drops are not charged again.
Closure-capture SSA aliases are free, but explicit field reads/borrows are charged;
no synthetic closure-layout fee applies.
Official Rust 1.85.1 anchors:
[map.rs:186–190,1680–1738](https://github.com/rust-lang/rust/blob/1.85.1/library/alloc/src/collections/btree/map.rs#L1680),
[node.rs:318–414,1175–1195](https://github.com/rust-lang/rust/blob/1.85.1/library/alloc/src/collections/btree/node.rs#L318),
[mem.rs:18–32](https://github.com/rust-lang/rust/blob/1.85.1/library/alloc/src/collections/btree/mem.rs#L18),
[navigate.rs:182–233,362–371,459–475,510–523,595–601,630–637,711–718](https://github.com/rust-lang/rust/blob/1.85.1/library/alloc/src/collections/btree/navigate.rs#L182).
Combining the partial component above yields CANDIDATE terminal
`BODY ≤ 461 + 128N + 8A + 452M + 390O`; external caller
`≤ 462 + 128N + 8A + 452M + 390O`.
Candidate coefficients are k0=461 BODY / 462 external, kv=128, ka=8, km=452, kc=390.
Count caller Call ONCE, not another invocation if already priced in ENTRY; no CLEANUP refund or re-debit follows.
461 is not ENTRY, full funding or physical-custody proof. ENTRY, opaque constructor,
lock/retry, whole-Q and adoption funding remain OPEN; no adopted tariff, Rust,
native/security approval or ordinary resource change follows.
Imported cleanup covers source/raw payload and two embedded charges/Budget-Arc tails ONCE. Context callback, outer Context-root Arc/root charge, working/controller charges and abandonment/caller guards remain DISTINCT cuts, not included by assumption. No aggregate ENTRY/Rfailure/Q follows until normalization compatibility and nonoverlapping ownership cuts are proved; frozen C=462 and all coefficients remain unchanged.
Calculation is not adoption: both candidate normalizations and this fixed component remain proposals,
without refunds or physical bounds; import + ENTRY + CLEANUP stays prefunded before
handoff, seeded once without a second debit. All security/execution holds remain.
Adoption/authoring still requires approval of the terminal-retirement tariff,
event decomposition, opaque boundaries, pinned closure and funding. No runtime
work is authorized; prior prefunding, single-seed and nonrefundable obligations stand.

## Funding and execution gates

The conditionally cleared 527 helper allowance and seven successful `usize` slots
retain only their narrow review scope, not whole-prefix coverage or a D199 body.
New class/adopter, phase guard, full owner layouts, Result/error/Python-exception
publication, moves and failure overlap remain **UNPRICED**, as do constructor
original-root issuance entry/upstream bootstrap and whole CHECK caller/debit/helper
invocation. No complete numeric quote is established or historical tariffs copied.
Before any future run, require a fresh qualified guard, exclusive worker and slot,
exact command and source snapshot, plus 24 GiB available-estimate launch headroom
(ongoing reserve 16 GiB; aggregate RSS cap 8 GiB; zero actual swapout/swap-used growth).
The owned Darwin estimate is `(free_count + inactive_count) * kernel_page_bytes`,
source `pllm.darwin.HOST_VM_INFO64.rev1.free_count+inactive_count.v1`, bounded by
checked `hw.memsize`. This matches psutil Darwin's formula, not a psutil memory
result; its version describes separately observed swap accounting only. Mach free
already includes speculative: add neither speculative, purgeable, file-backed nor
compressor pages. Raw-unused `free_bytes` remains `(free_count - speculative_count)
* kernel_page_bytes`, equal to vm_stat's displayed `Pages free` (no double subtraction).
Historical 64/66 MiB readings remain valid denials under the former FREE-only policy,
not diagnoses of available capacity. The estimate is no allocation reservation or
no-swap guarantee. Prior campaign/source qualification cannot be reused; fresh review
is required. Approximately 12.3 GiB estimated headroom still fails the 24 GiB launch gate.

### Current alternate-recovery design selection — policy only, no action grant

User73433 selected independently witnessed host restart and trusted launch freeze,
based on Astra report 8 (`ses_e7564497dffeHBmqIE0aeVLQFG`). User73465 explicitly
approved Astra14's prospective-accounting policy amendment: the historical campaign
is administratively **FAILED**, accounting **UNKNOWN**, with its entire remaining
entitlement extinguished: no refund, reuse, reset or deadline extension. Forfeiture
does not invent a consumption amount or establish historical ≤20 GiB compliance.
Original campaign, slot and budget records remain unchanged; historical cleanup
and compliance remain **UNKNOWN**, and canonical native-command slot **HOLD**.

Separately authorized prospective **per-campaign** accounting replaces mandatory
reconstruction or carry-forward of unknown historical growth; it makes no aggregate
historical-plus-future compliance claim. Current disk/headroom observations cannot
reconstruct past growth. No evidence has been supplied or boot qualification
asserted. Post-barrier local liveness cannot establish historical cleanup or
compliance.

The proposal must bind trusted host identity, pre/post-boot evidence and kernel
replacement chronology, excluding suspend/resume, restoration and surviving
execution domains. Trusted continuous admission freeze must cover relevant launchers,
schedulers and resumptions. Separately authorized read-only evidence must bind
canonical Git ancestry, slot, campaign, original budget, root manifest and writable
roots. Full independently witnessed restart, that continuous freeze and separately
approved exact retirement remain prerequisites for any new campaign. Recovery
operations require explicit finite funding and an exact grant; neither the old
unknown balance nor a future campaign allowance implicitly funds recovery.
Unidentified or unbounded recovery gaps remain blocking.

Caps remain unchanged: 8 GiB aggregate RSS, 24 GiB launch / 16 GiB ongoing
headroom, zero swapout/swap-used growth, 600 s, 20 GiB cumulative positive disk
growth per prospective campaign, and exclusive canonical slot. Any new campaign
requires its own separately approved finite allowance and qualified admission
under all these caps. Subsequent Astra policy review must freeze a feasible
proposal before any separate evidence request; funded feasibility and tooling
are not established. Collection, restart,
retirement and tests each require separate approval; every future grant must name
operator, host, scope, deadline, failure-HOLD behavior and exact permitted state.
Only this documentation policy record is approved: no collection, restart, signal,
slot or budget operational edit, retirement or test is authorized, nor any
filesystem scan or source implementation. This design neither
describes current `command_guard` semantics nor authorizes the current retirement
path; it introduces no recovery CLI command or tariff.

Command qualification (parent-reported isolated mock run): after source/isolation
review, the parent alone executed the approved exact 30-node selection under a
mock-only headroom exception: **225 passed in 0.18 s** (14 host functions: 89 cases;
16 guard functions: 136 cases), no selected node omitted. The same Astra reviewer
corrected the earlier 234 forecast as arithmetic error. All four source hashes
were unchanged before/after. One expected unknown `asyncio_default_fixture_loop_scope`
configuration warning followed disabled plugin autoload. Ordinary logs:
`target/headroom-mocks-pytest.log` and
`target/headroom-mocks-{before,after}-{vm-stat,swapusage}.log`.
`time -l` reported peak pytest RSS 58,785,792 bytes (56.0625 MiB), real 0.36 s.
OS Swapouts were 104373321 before/after; swap-used displayed 501.12M both times,
not exact-byte zero-growth proof. This was observed-only, not active aggregate-RSS
enforcement; snapshots could miss transient swap-used growth. The narrow mock-test
grant did not authorize RAW-prepayment authoring or qualify native/THOR, live Mach,
a whole guard campaign or whole GKM, and authorizes no further tests. Ordinary
24/16 GiB headroom, 8 GiB RSS and zero-swap-growth policy remains unchanged.

Historical diagnostic-mock R2 snapshot: **40 passed, 257 deselected**, exit 0,
0.12 s pytest / 0.24 s wall; peak RSS 61,014,016 bytes. Kernel pages were
16,384 bytes; Swapouts stayed 104373321 and swap-used stayed 501.12M at displayed
precision. Before/after SHA-256 matched: guard
`28049e985a25e5579d717e032a86e10adc7c6fe4efb5e8e6dde3f98126af8ac3`, test
`ebab834de509f739f3a2f6e4e916d8c1a7880a5f25e3b0fbba848b075f224933`.
Five ordinary R2 logs: `target/guard-diagnostic-mocks-r2-pytest.log`,
`target/guard-diagnostic-mocks-r2-before-vm-stat.log`,
`target/guard-diagnostic-mocks-r2-before-swapusage.log`,
`target/guard-diagnostic-mocks-r2-after-vm-stat.log`, and
`target/guard-diagnostic-mocks-r2-after-swapusage.log`.
The preceding diagnostic run remains historical **39 passed / 1 failed**: its
generic `caller` secrecy oracle collided with public `caller_HOLD`; only that test
was repaired, not production or the original failure record. The earlier 225-case
evidence applies only to its old snapshot, not this source. R2's below-headroom
mock exception was observed-only: no active aggregate-RSS enforcement, transient
swap or global-disk guarantee. Native/GKM 41 remain unrun; canonical HOLD and
unknown historical cleanup remain unchanged. Diagnostics are source-reviewed;
ordinary blocking-sink capture remains unresolved, not qualified by mock capture.

Historical diagnostic-mock R3 snapshot (independently verified by parent):
**51 passed, 257 deselected**, exit 0; 0.13 s pytest / 0.27 s wall; peak RSS
61,997,056 bytes (59.125 MiB). Kernel pages were 16,384 bytes; Swapouts stayed
104373321 and swap-used stayed 501.12M at displayed precision. One pytest unknown
`asyncio_default_fixture_loop_scope` warning came from plugin-disabled
configuration, not a product regression. Frozen before/after SHA-256: guard
`e62a8786e1914d2061635ada70d182a51b72c50d4fdd4b92253b10234912c92c`, tests
`1c2ea3b4087d2f0ee2db8fcb129e146b52ae035cf50929ab034b6211f8cc6bcc`.
Five ordinary logs: `target/guard-diagnostic-mocks-r3-pytest.log`,
`target/guard-diagnostic-mocks-r3-before-vm-stat.log`,
`target/guard-diagnostic-mocks-r3-before-swapusage.log`,
`target/guard-diagnostic-mocks-r3-after-vm-stat.log`, and
`target/guard-diagnostic-mocks-r3-after-swapusage.log`.
All 40 prior plus 11 new mocks cover static failed S/A/F attribution only in the
existing Outcome reason; no new emitter, blocking-sink capture or cleanup authority
is established. R1/R2/225 results remain historical and snapshot-scoped. R3 was
observed-only: no active aggregate-RSS enforcement, transient-swap/global-disk
guarantee or separate startup bound. Native/GKM 41 remain unrun; canonical HOLD and
unknown historical cleanup are unchanged; no campaign retry or recovery occurred.

This plan grants no native qualification, authoring, tests, native execution or
activation; correctness, security and whole-Q funding gates remain unclosed.

### Astra14 guard reporting source repair — authoring record and parent-audited R4

USER73115 explicitly approved only best-effort guard reporting repair and mock
regression authoring in `scripts/command_guard.py`, `tests/test_command_guard.py`
and this brief record. Existing edits are preserved. Only `main` (four sites) and
`telemetry_worker` (one exception site) change production behavior: status/error
reporting uses the existing `diagnostic` helper. Exception details are deliberately
omitted rather than invoking arbitrary exception formatting; terminal Outcome
reason and R3 phase labels remain intact. The helper is unchanged: at most one
512-byte write, only to an already-nonblocking FIFO/socket; loss/truncation is
accepted, with no retries, flag changes, new owners or drains. Worker stdout wire
records/reaping, ordinary child output, argparse, cleanup, return codes, deadlines,
slot policy and caps (600 s, 8 GiB RSS, 24/16 GiB headroom, zero swap growth,
20 GiB disk growth) remain unchanged. No lossless capture or whole-CLI wall-time
bound is claimed. Canonical HOLD/cleanup UNKNOWN remain; no recovery authority.

New exact test nodes under `tests/test_command_guard.py::`:

- `test_main_reporting_never_uses_blocking_textio` with suffixes
  `[context-delivered]`, `[context-lost]`, `[argv-delivered]`, `[argv-lost]`,
  `[control-delivered]`, `[control-lost]`, `[success-delivered]`, `[success-lost]`,
  `[hold-A-delivered]`, `[hold-A-lost]`, `[hold-F-delivered]`, `[hold-F-lost]`.
- `test_worker_failure_reporting_never_uses_blocking_textio[delivered]` and
  `test_worker_failure_reporting_never_uses_blocking_textio[lost]`.

Static expansion only: 12 main + 2 worker + 2 existing blocking-sink + existing
emitter 6 unsupported-sink / 6 bounded-record / 4 write-failure / 2 metadata-failure
= **34 cases across seven functions**. Strict TextIO fakes reject writes/flushes;
capture stubs check one reporting call, loss acceptance, reason/phase, return codes
and modeled custody. No real-process/socket/OS-probe fixture is added. Existing
emitter regressions supply the unchanged synthetic low-level boundary checks.

Pre-run frozen selection, **then UNRUN and awaiting separate execution approval**;
the exact command below was subsequently run once as R4, recorded below.
Working directory: `/Users/blairhudson/workspace/pllm`.

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -l .venv/bin/python -B -m pytest --noconftest -p no:cacheprovider -o addopts= tests/test_command_guard.py -k 'test_main_reporting_never_uses_blocking_textio or test_worker_failure_reporting_never_uses_blocking_textio or test_diagnostic_blocking_sink_skipped_without_changing_control_or_cleanup or test_diagnostic_emitter_' -q
```

Astra review 25 source-cleared the five reporting calls and 14 new cases, but did
not clear the prior bare-`python` selection's isolation. Corrected isolation and
selection then awaited re-review; **34 was a static count, not a pass result**. Plugin
autoload suppression alone is insufficient: conftest imports uvicorn/runtime,
so `--noconftest` is mandatory. Interpreter binding and isolation options above
come from recovered historical R3 author-review message 38850 (reviewer
e76a7286, 48); only the new exact OR selection is retained, not R3's broad `-k`.
The command binds the repository-root `.venv/bin/python`, not arbitrary PATH
`python`. This historical binding is not current-environment or native proof;
At that pre-run stage, interpreter identity required read-only confirmation under a future grant.
No installed matching wheel is claimed, and no new environment or runner is added.

Any future separate execution grant must require BEFORE/AFTER SHA-256 of guard,
tests and this document, plus ordinary `/usr/bin/vm_stat` and
`/usr/sbin/sysctl vm.swapusage` observations. `/usr/bin/time -l` supplies process
peak RSS only. Reject any observed breach of 8 GiB RSS or zero swap growth;
these checks are observed-only, not aggregate enforcement or proof against
transient growth. Unchanged caps remain 600 s, 24/16 GiB headroom and 20 GiB disk
growth. Native HOLD/cleanup UNKNOWN and the canonical slot remain untouched;
diagnostic repair grants no slot recovery. Guard status-report loss is accepted;
ordinary child blocking-log capture remains unresolved. No custom evidence bundle.

During the pre-run authoring/correction above, no execution, tests, collection,
imports, builds, probes or slot access occurred. That correction changed only this document; production and test bytes remained
unchanged. Only this document's SHA-256 and full-file `git diff --check` are allowed
for this freeze; its hash belongs in the handoff, not a self-referential record. Historical
R3 **51 passed** remains solely its old hash snapshot, not evidence for this repair.
STOP EDIT after freeze; parent Astra reviews before any separate execution grant.

**CURRENT — R4 approved mock gate completed; parent directly audited.** USER73161
approved one exact 34-case pure-mock run of the command above. Astra review 28
cleared corrected command/source isolation conditional on current startup/site;
startup source was checked and root `.venv` CPython 3.13.15 identity confirmed.
The sole executor finished once, exit **0**, no retry: **34 passed, 288 deselected**,
one unknown `asyncio_default_fixture_loop_scope` configuration warning; pytest
**0.12 s**, wall **0.25 s**. Darwin process peak RSS was **64,258,048 BYTES**.
Raw Swapouts **104373321 → 104373321**; swap used **477.12M → 477.12M**, unchanged
only at displayed precision. Parent audited ordinary output
`target/command-guard-reporting-r4-mocks.log` and seven companion logs.
BEFORE/AFTER SHA-256 matched this **TESTED SNAPSHOT**:
```text
70ef6adc389d33a40bff622977d8f34897f526ce6e5ef74a0d851843f2ec997a  scripts/command_guard.py
2ef75e9c22b49982f8309a4a43a380ef92ffba01edad09ca1f378a1ad103a571  tests/test_command_guard.py
396c6675df3f17ab2200a19da9023a04582fda910c33b24a52f9ab69716bf521  docs/plans/raw-check-prepayment-contract.md
```
Guard/tests remain unchanged and frozen. This post-result document revision was
NOT tested; R3 is historical, not this repair's proof. Acceptance covers only the
approved mock gate: observed-only RSS/swap checks, no aggregate enforcement or
transient-growth proof, lossless child capture, whole-CLI wall-time guarantee or
native qualification. No slot access/recovery occurred; HOLD/cleanup UNKNOWN and
all caps remain unchanged. No launch authority or execution grant remains.

### Astra66 correction — proposed BF16 numeric gate (NOT AUTHORIZED / UNRUN)

The prior Maverick HTTP/bit-classification mock recommendation was wrong for this
BF16 repair. Neither that recommendation nor the historical 34-pass guard-mock
record above qualifies RMS repair or authorizes numeric execution. Historical
results and source hashes remain snapshot-scoped; native HOLD/cleanup UNKNOWN and
all canonical cleanup/resource-guard requirements and caps remain unchanged.

Plain read-only inspection of `tests/test_semantic_numeric.py:131–228` confirms
eight production-oracle functions, statically expanding to **4+4+2+1+2+3+1+1=18**
cases. These are real BF16 numeric checks, not HTTP/bit-classification mocks:
reduction overflow, epsilon-addition overflow, mixed-batch rejection, finite-lane
acceptance, local exception translation, scalar-conversion overflow and exact
huge-integer unit-ratio acceptance. This count is not collection or a pass result.

**Exact requested command — PROPOSAL ONLY, NOT A GRANT; DO NOT RUN:**
Working directory: `/Users/blairhudson/workspace/pllm`.

```sh
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 /Users/blairhudson/workspace/pllm/.venv/bin/python -I -B -m pytest -c /dev/null --noconftest -p no:cacheprovider --rootdir=/Users/blairhudson/workspace/pllm -q \
  /Users/blairhudson/workspace/pllm/tests/test_semantic_numeric.py::test_bfloat16_rms_norm_rejects_float32_reduction_overflow \
  /Users/blairhudson/workspace/pllm/tests/test_semantic_numeric.py::test_bfloat16_rms_norm_rejects_float32_epsilon_addition_overflow \
  /Users/blairhudson/workspace/pllm/tests/test_semantic_numeric.py::test_bfloat16_rms_norm_rejects_one_overflowing_row_in_mixed_batch \
  /Users/blairhudson/workspace/pllm/tests/test_semantic_numeric.py::test_bfloat16_rms_norm_accepts_large_finite_single_lane \
  /Users/blairhudson/workspace/pllm/tests/test_semantic_numeric.py::test_local_bfloat16_rms_norm_translates_variance_overflow \
  /Users/blairhudson/workspace/pllm/tests/test_semantic_numeric.py::test_bfloat16_scale_translates_scalar_conversion_overflow \
  /Users/blairhudson/workspace/pllm/tests/test_semantic_numeric.py::test_local_bfloat16_scale_translates_scalar_conversion_overflow \
  /Users/blairhudson/workspace/pllm/tests/test_semantic_numeric.py::test_bfloat16_scale_accepts_huge_exact_integer_unit_ratio
```

Module imports precede fixtures: `ml_dtypes`, NumPy, Torch, `pllm.modeling`,
`pllm.runtime.semantic_executor`, `pllm.runtime.semantic_numeric` and
`pllm.runtime.transformer_client` are imported even for this narrow selection.
Runtime/native paths reach the lazy `_native_support` loader; importing `native.py`
does not itself automatically load `_native`. NumPy includes compiled code, not
"pure Python". `--noconftest` does not remove module dependencies; `-I` neither
disables site startup nor proves imports resolve to this checkout. Absolute test
paths and `--rootdir` do not establish production-import provenance.

Before any startup or numeric execution, obtain explicit parent authorization
binding the exact current worker, source snapshot, interpreter, site/startup,
imported production modules, test/dependency closure and native identity (including
whether/what is loaded). Satisfy canonical cleanup/resource-guard requirements and
all prior-authorization prerequisites before launch. Historical hashes, installed
wheel assumptions and the prior guard-mock exception cannot substitute for current
qualification. No current interpreter/site/native identity is asserted here.

Source-isolation test splitting would require separate approval and preservation
of all 18 production oracles, including real local-runtime exception translation;
no fake semantic module or weakened numeric oracle is an acceptable substitute.
No benchmark, build, recovery, split approval, test collection or Python startup
is inferred. No venv probing, process/slot inspection, environment dump, receipt
archive or custom verifier is proposed as a completion gate.

This correction changes only this plan; research documentation, source and tests
remain untouched by this authoring task. No tests, collection, imports, startup,
builds, network or runtime probes were performed. Freeze checks are scoped
`git diff --check` and this file's SHA-256 only; hash belongs in the handoff, not
this self-referential record. STOP EDIT after freeze; parent review and a separate
execution grant remain required. Requested 18-case gate remains NOT AUTHORIZED/UNRUN.

### Current source-only native prerequisite approval — User73658

User73658 extends the retained-seal prerequisite approval to SOURCE-ONLY inventory
and design of shared native Builder/factory internal custody and bound gamma/rotary
parameter lifetimes. Sol may implement only independently design-cleared stages
and author normal regressions; this is not blanket implementation clearance.
Preserve consuming APIs, arithmetic, Program identity, PRSS, original one-shot
permission, debits and all caps. No fresh budget, ticket adoption or digest authority
is granted. No test execution, builds, funding, native execution, probes, recovery,
writer-close change, Directory implementation or native HOLD clearance is approved.
Historical cleanup remains UNKNOWN and the canonical slot remains HOLD. This
source-only approval does not qualify historical results or authorize execution;
new regression and documentation review remain separate gates.

User73707 explicitly approves additive `ResidentAdmission<T>` / `AdmissionAttempt` source implementation and regression authoring in shared allocation, following Astra15/36 conditional independent 134+5 SOURCE-DESIGN clearance of the critical moves; independent production and test/oracle review remains pending. Private Empty/Pending/Admitted/Sealed custody retains the actual original-counter guard across abandoned exclusive borrowed attempts, with exact checked entry quotation, positive-only inline admission and prechecked whole-payload sealing through `enter_into`, `admit_inline` and `seal_into`; this is budget-reservation proof, not external backing custody, runtime proof or a global attempt/burn restriction. The caller must retire backing before the resident guard eventually drops and independently enforce sticky attempt policy; Pending capacity retry remains on that same counter only. Existing consuming and split-into APIs, payload-before-charge output field order, arithmetic, original debits and caps are preserved; no counter clone, fresh budget or adoption is introduced. New regressions and compile-fail examples are authored only, all UNRUN; source-norm files and frozen split-into regressions are untouched. Wrapper/state/handle/Option storage, move overlap and cumulative-work prepayment remain UNFUNDED and unquoted. This approval grants no execution, funding, native activation, recovery or HOLD change; canonical cleanup UNKNOWN and HOLD remain unchanged.

User73719 explicitly approves the additive resident parameter-custody SOURCE API and normal regression authoring, following Astra41 and independent13 conditional design clearance. Complete caller-owned full/tail sources move into an already empty resident before bind/price/admit; original owned rotary Vecs/scalars remain retained, RMS gamma remains borrowed, and builds borrow those same immutable parameters. Shared cores preserve consuming legacy APIs, original ledger/debits/caps/numerics and error precedence, including native errors before compiler postchecks; numeric summaries confer no permission, and there is no fresh ledger, ticket adoption or digest authority. First acquisition/source-slot population, workspace/result/whole-Builder custody, partial results, wrapper/overlap funding, cleanup and broader factory integration remain OPEN; caller backing-retirement duties survive. Regression/doctest examples are AUTHOR-ONLY/UNRUN, and independent production and oracle review remain required. No execution, builds, funding, probes, recovery, writer-close/Directory implementation, native activation or HOLD clearance is granted; historical cleanup UNKNOWN and canonical HOLD remain unchanged.

User73860 approves staged SOURCE authoring under the Astra163 packet: Stage1 is a generic schema emitter preserving existing configuration structs, visibility, Serde/default/codec contracts and generated private field IDs; independent Stage1 production and oracle review is required BEFORE Stage2 private borrowed diagnostic facts. Existing validation/lowering/numeric/default/codec bodies, consuming APIs, original ledgers, caps and error policies remain unchanged. Regression and macro/doctest fixtures are AUTHOR-ONLY/UNRUN, not compiled-macro or parity evidence. Inline layout/prepayment remains UNFUNDED; no no-heap-is-free claim, execution, funding, cleanup, recovery or HOLD clearance follows. Historical cleanup UNKNOWN and canonical HOLD remain unchanged.

The current user approval, following independent Astra review14 SOURCE-DESIGN clearance, authorizes only private rotary finalization custody repair and regression authoring in CORE PUMA. Existing `ResidentNativeSchedule::build_into` dispatches matching rotary construction/parameter variants through one shared geometry → Builder → coefficient-validation → recipe preparation helper, then immediately installs the returned actual Builder in independent inline full/tail resident finish slots before existing resident finish/transfer APIs. Legacy `Program::rotary_half` still consumes ordinary finish; nonrotary/mixed paths remain supported. Admitted → Burned → occupied-output refusal → construction order, immutable original parameter/count binding, original debit/caps, numeric contracts and full-output installation before tail are preserved. Successful Programs pass through resident Program slots into caller outputs with `ProducedCertificate::None`; no hook, assertion, callback, allocation or fallible check intervenes after taking an actual Program and before output installation. Ten new regressions are AUTHOR-ONLY/ALL UNRUN: installed two-table identity, real in-finish panic, explicitly private malformed-output error, failed/completed refusal matrix, occupied-output burn, second-finish tail panic, occupied/empty transfer controls, legacy error/instrumentation parity, full/tail parity and nonrotary/mixed controls. Existing 20 Builder and 23 parameter oracles are not rewritten; shared-helper comparisons are not independent numeric/runtime/PRSS evidence. Two added inline slots enlarge the owner: wrapper layout/prepayment, temporary/result overlap and original native parameter backing remain UNFUNDED and require review before composition. Prepare errors/unwinds BEFORE installation, recipe-local pending affine acquisition and first-table loss during second-table failure remain OPEN, as do broader native/whole-attempt custody and cleanup; no funded-capacity, cap-fit or full-custody claim follows. Abort/OOM termination/double panic are excluded. No execution, build, funding, activation, recovery or HOLD clearance is granted; historical cleanup UNKNOWN and canonical HOLD remain unchanged. Freeze STOP EDIT for independent PRODUCTION and ORACLE reviews.

### User73970 — source-only shared fixed-error preflight extraction

User73970 explicitly approves SOURCE-ONLY extraction and regression authoring in
`crates/pllm-models/src/source_budget.rs`, `source_mappings.rs`, and dedicated
private child `source_fixed_preflight_tests.rs`, with narrow module wiring and
this append-only approval record. The preceding 937 lines remain frozen.
Stage1 schema files (15 unit tests and 11 doctests), existing tests, rotary
production/frozen files, other dirty work and paper coverage remain untouched.

Astra136's read-only transitive-closure inventory identifies only Qwen2, Qwen3
and Llama as currently eligible, all Dense LastToken. Shared private eligibility
retains first registry match identity and factory-before-contract availability
order; it invokes neither decoding/lowering factories nor preflight callbacks.
The existing legacy boundary then checks the selected preflight before returning
the factory. Fixed codes are `NoAdmittedFactory`, `NoExpansionContract` and
`InvalidExpansion`; only legacy rendering allocates the existing constant-message
ModelError strings. Dense fixed checking borrows the document and uses stack
values/checked arithmetic. `SourcePreflight::Other` remains the original legacy
callback adapter and is NOT claimed transitively allocation-free.

The existing dense scalar summary body, scalar/shape/default/Serde behavior and
arithmetic remain unchanged. One shared zero/ceiling predicate serves fixed and
legacy adapters. Complete summary precedes caps: layers, hidden, intermediate,
attention heads, KV heads, head dimension, vocabulary, query width, KV width,
then all seven shape elements in existing array order. FullPrefix and LastToken
remain distinct. The shared InvalidExpansion code cannot prove interior error
precedence; that order is preserved by source inspection, not an observer test.

Eighteen new unit regressions are AUTHOR-ONLY/ALL UNRUN. Literal error variants,
messages and independent geometry expectations cover selection, synthetic missing
contracts, first-match identity/order, no factory invocation, legacy Other,
scalar/type/overflow/caps, optional KV defaults, workload/capacity failures,
head-width behavior, both readouts and exact/one-short shape boundaries. These
are not compiled-macro, parity, runtime or allocation-instrumentation evidence.
Independent PRODUCTION and ORACLE review remains required before execution.

Future SourceFacts capture must use the owner's original-attempt
`PrivatePreparedConfiguration::prepay_policy` before selection projection;
imported `Resources.work` is not ledger credit. This core does not fund inline,
storage or work and does not close capture custody. No capture, descriptor,
registry linkage, compiler wrapper, schema emitter or new public API is approved.
No fresh ledger, ticket, budget, retry, debit or policy change is introduced.
No test/build/rustc/rustfmt/import/probe/native/RNG/network execution, cleanup,
index/commit/worktree/terminal change or funding is granted. Final source review
is limited to file hashes and scoped whitespace checks, including untracked
files against `/dev/null`; no formatter runs. Historical cleanup UNKNOWN and
canonical HOLD remain unchanged. Freeze STOP EDIT; no execution follows.

#### Fixed-core independent source review status

Astra production34 independently marks shared fixed-error selection and dense
preflight production SOURCE-CLEAR. Astra oracle50 independently marks the
18 new unit regression oracles SOURCE-CLEAR after the test-only resolved-KV-width
repair: the last fixture admits its independently calculated 8 × 8 = 64 feature
width, leaving the intended 512-accepted/511-rejected tensor boundary unmasked.
There are 0 new doctests; ALL 18 tests remain UNRUN, with no compiler proof.
Independent production and oracle reviews are complete for THIS source
prerequisite only, superseding the pending-review status above for this scope.

Bounded source review preserves legacy selection order and the exact three
variants/messages: `NoAdmittedFactory` renders Unsupported("source has no admitted
Value factory"), `NoExpansionContract` renders Unsupported("source has no bounded
expansion contract"), and `InvalidExpansion` renders InvalidConfig("decoder source
expansion budget exceeded or invalid"). Dense fixed-core allocation freedom is
source-inspection-only; legacy Other callbacks and ModelError rendering can
allocate. Existing caps, defaults, ledger, debits, Serde and numerics are unchanged.

Review exclusions remain explicit: Other has a failure-only oracle;
NoExpansionContract coverage is synthetic, not production reachability; a
production-wrapper factory counter is absent; shared InvalidExpansion cases do
not prove interior precedence. No full numeric, allocation, runtime, custody or
funding proof follows. Registry/source-owner controller, gate, bounds bridge and
capture are neither implemented nor authorized by this approval. Original-attempt
prepayment, numeric lookup bounds, layout and retention on consuming refusal
remain OPEN. Historical cleanup UNKNOWN and canonical HOLD remain unchanged.
This documentation-only status record grants no execution or further source
implementation. STOP EDIT for independent documentation review.

#### User74052 — PREFIX EXCEPTION policy for next design step only

User74052 explicitly chooses PREFIX EXCEPTION in response to User74049/Astra39's
policy question, for the NEXT DESIGN STEP ONLY: an explicitly UNPAID, one-shot,
metadata-only admission prefix on an already-established ORIGINAL root/controller,
with NO document access, retries or error rendering. This records a design policy,
not an implemented one-shot gate. Logical-work bounds remain TO FREEZE; numeric
logical tariffs are NOT adopted. No elapsed-time, physical-RSS, lock, allocator or
atomic-retry guarantee follows. This exception does NOT prepay existing root setup;
original-root establishment and cleanup funding remain OPEN. Existing byte and
controller charges are NOT work authority.

Original work ledger, attempt, caps, fee and legacy API remain unchanged. A new
opt-in reservation post-debit marker is allowed only as a DESIGN TOPIC: it mints
nothing and grants no fresh ledger, refund, retry or reset. Host150 remains a
DESIGN-ONLY inert, nonextractable aggregate, with no handoff or capture; numeric
layout is not compiler-verified, and old consuming configure Err/unwind custody
remains OPEN. Marker, host, controller, bridge, lookup and decode authoring are
NOT approved.

Neither policy choice authorizes implementation, tests, execution, cap increases
or HOLD recovery. No global unpaid-prefix permission extends to other methods or
recovery. No source authoring, factory activation, build, test, import, model,
native or numerical execution, or cleanup-state change is granted. Historical
cleanup UNKNOWN and HOLD remain unchanged. Existing 18 core tests remain UNRUN;
this section authors zero tests. STOP EDIT for independent narrow documentation
review.

##### 2026-10-30 — Setup-metadata design exception only

User approval, “Allow design exception”, adopts ONLY a finite, explicitly UNPAID
setup-metadata envelope for pre-debit routing, quotation and reservation controls.
This extends, but is distinct from, the existing unpaid prefix on an ALREADY
established original root. No counts, tariffs, allowance value, admission, custody,
funding, capacity or physical bound are adopted. Astra's separate bounds freeze
is not yet adopted here. No global exception extends to other methods.

Existing root construction (`try_build`, `OnceLock`, `Mutex`, new-root atomics and
`Arc` allocation), payload processing and ALL cleanup remain OUTSIDE this exception
and OPEN; unknown cleanup is not free. Setup quotation or a coverage check proves
neither payload ownership nor payment. The first initial debit cannot pay for
earlier root construction retrospectively.

The integrated private install-into design is conditionally coherent, not authorized
source: destination occupied check → eligibility → admission/phase identity →
retained working budget plus coverage → legacy prepare → infallible install.
Preparation and retained-working charge owners are distinct and must belong to
the same original-root budget. Root setup,
pre-success custody, payment attribution, cleanup and storage values remain OPEN;
this does not settle capture implementation.

No new attempt, ledger or controller, repeat fee, retry, refund, cap increase,
configure handoff or bridge implementation is authorized. No implementation,
tests or execution approval follows. Existing 18 fixed-preflight tests remain ALL
UNRUN; HOLD and historical cleanup UNKNOWN remain unchanged. Documentation only;
STOP EDIT for parent independent review.
