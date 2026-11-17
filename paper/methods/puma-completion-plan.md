# PUMA common-Qwen completion plan

Status: approved for implementation, including the protected checked-range
profile, with Sol 6.1 subagents on independent workstreams. Baseline: `5dd11004` and
[guarded-activation validation](../../docs/research/puma-truncation-reference.md#guarded-piecewise-activations).
Completion still requires the evidence gates below.

## 1. Concrete finish line

Finish the D11 **public-weight** adaptation on the pinned
`Qwen/Qwen2.5-0.5B-Instruct` source used by `examples/papers/common.py`:

1. Admit the complete 24-layer decoder, full vocabulary and all eight selections
   under the shared **64-input/eight-output envelope**. Source-rendered ordinary
   chat with at most 32 inputs/two outputs is the first execution gate, not the
   completion workload. Historical 4+2 calculations below are not chat fixtures.
2. Execute through the ordinary three-worker SDK, gateway and CLI. Retain D6/D10
   Q18 arithmetic, D9 full-prefix recomputation, private feedback, stable argmax
   and final client-only token IDs.
3. Establish independent full-decoder integer correctness, including stochastic
   truncation, and failure/cancellation retirement. Final-token agreement alone
   does not establish hidden-tensor correctness.
4. Run the shared paper inputs through ordinary quality and response benchmarks,
   with three fresh-process performance replicates and complete role costs.
   Report quality failures and poor performance as results.
5. Publish the exact Experiment, build, source, workload and accepted/rejected
   counts with reusable experiments and ordinary benchmark output.

This qualifies an implemented, measured source-derived public-weight adaptation.
Private-weight full-Qwen capacity, the unavailable original evaluated graph,
representative task quality and independent cryptographic review have separate
qualification requirements. They cannot be inferred from these five checks.

## 2. Audited baseline blockers, including those after RMSNorm

| Blocker | Evidence / implementation at `5dd11004` | Required outcome |
| --- | --- | --- |
| Numeric admission | Public 4+2 source publication rejects selection 0, layer 1, step 29, RMSNorm; propagated Q18 residual interval is `[-343639410012, 343639400796]`. No worker starts. | One complete-model range strategy, validated across every layer and selection. |
| Lookup work | `public_lookup_work_floor` in `replicated_decoder/program.rs` prices 1,187 128-lane predicates per token, each using 28 ordinals. | Reduce actual execution costs and establish an admitted complete-program budget. |
| Frontend capacity | `replicated_decoder/sdk.rs` permits at most 32 input tokens; `lower_with_weights` permits at most 40 input-plus-output tokens. | Price and validate the shared 64+8 envelope across SDK, compiler, workers and memory admission. |
| Native capacity | 1,048,576 complete-program ordinals, 3,224,371,200 per-party AES blocks, 65,536 nodes, 64 MiB program storage and 1,048,576 values per secret owner are separate limits. | Check all limits, not only the first rejection. Every capacity change needs its own priced, tested envelope. |
| Weight execution | `public_source::multiply` scans immutable i64 weights with a 64 KiB buffer. Lookup invokes it separately for each token. | Bound and measure public file reads, both replicated-component products and live output owners. |
| Peer execution | `replicated_worker.py::run` uses HTTP advance/prepare/dispatch barriers; there is a 1,200-second attempt lifetime and a separate barrier-loop ceiling. | Bounded peer scheduling whose overhead and lifetime fit the intended request. Ordinals are not HTTP calls or network rounds. |
| Paper recipe | `examples/papers/puma.py` currently selects the default private-weight protocol with the shared 64+8 budget. | Explicitly select the qualified public-weight composition in the paper factory. |
| Correctness and measurement | Generated SDK controls pass; no complete pinned-Qwen execution, hidden-tensor oracle or matched response result exists. | Finish the full-source conformance and ordinary benchmark gates below. |

### Lookup-only cost calculation

The baseline schedule processes `sum(input_rows + selection)` lookup rows under
D9. With vocabulary 151,936 and width 896:

```text
tiles_per_lookup = 151936 / 128 = 1187
ordinals_per_lookup = 1187 * 28 = 33236
table_bytes = 151936 * 896 * 8 = 1089077248
```

| Geometry | Recomputed lookup rows | Lookup ordinals | Current ordinal ceiling multiple | Logical lookup file reads per worker |
| --- | ---: | ---: | ---: | ---: |
| 4+2 | 9 | 299,124 | 0.285× | 9.80 GB |
| 16+2 | 33 | 1,096,788 | 1.046× | 35.94 GB |
| 64+8 | 540 | 17,947,440 | 17.116× | 588.10 GB |

The 64+8 row is a hypothetical geometry extension of the current implementation;
the current frontend rejects it earlier. These are schedule calculations, not
serving measurements. Logical file reads are not physical disk traffic or network
bytes. At 64+8 the lookup additionally entails 147,025,428,480 ring64 multiply-adds
per worker across its two replicated components, before decoder body or head work.
Reusing a file scan can reduce reads without removing those products.

### Implementation update

The explicit checked-range, tiled lookup/selection and persistent-peer composition
has a coordinated installed wheel. Bounded native guards preserve valid operands,
retain sticky shared validity and mask all final tokens on failure. Generated
integer/AES oracles cover complete tensors and private feedback. The source-only
pinned 4+2 attempt advances to layer-eight attention before rejecting its product
bound; no full-checkpoint worker execution follows from this attempt.

P0's [complete accounting](puma-feasibility.md) finds a **no-go under the current
complete-request ceilings**, including node/storage and protected-work limits.
Compact equality reduces a measured 128-lane predicate from 69,768 to 19,920
encoded bytes, but the complete 64+8 graph still fails its directed-traffic floor.
These figures are native primitive measurements and public schedule calculations,
not whole-response results. The original engineering estimate remains conditional
on a successful revised feasibility gate.

Remaining implementation work must address the complete arithmetic preconditions,
bounded body execution with real live-owner pricing, and repeated private
comparison/conversion work. Shared templates may reuse public instructions; they
must never reuse MPC material. Capacity extensions require a separately priced
and tested envelope, rather than treating a larger constant as qualification.

### November 17 bounded prerequisite update

The first trained milestone now uses source-rendered ordinary chat with at most
32 input tokens and two outputs. Historical 4+2 remains a geometry calculation,
not an achievable ordinary-chat fixture or a reduced completion workload. The
64+8 finish line above remains separate. No trained PUMA run is established.

Model-free validation on Rust 1.85.1, locked/offline, Cargo `-j1`, includes:

| Focused selector | Result | Evidence boundary |
| --- | --- | --- |
| `shared_byte_ingress` | 21 passed | Same-root ingress, exact/short admission, retained terminal/H capacity and configured success. Failure-tail controls include primitive/manual adapters, not complete actual-importer failure coverage. |
| `peer_batch_tests` | 5 passed | Exact frame-batch construction, ordering and owner cleanup. Does not establish producer/queue pretransition admission or total native capacity. |
| `raw_append_leaf` | 2 passed | Shared typed writer preserves observed raw bytes, digest, size and sticky Bounds/Closed behavior. Private `max_i8` parity is supported by core-local controls/shared bookkeeping, not a new compiler observer. No authenticated publisher fixture is exercised. |
| `bounded_storage_selector` | 2 passed | Typed selector preserves legacy results/error text and source-policy Binding parity. Removes one allocating policy-error leaf; additional caller funding remains unproved. |
| `stream_transport_tests` | 13 passed | Nonblocking transport, absolute deadlines, strict EOF/child exit, cancellation, primary-error precedence and retained cleanup/reap controls. Not live Worker/SetupParty lifecycle or complete native integration. |

A later frozen-source Rust 1.85.1 packet separately passed:

| Crate / focused selector | Result | Evidence boundary |
| --- | --- | --- |
| `pllm-core` / `puma::prss::multiply::tensor::tests::shared_tests::` | 19 passed | Twelve inactive predicate-prefix regressions and seven retained shared-tensor tests: indexed lineage, local binding, pre-randomness refusal and retained failure/unwind ownership. Not a complete SiLU recipe or compiler route. |
| `pllm-core` / `puma::prss::multiply::tensor::construction_tests::` | 15 passed | Construction-resource/preflight contracts, exact and one-short controls, and unchanged tested numeric/error/digest boundaries. Does not prove compiler caller funding. |
| `pllm-compiler` / `source_storage_capacity::tests::` | 5 passed | Inactive checked selector-byte and nominal-layout inventories, independently reconstructed ABI/site families and legacy selector parity. No runtime accessor, Geometry wiring or complete capacity certification. |
| `pllm-models` / `owned_backing::retirement::tests::` | 9 passed | Inactive String/plain-usize-sequence logical retirement equations, actual capacity observations, child/release rows, overflow and unchanged real owners. No runtime obligation, query sponsorship or allocator/Drop-instruction guarantee. |
| `pllm-compiler --lib` / `real_shapes_` | 3 passed | Nonempty ingress shapes retain original cover during untouched collector destruction and external-phase unwind; overflow refuses before a new debit. Custody observations, not prepaid retirement-work or complete importer-failure evidence. |

Use the matching crate and literal filter with the ordinary locked/offline,
single-job Cargo command; large debug tests use `RUST_MIN_STACK=8388608`.
The two `lookup_authentic_width6_` compiler tests instead both failed at
`compact_lookup_norm_tests.rs:550`: `authenticated_entry_unfunded` followed by
`Binding`, before native construction. This is retained funding-barrier evidence,
not passing caller compatibility or sufficient-funding validation. No bypass was
added. These Rust changes postdate the installed-wheel Python results below.

The retirement packet ran before the subsequent Models fixed-error and Core
resource-view authoring. The Models command ran nine unit tests, with zero matches
in its integration targets. An initial Compiler command without `--lib` failed
compiling the existing standalone source-norm harness; it ran no tests. A later
`--lib` filter using the test filename also selected zero tests, which is not a
pass. The corrected `--lib real_shapes_` invocation ran all three intended tests;
the caught external-phase unwind printed its expected panic. The standalone
harness import failure remains separate from these passing unit tests.

The inactive prefix also exposes a SOURCE-fee integration hold: the current
universal validation quotation is 806,701,248 visits per call, versus the previous
1,198,272. Existing compact RMS/rotary callers derive their fees from that quote;
the early entry failures do not exercise those debits. A separately priced,
sealed proof-free leaf is now authored, with caller query/control composition to
close. Neither the Core passes nor this inactive leaf establishes whole-request
SOURCE-work fit; existing caps remain unchanged.

Later frozen-source Rust 1.85.1 validation used `--lib -j1 --locked --offline`:

| Crate / selector | Result | Evidence boundary |
| --- | --- | --- |
| `pllm-models` / `bounded_` | 40 passed | Includes retained bounded-canonical tests and fixed-error/control-inventory regressions; not 40 new quote tests. |
| `pllm-models` / all library tests | 298 passed | Includes the focused 40 and earlier retirement tests; not an additional disjoint total. Preserves tested legacy prices, errors, refusal order and bounded allocation behavior. |
| `pllm-core` / `resource_view_` | 7 passed | Private resource-view lifecycle, resident results and sealed producer boundaries at this frozen revision; not compiler funding. |
| `pllm-core` / broad tensor filter | 256 selected: 254 passed, 2 failed | Overlaps the earlier seven resource-view tests; counts are not additive. Wide owned-RMS admission regressions remain unresolved; not a passing Core packet. |

Independent Models source re-review cleared both the local nominal-maximum
witness and independent complete owner/peak reconstruction. The zero-byte
witness changes neither combined control bytes nor combined layout maximum;
historical bootstrap slots do not pay for new bounded controls. This establishes
the inactive Models leaf, not a Compiler payer, Geometry composition or selected fit.

A subsequent eight-file Compiler const-only composition is authored and frozen.
Independent source reviews cleared both the Models/ingress/caller/Geometry union
and the literal policy/SDK census with independently reconstructed test oracles.
Models controls enter once, maxima compose by maximum, and the checked candidate
retains conditional F3425/G with terminal `T0 + ControlBytes`. Its five bounded
regressions now pass **5/5**, and the ingress-alias regression separately passes
**1/1**. The 64-bit ABI measured `U=1888`, `ControlBytes=629756` and conditional
`Gwork=338073913600`. These validate const metadata and independent reconstruction,
not selected fit, a runtime payer or Geometry activation.

Core has an authored resource view with a sealed handoff from the genuine
resident RMS producer. Its construction-maintenance allowance is 37,568,
replacing 37,248; QUERY 256 and BODY 1,198,656 remain unchanged. At the initial
inline-layout revision, both 1×1024 fixtures failed owned-byte Limits admission before Executor construction/work
admission: inline indexed-extension ownership increases `size_of::<Executor>()`;
one fixture reports 267,384 owned bytes against
the unchanged 262,144 ceiling. Separate source review found two omitted scalar
inventory slots and insufficiently independent RMS/softmax legacy expectations.
Those repairs are now authored; they are not validated by the earlier seven
passing tests. A subsequent frozen-source packet passed `resource_view_` **8/8**,
`puma::prss::multiply::tensor::construction_tests::` **15/15**, and
`puma::prss::multiply::tensor::tests::shared_tests::` **20/20**. These are 43
distinct tests; the separately passing single legacy-vector filter is already
included in the construction 15. The shared filter now selects 20, rather than
the historical 19. Expected injected unwinds were caught in passing tests.
The legacy-vector regression independently serializes both RMS routes and
certified softmax against pre-extension Git provenance; it does not establish
trained numeric acceptance. Subsequent complete source-local census found further
helper/receiver/branch control cells; their typed inventory repair was not
validated by this packet. A separate one-test native
pre-layout diagnostic measured Executor=22,976 bytes and both exact wide-RMS
graphs at 267,384 owned bytes.

After the sparse descriptor-owner repair, the same diagnostic passed with
IndexedOwners/Extension=616, Program/ProgramOwner=13,640 and Executor=16,848
bytes. Both actual wide-RMS graphs required **261,256 / 262,144 owned bytes**:
the measured 6,128-byte reduction is physical ownership reduction, not a cap
increase or accounting subtraction. A first compile attempt failed on a borrowed
trait-object lifetime; the narrow object-bound correction compiled successfully.
The subsequent broad tensor filter passed **261/261** in 88.77 seconds,
including the diagnostic, both previously failing wide-RMS execution tests,
resource-view and construction tests. These counts overlap earlier packets.
The separate shared `linked_rows_` filter passed **3/3**, including exhaustive
1..=384 allocation extents and caught partial-clone unwind.

Independent source review passed sparse ownership, proof binding and custody,
but initially withheld resource-census acceptance: the metadata borrowed view and
ExactBuffer acquisition/finish controls lacked explicit phase inventories. The
follow-up repair now includes the overlapping `IndexedWorkView`, separate checked
phase sums/maxima, and independent actual Record-length 1/384 reconstruction.
Fresh bounded source re-review passed; the post-repair broad tensor rerun passed
**261/261** in 88.69 seconds, and linked storage passed **3/3** separately. These
reruns overlap the earlier results, not additional disjoint tests. Both wide-RMS
graphs still measure 261,256 owned bytes under the unchanged 262,144-byte cap.
QUERY/BODY/maintenance and the universal validation fee remain unchanged.
Compiler migration, original query sponsorship, complete simultaneous capacity
and request fit remain open; none of this is trained Qwen execution evidence.

The inactive retirement-query metadata also passed independent source review.
After a test-only owner-import repair, the retirement filter passed **16/16**:
seven new query-metadata tests and the nine retained evaluator tests. A subsequent
full Models library run passed **307/307**, including those 16;
counts are not additive. This validates checked logical query equations and
typed layout metadata, not payment of an actual query, acquisition of the genuine
retirement owner, partial-tree custody or full publisher accounting.

The private initial-payer Stage S implementation is authored and frozen after
independent custody/identity and accounting reviews. Its caller-resident owner
retains admission, terminal and early coverage across successful return; bounded
install retires configured payload before its request grant on refusal. The final
accounting repair adds the inner `Arc<Budget>` dereference and both atomic-load
arguments to production and independent typed inventories. Aggregate is **214
units / F(1070)**; creation remains 212 and post-debit 62. G is **F(4805)**, first
debit **F(4879)+1**, seed **F(2218)**; W74 and legacy routing are unchanged.

The latest frozen `stage_s_` Compiler filter passed **23/23** (14 source-owner and
nine Geometry tests, including the raw-boundary diagnostic), with exact/one-short payment, real install refusal/unwind,
retained-success and foreign-identity controls. Actual 64-bit **Ue=1888** was
compiled independently, not assumed from the earlier U. Measured payload operands
are terminal **884,284**, root **22,947**, attempt **1,139** bytes; logical seed
**218,933,705,216**, G **474,290,556,160**, first debit **481,594,926,849**. These
logical work and layout figures are not runtime, RSS or selected-source admission.
Before the shared bootstrap tariff repairs, genuine initial entry reported
**Err(Binding)** under unchanged caps. A same-input, test-only boundary companion located the
first raw refusal before decoder error mapping: the second
`byte_ingress_working_bytes` call, for configuration, returns SOURCE **Work**
before configuration parsing. The original root holds **983,979,690,241** visits;
the next **133,849,589,760** debit would require **1,117,829,280,001** against the
unchanged **1,000,000,000,000** limit. Refusal leaves used visits unchanged and
burns that root. Earlier seed, first reservation, experiment working-byte query
and experiment import succeeded on that same root. This is a work-fit blocker,
not evidence for an authentication repair. This is historical refusal evidence,
not the current entry result. The older successful custody fixtures use explicitly
separate test sponsorship.
Independent arithmetic shows that removing only the second working-byte query
cannot complete configuration: its unchanged BODY alone costs **148,571,034,880**,
against **16,020,309,759** remaining visits after Experiment import, before byte
work and the parser allowance. The bootstrap layout is input-invariant, but any
reuse must retain separate real working reservations, live custody/burn/peak
checks and implementation-derived replacement control payments. Shared layout
reduction therefore required replacement accounting, not query reuse or a cap increase.
Independent review has now accepted the frozen three-file const-layout cut: share one
formula through legacy-runtime and const-safe macro expansions in Types/Models,
then return a checked constant from the existing JSON bootstrap accessor. Legacy
dynamic expansions must remain unchanged; both ingress queries/reservations and
all fees/capacity quotes remain. Six focused regressions passed with Rust 1.85.1;
the broader Types and Models library runs passed **51/51** and **307/307**.
These counts overlap the focused tests. This validates the shared runtime/const
formula expansions and checked propagation, not a tariff reduction or entry fit.
The separately reviewed Stage B cut retains both queries/reservations and adopts
QUERY/CHECK/BODY units **108/79/401**. Its repaired plan separately assigns both
peak checks and runtime helper occurrences. Native typed reconstruction rejected
the proposed mapper partition: two complete five-cell endpoint families require
**322 bytes**, exceeding the assigned **232-byte** array. The actual acquisition
tuple and FixedControl endpoint checks passed before that failure. The replacement
regression now passes **1/1**: four disjoint 168-byte blocks use **672** of the
retained census pool's **8,840** bytes, with whole-type alignment, containment and
residual checks. The failed 322 > 232 candidate remains an explicit counterexample;
acquire arguments and FixedControl selves keep their separate original arrays.
The subsequent frozen revision also passed **1/1**, now asserting all twenty
corresponding census/mapper cells and actual native adapter whole tuples. The
inferred `Error::Native` constructor item measures 0 bytes/alignment 1; its whole
native Result receiver tuple measures 40/8, and `(NativeError,)` measures 1/1,
meeting their separately assigned native cells. These repeated 1/1 results cover
successive revisions of the same test, not two distinct tests. Custody
assignments retain raw payload endpoints through preparation and transfer into
separately paid JSON arguments before census-pool reuse. Independent final source
review cleared bounded tariff authoring under the stated logical convention.
The adopted G7 / QUERY108 / CHECK79 / BODY401 cut passed **23/23** ingress tests;
ENTRY1147, BYTE, capacity inventories and limits remain unchanged.
The borrowed-resource pool remains unassigned. The broader shared
Compiler JSON filter passed **29/29**, including the two Stage A tests.
The next historical refusal moved to `consume_stage_s`: used **823,205,755,221**
plus BODY **254,369,773,824** exceeded the same 1e12 cap. A second shared extraction
then moved two fixed Models bootstrap layout Results to module-private constants,
preserving dynamic validation, error mappings and capacity inventories. Full Models
tests passed **309/309**; three added actual-constructor ABI tests passed separately.
Four ordinary/bounded Compiler constructor-item witnesses also passed their focused
test. Independent reviews accepted separate ordinary and bounded endpoint/capacity
proofs and each consumer's own nominal maximum.

The resulting reviewed shared tariff removes **607** runtime-expansion units and
adds **2** whole-Result read units: core **405**, early query **437**, body **1367**.
Outer adapter allowance **89**, body tail **557**, both paid queries, live reservations,
nominal maxima and all caps remain. Post-edit `original_source` passed **151/151**;
the subsequent strict `stage_s_` filter passed **23/23**, with 14 tests overlapping
(160 distinct tests across the two filters).
The genuine tiny one-layer fixture now completes `prepare_stage_s` and its raw
boundary companion on the same unburned original ledger: **922,355,896,592 /
1,000,000,000,000** work used. This proves bounded fixture entry, not 24-layer
Qwen entry, complete request fit, production routing or trained execution.
Strict success/no-reentry and genuine-return destruction regressions now pass in
that 23-test run. Independent source review accepted their actual-entry sponsorship,
same-root identity, occupied-owner refusal without retained mutation, payload →
Context → grant destruction, resident guard coverage and watcher-adjusted baseline
reconciliation. The diagnostic itself still permits stopping at the first raw refusal.
The Stage T constructor audit found an additional concrete integration gap:
`prepare_stage_s` retires its imported experiment on return, whereas the existing
`HeaderPrepared::stage` receives a Geometry owner made through another original
root. Combining those two owners would not establish same-root custody. The
successor-entry plan now chooses a T-only composed owner and private same-Context
prepaid token, rather than changing shared grant/Stage-S/header layouts. It retains
the experiment and initial S residence until header/token retirement; token mint
requires a new genuinely prepaid consuming preparation and successful unique install,
not retrofit of an already-issued S grant. Independent custody review accepted the
bounded design. The inactive declarations now compile: `PrepaidSuccessor` retains
the exact Context Arc, spent flag and charge; the composed `StageTOwner` retains
header, experiment, successor, phase and initial S residence in that destruction
order. Independent source review accepted this types-only cut. Existing shared
owners and the three-field `RawHeaderStage` are unchanged; only the header type's
crate visibility was widened. The subsequent standalone T ABI filter passed
**4/4**, covering actual owner layouts, declared header operands, padded argument
tuples and endpoint families; direct private header-field projections remain a
separate witness. A private, inactive `PrepaidSuccessor::spend` now compiles and
passed independent source review for spent-first ordering, root/Budget identity,
eligibility and live-ledger checks, and guard cleanup. Source review found zero
calls, function-item escapes and token constructions. Its SP00–SP11 census is
frozen; the actual error-mapper ABI and complete descriptor remain open. No
constructor, issuer, adopted T fee or caller exists. Complete T typed tariffs,
funded behavior and integration remain pending.
The tiny fixture's **77,644,103,408** residual
work is a necessary fit screen, not T funding or a full-request budget.
The earlier **105/105** original-source packet predates Stage S; the **148/148**
packet predates the two bootstrap tariff cuts. The latest frozen
`original_source` filter passed **152/152**, including the 14 Stage S source-owner
tests and one T ABI test. The `stage_s_` filter again passed **23/23**; nine Geometry
Stage S tests are selected separately from the original-source tests.
These counts are not additive with earlier packets.

The sealed postconstruction resource-query surface is publicly re-exported on
the existing tensor route without exposing its fields or accepting arbitrary
Programs. Its borrowed-view accessor has a separate **128-event** tariff and
typed control inventory; QUERY **256**, BODY **1,198,656**, maintenance **37,568**
and the universal resource quote remain unchanged. The frozen focused resource
filter passed **14/14**, and the `work_preflight` doctest filter passed **17/17**,
including an external public consumer and targeted opacity/lifetime failures.
The added full/zero/tail scheduling regression now passes within the **288/288**
Core tensor run. It covers literal query/body counts **1/0/2**, both RMS origins,
separate accessor/body sponsorship and output-specific postchecks, with query/view/
body controls retired before output consumption. Independent re-review accepted
the full bounded frozen-plan source cut; the 15 current resource-view tests are
included in 288, not additive. A fresh PUMA-wide doctest filter passed **27/27**,
overlapping the earlier 17. An earlier full Core doctest run passed 69 and failed
one unrelated `rlwe_kernel::quote` prose table parsed as Rust. Converting only that
table to Markdown repairs the documentation: the fresh complete Core doctest run
passed **69/69**, with no ignored tests. Its count is one lower because the table
is no longer discovered as a Rust example; no executable example was disabled.
These results do not
provide SOURCE funding: Compiler integration still requires Stage T's genuine
original-root loan and caller/control coverage.

The inactive shared-shape extraction has 18 authored test definitions and passed
independent semantic/macro review. Production shape projection and its callers
remain untouched by this extraction. A separate custody repair replaces
counter-only test coverage with a real external Budget-backed sponsor across
nested-prefix, replacement, error and unwind retirement. Independent source
re-review passed its finite fixture census, exact/one-short pre-factory boundary,
same-root charge observations and post-drop baseline. After a test-only type
annotation repair, the mounted `source_shape_projection::tests::` filter passed
**18/18**. The earlier filename-style filter matched zero tests and is not
validation. A separate inactive checked-const row algebra now has six passing
tests for independent arithmetic, nominal size/alignment, distinct lexical sites
versus repeated execution, Option/Result residuals, direct conversion-item
boundaries and overflow. A callback ABI oracle was corrected to the actual fat
`&str` capture; production algebra was unchanged. The fresh combined projection
filter passed **24/24**, including the earlier 18 semantic tests. Direct-item
rows explicitly exclude callback receiver coverage. The subsequent inactive
F00–F09 finite-source inventory passed independent source review and its four
independent arithmetic/layout tests. The fresh combined projection filter passed
**28/28**, including the earlier 24. This inventories the fixed finite-bound
calculation and its helpers, not a complete query/body evaluator or adopted
coefficients. Complete projection tariff, first-query sponsorship, telemetry,
factory/cleanup accounting and publisher activation remain open.

A private pure `public_dot_operand_cap` helper now derives a sufficient symmetric
raw-dot input cap from the existing quarter-gap constant and a mathematical
complete-row L1 premise. Independent source review passed; eight tests passed for
inequalities, maximality, signed endpoints, finite enumeration and prefix products.
It has no production callers and does not authenticate coefficient statistics,
narrow unchecked views, emit guards or adopt a new numeric profile. Whole-model
numeric acceptance and resource fit remain separate gates.

The inactive checked-Rows Core implementation now reuses the existing extension descriptor
instead of adding a Program Option field. This avoids an unproved claim of spare
legacy initialization/presence-control work. The initial ten-owner source freeze
had 20 `checked_rows_` test definitions. Independent correctness/numeric/provenance/
custody review passed. Both finite accounting repairs are now source-reviewed:
separately charge the pre-take validator body and inventory complete mapped
conversion/Try/error control families. The repaired filter passed **21/21**:
construction 4, RMS/work 7, resident 7 and resource-view 3. The subsequent broad
latest frozen Core tensor filter passed **288/288** in 91.15 seconds, **including
those 21 Rows tests, two new strict normal-drop factory cases and the 15 resource-view tests**;
these counts are not additive. Both wide-RMS graphs still measure **261,256 /
262,144 owned bytes**. Old absence checks retain their exact
RMS/proof-free meaning; Rows resource-view classification is explicitly Unsupported.
The corrected Rows source hash census is 275 updates, 277 calls including init and
finalize, and 1718 public bytes. Its proposed opaque source-call logical row is
3468 writes, 17728 validation visits and 277 transitions, not SHA CPU work or paid
admission. The validator-body repair changes the proposed constructor reserve
from `52K + H` to `54K + H`; resources remains `8K`. These rows and conversion
controls remain unadopted. Earlier 261-test Core results predate these changes.
The older large normal-drop matrix permits `Bounds`, so that matrix alone does not
establish 384/512-lane admission. Two new strict cases establish the bounded factory
fixtures described below. Caps and Compiler selection remain unchanged. Validation
used repaired frozen inputs and a serial native slot. The audited Core manifest
has no Compiler dependencies (including dev dependencies), build script or Compiler
source includes, so unrelated Stage S Compiler edits do not block that command.
Compiler validation still requires its own dependency inputs frozen.

An independently source-reviewed transport screen distinguishes actual ordinary
binding destruction from delivery: Rust consumes encoded native owners, copies
bytes and drops those owners before returning Python bytes. Delivery alone earns
no issuer-lease credit. A harness retaining all originals refuses at 512 lanes by
the first PPA and at 384 by the second; these negative controls do not reject the
ordinary normal-drop candidates. After accounting for the PPA increment, current
native payloads must fit 94,208 bytes at 512 lanes or 135,168 at 384, including the
operand copy. Both PPA live-owner inventories, intra-operation peaks and separate
whole-request admission remain unverified. Updated encoded tests drop all exported
originals before delivery and the next peer call; they ran within the Core filter,
but their allowed refusal outcomes alone do not establish large normal-drop fit.
Two genuine-factory normal-drop cases now require strict success in the existing
Core test driver, without changing caps or frame destruction. Both passed within
the 288-test run: all six wide calls, completion, independent numeric oracle,
exact frame checks and zero live payload after cleanup. Owned admissions measure
**182,792 bytes at 384 lanes** and **236,040 at 512**, under the unchanged
**262,144-byte** cap. Neither case accepts `Bounds`. These are exact-fixture
factory/Executor results, not general batching or complete-Qwen admission.
Boundary ledger snapshots still cannot identify every internal allocation refusal
or operation-local peak (`ND-INNER-OBSERVATION`); that observation gap remains open.

Earlier Stage T bounded state/pricing and custody reviews preceded reconciliation
against actual frozen types. Their proposed source slot in `RawHeaderStage` and
separate request-entry claim storage are superseded by the composed T owner and
private exact-Context token described above. No shared header or grant field is
added. Integrated publication and any later zero-query claim still require a
genuine producer and prospective funding; the inactive declarations supply neither.
The current census distinguishes initial-paid post-working, model-replacement and
pre-header copy peaks, each retaining the fixed configuration copy window, and
separates initial from successor-complete cleanup leaves. Complete descriptors
remain unclosed until actual body correspondence and typed arithmetic checks.
Conditional source authorability is not adopted accounting or an integrated
funded claim. A further actual-source audit locates the producer-entry boundary
at bounded request BODY payment, before preparation and grant construction;
Geometry's request working-size query occurs even earlier. A future fixed
`T_entry` may be prospectively included in the initial original-ledger debit,
with its own literal typed census and refusal coverage. It cannot retroactively
charge an existing grant or pay controls that precede that initial debit. A
successor-only cut must not mint later claim eligibility. These are concrete
future payer boundaries, not implemented funding. This is not a query callback
or adopted tariff.
Existing grants lacking that allocation remain ineligible; no seed reopening or
late self-payment is allowed. Later import/compile integration must retain residual
custody across consuming failures. Before sharing SourceCustody, install that
residual into its by-value owner and retain a real caller-owned shared anchor
after payload fields through fallible finalization/publication. No late mutation
of already shared custody is assumed. Concrete frozen interfaces, adopted prices,
payment/retirement regressions and five-file query integration remain outstanding.

The earlier raw-admission Python oracle packet passed **237 tests** in 5.27 seconds using
`uv run --no-sync python -m pytest -q tests/test_replicated_decoder_oracle.py`.
This includes the independently source-reviewed raw-payload admission controls:
optional per-call limits, preallocation refusal, metadata assembly and propagation
through direct/client receive paths. The limit covers requested raw-buffer bytes,
not aggregate live storage, JSON/Python objects, native backing or RSS. This
supersedes the earlier 188-test pure-Python result; it is not a generated-native run.

The subsequent shared lexical census and bounded `SourceReadPolicy` owner packet
passed **534 tests** in 5.05 seconds against the matching installed wheel:

```sh
uv run --no-sync pytest -o pythonpath= -q tests/test_json_census.py tests/test_source_json_admission.py tests/test_replicated_decoder_oracle.py
```

The ordinary locked/offline Rust 1.85.1 single-job release wheel was rebuilt and
reinstalled; Python/native origins were verified under `.venv/site-packages`.
Clearing pytest's checkout `pythonpath` avoids the stale ignored checkout native
extension without deleting it. The shared scanner retains its lexical behavior;
the oracle's optional gate checks metadata before decoding and repeats root-key
and provenance validation after the copy-admission callback. The helper-owner
inventory is charged within a retained extended census-frame unit throughout the
admitted parse window, including helper-absent and failure phases, under the
existing frame allowance. Startup qualification and pre-census caller coverage
remain separate premises. On the tested host,
fixed controls measured 8178/8192 bytes, helper metadata 7209/8192, and the frame
plus two traceback allowance 1360/4096. Layout guards fail closed; these are
bounded logical-owner results under the reviewed trusted-host premises, not a
portable ABI, allocator/RSS or total 64-MiB guarantee. No fee or cap increased.
The initial 533-pass/one-failure run exposed two captured pointer slots missing
from an independent test frame-header calculation; the test-only correction
preserved exact equality and every production bound and passed source review.

Adjacent shared-parser/Forward ownership regressions separately passed **61 tests**
in 1.05 seconds using
`uv run --no-sync pytest -o pythonpath= -q tests/test_forward_json_admission.py tests/test_forward_runtime_ownership.py`.
Two initially stale disk probes were retargeted to the extracted shared
`bundle_storage` helper, preserving positive filesystem/cleanup and low-target,
ample-home pre-snapshot refusal assertions; independent source review passed.
One existing NumPy shape-assignment deprecation warning remains. These generated
fixtures do not establish PUMA worker execution or complete importer retirement.

Commands use ordinary tests, for example
`cargo test --locked --offline -j 1 -p pllm-compiler --lib raw_append_leaf -- --nocapture --test-threads=1`;
replace the filter with the other focused selectors for their corresponding
tests. Source inputs remain frozen for each command.

The fresh complete `stream_transport_tests` run passed **13/13** in 22.17 seconds
after independent actual-source review. The parent socket is configured
nonblocking before spawn; shared retry pauses check the same absolute deadline
before and after sleeping, with no timeout setters. The 300-second deadline,
20-ms maximum retry slice and two-second cleanup observation allowance remain
unchanged. Cleanup ignores only shutdown `NotConnected`; protocol shutdown and
other errors remain strict. Negative peers now retain the connection until their
intended trailing-byte or exit-status boundary; tests require actual phase
progress, EOF and exact-child status rather than accepting an earlier failure.
The injected unwind panic appears in a passing test. Polling adds slice latency;
neither scheduling nor kernel reap has a hard guarantee, and custody remains
retained through reap. This is passing transport-component evidence, not live
Worker/SetupParty lifecycle evidence. The separate storage-selector run passed
**2/2** in 0.03 seconds. An earlier compiler test's private core `max_i8()` call
was repaired without widening the API; that failed command executed no tests.

Production Geometry entry still needs its additional caller prepayment and
external error/unwind custody. The typed raw-write substitution removes one
native-error-formatting leaf, not the whole publisher's caller quotation or
width-square tariff. A reduced append quotation also needs retained original-funded
collector-retirement coverage when the next phase admission refuses.
Generated-native streaming remains held on complete
simultaneous native/Python owner inventory and pretransition/preparse admission.
SiLU and softmax independently exceed the unchanged complete-request ordinal cap
at the saved ordinary-chat envelope; inactive trie and conditional batching/proof
plans do not establish fit. No cap, routing, source authority, numeric/security
contract, activation or whole-method qualification is advanced by these tests.

## 3. Ordered delivery plan

Estimates are focused engineering days, not promised elapsed time. Budget
**8–13 days total if the feasibility gate passes**, with the first continuation
decision after one day. Host admission, checkpoint runs and review can extend
elapsed time.

| Milestone | Estimate | Dependency | Exit artifact / gate |
| --- | --- | --- | --- |
| P0. Complete cost and blocker inventory | 0.5–1 day | Existing implementation | Costed go/no-go decision for selected ordinary chat ≤32+2 and the shared 64+8 envelope. |
| P1. Complete-model numeric admission | 2–3 days | P0; approved checked-range contract | All layers and selections have checked primitive domains; positive and negative independent-oracle tests pass. |
| P2. Bounded bulk execution and capacity | 3–4 days | P0; P1 supplies the final numeric schedule | Complete 64+8 work/storage/transport admission; measured component scaling and generated three-worker regressions. |
| P3. Pinned full-Qwen conformance | 1–3 days | P1 and P2 | Actual ordinary chat ≤32+2, then separately admitted larger/boundary workloads, with complete correctness and lifecycle evidence. |
| P4. Shared quality and response campaign | 1–2 days | P3 | Standard CLI reports, repeatable Experiment and updated capability coverage. |

P2's lookup and transport work can be developed against generated fixtures while
P1 is under review. Full-checkpoint execution waits for both.

### P0 — decide feasibility before more layer-by-layer repairs

Extend the existing native schedule/resource diagnostics and ordinary benchmark
reporting to expose:

- Ordinals, AES/PRSS consumption, primitive invocations and live tensor/program
  owners, separately from peer rounds, frame bytes and HTTP control exchanges.
- Lookup, body linear, RMSNorm, softmax, SiLU/multiply, complete head and protected
  argmax costs for every full-prefix selection.
- Public source import/snapshot storage and logical file reads, charged to each
  owner. Distinguish shape-derived lower bounds, conservative estimates and actual
  measurements; failed range admission cannot mint an executable plan.
- All known workload, source, value, node, program, work, queue and timeout caps
  in one report, rather than discovering them through successive wheel builds.

Use admitted generated cases through shared benchmark facilities to measure native
work and the existing three-party peer path. Add any missing measurement facility
to the SDK rather than a PUMA-only benchmark runner. Keep extrapolations explicitly
separate from measured full-model results.

**Decision:** identify a bounded implementation route for the selected first chat
workload and 64+8, and price its dominant CPU, communication and memory costs. Screen directed byte
floors under the study's shared 100/40 Mbps access caps against the current
1,200-second attempt lifetime. A known lower bound beyond that lifetime blocks
the current execution contract; unknown dominant costs do not count as a pass.
Report that decision before spending the remaining implementation budget.

### P1 — resolve the entire numeric graph once

First assemble a per-operator range map for all layers. Timebox a source-only proof
attempt to half a day within P1, using existing row/energy certificates, including
the joint gate/up and down-projection residual. Its goal is a complete graph result.

If those proofs remain insufficient, the approved completion route is a separately
identified **protected checked-range profile**. The subsequent instruction to
proceed with the complete plan approves this contract:

1. Freeze public source/workload-bound ranges in an immutable component/artifact.
   Select them from public constraints and public development data, never a private
   request. Workers validate the same profile and source commitments.
2. Before any operation whose safety relies on a range, evaluate membership inside
   MPC. Comparisons and selection themselves must be safe over the entire incoming
   domain, including failure cases and subtraction/wrap endpoints.
3. Privately select a proven-safe fallback operand for an invalid value **before**
   unsafe multiplication, accumulation or truncation. Keep a sticky shared validity
   bit. Checking an already wrapped result or merely rejecting at the end is
   insufficient.
4. Execute the same public schedule in valid and invalid cases. Only the client
   receives aggregate validity at the final acceptance boundary. Invalid requests
   return no model tokens or reusable state; no intermediate failure location or
   validity is opened to workers. Account for the client's final success/failure
   disclosure in the new contract.
5. For accepted requests preserve the approved Q18 equations, rounding law and
   tails. Charge fresh guard products, comparisons, frames and PRSS work under a
   distinct execution identity. Extra draws can change a particular stochastic
   realization; do not claim old/new bit parity without a coupled-tape argument.

Independent tests cover all boundary predicates, failed guards before unsafe work,
sticky failure through later layers, final acceptance and cancellation. Public
held-out range failures remain failures in quality reports; they cannot be dropped
or used to retune the profile retrospectively.

**Gate:** native numeric analysis validates all 24 layers and every requested
prefix, including final normalization/head/argmax, with fully proved primitive
domains. Diagnostics must not bypass resource checks to issue a runnable program;
complete source-publication admission requires P2 as well.

Primary files: `crates/pllm-core/src/puma/{tensor.rs,tensor_numeric.rs,work.rs}`;
`crates/pllm-compiler/src/replicated_decoder/{program.rs,source.rs,sdk.rs}`;
the existing Python protocol component, publisher and final-response acceptance.

### P2 — fix execution granularity, then justify capacity

Implement and measure these changes in order:

1. **Bulk private lookup:** add bounded public row/vocabulary tiles to the native
   executor. Form fresh shared predicates, stream public weights across several
   rows and accumulate complete exact embeddings. Keep complete vocabulary access
   independent of private IDs. Price selector, accumulator and file-buffer overlap.
   Preserve D9's recomputation; this is no cross-selection prefix cache.
2. **Predicate and local-operation batching:** fuse reusable local Boolean maps
   and batch independent primitive phases where the admitted payload permits.
   Prove any narrower predicate width from the public token domain. Preserve
   source PRSS fill semantics and fresh material. Measure CPU/AES/bytes/rounds as
   well as ordinals so a changed accounting granularity cannot impersonate a win.
3. **Bounded peer streams:** replace coordinator HTTP round trips per primitive
   with authenticated long-lived peer delivery and native phase/ordinal readiness.
   Retain bounded queues, backpressure, ordered acceptance and whole-attempt
   retirement. Slow peers must not receive a later phase before its prerequisites.
   Reuse role supervision, credentials and accounting; no second public runtime.
4. **Complete-program capacity:** derive the remaining node, value, storage, AES,
   sequence, frame and lifetime needs from the final schedule. If full-prefix
   schedule metadata duplicates dominate, use bounded public iteration/templates
   while charging every fresh selection. Extend only demonstrated resource
   envelopes, consistently across native admission, SDK, workers and benchmarks.

The 32-input/40-total frontend limits cannot simply be changed to make a failing
benchmark start. Nor can the request be split into independent budgets to erase
its complete-program cost. A larger envelope requires bounded allocations,
checked counters, measured work, failure tests and a distinct capacity contract.

**Gate:** the 64+8 envelope passes every capacity check; measured generated cases
reconcile schedule and transport counters; replay, cross-session substitution,
slow-peer, expiry and cancellation tests retire every owner. If the measured route
cannot fit the chosen request/resource budget, stop and record the blocker before
launching expensive checkpoint runs.

Primary files: native `replicated_decoder/{program.rs,runtime.rs,public_source.rs}`,
`puma/{tensor.rs,work.rs}`, the replicated Python worker/transport, shared role
supervision, `benchmark_memory.py` and `benchmark_targets.py`.

#### Source-only compact publication obligations

**Status: static source only.** The following inventory records branches,
representations and unresolved coverage obligations in
[`compact_request_sealed.rs`](../../crates/pllm-compiler/src/replicated_decoder/compact_request_sealed.rs).
It establishes no fees, admission result, native entry funding or execution
qualification.

| Source lines | Reached operation or boundary | Coverage obligation |
| --- | --- | --- |
| [543](../../crates/pllm-compiler/src/replicated_decoder/compact_request_sealed.rs#L543) | Completed-state refusal before quotation or growth. | Cover the executed check and its reached cleanup under prior owners. |
| [547–566](../../crates/pllm-compiler/src/replicated_decoder/compact_request_sealed.rs#L547-L566) | Layout construction and checked quotation refusals before growth. | The executed prefix needs prior coverage; its calculated demand is not an advance debit. |
| [579](../../crates/pllm-compiler/src/replicated_decoder/compact_request_sealed.rs#L579) | Lease-growth refusal. | Future prepayment cannot retrofund the reached prefix or refusal cleanup. |
| [583–588](../../crates/pllm-compiler/src/replicated_decoder/compact_request_sealed.rs#L583-L588) | Publication and retained-resolution prepayment failures. | Original owners remain retained through the reached cleanup; complete coverage remains to be shown. |
| [589](../../crates/pllm-compiler/src/replicated_decoder/compact_request_sealed.rs#L589) | Output reservation failure. | Include reservation work, retained owners and reached destruction. |
| [597–608](../../crates/pllm-compiler/src/replicated_decoder/compact_request_sealed.rs#L597-L608) | Custody `Box` and Program `Arc` construction. | Separate allocation, initialization and cleanup obligations; these are not ordinary returned-`Result` refusal branches. |
| [609–627](../../crates/pllm-compiler/src/replicated_decoder/compact_request_sealed.rs#L609-L627) | Conditional binding access, declared-digest, schedule-length, custody-presence and public-source check refusals. | Count each complete reached path, including its prefix and cleanup. |
| [628–636](../../crates/pllm-compiler/src/replicated_decoder/compact_request_sealed.rs#L628-L636) | Mutable `Arc` projection, missing custody and installed-variant refusal. | Preserve the actual installed owner and its still-uncommitted attempt through refusal. |
| [637–638](../../crates/pllm-compiler/src/replicated_decoder/compact_request_sealed.rs#L637-L638) | Commit the same installed `Attempt` and return the Program. | Preserve the identity and publication boundary; this source inventory supplies no execution result. |

The quotation at
[line 562](../../crates/pllm-compiler/src/replicated_decoder/compact_request_sealed.rs#L562)
includes two `Result<&mut Self, Error>` representations; lease growth occurs later
at line 579. One installed-projection chain is identified: the resident match at
[192–196](../../crates/pllm-compiler/src/replicated_decoder/compact_request_sealed.rs#L192-L196)
through the call and destination at 632–636. The second representation remains
**unassigned**. This supplies no fungibility, call/return aliasing or work-credit
inference.

Local sealed-field destruction follows the declaration order at
[53–58](../../crates/pllm-compiler/src/replicated_decoder/compact_request_sealed.rs#L53-L58):
`Attempt`, recipes, finalized-owner `Arc`, then covering charge. In
[`request_schedule.rs:317–341`](../../crates/pllm-compiler/src/replicated_decoder/request_schedule.rs#L317-L341),
an uncommitted `Attempt` stores `BURNED` before its origin `Arc` field retires.
This is local ordering only, not final backing release, global final-guard ordering
or whole-cleanup funding proof.

The public-source check at
[`public_source.rs:429–477`](../../crates/pllm-compiler/src/replicated_decoder/public_source.rs#L429-L477)
performs **both binding checks before actual input traversal**: the `filter_map`
iterator is lazy. Traversal then inspects all input recipes, filters public
descriptors, performs keyed file lookup and digest/size checks, and checks a
nonzero exact descriptor count. The all-input count and the yielded
public-descriptor count are distinct obligations; neither can stand in for the
other.

Independently established dynamic `N`/`M` pricing remains separate and does not
cover unresolved leaves by implication. The historical 64-visit metadata
allowance remains **unproved**: neither adequacy nor a counterexample or overrun
has been established here. Normative leaf checks, resize and destruction coverage,
and prior-prefix funding remain open. This inventory changes no rates, caps,
completion status or permissions; execution HOLD continues.

### P3 — actual checkpoint execution and independent correctness

1. Extend the existing independent stochastic-integer oracle to the complete
   semantic decoder, including streamed public products, attention, normalization,
   rotary, guarded activation, head and stable argmax. Do not call production
   arithmetic helpers from the reference.
2. In a test-only harness reconstruct party shares and compare every admitted
   layer boundary, complete logits, full-prefix attention K/V, selected IDs and
   feedback history against the independently replayed randomness. Exercise both
   generated adapters and the pinned checkpoint. This is conformance access, not
   a production hidden-tensor export or persistent-KV contract.
3. Run the ordinary three-process installed-wheel path at 4+2, then 16+2, then the
   64+8 boundary. Freeze public cases first, include nonzero/nonconstant data, and
   retain failures. Verify final-only disclosure through SDK and gateway.
4. Inject malformed/missing peer frames, stale sessions, mid-operation and
   final-acceptance cancellation, invalid ranges and partial source publication.
   Verify native buffers, authenticated files, pending replies and role processes
   retire before admission capacity is refunded.

Build/test instrumentation has a separate identity and cost from serving samples.
Final production checks use the installed wheel, source pins and ordinary role
supervisor. Host admission and swap monitoring remain active throughout.

### P4 — finish through the shared paper tools

- Make `examples/papers/puma.py` explicitly select the qualified public-weight
  component and any approved range/capacity profile. Keep its model, public inputs
  and workload envelope sourced from the shared paper files.
- Execute `pllm benchmark quality` on the shared prompts and same-token positions
  against the existing streamed FP32 reference. PUMA supplies final-token agreement
  through fresh one-selection requests; top-k and logit-error metrics stay unknown.
  This is a fidelity diagnostic, not a retrospective task-quality qualification.
- Execute `pllm benchmark run` with frozen shared cases and three fresh-process
  replicates. Include publication, client and three-worker CPU/RSS, private files,
  directed peer and client traffic, startup-through-cleanup time and time to final
  accepted output. Buffered output supplies no streaming decode-TPS measurement.
  Classify observed client memory from the largest complete-client replicate peak;
  topology RSS uses simultaneous samples, not a sum of independent lifetime peaks.
- Use one matched invocation/cohort for controls where supported. Retain different
  numeric, topology and acceptance identities and the comparison API's ranking
  restrictions. Source-present delivery, TCP streams and physical wire keep their
  separate scopes.
- Validate the exact published commands, standard replay and benchmark coverage.
  Update `IMPLEMENTATION-STATUS.md`, the PUMA contract and paper recipe links once.
  Use the shared evidence registry and existing build receipts, rather than a new
  runner or archive system for each admission error.

## 4. Completion control

Commit validated milestones, with focused native tests during development and one
installed-wheel regression pass per completed integration slice. Do not rebuild
and archive a full campaign merely because another individual layer admits.

P0 produces the first firm continuation decision. P1/P2 failure produces a precise
blocked status and the remaining work estimate; it does not silently narrow the
common workload, weaken arithmetic, or reset the agreed effort budget. P3 failure
is a correctness/lifecycle blocker. P4 can legitimately finish with poor measured
quality or speed, but those results cannot qualify a quality-matched or practical
performance claim. Completion is recorded only against the five checks in Section 1.
