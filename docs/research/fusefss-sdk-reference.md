# Bounded FuseFSS decoder and ordinary SDK

`FuseFssSourceCpu` selects a model-neutral ring50/Q12 decoder with a separate
trusted offline dealer and two online additive-share holders. The generated
GPT-2 and dense-RoPE SDK paths execute two fixed-count selections through three
actual child processes, including private feedback. SDK/gateway failure and
cancellation tests exercise the same native owners and authenticated HTTP path.

## Numeric and source binding

The two static profiles are
`fusefss.gpt2-complete-ring50-q12.v1` and
`fusefss.dense-rope-ring50-q12.v1`. Their approved equations are D13 in
[`paper/methods/DECISIONS.md`](../../paper/methods/DECISIONS.md).
Original F32/F16/BF16 bytes stream through bounded native import. Source,
tokenizer, geometry, numeric identity and program commitments precede private
input issuance. Public weight paths remain in source mappings; method lowering
consumes semantic operators rather than model-specific node names.

Every contraction accumulates Q24 products, floors once to Q12, then adds its
Q12 bias. Attention scale has a separate product and floor. Dense RMSNorm retains
the source table, unaveraged square sum plus integer one and separate affine
floors. RoPE floors each signed coefficient product before addition. These are
explicit method numerics, not FP32 checkpoint equivalence.

The opt-in `fusefss.dense-rope-checked-ranges-ring50-q12.v1` profile adds D14's
private RMSNorm guards. Each complete row must satisfy
`|x| <= floor_sqrt((2^49 - 2) / width)`; the bound is 792,648 Q12 codes at width
896. A failed guard selects an entire zero row before normalization. Secret
validity persists across every padded row and selection, and masks all final
tokens. The client raises `NumericDomainRejected` on final failure without
releasing earlier tokens or a failing location. Accepted requests retain both
D13 floors and tables. Other arithmetic obligations still require admission.
The checked-profile factory is `examples/papers/fusefss.py:qwen_checked_smoke`;
native generated parity does not establish trained-checkpoint acceptance.

Five frozen generated models cover tied/untied GPT-2, Qwen2, Qwen3 Q/K norm and
Llama structures. The independent semantic interpreter checks 918 complete
observations across three full-prefix selections each, including physical padding,
causal GQA, full heads, feedback and final IDs. The older 5,125 internal-register
exchange has a different observation contract and is not claimed as passing.
The strict semantic contract lives in
[`paper/reproduction-contracts/fusefss/decoder/SEMANTIC_EXCHANGE.md`](../../paper/reproduction-contracts/fusefss/decoder/SEMANTIC_EXCHANGE.md).

### D64 conditional full-prefix population lemma (design only)

This candidate uses existing `Schedule` and compiled-node shape semantics, not a
new execution strategy. [D64](../../paper/methods/DECISIONS.md#d64-fully-costed-fusefss-prefix-support-for-actual-request-envelopes) requires complete
cost admission and independent review before activation; it grants no cap/source
change or implementation. **Independent Astra review: partial documentation census PASS**
for this subsection only; not funding, fit, whole-method, security,
source-faithfulness or native admission qualification. Existing limits remain unchanged.

For positive initial tokens `I` and selections `O`, checked arithmetic validates
`T = I + O - 1`; selection `s` has logical prefix `p_s = I + s`, `0 <= s < O`
([shared geometry](../../crates/pllm-models/src/lib.rs#L1495-L1545)). This metadata
validation neither constructs a graph nor admits native execution. Current native
hard/default sequence cap is 16, with independent graph/state checks
([compiler](../../crates/pllm-compiler/src/fss_decoder/program.rs#L522-L574));
[RoPE source cap](../../crates/pllm-compiler/src/fss_decoder/source.rs#L603-L621)
and [Python early cap](../../python/pllm/runtime/fss_source.py#L554-L556) also remain 16.
Thus D64 examples below are NOT currently admitted or constructed runs.

Every selection creates fresh node arrays at full compiled shapes using extent
`T`, not `p_s` rows to allocate or price. `Last` reads row `p_s - 1`; nonfinal
feedback writes token position `p_s`. For `I=30,O=2`, prefixes are 30/31 and BOTH
passes use `T=31`; for `I=32,O=2`, prefixes are 32/33 and BOTH use `T=33`
([allocation/feedback](../../crates/pllm-compiler/src/fss_decoder/runtime.rs#L878-L953),
[Last](../../crates/pllm-compiler/src/fss_decoder/runtime.rs#L1686-L1690)).
This `Schedule` retains no interselection activation K/V cache. Tokens, finals,
validity, Program/source, Driver/visitor, context, resources, obligation hash and
source-table cache can persist; table reuse and fresh private material are distinct
from activation K/V reuse ([Schedule](../../crates/pllm-compiler/src/fss_decoder/runtime.rs#L418-L431)).

**Conditional completed-traversal population.** For each `Scores` node `n`, let
`H_n = node.shape[1]`, `d_n = left.shape[3]`. Serial loops visit all `T*T` query/key
pairs per head per selection: `N = O*T^2*sum_n(H_n)` scalar dot outputs and
`M = O*T^2*sum_n(H_n*d_n)` scalar product terms. Per node, two selections give
`1922*H_n` outputs at `T=31`, versus `2178*H_n` at `T=33`, not `30^2+31^2`
times heads and not single-query decode. Counts apply once per completed Schedule
traversal, not multiplied by parties, dealer or repeated preflight traversals.
Repeated preflight traversals incur separate work; this population does not price
them or make them free.
[Score/softmax/value loops](../../crates/pllm-compiler/src/fss_decoder/runtime.rs#L1522-L1684)
also give each Softmax node (`H = node.shape[1]`) `O*H*T` width-`T` rows, including
padded queries; active key predicate is `k <= q && k < p_s`. Each Values node has
`O*H*T*d` scalar dots of length `T`, with `H = node.shape[1]`, `d = node.shape[3]`,
distinct from Scores' left-operand `d_n`. Wavefront `dot_lanes` has the same output population,
but different kernels/batching: these counts are NOT Driver/material/frame counts.
Renew checks when profile or schedule changes.

**Raw node-array peak only.** Let `b_i = 8*elements(shape_i)` and `L_0 = 0`.
Indices follow compiled node order; within one selection, `L_i` is live prior-node
payload bytes immediately before allocating node `i`.
Then `P = max_i(L_i+b_i)` and
`L_{i+1} = L_i+b_i-sum_{j<i:last_use_j=i}(b_j)`: allocate current output before
retiring earlier last-use arrays
([compiler recurrence](../../crates/pllm-compiler/src/fss_decoder/program.rs#L1473-L1495)).
`P` is neither total reservation nor physical peak. The
[live-array quote](../../crates/pllm-compiler/src/fss_decoder/resource_quote.rs#L185-L207)
adds token/output backing, gather workspace, array slots and wavefront backing;
other Source/Driver/material/retirement owner peaks remain unproved here.

Allocation, geometry, proof or Driver errors can truncate traversal: this is a
success-population lemma, NOT an all-outcome/WVT bound, fee, fit, security,
physical-peak, whole-method or native qualification. Signed-SiLU construction stays
held, lifecycle remains UNSELECTED, and source-faithful execution-strategy
qualification remains unknown. No broader approval or execution follows.

### D64 conditional admission dependency map

**Source census only; independent Astra review: PASS for this dependency map only.**
Component-coefficient review and whole-D64 admission closure remain incomplete.
Independent Astra multiplicity review: PASS for 12 walks/request, not equal phase
costs. This map extends the reviewed fixed-`T` census above, not its review clearance.
It connects existing owners and quote sites; it adds no rate, cap, registry, driver,
security assumption, fit result, execution authorization or change to any HOLD.
Unknown terms are not zero.

**Call DAG, ordinary successful route.** One completed `Schedule` walk already
contains ALL `O` selections, nodes and checks; do not multiply it by `O` again.
The [compile tail](../../crates/pllm-compiler/src/fss_decoder/program.rs#L1491-L1507)
calls [preflight](../../crates/pllm-compiler/src/fss_decoder/runtime.rs#L254-L287).
[Role bind](../../python/pllm/runtime/fss_worker.py#L875-L911) constructs
[`FssRoleRequest`](../../crates/pllm-python/src/fss_transport.rs#L590-L617), whose
admission uses [transport `Scan`](../../crates/pllm-compiler/src/fss_decoder/transport.rs#L287-L377).

| Successful request branch | Completed Schedule walks |
| --- | ---: |
| Three role source imports → compile/preflight | 3 |
| Three role binds → admission Scan | 3 |
| Dealer preparation → readmission Scan → generation | 1 + 1 |
| Two holder runs → readmission Scan → execution | 2 + 2 |
| Ordinary role-route total | 12 |
| Decoder-only cold topology startup → compile/preflight + quote Scan | 2 once/startup, not per request |
| Optional additional imported-program quote | +1 per call |

Dealer/holder reentry sites are [`prepare_with_delivery`](../../crates/pllm-compiler/src/fss_decoder/transport.rs#L565)
and [holder readmission](../../crates/pllm-compiler/src/fss_decoder/transport.rs#L928).
[Cold qualification](../../python/pllm/runtime/servers.py#L1116-L1147) imports,
prices, then retires its separate owner. Only [`start`'s `fss_public` AND `capacity="decoder"` branch](../../python/pllm/runtime/servers.py#L1194-L1218)
calls `_qualify_fss_decoder`; Capsule/default uses metadata admission, without these
two walks. Qualification uses configured MAX geometry, potentially unlike actual
request geometry. [Optional quote](../../python/pllm/runtime/benchmark_delegated.py#L360-L375)
reaches the [native pricing Scan](../../crates/pllm-python/src/fss_transport.rs#L807-L812).
Program retention within a role avoids recompilation, not repeated Scan walks.
Cost is a sum of separate phase cost vectors: three compile/preflight + six admission
Scans + one generation + two holder executions per request, plus optional separate
startup compile/preflight and quote vectors. Never price 12 identical costs or 14
same-geometry walks. Driver, reads, public contractions and callbacks differ;
No-Driver/dealer paths may skip public contractions while kernel/quote/cache/hash
work and allocations remain. Structural counts are not observed candidate completion:
failed `T=31/33` candidates cannot claim 12/14 completed walks. Cap 16 and signed-SiLU
gates still refuse; count each actual failure prefix instead.

**Completed node partition.** Rows below are per node, per selection; sum over
validated metadata and selections within each phase vector; failed prefixes need their own census.
`R = input elements / input width K`, `C = output width`; embedding has vocabulary
`V` and width `E`. `H,d` follow each node's operands/shapes as above, not model labels.

| Node/path | Work population, not primitive coefficients |
| --- | --- |
| Resident serial affine | `R*C` public dots, each length `K`; public coefficients are not private multiplications. |
| File serial affine | `R*K*C` local multiply/add pairs plus `R*C` fresh floors; file/statistic reads separate. |
| Wave/tensor affine | Same complete public contraction then floors; bounded lanes change Plan grouping. |
| Token embedding | `T` selector Plans plus `T*V*E` local pairs; file batches also retain gather/copy work. |
| Scores, serial | `T^2*H` private-dot Plans, each length `d`. |
| Values, serial | `T*H*d` private-dot Plans, each length `T`, NOT `T^2*H*d` Plans. |
| Softmax | Full output population `H*T^2`; row active width `a=min(q+1,I+s)`; profile/batch bounds determine actual Plans. |

Source: [embedding/affine](../../crates/pllm-compiler/src/fss_decoder/runtime.rs#L980-L1224),
[Scores/Softmax/Values](../../crates/pllm-compiler/src/fss_decoder/runtime.rs#L1522-L1684).
Wave/tensor `dot_lanes` preserves contraction populations, not serial Plan counts.
Norm, activation, selection, guards and final masking remain required; detailed
unreviewed primitive coefficients deliberately omitted. No invented party multiplier.

**Existing reservation tariff, not a disjoint-backing proof.** Reuse
[`Quote::live_array_reservation`](../../crates/pllm-compiler/src/fss_decoder/resource_quote.rs#L185-L207):
`L = P_arrays + 8*(T+O) + node_slots + gather + frontier`, where `P_arrays` is the
last-use array peak above. `node_slots` prices `Option<Zeroizing<Vec<u64>>>` slots;
`gather` is the function's capacity-dependent maximum, including Decoder reader
buffers; `frontier` is its wavefront-only tournament/lane backing (zero otherwise).
No replacement workspace or second ledger is introduced.

[`Quote::finish`](../../crates/pllm-compiler/src/fss_decoder/resource_quote.rs#L331-L358)
accumulates the existing tariff as
`3*M + 2*L + max_k[max(R_k + W_k + 2*P_k, 3*D_k) + 3*C_k]`:

| Symbol | Existing field/site, not a new charge |
| --- | --- |
| `M` | `resources.metadata_bytes`, including source, energy and compiler/validation scratch; [compiler inventory](../../crates/pllm-compiler/src/fss_decoder/program.rs#L1456-L1470). |
| `R_k` | `native.combined_reservation_bytes`. |
| `W_k` | Wave `additional_reservation_bytes`, zero only when absent. |
| `P_k` | Native `plan_reservation_bytes`; unrelated to `P_arrays`. |
| `D_k` | Native `template_construction_bytes`. |
| `C_k` | `tables.retained_bytes + tables.metadata_reservation_bytes` at finish. |

This conservative TARIFF is not RSS or unique physical backing. Unique backing,
gather/returned-output/input/source-cache overlaps and retirement remain open;
ordinary compiler Vectors are not automatically Budget leases. Transport's
[separate peak](../../crates/pllm-compiler/src/fss_decoder/transport.rs#L334-L350)
also retains worker/dealer exports, frame copies, wave and receive owners.

**Material/tape dependency.** [Scan](../../crates/pllm-compiler/src/fss_decoder/transport.rs#L287-L324)
derives `N` secret steps, `T_packet` cumulative packet bound, `P_packet` maximum
packet and `F` peer rounds from actual Plans/waves, not scalar output counts.
[External-AAD tape price](../../python/pllm/runtime/fss_tape.py#L170-L212) per holder:
`D_tape = min(T_packet, N*P_packet) + 40*N`. Two holders retain two tape copies;
live append/take buffers, metadata/AAD, exports and frames require separate overlap
accounting. The existing 32 GiB per-tape ceiling is distinct from aggregate
64 MiB native live admission; neither pays for the other or proves whole-host fit.

**Before bootstrap.** Source IEEE payload `b*n` (`b` bytes/element) converts to Q12 `8*n` retained
coefficient payload, not the complete source-owner size. Source import, rotary,
hashing and preparation precede later Schedule bootstrap: [source seal/compile](../../crates/pllm-compiler/src/fss_decoder/import.rs#L187-L244),
[`preparation_peak` / `construction_limit`](../../crates/pllm-compiler/src/fss_decoder/resource_quote.rs#L135-L175),
[`bootstrap`](../../crates/pllm-compiler/src/fss_decoder/resource_quote.rs#L219-L250); [Q12 conversion](../../crates/pllm-compiler/src/fss_decoder/source.rs#L94-L107).
Local phase gates are not retroactive payment for prior work; borrowing and hashing
mint no funding. Proof/numeric workspace remains a separate admission obligation.

**Lifecycle boundary.** Existing [`_CleanupSlot`](../../python/pllm/deployment/resources.py#L532-L595)
retains callback/payload only once installed; caller must drain producers and slot.
It proves neither constructor rollback nor final-parent policy. Worker
[`_Attempt._close`](../../python/pllm/runtime/fss_worker.py#L226-L301) owner-clears
and dependent tape/source-close uncertainties remain OPEN, not authorized fixes.
Mandatory preacquisition shells and failed-parent shutdown remain UNSELECTED
proposals, not adoption. Original-root work entry, repeated-walk prepayment,
all-outcome overlap/retirement and independent security remain gates. No broader
review, funding, source-faithfulness, fit or execution qualification follows.

## Ownership and transport

The dealer receives no private input or masked online openings. It delivers only
each recipient's fresh material to that holder's encrypted, ordered one-use tape.
After acknowledged setup, the client retires the dealer before sending either
input share. Online peer exchanges are direct between holders; selection and
feedback stay shared. Only the final fixed-count IDs reconstruct at the client.

Native canonical codecs check context, role, schedule and exact lengths. Failed
imports burn claims; cancellation drains callbacks and HTTP owners before capacity
refund. Tapes authenticate records, consume in order and reject replay. Returned
frames retain native leases; Python/HTTP copies have separate conservative owner
allowances. Tape deletion and reference retirement do not claim secure physical
erasure. Confidential authenticated remote channels and non-colluding operators
remain deployment assumptions; co-located benchmark children do not prove them.

The native live-material allowance remains 64 MiB. Request-long exact-key replay
ledgers now support bounded indexed heap or private-file storage; claims survive
individual kernel retirement. Tape admission derives record counts from complete
packet, metadata, framing and authentication-tag costs. Capsule retains its
256 MiB cumulative-packet and 1 GiB file ceilings. Explicit Decoder capacity has
a 32 GiB per-holder framed-tape ceiling and separately priced replay files.
Source, program, table/cache, callback and export owners retain separate bounds.
Shared immutable public tables do not permit mask/key reuse.

Decoder tapes use a distinct v2 encryption domain: the complete original metadata
remains AES-GCM additional authenticated data, supplied again by the native-bound
reader. Disk retains the 24-byte header, ciphertext and 16-byte tag per record.
Pricing is `min(cumulative_packet_bound, record_count * packet_bound) + 40 * record_count`;
metadata input and live-buffer bounds remain charged. Capsule and direct tape
callers retain inline metadata and the original v1 format by default.

The explicit `material_delivery="batch-v1"` choice coalesces up to 64 original
material records per authenticated HTTP body, bounded to 1 MiB including framing.
It reuses shared length-prefixed framing, native erasing packet owners and the
existing sequential tape append. Both holders acknowledge a complete batch before
the dealer advances; partial acceptance, lost acknowledgement or cancellation
retires the whole attempt. Larger admitted records use the existing singleton
path after queued records drain. Native queues and Python copies receive separate
charges under the existing aggregate ceiling. The checked SDK contract binds the
choice; numeric Program/source-body identities and default single delivery remain
unchanged. Batching does not merge one-use records or online peer exchanges.

The separate `online_schedule="wavefront-v1"` option executes public dependency
groups of the original consuming primitives. At most 256 ready primitives,
2 MiB of decoded party material and a 1 MiB framed opening fit one group; the
unchanged aggregate 64 MiB ceiling includes schedules, registers, exports and
retained frames. Exact kernel resources may reduce the group further. Every
primitive retains its own context, original packet and fresh one-use material.
Malformed groups, replay, partial replies and cancellation consume the active
owner. Public schedule construction and repeated metadata also count as work.

Bounded lane templates combine independent contractions, floors, activations,
normalization rows and attention work. Public matrix products retain complete
accumulation before their floor. Adjacent-pair private selection preserves the
lowest original index on ties and carries odd tails unchanged. Logical tree
depth and actual tile/exchange counts remain separate. Checked-range validity
stays secret and sticky through complete feedback and final masking. D15 records
the approved execution adaptation; numeric source identities stay comparable,
while Program, schedule and admission commitments bind the new order. Serial
execution remains the default. Full dense conformance fixtures use Decoder's
existing cumulative-work envelope; the new construction work can exceed Capsule
even when its live payload fits.

`online_schedule="tensor-v1"` additionally retains one immutable instruction and
register-description template per tile. Lane-specific private registers, keys and
wire contexts remain independent; logical instruction, register and work limits
still apply. Public dependency spans also retain one template; bounded virtual
lane/PC lookup reconstructs the same expanded dependency order without allocating
one ready-work record per lane. Sorting, address lookup and hashing remain charged.
A ready set may contain up to 4,096 primitives, constrained by the
same 2 MiB decoded-material, 1 MiB frame and 64 MiB aggregate bounds. Construction
prices the overlap between old and compact metadata before releasing either
owner, and whole-topology admission charges all three construction owners.
The ordered selection tree uses strict score comparison with left-on-tie payload
movement. Its compiler-owned ascending leaves and adjacent-subtree reductions
prove lowest-original-index ties, including odd tails, without privately comparing
index payloads again. The general arbitrary-index selector remains separate.
This selector and compact templates have a distinct Program/schedule identity;
the original Q12 numeric source contract remains unchanged.

Validated public lookup chunks now share immutable, independently leased native
backing across dealer issuance and holder import. Only original public values
are reused: every invocation still issues fresh masks, keys and private bases.
Raw-table generation, lowered payload construction, chunk headers/Arc counters,
retained backing and per-issuance/import handles have separate accounting. The
last Plan, cache or party handle retains its backing charge; exported packets
and import-transition copies retain their own charges. Duplicate one-use imports
reject before parsing or allocation, including after a failed import.

Normal regressions cover exact-budget overlap, retained-owner drop order,
allocation failure and cached/legacy parity. The opt-in generated-Qwen complete
decoder check also preserves frozen observations and private feedback. This is
storage reuse, not fewer protected operations, rounds or fresh private material.
Historical trained-source rejection and WAN results below retain their original
builds; they do not establish admission or speed for this ownership change.

Complete-vocabulary one-hot evaluation folds the DCF's natural-order leaves
directly into the existing output instead of retaining a second full-domain
array. Every padded leaf, PRG expansion, correction and logical work charge is
preserved. First/previous/current leaf storage costs 24 logical bytes per party;
at vocabulary 151,936, the removed heap array was 2,097,152 bytes per party.
Independent scalar-point tests compare every leaf and each party's output bytes,
not only reconstruction. This does not by itself prove a smaller complete-topology
reservation: transport pricing takes the maximum across all kernel families.

`capacity="decoder"` streams Q12 coefficients and row statistics into shared-core
immutable files. Public contractions reuse `pllm_core::streamed_matrix`; the
complete product still precedes the original FuseFSS floor. Complete vocabulary
lookup reuses the consuming one-hot primitive, and tiled stable selection retains
every head row. Shared source/tokenizer validation permits unmapped reserved head
rows without removing them from selection. The existing tokenizer omits such IDs
from text; the selected IDs remain unchanged.

Shared integer matrix/row-energy certificates tighten public admission bounds.
Their provenance follows complete semantic rows through supported views and
operators. They do not remove a normalization's input-domain obligation or change
the source tables, epsilon, two affine floors, or private multiplication schedule.

### Original-request custody contract (design only; activation held)

This section proposes custody changes, not current implementation guarantees.
Approval covers this design text only: source changes, tariffs, registries, trust
changes, API selection, builds and execution remain held. Existing ownership and
transport contracts above remain in force; no full-method qualification follows.

**Sealed provenance.** The existing route is `SourceBuilder::seal_source` →
`ImportedSource` → `seal_with_schedule` → `ImportedProgram` → `NativeProgram`.
Imported structure fields and `SourceExecutionBinding::new` are public today;
that binding commits semantic geometry, not actual payload. SDK publisher, file
and tokenizer identities remain caller-authentication claims, not native proof.
Propose a private, indivisible `ImportedProgram` retaining genuine import source
and tying actual `Program`, input-stream digest, execution binding and
`CompileOwnership`. Native custody would retain `Option<Arc<ImportedProgram>>`;
no constructor could accept independently supplied Program, binding or digests.
This supplies local provenance only, not new cryptography or external SDK
authentication. The sealed issuer route does not exist; construction stays held
until that route and every live caller migrate together.

**One original request.** Proposed private `SourceRequestCustody` wraps the
existing channel `Request`, authenticated source, role, delivery/receive owners,
spend, budget roots and lifecycle/admission state. Move existing accounting from
`Schedule`; do not add a parallel ledger. Current `Request` does NOT yet own this
ledger or sealed import; issuer/API migration remains prerequisite.
`Request::parse` and `ClientRequest`
validate framing only and cannot mint authority. Proposed lifecycle is
Pending → Admitted → Running → Terminal. Failure is terminal, with no same-owner
retry; this does not select or approve a new API. Debits never roll back. Dropping
a terminal owner neither resets spend through a fresh wrapper nor proves replay
protection across wrappers or restarts.

**Cumulative spend versus pass state.** `RequestSpend` would retain existing
`work`, `public_copy_bytes`, `material_bytes` and `frame_bytes` cumulatively,
using their exact formulas and debit order. Material/frame charges include
conservative admission projections; they are not measured delivered bytes.
`PassState` remains local: kernel ordinal, primitive count, transport counts,
selection node, validity, obligation hash and transcript. Runtime invocation
context using `resources.kernels` must keep the unchanged pass ordinal, not
substitute cumulative spend. Metadata, live-array and source-cache inventories
and peak overlap are live ownership terms, not sums over retired passes.
Original `Budget` high-water evidence stays monotonic across pass-local inventory/stat resets; this neither sums retired storage nor adds a cap.
Every pass retains a cold local cache and its own accounted statistics, preserving
hit/miss hashes. Cache construction debits the same persistent request owner at
the existing covered boundary; no warm-cache optimization or fabricated statistics.

Budget roots would be created once before the first cache. Each pass borrows
split roots and mutable spend fields; proposed `PassTables<'b>` owns a borrowed
cache. All plans, material and outputs retire before pass return. No self-reference,
unsafe lifetime extension or replacement `Budget` is permitted. Escaping owners
use existing `OwnedBudget` only when genuinely necessary, with bootstrap funding.
Separate guard, batch and receive capacities remain separate. Storage-lease reuse
may release live storage, never refund work.

Required order: source authentication/import → sealed join and request-role checks
→ existing debit → covered allocation → visit/callback → execute. Source-import
preflight precedes custody funding and remains a separate obligation. Preserve
refusal prefixes, overflow behavior and residual budgets; identify uncovered work
explicitly rather than claiming custody covers it. PRSS nonce, recipient, context,
`ImportGuard` and `issue_step` remain unchanged; partial delivery cannot retry.

**Constructor cancellation slot — PROPOSED, not implemented.** Slot plumbing is
selected for FURTHER DESIGN ONLY, not API implementation/adoption or funding.
In [`crates/pllm-python/src/fss_transport.rs`](../../crates/pllm-python/src/fss_transport.rs),
`Control` is created AFTER admission and owns no original source/spend/admission
payload; synchronous `__new__` returns after admission. In
[`crates/pllm-compiler/src/fss_decoder/channel.rs`](../../crates/pllm-compiler/src/fss_decoder/channel.rs),
`Request` carries framing/context and phase, not that custody. In
[`python/pllm/runtime/fss_worker.py`](../../python/pllm/runtime/fss_worker.py),
application `active` roots the longer-lived `_Attempt`, but `_close` continues
dependent tape/source cleanup after errors, clears native/source fields, and loses
the reservation field if its final `reservation.close()` raises. No current
endpoint preserves all original source, spend and backing through wrapperless
constructor failure AND true final-parent drop. A proposed one-use,
process-bound cancellation slot and `RoleRequest` share one `Arc<Control>` and one
Mutex state: Pending/Admitted/Running/Terminal plus separate cleanup status.
The cancelled Atomic is only a polling flag, never publication/run authority.
`SourceRequestCustody` retains the original `Request`, sealed source binding,
authentic role, moved four cumulative spend fields above and original budget roots.
Exactly one owner location exists at a time: partial-constructor guard, state
escrow, run guard or cleanup guard; a result has no independent authority.
`state.owner: Option<_>` is valid only with exclusive constructor/state/run/cleanup
transfers. Reject second attach, replacement and concurrent retirement claims;
NEVER overwrite occupied escrow.
An admission `Option` alone misses components acquired before a complete owner:
every partial component must enter guard custody immediately.

**Independent lifecycle review — REQUIRED CORRECTIONS, not clearance.** These
are proposed requirements, not current enforcement. Attach
and worker registration occur under the state Mutex BEFORE source clone/acquisition,
not merely before detachment. Cancellation winning rejects acquisition; attach
winning obliges close to await the registered worker. Never Confirmed → attach.
The constructor deposits its owner and completes registration atomically before
run claim. Panic, admission failure, late publication and Python-wrapper failure
retain partial/full candidates before worker acknowledgement, including cleanup
unwind. Under the same Mutex, cancellation winning forbids publication: late
candidates enter escrow only. Post-publication cancellation rejects new run,
metadata, source-borrow and callback claims while preserving live borrowers.
Confirmed requires closed registration gates, zero constructor/run/borrower/callback
participants AND successful cleanup; published, None or Terminal does not mean drained.
Count scheduled callbacks BEFORE dispatch; nested registration retains its parent
registration or refuses. No callback, owner destruction or drain under the state
Mutex. Python callbacks REQUIRE the GIL; blocking waits release it. Reject/defer
dependency self-drain, including event-loop callback → close → worker → callback;
thread-ID self-join detection alone is insufficient.

**SDK ordering and failed drains.** Install the slot before scheduling bind; register
the bind job before possible eager execution or yield. `_Attempt.signal` cancels
the installed slot before its existing stop-flag early return. Close that slot even
if construction raised before `attempt.native` assignment. Failed slot/native/job
drain stops DEPENDENT tape/source close and refunds; merely retaining fields is
insufficient. Clear each slot/native/source/tape/job/reservation manager field
ONLY AFTER its cleanup succeeds; failed dependencies retain owners AND backing.
Clear the reservation ONLY AFTER `reservation.close` succeeds;
failure retains the rooted active attempt and blocks replacement. A failed cached
close task preserves escrow; retry policy is unselected. Retain the source Arc
through borrowers; Python bytes/HTTP copies have separate reservation obligations.

**Shared parent-retirement endpoint — UNSELECTED candidate.** Reuse shared
source/request lifecycle machinery, not a paper-specific parallel owner. Reserve
one finite parent slot BEFORE any acquisition; the parent roots `Control`
independently of client handles. Register the partial-constructor guard before
acquisition; every acquired component enters its custody immediately. Exclusive
Partial → Full transfer preserves the original source, request, cumulative spend,
budget roots and backing; constructor/state/run/cleanup locations above never
duplicate authority. A cancellation-winning late constructor deposits partial/full
custody before acknowledgement and never publishes a result.

Claim cleanup ONLY after bind/constructor, run, borrowers and callbacks are globally
drained and registration gates closed. An in-place borrowing cleanup guard leaves
the payload rooted in the slot as Failed on error or unwind; dependent owners,
charges and reservation backing are not cleared/refunded. Release the slot only
after ALL cleanup succeeds. No callbacks, destructors or waits under the Mutex.
Cancellation requests retirement; it does not interrupt native execution or prove
drain. Last client Arc requests retirement while the parent remains alive.

**TRUE LAST-OWNER DROP — BLOCKER.** Parent rooting moves the final ownership
boundary; it does not solve it. True final owner cannot return from `Drop` while
retaining failed cleanup without an external root or deliberate leak; neither is
selected here, nor is a leak/abort policy prescribed. Finite slot occupancy does
not imply finite retention duration. An unresolved parent must remain owned;
shutdown cannot claim drain. No deadline release, automatic retry or leak-as-cleanup.
Mandatory preacquisition slots and fail-closed parent lifetime/shutdown require
NEW DESIGN approval, beyond the existing further-design slot selection. Source,
issuer, funding, API adoption and implementation remain held; prior approvals
are unchanged. This candidate establishes no current survival guarantee.

**Independent funding review — BLOCKERS.** `768S` covers only six `128S` lookup
constructor allowances, not pending/control/attach/publication/drain/cleanup.
Freeze actual native/Python representations, capacities, overlaps and last-drop
costs. Bound TOTAL registrations/attempts, not just concurrency: constructor/worker
dispatch, borrowers/callbacks/handlers, cancel/close and rejected attempts, drain
participants, retries and wakes. Condvar spurious wakes, repeated public calls,
arbitrary foreign callbacks and `to_thread`'s shared executor provide no finite
physical-work proof. A logical event contract alone does not fund the whole
lifecycle. Caps/tariffs remain unchanged; pass-local ordinals/transcripts/caches
stay separate from persistent four-field spend. No refund/reset of failed work.

**Migration boundary.** All live public `admit*`, `prepare*`, `execute*`,
`visit_material` and Worker callers must become owner-bearing; pricing remains
explicitly non-executing. SDK `fss_worker` burns IDs before work, but native direct
reparse bypass remains unresolved; all live calls require genuine custody, with no
ownerless compatibility bypass. No new registry/receipt supplies authentication,
and no cross-wrapper/restart replay
claim is approved. Migration spans channel/runtime, `fss_transport`, resource
quotes, import, SDK, `fss_source`, Python/native and direct Rust callers/tests.
Audit `Request`/owner size charges and Python ABI. Unchanged wire format is an
explicit proposal, not implied ABI approval.

**Held gates and future checks.** Activation requires sealed-issuer migration,
separate cancellation API implementation/adoption approval, resolution of the
lifecycle/funding blockers above, and feasibility under
unchanged caps with cumulative existing charges. Proposed regression coverage:
forged or mismatched provenance; cumulative failure without debit rollback;
pass-local ordinal/hash/cache parity; no issuance retry after failure; cancellation
and late-success drain; framing-only client authority; non-executing pricing.
These are future tests, not checks run or results established by this design.

The existing six constructor tariffs totaling 768S are already source-cleared;
this proposal does not claim they are absent or newly implemented. An extra six
and new +768S remain held. No approval or funding claim covers the whole twelve,
source import or preconstruction work. Implementation, tariff/registry/trust edits,
concrete API adoption and execution require separate approval; this design activates none.

## SDK and benchmark interface

`FuseFssSourceDecoder`, `TrustedDealerTwoShareRoles` and
`FixedCountPrivatePrefix` use ordinary component discovery, immutable
serialization and Experiment resolution. The typed profile selects all required
slots. Unsupported sampling, tools, retained KV and continuation reject.
Streaming API requests buffer text until complete final disclosure; EOS does not
shorten the selected count.

The reusable [generated factory](../../examples/papers/fusefss_smoke.py) and
[ordinary benchmark outputs](../../paper/artifacts/fusefss-sdk/README.md) exercise
both numeric profiles. The CLI, `BenchmarkTarget.response` and shared response
driver use the same role supervisor and resource admission. CPU/RSS is reported
per owner, dealer/holder/client application bodies are counted once per directed
edge, and tape I/O stays separate from network totals. Reports retain
setup-inclusive final disclosure and leave streaming TTFT/decode throughput null.
Metadata preflight and completed native program identity have separate scopes;
an absent role resource estimate remains unknown.

Worker phase counters separately measure dealer preparation, delivery and seal
callbacks, and each holder's native execution. Preparation outside callbacks
includes planning, issuance, encoding and bridge work. The reported maximum holder
duration is not the client online interval; client online time remains unknown.

A separate generated dense-RoPE sample records 3,875 material records and 4,513
peer frames per holder. Dealer preparation takes 5.396 seconds, including 5.279
seconds in delivery callbacks (97.82%). Maximum holder-native execution takes
3.683 seconds. The original ordinary report is
[`dense-phase-single.json`](../../paper/artifacts/fusefss-sdk/dense-phase-single.json).
Delivery callbacks include HTTP, receiver append and scheduling. This small
serial-HTTP sample supplies neither trained-Qwen timing nor a measured batching
improvement.

A subsequent fresh-client single/batch pair uses the same generated source,
numeric contract, private cohort and 1+2 workload. Dealer preparation falls from
5.393 to 0.390 seconds, delivery callbacks from 5.277 to 0.288 seconds, and full
request time from 9.680 to 4.796 seconds. Output digests agree; online bodies stay
7,681,488 bytes and each holder reads/writes the same 6,189,173 tape bytes. Covered
all-link bodies fall from 21,618,568 to 19,909,435 bytes through reduced delivery
envelopes and acknowledgements. Aggregate cold-first CPU falls from 11.022 to
6.956 seconds; maximum holder-native time rises from 3.565 to 3.656 seconds.
Both samples observe zero swap growth. This one-pair generated screen is separate
from trained-Qwen execution and independent confirmation. The ordinary report and
exact factories are linked from the [benchmark README](../../paper/artifacts/fusefss-sdk/README.md).

Decoder launch first admits a bounded source-qualification phase. One temporary
native owner supplies actual program and transport prices, then retires before
fresh whole-topology memory/disk admission and child launch. Workers independently
import and recheck the pinned source. Qualification time and CPU belong to ordinary
cold setup. A metadata-only preflight keeps conservative unknown transport costs;
it does not certify a whole request. Explicit Decoder deadlines may extend to
86,400 seconds, with remaining time reduced by imports and setup rather than
renewed by peer traffic.

Shared disk leases retain pending capacity and credit only verified physical
backing of live, same-device files, deduplicated by file identity. File length or
global free-space changes do not supply that credit; sparse replay capacity and
unwritten tapes remain pending. These are process-local admission reservations,
not filesystem quotas.

[`examples/papers/fusefss_conformance.py`](../../examples/papers/fusefss_conformance.py)
supplies the same frozen generated Qwen2 source for Capsule and Decoder. Ordinary
three-child SDK checks preserve three independent-oracle selections, private
feedback, more than 4,096 consumed tape records and cleanup. Decoder also exercises
file-backed source/replay owners and native-priced launch. Native tests compare
all 186 complete Qwen semantic observations; these are generated-model checks.

## Qualification boundary and next gate

No trained GPT-2/Qwen quality, matched paper performance or independent security
qualification is established. The static profile's earlier 1+1 source import
rejected numeric admission before worker launch. Its unresolved early obligation
is the next RMSNorm input after layer-zero MLP; later conditional propagation does
not discharge that obligation.

The separate D14 profile now imports and compiles pinned Qwen2.5 at 4+1 under
Decoder capacity, with zero observed swap growth. Its immutable source occupies
3,952,262,144 coefficient bytes plus row statistics. Native admission prices
2,602,128 kernels, 11,668,738 material records per holder and 13,424,643 peer frames
per holder. These are complete scheduled counts, not execution measurements.
Private range acceptance still requires the actual request to run. The initial
inline-metadata tape quote requires 71.53 GB of whole-topology disk including the
2 GiB reserve, and rejects the observed host before child launch.
External-AAD tape pricing reduces that quote to 47,630,190,304 bytes while
preserving all material records, complete metadata authentication and replay
storage. This is admitted storage arithmetic, not measured request execution.

A subsequent checked-range/batched Qwen attempt passes source and topology
admission and reaches online execution. After approximately 66 minutes in the
measured request window, macOS warning-level memory pressure triggers retirement
before final acceptance. The four-hour deadline remains unexpired; the pressure
observation alone does not attribute its cause to a process or allocation. The
[ordinary failed checkpoint](../../paper/artifacts/fusefss-sdk/qwen-checked-batch-pressure.json)
retains an incomplete status and no report. No token, numeric acceptance,
completed resource measurement or quality result follows. All role processes
retire. Online gate batching and callback-cost calibration are the next execution
gate; offline material batching does not reduce these peer exchanges.

The integrated revision passes 1,025 Rust library tests across core, garble and
compiler, 816 installed Python checks, and two additional three-child Decoder
tests. The latter consume more than 4,096 v2 tape records and enforce late
checked-range rejection followed by successful fresh-request recovery. These
tests retain their generated-model scope.

Trained execution, independent integer parity, quality and ordinary matched
benchmarks remain open. The common 64+8 envelope still exceeds sequence/selection
limits. The reusable pinned-source factory is
[`examples/papers/fusefss.py`](../../examples/papers/fusefss.py).
The [completion plan](../../paper/methods/fusefss-completion-plan.md) keeps these
remaining method gates explicit.

## WAN execution gate

Full-model completion requires measured execution with both access-rate caps and
propagation delay. The shared native relay now covers the complete FSS graph:
client controls, dealer-to-holder material delivery and both online peer paths.
Each party shares its upload/download allowance across peers. Same-party traffic
remains local. `--link-conditions` selects a delay-only `LinkConditions` document;
`--docker-network-profile` remains an alias.

```sh
uv run --no-sync pllm benchmark run \
  --experiment examples/papers/fusefss_smoke.py:dense_batch_smoke \
  --factory --trust-python --backend native --wan \
  --link-conditions examples/benchmarks/wan_20ms.json \
  --prompt hello --max-output-tokens 2 --timeout 600 \
  --capture-output-digest --output target/fusefss-generated-wan.json
```

This selects 100 Mbps download/40 Mbps upload per party and 20 ms of added delay
per inter-party direction: 40 ms added request/reply RTT. Propagation is pipelined
after rate admission, not a serial sleep for every read chunk. A shared eight-MiB
queue allowance, charged within the existing 32-MiB relay owner, bounds delayed
reads. Bandwidth-delay products beyond that allowance reject before startup.
Cancellation drains queued owners; reports bind and validate the selected delay,
rates, role routes and queue bounds. Native loss and a second rate limiter reject.

This is local TCP-payload emulation, not Internet measurement. TCP handshakes,
kernel headers, retransmissions and checkpoint distribution remain outside its
scope. Setup material, client/relay work and all routed role bodies remain charged.
Buffered final disclosure still supplies no streaming TTFT or decode throughput.
The scalar online schedule is not WAN-qualified; tensor-wide dependent rounds,
balanced private selection and preprocessing volume remain execution gates.

The matched online-schedule factories are
`examples/papers/fusefss_smoke.py:dense_batch_smoke` and
`examples/papers/fusefss_smoke.py:dense_wavefront_smoke`. The pinned checked-source
candidate is `examples/papers/fusefss.py:qwen_checked_wavefront_smoke`. Selection
alone supplies no trained acceptance or WAN performance result.

A separate fresh-client matched pair runs both generated factories at shared
100/40 Mbps caps and 20 ms per direction (40 ms added RTT). Both produce the
same output commitment with zero observed swap growth; native and SDK controls
above supply the independent tensor/selection and rejection evidence.

| Measured scope, one sample per arm | Serial | Wavefront |
| --- | ---: | ---: |
| Full request, including offline preparation | 222.505 s | 34.583 s |
| Maximum holder native-run elapsed | 212.013 s | 24.251 s |
| Dealer preparation elapsed | 8.801 s | 8.469 s |
| Covered all-link application bodies | 19,909,435 B | 14,022,118 B |
| Covered online application bodies | 7,681,488 B | 1,716,046 B |
| Startup-through-response TCP streams | 24,273,664 B | 14,584,395 B |
| Client lifetime peak RSS | 112,312,320 B | 112,295,936 B |

This is a generated hidden-width-two, one-layer, three-token-vocabulary 1+2
workload. Its 6.43-fold request reduction and 8.74-fold holder reduction do not
project full-Qwen performance. Client RSS is effectively unchanged. Fresh
preprocessing remains charged; checkpoint distribution, physical TCP/IP overhead,
Internet measurements and independent operators remain outside scope. The
ordinary report is [`fusefss-wavefront-wan-pair.json`](../evidence/fusefss-wavefront-wan-pair.json).

```sh
uv run --no-sync pllm benchmark run \
  --experiment examples/papers/fusefss_smoke.py:dense_batch_smoke \
  --experiment examples/papers/fusefss_smoke.py:dense_wavefront_smoke \
  --factory --trust-python --backend native --wan \
  --link-conditions examples/benchmarks/wan_20ms.json \
  --prompt hello --max-output-tokens 2 --timeout 600 \
  --capture-output-digest --isolate-candidates \
  --output target/fusefss-wavefront-wan-pair.json
```

### Trained-source scheduling gate: wavefront

Source-only qualification of the pinned checked Qwen candidate at 4+1 completes
range construction but rejects the kernel walk at layer 23 SiLU: cumulative public
copies reach 34,371,192,850 bytes, above the unchanged 34,359,738,368-byte ceiling.
This includes the new public schedule construction; it is not a private-value
rejection or a completed request. No workers launch.

A separate 1+1 source/pricing control admits 29,334 peer exchanges per holder,
4,144,944 fresh primitive records per holder, and a 64,688,850-byte combined native
transport reservation under the 64 MiB ceiling. Native packet pricing is bounded
by 5,181,796,140 bytes per holder, before HTTP metadata; that bound is neither an
observed transfer nor a transfer lower bound. Complete source/tape/guard/reserve
disk pricing is 25,317,061,976 bytes. Qualification retires its source owner and
observes zero swap growth, without launching workers or executing private input.

At 40 ms RTT, the current sequential HTTP exchange schedule alone implies about
1,173 seconds (19.6 minutes) of latency waits for that 1+1 control. The logical
selection tree therefore does not make the bounded tiled schedule WAN-practical.
This motivated compact public schedule ownership and wider bounded tensor waves;
the subsequent qualification below measures that implementation separately.
Fresh preprocessing delivery remains a separate cost gate. These estimates do
not qualify trained acceptance, quality or serving.

### Compact tensor schedule: generated WAN pair

A new fresh-client pair compares `dense_wavefront_smoke` with
`dense_tensor_smoke` under one private cohort, the same generated 1+2 workload,
shared 100/40 Mbps caps and 20 ms propagation per direction. Output commitments
match, every report check passes, and both memory guards observe zero swap growth.

| Measured scope, one sample per arm | Wavefront | Tensor |
| --- | ---: | ---: |
| Full request, including offline preparation | 35.749 s | 34.732 s |
| Maximum holder native-run elapsed | 25.299 s | 24.503 s |
| Dealer preparation elapsed | 8.656 s | 8.387 s |
| Aggregate startup-through-first-response CPU | 9.769 s | 7.310 s |
| Covered all-link application bodies | 14,022,118 B | 13,916,321 B |
| Covered online application bodies | 1,716,046 B | 1,690,014 B |
| Startup-through-response TCP streams | 14,584,397 B | 14,467,175 B |
| Client lifetime peak RSS | 113,410,048 B | 111,804,416 B |

This separate pair observes a 2.84% request-time reduction. It does not isolate
template sharing from ordered-selection changes or establish a causal memory
gain. Its tiny generated geometry supplies no trained-model speed projection;
checkpoint distribution, physical TCP/IP overhead, Internet measurements and
independent operators remain outside scope. Buffered disclosure supplies neither
streaming TTFT nor decode throughput. The ordinary report is
[`fusefss-tensor-wan-pair.json`](../evidence/fusefss-tensor-wan-pair.json).

```sh
uv run --no-sync pllm benchmark run \
  --experiment examples/papers/fusefss_smoke.py:dense_wavefront_smoke \
  --experiment examples/papers/fusefss_smoke.py:dense_tensor_smoke \
  --factory --trust-python --backend native --wan \
  --link-conditions examples/benchmarks/wan_20ms.json \
  --prompt hello --max-output-tokens 2 --timeout 600 \
  --capture-output-digest --isolate-candidates \
  --output target/fusefss-tensor-wan-pair.json
```

The native composition suite passes 110 checks, including compact-versus-expanded
dependency order, 1,024-lane execution, fresh issuance and allocation-failure
retirement. The 118 decoder checks include all five frozen semantic fixtures,
checked-range rejection and stable ordered selection. The rebuilt installed wheel
passes 408 focused Python checks plus four three-process SDK/gateway checks.

### Historical trained-source scheduling gate: tensor

The pinned `examples/papers/fusefss.py:qwen_checked_tensor_smoke` 4+1 source
qualification completes public range construction but rejects the kernel walk at
the output head: cumulative public copies reach 35,178,175,330 bytes against the
unchanged 34,359,738,368-byte ceiling. This is a resource rejection before worker
launch, not a checked private-input rejection or accepted request.

A separate source/pricing control selects the same factory with
`budget__max_input_tokens=1`. Source pins, numeric profile, vocabulary and selection
count match the earlier wavefront control. Tensor pricing admits 19,229 peer
exchanges per holder (34.45% fewer), 3,081,399 fresh primitive records per holder
and a 61,851,153-byte combined native transport reservation. Complete native
material is still 7,451,946,472 bytes. The per-holder packet allowance is
4,168,085,820 bytes, excluding HTTP metadata; it is an upper-bound price, not a
measured transfer or traffic lower bound. Two encrypted tapes plus replay guards
require 9,136,331,688 bytes, with source-file ownership charged separately.
Qualification retires its source owner and observes zero swap growth; it runs no
private input and launches no workers.

At 40 ms RTT, sequential peer exchanges still imply about 769 seconds (12.8
minutes) of waiting before compute. Compact templates and the ordered tree do
not make this trained workload WAN-practical. Further round and fresh-material
reductions, four-input resource admission, trained acceptance and quality remain
open; the capacity ceilings have not changed.

The ordinary [generated response report](../evidence/fusefss-generated-wan.json)
records two final outputs in 222.124 seconds, with 7.766 seconds of dealer
preparation and 212.664 seconds for the maximum holder-native run. Covered
all-link bodies are 19,909,435 bytes, including 7,681,488 online bytes. All runtime
checks pass and observed swap growth is zero. This single transport qualification
run, which overlaps short regression checks, is a negative practicality result;
it supplies no trained-Qwen timing or matched speed comparison.

### Shared-public-chunk source retry

After immutable public-chunk reuse, the installed-wheel pinned-Qwen source tests
pass at both 1+1 and 4+1 under the unchanged numeric, work, public-copy and native
memory ceilings. These are the ordinary opt-in
`test_pinned_qwen_checked_tensor_source_admission` cases in
[`tests/test_fss_checkpoint_admission.py`](../../tests/test_fss_checkpoint_admission.py).
They execute the full native source/program admission walk, not just metadata
preflight. The earlier 4+1 public-copy rejection above remains historical evidence,
not the current source-admission outcome.

Source qualification retains the ordinary memory/watchdog policy and 2 GiB disk
floor, but creates no tapes or workers. A separate opt-in
`test_pinned_qwen_checked_tensor_topology_admission` obtains native transport prices
from the live source owner, retires that owner, then rechecks and reserves complete
topology memory and disk. Source-only success does not discharge that gate.
Neither test makes material, launches roles or executes private inputs. An accepted
checkpoint-backed SDK request, complete integer parity, quality and matched WAN
performance remain separate outstanding gates.

Before the one-hot staging fold, the complete-topology test passes at 1+1 with
the ordinary 25% safety margin:
required host memory is 4,068,474,880 bytes and required disk is 23,204,557,736
bytes. Native transport's combined logical reservation is 62,577,393 bytes,
below the unchanged 67,108,864-byte ceiling. These are admission prices and
process-local logical reservations, not measured RSS, allocated tapes or a
cross-process/filesystem quota.

The 1+1 control is not an ordinary SDK serving workload. Responses renders role
prefixes before tokenization: with the pinned tokenizer, each of `A`, `hi`, `!`
and `hello` becomes three input tokens after the unchanged `user: ` prefix.
That exceeds this control's one-input-token budget. Do not bypass rendering or
submit raw token IDs to turn admission into an SDK acceptance claim.

At 4+1, that pre-fold rebuilt-wheel retry completes source admission but rejects in native
transport pricing at `aggregate_reservation`, with `Resource`: 68,913,145 bytes
against the unchanged 67,108,864-byte limit, a 1,804,281-byte excess. The final
host/disk admission is not reached. The public owner terms are:

| Combined reservation term | Bytes |
| --- | ---: |
| Metadata | 25,227,438 |
| Live arrays | 15,943,552 |
| Public source-table cache | 1,208,520 |
| Transport peak | 13,330,075 |
| Fixed metadata | 65,536 |
| Native batch backing | 2,123,880 |
| Python batch backing | 11,014,144 |

These sum to the rejected reservation; they are logical admission charges, not
measured RSS. No cap, margin or owner charge is reduced to accept the request.

After the one-hot staging fold (`607813fd`), fresh installed-wheel retries retain
the same caps, source pins, workload and guards. The 1+1 complete-topology control
passes again, with a 60,448,276-byte native combined reservation and unchanged
host/disk prices above. The 4+1 native transport reservation now passes at
66,784,028 bytes, leaving 324,836 bytes below the 67,108,864-byte cap. The measured
admission-price reduction is 2,129,117 bytes for each workload; this is the result
of the complete pricing walk, not multiplication of the removed per-party array.

The 4+1 retry then reaches final host/disk admission and rejects **disk headroom**:
45,517,613,936 bytes are required, including the unchanged 2 GiB reserve, against
41,405,534,208 available bytes. Required host memory remains 4,068,474,880 bytes;
host memory admits with normal observed pressure. No tapes, material or workers
are created. Native aggregate admission is therefore cleared for this workload,
but complete-topology admission, private-request acceptance and quality remain
unqualified. Ordinary pytest output is retained under ignored
`target/fusefss-one-hot-fold-topology-{1,4}.log`.

The separately bounded `qwen_checked_tensor_minimal_smoke` factory uses 3+1,
enough for the pinned tokenizer's ordinary `user: hi` rendering. Its guarded
installed-wheel complete-topology admission passes at a 64,155,060-byte native
combined reservation, 4,068,474,880 required host bytes (normal 25% margin), and
37,476,300,632 required disk bytes (including the unchanged 2 GiB reserve).
The source qualification owner retires before the full-graph reservations.
This test creates no workers, tapes or material; it establishes neither a private
response nor quality, and does not qualify the original 4+1 workload. Ordinary
pytest output is `target/fusefss-one-hot-fold-topology-3.log`.

The subsequent ordinary 3+1 `benchmark run` attempt used that factory, prompt
`hi`, one output token, no warmups, one repetition, a 1,800-second timeout, and
the normal 25% memory margin. It stopped with CLI exit 6:
`benchmark stopped: macOS memory pressure is warning`. Its ordinary progress
report has `status=failed`, `complete=false` and no candidate result; this is not
private-response acceptance or a performance measurement. The report and CLI
output are `target/fusefss-qwen-3-plus-1-sdk.{json,log}`. Pressure guards remain
enabled; a fresh capacity/pressure check is required before retrying.

Future ordinary failed benchmark JSON can include optional
`failure.diagnostics` (`pllm.benchmark_failure_diagnostics.v1`). This fixed,
allowlisted record is capped at 4 KiB and captures the first failure before cleanup: observed lifecycle
phase and role, elapsed time, live qualification priced/completed/retired flags,
spawned/ready roles, the triggering watchdog observation, and cached owned-process
RSS samples/peaks when available. Unknown observations stay null; no prompt,
payload, credentials, source paths or event history are included. Capture adds no
process scan and does not change admission or pressure guards. Cancellation is
published before diagnostic reads; contended diagnostic locks yield unknown
observations rather than delaying shutdown. Request failures are observed before
request cleanup; final-check failures use a separate pre-teardown fallback that
cannot overwrite an earlier failure. Sampled RSS is not
an allocation census, can miss peaks/count shared pages, and excludes OS file
cache. Phase breadcrumbs do not identify the cause of pressure or prove private
execution. These diagnostics cannot retrospectively locate the older failed run.

Program-pricing admission now has a typed diagnostic path sharing the original
admission walk and limits. Its bounded public record identifies the enclosing
stage, error category and, only when known, kernel ordinal, observed bound and
aggregate owner terms. Unknown nested bounds remain unknown. The pricing-only
Python exception uses a fixed 1,024-byte allowlisted formatter; live role and
callback errors retain their generic rejection. Successful admission identities,
prices and delivery schedules are unchanged. The diagnostic retry located the
pre-fold aggregate failure; it does not establish trained-request acceptance. Focused
installed-wheel native transport, diagnostics, benchmark and boundary guards pass
126 tests. No worker launch or private execution occurs in the admission retry.
