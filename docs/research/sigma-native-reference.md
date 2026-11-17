# Bounded SIGMA native references

Source arithmetic: [SIGMA](https://petsymposium.org/popets/2024/popets-2024-0107.php),
§3.2.1 masked multiplication and §4.1.1 matrix multiplication. The reference
independently implements these equations; upstream code is not a dependency.

The original matrix reference is a **single-process, coupled two-share harness**.
The separate `matrix::party` API now owns each party's material independently,
reusing the same equations and bounded contraction core. Neither API supplies a
SIGMA decoder, independently hosted implementation, SDK component or security
qualification. The approved GPU-derived DPF port is a separate primitive under
decision D16; it is not used by this matrix product.

## Arithmetic and reuse

`pllm_garble::masked_wire_reference::matrix` extends the scalar negative-mask
product algebra without scalar difference openings. For caller-supplied masks
`RX`, `RW`, `RY`, issuance shares:

```text
A = -RX
B = -RW
C = RX·RW + RY
Y_party = A_party·W_hat + X_hat·B_party + C_party
          + [party == 1] X_hat·W_hat
```

With `X_hat = X + RX` and `W_hat = W + RW`, reconstruction gives the complete
`XW + RY mod 2^n`. Widths 1–64 and input/output or output/input coefficient
layouts are explicit. No intermediate floor, rescaling, clipping, bias or
float conversion is supplied. This is not a complete private linear layer.

Scalar products retain their existing expression, random draw order and encoding.
Resident private contractions reuse the allocation-free ring64 traversal in
`pllm_core::streamed_matrix`; private coefficients never enter a public-file
owner. Existing streamed public products keep their arithmetic order and layout.

## Ownership and admission

Each input, weight and output matrix is limited to 256 elements. Complete work
is priced as six contractions and rejects above `2^20` MACs. Per-party logical
key size is `n*(a+b+c)` bits, where `a`, `b`, `c` are the three matrix sizes;
physical storage uses u64 elements rather than packed keys.

The shared logical budget reserves `8*(6a+6b+7c)` payload bytes plus typed owner
metadata, 16 KiB stack allowance and per-allocation allowance under the existing
64 MiB ceiling. It includes admitted caller masks, masked operands and retained
clear inputs as well as native overlap. Excess caller capacities, aliases and
unrelated allocations remain additional owners. These allowances are not process
RSS, allocator or stack guarantees.

Only one operation is live process-wide. Borrowed non-Send handles retain the
complete reservation until the operation owner drops, after output buffers.
Duplicate imports, malformed bindings, cancellation and dropped unfinished
handles retire the coupled operation. Fallible exact-capacity allocations and
zeroizing buffers preserve erase/deallocate-before-refund ordering.

Issuance freshly samples correlation splits and an operation nonce. It accepts
the masks from its caller: **it does not sample them or certify that the caller
has not reused them**. There is no authenticated transport, replay persistence or
malicious-output verification. Both materials reside in one operation owner;
the evaluator API is not an isolated worker security boundary. Production
integration still needs context-bound import/export, fresh wire-mask ownership
and complete transport/failure lifecycle.

### Separate party-local matrix owners

`matrix::party::issue_local` returns two non-cloneable owners. Each holds only
its own A/B/C material; neither accesses a sibling or a coupled secret owner.
`begin` consumes fresh material and emits one output-share frame. `finish`
validates the complete peer frame before reconstructing the common masked output
in its existing local allocation. Context, fresh issuance, operand digest, sender,
phase, exact count and canonical residues are bound. Failure consumes only the
local party; the coordinator must cancel its sibling explicitly.

Frames and masked outputs independently retain split logical reservations and
the shared single-live-operation guard until erasure and deallocation. The
original coupled adapter and its allocation quote remain unchanged. Both APIs
share the same process-wide gate: a retained frame or output blocks new issuance.

For `P=8*(a+b+c)` and `F=76+8*c`, each party reserves
`P+8*c+F+owner_metadata+5*64+8192` bytes. The dealer additionally reserves
`8*c+dealer_metadata+64+16384` bytes and refunds this scratch only after its
buffers retire. `resources()` reports the combined peak and each component.
Caller-owned masks, common operands, retained clear values and excess backing
capacities remain additional outer owners. These are logical bounds, not RSS.

The local dealer still accepts caller-supplied masks: separate owners do not
certify freshness. This raw API has no material importer, authentication,
malicious-output verification, independent hosting or persistent replay ledger.
The typed consuming handoff below retains additional provenance. Public-API tests
chain both matrix layouts into the existing TR/GapARS/DReLU owners against
independent ring and signed-floor equations; that untyped fixture is not graph
provenance or a production composition.

### Bounded typed local composition

`masked_wire_reference::typed` wraps the original matrix and arithmetic owners;
it does not translate FuseFSS shares or change their arithmetic. Shared
`pllm_core::edge_contract` descriptors bind source, shape, numeric representation
and weight visibility. Descriptors are public declarations, not executable
capabilities, authenticated source identities or range proofs.

The local trusted dealer owns fresh root masks and returns separate, non-cloneable
ring capabilities for each party. Consumer, role, source/plan/session, edge,
numeric and mask identities bind each consuming operation. Outputs retain their
original native backing, logical leases and matrix singleton guards. Ring and
XOR-masked predicate outputs have distinct Rust types. A failed party consumes
its local state; the caller must still cancel its sibling.

`BoundRange<LocalAssumption>` records a developer premise checked by the local
root encoder, **not a compiler-derived or checkpoint-certified proof**. Checked
interval multiplication and complete-dot accumulation can retain signed no-wrap
provenance; explicit `MatrixRange::RingOnly` discards it. TR reduces the ring and
fractional width while preserving signed-floor interpretation. GapARS accepts
only a derived unwrapped interval inside the required quarter-ring domain;
ring-only provenance cannot qualify it. DReLU returns a typed predicate, not a
ring operand.

Aggregate logical admission prices root masks, encoded owners, metadata and
native operation overlap before issuance. Caller-owned clear buffers and excess
capacities remain separate. This single-consumer adapter supplies no fanout,
remasking, unrestricted byte imports, authenticated transport, compiler schedule,
decoder or SDK activation. Its local matrix → TR/qualified GapARS → DReLU controls establish
bounded ownership and arithmetic composition only, not composed security,
checkpoint ranges or trained-model execution.

### Consuming local matrix-material handoff

The bounded handoff retains typed provenance rather than adding a public
raw `Party::import(bytes, metadata)` constructor. `MatrixParty::export_material`
consumes a fresh owner and returns a handoff whose parts are a non-cloneable,
nonserializable import permit and a zeroizing secret packet. Import consumes both
on success or failure; imported owners cannot be exported again. Original typed
input capabilities remain required for execution.

The descriptor binds both complete input bindings, the output binding, approved
implementation, role, matrix geometry/layout, interval provenance and native
issuance. The `PLLMSM01` envelope carries that descriptor digest, native issuance,
exact body length, and canonical u64 little-endian A/B/C arrays, followed by a
SHA-256 checksum. Its fixed overhead is 120 bytes. The permit additionally retains
the complete original packet digest, so rewriting material and recomputing the
checksum cannot authorize replacement material.

This is same-budget, process-local ownership continuity, **not authenticated
delivery or cross-process/restart replay prevention**. Bytes and public descriptors
cannot mint permits. A future remote importer must authenticate issuer, recipient,
issuance and descriptor and preserve receiving-endpoint replay state. Secret
material also requires confidential delivery.

Native execution reservations and typed output reservations transfer without a
refund/re-reserve gap. Packet and permit independently retain the original matrix
singleton guard; no drop-and-reacquire admission occurs. Additional handoff
admission charges packet backing, owner metadata, allocator and bounded codec
workspace before encoding, including material/packet export overlap and
packet/restored-material import overlap. Caller-created byte copies remain
separate owners. These are logical capacity charges, not process-RSS bounds.

With `P=8*(a+b+c)`, the packet backing is `120+P` bytes. The native incremental
quote adds both native owner sizes, a 64-byte allocator allowance and 8 KiB
workspace. The typed quote adds permit/handoff/descriptor metadata, the packet
wrapper and another 8 KiB workspace; `matrix_material_resources` exposes each
component. Original native material and typed wrapper/output reservations remain
additional live charges throughout. Exact-charge tests independently retain the
packet or permit, preventing admission credit from moving to the shorter-lived
owner.

Packet storage and envelope mechanics reuse `material_wire`; FuseFSS keeps its
original public paths, `PLLMFW01` bytes, budgets, replay ledger and width-50 body
reader. SIGMA's body validator separately handles canonical widths through 64.
Arithmetic-material serialization and remote import remain later gates.

### Remaining decoder gates

The typed chain is not a complete masked-wire algebra. DReLU produces a common
XOR-masked predicate, not an arithmetic bit. The direct Select below preserves
that distinction without a predicate conversion. D19's bounded source lookup
below has its own bounded native executor. D20's fixed GELU/SiLU region below
composes that lookup internally; general typed lookup and complete nonlinear-row
coverage remain absent. FuseFSS additive-share operations are not substitutes.
Source CPU activation performs
the early floor before absolute value and clamping, as specified in
`paper/reproduction-contracts/shared-profile-resolution.md`.

Complete source nonlinear rows, private-weight embedding/head, stable selection
and private feedback still need executable composition and compiler-derived
ranges. Existing single-consumer and matrix-singleton constraints cannot be
silently relaxed for residual fanout or subsequent matrix operations. The approved
full-prefix, fixed-count, final-only disclosure policy does not authorize
persistent KV, streaming token disclosure or a new Qwen numeric mapping.

## Bounded direct five-word Select

`masked_wire_reference::selection` implements the pinned source's five-word
selection under `pllm.sigma.five-word-select.v1`. It accepts a common XOR-masked
predicate `bhat = b XOR beta` and an independently masked ring input
`xhat = x + rx`, returning the common masked ring result `b*x + rz`. The trusted
local dealer samples fresh full-ring `rz` and separately shares these five words:

```text
beta, rx, beta*rx + rz, 2*beta, -2*beta*rx
```

For each party, the local contribution is
`k2 - k0*xhat - k1*bhat + [bhat=0]*(k3*xhat+k4)`; party One additionally adds
`bhat*xhat`. One peer exchange reconstructs only the masked result. In particular,
the evaluator never treats the XOR-masked predicate as an arithmetic share.
All equations wrap in the declared `2^n` ring; no truncation is supplied here.

Input widths remain `3..64`, lanes `1..256`, under the existing combined 64 MiB
logical ceiling. Geometry, three distinct wire identities, canonical bits and
residues precede allocation and entropy. Frames bind both complete input views,
role, phase, plan/session/operation, wire identities, index and a fresh nonce.
Malformed inputs, repeated calls, bad replies and cancellation consume local
state; the caller explicitly cancels its sibling. Retained frames and outputs
keep independent leases through erasure and deallocation. There is no material
codec, authenticated delivery, malicious-output check or durable replay ledger.

With `L` lanes and `S(T)=size_of::<T>()`, native reservation is:

```text
F = 76 + 8L + S(Frame) + 64
O = 8L + S(MaskedOutput) + 64
P = S(Party) + 64L + 128 + F + O + 8192
D = S(OutputMasks) + 24L + 64
combined = 2P + D
```

The party's `64L` includes `40L` private words, `8L` local output and `16L`
borrowed inputs; the dealer's `24L` includes output masks and borrowed masks.
Each party receives `5*n*L` logical key bits. Issuance uses one 32-byte nonce draw
and six 8-byte draws per lane (`32+48L` bytes), no AES/DPF work, and five dynamic
allocations. Begin allocates one frame; finish allocates none. These logical
charges are not wire serialization, process RSS or whole-model costs.

`typed::LocalDealer::select` consumes one predicate-mask capability and one ring
mask; `SelectionParty::begin` consumes the corresponding two evaluator
capabilities. Source, session, namespace, role, consumer and full bindings remain
checked. Output width and Q scaling equal the ring input's; its signed interval
is the conditional zero/input hull, while absent provenance stays absent.
Original native owners move intact; wrapper and native peak charges coexist with
borrowed input owners. Separate roots establish no predicate/input relation and
do not implement activation fanout. The separately approved fixed prefix below
supplies only its precommitted internal uses; this primitive is not GELU/SiLU,
a decoder or an SDK component.

### Distinct consuming deferred Select

`selection::issue_deferred` binds a distinct boundary and returns the same
five-word correlations, not a reconstructed-output owner. Fresh-only
`Party::into_deferred` consumes its party and returns that party's additive share
of `b*x+rz`. It moves preallocated backing without a frame or allocation. Wrong
mode, bad inputs and repeated execution consume local material; the caller still
retires its sibling. The share exposes only its own low-level values and
role/binding/geometry, never a peer share. This boundary alone does not enforce
matching common input views; enclosing lookup validates those views first.

`resources_for` retains the reconstructed formula above. Deferred mode replaces
`F+O` with `8L+S(DeferredShare)+64`, and reports zero emitted frames. Both original
party and returned share metadata are priced through transition overlap. Share
backing erases and deallocates before its split reservation refunds. No transport
payload can construct a share capability or relabel a `MaskedOutput` as a share.

## Bounded source lookup — D19

`masked_wire_reference::source_lookup` independently implements the approved
`LUT_dfpet` correction with D16's ascending GPU127 expansion. The pinned source,
approved algebra and physical frame contract are in
`docs/plans/sigma-source-lookup.md`; D19 records approval in
`paper/methods/DECISIONS.md`. The complete garbling suite passes 513 unit tests
(eight opt-in skips), 25 external/oracle tests and 17 documentation tests.
These are bounded local correctness and ownership checks, not security review.

Geometry is exactly 8- or 10-bit indices, 3–64-bit outputs and 1–256 lanes.
Caller-supplied tables contain exactly 256 or 1024 canonical ring residues.
The primitive neither generates source-libm activation tables nor certifies
their approximation quality. Full shape, canonical masks/table, distinct wires
and index-overflow checks precede aggregate reservation, allocation and entropy.

For common `Xhat=x+rin`, the DPF target is `-rin`. Both parties scan the complete
ascending expansion against `T[(i+Xhat) mod N]`, without private-indexed table
gathers or reversed terminal offsets. Signed full-population counts, not assumed
local ±1 counts, supply the modulo-four sign correction. The first exchange
opens both `R=σ*T[x]+r` and `C=B XOR q`, where `B=(σ+1)/2`.
Fresh full-ring `r,u` and Boolean `q` retain the source's mask dependency:

```text
Select output: S = B*(σ*T[x]) + u
final mask:    rho = 2*u-r
reconstructed: 2*S-R = T[x]+rho
deferred:      2*z0-R, 2*z1, with z0+z1=S
```

Only party Zero subtracts common R in deferred mode. Derived rho is marginally
uniform, with `rho mod 2 = r mod 2`; no independent remask is implied. Deferred
shares have distinct consuming owners, not reconstructed outputs or decoded frame
payloads. Later reconstruction remains an additional operation and cost.

The first `PLLMLU01` frame is `80+16L` bytes: complete invocation/input digests,
role, phase, lane count, widths/boundary and lane-major u64 arithmetic/correction
words. Canonical correction words are exactly 0/1. Reconstructed mode then uses
Select's unchanged `76+8L` frame. Both-direction totals are `312+48L` bytes for
reconstructed and `160+32L` lookup-local bytes for deferred. Neither is upstream
wire parity, authenticated transport or complete decoder traffic.

`resources` prices actual inline GPU127 lane owners, both immutable table copies,
mask/working vectors, original nested Select owners, frame/output metadata,
borrowed receive/input/table allowances and transition overlap. One lane-serial
N-byte full expansion uses the admitted exact-capacity allocator; nonempty or
wrong-capacity scratch rejects before traversal. Aggregate admission reserves
once under 64 MiB and splits original leases into Select rather than releasing
and reacquiring capacity. Retained frames and outputs keep their own charges
after cancellation or dealer-first drop. These are logical local-owner bounds,
not measured process RSS or guarantees for arbitrary caller/transport copies.

Regression tests independently inventory padded owners across every admitted
geometry, check exact/one-byte-short admission, isolate caller table/mask mutation,
measure issuance/expansion AES counts, and inject entropy/allocation failures.
Moderate n10 lanes sweep all allocation callbacks; maximum lanes cover issuance,
scan boundaries and all frame/Select transitions. Both lookup boundaries retain
their original leases through drop permutations.

Full table/numeric identity, source pin, geometry, boundary, plan/session/operation,
every wire, index and fresh issuance bind execution. Complete received frames
are validated before mutation. Failure burns all remaining local stages; the
caller cancels its sibling. No arithmetic-material codec, authenticated delivery,
persistent replay protection, standalone typed lookup, decoder/SDK selection
or independent security qualification is supplied.

## Fixed owning activation prefix

`typed::LocalDealer::activation_prefix` consumes a Q12 input-mask capability and
issues separate party-local owners for exactly `Y=TR6(X)`, `D=[Y>=0]`,
`R=Select(D,Y)`. Input widths are 12–64, lanes 1–256; Y/R use the reduced
(n−6)-bit Q6 ring. Existing Q12 metadata rejects widths 9–11. Signed floors,
zero's nonnegative predicate and D16/D17 masks remain unchanged. This prefix
selects Y, not the original X selected later in complete source activations.

`PrefixSpec` binds complete input/intermediate/output descriptors and the next
consumer. Shared core `ConsumerUse` metadata declares each operand position and
ordinal but grants no authority. A private four-slot ledger additionally binds
actual source/session/namespace/issuance/mask/role capabilities: X/TR, Y/DReLU,
D/Select, Y/Select. Each party consumes its external X once and retains the same
original Y backing for both internal reads. No intermediate capability escapes;
there is no exported fanout, cloning, remasking or arbitrary graph execution.

All three native issuances and their wrappers/workspace are admitted together
before randomness. The incremental peak is the sum of the unchanged TR6,
DReLU and Select combined reservations, two party-wrapper reservations, a final
mask-wrapper reservation and issuance workspace. Each party wrapper prices its
inline owner, final common-output wrapper, four retained frame/step wrappers and
8 KiB transition allowance. Issuance additionally prices two inline party owners,
the spec, four bindings, three native contexts, final mask wrapper and 16 KiB
workspace. Existing input/backing reservations remain independently live.
Native leases move intact; cancellation erases/drops local state before refunding
workspace. A retained cancelled or completed party keeps its inline metadata
charge until drop; externally retained frames keep their native/wrapper leases.
Matrix-backed X retains its original singleton until that backing retires; final
arithmetic R does not acquire a synthetic matrix guard.

Execution has four peer exchanges: two TR phases, DReLU, Select. Eight directed
frames cost exactly `8*(76+8*lanes)` bytes, excluding downstream consumption.
Failures burn every remaining local stage; the caller must cancel its sibling.
Outputs remain opaque common-masked capabilities usable by existing consuming
typed operations. This supplies local ownership composition, not authenticated
transport, compiler/checkpoint ranges, security qualification or a runnable SDK
decoder. Numerical and public-API contracts/tests:
[`sigma-activation-prefix-contract.md`](sigma-activation-prefix-contract.md).

D19 separately supplies the native lookup above, not an extension of this region.
D20 separately approves the full owning region below:
[`../plans/sigma-full-activation.md`](../plans/sigma-full-activation.md).

## Fixed full GELU/SiLU owning region

`typed::ActivationSpec` selects only GELU or SiLU under the approved D20
ring50/Q12 input/output contract. Encoded input must lie in `[-2^37, 2^37)`.
`LocalDealer::activation_from_values` validates the complete clear input, geometry,
table/template identity and aggregate capacity before root allocation or entropy.
`LocalDealer::activation` consumes an existing mask owner with the same bound;
that inherited local developer premise is not a compiler or checkpoint proof.

The fixed schedule projects input and mask privately into ring38, takes TR6 to
ring32/Q6, and computes `D=[Y>=0]`, `P=D*Y`, and unsigned `A=2P-Y`.
Clipping uses `K=256` for GELU or `1024` for SiLU and a narrow Select operand;
the original wide input stays live for the final deferred `D*X` selection.
The source lookup retains its masked arithmetic/sign correction and correlated
mask `rho=2u-r`. Matching-role deferred owners combine as `D*X-T[J]` in ring50,
with final mask `rF-rho`, no extra rounding, and no final shift.

Eight exchanges emit sixteen directional frames, exactly `1224+144*lanes`
bytes. The final `PLLMAC01` frame namespace is distinct from native primitive
frames. Complete frame validation precedes mutation; malformed inputs, replay,
out-of-order calls and cancellation burn every remaining local stage. The caller
must retire its sibling. Final results remain opaque typed common-masked owners.

A private fixed-use ledger binds actual source, mask, role, invocation and operand
slots. No intermediate escapes and no external capability is cloned. Original
backing and leases survive until last use; the original matrix singleton guards X
until final selection consumes that backing, not the unrelated activation result.
Aggregate reservation precedes all stage issuance and splits into the existing
native owners without refund/reacquire. It prices both parties, both table copies,
all retained/received/emitted frames, transition workspace and final owners.
Cancellation and completion retain party metadata charges until party drop.

Immutable equation tables are shared with FuseFSS through method-neutral table
accessors, not shared protocols or numeric identities. Normal native, external
integer-oracle and lifetime/fault tests cover this bounded region. The integrated
garbling suite passes 554 unit checks, 35 external/oracle checks and 20 doctests
(eight opt-in tests skipped). It does not supply protected float
conversion, authenticated material/peer transport, compiler ranges, a complete
decoder, SDK selection, or independent security qualification.

## Local complete-dot / activation composition (D22)

The separate `typed::LocalLinearActivationSpec` and
`LocalDealer::local_linear_activation_from_values` compose a complete private
ring50/Q24 matrix sum, one exact GapARS12 to ring50/Q12, and the unchanged D20
GELU/SiLU owner. There is no per-term floor, bias or changed activation profile.
Complete endpoint-product intervals must admit the quarter-ring promise before
issuance; dimensions, each 256-element matrix limit, contraction work and the
existing 64 MiB native ceiling remain enforced. These intervals and source labels
are checked **local developer declarations**, not authenticated checkpoint proofs.

The fixed local schedule has eleven exchanges. A four-use ledger consumes
original capabilities; native backing, matrix singleton guards, retained frames
and zeroizing owners keep continuous reservations. Singleton/issuance admission
precedes root entropy, and cancellation retires both local parties. Exact-budget
reserved transfer preserves budget identity without refund/reacquire. The
native API supplies no authenticated delivery or isolated worker transport.

Owner-defined preconstruction quotes separately cover activation and linear-spec
metadata validation, hashing and constructor/return overlap. They are available
before constructing a declaration, use checked sums and sequential phase maxima,
and do not themselves reserve capacity. Caller-retained copies and source/compiler
overlap remain separate; logical storage is not stack, allocator or process RSS.

The independent fixture selects 256 coordinates of one pinned Qwen gate row. It
is a **checkpoint-derived submatrix**, not its original 896-wide contraction,
complete MLP or decoder. Native unit tests import the frozen independent oracle
and test full dot/floor/activation behavior, both layouts, failure retirement and
reservation continuity. Source certification remains exclusively compiler-owned.
The bounded `sigma_linear_activation::execute_sigma_linear_activation` facade
derives its native declaration and coefficients only from its checked source and
certificate. Its incremental compiler quote reserves before source hashing or
copies, includes import scratch and sequential metadata/proof/native-handoff
overlap, and retains the original guard across atomic native-capacity transfer.
An already borrowed semantic plan and immutable inventory are baseline owners,
not newly allocated backing; their initial construction is outside this boundary.
Transferred metadata and source/coefficient backing remain charged until their
last frame, mask or output owner retires. These are logical added-owner bounds,
not whole-compiler RSS or allocator guarantees. No native `LocalAssumption`
provenance is elevated by source labels or metadata hashes. Publisher pin trust,
authenticated transport and complete-decoder/SDK admission remain separate.
Approved boundary: [`../plans/sigma-compiler-linear-activation.md`](../plans/sigma-compiler-linear-activation.md).

## Separate GPU-derived DPF primitive

Decision D16's private `compact_dpf::gpu127` module implements the approved
CPU port under `pllm.compact-dpf.source-gpu127.v1`. Widths 8–63 retain 127-bit
secret seeds, randomized complementary root controls and two complete
AES-expanded terminal correction words. AES plaintext tags are 0 and 2 with
the pinned GPU word/byte mapping; existing compact-DPF defaults stay unchanged.

For `d = n-7`, each logical key costs `129*d+255` bits. The distinct PLLM codec
adds an 18-byte header and canonical zero tail padding: 66–953 encoded bytes.
Paired issuance uses `4*d` AES blocks; point and unsigned strict-less evaluation
each use `d`. Full expansion admits only widths 8–10, at most 1,024 output bytes
and 14 AES blocks per key. Fixed inline key owners and stack/AES workspace remain
additional to encoded bytes; allocation limits do not bound whole-process memory.

Issuance uses one paired OS-random draw. Encoding consumes its owner; internal
evaluation borrows it for table scans. This private algebra does not enforce
authenticated session binding or one-use transport: those remain outer-runtime
duties. Same-profile payload substitution is not detected by its canonical codec.
There is no upstream wire-byte equivalence, GPU performance or independently
approved security proof. The source-bound conditional proof candidate below does
not change that qualification. D17 separately approves complete-output-ring arithmetic masks; its source
audit remains in `paper/reproduction-contracts/sigma-arithmetic-masks.md`.

## Bounded TR / GapARS / DReLU adaptation

`masked_wire_reference::arithmetic` uses the distinct
`pllm.sigma.gpu127-full-ring-masks.v1` identity and D16 GPU127 keys. It is not
the pinned CPU implementation or the existing Q12 gated-block profile; defaults
remain unchanged. `rescale.rs` factors shared masked-floor equations, not a new
rounding rule. Native tests below qualify bounded arithmetic and ownership,
not a composed protocol or decoder.

Admission covers 1–256 lanes and input widths `3 <= n <= 64`. TR admits
`0 < s < n`, returning exact unsigned floor modulo `2^(n-s)` (also the reduced-ring
representation of signed floor). GapARS admits `0 < s < n-1`, returning signed
floor in the original ring **conditional on the unwrapped private input promise
`-2^(n-2) <= x < 2^(n-2)`**. That promise is not enforced or proved by this owner.
DReLU returns the XOR-masked bit `[signed x >= 0]`, including zero.

Comparison width is `c=max(s,8)` for TR/GapARS and `c=max(n-1,8)` for DReLU.
Comparisons below eight bits publicly zero-extend threshold and query into the
GPU127 domain; they do not truncate or change the comparison. Issuance samples
fresh uniform masks over all `n-s` TR output bits, all `n` GapARS output bits,
or one DReLU bit. Caller-supplied input-mask freshness remains a caller obligation.

For `L` lanes, `d=c-7` and output width `m`, checked costs are:

| Cost | TR | GapARS | DReLU |
| --- | --- | --- | --- |
| Logical key bits per party | `L*(129*d+255+2*m)` | `L*(129*d+255+3*m)` | `L*(129*d+256)` |
| RNG byte requests (including nonce, roots, masks and splits) | `32+64*L` | `32+72*L` | `32+48*L` |
| Frames per party | 2 | 2 | 1 |

Each operation requests `m*L` output-mask bits; RNG request bytes are not an
entropy/security claim. Paired issuance costs `4*d*L` AES blocks; evaluation costs
`d*L` per party. Each frame is exactly `76+8*L` bytes, totaling
`2*frames_per_party*(76+8*L)` across both directions. Party material physically
stores `L*sizeof(Lane)` bytes, including inline keys and three u64 constants per
lane even where unused; logical key bits are not encoded delivery bytes.

`resources()` separately prices dealer, each party and combined logical live
reservations before randomness/allocation, under the shared 64 MiB ceiling. With
`F=76+8*L+sizeof(Frame)+64`, `O=8*L+sizeof(MaskedOutput)+64`, party reservation is
`P=sizeof(Party)+L*sizeof(Lane)+24*L+3*64+frames_per_party*F+O+8192`;
dealer reservation is `D=sizeof(OutputMasks)+16*L+64`, combined `2*P+D`.
Borrowed input masks (`8*L`) and both masked-input views (`16*L`) are included
allowances, not transferred ownership; excess backing capacities and unrelated
owners remain additional. These are logical allowances, not RSS/allocator limits.

A local trusted research dealer sees input masks and returns a separate output-mask
owner plus two separate party-local owners, each holding only its own key and
constants. This differs from the coupled matrix owner. Frames bind public
plan/session/operation/wires/index, nonce, input digest, sender and phase, but are
not authenticated. Successful completion or a failed call consumes local execution
state; cancellation burns it. Orchestration must also retire the other party on
failure. Frames and outputs retain split leases independently; cancelled/consumed
party owners retain their remaining reservation until drop. Zeroizing owners
erase/deallocate before lease refund. No material codec, authenticated material
delivery or peer transport, persistent replay protection, malicious-output
verification, SDK integration, runnable SIGMA decoder or composed security proof
is supplied by this source slice.

## Paper/source security review amendment

This amendment records the reviewed Astra53 paper/source plan, the AstraGPU14
argument with independent Astra59 **PASS conditional-proof-candidate** review,
and AstraSim14's independent primary-image **PASS** check. These are AI review
preparation results, not external or production security approval. They do not
upgrade functional source review into cryptographic qualification. No native
execution accompanies this documentation amendment.

### Primary-paper map and D17 scope

The primary reference is the [SIGMA paper PDF](https://petsymposium.org/popets/2024/popets-2024-0107.pdf).
Page numbers here are **printed paper pages**, not PDF viewer indices or ignored
local artifact paths. The arithmetic map is §4.2–§4.3, printed pp. 66–68:

- **GapARS:** Theorem 3, Figure 2 and Lemma 2 use the unwrapped quarter-ring
  premise `-2^(n-2) <= x < 2^(n-2)`. Add `2^(n-2)`, apply GapLRS, then subtract
  `2^(n-f-2)` in the original `Z_(2^n)` output ring. The premise is necessary for
  this reduction; it is not supplied by a masked residue or functional tests.
- **TR:** Theorem 4 and Equation (3) reduce modulo `2^(n-f)`. The wrap term is
  a multiple of that modulus and vanishes, so this reduced-ring operation needs
  no GapARS premise. The paper omits the detailed TR protocol; the local code's
  shared masked-floor equations must be reviewed as implementation algebra,
  not attributed to printed protocol pseudocode.
- **No-gap ARS:** Theorem 5 is TR followed by sign extension. None of the three
  current gate variants implements this no-gap, original-ring protocol. Having
  TR and DReLU separately does not implement it; no TR+DReLU shortcut is claimed.
- **DReLU:** Figure 3 and **Theorem 6**, not Theorem 5, specify the predicate.
- **Appendix discrepancy:** §4.2.1 points to Appendix E.1 for GapLRS. The actual
  printed E.1 concerns DReLU, while E.2 contains LRS with a gap (including its
  simulator on printed p. 76). This citation discrepancy must remain explicit.

For masks, §2.4.1's appropriate-wire-group convention and §2.4.3/§4's masked
output reconstruction support full output-group masks: GapARS uses `Z_(2^n)`,
TR uses `Z_(2^(n-f))`, and DReLU uses one bit. D17 corrects/aligns the inspected
source locally to that contract; it is **not a new masking theorem**. The carry
mask is a different object: its secret is a bit, but its arithmetic sharing lives
in the output ring. Each individual arithmetic share is full-ring uniform,
not restricted to `{0,1}`. Boolean carry messages use its low bit only.

Read source anchors: `crates/pllm-garble/src/masked_wire_reference/arithmetic.rs`
`split` (lines 198–206), output-mask and carry issuance (377–409), and `begin`
carry messages (557–564); `rescale.rs` lines 3–5 and 16–27 implement the shift,
carry and wrap corrections. D17's output-view repair and the equations' functional
correctness are distinct from a joint-view security proof.

### GPU127: source-bound conditional proof candidate

Scope is exactly `crates/pllm-garble/src/compact_dpf/gpu127.rs`: public
`8 <= n <= 63`, fixed role `b`, Boolean payload `beta=1`, and ideal independent,
fresh issuance randomness. Anchors are `Profile::new/depth/ordinary` (81–98),
`cipher/child/children` (248–279), `issue_with` (329–379),
`correction/descend/terminal_block` (382–405), `point/less` (415–457), and
`write/decode` (595–673). Other compact-DPF profiles and arbitrary payloads or
groups are outside this argument; distribution equivalence with generic
`compact_dpf` is not asserted.

For a 127-bit seed `s`, the implemented joint expansion is

```text
G(s) = AES_LE(s << 1)(0^128) || AES_LE(s << 1)(02,00,...,00).
```

The seed is the **secret AES key**, with low key bit zero; this is neither a
public fixed-key PRP nor feed-forward construction. The concrete assumption is
that this restricted-key, **joint 256-bit** output for uniform `s` is
computationally indistinguishable from uniform 256 bits, with AES-128 fixed.
Two separate marginal assumptions do not suffice. An asymptotic PPT theorem
would additionally require a parameterized family; fixed AES-128 alone does not
define one.

**Root and ordinary-step invariant.** Issuance has independent 127-bit root
seeds and randomized complementary controls. Either party's packed root is
uniform 128 bits. For target branch `a`, let `A_j` be the exposed party's seed
child and `U_j` the ideal hidden party's seed child. The seed correction and
next hidden seed obey

```text
Cs = A_(1-a) XOR U_(1-a)
h' = U_a XOR ((1-t) * Cs)
U_(1-a) = Cs XOR A_(1-a)
U_a = h' XOR ((1-t) * Cs),
```

where `t` is the exposed party's current control and multiplication denotes
conditional XOR. For fixed exposed prefix this inverse is bijective. The two
independent hidden child-control bits similarly map bijectively to the two
control corrections. Hence the complete ordinary correction word is uniform
129 bits, and `h'` is uniform and independent of the exposed prefix **including
that correction word**. The next hidden control is determined as the complement
of the exposed next control; it is not an additional secret uniform bit.

The root-to-terminal hybrid marginalizes earlier hidden random children and
represents the next hidden seed implicitly through this invariant. Each next
PRG challenge output suffices to form the future correction words: no hidden-state
oracle and no challenge seed are required. There are `d=n-7` challenges,
**including the terminal expansion**, rather than only `n-8` ordinary steps.

**Terminal simulator.** With exposed terminal seed `s_b`, target branch
`a=(alpha>>7)&1`, offset `r=alpha%128`, and jointly uniform 256-bit `(R_0,R_1)`,

```text
T_j = G_j(s_b) XOR R_j XOR [j=a] * e_r.
```

Both entire terminal words are jointly uniform, including low bits: terminal
low bits are data, not discarded control bits. The alpha-independent single-key
simulator samples one uniform 128-bit root, `n-8` independent uniform 129-bit
ordinary correction words and uniform 256 terminal bits. It uses source-canonical
zero unused slots and the same public framing/role/width and zero tail padding.
For each fixed alpha,

```text
Adv(real_alpha, simulator) <= sum_(i=1)^d Adv_G(B_(alpha,i)).
Adv(real_alpha, real_alpha') <= sum_i Adv_G(B_(alpha,i))
                              + sum_i Adv_G(B_(alpha',i)).
```

The latter is at most `2*d*epsilon` only if **all** those reductions have
advantage at most epsilon at their actual runtimes, including reduction overhead.
Public G evaluation, seed guessing and adaptive evaluations under adversary-chosen
keys remain available to distinguishers/reductions. Do not condition on seed
uniqueness; this embedding adds no separate birthday term. These are conditional
single-key privacy bounds, not an unconditional AES security estimate.

**Perfect paired correctness.** Along the target path, party controls stay
complementary. On the first off-target branch both seeds and controls become
equal and remain equal. The full terminal correction therefore reconstructs
one one-hot bit exactly on the target branch and zero elsewhere, giving
`point_0(x) XOR point_1(x) = [x=alpha]` without a PRG assumption.

For `less`, index the `d` traversed branch bits by `x_1,...,x_d`, set `x_0=0`,
and let `E_(i-1)` indicate equality of the preceding query/target prefixes.
The paired source accumulator before terminal parity is

```text
P = 1 XOR XOR_(i=1)^d E_(i-1) * (x_(i-1) XOR x_i).
```

At the first mismatching branch `j`, this telescopes to `1 XOR x_j = [x<alpha]`;
terminal contributions cancel. If no branch mismatches, `P=1 XOR c` for
`c=x_d`. Terminal parity contributes `[r_alpha <= r_x]` when `c=0`, or
`[r_alpha > r_x]` when `c=1`. Both cases yield `[r_x < r_alpha]`. Equality yields
zero; offset 127's empty suffix is explicitly zero, avoiding a shift by 128.
Thus paired `less` evaluates unsigned strict less, not less-or-equal. Each party
uses `d` half-expansions. Public zero extension of `k<8` query and threshold to
eight bits preserves these predicates but costs **one** half-expansion, not the
paper formula's zero-expansion special case.

Printed p. 64, Theorem 2 (visually recovered from the image-only local PDF's
fourth page in the reported review), states existence/cost of strict-less DPF
with beta=1; it does not print this `Eval` pseudocode. The algebra above is the
implementation-specific candidate supplied for review, not a quotation of that
theorem. Local deterministic evaluation/postprocessing transfers conditional
key privacy to evaluation views within that scope. Consuming an owner enforces
lifecycle use, **not cryptographic freshness**.

### Printed E.2 simulator versus real arithmetic-share view

The independently visually checked primary image is Appendix E.2, printed
p. 76. Its step 1 samples the arithmetic carry-mask share `A=r_b^(w)` and peer
`wHat` from `{0,1}`; step 5 uses A arithmetically, step 6 samples `m_b` uniformly
from `U_N`, and step 9 outputs A itself. Figure 2 on printed p. 66 and §2.1.3
instead specify arithmetic sharing of a bit secret in `Z_Q`. This requires each
individual share A to be uniform over the full ring; the secret's bit domain
does not restrict either arithmetic share's domain.

This discrepancy survives output conditioning. Fix a valid masked input and
underlying masks. Let A be the arithmetic carry-mask share and R the independently
randomized additive correction share (Figure 2's `r_b`; source `correction[b]`).
For either role, R is uniform over the output ring and independent of A, the DPF
key and carry messages. The local output share satisfies

```text
Y_b = C + R + T_b - U_b mod Q.
```

For each choice of A and the other sampled material, each fixed Y_b determines
exactly one R. Thus conditioning on the masked input and Y_b preserves A's
full-ring uniform distribution. Adding the compatible
other output share adds no constraint when their sum is already fixed.
Conditioning on R itself would be a different experiment, not the printed
simulator's input. For canonical n-bit residues, `D(A)=[A>=2]` therefore has
real-versus-printed-simulator probability gap `1-2/2^n` for `n>1`.

This is a **printed-simulator/real-view mismatch**, not a distinguisher between
secret inputs, a protocol attack or an impossibility theorem. The local code's
full-ring split is visible at `arithmetic.rs:198–206,384–385`; using only its low
bit in the Boolean message at 563–564 does not turn that stored arithmetic share
into a Boolean share.

Proposed paper-simulator correction only: sample A over the full ring, use
`A mod 2` in Boolean step 3, retain full A in arithmetic step 5, and program gamma
as in step 8. A corrected **joint-view proof remains unresolved**. This proposal
is neither a proved repair nor a requested code change.

### Single-gate common-output view: conditional candidate

SingleGate proposal 25 received independent Astra17 **PASS** source/math review.
This candidate uses a **different interface** from Appendix E.2's prescribed
local output share ([primary paper](https://petsymposium.org/popets/2024/popets-2024-0107.pdf),
printed p. 76): preprocessing delivers complete own material before masked
input X; the legitimate **common masked output Y** is available before the final
peer output frame. This is not paper ideal-functionality equivalence, closure
of the corrected E.2 joint simulator, or cryptographic security signoff.

Scope is one valid TR, GapARS or DReLU, a trusted dealer, fresh independent
issuance, one static semi-honest role, applicable numerical premises, and the
fixed-beta=1 GPU127 **full-key** simulation assumption above. No additional
hidden-mask-related auxiliary information is supplied to the corrupted role or
used in input selection; future correlated preprocessing is excluded.
Here r denotes the input mask, c the
carry-mask **bit**, and R the **full output mask**. This R is different from the
preceding subsection's additive correction share (Figure 2's `r_b`).

Own arithmetic constants `(A,B,C)` mean carry share, wrap share and additive
correction share respectively; C, not this subsection's R, corresponds to that
preceding correction-share notation. For TR, A and C are independent uniform
elements of `Z_Q` and B is zero. For GapARS, A, B and C are independent uniform
elements of `Z_Q`. For DReLU, the own constant D is a uniform bit. For either
role, fresh independent additive/XOR split coins and the target-minus-left
bijection make the **joint own constants** independent of `(r,c,R,K)`, where
K is the full own DPF key.

Let Z be the actual gate-adjusted masked input and let the opened carry be
`q = c XOR [Z_low < r_low]`. Conditional on own material, fixing compatible
`r,x,X` gives

```text
Pr[q=u, Y=v] = 1/(2Q),    Y = F(x) + R mod Q.
```

Fresh independent c and full-ring R give this factorization. DReLU instead
has a uniform common masked output bit and no carry round. The simulator samples
the full key and own constants **once** in preprocessing, finalizing their bytes
and hashes before X and Y. At X it samples q, computes the actual own carry
message `h_b`, and supplies peer message `q XOR h_b`. It computes the actual own
output share `p_b` with unchanged material. At legitimate Y it supplies
`Y-p_b mod Q`, or `Y XOR p_b` for DReLU. Key indistinguishability extends through
this postprocessing, including adaptive own-key evaluation, all own-material
serialization and frames.

Source anchors are
`crates/pllm-garble/src/masked_wire_reference/arithmetic.rs:198–206,375–434,543–605`
and `rescale.rs:16–27`. The material boundary is
`crates/pllm-garble/src/masked_wire_reference/arithmetic/material.rs:170–186,225–250`,
with shared envelope handling in `crates/pllm-garble/src/material_wire.rs:155–185`.
The actual material descriptor hashes public spec/profile/role and arithmetic
binding; that binding hashes public context and issuance nonce. Export maps the
descriptor to envelope binding and arithmetic binding to envelope nonce. No
pair-material commitment or secret MAC exists in this inspected boundary. These
hashes are deterministic functions of own/public bytes: this simulation extension
needs no hash inversion or collision assumption. That observation does **not**
establish forgery resistance. The actual Party owns no input-mask shares,
honest-party material or dealer masks.

The different interface cannot generally accept an independently prescribed
internal `p_b` after preprocessing and X are fixed: arithmetic has at most two
possible local output values, so an independent uniform prescription matches
with probability at most `2/Q` (vacuous at `Q=2`); DReLU has one possible local
value. This is a timing/interface obstruction, not a secret-input attack.
Full-ring R is essential to the uniform-output factorization above. Bounded R
is not inherently incompatible with every conditional-view simulation supplied
with legitimate Y.

This candidate is **single-gate only**. The actual typed region
(`crates/pllm-garble/src/masked_wire_reference/typed/region.rs:389–407` and
`crates/pllm-garble/src/masked_wire_reference/typed/activation.rs:1226–1249`)
directly uses an output mask R as the next input mask r, without a client remask.
Generic joint conditioning on future preissued correlated material remains
**open**; the following finite-prefix candidate covers one specified dependency
pattern. The single-gate result alone implies no dependent-wire composition,
security approval or activation.

### Finite dependent-mask prefix: conditional joint-view candidate

This source/math-reviewed conditional candidate covers only a fixed, successful,
valid workload of the existing **TR6 → DReLU → reconstructed Select** prefix.
Input widths are `12 <= n <= 64`, reduced widths `m=n-6` are 6–58, and lanes
are `1 <= L <= 256`; signed descriptors move from Q12 to Q6, with output ring
`Z_Q`, `Q=2^m`. It retains the preceding GPU127 full-own-key premise, one static
semi-honest role, a trusted dealer and fresh independent dealer coins. It adds
no AES assumption upgrade or cryptographic signoff.

The role owns its three native parties, common X/Y/D and ledger, not input-mask
shares, dealer masks or the honest party's key. Permitted public metadata is
explicitly the full spec/context, descriptors, roles, session, namespace,
issuance counter/nonces, mask IDs, consumer and **declared intervals**, ledger
digests and derived framing. Intervals are public assumptions, not private facts.
Context/ledger hashes cover public typed uses; mask IDs hash namespace, counter
and descriptor, **not mask values**. See
`crates/pllm-garble/src/masked_wire_reference/typed.rs:465–487`,
`typed/internal_use.rs:12–35` and `typed/region.rs:344–385` under that same source
directory. Additional hidden-mask auxiliary information outside this declared
interface is excluded.

For fixed valid clear workload x, the common trace is

```text
y = TR6(x),                 d = [signed_m(y) >= 0]
X = x + r_x mod 2^n,        Y = y + rho mod Q
D = d XOR beta,             Z = d*y + sigma mod Q.
```

Here rho and sigma are fresh full-ring masks and beta is a fresh bit. The **same
rho** passes from TR output to DReLU input and Select's arithmetic operand; the
**same beta** passes from DReLU output to Select's predicate. The same stored Y
fans out, without resampling or a client remask
(`typed/region.rs:389–407,505–540`).

Let `G=(r_x,rho,beta,sigma)`, let c be TR's carry-mask bit, and let `K_T,K_D` be
the real own keys. Write own TR constants as `(A,0,C)`, DReLU constants as
`(E,0,0)`, and Select constants as `W_0,...,W_4`. The seven ring coordinates
`A,C,W_0,...,W_4` are jointly independent uniform elements of `Z_Q`; E is an
independent uniform bit. Thus the joint own-constant vector S satisfies

```text
Pr[S=s | G,c,K_T,K_D] = 1/(2*Q^7).
```

For role Zero this follows from fresh split coins. For role One, the
coordinatewise target-minus-left bijection establishes the same **joint** law
across all gates. Select's totals are
`(beta,rho,beta*rho+sigma,2*beta,-2*beta*rho)`, each independently freshly split;
in particular `W_3` generally does **not** equal `2*W_0`. Its actual own share is

```text
v_b = W_2 - W_0*Y - W_1*D
      + [D=0]*(W_3*Y+W_4) + b*D*Y mod Q.
v_0 + v_1 = (D XOR beta)*(Y-rho) + sigma = Z mod Q.
```

This algebra needs no additional Select cryptographic assumption. Source:
`crates/pllm-garble/src/masked_wire_reference/selection.rs:413–431,479–491,528–537`;
arithmetic sampling and evaluation:
`crates/pllm-garble/src/masked_wire_reference/arithmetic.rs:193–206,374–418,543–605`.

**Causal simulator.** All simulated own keys, constants, bytes and metadata are
fixed **before X**. Key widths are 8 for TR6 and `max(8,m-1)` for DReLU. At X,
compute the actual own carry message `h_b`, sample uniform q, and compute actual
own TR share `p_b` with unchanged material. On legitimate Y, compute actual own
DReLU share `e_b`; on legitimate D, compute actual own Select share `v_b`.
The peer payloads, in chronological order, are

```text
q XOR h_b,    Y-p_b mod Q,    D XOR e_b,    Z-v_b mod Q.
```

Each legitimate common value is available before its corresponding peer frame.
These are not independently prescribed local output shares, and preprocessing
is not programmed late. Source framing/material are deterministic functions of
own/public bytes; the region itself introduces no new export codec.

With real keys, q is uniform even given **all preissued own constants**, G,
keys and the legitimate trace: future materials reveal no c. Every remaining
peer payload is then forced. This gives an **exact joint real-key coupling**,
not merely matching marginal distributions. Replace the `2L` own keys by a
sequential hybrid. Each reduction knows G and the fixed workload, simulates
other keys using fresh seeds, and supplies threshold-dependent auxiliary
material independent of the challenge issuance coins. It never needs the
honest challenge key. The distinguishing bound is

```text
Adv <= L * (epsilon_GPU(8) + epsilon_GPU(max(8,m-1))),
```

with each epsilon evaluated at its reduction's actual resources. Full own-key
inspection, copying and adaptive evaluation remain within the existing premise.

For fixed valid workload, Y/D/Z are jointly uniform under fresh full rho, beta
and sigma **before conditioning on real keys**, not necessarily uniform given
real keys. In the final simulated-key hybrid, the trace is independent of own
material. Privacy of X additionally requires fresh uniform `r_x`. Full-ring
masks are necessary for **this uniform-trace factorization**, not for every
possible conditional-view simulator. State consumption is not cryptographic
hiding.

This is a narrow **common-trace interface** candidate, not equivalence to the
[paper's](https://petsymposium.org/popets/2024/popets-2024-0107.pdf) prescribed-share
ideal interface (Appendix E.2, printed p. 76). It establishes no whole-activation,
GapARS-chain, deferred-Select or general-graph claim, source authority, physical
isolation, discharge of numerical premises, gate 3886 closure, authentication,
activation or independent cryptographic security signoff.

### Complete fresh SiLU owner: conditional joint-view candidate

This source-bound conditional candidate extends the historical finite-prefix result through the **returned final common masked output of one fresh, preissued SiLU owner**. Reported independent AI planning/proof reviews passed the clipping cut and its Lookup continuation. Those reviews are not external cryptographic signoff. Scope is a fixed valid workload, identical public declarations when comparing workloads, `1 <= L <= 256`, one static semi-honest evaluator, trusted dealer, fresh independent issuance coins, and the existing [GPU127 full-own-key premise](#gpu127-source-bound-conditional-proof-candidate). It is not a generic graph-composition result.

**Actual schedule and view.** Set `M=2^32`, `N=1024`, `W=2^50`; arithmetic expressions use their explicitly indicated ring, bits use XOR. Input is a ring50/Q12 encoding of fixed signed `x ∈ [-2^37,2^37)`. Project its value and mask into ring38, then TR6 produces ring32/Q6. Do not substitute ring50→ring44 TR6. All seven top-level native owners are issued before X: TR6, first DReLU32, reconstructed Select32, clipping DReLU32, reconstructed Select10, deferred Lookup10→50 (including its nested deferred Select50), and final deferred Select50.

The view contains all four own GPU keys, every own scalar coordinate, known Lookup mask shares `(r_b,q_b)`, all own preissued Select words, remembered pre-erasure contents, arbitrary efficient own-key inspection/evaluation, local scan/count/working values, frames and common masked intermediates. Public metadata includes specs, table, descriptors and declared intervals, roles, context/session/namespace, counters/nonces, mask IDs, use ledger and derived framing. Dealer masks, honest-party material, extra upstream-mask shares and future-consumer material are excluded. Source ownership/consumption is not a cryptographic freshness argument.

**Wires and reused masks.** Hats denote common masked values. Let `r_x ∈ Z_W`, `rho,sigma ∈ Z_M`, `eta ∈ Z_N` and bits beta,gamma be fresh independent masks. Then

```text
Xhat=x+r_x mod W; XNhat=Xhat mod 2^38
y=floor(x/64); Yhat=y+rho mod M
d=[y>=0]; Dhat=d XOR beta
p=d*y; Phat=p+sigma mod M
a=2p-y=abs(y); Ahat=2Phat-Yhat mod M
lambda=2sigma-rho mod M
c=a-N; Chat=Ahat-N mod M
h=[signed32(c)>=0]; Hhat=h XOR gamma
e=1-h; Ehat=Hhat XOR 1
q_clip=(c+1) mod N; Qhat=(Chat+1) mod N
v=e*q_clip mod N; Vhat=v+eta mod N
j=v+N-1 mod N; Jhat=Vhat+N-1 mod N=j+eta mod N.
```

XN uses `r_x mod 2^38`; A/C share lambda; E retains gamma; Q uses `lambda mod N`; J retains eta. No fresh entropy is assigned to XN/A/C/E/Q/J. Same Y feeds first DReLU, prefix Select and A; same C feeds clipping DReLU and Q; same original Dhat/Xhat survives for final wide Select. In particular `lambda+rho=2sigma mod M` and `Chat=2Phat-Yhat-N mod M` hold exactly, including their parity constraints.

Numerically, `y ∈ [-2^31,2^31-1]`, unsigned `a ∈ [0,2^31]`, and `c ∈ [-1024,2^31-1024]`. Hence `h=[a>=1024]` and `j=min(abs(y),1023)`, including minimum signed y. These are source/range correctness statements, not a privacy premise discharged by tests, a compiler range proof, an approximation guarantee, or permission to adopt another profile.

**Complete own scalar law.** Lookup independently samples `r,u ∈ Z_W` and bit q; final Select independently samples `f ∈ Z_W`. Dealer masks are `rho_L=2u-r` and final `f-rho_L`. Lookup owns additive r_b and XOR q_b shares. Prefix, clipping, nested Lookup and final Select each independently split their five totals:

```text
(beta,rho,beta*rho+sigma,2beta,-2beta*rho)                  in Z_M
(gamma,lambda mod N,gamma*(lambda mod N)+eta,2gamma,
 -2gamma*(lambda mod N))                                  in Z_N
(q,r,q*r+u,2q,-2q*r)                                      in Z_W
(beta,r_x,beta*r_x+f,2beta,-2beta*r_x)                      in Z_W.
```

Let G contain all total masks `r_x,rho,beta,sigma,gamma,eta,r,q,u,f`, with lambda/rho_L derived; let chi be TR's carry-mask bit. The complete nonzero-capable own scalar vector S_own has seven M coordinates (two TR and five prefix Select), three bits (two DReLU constants and q_b), five N coordinates (clipping Select), and eleven W coordinates (r_b and two five-word Select vectors). Other arithmetic constants are canonical zero. For either role, even conditioned on all **paired** real GPU keys,

```text
Pr[S_own=s0 | G,chi,paired keys] = M^-7 * 2^-3 * N^-5 * W^-11.
```

Independent split coins prove this directly for role Zero; coordinatewise total-minus-left/XOR-left bijections prove it jointly for role One. This includes already known r_b,q_b and future preissued material. A local Select word is not computed from related local words merely because their totals are related; in particular local word 3 need not be twice word 0.

**Exact real-key prefix coupling.** Fix all own material before Xhat. With real keys, immediately before the TR carry opening, chi is uniform conditioned on G, all own material, own keys and only the prior legitimate common trace. The opened TR carry omega is therefore uniform at that boundary; its peer payload is `omega XOR own_carry`. No claim that chi remains hidden after observing omega under conditioning on all proof-only total masks is made. Subsequent arithmetic peer payloads are legitimate common result minus actual own share, or XOR for DReLU. Chronologically these reconstruct Yhat, Dhat, Phat, Hhat and Vhat using unchanged material. A/C/E/Q/J are derived rather than resampled. The owner's transition after Vhat immediately performs its local Lookup scan and emits its first Lookup frame. All local evaluation/framing uses the actual fixed own keys and bytes.

**Lookup and final correctness.** Write `t=T[j] ∈ Z_W`, `g=d*x-t mod W`. Lookup key target is alpha=`-eta mod N`. For paired Boolean expansions a_b(i), define

```text
s=a_0(alpha)-a_1(alpha) in {+1,-1}; B=(s+1)/2
A_b=(-1)^b * sum_i a_b(i)*T[(i+Jhat) mod N] mod W
n_b=(-1)^b * sum_i a_b(i).
```

The actual ascending scan gives `A_0+A_1=s*t` and integer `n_0+n_1=s`. Individual signed populations are not assumed ±1. The source's modulo-four correction functions have residues `[0,1,1,0]`, whose XOR under these sums is B. With `kappa_b=correction(b,n_b)`, own outgoing pair is `(a_out,c_out)=(A_b+r_b,kappa_b XOR q_b)`. First peer receipt reconstructs

```text
R=s*t+r mod W; C_L=B XOR q.
```

For W-ring Select define

```text
Sel_b(K,c,z)=K2-K0*z-K1*c+[c=0]*(K3*z+K4)+b*c*z mod W.
```

Let U be own nested Lookup words and F be own final Select words (distinct from common Vhat). Local continuation is exactly

```text
z_b=Sel_b(U,C_L,R)
ell_b=2*z_b-[b=0]*R mod W
k_b=Sel_b(F,Dhat,Xhat)
w_b=k_b-ell_b mod W.
```

Only role Zero subtracts R. There is no nested output exchange in this deferred path. Paired sums give

```text
z_0+z_1=B*(s*t)+u
ell_0+ell_1=t+2u-r=t+rho_L
k_0+k_1=d*x+f
w_0+w_1=g+f-2u+r=O.
```

O is the returned final common masked output, not a prescribed local share or an internally reconstructed Lookup output.

**Exact downstream entropy and joint law, before any key hybrid.** Let proof-only Z contain fixed x, upstream total masks, chi, all paired real key generations, metadata and independent nonces, excluding downstream r,q,u,f and split coins. The scalar law implies, at every compatible positive-probability conditioning,

```text
Pr[r=a,q=e,u=c,f=d | Z,S_own] = 1/(2*W^3).
```

The view V_cut through first outgoing Lookup is a deterministic function of Z,S_own and independent inspection coins. Its dependence on downstream material is through the listed own split coordinates, not hidden totals. Thus this downstream product law survives conditioning on V_cut, including its known r_b,q_b and complete preissued own keys/words. In particular

```text
Pr[R=a,C_L=c | V_cut] = 1/(2W).
```

For fixed proof-only Z,S_own,u, the joint map and inverse are

```text
(r,q,f) -> (R,C_L,O)=(s*t+r, B XOR q, g+f-2u+r)
r=R-s*t mod W; q=C_L XOR B; f=O-g+2u-r mod W.
```

This is a bijection on `Z_W × {0,1} × Z_W`, so

```text
Pr[R=a,C_L=c,O=o | Z,S_own,u] = 1/(2W^2)
Pr[R=a,C_L=c,O=o | V_cut] = 1/(2W^2).
```

For L lanes the joint probability is `1/(2^L*W^(2L))`. These are exact **real-key** identities, not uniformity asserted only after replacing keys. Z/s/t/u and the inverse are proof tools, never simulator inputs. The inverse counts compatible hidden totals; it neither reissues f nor programs honest material online.

**Causal continuation.** After the already fixed own outgoing pair, sample uniform R,C_L and supply peer pair `(R-a_out,C_L XOR c_out)`. Compute z_b,ell_b,k_b,w_b from existing U/F and retained Dhat/Xhat; emit w_b. Only then, at its legitimate final-output boundary, receive O and supply peer `O-w_b mod W`. Frame namespaces, roles, counts, digests and local state transitions follow source. The joint law makes R,C_L uniform even conditioned on legitimate O, so no earlier knowledge of O is needed. An unconditional sampler instead samples uniform O under that same law. No hidden sign, dealer mask, honest key/share, reconstructed Lookup output or independently prescribed local deferred share is supplied to the simulator.

This exactly preserves hidden `rho_L mod2=r mod2`, actual `ell_1 mod2=0`, and `ell_0 mod2=R mod2`. Hypothetically reconstructing `O_L=t+rho_L` would force `O_L+R` even; the actual path does not reconstruct O_L. These observations refute an independent-mask/independent-local-share shortcut, not uniform R,C_L alone and not an established secret-input attack. Fresh f is the existing final Select mask, not an added remask.

**Four-key bound and fixed-workload scope.** Replace the same four preissued own keys once, sequentially:

| Family | Width | Target |
| --- | --- | --- |
| TR6 | 8 | `r_x mod64` |
| First DReLU | 31 | `(-rho mod M) mod2^31` |
| Clipping DReLU | 31 | `(-lambda mod M) mod2^31` |
| Lookup | 10 | `-eta mod N` |

Targets may correlate through masks; independent issuance coins and known fixed-workload masks suffice for the reductions. They never need the honest challenge key. In particular the Lookup continuation has already marginalized its hidden sign. The baseline full-own-key premise covers inspection, copying and adaptive own-key evaluation. No late key, related-key assumption or stronger AES premise is added. For a fixed workload,

```text
delta_x <= L*(epsilon_GPU(8)+2*epsilon_GPU(31)+epsilon_GPU(10)).
```

Each epsilon uses its reduction's **updated actual runtime**, including local full expansion, continuation, framing and adversarial inspections. Depth sum `1+24+24+3=52` gives `52L*epsilon_G` only if every baseline joint-expansion reduction has that bound at those resources; it is not an unconditional AES estimate or new native count.

With fresh uniform r_x, the final simulated-key hybrid has independent uniform base common Xhat/Yhat/Dhat/Phat/Hhat/Vhat; all affine descendants remain correlated. The exact suffix adds uniform R,C_L,O. Thus the complete finite-owner simulated transcript is independent of fixed x. Comparing two fixed valid workloads x,x' with **identical public declarations/table/declared intervals** yields bound `delta_x+delta_x'`. Earlier common-trace coupling plus this exact suffix is composition at this one preissued owner, not a theorem for arbitrary DAGs.

**Source anchors and retained holds.** Under `crates/pllm-garble/src/masked_wire_reference/`: `typed/activation.rs:1221–1319,1332–1368` binds issuance/masks; `:1456–1589` binds the prefix and immediate scan; `:1591–1689` binds deferred difference/final return. `arithmetic.rs:375–434,543–605` and `selection.rs:413–431,479–491,549–570` bind splits and local evaluations. `source_lookup.rs:575–625,715–722,885–931,974–1009` and `source_lookup/correction.rs:311–334,357–431,480–527` bind lookup issuance, scan, known shares and role-Zero subtraction.

The endpoint excludes downstream consumers/preprocessing, dealer disclosure, decoding, inherited correlated history, adaptive workloads, authenticated transport and whole-method security. Inherited inputs retain only the conditional interface when the required fresh downstream law is independently justified. Prescribed-local-output-share APIs are not automatically equivalent to this common-output interface. The printed Appendix E.2 simulator repair/joint proof remains unproved. No numerical-premise discharge, profile adoption, native evidence/count increase, Source-authority promotion, compiler/SDK activation, independent cryptographic approval or TEN-PAPER qualification change follows. Gate 3886 and existing external-review/security holds remain open.

### Remaining security and activation gates

Gate **3886 remains open**: the exact GPU127 conditional argument above is an
AI-reviewed proof candidate, not explicit independent cryptographic security
signoff. Precise accepted assumptions/scope and review authority remain pending;
explicit independent cryptographic security review is still required. This does
not introduce a new human-only gate.

The corrected joint simulator remains outstanding. Extending output-share
privacy to legitimate common masked outputs must use the actual shared output:
the peer share is `yhat-y_b` in the ring, or `yhat XOR y_b` for XOR shares.
One cannot invent an independent yhat and call that an extension of the real
joint distribution. Beyond the scoped fixed-prefix candidate above, dependent-wire
composition remains unresolved and must track output-mask/input-mask correlations,
freshness, ranges, permitted disclosures and authentication boundaries. The
paper's §3.2.3 sequential-composition claim does not automatically cover current
state machines and frames.

The complete-fresh-SiLU-owner candidate above narrows that composition hold only for its explicitly conditioned, preissued seven-owner execution through its final common masked output. It does not repair the printed prescribed-share simulator, establish generic dependent-wire composition, or cover downstream material, inherited history or output decoding. Earlier prefix-only limitations describe those earlier candidates; this later finite extension leaves the remaining security and activation gates intact.

The reported **66** native functional/lifecycle checks below cannot establish
these proof claims and are neither rerun nor increased here. The independent
whole `NumericGolden` remains **unlocated**; frozen GPU codec goldens are not
that golden. Compiler/SDK activation remains **inactive**. No new single-party
privacy attack, production-security approval, whole-golden qualification or
publication promotion follows from this amendment. Existing decoder, transport,
composition and admission gates remain unchanged.

## Later bounded increment: consuming arithmetic-material handoff

This increment supersedes only the preceding arithmetic source slice's **“No
material codec”** statement. It does not change the TR/GapARS/DReLU equations,
GPU127 controls, original resource contracts or GapARS's conditional unwrapped
quarter-ring promise. `arithmetic::material` adds fresh-owner export for the
borrowed-budget `Party` API: `export_material` consumes the owner, and
`Handoff::into_parts` returns a secret `Packet` and a non-cloneable
`ImportPermit`. Import consumes both, including on failure; begun or consumed
owners cannot export. This is local ownership continuity, not remote delivery.

The `PLLMSA02` envelope reuses the shared canonical `material_wire` packet and
GPU127 key codec. Its descriptor binds gate/shift, widths, lanes, role, arithmetic
and key-profile identities and the original context binding. The permit retains
the complete packet digest, so rechecksummed replacement bytes do not authorize
new material. For key codec size `K` and `L` lanes, body size is `L*(K+24)` and
packet size is `120+L*(K+24)`: one canonical key and three u64LE constants per lane.
These delivery sizes differ from resident `Lane` storage and logical key bits.
Output masks, including dependent composition masks, stay with their original
dealer/composition owners; no new mask authority is minted by import.

The original party reservation remains continuously live and moves unchanged
through permit to restored party on the same budget. Additional admission covers
packet backing, one temporary encoded key, concrete owner/return/transition
metadata, allocator allowances and 8 KiB codec workspace. Packet metadata has an
independent lease: dropping the permit first cannot refund the retained packet's
backing or wrapper charge. Native execution backing, original output-mask owners
and caller copies remain separate live owners. These are logical owner charges,
not physical stack, allocator-overgrant, RSS or CPU guarantees.

Export borrows keys for encoding, then clears every original lane's key secret
fields and constants **in place before freeing the original Vec**. Success and
later-key allocation-failure observers check the same still-live pointer and
length, initially nonzero fields, exactly one cleanup observation and all-zero
original fields before deallocation. Assertions occur after export returns; no
freed-memory reads or substitute scratch wipe stand in for the original slots.
This establishes bounded logical zeroization, not physical CPU/register/stack
erasure.

### Reported serial verification packet

All source reviews passed before the separate native execution packet. Source
review is not native execution evidence. The fresh reported Rust 1.85.1 packet
passed **66 checks**, partitioned below; this documentation edit reran none.
All ten ordinary commands matched their expected counts, with zero failures,
retries or unrun commands. Inputs were frozen for that packet; no fixes or
control changes occurred during the run.

| Partition | Passing checks | Scope |
| --- | ---: | --- |
| SIGMA arithmetic-material units | 8 | All gates/widths, canonical rejection, fresh-only export, admission/drop continuity and both original-slot retirement observers |
| Original GPU127 controls | 3 | Frozen GPU codec golden/roundtrip, malformed codec and entropy/allocation/consumed-owner controls |
| Floor-chain integration | 2 | Existing direct chain plus consuming handoff chain |
| Existing negative compile-fail documentation controls | 3 | Packet/permit cloning and consumed-permit reuse reject; not positive controls |
| **SIGMA subtotal** | **16** | Not an additional packet |
| Additional GPU127 legacy draw-order control | 1 | Historical bytes, random fields and draw order |
| Shared packet wire | 15 | Legacy/self-owned packet ownership transitions |
| Isolated native typed comparison | 12 | Five reference checks plus seven packet-ownership checks |
| Existing backend wire | 22 | Compatibility and request-guard lifecycle |
| **Separate ownership-bridge subtotal** | **49** | Last three rows, part of the same 66 |
| **Total** | **66** | `16+1+49`, not `66+49` or `66+16` |

Reproduce selectors with the ordinary direct-toolchain settings below. Use the
Rust 1.85.1 arm64 toolchain directory for `TOOLCHAIN`; invoke Cargo directly, not
a rustup-selected default. Each command had its own 600-second timeout, one build
job and one test thread; the worker stack setting was 8 MiB.

```sh
TOOLCHAIN="$HOME/.rustup/toolchains/1.85.1-aarch64-apple-darwin"
export PATH="$TOOLCHAIN/bin:$PATH"
export RUSTC="$TOOLCHAIN/bin/rustc"
export RUSTDOC="$TOOLCHAIN/bin/rustdoc"
export RUST_MIN_STACK=8388608

"$TOOLCHAIN/bin/cargo" test --release --locked --offline -j 1 -p pllm-garble --lib masked_wire_reference::arithmetic::material::tests:: -- --test-threads=1
"$TOOLCHAIN/bin/cargo" test --release --locked --offline -j 1 -p pllm-garble --lib compact_dpf::gpu127::tests::frozen_source_gpu_codec_vector_and_canonical_roundtrip -- --test-threads=1
"$TOOLCHAIN/bin/cargo" test --release --locked --offline -j 1 -p pllm-garble --lib compact_dpf::gpu127::tests::malformed_codec_profile_version_role_width_padding_truncation_and_substitution -- --test-threads=1
"$TOOLCHAIN/bin/cargo" test --release --locked --offline -j 1 -p pllm-garble --lib compact_dpf::gpu127::tests::preflight_entropy_failure_allocation_hooks_and_consumed_owner_erasure -- --test-threads=1
"$TOOLCHAIN/bin/cargo" test --release --locked --offline -j 1 -p pllm-garble --lib compact_dpf::gpu127::tests::frozen_legacy_bytes_random_fields_and_draw_order_remain_unchanged -- --test-threads=1
"$TOOLCHAIN/bin/cargo" test --release --locked --offline -j 1 -p pllm-garble --lib material_wire::tests:: -- --test-threads=1
"$TOOLCHAIN/bin/cargo" test --release --locked --offline -j 1 -p pllm-garble --lib fusefss_backend::predicates::typed_comparison:: -- --test-threads=1
"$TOOLCHAIN/bin/cargo" test --release --locked --offline -j 1 -p pllm-garble --lib fusefss_backend::wire::tests:: -- --test-threads=1
"$TOOLCHAIN/bin/cargo" test --release --locked --offline -j 1 -p pllm-garble --test sigma_masked_floor_chain -- --test-threads=1
"$TOOLCHAIN/bin/cargo" test --release --locked --offline -j 1 -p pllm-garble --doc masked_wire_reference::arithmetic::material -- --test-threads=1
```

Expected command counts, in order: `8, 1, 1, 1, 1, 15, 12, 22, 2, 3`.
Warnings were visible and unsuppressed: Core 64 in each command; Garble 39 in
the first eight and 91 in integration/documentation commands. These are warning
reports, not additional checks. The frozen GPU codec golden **was run**.
Shared frozen-wire/native-codec roundtrips and layout twins do **not** provide an
independent frozen full native-typed-comparison golden.

Arithmetic uses the legacy shared-packet branch plus its own wrapper metadata;
the isolated comparison's native opaque packet import is a separate bridge.
Passing each boundary does not establish a four-file end-to-end composition.
See [Shared packet ownership-bridge evidence](../../paper/methods/fusefss-completion-plan.md#shared-packet-ownership-bridge-evidence)
for its separate scope. Neither packet supplies remote authentication, durable
replay state, compiler/checkpoint provenance, complete nonlinear composition,
complete CPU/`SourceWork` funding, security review, model admission, wheel/SDK
qualification or activation. Publication qualification remains **0/10**.

## Correctness checks

Normal tests cover rectangular independent u128 equations, signed/wrapping
values, both layouts, tiny exhaustive scalar mask/share cases, legacy adapter
agreement, resource counters, malformed bindings and inputs, cancellation,
allocation/randomness failures and reservation retention. Replay with:

```sh
cargo test --release --offline -j 1 -p pllm-core --lib streamed_matrix::tests:: -- --test-threads=1
cargo test --release --offline -j 1 -p pllm-core --lib edge_contract::
cargo test --release --offline -j 1 -p pllm-garble --lib masked_wire_reference:: -- --test-threads=1
cargo test --release --offline -j 1 -p pllm-garble --lib fusefss_backend:: -- --test-threads=1
cargo test --release --offline -j 1 -p pllm-garble --lib compact_dpf:: -- --test-threads=1
cargo test --release --offline -j 1 -p pllm-garble --test sigma_masked_floor_chain
cargo test --release --offline -j 1 -p pllm-garble --test sigma_matrix_party
cargo test --release --offline -j 1 -p pllm-garble --test sigma_material_handoff
cargo test --release --offline -j 1 -p pllm-garble
```

The FuseFSS run checks the extracted shared logical allocator's existing users.
The DPF run adds independent AES wordwise/NIST/frozen OpenSSL vectors, complete
width-8/9 point and strict-less tables under both root phases, wide endpoints,
bounded expansion/work counters, canonical parsing and erasure/failure controls.
The complete `cargo test --release --offline -j 1 -p pllm-garble` run passes
490 unit tests, fifteen external/oracle checks and seventeen doctests; eight opt-in exporters or
measurements remain ignored. Its fourteen arithmetic tests cover every legal
small-ring shift/input/mask, exact RNG requests and measured AES block counts,
64-bit boundaries, all-gate entropy/allocation failures and retained-owner leases.
The external chain passes common masked outputs and dealer-only masks through
TR, GapARS and DReLU without opening intermediates to either evaluator, against
independent signed floor equations. The separate matrix-party external fixture
and typed local controls additionally cover matrix chaining, full-dot interval
overflow, role/consumer/source/mask mismatches, consuming failure and retained
budget/singleton lifetimes. Compile-fail controls reject cloning, metadata-built
capabilities and predicate-mask use as ring material. Unit tests share a test-only
matrix mutex; the production singleton is unchanged and ordinary parallel test
execution passes. Material-handoff controls additionally cover original inputs,
both matrix layouts, width-64 canonical residues, descriptor/packet substitution,
imported-party input rejection, single-use/re-export rejection, all allocation
failures and exact independent retained charges. A restored ring-only output
retains its absent interval and cannot qualify GapARS. Golden-envelope checks
preserve historical FuseFSS bytes. All 458 shared-core library tests pass in a
serial run, including fifteen edge-metadata and eight traversal checks; 280
independent source-terminal/arithmetic-mask negative controls also pass. These
do not establish general compiler graph provenance.
Direct Select controls additionally check exhaustive small-ring equations,
ring64 endpoints, exact native charges, independent DReLU/input roots,
typed hull provenance, bad bindings, out-of-order calls, malformed frames,
allocation/cancellation failures and retained frame/output leases. The typed
finish-before-begin path returns `Context` and consumes the owner; retries return
`Consumed`. The library Clippy checks for core/garble and compiler library check
also pass with release/offline single-job execution and warnings denied for Clippy.
Fixed-prefix controls additionally cover independent signed-floor/selection oracles,
all three issuances' entropy failures, allocation failures, zero-draw preflight,
late DReLU/Select frame substitution, private-use binding and original Y
address/lease continuity. Exact-budget controls retain all eight directed frames
and reconcile each drop's refund. Matrix guards remain with their actual input
backing; compile-fail checks reject intermediate escape. Public tests follow the
complete four-exchange schedule without exposing hidden intermediate values.
An independent native-predicate check observes D before final selection: X=-1
is negative after TR6, while every X=0..63 gives Y=0 and a true predicate, with
both Boolean masks and both roles. Compile-fail checks additionally reject
post-begin input reuse and public access to final payload/dealer-mask backing.
Filtered native reruns overlap the complete suite, not additional distinct checks.
Primitive parity supplies no trained-checkpoint, quality, WAN, performance or
whole-client memory result. Complete source-specific truncation, nonlinear
programs, compiler scheduling and ordinary SDK/gateway/benchmark execution remain
required for runnable SIGMA.
