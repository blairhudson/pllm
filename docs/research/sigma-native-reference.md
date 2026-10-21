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
that distinction without a predicate conversion. Source table lookup and full
nonlinear composition still need executable implementations; FuseFSS
additive-share operations are not substitutes. Source CPU activation performs
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
Native leases move intact; cancellation erases/drops local state before wrapper
refund, and externally retained frames keep their own native/wrapper leases.
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

Lookup remains unimplemented and separately approval-gated. The pinned source
opens a masked value **and** masked sign correction, then derives final mask
`rho=2*u-r`, uniform marginally but parity-correlated with internal r. Reconstructed
and deferred-share output boundaries are distinct; neither silently replaces the
other in full GELU/SiLU. See the unapproved next-gate audit in
[`../plans/sigma-source-lookup.md`](../plans/sigma-source-lookup.md).

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
There is no upstream wire-byte equivalence, GPU performance or security-proof
claim. D17 separately approves complete-output-ring arithmetic masks; its source
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
