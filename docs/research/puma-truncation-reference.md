# PUMA candidate: bounded replicated arithmetic

`pllm_core::puma` implements three-party scalar `TruncAPr`; `puma::prss`
implements its pinned pairwise setup and bounded sequential stream owner.
It is independent Rust, with no upstream runtime dependency. Specification:
`secretflow/spu@a42c261e50fe0602ccdec4b188b8dc2cbcfbb7b4`,
`libspu/mpc/aby3/arithmetic.cc:750–941`, as resolved in
[source contract §3.1](../../paper/reproduction-contracts/shared-profile-resolution.md#31-truncapr-is-stochastic-with-a-derivable-exact-law).
This is the pinned **candidate**, not proof of the evaluated PUMA build.

**Decision update, 17 October 2026:** [D6](../../paper/methods/DECISIONS.md#d6-approved-puma-source-derived-q18-profile)
approves `puma.source-derived-q18.v1`, exactly
[source resolution §3.3](../../paper/reproduction-contracts/shared-profile-resolution.md#33-approved-p-n1-profile-source-derived-q18),
including its fixed rounding/constants/order and corrected softmax/normalization
formulas. P-N1's choice is resolved for that named interpretation, not the missing
evaluated graph. [D9](../../paper/methods/DECISIONS.md#d9-fixed-count-private-token-feedback)
approves fixed-count private full-prefix feedback, lowest-index protected argmax
and final client-only IDs. Those approvals did not themselves change native inputs or
previous test/measurement record.

## Numeric contract

Unsigned wrapping ring64 represents signed two's-complement X. Require
`-2^62 <= X < 2^62`, `1 <= b <= 62`, and at most 1024 lanes. These bounds
apply to the reconstructed value, never individual random share components.
The compiler/caller must establish this public range obligation; the runtime
does not open, reconstruct or test private X to enforce it. For products and
accumulators, the obligation concerns the **intended unwrapped integer before
ring reduction**. An overflowing integer that wraps into the quarter-gap does
not satisfy it. Q18 multiplication uses b=18 only after proving that raw-product
bound; accumulation requires its own bound before any modulo reduction.

For fresh uniform R in ring64, C=(X+2^62+R) mod 2^64 and

```text
Y = ((C mod 2^63) >> b) - ((R mod 2^63) >> b)
    + (MSB(C) xor MSB(R))*2^(63-b) - 2^(62-b)
  = floor(X/2^b) + [(X mod 2^b)+(R mod 2^b) >= 2^b].
```

Thus upper rounding has probability `(X mod 2^b)/2^b`; exact multiples are
exact. There is no additional large-wrap error within the admitted quarter-ring
gap. `oracle::{distribution,truncate}` is a separate wide-integer, deterministic
**conditional-on-mask** oracle, not a replacement exact truncation protocol.
For X=-9,b=3, R=0 returns -2; low R=1..7 returns -1 (7/8 probability).

## Party-local protocol and material boundary

Party i owns only `(x_i,x_(i+1))`. A public pivot relabels A,B,H cyclically.
The low-level API consumes a public pivot; `prss::SourceParty::plan` now derives
it from the pinned public PRG and binds its cursor into the operation.
There is no global runtime/reconstruction function in production. H is the
third online party, not a dealer. H's authorized view includes the full random
mask R; H never receives C or an evaluator opening.
The source threat model permits at most one static semi-honest corruption.
Consistent replicated inputs, correct setup and authenticated private channels
are assumptions; this primitive provides no malicious-party verification.

The low-level `LocalMaterial` adapter imports **one party's** PRSS outputs.
The higher-level `SourceParty` issues these internally and never detaches them:

| Party | Authorized fresh components |
| --- | --- |
| A | rA, yA shared only with H |
| B | rB, yB shared only with H |
| H | rA,rB,yA,yB from its two pairwise streams |

R=rA+rB. The numeric law above assumes ideal fresh uniform words; source PRSS
implements their computational pseudorandom analogue. Low-level raw imports do
not certify freshness. `SourceParty` enforces source cursor issuance and retires
keys with the online owner. Test-only deterministic fixtures know all shares;
production has no aggregate dealer, deterministic seed constructor or key registry.

1. `begin`: A/B send their masked input contributions to each other. H draws
   private splitting words, adds MSB(R)/middle(R) to A's
   pieces and negates the original words for B, exactly the source equations.
   H sends two correlation frames and retains output `(yB,yA)`.
2. `open`: each evaluator supplies the peer masked contribution and its H
   correlation frame at one round barrier, forms C, computes its additive
   high-bit correction and subtracts yA/yB. It sends a `Reshare` frame.
3. `reshare`: A/B combine those pieces. Outputs are `(yA,z)`, `(z,yB)`,
   `(yB,yA)` in pivot order. `take_output` moves one local pair once.

Six directed online frames carry 8 ring words per lane in total, excluding
headers, setup and transport. The source owner fills H's complete `2*n` array
with one private-PRG call (all rb words, then all rc words). The low-level raw
material adapter retains its separate fresh-OS private-splitting path.

## Pinned PRSS semantics and custody

Additional immutable specification: SPU `libspu/mpc/common/prg_state.{h,cc}` and
`bazel/repositories.bzl`; the latter pins
`secretflow/yacl@5418371c4335f4a64fbd0bdabb0efd94da2af808`.
YACL `yacl/crypto/tools/prg.{h,cc}`, `crypto/base/symmetric_crypto.cc`, and
`crypto/utils/rand.{h,cc}` fix the following **64-bit little-endian** profile:

- Each role draws independent 128-bit public contribution, private key and own
  pairwise key. Native implementation uses separate OS entropy calls; no key is
  derived from public metadata or a central master seed. Public seed is the
  three contributions' sum modulo 2^128, not XOR.
- Actual source sends `self_seed` to **PrevRank** and receives **NextRank**'s
  seed; its prose comment is reversed. Each owner holds exactly its own and next
  edge keys, plus its private key. The third edge key is absent.
- Each fill starts AES-128-CTR at `LE128(cursor)` with raw little-endian key
  bytes. Inside that call, counter blocks increment **big-endian**, matching
  EVP CTR. Return cursor is `cursor+ceil(bytes/16)`; unused partial-block bytes
  are discarded. A buffered conventional CTR stream is not equivalent.
- Per truncation: public cursor +1 for one u64 pivot; pair cursor
  `+2*ceil(n/2)` for separate n-word mask and output fills. A uses First, B
  Second, H Both; selecting one stream still advances the shared logical cursor.
  H's private cursor advances n blocks; other roles' private cursors do not.
- Source computes `(raw_pivot+1)%3` and `(raw_pivot+2)%3`. The two u64 values
  that overflow alias roles in the source. This bounded owner rejects and burns
  them, without repair/resampling (public abort probability 2/2^64 under ideal
  uniform pivot words). The admitted numeric truncation law is unchanged.

`SetupParty` performs public contribution gathering, then one directed secret
seed exchange. Canonical role-ordered contributions, profile and original
Context bind a fresh setup digest. Operations additionally bind exact public,
mask and output cursor starts with distinct purpose labels. Those labels bind
metadata; they do **not** alter the source AES inputs or invent a KDF.

`SourceParty` owns PRSS and `PartyRuntime` together. `plan` burns a public draw;
`begin` reserves pair/private ranges and AES work **before** material allocation.
All online failures/cancellation retire both owners. There is no Clone, seed
export/import, persistence, cursor reset, detachable material or source fork API.
New setup always draws fresh OS seeds; its fresh contribution transcript rejects
old seed messages even if caller Context is repeated. Process-ID checks retire
forked owners. Arbitrary process-memory snapshot rollback is not a persistence
feature and cannot be made safe by these in-memory types.

Admission remains 1024 lanes/4096 operations. Additional cumulative work ceiling:
`(3*1024+3)*4096` AES blocks, locally reducible. Each evaluator uses `2*ceil(n/2)`
pairwise AES blocks; H uses `4*ceil(n/2)+n`, plus each role's one public block.
Fixed-size setup allocates no payload buffers. The direct `SourceParty` API admits
sequential TruncAPr-only schedules. Its compound owner below additionally composes
elementwise/matrix multiplication, typed Boolean AND and arithmetic-to-Boolean
carry circuits on that same stream.
Other kernels still need their source-ordered consumption contracts.

### Native Q18 multiply → truncate

`puma::multiply` now composes the existing source PRSS and truncation owners
with replicated multiplication. It implements the frozen
[multiply/truncate specification](../../paper/reproduction-contracts/puma/README.md)
inside `pllm-core`, for subsequent decoder integration.

`InputBounds` checks all four unwrapped signed-i128 product corners for every
lane before randomness. The whole product interval must lie in `[-2^62,2^62)`.
It also derives the all-mask rounded output interval. These public bounds remain
caller/compiler obligations; no private operands are opened to check membership.

Each role owns two adjacent shares of each operand. Its fresh pairwise zero share
masks the three local cross-products before one rotation to the previous role.
The received next-role component enters the existing truncation owner without
reconstruction or a second product snapshot. The execution ID is
`puma.spu-a42c261.mul-aa-truncapr18.ring64.quarter-gap.replicated3.v1`.
Setup, interval envelopes, phase/ordinal and separate zero/mask/output ranges
bind that identity; the approved Q18 numeric profile and AES algorithm remain.

`plan` admits the complete operation without consuming PRSS. `begin` consumes
the product's Both fill and emits the rotation. Only `rotate` draws the public
pivot and starts TruncAPr. Helper private splitting precedes output-mask fills;
its prepared buffer becomes the existing correction buffer. This retains exact
source ordering without an additional allocation or fresh-entropy fallback.
For `q=ceil(n/2)`, successful pair/public/private consumption is `3q/1/n` for the
helper and `3q/1/0` for evaluators. AES work is `6q+n+1` or `4q+1`, respectively.
Nine directed frames carry `11n` ring words before headers/setup/transport.

One non-cloneable compound owner couples all phases, keys, counters and failures.
Its unchanged 256 KiB ledger retains original rotations, truncation frames and
outputs through erasure and deallocation. Admission uses the joint conservative
`4096 + 128n` bound plus earlier live payloads. The original truncation-only
role estimate is insufficient: a helper retaining its outgoing rotation reaches
11 words/lane, versus the truncation helper's 10. The 1024-lane regression measures
this overlap. Retaining eight complete output pairs blocks the next full-size
operation before PRSS consumption. Every cancellation phase retires the chain.

`PartyRuntime.plan_matmul` also binds row-major matrix products to this same
owner and stream. Pinned `arithmetic.cc:628–660` and `fxp_base.cc:264–269`
specify one output-shaped zero fill and rotation, followed by truncation of
each **complete** dot product. Operand position, all three dimensions and the
public unwrapped accumulator bound enter a distinct matrix execution identity.
Each input/output has at most 1024 entries. Joint admission additionally prices
both replicated input tensors; `Plan.admitted_bytes()` reports that bound.
Matrix tests compare raw reconstructed dot products with independent signed-i128
sums, then check the stochastic output law, mixed elementwise/matrix cursors,
operand substitution and allocator-failure retirement through the exported API.

Native tests read the existing frozen independent vectors directly, checking
product components, complete output shares and all stream cursors across odd/even
chained shapes. Fresh OS-key exported-API tests cover up to 1024 lanes, the signed
rounding law and frame/work reconciliation. Allocation-index sweeps inspect
erasure and still-held reservations inside deallocation; compile-fail checks
reject cloning setup, runtime and Boolean-share owners. The current suite passes
**69 native checks** (35 unit, 8 truncation, 3 PRSS, 2 compound exported-API,
18 allocation, 3 compile-fail). The frozen multiplication oracle's 14 checks and
two independent Boolean/carry transcript checks also pass.
Focused formatting passes; Clippy passes with the existing unrelated
`needless_range_loop` warnings allowed. Replay uses the commands below.

Graph-wide range derivation, other mixed kernels, confidential authenticated
transport, protected feedback and ordinary SDK/Experiment execution remain open.
This native integration supplies no checkpoint quality or serving measurement.

### Typed Boolean products on the same stream

`PartyRuntime.plan_and`, `begin_and`, `receive_and` and `take_boolean_output`
implement the same source's
[`AndBB`](https://github.com/secretflow/spu/blob/a42c261e50fe0602ccdec4b188b8dc2cbcfbb7b4/libspu/mpc/aby3/boolean.cc).
Each party computes three local AND cross-products, XORs its adjacent fresh
edge words, then rotates the masked component. No party reconstructs a value.
Boolean shares have their own non-cloneable type and frame namespace; operand
position, nominal widths, setup, phase, ordinal and pair cursor bind admission.

The source chooses an 8/16/32/64-bit backing type from the larger public operand
width. Individual shares fill that type even when the reconstructed value has
fewer significant bits. One Both fill advances the shared pair cursor by
`ceil(n*backing_bytes/16)` and uses twice that many AES blocks per party. Public
and private cursors do not advance. Tail bytes retire after each fill, so replacing
these typed draws with ring64 draws would change subsequent truncation material.
Native payload storage remains fully charged u64 words, including narrow shares;
typed AES consumption is not a wire-byte or allocation-saving claim.

Independent scalar AES/XOR fixtures check complete rotation hashes across all four
backing types and partial-block transitions. Native checks cover every nominal
width 0–64 and up to 1024 lanes. Fresh OS-key exported-API execution interleaves
Boolean and Q18 matrix/elementwise operations on one source owner. Allocation
failure, substitution, replay, wrong output extraction and cancellation retire the
coupled keys; original frames remain charged until erasure and deallocation.
The existing 256 KiB ceiling is unchanged.

### Arithmetic-to-Boolean carry circuits

`PartyRuntime.plan_conversion`, `begin_conversion` and `advance_conversion`
implement the pinned source's full ring64 `A2B` and `MsbA2B`. A masked split
rotation precedes the source's Kogge–Stone addition or halving carry circuit.
Seven AND rotations use the existing Boolean primitive; their public vmap lane
order, backing widths and partial AES fills preserve the shared cursor schedule.
All intermediates remain party-local adjacent shares. The final Boolean owner
comes from `take_boolean_output` without reconstruction. These are arithmetic
conversion and sign primitives, not arbitrary signed comparisons or nonlinear
operators.

Admission checks the complete eight-round work envelope before randomness,
limits inputs to 512 lanes and prices `4096 + 256n` bytes plus previous live
payloads. This includes all original outgoing frames retained through completion.
Compound gate metadata stays inline within the fixed reserve. Round-specific
bindings reject replay/substitution; malformed frames, allocation failure and
cancellation retire the common PRSS owner. The per-runtime 256 KiB limit remains.

Independent OpenSSL/scalar fixtures check complete rotation transcripts, including
random padding bits in narrow carry stages. For three lanes, full A2B consumes
22 pair blocks and MSB consumes 11; neither consumes public/private PRSS. Native
checks cover signed endpoints, wraparound, odd lanes and maximum tiles. Exported
OS-seeded tests interleave both circuits with Boolean AND and Q18 matrix products;
allocation-index sweeps check every role and every circuit phase. Native frames
still store/charge u64 words (42n words across all parties), independently of the
source's narrower Boolean storage types. Transport/wire metrics and whole-model
range/fidelity claims require the subsequent SDK composition.

### Independent primitive and transcript vectors

For a zero AES key, eight bytes at cursor 0 are `66e94bd4ef8a2c3b`; a **new**
eight-byte fill at cursor 1 is `47711816e91d6ff0`. By contrast, a single 16-byte
fill at cursor 0 continues with `884cfa59ca342b2e`. Tests also cover starts
2,255,256 and odd lengths. A separate 4128-byte fill at cursor zero checks the
**within-fill** big-endian carry: block 255 is
`f70ddef93ba62588242a0e67d0d645e0`; block 256 is
`fb56cc09b680b1d07c5a52149e29f07c`. These differ from new fills starting at
little-endian cursors 255/256. SHA-256 of all 4128 ciphertext bytes:
`b844e6a17d006cd080e7adcdf6d82be099aead7190912154e5358900dc3bdc04`.
These vectors came from standard OpenSSL encryption
of zero bytes with `-aes-128-ctr -K <key> -iv <LE128(cursor)>`, not upstream code.

Full fixed fixture: public contributions `(1,2,3)`, private keys `(101,102,103)`,
edge keys `(11,22,33)` as LE128. First public word is `5284552621516802438`,
so `(A,B,H)=(2,0,1)`. At n=3,b=3, input `(-9,1,2^62-1)` yields
`(-1,0,576460752303423488)`. `prss_tests.rs` freezes all helper correction and
output-share words from independent OpenSSL plus integer arithmetic, not merely
the reconstructed result. Odd/odd/even chained fills check exact cursors.

## Binding, retirement and bounds

`Operation` binds profile, session, complete schedule digest, request phase,
monotonic ordinal, pivot, shift and lanes. Each runtime starts at ordinal zero;
one operation is live at a time. Every frame checks binding, sender, recipient,
message phase and exact size before arithmetic. Typed frames are not an
authenticated wire codec. Future transport must authenticate roles, keep
payloads confidential, bound receive allocation and cancel on decode failure.

Any runtime transition failure permanently retires that runtime; the ordinal is
burned before payload allocation/material use. Cancellation/drop wipes retained
heap buffers. Entropy/allocation failure has no retry or fallback. The caller
must propagate cancellation to the other two runtimes, burn producer tickets,
discard already returned/in-flight frames and avoid releasing a partial request.
Local failure cannot remotely erase another process. Caller-owned input copies
and stack/register copies are outside the heap-wiping guarantee.

### Live payload accounting repair — 17 October 2026

The earlier per-operation estimate was **not a retained-payload bound**. The
independent audit found that a fixed-helper P2 operation at 1024 lanes can leave
32 KiB of original outgoing frames after output extraction. Nine retained
operations exceed 256 KiB; 4096 could retain 128 MiB. The earlier 31-test result
below did not establish a lifetime bound. This repair preserves the arithmetic,
source AES consumption, role custody and one-use retirement behavior.

Limits remain **1024 lanes, 4096 operations and 256 KiB per runtime**. One
runtime ledger enforces `live charged payload bytes <= max_bytes - 4096`:
**258048 bytes** at the unchanged default ceiling. The 4096-byte fixed reserve
covers owner/ledger metadata. This is a per-runtime payload contract, **not a
process-global or RSS cap**.

- Each runtime/source-PRSS dynamic word buffer acquires a reservation before
  its fallible allocation. The exact requested capacity is priced; an unexpected
  capacity is rejected before secret copying. Allocation failure refunds the
  unused reservation and retires the owner, with no cursor rollback or retry.
- Reservations live inside payload ownership, following original `Message` and
  `ReplicatedShare` buffers through return, output extraction, cancellation and
  runtime destruction. Erasure and real vector deallocation occur **before**
  refund. Dropping a runtime cannot forgive its exported original buffers.
- Outer imports (`from_local`, raw material constructors and `from_received`)
  are fallibly bounded per object. Their caller allocations need a separate
  outer policy until admission. Accepted original input buffers are charged
  before any PRSS expansion; imported material and incoming frames are charged
  before arithmetic or further owned allocations. Receiving an original frame
  keeps its issuer reservation and adds a receiver reservation until free.
  That deliberate overlapping accounting is not a process-wide memory sum.
- `payload_usage()` is a read-only, clonable ledger handle; it exposes charged
  live/peak/limit bytes, not secret data or reset/refund authority. Payloads,
  shares, seed messages and material remain non-cloneable.
- **Fixed-allocation exception:** one fixed-size `Arc` ledger allocation at
  runtime construction uses standard Rust's abort-on-OOM behavior. Stable Rust
  does not make that constructor fallible here. It fits the fixed reserve;
  cloning handles/reservations adds no allocation. Variable payload allocations
  use fallible exact reservations. Allocator overhead, caller containers/header
  storage, independent caller/transport copies and other runtime instances are
  outside this per-runtime payload cap and require their own outer bounds.

The existing shape-only conservative estimate `4096 + 128*lanes` is unchanged.
`Limits::role_working_bytes` additionally reports an operation's working set
including its current original outgoing frames: H uses at most **10 words/lane**
(2 input + 4 material + 4 outgoing); A/B use at most **8 words/lane** at opening
(2 retained state + 1 earlier outgoing + 3 incoming + 2 new reshare buffers).
These estimates do not replace lifetime accounting for earlier payloads.

At 1024 lanes, the fixed-helper regression retains six original frame pairs;
the seventh operation rejects and retires within the ceiling. Dropping retained
frames before exhaustion permits later fresh ordinals. Eleven retained output
pairs alone similarly block the next operation. Refund after failure never
revives the retired owner. Tests observe zeroed bytes and still-held charges
inside the allocator's deallocation callback, including after its system free.

Context/session freshness across restarts is enforced by fresh setup without
seed persistence; authentication and abort propagation belong to transport.

## Verification and remaining gates

```sh
cargo test --locked --offline -p pllm-core --lib puma::
cargo test --locked --offline -p pllm-core --test puma_truncation --test puma_prss --test puma_multiply --test puma_allocation
cargo test --locked --offline -p pllm-core --doc puma
```

Historical pre-repair result: **31 tests passed** (13 unit, 8 truncation,
3 PRSS, 7 allocation). Its nine-file freeze, superseded by this audit repair:
`bafabbdd285eb4d9f5bd6f544bb17f870ebc9b50941af5c47760cefb2ceafaac`.

Repair verification: **37 tests passed** (14 unit, 8 truncation, 3 PRSS,
12 allocation); focused formatting checks passed. Allocation-index sweeps cover
all 14 role-specific `begin` allocation failures and four `open` failures after
earlier successful expansions, plus five success endpoints. They verify rollback,
erasure, transferred-input/incoming-frame pricing and sticky cursor retirement.
Retained-original-frame/output exhaustion, refund and owner-drop cases are covered.

**Repaired Rust inputs frozen for parent integration:** ten PUMA Rust files,
all `crates/pllm-core/src/puma/*.rs` and `crates/pllm-core/tests/puma_*.rs`.
SHA-256 of sorted `relative_path SP file_sha256 LF` rows:
`1b40c646243d300f217958e1db59a64c131e4d3ea277de5e1f345967d099ede2`.

Checks include 247,552 exhaustive small-ring equation/mask cases; actual ring64
party transcripts for all pivots, signed gap endpoints and wrapped masks;
exhaustive low-residue frequencies; replay/role/phase/context rejection;
cancellation, entropy/allocator failure and admission-before-allocation; source
AES/transcript vectors, seed direction/separation, stale setup rejection and
logical/AES-work cursor exhaustion.

P-N1/T-N1 adoption choices are closed for D6/D9's named contracts. Complete numeric
graph implementation, model-wide range proofs, authenticated wire transport,
mixed-kernel cursor scheduling, protected decoder/token loop, ordinary SDK
integration and independent original/common-model quality remain explicit gates. This
primitive supplies no composed-Qwen accuracy, security or performance certification.
