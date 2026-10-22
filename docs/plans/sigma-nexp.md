# SIGMA bounded nExp owning region

Status: **UNAPPROVED PROPOSAL — DOCUMENTATION ONLY**.
Recorded 22 October 2026. Parent review and explicit user approval are required
before implementation. No decision number is assigned. This document adopts no
executable, numeric profile, source artifact or validation result.
D17/D19/D20/D22/D27 remain unchanged and do not authorize this new composition.

Source audit: [`../research/sigma-normalization-softmax-audit.md`](../research/sigma-normalization-softmax-audit.md).
No code edits, builds, tests or benchmarks accompanied this proposal.

## Next user decision — bounded nExp only, after D27

**Proposed decision wording; not an approval record:**

> Approve **conditional local nExp implementation only**: 1–256 ring50/Q12 lanes,
> original-owner `0 <= X < 2^38` provenance, paper Select16 clipping, private
> byte coordinates, two reconstructed D19 lookups, fresh-masked consuming product
> and reconstructed GapARS50,12. Approve the equation, 18 uses, eleven exchanges
> and 22 frames below, including explicit CPU/paper departures; retain the
> 64 MiB ceiling, continuous charges and D27 role/abort/barrier/source patterns.
>
> Require reviewed canonical records, actual layout/quote census and exact floor
> evidence **before their construction/hash/issuance boundaries**. Joint-transcript
> privacy remains unqualified. No new D22/D27 extraction or range authority,
> post-D22 Multiply, softmax/normalization, transport, decoder or SDK approval.

The smallest mathematical scope is exactly
`floor(E0[min(X,65535) mod 256] * E1[floor(min(X,65535)/256)] / 4096)`.
The reusable product implementation must first admit only this gate's two
distinct ring50/Q12 payload owners, each in `[0,4097)`; `P` has interval
`[0,16777217)` in Q24 and final E has `[0,4097)` in Q12. Reusable code does not
authorize other products or a square using one capability twice.

### Current API check: what D27 actually removes

Read-only recheck against implementation committed in `b0f07f24`; no validation
was rerun for this packet. The two eleven-exchange schedules are different
regions: D27's count is not nExp implementation or source-wire evidence.

| Previously missing mechanism | Current API/evidence boundary |
| --- | --- |
| Consuming single-role D22 handles | Native `LocalLinearActivationRun::into_roles` / `into_roles_from_reserved`; compiler `SigmaExecutionRun::into_roles(budget)` exist. Do not propose another D22 role scaffold. |
| Source retention across split and returned owners | `execution_roles.rs::RetainedRoles` keeps an Arc to the **same** original checked source and ExternalCharge; mask already shares it. No Arc-uniqueness assumption or source-copy replacement. |
| Immediate sibling burn and final barrier | Native `linear_activation/roles.rs::Coordination` clears both mailboxes on error/cancel/drop/unwind; `finish` requires both Ready. Caller polls run outside both mutexes; locked native work uses internal polling. Retained frames keep actual guards/leases; previously borrowed bytes cannot be revoked. |
| Split preconstruction admission | Native `local_linear_activation_role_split_resources` and compiler `sigma_execution_role_split_resources` quote added layouts before allocation; native receives an exact reserved slice. Original source charge is neither refunded nor reacquired. This prices **D27**, not the proposed nExp owners. |

**Not removed:** D27 role outputs expose only role/provenance and fallible
descriptor access; they have no consuming nExp/native extraction API. D20 creates
its output Binding with `interval=None` (`typed/activation.rs::activation_reserved_with`).
D22 certifies the MLP complete sum and activation input, not a nonnegative
activation output, score difference or norm energy. Matching metadata, a
successful role run or a fresh root with copied values cannot supply that proof.

For this first gate, a new clear-value entry must check the whole declared input
slice before calling reserved root preparation: existing public
`prepare_root(...).encode(...)` prepares masks **before** encode checks values.
It cannot support the proposed zero-operation-draw rejection claim by itself.
An inherited entry may exist only for actual matching mask/role owners whose
original private Binding interval is contained in `[0,2^38)` and whose consumer
matches this region. Its provenance stays LocalAssumption unless separately
certified; no descriptor-based import or automatic D27-to-nExp adapter is allowed.
If no genuine upstream owner satisfies that check, inherited admission stays
fail-closed; the bounded clear-root gate does not require inventing a new
producer certificate or another D27 adapter.

### Approval choices versus implementation evidence

The decision must explicitly select narrow Select16 rather than CPU Select50,
the existing two-exchange TR, common-masked final GapARS output rather than the
paper's returned shares, private unsigned coordinate interpretation, the fixed
uses, product mask/split rule and byte domains below. It must not silently select
the CPU softmax widths or normalization epsilon/affine policy.

This packet is **not ready for unconditional execution approval**: new canonical
record field order and exact owner layouts/quote formulas remain design-freeze
items, not invented byte totals. A conditional yes permits their bounded
implementation/review, not bypassing them. Engineering acceptance and outstanding
privacy questions are separated below so neither becomes a claim of settled
cryptography or another role-scaffolding prerequisite.

## Evidence identity and bounded scope

Mathematical target: `papers/r13-sigma.pdf`, one-based PDF p.9, Fig.8 and §5.2.2;
recorded paper SHA-256
`f911b715b0e0523fa4d77f21c28c6a0440c5be781e67b5b30419f20e505a6ebf`.
Author source:
[`mpc-msri/EzPC@aa0f2f860266fb79901405ef5e9965b8c9fe2e88`](https://github.com/mpc-msri/EzPC/tree/aa0f2f860266fb79901405ef5e9965b8c9fe2e88),
under `GPU-MPC/ext/sytorch/ext/llama/`:

- `api.cpp:2222–2253`: clip, TR16,8, two 8-bit tables, product and shift.
- `api.cpp:2264–2266`: `nExp` immediately calls `nExp_SIRNN`; the following
  direct 16-bit LUT code is not its executed branch.
- `api.cpp:2989–3036`: positive-input clipping.
- `api.cpp:2121–2208`: lookup correction/reconstruction.
- `api.cpp:1185–1236`: masked elementwise multiplication/reconstruction.

These are immutable specification citations, not author runtime adoption or an
evaluated 50/12 invocation lock. Keep CPU numerical equations, D16's separately
approved GPU-derived DPF and D17's full-ring mask adaptation distinct.

Three proposed adaptations must not be described as verbatim source execution:

| Boundary | Paper / pinned CPU artifact | Proposed PLLM composition |
| --- | --- | --- |
| Clip arithmetic Select | Paper Fig.8 uses Select16. Pinned `api.cpp:2224` calls `SlothClip` with global output width; `3026` calls global-width Select, hence Select50 under the proposed ring50 setting. | Follow the paper's narrow Select16, not that global Select50 call. |
| TR16,8 | Pinned CPU `TruncateReduce`, `api.cpp:2850–2913`, reconstructs in one exchange. | Existing D17/GPU127 arithmetic owner uses carry and output exchanges: two. |
| Final GapARS12 | Paper Fig.8 returns party shares of the masked result. | Existing arithmetic owner reconstructs an opaque common-masked result, adding that explicit output boundary. |

Consequently, eleven exchanges below describe an **UNAPPROVED PROPOSED PLLM
composition**, not a verbatim paper/CPU schedule or an observed runtime.

Proposed geometry: **1–256 independent lanes**, under the existing combined
**64 MiB logical live ceiling**. Input and output are ring50/Q12. The external
input must have a bound, fixed before private values enter execution:

```text
0 <= X < 2^38
```

This represents a nonnegative Q12 difference, not arbitrary signed activation
input. For clear-root admission, the encoder **must check every integer code
against `0 <= X < 2^38` before operation entropy or secret allocation**; the
complete immutable declaration must also pass preflight. Label the premise as a
local developer assumption, not a compiler proof. Invalid roots reject without
an operation nonce, masks, keys or secret buffers. Borrowed caller-owned clear
storage predates this boundary and remains separately charged.

For inherited inputs, consume the original matching mask and evaluator owners
with source-/edge-bound interval provenance contained in `[0,2^38)`. No copying
into new roots, reconstructed-secret handoff, mask replacement or relabeling of
metadata is permitted. Neither masked payloads nor observed extrema establish
this provenance. **D22/D27 supply no attention or nExp authority**: the MLP handoff
cannot certify nonnegative score differences or authorize this region. A future
compiler certificate is separate work. No row maximum, inverse, norm, attention
score generation or full decoder is included.

## Exact integer schedule

Let `K=65536`. Every division below is floor; tables already contain floored
payloads:

```text
X40 = X mod 2^40                         # private projection, same clear value
H   = [X40-K >= 0]                       # DReLU40, XOR-masked
B   = NOT H
V   = Select16(B, (X40-(K-1)) mod 2^16)
C   = V + (K-1) mod 2^16                 # min(X,65535), unsigned16/Q12
J0  = C mod 256
J1  = TR16,8(C)                          # unsigned high8, not signed ARS
U0  = reconstructed LUT8_to_50(E0,J0)
U1  = reconstructed LUT8_to_50(E1,J1)
P   = reconstructed elementwise_product50(U0,U1)
E   = GapARS50,12(P)
```

`X40-K` fits signed40 throughout the proposed input domain. The predicate is
computed before narrowing the Select arithmetic operand and its mask to 16 bits.
Projection reduces both common masked values and dealer masks; it neither
recovers nor sign-extends a masked secret. Public constants change common
values, not additive masks; predicate complement changes the common XOR bit,
not its Boolean mask. All affine mask relations must be derived exactly.

`J0/J1` are private table-coordinate views bound to C, not independent roots.
The concrete mapping and ordered uses are specified below. In particular,
ordinary ring8/Q12 is invalid, and the public typed TR's signed output metadata
does not describe unsigned `J1`. C's original owner stays live through both reads.

```text
E0[j] = floor(4096*exp(-j/4096)), j=0..255
E1[j] = floor(4096*exp(-j/16)),   j=0..255
E     = floor(E0[J0]*E1[J1]/4096)
```

Lookup payloads are ring50/Q12; `P` is the complete Q24 product with
`0 <= P <= 2^24`, including equality at X=0. This proves its quarter-ring
GapARS50 domain `[-2^48,2^48)`. Output is an opaque typed common-masked
ring50/Q12 capability with half-open encoded interval **`[0,4097)`**. No extra
floor, clipping, float conversion or final remask follows GapARS12.

## Table proof and source identity: still required

Existing literals are in
`crates/pllm-garble/src/fusefss_backend/composition/source_table_values.rs`,
accessed by `source_tables.rs:31–36`. Reuse requires extraction into method-neutral
immutable table ownership, not importing FuseFSS's protocol, profile or budget
authority. The existing source interpretation records hashes over successive
u64 little-endian entries:

| Table | Recorded SHA-256 |
| --- | --- |
| E0 | `e094819aa436afcf59d46bfd2937b91860045f5c905387d98e271731259a963b` |
| E1 | `21a20b930ba1cd6dd57601ec54b4b6b85f206e553304b7c58da5dff2f45c5ed9` |

**Outstanding acceptance proof:** every selected literal must equal its exact
floor formula, including boundary entries. Hash agreement proves byte identity,
not transcendental floor correctness. Bind the selected bytes, equations,
source pin, scale, coordinate mapping, ring, template and complete invocation.
Existing fixture/oracle history is useful evidence to inspect, not validation
of this proposed region. Do not generate online libm tables or claim author
compiler/libm byte identity. No fitted approximation or substitute table/profile
is authorized.

Reuse/extract the existing exact rational machinery in
`paper/reproduction-contracts/fusefss/gate-set/reference.py:138–174`, not a new
proof framework: `exp_bounds(Fraction(j,divisor))` encloses the positive
exponential, then `certified_floor(4096/hi,4096/lo)` must resolve the same integer
for each entry, with divisors 4096 and 16. Compare all selected literal bytes and
their recorded hashes against those exact certificates. Dual Decimal precision
alone is regression evidence, not this floor proof. Extraction shares public
mathematics only; no FuseFSS protocol or profile authority transfers. This audit
did not execute that proof against the proposed region's selected literals.

## Shared product owner, not a matrix-singleton workaround

Reuse negative-mask equations in
`crates/pllm-garble/src/masked_wire_reference/product.rs:1–20`:

```text
A = -r0
B = -r1
C = r0*r1 + rP
party_p = A_p*U1hat + U0hat*B_p + C_p + [p=1]*U0hat*U1hat
```

Their sum is `U0*U1+rP mod 2^50`. A proposed reusable native/typed elementwise
owner must consume the actual matching dealer/evaluator capabilities, sample a
fresh full-ring output mask, preserve role separation and reconstruct only the
masked product. Existing legacy scalar multiplication supplies algebra, not
this new owner or provenance proof.

For **each lane and issuance**, sample `rP` independently and uniformly in
`Z/(2^50)`, independent of both lookup randomness streams and all coefficient
splits. Compute A/B/C modulo `2^50`; choose fresh independent uniform ring50
shares `A_0,B_0,C_0`, then set `A_1=A-A_0`, `B_1=B-B_0`, `C_1=C-C_0` modulo
`2^50`. The sharing draws and `rP` must not be reused across coefficients,
lanes, invocations or retries. A party owns only its three coefficient shares;
the dealer mask owner retains `rP`. The common input masks are the actual
lookup-output masks, not newly chosen replacements.

Keep this private-product owner **distinct from the matrix singleton**. Do not
use repeated 1x1 matrices, release/reacquire a matrix guard, copy guarded input
payloads into new roots, or relax existing singleton rules. A guarded upstream
owner remains guarded through its last actual use. No security elevation follows
from extracting shared algebra.

## Mask, capability and reconstruction contracts

- Consume each external capability once into a fixed owning region. Retain
  original backing, source identity, mask identity, roles, leases and any guard
  through the precommitted last use. No exported fanout, capability cloning,
  arbitrary graph execution or remasking.
- Reuse private fixed-use ledger mechanics beside
  `masked_wire_reference/typed/internal_use.rs`; bind ordered operand slots,
  projections, affine relations, both lookup uses and final product/shift.
  Public `ConsumerUse` metadata alone grants no authority.
- Use unchanged D16 GPU127 mapping and D17 full-output-ring arithmetic masks.
  Boolean predicates retain independent one-bit masks and their distinct type.
  Do not treat XOR-masked predicates as arithmetic shares.
- Both D19 lookups use the **reconstructed** boundary. Each opens masked
  `R=sigma*T[j]+r` and `Csign=Bsign XOR q`, then masked Select output S, returning
  `2*S-R=T[j]+rho`, with `rho=2*u-r`. Independent full-ring r/u and Boolean q
  preserve D19's dependency: rho is marginally uniform but parity-dependent on r.
  Separate lookup invocations need fresh material; no independent-remask claim.
- Product consumes those common-masked outputs, not deferred shares or decoded
  frame bytes. Reconstruct only its masked result. GapARS consumes that owner
  and provides the final opaque common-masked output. No intermediate clear
  table index, payload, clipping predicate, product or output is disclosed.
- Bind source/table/numeric identity, plan/session/operation, issuance, every
  wire/use, geometry, lane index, consumer and role. Matching common-input views,
   mask provenance and complete table-domain coverage need explicit proofs and
   negative controls; bytes or metadata cannot mint capabilities.

D19's `rho mod 2 = r mod 2` dependency is **not a demonstrated attack**. Marginal
uniformity and integer correctness nevertheless do not prove privacy of the
joint two-lookups → product → GapARS transcript. Independent composition review
must account for both lookup transcripts, correlated internal masks, fresh
product splits/output mask and final reconstruction. No composed security claim
is made here, and functional/runtime tests cannot discharge that obligation.

### Concrete joint-view questions; no answer is adopted

Review one noncolluding evaluator's complete issued material and online view,
with the local trusted-dealer premise stated explicitly; D27 does not establish
independent hosting or privacy against a dealer observing masked traffic.

1. Can the claimed FSS/lookup security assumptions cover **both** correlated
   coordinates of the same C, both correction openings and nested Selects,
   together with that role's product coefficients and final GapARS messages?
   Fresh lookup streams do not make J0/J1 independent secrets.
2. For each lookup, `Uhat=T+2u-r` and `R=sigma*T+r`, with `sigma` equal to ±1,
   imply `Uhat mod 2 = R mod 2`. Account for this exact joint relation and
   `rho mod 2 = r mod 2`; neither marginal uniformity nor this relation alone
   proves privacy or an attack. Does the product's independent mask/splitting
   suffice under the actual joint material/view, not just isolated equations?
3. Does the argument include the reconstructed product, GapARS carry/output
   openings, role-specific public-product term, reused parent-coordinate backing
   and error/abort behavior? Which leakage and semi-honest/FSS assumptions are
   needed? Do not infer malicious security or authentication from digest checks.

Functional local prototyping, if explicitly approved conditionally, must remain
labelled as such while these questions are open. No joint-view simulation proof
or composed security qualification is supplied by this document.

## Private coordinate metadata and fixed ordered uses

Reuse the private owner/ledger pattern in
`masked_wire_reference/typed/activation.rs:927–989,1392–1456` and
`typed/internal_use.rs`, not a public projection or fanout API. Each private
coordinate record binds its actual parent C binding, source, issuance, role,
lane range, target table hash, mapping tag and resulting mask binding:

| Private record | Representation and mapping |
| --- | --- |
| C | Unsigned ring16/Q12 code, proved `0 <= C < 65536`; common additive mask in ring16. |
| J0 | Unsigned ring8/**Q0 integer coordinate**, tag `low8.v1`; clear coordinate `C mod 256`, masked word `Chat mod 256`, dealer mask `rC mod 256`. E0 supplies the `j/4096` interpretation. |
| J1 | Unsigned ring8/**Q0 integer coordinate**, tag `high8-tr16-8.v1`; clear coordinate `floor(C/256)`. Use native TR16,8 with fresh full-ring8 output mask; E1 supplies the `j/16` interpretation. |

Q0 describes the table coordinate, not a numeric rescaling of the original
real value. Both coordinates have domain `[0,256)`. The high-byte step must use
the native unsigned TR equation behind a **private region adapter**, preserving
its original output owner/mask; do not call public `typed::tr` then relabel its
signed result (`typed.rs:1173–1183`). A signed interpretation would misread
indices 128–255. No public
edge widening, ring8/Q12 declaration, general cast, mask sign extension or
ordinary external capability is introduced.

For private `ConsumerUse`/Binding declarations, coordinate numeric metadata is
unsigned `(bits=8,fractional=0)`. Its declarative mapping digest binds the mapping
tag, parent edge identity, target table hash and integer domain. A separate
runtime provenance digest binds that declaration to the actual C Binding and
the resulting coordinate/mask Binding under the coordinate domain. The private
ledger binds this provenance and role. Do not insert runtime mask identities
into the immutable spec or change the original public source identity: the spec
declares edges/mapping rules; issued bindings/ledgers prove their actual-owner
association. This avoids a spec/binding digest cycle and declares only the
proposed private adapter contract, not new authority for typed constructors.

The template has exactly 18 ordered uses, burned before checking, with role-local
ledgers 0/1 and dealer mask ledger 2. Each slot binds actual owners and its operand
ordinal; projection/affine slots bind the parent and derived mask relation too:

| Slot | Consumed/authorized use |
| ---: | --- |
| 0 | External X capability enters region once. |
| 1 | Original X → private X40 projection, value/mask modulo `2^40`. |
| 2 | X40 → signed40 affine A=`X40-K`; additive mask unchanged. |
| 3 | A → DReLU40 H. |
| 4 | H → B XOR complement; Boolean mask unchanged. |
| 5 | Retained X40 → W16=`(X40-(K-1)) mod 2^16`; mask reduced modulo `2^16`. |
| 6 | B → Select16 predicate, operand0. |
| 7 | W16 → Select16 arithmetic value, operand1. |
| 8 | Select output V → C=`V+K-1 mod 2^16`; mask unchanged. |
| 9 | Retained C → J0 low8 coordinate. |
| 10 | Same original C → native unsigned TR16,8. |
| 11 | Original native TR output → private J1 coordinate record, preserving backing/mask. |
| 12 | J0 → reconstructed E0 lookup. |
| 13 | J1 → reconstructed E1 lookup. |
| 14 | U0 → product operand0. |
| 15 | U1 → product operand1. |
| 16 | Product P → GapARS50,12. |
| 17 | GapARS output E → final opaque typed handoff, without another frame. |

Keep X40 through slot 5 and C through slot 10; both lookups' original outputs
survive until product consumes them. Retained frames can keep backing/leases
longer. Public metadata cannot authorize additional reads or reconstruct a
coordinate capability. No intermediate is exported.

## Proposed local wire contract — also unapproved

Reuse the existing typed activation wrapper convention: a consuming local frame
enum owns one unchanged native frame and its original lease
(`typed/activation.rs:959–977,1399–1412`). For role execution add D27's shared
abort guard and fallible byte access: forward only while live, retain the same
source/charge and native frame backing even after abort.
There is **no extra serialized region envelope and no final region exchange**.
The wrapper's private state accepts exactly the ordered phases below. This is a
local peer codec proposal, not authenticated transport or a material importer.

Proposed domains/version strings:

- Region: `pllm.sigma.nexp-region.v1`.
- Private coordinate metadata: `pllm.sigma.nexp-coordinate.v1`.
- Reusable product: `pllm.sigma.elementwise-product.v1`.
- Product common-input digest: `pllm.sigma.elementwise-product.common-inputs.v1`.
- Product magic: eight ASCII bytes `PLLMSP01`; product output phase byte 1.

Let `D(parts...)` be the existing typed digest convention
(`typed.rs:129–136`): SHA-256 of each part preceded by its u64-LE byte length.
All domain strings below are ASCII, without implicit trailing NUL. Scalars use
canonical little-endian bytes; roles are 0/1 (dealer ledger role 2 is never a peer
sender). Lane vectors are ordered u64-LE words, all checked before hashing.

The region binding is
`D(region-domain, spec-digest, original-X-binding, final-output-binding,
ledger0-digest, ledger1-digest, dealer-ledger-digest, fresh-region-nonce32)`.
The immutable spec digest binds source pin and paper-equation target, both exact
table hashes, complete input/output descriptors and required interval, all
declarative private wire/mapping/use records, numeric widths/scales, lane count/first index,
plan/session/operation, implementation and next consumer. Binding digests use
the existing complete private Binding convention, including source, namespace,
mask identity, issuance, consumer and interval. Nonces are freshly dealer-issued
after preflight; they supply uniqueness, not authentication.

The immutable spec contains no runtime Binding, mask identity or issuance; those
enter the separate region/ledger bindings only after spec construction. Exact
canonical field encodings/order for the new spec and coordinate-provenance
records must be frozen before their constructor/hash boundary using existing
edge/use digest conventions.
**This remaining digest-record encoding blocks unconditional activation**;
the product/header byte contract below does not silently resolve it.

For global stage ordinal `s=1..11`, use
`stage-operation=D(region-binding, ASCII("stage.v1"), u16le(s))` in native
contexts. The two phases of a single TR/GapARS owner share its first-stage
operation binding; the second stage is enforced by owner state/local phase.
Likewise each lookup's nested Select is bound by unchanged D19 context derivation.
Thus repeated native phase bytes do not permit cross-stage substitution.

For product stage 9, define its common invocation binding as
`D(product-domain, region-binding, stage-operation, U0-binding, U1-binding,
P-binding, bytes(50,24), u64le(L), u64le(first-index), fresh-product-nonce32)`.
**Both ordered input bindings are mandatory**, even though multiplication is
commutative. Its common-input digest is
`D(product-input-domain, product-binding, U0-binding, U1-binding,
u64le(L), lane-major-U0hat-bytes, lane-major-U1hat-bytes)`.
Only common masked words enter this digest, never private shares or masks.

The new product frame uses the native 76-byte header layout:

| Byte range | Field |
| --- | --- |
| `[0,8)` | `PLLMSP01` |
| `[8,40)` | Product invocation binding32 |
| `[40,72)` | Ordered BOTH-input common digest32 |
| `[72,73)` | Sender role0/1; receive must equal `1-local_role` |
| `[73,74)` | Product output phase1 |
| `[74,76)` | L as u16-LE, `1 <= L <= 256` |
| `[76,76+8L)` | Own masked-product additive contribution, lane-major u64-LE; each word `<2^50` |

Exact length is `76+8L`, no padding or trailing bytes. Full peer length, magic,
both digests, opposite role, phase, lane count and every canonical residue must
validate **before any output mutation or share addition**. Then add matching
role contributions modulo `2^50` to reconstruct only `P+rP`. Same-role frames,
swapped/mismatched input views or bindings, changed issuance, replay and repeated
execution burn local state; coordinator retires its sibling.

For the proposed owning region, sibling retirement is an **enforced internal
coordinator obligation**, not an instruction for an evaluator caller to remember.
Follow D27's abort-first, both-mailbox retirement and two-Ready output barrier;
escaped output wrappers must fail closed and their native backing must be erased
on abort. Caller cancellation callbacks run outside coordination and role locks,
with unwind-triggered abort; native locked work uses a non-reentrant internal
poll. Keep source pins, the same source charge and new split reservations until
their actual final coordinator/frame/output owners drop. Extract reusable
private cancellation/retention mechanics where appropriate rather than building
a parallel public ownership/import framework. This adds no twelfth exchange.

The complete proposed wrapper dispatch order is:

| Global stage | Native magic / local phase | Payload per lane | Bytes per direction |
| ---: | --- | --- | ---: |
| 1 | `PLLMSA01` /1, clip DReLU | Canonical bit | `76+8L` |
| 2 | `PLLMSL01` /1, clip Select16 | Canonical ring16 word | `76+8L` |
| 3 | `PLLMSA01` /0, TR carry | Canonical bit | `76+8L` |
| 4 | `PLLMSA01` /1, TR output | Canonical ring8 word | `76+8L` |
| 5 | `PLLMLU01` /0, E0 lookup | Ring50 R, then Boolean correction | `80+16L` |
| 6 | `PLLMSL01` /1, E0 nested Select | Canonical ring50 word | `76+8L` |
| 7 | `PLLMLU01` /0, E1 lookup | Ring50 R, then Boolean correction | `80+16L` |
| 8 | `PLLMSL01` /1, E1 nested Select | Canonical ring50 word | `76+8L` |
| 9 | `PLLMSP01` /1, product | Canonical ring50 contribution | `76+8L` |
| 10 | `PLLMSA01` /0, GapARS carry | Canonical bit | `76+8L` |
| 11 | `PLLMSA01` /1, GapARS output | Canonical ring50 word | `76+8L` |

Existing native headers retain the same byte 0–75 fields (magic, native invocation
binding32, common-input digest32, sender, native phase, u16-LE lane count), not
the new product's digest domain. Lookup adds bytes 76–79: index width 8, output
width 50, reconstructed boundary 0, reserved 0; payload is lane-major R/correction
pairs. The nested Select retains its own invocation/input digests. Native
canonical rules remain unchanged: bit words are exactly 0/1; ring-n words have
all higher bits zero. References: `arithmetic.rs:522–563`, `selection.rs:18–23`,
`source_lookup.rs:730–773`. Validate the expected inner frame completely before
mutation, including contexts tied to the same region and issuance.

There are 22 directional frames and proposed total peer bytes
`2*(9*(76+8L)+2*(80+16L)) = 1688+208L`. This excludes dealer material, local
wrapper/workspace charges and any future transport framing. It is a proposed
codec calculation, not measured traffic, source-wire equivalence, replay
persistence, integrity protection or malicious-output verification. Product
material serialization/import is not authorized or supplied by this contract.

## Derived sequential exchanges and continuous admission

| Stage | Sequential peer exchanges |
| --- | ---: |
| Clip DReLU40 | 1 |
| Clip reconstructed Select16 | 1 |
| TR16,8 | 2 |
| Reconstructed E0 lookup | 2 |
| Reconstructed E1 lookup | 2 |
| Masked product reconstruction | 1 |
| GapARS50,12 | 2 |
| Total | **11** |

This is the **derived PROPOSED PLLM composition schedule**, not a verbatim
source schedule, observed timing, implemented frame accounting or a performance
result. Select width, TR exchanges and final reconstruction adaptations are
listed above. Batching the lookups or changing boundaries needs separate review.
The proposed wire contract must be approved, implemented and charged; existing
transport codecs do not automatically cover it.

### Owner-defined quote before construction and hashing

Following the owner-defined metadata/hash quote pattern in
`typed/activation.rs:62–124,639–795` and
`pllm-core/src/edge_contract.rs`, the proposed region must expose an allocation-
and hash-free checked quote before constructing its spec, coordinate records,
use arrays, bindings or digests. It must price named concrete input/output/Result
slots and owner metadata, not an unexplained constant workspace allowance.

The quote covers region/product declarations; all fixed coordinate/wire records;
the 18 uses and three role-bound ledgers; retained original descriptors/source
pins/table sources; constructor/return overlap; table validation and rational-
certificate comparison inputs; both SHA states/encoding buffers for nested
ordered-input and binding hashing; canonical common-input hash traversal;
native resource-query input/output records; aggregate-admission calculation;
and constructor-to-native handoff overlap. Public rational certificate generation
is an ordinary offline proof task, not an online region routine or copied
evidence archive. Its retained selected literals/proof identities still count
where the region uses them.

For each sequential phase, compute retained owners plus its simultaneously live
scratch/return records with checked arithmetic; take maxima only across phases
that cannot coexist. Caller-retained copies, preexisting inventory/source backing
and outer compiler overlap remain additional. Reserve metadata/hash capacity
**before** spec construction, table validation/hashing or source copies, retain
the guard through final native capacity transfer, and reserve execution overlap
before any operation entropy/secret allocation. Transfer/split original leases
atomically without refund/re-reserve. This quote is a required owner API, not an
implemented API or completed byte inventory. **Exact owner layouts and quote
formulas block unconditional activation**. Conditional design implementation
permission does not satisfy that gate: the inventory must be reviewed before
its corresponding construction/hash/issuance boundary. No unstated metadata or
workspace exemption is authorized.

Validate full geometry, domains, table/template identity and checked resource
sums, then reserve aggregate capacity **before entropy or secret allocation**.
Price roots, masks, original backing, two immutable lookup table copies per
table as required by actual owners, DPF keys, native and typed wrappers,
projections, ledger, product material, workspace, retained sends, borrowed
receives, outputs and every transition overlap. Caller-retained source/capacity
owners remain additional charges. Split original leases without a
refund/re-reserve gap, under the unchanged 64 MiB combined logical ceiling.
This is not process RSS or allocator/stack qualification.

Erase and deallocate backing before refunding its lease, including partial
issuance failure. Any malformed/replayed/out-of-order frame, mismatched use,
allocation/entropy failure or cancellation burns remaining local execution.
The internal coordinator synchronously retires both siblings. Retained frames,
outputs and cancelled party metadata retain their charges through actual drop.

Frame validation, checksums and consuming ownership do not authenticate delivery,
establish independently hosted parties, prevent durable replay or prove composed
security. No legacy transport reuse or security qualification is claimed.

## Implementation-ready acceptance gates; none performed here

These are ordered gates, not authorization to start implementation now. A
conditional implementation decision must preserve every gate; an unconditional
security/availability claim requires actual evidence, not just that decision.

### Gate A — freeze declarations and price before their first use

- Freeze canonical spec, coordinate/provenance and use-record field encodings;
  distinguish declarative operation tags from native stage-operation digests.
  Use D20's acyclic order: immutable descriptors/spec → predeclared private
  Binding identities/intervals and provenance/ledger digests → region binding →
  native stage contexts → issued native owners. Private Binding consumers must
  not depend on the later region/stage digest that includes those Bindings.
  Verify actual returned owners against those predeclared identities; metadata
  alone never mints them. If this order cannot cover the new records, stop for
  design review rather than hash a cycle or drop a binding field.
- Publish pure owner-defined constructor/hash/resource-query quotes from the
  actual region, coordinate, product, role-mailbox, coordinator, frame/output
  and source-retention layouts. For each phase quote
  `retained owners + simultaneous scratch/Result/origin/destination slots`;
  use a maximum only for disjoint phases. Add original caller-owned/transferred
  capacities and transition overlap explicitly. No D27 size or unexplained
  workspace constant substitutes for this new census.
- Reserve that metadata/source overlap before its construction/hash/copy; grow
  the same aggregate positively before issuance and split the exact native
  slices without refund/reacquire. Prove `total <= 64 MiB` with exact-fit,
  one-byte-short and overflow tests at each initial/growth/split boundary.
  A short/new reservation cannot refund the existing input/source owners.

### Gate B — independent numerical, ownership and wire acceptance

1. Independent integer oracle for the exact two-table/floor equation, with
   independent literal floor proofs and complete frozen-table/hash comparison.
2. Zero, 255/256 and high-byte split boundaries; 65535/65536 clipping boundaries;
   maximal admitted input; exact product endpoint 2^24 and output 0/4096.
3. Negative controls for wrong source/table/profile/scale/domain, signed high-byte
   extraction, moved floors, predicate-as-arithmetic confusion, masked-value
   range admission, wrong provenance and substituted material.
4. Forced masks 0/1/half/max, both predicate masks, separate fresh lookup masks,
   affine/projection mask equations and complete reconstruction transcripts.
5. Normal native/public-API runtime checks, consuming-API compile-fail controls,
   both roles and lane endpoints 1/256, replay/failure burn and sibling retirement.
6. Exact-budget/one-byte-short preflight, zero-draw rejection, allocation/entropy
   fault sweeps, retained-owner drop orders and erasure/refund ordering. Preserve
   any original upstream guard without manufacturing an otherwise inadmissible
   input to test it. Charge actual new product frames and all live owners.

Additionally make the following controls executable, not merely descriptors:

- Root rejection for `-1` and `2^38`, before any operation nonce/mask/key/secret
  buffer; success at 0 and `2^38-1`. For inherited inputs reject absent/wider
  provenance, changed source/session/issuance/consumer, copied masked payloads,
  exchanged roles and a D27 output descriptor offered as an input capability.
- Check product contribution algebra for both roles modulo `2^50`, mask and
  split endpoints, fresh independent draws across all four ring50 samples
  (`rP,A0,B0,C0`) per lane/issuance, and separate region/product nonces. RNG byte
  requests and secret physical material must match the chosen owner quote; a
  distribution test is not a proof of uniformity or independence.
- Check all 18 ledger uses, both coordinate/mask relations and retained parent
  backing; no early C/X40 retirement or independent-root reconstruction. Assert
  both LUT outputs' original owners survive their consuming product use, and
  product/GapARS carry the derived `[0,2^24+1)` / `[0,4097)` intervals.
- Record all eleven exchanges and exact `1688+208L` directed bytes at L=1/256;
  assert source-versus-composition adaptations separately, not source-wire parity.
  Reject every product-header field substitution, swapped ordered inputs,
  noncanonical ring/bit residues, replay, repeated/reordered stages, truncation
  and trailing bytes before output mutation. No hidden LUT/product opening or
  final wrapper exchange is allowed.
- Public role tests must execute the whole new region, require both Ready before
  finish, and burn **both** siblings on the first error, cancellation, unfinished
  drop or callback/native unwind, including after one output is locally ready
  or has escaped. Test callback inspect/cancel reentrancy with bounded waits;
  retained frames keep original leases and any inherited singleton after abort
  (clear roots create no matrix singleton). Isolate
  coordinator-only, frame-only and output-only source retention; exclude added
  role charges when checking survival of the original source charge. Private
  tests own pointer/erase/entropy/no-refund-gap observations unavailable through
  public counters. Compile-fail clone/reuse/raw extraction/deferred-as-common
  controls remain required; existing D22/D27 and SDK-negative tests stay unchanged.

### Gate C — separate privacy disposition

Record an independent disposition of the joint-view questions above under an
explicit adversary/dealer/leakage model. Unresolved privacy blocks security or
protected-deployment claims even if every Gate B test passes. Neither D27's
local role isolation nor successful masked arithmetic settles this gate.

## Remaining approval / activation gates — not met

- Explicit approval of Select16 versus CPU Select50, two-exchange PLLM TR,
  reconstructed final GapARS, private unsigned-coordinate adapter, fixed uses
  and the proposed product/wrapper domains and byte contract.
- Before activation: exact owner-defined metadata/hash/native-transition quote
  formulas and a complete simultaneous-live owner inventory, before construction
  or hashing.
- Before activation: frozen canonical field encodings/order for new immutable
  spec and private coordinate-provenance records, with no runtime/spec digest
  cycle; the proposed product frame's both-input/header encoding does not
  complete those records.
- Review of original-owner inherited-range admission; clear roots must reject
  every out-of-domain code before operation entropy/secret allocation. No D22
  or D27 extension to attention/nExp is assumed. The role-split, original-source
  retention and enforced sibling-abort mechanisms themselves are already present
  for D22; only their new nExp-specific composition/admission remains unapproved.
- Reuse/extraction and execution of existing rational floor certificates against
  every selected literal; ordered mapping/source/table/mask provenance proof.
- Before security qualification: independent privacy review of both D19 lookup
  transcripts followed by freshly split product and reconstructed GapARS,
  without treating rho parity dependence as a shown attack or correctness as
  security evidence.
- After approval, implementation and the independent oracle, negative controls,
  normal runtime, lifetime/fault and exact-budget tests above. None run here.

No implementation, validation, standalone protected softmax, normalization,
trained-checkpoint quality, authenticated transport, decoder or SDK availability
follows from approval or from this document. No FuseFSS equations/execution,
new numeric profile, source shortcut or unauthorized substitution is permitted.
