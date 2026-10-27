# BumbleBee encrypted OLT: full-U decision packet

**Frozen scope: user62077, 26 October 2026; source-only design.** No crypto
implementation, dependency/backend adoption, compilation, RNG, key generation,
ciphertext execution, benchmark, model execution or numerical-profile adoption.
Only this document is changed. D53 selects design work on bounded unbiased
rejection; it is not implementation permission. D47's 32 MiB and `2^28` are
**clear-reference limits, not encrypted-region ceilings**.

Disposition: the dataflow and implementation contracts below are selectable for
review. **An executable encrypted implementation is NOT READY:** original-engine
birth/erasure, exact switch-down/error, transform resources, circuit privacy and
whole-region admission remain UNKNOWN. Unknown fit refuses before construction.
Calling a path “faithful” cannot discharge any of these obligations.

## 1. Evidence and ownership boundary

Local source baseline: `2702f9810af29a2647bce2136aee063242087f6a`; uncommitted
work is not represented by that commit. No claim that this source pass ran code
or reauthenticated a complete upstream distribution.

Consumed Astra material: the four existing text notes under
`target/bumblebee-pinned-phase-astra/`: `observations.md`,
`two-native-leaf-contracts.md`, `sampling-definitions-followup.md`, and
`rand-0.10.2-uniform-law.md`. These are recoverable local inputs to the requested
`ses_e8aacf8c4ffeis63h2Yu3uZrMf` packet, not new provenance certificates. The
session transcript itself was not available through this session's tools; no
unseen transcript assertion is treated as evidence. Durable source locators below
carry the findings; ignored target notes are not runtime dependencies.

COMMON secure HE owner `ses_e8aacf88fffedgLItBTRxeOmmi` owns original engine
birth, entropy/expansion, constructor-copy erasure and descendant freeze. This
packet neither duplicates that complete audit nor claims its concurrent result.
The existing [secure sampling design](he-secure-sampling-design.md) is an interface
input, not proof that a BFV-capable COMMON engine exists. Its CKKS/Poulpy secret,
torus mask and noise laws do not transfer to this OLT.

Final-paper identity inherited from the existing design: NDSS 2025,
`papers/r22-bumblebee-final.pdf`, SHA-256
`184bbe5104f183d4df66db2413f89c96c6c814e0b0ec6297f19745a1fa6e203b`.
App.E's OLT sizes and D21's recovered original OLT/packing equations are recorded
in [D37 B1/B2](bumblebee-executable-profile-d37.md#b1-source-size-ring-and-explicit-ordered-tuples)
and [indexing corrections](bumblebee-indexing-corrections.md).
This text-only pass did not reopen the PDF or invent new paper quotations.
Exact primes and sampling choices below are proposals, not author configuration.

Pinned specification input throughout:
`tlepoint/fhe.rs@98a087f1bf5f7fa95196ebe16b0e1b0eeb632442` (abbreviated F).
F is **not** an upstream runtime dependency to adopt, vendor or execute.

## 2. Contract selected for this proposal, not user-adopted law

Keep `N=8192`, `t=2^64`, shape `(3,65,129)`, window `(1,64,128)`, and **ordered**
data limbs `[562949952847873,562949952798721,562949952700417]`.
Their product is `q=178405960980747204108026560376635421240770561`.
Canonical coefficients are in `[0,t)` or `[0,q)`, never implicitly centered.
Packing negacyclic signs are applied in ring-t **before U**.

The proposed encoding is coefficientwise

```text
U(m) = floor(q*m/t), 0 <= m < t
Down(x) = floor((t*(x mod q) + floor(q/2))/q) mod t.
```

Odd q rules out half ties. U is neither `floor(q/t)*m` nor nearest-Up.
**For this proposal only**, full-U equations supersede D37 B3's OLT Delta
attachment, including its `-r_q*K` scaling-carry obligation as an account of
this path. Preserve that legacy oracle/counterexample and D37 history unchanged.
No other D37 arithmetic, SIMD bOLE law or complete profile is adopted here.

Recommended source-law branch for a later decision: F's CBD small-polynomial law
for **both secret and public-encryption ephemeral**, with explicitly proposed
variance `v=10`; preserve canonical full-width ct×plaintext and RNS/Garner
switching, then implement exact client CRT/Down instead of F's approximate
decoder. `v=10` is a proposed concrete choice here, not a recovered paper choice.
COMMON secure expansion may change transcripts; it must not silently change
these integer laws. No ChaCha8 exception, compressed seed replay, or obsolete
preserve-stream draft is selected. No secure expander identity is selected by
this document in place of COMMON's pending original-birth freeze.

### Source contrasts that cannot be hidden

| Boundary | Pinned source and short quote | Disposition |
| --- | --- | --- |
| Whole encoding | [F plaintext.rs L145–169](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe/src/bfv/plaintext.rs#L145-L169), [parameters.rs L688–728](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe/src/bfv/parameters.rs#L688-L728) | `v=(q mod t)*m mod t`, multiplied by `(-t)^-1 mod q`, yields U exactly; not Delta. |
| Actual ct×pt | [F ops/mod.rs L130–143](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe/src/bfv/ops/mod.rs#L130-L143), [plaintext_chunks.rs L42–115](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe/src/bfv/plaintext_chunks.rs#L42-L115) | Every component multiplies the canonical plaintext polynomial; no U on this multiplier, ct×ct scaler, centering, Q18 truncation or reconstructed-small-value shortcut. |
| Secret/ephemeral | [F secret_key.rs L47–59](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe/src/bfv/keys/secret_key.rs#L47-L59); [public_key.rs L69–79](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe/src/bfv/keys/public_key.rs#L69-L79): `Poly::<Ntt>::small` for u/e1/e2 | D37's iid ternary secret/u is different. For v=10, CBD support is [-20,20], not {-1,0,1}. |
| CBD definition | [F fhe-util/src/lib.rs L37–83](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe-util/src/lib.rs#L37-L83) | `X=Binomial(2v,1/2)-Binomial(2v,1/2)`, variance v; one signed sample projected consistently into every RNS limb. Not independently sampled small residues. |
| Galois operation | [F galois_key.rs L80–110](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe/src/bfv/keys/galois_key.rs#L80-L110): `key_switch_ntt_assign`, then `switch_down_to` | Need substitution, actual switch and its context transition; clear automorphism alone is not this operation. |
| Key rows | [F key_switching_key.rs L124–218](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe/src/bfv/keys/key_switching_key.rs#L124-L218): `ctx_ciphertext.moduli().len()`, `rns.get_garner(i)` | Three RNS/Garner rows for three data limbs, not D37's eight radix-2^20 rows. Its gadget equation, row counts and switch-error bound cannot transfer. |
| Decoder | [F rns/scaler.rs L54–59](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe-math/src/rns/scaler.rs#L54-L59): “off-by-one errors” | Approximate fixed-point RnsScaler is not exact global Down. Test agreement away from boundaries would not prove equivalence. |

The source-law proposal retains D37's candidate special prime
`p=562949952274433`, with ordered key context `[data limbs,p]` and no automatic
extra data level. Four-limb key-context feasibility, exact Garner weights and
special-prime drop/error still require proof against actual source. This is a
specific proposed target, **not** a claim that F implements D37 B2.1. If proof
fails, refuse this target; do not substitute eight radix rows or another modulus.

## 3. Original OLT → retained shares, both cross directions

Let `A=A0+A1`, `B=B0+B1` in ring-t. Retain distinct originals A0,A1,B0,B1;
never open either operand. Cross directions are ordered `(0,1)` then `(1,0)`.
In direction `(i,j)`, Pi owns original secret si and encrypted lhs Ai; Pj owns
the canonical plaintext multiplier Bj and performs evaluation/masking.

1. **Pack original shares.** D21 lhs stride `nw*mw`; rhs block is
   `V[beta,gamma]`; negative exponents use ring-t negation. Three row blocks,
   two inner blocks, two column blocks: six lhs encryptions per direction,
   four rhs plaintext blocks, twelve ct×pt products. Reuse each original lhs
   encryption across its two column products; no hidden re-encryption.
2. **Encode/encrypt actual lhs.** Pi encrypts U of the complete packed lhs
   coefficients under its original public key. Phase convention `(c0,c1)` is
   `c0+c1*si=U(lhs)+Eenc mod q`. Do not encrypt/re-encode the dense answer.
3. **Accumulate inner products first.** For each of six output blocks, sum both
   actual ct×pt products. All canonical full-width share coefficients multiply
   both components. No per-inner-block Down, rounding or Q18 rescale.
4. **Trace/interleave whole ciphertexts.** Pad six inputs to eight using
   structural zero (no fresh encryption), stride 64. Match existing D21 schedule:
   three initial stages per padded input, then 4+2+1 merges: **31 Galois calls
   per direction**, 62 total. Key exponents `[257,513,1025,2049,4097,8193]`
   under each original secret; twelve logical keys, not a shared secret.
   Apply each automorphism to actual components and switch back to si using
   the selected Garner path. Preserve the whole phase and actual drop error.
5. **Fresh masked H2A.** Pj samples an independent uniform full R_q polynomial
   R_ij (all 8192 coefficients, including padding), and independently produces
   a real public-key encryption of zero under Pi's original key. Send
   `ct_return=ct_trace+Enc_pki(0)+(R_ij,0)`. Trivial zero is not fresh zero.
   No masks, ephemeral/error streams or fresh-zero ciphertext are reused across
   directions. Fresh-zero alone does not prove circuit privacy.
6. **Original-client phase and exact Down.** Pi consumes a same-owner masked
   ciphertext loan, forms exact `c0+c1*si`, inverse transforms to coefficient
   RNS, reconstructs each coefficient globally, then applies exact Down.
   Stop before F's scaler. Pi retains `D_ij=Down(phase_return)`; Pj separately
   retains `S_ij=-Down(R_ij) mod t`. Only then ParseMat/crop each party's own
   result; never reconstruct and resplit the cross term.
7. **Separate final parties.** P0 retains
   `C0=A0*B0+Parse(D_01)+Parse(S_10)`; P1 retains
   `C1=A1*B1+Parse(S_01)+Parse(D_10)`, all mod t. Local products are exact.
   No coordinator holds a production capability for both shares. Clear research
   tests may compare them but cannot mint private `Share` values.

### Full-U error obligation

Write `U(x)=q*x/t+epsilon(x)`, `-1<epsilon<=0`. For integer negacyclic
`x*y=M+t*K`, the `q*K` term vanishes modulo q. There is **no Delta `-r_q*K`
term** in this path. Keep `epsilon(x)*y`, amplified encryption error, actual
trace/switch errors and fresh-zero error; do not replace them by sampled noise.

The existing clear result bounds its zero-crypto-error phase residual by
`|E_U|<2*N*(t-1)`, with scaled contribution below 1/16. Masked exact Down then
has circular error at most one per cross direction, two after the two cross
terms. This is the **clear theorem only**. Encrypted acceptance requires a new
bound on the selected whole trace/merge map and correlated key-row errors:

```text
P_ij = T(sum_beta ((U(A_i,beta)+Eenc_i,beta)*B_j,beta))
       + E_actual_switches + E_fresh_zero   (mod q)
D_ij + S_ij = Down(P_ij+R_ij)-Down(R_ij)   (mod t).
```

T includes inverse-64 normalization, all padded stages and merge signs. Prove
its exact selector identity for U's residual and its effect on arbitrary
encryption/switch errors; do not divide an error bound by 64 by assertion.
Reused encryption and evaluation-key rows correlate errors. With CBD secrets,
D37's ternary `||s||_1<=8192` is invalid; the crude support bound is 20N at v=10.
Exact special-prime rounding, its correlation with s, final correctness tolerance,
security level and circuit-privacy argument are **UNKNOWN**. Exact Down removes
decoder uncertainty only. No ±1 encrypted theorem or malicious-peer authenticity
claim follows from clear arithmetic or an original local key receipt.

## 4. Reuse decisions and concrete Sol owners

All filenames below exist unless explicitly marked **new/proposed**. Do not
change `lib.rs`, COMMON owners or another paper's files in this design tranche.

| Existing owner | Exact reuse / missing proof |
| --- | --- |
| `crates/pllm-core/src/coefficient_phase_reference.rs` | Constants, full-U canonical packing, two directions, actual clear products, complete-inner accumulation, separate Down outputs and party-local assembly. It requires `CryptoError::ExplicitZero`; **do not relax that to admit encrypted phases** or call clear `phases()` encrypted evidence. |
| `crates/pllm-core/src/coefficient_phase_reference/wide.rs` | Reuse `Rns3::{reconstruct,encode,down}` as clear oracle for three ordered limbs. Four-word bounds cover data CRT/Down. File explicitly says “non-constant-time clear”; it is not qualified for secrets, masked private phase, four-limb key switching or constant-time decryption. Extract reviewed method-neutral exact arithmetic later rather than creating a second paper CRT; secret-side timing/erasure review is required. |
| `crates/pllm-core/src/coefficient_phase_reference/work.rs` | Reuse named, fixed-width expanded clear-operation accounting. Extend a shared inventory with separately priced NTT, Garner, sampler/expander, codec and erase costs; existing Q does not include them. |
| `crates/pllm-core/src/coefficient_layout/trace.rs`, `coefficient_phase_reference::coordinates` re-export | Reuse D21 public addresses, signs, stride, padded schedule and clear trace oracle. Encrypted callback/operation lowering must preserve this sequence. No production degree-cap relaxation, private-key operation or ciphertext factory exists here. |
| `crates/pllm-garble/src/coefficient_phase_reference.rs` | Reuse `Supplied` original-vector capacity checks and `ClearRegion`/original-reservation pattern; its caller-supplied clear masks have no freshness. Do not return encrypted/private data through its public clear getters. |
| `crates/pllm-garble/src/shared_allocation.rs` | `OwnedBudget`, `OwnedExternalCharge::try_build`, `OwnedCharged<T>` and original split/transfer semantics. Payload field precedes charge; charge release follows payload destruction. This proves ordering, **not zeroization**, whole allocator RSS or original secret birth. Add secret payload wipe inside the actual owner. |
| `crates/pllm-compiler/src/encrypted_level_schedule.rs` | Reuse `PublicCiphertext{Owner,State,Capability,Graph,TraceStep}` backing liveness, unknown-cost refusal and immutable SSA validation. Existing `PublicCiphertextExecution` has CKKS/hybrid/view only: needs a separately reviewed method-neutral exact-ring execution contract before BFV lowering. Do not fake CKKS depth, hybrid conversion or treat a digest/state label as key provenance. |

Future minimal ownership split: shared exact-ring arithmetic/operation contracts
in core/compiler; original private engine/source in COMMON; paper-specific OLT
lowering adjacent to coefficient layout; original-counter adapter in garble.
Proposed new files, only after separate implementation permission:
`crates/pllm-core/src/coefficient_encrypted_olt.rs`,
`crates/pllm-core/src/coefficient_encrypted_olt/tests.rs`,
`crates/pllm-garble/src/coefficient_encrypted_olt.rs`,
`crates/pllm-garble/src/coefficient_encrypted_olt/tests.rs`, and
`crates/pllm-compiler/tests/coefficient_encrypted_olt.rs`.
These consume COMMON interfaces; they must not become a second HE engine or RNG.
Module registration is later work, not authorization to edit held `lib.rs` now.

## 5. COMMON interface required, not presumed implemented

The interface names here are requirements, **not callable current APIs**:

* An original non-Clone client/key-epoch owner, issuing purpose/role/parameter/
  child-ordinal-bound secret, public-key and evaluation-key descendants at birth.
  Actual original allocation counters and immutable key identity travel with
  every loan. Imported labels, equal bytes or a supplied digest cannot mint it.
* Fallible, bounded uniform/CBD draws conditional on future approval of COMMON's
  secure expansion; exact engine remains pending. Require exact CBD projection
  and unbiased uniform residues. Typed entropy/attempt/
  byte/work exhaustion; no unwrap, modulo fallback, zero substitution or reseed.
  Original and every descendant constructor must use the same failure path.
* Exact-ring transforms, ct×pt, substitution/Garner/drop, fresh-zero/mask and
  original-client masked-phase consumption. Quote actual original capacities,
  transform preparation, scratch and constructor overlaps before factories.
* `MaskedCipherLoan -> opaque MaskedPhase -> consumed exact Down output` bound
  to the original client borrow. Reject wrong component count (must be two),
  degree, ordered limbs, level, canonical representation, key/role/epoch,
  foreign counter, missing fresh-zero/mask lineage or spent ticket **before
  copy/NTT**. No phase getter, secret export, serialization or clone. Returning
  only exact Down residues does not itself establish secret-independent timing.
* Output policy keeps party shares separate; keys and temporary masked phases
  retire under original ownership. No callback while an owner lock is held.

UNKNOWN until COMMON's frozen interface and evidence are supplied: expander/
dependency identity, finite child schedule and root-to-child domain encoding,
every seed/state/constructor copy's erasure, concrete transform allocation/work,
exact switch/drop implementation, secure original-generation capability. Missing
fields are hard admission blockers, never zero-cost placeholders.

## 6. Finite closure inventory: multiplicities, draws, frames, erasure

This inventory freezes **logical multiplicities**, not an invented native quote.
First proposal is serial across directions, one request/epoch, uncompressed
original keys only; no bOLE keys, ct×ct/relinearization, bootstrap, compressed
imports or retries. Reordering/parallelization needs a new peak calculation.
Let `P_L=8*N*L`; `P3=196608`, `P4=262144` bytes. Payloads are lower-level facts,
not native capacities, Shoup/preparation sizes or total admission.

| Owner/stage | Finite inventory | Required lifetime / erase accounting |
| --- | --- | --- |
| Original inputs | Two lhs of 195 ring64 words, two rhs of 8385 words: 137280 bytes | Original backing/capacity retained through both directions and local products; charge views' full parent. Wipe private copies before release. |
| Context/setup | Ordered 3-limb data and proposed 4-limb key contexts for two owners; exact transform/twiddle/prepared forms | Actual sharing/dedup and capacities UNKNOWN. Include primality/context checks, tables, metadata, setup and destruction; don't count equal contexts twice unless actually copied or deduplicate merely by label. |
| Secrets | Two original N-coefficient CBD secrets, 131072 signed payload bytes | Original plus every signed/NTT/projected/automorphed secret copy; 12 Galois-source substitutions plus context-up transitions. Wipe originals last, after all loans; transformed copies on every exit. |
| Public keys | Two size-two key-context keys: 4P4 raw bytes; retained data restrictions: 4P3 if both kept | Symmetric key generation: two uniform key polynomials, two CBD errors. Include seed, projected s, a*s and output/result overlap. Raw/data copies coexist while charged. |
| Galois keys | 2 owners × 6 exponents × 3 Garner rows × 2 polynomials = 72P4 = 18874368 raw bytes | 36 uniform polynomials + 36 CBD errors; NTT-Shoup auxiliary payload/preparation is additional, as are c1 clones, zero replacement, a*s power-basis conversion and weighted source copies. Immutable reused rows remain charged; don't resample errors per call. |
| Encrypted lhs | 12 actual public encryptions total, 6 per direction; each size two at data level | 36 CBD draws for u/e1/e2, no uniform-a draw for this PK path. Retain original ciphertext shared by two column products; account serialization and outstanding peer ownership. |
| Product/trace | 24 ct×pt products; 12 accumulated blocks; 8 padded slots per direction, 31 switches per direction; one packed result per direction | Price old/new ct, substituted c1, Garner decomposition/reconstruction, switch/drop, inverse64 and merge scratch/copies. Structural padding may still allocate. Actual peak depends on backend reuse, never assumed in-place. |
| Fresh H2A | Two fresh zero public encryptions; two independent R_q masks | Six CBD draws; two uniform 3-limb polynomials. Charge unmasked/zero/mask/return overlap, mask representation conversion and both parties' outputs. Retain mask until sender's exact Down; wipe all private descendants. |
| Client phase | One borrowed return ct, original secret loan; S_ntt, product/phase and coefficient phase, inverse transforms | A three-polynomial subset is 3P3, not complete scratch. Reconstruct whole coefficient then Down; 4-word temporaries/copies must be reviewed and wiped. Return loan and phase die before charge; secret borrow prevents early retirement. |
| Retained outputs | Four N-word cross shares before crop: 262144 bytes; four cropped cross matrices plus two final matrices: 2322 words = 18576 bytes | Keep D01/S01/D10/S10 distinct. Quote simultaneous full/cropped/local/final output transition; final original owners persist until consumer retirement. No joint production getter. |
| Control/constructor | Tickets, stream states, seed/hash state, private fixed arrays, Result/Option/Vec descriptors, inline owner padding, error/unwind slots | Every move/copy and partially initialized output enters quote and wipe review. Existing 64-KiB clear control envelope is not a crypto allowance. Allocator caches/prepared arenas remain charged until actually released. |
| Frames/transport | 2 public keys + 12 Galois-key frames + 12 input-ct frames + 2 return-ct frames; optional data-key restriction frames only if actually sent | First finite schedule: one frame in flight per direction, no retransmission or batching. Include framing/length checks and full sender-buffer + encode-buffer + receive-buffer + decoder/native-copy overlap; actual byte ceilings UNKNOWN until canonical uncompressed codec selected. No zero-copy assumption. Secret/seed/mask/plain phase never framed. |

The two directions use separate original secrets; the evaluating peer holds the
other's public/evaluation material. Cross-process copies each consume their own
original receiver allocation guard; a coordinating aggregate quote includes both
parties, all retained key frames and simultaneously live send/receive backings.
No “small evaluator” claim may exclude client setup or the sibling direction.

### Draw schedule and bounded failure

For v=10, one CBD polynomial consumes 5120 u64 words = 40960 bytes under ideal
word input, independent of RNS limb count; pool leftovers follow F's law. Logical
small draws total **82**: 2 secrets + 2 PK errors + 36 key-row errors + 36 input
encryption small polynomials + 6 fresh-zero small polynomials. CBD stream-byte
subtotal is 3358720, not an entropy-quality or entire-expander claim.

Uniform outputs: **38 four-limb polynomials** (2 PK + 36 key rows) and **2
three-limb masks**, or 1294336 scalar residues. Recommended finite law is the
locked rand0.10.2 scalar Lemire map: u128 product of uniform u64 and p, accept
low half `>= 2^64 mod p`, return high half; at most **128 attempts per scalar**.
Use COMMON's independent implementation, not an upstream runtime or global
`rand/unbiased` toggle. Conditional successful outputs are unbiased; failure burns.

Source: [rust-random/rand@1540ea327e8beaf0694ea64f6d9eb8eaadd47bd5,
uniform_int.rs L100–156](https://github.com/rust-random/rand/blob/1540ea327e8beaf0694ea64f6d9eb8eaadd47bd5/src/distr/uniform_int.rs#L100-L156).
This publication commit is the inherited rand0.10.2 source mapping, not an archive
integrity certificate. Stored scalar Uniform differs from sample_single and is
unbiased regardless of the latter's feature branch.

Uniform maximum: 165675008 u64 attempts, **1325400064 generated bytes** before
seed/expander/control costs. These bytes can be streamed; they are not retained
scratch or fresh OS entropy. Since all proposed limbs are below 2^49, ideal
per-scalar exhaustion is `<2^-1920`; union bound over this finite schedule is
`<1294336*2^-1920 <2^-1899`. This is an ideal-independent-word calculation,
not a security theorem for a finite-seeded expander or arbitrary adversarial seed.

F's historical seed graph separately has 2 PK seed fills, 12 Galois parent
fills, 36 row-child fills, per-row hash/ChaCha8 expansion, and small-encryption
draws from caller RNG; seed/state constructor copies are additional. No compressed
replay is allowed in this proposal. COMMON's new secure graph must give an exact
finite replacement schedule including all those logical descendants, parent/
child seed draws, stream block rounding, state buffers and all constructor copies.
That replacement total is **UNKNOWN**, not the byte subtotal above. Arbitrarily
chosen “one seed per polynomial” is not evidence of actual constructor behavior.

### Original counters, sibling burn, retirement

Before entropy or any payload factory, one original epoch owner reserves the
complete two-party closure: retained keys/contexts, actual peak capacities,
bounded generated bytes/attempts/work, frames and output-retirement overlap.
Checked child slices can transfer an existing reservation, never acquire a new
budget or refresh spent work. Validate identity/length/capacity first; consume
invocation tickets before draws and value-dependent work. Used work/draws never
refund; live-memory guards release only after payload erase/drop.

Any entropy, sampler, constructor, codec, transform, admission, cancellation or
unwind failure poisons the original epoch: burn outstanding sibling tickets,
invalidate pending outputs and stop both directions. No retry of the other
direction, regenerated child, adopted received key, retained partial result or
fresh dummy full-cap reservation can bypass that burn. Retire partially built
payloads first, then their child guards, finally the original epoch guard. Drain/
erase queued private buffers under the same original allocation authority.

An ordinary failure-state test must observe original-counter identity and event
order, not merely the final used-byte value. `OwnedCharged`'s field order is
reusable evidence of drop ordering; complete secret-copy erasure and epoch-wide
revocation still require COMMON implementation evidence.

## 7. Implementation decision and ordinary qualification plan

**Select contracts before code:** recommended branch is full-U, canonical ct×pt,
CBD(v=10) secret/u/errors, 3-row RNS/Garner with the proposed 4-limb key context,
fresh-zero plus all-coefficient masks, original-client exact global CRT/Down,
uncompressed keys and serial finite closure. All are proposal selections, not
additional user adoption. COMMON must first deliver the exact-ring/source/erase
interfaces above. Do not start crypto code while UNKNOWN gates remain.

Once those contracts and separate implementation permission are granted, Sol's
ordinary tests belong in the explicit new files in section 4, extending existing
core/garble shared-allocation and compiler trace tests only where method-neutral:

1. Tiny-degree independent integer oracle versus actual component arithmetic:
   canonical extremes/full shares, signed packing, partial blocks, actual U×pt;
   preserve the Delta counterexample and reject nearest-Up substitution.
2. Whole trace/merge sequence versus independently derived coefficient selector;
   all six exponents, padded stages, Garner rows and special-prime drop. Independently
   calculate error; no expected phase derived by packing the desired dense result.
3. Original masked-phase consumes exact c0+c1*s, CRT/Down boundary/wrap cases and
   noncanonical/limb-order/level refusal; demonstrate why approximate-scaler equality
   is not an assumption. Clear wide oracle is not private constant-time evidence.
4. Both directions and **separate** D/S outputs and final C0/C1 versus independent
   dense ring64 oracle; no reconstruction/resplit implementation. Check proposed
   encrypted error bound only after actual proof exists; don't borrow clear ±1.
5. Finite byte-tape sampler mapping, rejection exhaustion, entropy failure and
   every constructor cut point; no production RNG/key/ciphertext work in such
   mechanism tests without separate permission. Check foreign/spent owner refusal,
   sibling burn, no reset/refund and payload-first erase/drop including unwind.
6. Ordinary compiler/resource tests: exact-ring capabilities, shared full backing,
   scratch/constructor/frame/output overlap, unknown-cost refusal and retained
   keys; crypto registration does not follow from passing a public graph validator.

Proposed reusable experiment **to be added later**, not an existing command:
`experiments/bumblebee_encrypted_olt.py`. Once implemented, contracts closed and
execution separately admitted, its ordinary invocation would be:

```sh
uv run python experiments/bumblebee_encrypted_olt.py
```

The script must consume the selected immutable contract, obtain a complete finite
prequote before factories, then run one original two-direction region, report
ordinary benchmark timing/bytes and per-direction numeric error, and retain no
secret material. Current invocation is intentionally **not runnable**: file and
crypto interfaces do not exist. No phantom current flags, custom verifier,
receipt archive, copied source tree or new replay/provenance gate is prescribed.
Use normal uv/Maturin build commands after permission, not a bespoke build driver.
The old nine clear tests are historical context, **not progress on this closure**.

## 8. Remaining choices and blocking evidence

| User choice (not silently adopted) | Recommended option | Independent evidence still needed |
| --- | --- | --- |
| Source CBD versus D37 ternary secret/u | CBD(v=10), explicitly source-law-derived, no paper-evaluated identity claim | Selected variance/security review, full noise propagation; ternary branch would require separate new numeric and sampling contract. |
| Garner versus eight radix-2^20 rows | Three Garner rows per Galois key, matching F's multi-limb structure | Exact weights, special-prime/drop rounding/error and context legality; cannot use D37 radix bound. |
| Exact versus approximate decoder | Original-client exact global CRT/Down | Secret-side constant-time/erase implementation and original-phase owner; no backend approximate-equivalence claim. |
| Secure engine/stream identity | Consume COMMON original-birth freeze, no ChaCha8 exception or compressed replay | Exact chosen implementation/dependency/domain graph, bounded descendant schedule and constructor-copy erase proof. D53 alone closes none of these. |
| Encrypted resource ceiling and qualification lane | Choose only after a complete instantiated native quote; unknown refuses | Exact capacities/NTT/Garner/frames/work/erase/seed totals and compiler costs; no reuse of 32 MiB/2^28 as crypto ceiling. |

The first three choices define an implementable *specification direction*; they
do not close proof or grant work. The fourth is owned by COMMON, not a second
BumbleBee sampler. The fifth cannot be replaced by a guessed 600-second/8-GiB
or other legacy budget. Until evidence closes, status remains **source-only,
not executed paper, not implementation-ready encrypted runtime**.

## 9. Bounded source pass / document freeze

One text-only source pass; no tensor/index/artifact fetch, repository download,
native job or executable arithmetic experiment. Two public exact-pin text source
fetches (key_switching_key.rs and rns/scaler.rs); remaining evidence came from
existing text files. Conservative ordinary count: at most 28 distinct text
inputs, with a 2 MiB read envelope including repeat slices (below 32 files /
16 MiB). A whole-file `wc -c` upper inventory for 22 local candidates totaled
497845 bytes; selected slices rather than every complete file were returned.
No continuing source sweep. The 128 KiB returned-output ceiling was not tracked
by a cumulative byte meter, so compliance with that output sublimit is **not
retrospectively certified**. This is an accounting limitation, not crypto evidence.
No receipt collection or independently authenticated source manifest was made.
Review completed within the single 60-minute tranche. Documentation whitespace
check returned no diagnostics; manual local link/path checks cover the three
linked existing design documents. No passing crypto test or execution is claimed.
