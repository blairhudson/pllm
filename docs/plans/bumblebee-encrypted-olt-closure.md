# BumbleBee encrypted OLT: full-U decision packet

**Frozen scope: user62077, 26 October 2026; source-only design.** No crypto
implementation, dependency/backend adoption, compilation, RNG, key generation,
ciphertext execution, benchmark, model execution or numerical-profile adoption.
Only this document is changed. D53 selects design work on bounded unbiased
rejection; it is not implementation permission. D47's 32 MiB and `2^28` are
**clear-reference limits, not encrypted-region ceilings**.

Disposition: the dataflow and implementation contracts below are proposals for
review, except section 16's D67 wider-family design selection and section 17's
explicit user selection of the **Untimed reuse** and private **Residue classes**
component designs. These design-only selections are not
runtime, parameter, engine or protocol-security approval and grant no implementation
permission. **An executable encrypted implementation is NOT READY:** original-engine
birth/erasure, parameter acceptance/construction and total error, transform resources, circuit privacy and
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

This documentation consolidation also records the completed independent Astra
review findings supplied to this pass: the conditional two-component privacy
lemma, canonical full-U/Delta distinction, arbitrary-error trace map and support
bounds, and unresolved security/source/ownership gates below. Completion of those
reviews is not approval of this protocol. Astra51's newly supplied pinned
source-text findings close the helper-body/equation gaps below, conditional on
compatible source-q/target-pq contexts; they do not instantiate this profile.
This integration performs no new source-body traversal. User70171 authorizes
retrieval and M32 policy only, not BFV approval or runtime adoption.

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

Precisely, CBD(v=10) is `X=sum_{h=1}^{20}(B_h-C_h)` for 40 independent
Bernoulli(1/2) bits: variance 10, support [-20,20]. One signed coefficient is
projected into every limb; independent per-limb small draws are a different law.

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
extra data level. Section 3 records conditional source-derived Garner weights,
drop/error equations and indexed q/pq placement. Parameter acceptance and actual
four-limb engine construction remain unvalidated. This is a
specific proposed target, **not** a claim that F implements D37 B2.1. If proof
fails, refuse this target; do not substitute eight radix rows or another modulus.
The four proposed 49-bit primes have `p_i=k_i*2^14+1`, with ordered
`k_i=[34359738333,34359738330,34359738324,34359738298]` (respectively
`2^35-[35,38,44,70]`). This establishes NTT congruence only: primality, roots,
context legality and intermediate arithmetic bounds remain open.

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
   inverse-64 first, then initial exponents 1025,513,257 per padded input and
   merges 2049,4097,8193 (4+2+1): **31 Galois calls
   per direction**, 62 total. Key exponents `[257,513,1025,2049,4097,8193]`
   under each original secret; twelve logical keys, not a shared secret.
   Apply each automorphism to actual components and switch back to si using
   the selected Garner path. Preserve the whole phase and actual drop error.
5. **Fresh masked H2A.** Pj samples an independent uniform full R_q polynomial
   R_ij (all 8192 coefficients, including padding), and independently produces
   a real public-key encryption of zero under Pi's original key. Send
   `ct_return=ct_trace+Enc_pki(0)+(R_ij,0)`. Trivial zero is not fresh zero.
   No masks, ephemeral/error streams or fresh-zero ciphertext are reused across
   directions. This is the reviewed candidate transcript, not an approved private
   protocol: fresh CBD zero encryption and phase-only flooding do not prove
   circuit privacy; see the conditional lemma below.
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

Use canonical unscaled `B in [0,t)^N`; no centered substitution. Write
`U(A)*B=q*A*B/t+epsilon(A)*B mod q`, `-1<epsilon<=0`. For integer negacyclic
`x*y=M+t*K`, the `q*K` term vanishes modulo q. There is **no Delta `-r_q*K`
term** in this path. Keep `epsilon(x)*y`, amplified encryption error, actual
trace/switch errors and fresh-zero error; do not replace them by sampled noise.

The Delta substitution fails even without crypto error. For scalar
`a=b=t-1`, `r=q mod t=0xfc73a11bdfe2c001`, `Delta=floor(q/t)`, its phase is
`Delta-r*(t-2)` rather than desired `q/t`. The scaled residual is
`-t*r*(t-2)/q-r/q`, with magnitude between `2^44` and `2^46`: a noiseless
counterexample to the circular ±1 claim, not a reason to center B.

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

The reviewed arbitrary-input selector is
`T(F)_{64*k+j}=F_j[64*k]` for `0<=j<6`, and zero for `6<=j<64`.
Inverse-64 precedes the trace: selected arbitrary encryption errors are **not
divided by 64**. The 31 switches include the padded lanes. For each original lhs,
`E_b=e_pk*u_b+e1_b+e2_b*s`, conditional on the **actual data-level e_pk**.
The bound `||E_b||_infinity<=800*N+20=6553620` presumes both PK error and
secret have CBD support 20; this is unverified for the effective level-switched
PK. Mapped `bfv-public-key.rs` 25–27 creates the PK at level-0 pq; 58–63 switches
it for level-1 q. Supplied Astra34 exact-pin review confirms direct rounded
component drop: `e_pk=e_eff=(e-r(b)-r(A)*s)/p` with compatible integer lifts
(§13), not an approximate RnsScaler path. The effective error is not iid CBD;
its joint rounding law with A, e and s remains unresolved. The same secret, PK error, original lhs
encryption and Galois rows are reused; independent-switch assumptions are invalid.
These support bounds and H below remain conditional theorems, not instantiated
engine proofs; no direct-q or literal-restriction profile is approved.

Writing each switch/drop phase error as xi_l, its later trace/merge propagation
as L_l, and fresh-zero phase error as zeta, the residual is

```text
T(sum_b ((epsilon_b+E_b)*B_b)) + sum_l L_l*xi_l + zeta.
sum_l ||L_l|| <= 469,
H = 2*N*(t-1)*(800*N+21) + 469*K + Z
```

Here `||xi_l||_infinity<=K`, `||zeta||_infinity<=Z` are required premises;
the conditional source-derived K below is not an instantiated-profile proof.
H is a sufficient support bound, not an actual joint-tail estimate. Encryption
support alone is about `2.05e5` plaintext codes, so this bound cannot certify ±1;
its looseness is **not a failure witness**. Actual joint tails remain necessary.
For canonical global Down with odd q there are no half ties; circular
`t*H/q<1` suffices for ±1 per direction. No encrypted ±1 theorem follows yet.

### Astra51 pinned Garner/drop equations — conditional, not instantiated

Source-text provenance uses F's exact pin above, not a new runtime dependency:

- `crates/fhe-math/src/rq/switcher.rs#L20-L30`, blob
  `6af4d88fa8a93cd8766f9338efa75d12cfe73f4a`: chooses destination/source
  ratio and delegates to Scaler.
- `crates/fhe-math/src/rq/rns_component.rs#L19-L100`, blob
  `334158c537c579af2787b4e2150073caf2fb04fc`: unsigned inverse-NTT lift and
  target embedding, reusing a matching row; rejects lazy input/degree mismatch.
- Existing mapped Scaler text
  `target/bfv-transform-source-inputs/crates--fhe-math--src--rns--scaler.rs`,
  lines 104–145, 225–256, 307–404, and `bfv-galois-key.rs`, lines 35–48,
  supply the scaling algebra and first substitution. These are supplied review
  inputs, not runtime dependencies or a new evidence archive.

The user-reported independent Astra review `ses_e7bf14243ffenmrtiColWiInF9`
conditionally clears the complete C1(q) → C0(pq) → one last-p-drop algebra and
source path, with no counterexample found. Switcher uses ratio p and delegates
degree/input-context checks; digits are unsigned canonical, and the caller
satisfies the lazy-output NttShoup requirement. This confirms the phase and K
equations below under their N=8192/CBD-support-20 premises, not instantiated-engine
acceptance. Builder BigUint evidence is mapped provenance; supplied blob IDs
were not independently hashed here. No new retrieval or rehashing is claimed.
Parameter construction/primality, encrypted ±1 correctness and joint circuit
privacy remain unqualified; the 469 trace multiplicity is outside that review.
No engine approval follows.

#### Indexed conditional profile (supplied 67 source review)

Existing exact-pin text inputs under `target/bfv-two-leaf-source-inputs` supply
the following definitions; upstream URLs identify source, not vendoring. This
follow-up performs no additional retrieval or source traversal.
Use **N=8192, not 4096**, and ordered
`M=[562949952847873,562949952798721,562949952700417,562949952274433]`:
q is the product of the first three limbs; p is **last**.

| Pinned definition | Conditional placement/consequence |
| --- | --- |
| `parameters.rs` 649–656; 689–691, 742–743, 765–769; 198–202 | Preserves supplied order, builds prefixes then reverses level order, with direct level indexing: `C0=pq` (first four), `C1=q` (first three), `C2` (first two), `C3` (first one). |
| `galois_key.rs` 29–45 | `ciphertext_level=1`, `galois_key_level=0` selects q → pq, ratio p. |
| `key_switching_key.rs` 114–121, 147–148; 194–201, 216–218 | Three decomposition rows stored over four limbs; weights use the first-three-limb Garner basis. |
| `rq/context.rs` 86–102 | Child removes the **last** limb p. Prepending p is a different, wrong profile: it removes q2 instead. |
| `galois_key.rs` 92, 101–109; `rq/mod.rs` 529–537 | Reachable child is one drop away; drop both switched components back to q, then add substituted c0; output level is 1. |

This closes the **conditional definition/placement** gap, not parameter
acceptance or construction. The four distinct 49-bit values are 1 mod 16384,
supporting N=8192's degree congruence; primality, pairwise coprimality,
plaintext compatibility and NTT-constructor acceptance are not verified.
Independent Astra21 review closes the speculative builder-width question:
mapped `bfv-parameters.rs` 437–439 has
`ParametersBuilder::plaintext_modulus(impl Into<BigUint>)`; 37–49 has
`PlaintextModulus::try_new` retain oversized values as BigUint with `small=None`.
Thus `t=2^64` is directly representable, not a new numeric adaptation or zero
sentinel. Machine-word `Modulus::new`'s `p<2^62` restriction does not apply to
this oversized-plaintext path. `ParameterConfiguration::build` 689–728 uses
BigUint inversion and `ScalingFactor::new(t,Q)`. However,
`PlaintextModulus::ntt_operator` 73–76 supplies **no plaintext NTT for t=2^64**;
no SIMD support or universal plaintext-API compatibility follows. Encoding
semantics, actual NTT/primality checks and parameter acceptance/construction
remain gates. No four-limb engine was constructed or approved.

Let `q_i` be the three ordered data limbs and
`g_i=(q/q_i)*inverse(q/q_i mod q_i)`. The digit polynomial `d_i=[c]_{q_i}`
has canonical, not centered, coefficients, bounded by
`D=(562949952847872,562949952798720,562949952700416)`.
**Provided source context q and target context pq**, Scaler has
`omega_i=p*g_i`, `gamma=p*q=0 mod pq` and zero fractional corrections:

```text
Switch(x) = p*sum_i [x]_{q_i}*g_i = p*x mod pq.
```

Canonical versus centered source lifts differ by q, hence by pq after scaling;
there is no upward-rounding/tie term here. A same-level ratio is 1, not p.
Galois key generation first substitutes `f=p*sigma(s)`: orientation is
substituted secret → original secret. In this **single-component key-switch
lemma**, `c` denotes the already automorphed ciphertext component `sigma(c1)`,
not the full ciphertext phase. Row phases and the accumulator obey

```text
b_i+a_i*s = p*g_i*sigma(s)+e_i,
sum_i d_i*p*g_i = p*c mod pq,
v0+v1*s = p*c*sigma(s)+E mod pq,  E=sum_i d_i*e_i.
```

All products are polynomial-ring products. For componentwise centered drop,
odd p gives `h=(p-1)/2`, with no half ties:

```text
r_j = ([v_j+h]_p)-h,  Dp(v_j)=(v_j-r_j)/p,
Dp(v0)+Dp(v1)*s = c*sigma(s)+xi mod q,
xi = (E-r0-r1*s)/p = E/p+rho0+rho1*s,  rho_j=-r_j/p,
K <= 20*N*sum_i D_i/p + (1+20*N)*h/p < 573441.
```

The numerator is coefficientwise divisible by p with **compatible lifts**;
`E/p` is a rational decomposition term, not independent modular division by p.
After dropping both components, the full Galois operation adds `sigma(c0)`:

```text
output phase = sigma(c0)+sigma(c1)*sigma(s)+xi
             = sigma(c0+c1*s)+xi mod q.
```

The K bound uses exactly N=8192 and the stated conditional parameters/CBD
support. The external trace-propagation bound gives at most `469*K`, with no
independence assumption; neither bound qualifies an instantiated engine.

The indexed definitions establish q/pq placement and the adjacent last-p drop
conditionally, but parameter acceptance/construction is not validated. That
remains a hard gate; the conditional bound is not an instantiated profile.
It does not close amplified encryption error, the support
bound's failure to certify `t*H/q<1` or ±1, correlated trace-error tails,
two-component privacy, entropy, resources or a secret-qualified engine. Exact
Down removes decoder uncertainty only, not these obligations or malicious-peer
authenticity.

### Conditional two-component privacy obligation

Let the decryptor's prior view V include its secret, key-generation coins,
public and evaluation keys, original ciphertexts and their coins, and prior
transcript. Fix evaluator input B and evaluated ciphertext `(F0,F1)`. For fresh
zero encryption `(Z0,Z1)` and ideal independent uniform full-ring R, the returned
ciphertext Y obeys

```text
Y = (F0+Z0+R, F1+Z1),
Law(Y | V,B) = Uniform(R_q) x Law(F1+Z1 | V,B).
```

This holds regardless of correlation between Z0 and Z1. Conditional statistical
distance between returned ciphertext laws for two B values equals exactly the
distance between their translated Z1 laws (for the fixed prior view). Uniform
first-component masking therefore does not discharge the second-component
obligation. `H=Y0+s*Y1` is uniform and independent of Y1 conditional on V,B;
conditioning on allowed `D=Down(H)` does not remove that obligation either.
Here H denotes the masked phase, not the support bound above.

The required B-independent simulator must cover the **joint** transcript within
the actual allocated transcript-distance budget, preserving correlations from
reused lhs ciphertexts and Galois rows. Fresh CBD zero encryption or flooding
only the phase does not prove this law. This is a missing privacy proof, **not a
practical attack assertion**. Any revised protocol covering both components needs
explicit approval and a new error, modulus and resource review; no such revision
is selected here.

## 4. Reuse decisions and concrete Sol owners

All filenames below exist unless explicitly marked **new/proposed**. Do not
change `lib.rs`, COMMON owners or another paper's files in this design tranche.

| Existing owner | Exact reuse / missing proof |
| --- | --- |
| `crates/pllm-core/src/coefficient_phase_reference.rs` | Constants, full-U canonical packing, two directions, actual clear products, complete-inner accumulation, separate Down outputs and party-local assembly. It requires `CryptoError::ExplicitZero`; **do not relax that to admit encrypted phases** or call clear `phases()` encrypted evidence. |
| `crates/pllm-core/src/coefficient_phase_reference/wide.rs` | Reuse `Rns3::{reconstruct,encode,down}` as clear oracle for three ordered limbs. Four-word bounds cover data CRT/Down. File explicitly says “non-constant-time clear”; no enforcing private custody or four-limb CRT/drop exists here. Section 17 selects reuse of this method-neutral exact arithmetic inside future authorized opaque custody, with timing and comprehensive transient-copy erasure excluded, owned buffers zeroized. No constant-time decryption or present secret qualification follows. |
| `crates/pllm-core/src/coefficient_phase_reference/work.rs` | Reuse named, fixed-width expanded clear-operation accounting. Extend a shared inventory with separately priced NTT, Garner, sampler/expander, codec and erase costs; existing Q does not include them. |
| `crates/pllm-core/src/coefficient_layout/trace.rs`, `coefficient_phase_reference::coordinates` re-export | Reuse D21 public addresses, signs, stride, padded schedule and clear trace oracle. Encrypted callback/operation lowering must preserve this sequence. No production degree-cap relaxation, private-key operation or ciphertext factory exists here. |
| `crates/pllm-garble/src/coefficient_phase_reference.rs` | Reuse `Supplied` original-vector capacity checks and `ClearRegion`/original-reservation pattern; its caller-supplied clear masks have no freshness. Do not return encrypted/private data through its public clear getters. |
| `crates/pllm-garble/src/shared_allocation.rs` | Actual seams are `OwnedBudget`, `split_native`, `OwnedExternalCharge::try_build` and `OwnedCharged<T>`; there is no `SharedAllocation` type. Payload field precedes charge; charge release follows payload destruction. This proves accounting/drop ordering, **not wiping**, whole allocator RSS, secret birth or allocation authority. `OwnedExternalCharge` neither owns nor erases the external payload; actual payload owners must wipe. |
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

Security remains conditional on the exact RLWE instance and related-secret/KDM
assumptions for Galois material. Neither 128-bit security nor a `2^-40`
correctness/privacy target is proved. COMMON must freeze ChaCha20 implementation,
version, features, domain separation, entropy source, finite descendant graph and
erasure behavior; this is a requirement, not an adopted engine. ChaCha8 and Poulpy
inheritance do not satisfy it. PRG distinguishing advantage must be allocated
separately from statistical statements proved under ideal independent bits.

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
| Client phase | One borrowed return ct, original secret loan; S_ntt, product/phase and coefficient phase, inverse transforms | A three-polynomial subset is 3P3, not complete scratch. Reconstruct whole coefficient then Down. Section 17's selected component must zeroize owned phase/scratch/output buffers; comprehensive erasure of transient four-word/stack/register copies is explicitly excluded. Return loan and phase die before charge; secret borrow prevents early retirement. Distinct COMMON key/constructor-copy obligations remain unchanged. |
| Retained outputs | Four N-word cross shares before crop: 262144 bytes; four cropped cross matrices plus two final matrices: 2322 words = 18576 bytes | Keep D01/S01/D10/S10 distinct. Quote simultaneous full/cropped/local/final output transition; final original owners persist until consumer retirement. No joint production getter. |
| Control/constructor | Tickets, stream states, seed/hash state, private fixed arrays, Result/Option/Vec descriptors, inline owner padding, error/unwind slots | Every move/copy and partially initialized output enters quote and wipe review; section 17 explicitly excludes comprehensive transient-copy erasure for its selected arithmetic component, not COMMON seed/state/constructor-copy requirements. Existing 64-KiB clear control envelope is not a crypto allowance. Allocator caches/prepared arenas remain charged until actually released. |
| Frames/transport | 2 public keys + 12 Galois-key frames + 12 input-ct frames + 2 return-ct frames; optional data-key restriction frames only if actually sent | First finite schedule: one frame in flight per direction, no retransmission or batching. Include framing/length checks and full sender-buffer + encode-buffer + receive-buffer + decoder/native-copy overlap; actual byte ceilings UNKNOWN until canonical uncompressed codec selected. No zero-copy assumption. Secret/seed/mask/plain phase never framed. |

The two directions use separate original secrets; the evaluating peer holds the
other's public/evaluation material. Cross-process copies each consume their own
original receiver allocation guard; a coordinating aggregate quote includes both
parties, all retained key frames and simultaneously live send/receive backings.
No “small evaluator” claim may exclude client setup or the sibling direction.
Both parties' original/transformed secrets, raw and prepared keys, transform/CRT
copies, errors, codec buffers, cancellation paths and output overlaps require
pre-factory quotes, wipe obligations and nonrefundable draw/work accounting on
their original ledgers. Accounting transfers do not transfer D47's clear budget
to crypto or authorize remote erasure; each remote owner must enforce its own
retirement and wipe contract.

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

**Later bounded exception: user71342 explicitly approves PUBLICRNSDROP source/test
authoring only.** The approval is identified by that user turn; no calendar date
was supplied for it. Shared `coefficient_layout::RoundedLimbDrop` projects one
retained odd limb q after removing one coprime odd limb p of equal degree, using
existing validated rings (`3<=m<2^61`, default Legacy4096, explicit Public8192).
For `h=(p-1)/2`, center the p residue to r and compute
`((vq-(r mod q))*inverse(p mod q)) mod q`; independently test against global
BigInt CRT followed by `floor((x+h)/p) mod q`. Reduce the center modulo q before
subtraction, including p>q. Construction/application are allocation-free source
contracts; inputs and lengths are fully validated before writes. Error atomicity
is per retained-limb projection only, not an implicit multi-projection transaction.
Reuse shared modular multiply/subtract and extract the existing bounded Euclid
from `coefficient_phase_reference/wide.rs` into the same shared modular owner,
preserving Rns3 error mapping/numerics. Existing Rns3 global CRT/Down, plaintext/
torus conversion and power-of-two repacking do not implement this drop. No fixed
paper-specific 3+1 API, unlimited basis, primality/root requirement or transform
dependency is added. Legacy consumer caps remain unchanged. This public untimed
algebra does not authorize secret issuance, HE/ciphertext integration, security or
parameter adoption, funding, runtime execution or cap increases. Existing dirty
NTT8192 source remains separate and unrun. New source/tests are **UNRUN**: no
build, test, formatting, syntax/import check, shell/native/model/network command,
staging or commit is authorized in this tranche. Execution remains held with
64 MiB free versus 24 GiB required; source is frozen for independent Astra review.

**Select contracts before code:** recommended branch is full-U, canonical ct×pt,
CBD(v=10) secret/u/errors, 3-row RNS/Garner with the proposed 4-limb key context,
the reviewed fresh-zero plus all-coefficient-mask candidate (privacy unclosed),
original-client exact global CRT/Down,
uncompressed keys and serial finite closure. All are proposal selections, not
additional user adoption. COMMON must first deliver the exact-ring/source/erase
interfaces above. Do not start crypto code while UNKNOWN gates remain.

No encrypted OLT approval exists. Deterministic public-buffer modular arithmetic,
NTT, ct×pt and monomial kernels are at most a **possible separate user-authorized
authoring scope**; this packet grants none. Such work would not qualify any
secret-bearing path. Existing CRT is non-constant-time, and the compiler contract
currently covers CKKS/hybrid/view, not exact-ring execution. Crypto implementation
remains blocked regardless of public-kernel progress.

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
    limb-order/level refusal; preserve clear/ciphertext noncanonical rejection while
    testing section 17's private all-u64 residue semantics. Demonstrate why
    approximate-scaler equality is not an assumption. Clear wide oracle is not
    private constant-time evidence; selected untimed component claims none.
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
| Garner versus eight radix-2^20 rows | Three Garner rows per Galois key, matching F's multi-limb structure | §3 supplies conditional weights/drop/error equations and indexed q/pq/last-p-drop definitions; BigUint directly represents t=2^64, but parameter acceptance/construction remains unvalidated. Cannot use D37 radix bound or claim instantiated total-error closure. |
| Exact versus approximate decoder | Original-client exact global CRT/Down; section 17 selects Untimed reuse and private Residue classes, design only | Genuine original-phase issuer/custody, owned-buffer zeroizing retirement and complete original-ledger admission remain missing. Timing and comprehensive transient-copy erasure are explicitly excluded for this component; other privacy/issuer/admission gates unchanged. No backend approximate-equivalence claim. |
| Secure engine/stream identity | Consume COMMON original-birth freeze, no ChaCha8 exception or compressed replay | Exact chosen implementation/dependency/domain graph, bounded descendant schedule and constructor-copy erase proof. D53 alone closes none of these. |
| Two-component circuit privacy | No approved replacement; candidate transcript remains unclosed | B-independent joint simulator within actual distance allocation, including translated Z1 laws and reused-lhs/Galois correlations; any protocol revision needs explicit approval plus new error/modulus/resource review. |
| Encrypted resource ceiling and qualification lane | Choose only after a complete instantiated native quote; unknown refuses | Exact capacities/NTT/Garner/frames/work/erase/seed totals and compiler costs; no reuse of 32 MiB/2^28 as crypto ceiling. |

The first three choices define only a *specification direction* (including section
17's explicitly selected component design, not an instantiated decoder); they
do not close proof or grant work. The secure engine is owned by COMMON, not a
second BumbleBee sampler. Circuit privacy remains a separate blocker. The
resource ceiling cannot be replaced by a guessed 600-second/8-GiB
or other legacy budget. Until evidence closes, status remains **source-only,
not executed paper, not implementation-ready encrypted runtime**.

## 9. Bounded source pass / document freeze

The following records the prior source pass, not new activity in this
consolidation. This consolidation changes only this document using supplied
independent review findings; no code, tests, builds, native execution, network,
source-body traversal, index, commit or new worktree. That historical consolidation
made no new decision or approval; section 17 separately records the later explicit
component-design selection and bounded local source inspection, not runtime adoption.

Prior pass: one text-only source pass; no tensor/index/artifact fetch, repository download,
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

## 10. Conditional computational masked-client-view lemma — review only

Supplied independent Astra findings **291 and 295**, with output refinement
**305**, support the following conditional lemma, not whole-circuit privacy,
numeric correctness, parameter security or engine approval. This supplements
§3's missing-proof obligation: fresh CBD zero encryption is not automatically
insufficient computationally, but neither does freshness alone prove privacy.
The earlier conditional statistical/support lower bound remains valid.

**Premises and claim.** Let `a` be uniform in the actual `R_q`, and assume
decisional RLWE for `(a,a*u+e)` with independent CBD(v=10) `u,e`, matching the
actual ring and modulus. A PPT, challenge-compatible sampler `G(a)` must produce
the full prior view `V`, evaluator input `B`, and evaluated components `F0,F1`,
independently of fresh `u,e`. Let `Z0` be the corresponding fresh-zero first
component and `R` an independent uniform full-ring mask. Then

```text
Y0 = F0+Z0+R,   Y1 = F1+a*u+e
(V,Y0,Y1) ≈c (V,U0,U1),   U0,U1 independent uniform in R_q.
```

Full-ring masking makes `Y0` uniform independently of the remaining view, even
when `Z0` and `Z1` correlate; RLWE replaces `a*u+e`, and translation by `F1`
preserves uniformity. This is a joint, averaged-view computational statement,
not a guarantee for every fixed `V`. Known secret `s` and previous client coins
may belong to `V` **only if `G(a)` efficiently samples their actual joint law**.
“All client coins” excludes the other role's fresh zero-encryption and mask
coins: revealing fresh `u,e` directly reveals `F1=Y1-a*u-e`.

**Source-distribution gate.** Supplied mapped F `bfv/public_key.rs` 69–79 has
`Z1=a*u+e2`; earlier public-key level handling at 58–63 may change effective
`a`. Mapped `bfv/keys/secret_key.rs` 125–148 instead retains a seed with
`Expand(sigma)=a`. Exposing that seed while embedding an arbitrary RLWE
challenge `a` is not samplable by the ordinary RLWE reduction. Application
therefore needs an explicit reviewed seeded/effective-a assumption or a proof
mapping the actual joint distribution. None is adopted here; these are supplied
source locators, not new source verification or permission to change source.

**Permitted-output refinement.** For known `s` and permitted client share
`L=Down(Y0+s*Y1)`, a candidate ideal simulator samples `P` uniformly from the
exact fiber `Down^-1(L)`, samples independent uniform `U1` in `R_q`, and returns
`(P-s*U1,U1)`. With `L` drawn from the ideal joint output law, this preserves
the joint averaged-view/output correlation; it is not a pointwise conditional
indistinguishability claim for every fixed `V,L`. The ideal must also match the
server's `-Down(R)` share (hence `Down(R)`) and later messages jointly. This
client-marginal simulator alone does not establish that protocol-level law.

Under the assumed coefficient map `Down(x)=floor(t*x/q+1/2) mod t`, canonical
`0<=x<q`, fibers are balanced (sizes `floor(q/t)` or `ceil(q/t)`). Exact uniform
fiber sampling can use weighted interval ranks, but requires an exact uniform
integer primitive or expected-polynomial fair-bit sampling. Bounded exact
fair-bit sampling is impossible for non-power-of-two fiber sizes. No such
implementation is approved, and the actual Down convention/source binding
still needs verification, rather than assuming this map is implemented.
For `r=q mod t`,

```text
TV(Down(U_q),U_t) = r*(t-r)/(q*t) <= t/(4*q),
TV for N coefficients <= N*t/(4*q).
```

The illustrative `q≈2^147`, `N=8192`, `t=2^64` gives about `2^-72`, not a
parameter certificate or permission to replace the exact joint output law.

**Scope and remaining gates.** Multiple fresh encryptions sharing `a` admit a
hybrid only with fresh independent coins and efficient continuation sampling;
the full view must retain reused-ciphertext and Galois-key correlations.
Advantages add across hybrids; this proves neither statistical `2^-40` privacy
nor concrete 128-bit security for the proposed parameters. Small-support
distance from uniform conditional on `V` can remain near one information-
theoretically without contradicting this computational lemma. This is a
masked-client marginal, not ordinary unmasked-ciphertext circuit privacy.
Flooding is not necessary for this conditional lemma; no engine clearance
follows. Exact OLT joint outputs, rounding carry `0,±1`, and encrypted ±1
correctness remain **OPEN**. No funding, pinned version, assumption adoption,
implementation authorization or approval changes. **STOP EDIT: independent
review required.**

## 11. Proposed public-A conditional tape lift — independent review only

Supplied independent **Astra10 + review7** support this conditional mathematical
review supplement to §10. It adopts no key-generation profile, entropy ABI,
sampling/storage change, cap or engine. This records review, not activation.

**Proposed source law.** For public A only, assume direct independent uniform
OS-returned 64-bit words; retain every returned word in the client view, excluding
OS internal state. There is no userland seed relationship for these words.
Calling a real OS API alone proves neither uniformity nor independence. Other
seeded ancestors still require honest conditional generation: no blanket erasure,
CSPRNG claim, or assertion that a seed ancestor is fixed without proof.

**Coordinate law and challenge lift.** For a legal integer limb `1 <= ell <= W`,
define `W=2^64`, `M=floor(W/ell)`, `T=M*ell`, `r=W-T`, `rho=r/W`.
Let `K >= 1` be an unselected finite integer cap (unrelated to §3's error K).
Stop at the first word `z<T`, output `z mod ell`; abort after K rejections.
Conditional on success, the residue is exactly uniform. Given challenge residue
`a in [0,ell)`, let L count preceding rejected words. Its conditional law is

```text
Pr[L=l | success] = (1-rho)*rho^l/(1-rho^K),  0 <= l < K.
rejected words iid Uniform([T,W)); J independent Uniform({0,...,M-1});
accepted word z = a + J*ell.
Pr[each compatible length-(l+1) tape | a,success]
    = ell / (W^(l+1)*(1-rho^K)).
Pr[abort] = rho^K; conditional abort tape = K iid Uniform([T,W)) words.
```

When `r=0`, success is certain and L=0; never invoke an empty rejected-interval
sampler or condition on impossible abort (including legal ell=1 or ell=W).
CRT requires legal pairwise-coprime limbs. Lift p independently; effective DATA
`a` must be literal `A mod q`, not rounded modulus switching.
This remains valid conditional mathematics but DOES NOT ATTACH to the mapped PK:
§13 confirms `a_eff=D_p(A)`, not literal restriction. Neither tape lift nor the
bounded whole-experiment variant below solves rounded-challenge embedding.

**Finite public-A event only.** Two PK A polynomials × four limbs × 8192
coordinates give `n=65536` tapes, 16384 per limb—not all protocol entropy.

```text
delta_K = 1 - product_ell (1-rho_ell^K)^16384
        <= 65536*rho_max^K.
ell_max=562949952847873 < 2^49 => delta_K < 2^(16-15*K).
```

Arithmetic checked against §2/§3's existing ordered values (no primality or
constructor proof); each has `M=32768`:

| ell | r = 2^64 - 32768*ell |
| --- | --- |
| 562949952847873 | 18790449152 |
| 562949952798721 | 20401061888 |
| 562949952700417 | 23622287360 |
| 562949952274433 (p) | 37580931072 |

Thus `r_max=37580931072 < 2^36`, giving `delta_K < 2^(16-28*K)`.

**Always-success conditional simulation and runtime distinction.** Sample L using capped independent
acceptance flags with success probability `1-rho`: STOP at first acceptance;
restart only after K failures. Expected trials `1/(1-rho^K)`; expected total
flags `1/(1-rho)`. Drawing all K flags instead costs `K/(1-rho^K)` flags.
These internal simulator restarts are not modeled client retries; simulator
private flags are not client coins. For these limbs J uses 15 fair bits.
For r>0, sample rejected-interval offsets by bit rejection with acceptance >1/2
(power-of-two intervals accept surely). Exact fair-bit simulation is expected
polynomial with an exponential runtime tail, not strict PPT or a CPU bound.
Strict PPT requires an explicit polynomial cutoff and a separately bounded
negligible truncation error; no such cutoff is selected here.

For global abort, generate honest preliminary success/failure flags and tapes,
embedding the challenge ONLY on success; alternatively sample first failure
directly. NEVER retry until the rare global-abort event occurs. Match partial
success, first failure, cleanup and all observed metadata. Provider failures and
their observables need separate treatment, not inclusion in ideal `rho^K` bounds.

**Joint-view gate.** Generate keys, secret, PK error, reused Galois rows and
original encryptions once, preserving every correlation. The actual prior view
must be jointly samplable from legitimate client inputs/public setup. Fixed B
as nonuniform advice is a reduction device only, not an ideal Simulator input.
Correlated B requires a legitimate efficient joint setup/input sampler independent
of fresh noise. Full uniform masking yields a MASKED CLIENT MARGINAL ONLY:
joint server share, ideal joint outputs, fiber law and encrypted ±1 remain OPEN.
This conditional lift does not resolve §10's other source-distribution gates.

**Unfunded storage/ownership proposal.** Two retained four-limb As occupy
`2*4*8192*8=524288` payload bytes. Worst returned public-A word tapes occupy
`65536*K*8=524288*K` bytes BEFORE headers, capacity slack and copies.
Conditional reduction flags are not runtime allocations. No K is chosen, so no
finite funded runtime quote exists. Existing whole-region original owners must
precharge actual capacities, preconstruction/transport overlap, work and cleanup;
provider failure/observables remain separately charged and modeled. Maverick's
32-draw policy does NOT select BumbleBee K; §6's earlier proposed sampler cap
does not select this alternative either. Approval is required before entropy,
sampling, storage, ABI or runtime changes. No engine approval.

### Optional stronger whole-experiment embedding — Astra9 review

Independent **Astra9** clears exact bounded public-A embedding for the WHOLE
experiment retaining honest cap-abort mass. Unlike the always-success conditional
sampler above, this public-A stage needs no simulator cutoff or truncation loss.
Construct preliminary honest capped word tapes until global first failure.
On abort, keep chronological partial setup, coins and cleanup honest; ignore
the challenge. Abort terminates/burns, with no external continuation.
On all-success, for each q-limb coordinate of ONE hybrid-target PK, preserve
its rejected prefix, L and `J=floor(accepted_word/ell)`, replacing the accepted
word by `J*ell+a_challenge`. Keep p's residue/tape independently honest and
the other PK independently honest: NEVER reuse one challenge for both PKs.
Conditional on global success, prefixes/L/J are independent of uniform accepted
residues; hence replacement preserves the EXACT joint A+tape+view law.
Generate other keys, noises and reused rows honestly around final A, preserving
all correlations; repair ALL copied words, metadata and derived consistency.
Discarded preliminary words are simulator PRIVATE, not visible client coins.

At most `n*K` uniform 64-bit word calls plus `O(n)` fixed-width replacement:
a strict bounded random-word stage, NOT a provider-latency or whole-engine bound.
Abort contributes zero hybrid gap; `full_gap=Pr(success)*conditional_gap`,
with no extra embedding abort. Global all-success means precisely this capped
sampling stage; further A-dependent validation/failure must be separately
modeled, not presumed independent. Legal coprime CRT, literal `A mod q`, ideal
OS-word assumptions and remaining ancestors' honest conditional generation remain
required. Other whole-protocol samplers still need their own PPT proofs.
This optional stronger mathematical contract selects no cap, profile or funding;
it changes no runtime, entropy ABI or engine approval.

## 12. Conditional legal-input variance diagnostic — Astra17 review

Independent **Astra17** identifies the effective-PK-error source-attachment gap
above. This diagnostic preserves numerical premises, not an instantiated proof,
new profile, implementation permission or engine approval. Source locators below
are supplied mapped review findings, not new source verification.

Take legal input `A=0`, B a 65×129 matrix of canonical `t-1` values.
`coordinates.rs` 100–107 gives

```text
B0 = (t-1)*sum_{r=0}^{8191} X^r,
B1 = (t-1)*sum_{c=0}^{127} X^(64*c),
W = [B0*E0+B1*E1]_0.
```

Supports have 8192 and 128 coefficients. Matrix entry (0,0) maps to coefficient
0 (111–117); clear reference 288–304 uses canonical multipliers. The trace selects
W, NOT W/64. For fixed shared actual data-level PK error e and secret s, assume
fresh independent `u_b,e1_b,e2_b` have CBD variance 10. Let C(f) be the integer
negacyclic multiplication matrix, and define row vectors

```text
a_b = row_0(C(B_b)*C(e)), d_b = row_0(C(B_b)*C(s)),
r_b = row_0(C(B_b)),                         b in {0,1}.
E[W | e,s] = 0,
Var(W | e,s) = 10*sum_b (||a_b||_2^2+||d_b||_2^2+||r_b||_2^2),
E[exp(theta*W) | e,s] = product_{b,i}
  cosh(theta*a_bi/2)^40 * cosh(theta*d_bi/2)^40 * cosh(theta*r_bi/2)^40.
```

Mapped `fhe-util-lib.rs` 50–71 uses two 20-bit popcounts: 40 disjoint fair bits,
support [-20,20], variance 10. `rq/mod.rs` 339–346 projects the SAME SIGNED draw
across limbs. If additionally e and s have iid CBD coefficients and are mutually
independent (and independent of fresh draws), THEN

```text
Cov(E_b,E_c) = indicator(b=c)*(200*N+10)*I,
Var(W) = (200*N+10)*8320*(t-1)^2,
SD(t*W/q) ≈ 0.223 plaintext codes.
```

Zero cross-covariance is NOT independence. This unconditional formula and SD
do NOT attach to the current level-switch path until the actual e law is proved.
Under the previous CONDITIONAL switch/fresh-zero bounds, write remaining error V:
`|V|<=Vmax=469*573441+6553620=275497449`, `delta=t*Vmax/q<2^-54`.
With R independent of the ENTIRE unmasked phase, the reviewed weaker bound is

```text
Pr[circular error >= 2] >= (1/2-1/q)
  * Pr[1.5+delta <= abs(t*W/q) <= 2.5-delta].
```

For exact Down use the integer lift `D(n)=floor((t*n+(q-1)/2)/q)`.
Write `t*k=L*q+j`, `0<=j<q`. Since `gcd(t,q)=1`, uniform R gives
`D(R+k)-D(R)=L` or `L+1` with probabilities `1-j/q` and `j/q`.
Reduction modulo t removes q-wrap. Condition on both W and V: their correlations
are harmless because R remains independent. A stronger 1/2 factor is valid,
but the displayed bound retains the reviewed weaker factor.

Neither variance nor a Gaussian heuristic proves failure. The historical proposed
route was to establish the effective e law, then prove an annulus probability;
§15 instead supplies a rigorous tail argument under an explicit ideal joint law.
The old covariance/SD above remain a conditional historical diagnostic. The global `2^-40`
correctness/privacy target remains unproved; no per-coefficient budget is approved.
A hypothetical `2^-40` single-coefficient benchmark is not an adopted allocation.
Literal restriction or direct-q PK generation would require an explicitly reviewed
adaptation, not an assumed source fix. No numeric run or runtime change follows.

## 13. Pinned direct-drop PK error attachment — Astra34 review

Supplied **Astra34** retrieved F's exact pin
`98a087f1bf5f7fa95196ebe16b0e1b0eeb632442`. Its reviewed source is
[`crates/fhe-math/src/rq/modulus_switch.rs` 20–87](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe-math/src/rq/modulus_switch.rs#L20-L87).
The parent retrieved and reviewed this exact pinned body, confirming lines 20–87;
this records source review, not execution.

The NTT direct-drop path inverse-transforms the last row (24–37), centers its
remainder with `h=(p-1)/2`, embeds/retransforms it (47–57), subtracts and multiplies
by inverse p (58–60), then removes the row (67–68). Adjacent contexts take one
call (74–77). This is EXACT rounded drop, not RnsScaler or an approximate
off-by-one path. Coefficientwise, with compatible integer negacyclic lifts:

```text
r(x) = ([x+h]_p)-h,                 h=(p-1)/2,
D_p(x) = (x-r(x))/p mod q.
b+A*s = e mod pq; choose lift b=e-A*s.
a_eff = D_p(A),                    b_eff = D_p(b),
b_eff+a_eff*s = e_eff mod q,
e_eff = (e-r(b)-r(A)*s)/p mod q.
```

The numerator is coefficientwise divisible by p; this is compatible-lift integer
division, not modular division by p. Mapped
[`secret_key.rs` 133–149](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe/src/bfv/keys/secret_key.rs#L133-L149)
supplies the original PK equation;
[`ciphertext.rs` 167–202](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe/src/bfv/ciphertext.rs#L167-L202)
switches BOTH components of the level-0 pq PK to q. Under original CBD support
20 and `N=8192`, a compatible effective-error lift satisfies

```text
||e_eff||_infinity <= [20+(1+20*N)*(p-1)/2]/p < 81921.
```

This support statement does NOT make e_eff iid CBD: rounding correlations jointly
involving A, e and s still require a law. For fixed actual e_eff,s and independent
fresh CBD draws, `E=e_eff*u+e1+e2*s` and §12's conditional zero mean/MGF remain
valid. Its unconditional covariance/0.223 estimate and the old E support, H and
V bounds remain conditional, NOT mapped-engine qualifications. No recomputed H,
tail proof, rounded-challenge embedding or encrypted ±1 closure is claimed.
Literal restriction/direct-q generation would need explicit reviewed adaptation;
none is adopted. Parent's next Astra joint-law review is separate mathematical
work, not implementation permission. No new profile, sampler ABI, cap, funding,
parameter construction, runtime change or engine approval follows.

## 14. Optional rounded-coordinate extension — Astra36/Astra39 review

This reviewed MATHEMATICAL extension is not an adopted setup, adaptation or source
ABI. §11's literal-restriction theorem still does NOT directly attach to the
mapped source; the construction here is distinct from its always-success
conditional fair-bit sampler, whose expected-time/cutoff caveat remains intact.

**Astra36 coordinate identity.** Let `Q=p*q`, `h=(p-1)/2`, with legal coprime
CRT limbs. For uniform `a in R_q` and independent uniform centered coefficient
vector `r in [-h,h]^N`, the following gives a bijection onto R_Q:

```text
A = p*a+r mod Q,       D_p(A)=a,       r(A)=r.
b=e-A*s mod Q;         r(b)=r(e-r*s),
e_eff = D_p(e-r*s)     (small integer lift).
```

Here `D_p(x)=(x-r(x))/p` on integer lifts; products are negacyclic.
If r,e,s are jointly drawn independently of a, then a is independent of
`(e_eff,s)`. Effective error may depend on s and is NOT thereby iid CBD.
This does not grant free choice of the b component of an unknown-secret RLWE
challenge PK. Challenge-compatible joint-view generation remains OPEN, including
Galois rows, reused ciphertexts, 31-switch schedule, full coins and known secret.

**Astra39 exact bounded tape construction.** Use §11's stronger WHOLE-experiment
stage, not repeated conditional sampling. Run the original honest capped-word
stage until first failure; on abort preserve chronological partial tapes and
observables unchanged, ignore the challenge, and terminate/burn as before.
On success keep the ENTIRE original p-limb tapes unchanged; set each centered
`r=center_p(p_accepted_residue)`. For ONE hybrid-target PK and independent uniform
challenge `a in R_q`, define `A=p*a+r mod Q`. For each q-limb coordinate replace

```text
accepted_word = J_ell*ell+z
             -> J_ell*ell+(A mod ell).
```

Keep quotient, rejected prefix, length and acceptance decisions unchanged.
Given the entire fixed p tape, r is fixed; `a -> (p*a+r) mod q` is a bijection
because `gcd(p,q)=1`. The resulting q residues are CRT-uniform independently of
the ENTIRE retained p tape. Thus the full A+tape joint law and honest abort mass
are EXACT, with `D_p(A)=a`. Keep the other PK independently honest: never reuse
one challenge for both PKs. Repair all public copies and descendant consistency;
overwritten preliminary values are simulator-private, not visible client coins.

At most `n*K` original uniform 64-bit word calls plus `O(n)` fixed-width CRT/update
work: no fresh uniform-r sampling, restarts, cutoff or truncation loss FOR THIS
STAGE ONLY. This is not a provider-latency/whole-engine bound; remaining protocol
samplers retain their PPT obligations. Actual seeded-PK ancestry and returned
coins must match jointly; ideal direct tapes are not presumed to be the actual
OS/seeded API. Further validation/failure and cleanup observables retain §11's
separate modeling obligations. No unconditional iid-noise, RLWE privacy, joint
share, encrypted ±1 or engine claim follows. No cap, profile, funding, setup,
source ABI or implementation approval is granted. The subsequently reviewed
conditional numeric-moment theorem is recorded in §15, not adopted as a profile.

## 15. Conditional rigorous per-direction failure theorem — reviewed model only

The supplied original moment proof was independently **CLEARED by reviewer41**
(`ses_e7b86a27fffeSQk9kViA9Gracx`). This is a conditional mathematical theorem,
NOT an adopted numeric law/target or a mapped-engine failure claim. It uses no
independent rounding-noise assumption, Gaussian approximation or numeric run.

**Explicit model.** Use §2's fixed q,t and `N=8192`, odd `p>=2^48`, exact drop
and Down, and an ideal uniform original `A_pk in R_pq`. Its centered remainder
r has iid uniform coefficients in `[-h,h]`, `h=(p-1)/2`. Original e and secret s
have independent CBD(v=10) coefficients, independently of r. For b=0,1, fresh
`u_b,e1_b,e2_b` are independent CBD(v=10), also independent of r,e,s. Retain
shared r,s across b; this is NOT proof of the actual seeded F source law.
Assume the conditional trace/switch bounds `K=573441`, `sum_l ||L_l||<=469`.

Take input A=0 and canonical rhs B, 65×129 entries all `t-1`, as in §12.
Input zero removes the encoding residual; distinguish A from public A_pk.
The retained first output coefficient is W, with NO division by 64. Set

```text
H0 = sum_{i=0}^{8191} X^i,       H1 = sum_{j=0}^{127} X^(64*j),
B_b = (t-1)*H_b,                E_b=e_eff*u_b+e1_b+e2_b*s,
W = [sum_b B_b*E_b]_0,          c=t*(t-1)/q.
e_eff = -(r*s)/p+g,             g=e/p+eta,
eta = -r_p(e-r*s)/p,            ||g||_infinity < a=1/2+20/2^48.
Y = t*W/q = ell+Q+F,
ell = -c*[sum_b H_b*(r*s*u_b)/p]_0,
Q   =  c*[sum_b H_b*g*u_b]_0,
F   =  c*[sum_b H_b*(e1_b+e2_b*s)]_0.
```

All products are integer negacyclic; r_p is centered remainder modulo p.
The scalar ell here is a moment term, not §11's limb. In particular g is a
function of r,e,s, NOT independent rounding noise. Write `||X||_k=(E|X|^k)^(1/k)`
for random scalars and C(H) for integer negacyclic multiplication matrices.

**Second moments, with correlations retained.** Direct expansion using the
independent groups r,s,(u0,u1) and `Var(r_i)=(p^2-1)/12` gives

```text
E[ell^2] = c^2*N^2*100*8320*(1-p^-2)/12 in (16.90,17.00),
4.11 < ||ell||_2 < 4.13,        1.907e-6 < c < 1.91e-6.
||C(H0)||_op = csc(pi/(2*N)) < .637*N,
||C(H1)||_op <= 128.
```

The support sizes sum to 8320. Conditional on r,e,s, Q is linear in fresh u;
the operator bounds and `||g||_2<=sqrt(N)*a` give a POINTWISE variance bound:

```text
E[Q^2 | r,e,s] <= c^2*10*N*a^2*((.637*N)^2+128^2) < 1.43^2,
E[F^2] = c^2*8320*(10+100*N) < .16^2.
||Y||_2 >= ||ell||_2-||Q||_2-||F||_2 > 4.11-1.43-.16 = 2.52.
```

Reverse Minkowski does not discard correlations among ell,Q,F.

**Fourth moments.** Every centered-uniform or CBD coordinate has fourth moment
at most three times variance squared. Iterating the multilinear fourth-moment
bound over independent groups r,s,(u0,u1) yields

```text
E[ell^4] <= 27*(E[ell^2])^2.
||Q||_4 <= 3^(1/4)*1.43.
```

The second bound uses Q's conditional linearity and the pointwise variance bound,
not independence of effective-error coefficients. Split F into its fresh-e1
linear part F1 and its fresh-e2/secret bilinear part F2. Their bounds give

```text
||F1||_2 < .001,   ||F2||_2 < .16,
||F||_4 <= 3^(1/4)*.001+sqrt(3)*.16 < .322,
||Y||_4 < (5/2)*4.13+(4/3)*1.43+.322 < 13.
```

Here `27^(1/4)<5/2` and `3^(1/4)<4/3`. Cauchy–Schwarz applied to Y² on its
tail, with `E[Y^2]>2.52^2` and `E[Y^4]<13^4`, now implies

```text
Pr(|Y|>2.51) >= (E[Y^2]-2.51^2)^2/E[Y^4]
             > 1/(400*13^4) > 2^-24.
```

These are analytic rational bounds; no Gaussian, Monte Carlo or annulus-probability
assumption is used. Shared r,s and rounding correlations remain in the model.

**From the tail to every-mask failure.** The integer effective-error lift obeys
`||e_eff||_infinity<=81920`. Encryption and fresh-zero error supports therefore
give Z below; this replaces the old support premise ONLY within this model:

```text
Z = 13,425,049,620 < 2^34,
|V| <= 469*K+Z,          delta=t*(469*K+Z)/q < 2^-47,
|Y| <= c*8320*Z < 2^30.
```

Thus on the tail event, `2.5<|x|<2^30+.01` for `x=t*(W+V)/q`, even when W,V
depend on each other. For the exact integer lift
`D(k)=floor((t*k+(q-1)/2)/q)`, for EVERY integer mask R the increment
`D(R+W+V)-D(R)` belongs to `{floor(x),ceil(x)}`, modulo t after q-wrap.
Both possibilities have circular absolute error at least 2 and less than t/2.
No averaging over R or independent-noise assumption is needed for this step.

Consequently per-direction ±1 fails with probability **greater than 2^-24**.
Any global contract requiring ALL per-direction coefficients within ±1 cannot
meet `2^-40` under THIS conditional model; no assigned per-coefficient budget is
needed. Neither that global target nor a numeric profile is adopted here.
The per-direction theorem alone is NOT a final reconstructed `C0+C1` failure
theorem: cancellation and a final clear ±2 contract may differ. The following
extension requires an additional conditional uniform-mask law.
The candidate under this proposed ideal joint law faces a concrete conditional
numeric counterexample, not a whole-BumbleBee-paper/backend failure finding.
Actual source/joint-view attachment, RLWE privacy, joint outputs and engine
qualification remain separate. No implementation, cap, funding or approval follows.

### Final-share extension — additional conditional uniform-mask premise

**A5 and independent Astra44** (`ses_e7b86a27fffeSQk9kViA9Gracx`) both clear
this mathematical extension. Take legal party shares `A^(0)=A^(1)=0`,
`B^(0)=0`, `B^(1)` all canonical `t-1` in the 65×129 shape. Ideal and local
products vanish. Party B^(0) is NOT packed block B_0: §12's B_0/B_1 both belong
to the packing of party B^(1). Direction 01 has the preceding W and bounds.
In direction 10, B^(0)=0 removes initial encryption/encoding phase, NOT switch
or fresh-zero errors. Retain `|V10|<=469*K+Z` and both keys' effective-error and
support premises. Write `x01=t*(W+V01)/q`, `x10=t*V10/q`; hence `|x10|<=delta`.

Additionally require each full-ring mask to be uniform CONDITIONAL on the ENTIRE
JOINT unmasked phases, including fresh-zero and switch errors, not just marginally
uniform. Conditional mutual mask independence is sufficient, not necessary.
Given all phases, for integer phase k write `t*k=L*q+j`, `0<=j<q`.
The rank `t*R mod q` is uniform because `gcd(t,q)=1`; exact Down obeys

```text
D(R+k)-D(R) mod t = L      with probability 1-j/q,
                   L+1    with probability j/q.
```

There is NO 1/q approximation error. Negative k and wraps are covered by
`D(x+q)=D(x)+t`. On the previous tail, `2.5<|x01|<2^30+.01`, so conditional
`Pr(circular |E01|>=3)>1/2`. Integrating over the tail gives
`Pr(circular |E01|>=3)>2^-25`. For direction 10 the same exact increment law
gives `Pr(E10!=0)<=delta`. No global independence between directions is required.
The documented §3 assembly has final error `E01+E10 mod t`; a union bound yields

```text
Pr(circular |C0+C1|>=3) > 2^-25-delta > 2^-26.
```

On E10=0 cancellation is impossible; relevant direction-01 increments are below
t/2, so integer and circular magnitudes agree. The ideal product here is zero.
Thus THIS conditional model rules out a `2^-40` failure guarantee for FINAL ±2,
IF that numeric contract is required. §3's ±1/±2 result is a clear theorem only,
NOT an adopted encrypted target. No per-coefficient budget is assigned or changed.
This is not a mapped-engine failure/security finding, profile adoption, new mask
ABI, funding or execution approval. No runtime, key, sampling or numeric change
is authorized; parameter-repair planning remains separate and unapproved.

### Nearest-full-U encoding-only candidate — conditional zero-input disposition

**DESIGN ONLY; unadopted candidate.** Replace only the lhs encoding
`U(m)=floor(q*m/t)` by `U_nearest(m)=round(q*m/t)`, deterministic nearest
rounding of the FULL ratio with a fixed tie rule, NOT nearest-Up. Keep all
remaining deterministic steps, the entire joint RNG law, parameters, operation
and input order identical: rhs remains unscaled canonical; packing, masks,
both-component PK drop, switches, trace, fresh-zero and Down remain unchanged.
No zero-skipping or change in sampler consumption is allowed.

Use [§15](#15-conditional-rigorous-per-direction-failure-theorem--reviewed-model-only),
**Explicit model**, and its following whole-matrix `A=0` witness, not merely
one zero coefficient. Packing, signs and padding preserve zero, and
`U(0)=U_nearest(0)=0` exactly. For the
[Final-share extension](#final-share-extension--additional-conditional-uniform-mask-premise),
both COMPLETE party lhs shares `A^(0)=A^(1)=0`; retain that paragraph's rhs
shares and assembly. Under the stated identical-step/joint-law conditions,
coupling with the same randomness gives identical ciphertexts, phases and
decoded errors for THIS witness. This is not general encoding equivalence.

Inherit exactly §15's **Explicit model**, **Second moments, with correlations
retained**, **Fourth moments** and **From the tail to every-mask failure**
premises and proof, including compatible integer lifts for `e_eff`, both-component
PK drop, the noise/remainder joint law and shared correlations, and the
31-switch/trace bounds `K=573441`, `sum_l ||L_l||<=469`. None is discharged
by changing the encoding. Thus conditional directional ±1 failure remains
`>2^-24`; this directional conclusion needs no mask independence.

Conditional FINAL ±2 failure remains `>2^-26` ONLY with the additional
**Final-share extension — additional conditional uniform-mask premise**:
full-ring masks uniform conditional on the ENTIRE joint unmasked phases,
including switch and fresh-zero errors, plus that extension's support and
assembly premises. No global independence is inferred; its existing proof
and distinction between sufficient and necessary mask conditions stand unchanged.

Therefore nearest-full-U ENCODING ONLY is conditionally insufficient for any
selected guarantee tighter than the applicable bound (e.g. failure `<=2^-40`
IF selected). No target, law or tolerance is adopted. This explicitly disposes
of an already implicit witness; it is not a new lower bound, new independent
encrypted-failure result, actual-engine finding, security/privacy or noise
instantiation, or B-N1 closure. Parameter, protocol or noise-law changes need
new review and adoption; no automatic approval of
[§16's wider family](#16-selected-deterministic-parameter-repair-design-family) follows.
The nearest candidate remains unadopted; `coefficient_phase_reference` source
and its whole floor-U behavior remain unchanged. No execution is claimed.

## 16. Selected deterministic parameter-repair design family

The user selected this substantial-adaptation DESIGN family on 5 November 2026
under [D67](../../paper/methods/DECISIONS.md#d67-bumblebee-wider-full-u-exact-down-design-target):
`bumblebee.olt-wide3x60-p35-fullU-exactDown.v1`. FAMILY SELECTED, DESIGN ONLY;
the concrete numerical profile remains UNADOPTED. D7's final-paper/evaluated-
profile mapping remains separately unresolved; this is not whole-model tolerance
qualification or runtime/source-implementation approval.

Independent **Astra46** (`ses_e7b86a27fffeSQk9kViA9Gracx`) clears this candidate
algebra only. Fix N=8192, t=2^64, packing, full-U, trace, 31 switches and two
directions. Consider distinct prospective NTT primes `q_i in [2^59,2^60)`
(i=1,2,3), `p in [2^34,2^35)`, all 1 modulo 16384. Then
`q=product_i q_i>=2^177`, `Q=p*q<2^215`. Existence, primality, construction,
context legality and backend acceptance are NOT established. The substantially
changed widths are selected as a design family only; concrete profile adoption
and implementation still require separate approval.

Under the exact compatible-integer-drop premise,

```text
||e_eff||_infinity <= 81920.5-81900.5/p < 81920.5,
integer ||e_eff||_infinity <= 81920,
E_enc support = Z = 20*N*81920+400*N+20 = 13,425,049,620 < 2^34.
```

Assume the source unsigned-Garner identity and support premises
`epsilon_KS=(sum_i d_i*e_i-r_b-r_a*s)/p`, `||d_i||_infinity<=q_i-1`,
`||e_i||_infinity<=20`, `|r_a|,|r_b|<=h=(p-1)/2`. The new bound is

```text
K = ceil((20*N*sum_i(q_i-1)+h*(1+20*N))/p)
  <= 3*(20*N)*2^26+81921 < 2^45.
H = 2*N*(t-1)*(Z+1)+469*K+Z,
t*H/q < (Z+1)/2^35+2^-59+2^-79 < 1/2+2^-59+2^-79 < 1.
```

Never reuse the old 573441 in this candidate. H assumes the unchanged arbitrary-
input trace selector/propagation 469 and full-U residual `|U(A)-q*A/t|<1`.
Conditional algebra gives deterministic directional circular ±1/final ±2 for every
mask and supported noise under the recorded premises; distributional
independence is not needed for this support bound. The source-body attachments
and reviewed composition below are conditional, not an instantiated wider-limb
engine proof.

Source-feasibility review `ses_e7b7cbd0dffe4rGmlmQmWnBsyw` was CANCELLED:
no completion or compatibility inference follows from that cancelled review.
Subsequent two bounded read-only audits supplied conditional lower-level and BFV
source-body attachments at F's pin `98a087f1bf5f7fa95196ebe16b0e1b0eeb632442`:

| Recovered source support | Conditional consequence / remaining limit |
| --- | --- |
| [zq/mod.rs 83–98](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe-math/src/zq/mod.rs#L83-L98), [ntt/mod.rs 80–84](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe-math/src/ntt/mod.rs#L80-L84), [rq/context.rs 47–65](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe-math/src/rq/context.rs#L47-L65) | `m<2^62` permits 60/35-bit widths; prime `1 mod 2N` and power-of-two `N>=8` are checks, not constructed parameters. |
| [ntt/native.rs 256–356](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe-math/src/ntt/native.rs#L256-L356) | u128 modular multiplication and native lazy `<4m`/`<2m` arithmetic fit these widths; only 100 seeded root candidates can fail. Optional `tfhe_ntt::Plan` internals UNINSPECTED; no backend qualification. |
| [rq/mod.rs 170–172](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe-math/src/rq/mod.rs#L170-L172), [241–242](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe-math/src/rq/mod.rs#L241-L242), [rns/mod.rs 57–148](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe-math/src/rns/mod.rs#L57-L148) | Per-limb `Array2<u64>` plus BigUint products, quotients, Garner and CRT: no fixed-total-width obstacle at `Q<2^215`. |
| [parameters.rs 437–446](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe/src/bfv/parameters.rs#L437-L446), [540–656](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe/src/bfv/parameters.rs#L540-L656), [688–743](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe/src/bfv/parameters.rs#L688-L743), [context.rs 86–102](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe-math/src/rq/context.rs#L86-L102) | BigUint `t=2^64`, explicit `[q1,q2,q3,p]`, conditional valid-prime checks; reversed prefixes give C0=pq/C1=q and child removes last p. Encoded data must explicitly request level 1, not default 0. |
| [key_switching_key.rs 114–148](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe/src/bfv/keys/key_switching_key.rs#L114-L148), [194–218](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe/src/bfv/keys/key_switching_key.rs#L194-L218), [367–376](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe/src/bfv/keys/key_switching_key.rs#L367-L376), [rns_component.rs 55–97](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe-math/src/rq/rns_component.rs#L55-L97) | Exactly three data Garner rows; unsigned component lifts give `0<=d_i<=q_i-1`. CBD(v=10) support 20 remains a premise. |
| [rq/switcher.rs 22–30](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe-math/src/rq/switcher.rs#L22-L30), [galois_key.rs 101–109](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe/src/bfv/keys/galois_key.rs#L101-L109), [modulus_switch.rs 24–60](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe-math/src/rq/modulus_switch.rs#L24-L60) | Upward ratio p is exact despite generic RnsScaler: integer-ratio fractional corrections vanish or are multiples of pq. Galois last-p drop uses exact component-centered division. |
| [plaintext.rs 145–169](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe/src/bfv/plaintext.rs#L145-L169), parameters.rs 692–718 above | Full-U identity uses the canonical coefficient representative, as qualified in the reviewed composition below; it is not unrestricted integer equality for a modular expression. |

**Decoder distinction.** [secret_key.rs 250–290](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe/src/bfv/keys/secret_key.rs#L250-L290) calls approximate
[RnsScaler](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe-math/src/rns/scaler.rs#L165-L173) (fraction/scalar rounding also 245–256, 307–366): no exact-Down guarantee or candidate mismatch established.
This blocks treating the mapped SOURCE decoder as exact; §2 already proposes
exact CRT/Down instead. Section 17 now selects reuse under an untimed private
component contract, design only; instantiated adaptation/implementation remains
UNAPPROVED, not a newly adopted approximate-decoder path or approximate-proof-only gate.
Completed PLLM source audit: [`wide.rs::Rns3::new` 162–180](../../crates/pllm-core/src/coefficient_phase_reference/wide.rs#L162-L180) accepts three ordered odd coprime limbs `<2^61`, `2^128<=q<2^191`; candidate three 60-bit limbs fit conditional on primes.
Its `reconstruct` 188–203, `encode` 205–208 and `down` 213–245 provide exact clear arithmetic: `q<2^180`, `t*x<2^244`, no 49-bit constants; not four-limb CRT/drop over Q.
`Wide` 5–7 exposes a public array with Copy/Debug; 113–140 and 214–243 use value-dependent division, branches and correction exit, with no eraser: no enforcing secret custody, constant-time or comprehensive copy-erasure contract.
Higher phase-reference owner hardcodes the old 147-bit profile; garble clear-phase getters expose ordinary Vec storage. No existing enforcing secret decoder was found.
[`shared_allocation.rs` 524–564](../../crates/pllm-garble/src/shared_allocation.rs#L524-L564) prepays inline/payload storage before charge release; payload must supply secret ownership/erasure.
Prefer reuse/extraction of these method-neutral numeric contracts over parallel paper-specific CRT.
Section 17 selects the untimed/output-only custody design and its explicit erasure
scope; concrete opaque zeroizing scratch/cleanup, genuine issuance and complete
original-owner byte+work admission remain missing implementation obligations.
Existing public clear-reference entrypoints remain reference-only; no implementation or engine activation is inferred.

**Reviewed conditional wider-profile composition (D61 documentation only).**
Two independent Astra reviews clear the source/proof composition for the D67
SELECTED DESIGN family `bumblebee.olt-wide3x60-p35-fullU-exactDown.v1`, a
substantial BumbleBee adaptation with concrete profile UNADOPTED. It retains the
N/t/prospective q_i/p ranges above, `q=product(q_i)` and `Q=p*q`; it does not
construct primes or adopt an instantiated profile.

| Attachment at F's pin above | Recovered mechanism / conditional use |
| --- | --- |
| [plaintext.rs 145–169](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe/src/bfv/plaintext.rs#L145-L169) | Select context at 145, match coefficients at 148; Small at 154 uses `q_mod_t`/scalar multiply/embed at 155; Large 158–163 uses scalar multiply with `q_mod_t`/BigUint embedding; into NTT at 167, multiply delta at 168. This body was recovered verbatim from the archived original primary-source tool response, not freshly fetched here. |
| [parameters.rs 649–656](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe/src/bfv/parameters.rs#L649-L656), [689–743](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe/src/bfv/parameters.rs#L689-L743), [763–769](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe/src/bfv/parameters.rs#L763-L769) | Ordered `[q1,q2,q3,p]` gives C0=Q/C1=q; explicit data level 1/key level 0. Delta at 692–718 is `-t^-1 mod q`. |
| [fhe-util/src/lib.rs 39–79](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe-util/src/lib.rs#L39-L79) | CBD(10) support ±20. Effective dropped PK error is bounded by 81920 above, NOT iid CBD; encryption support is the existing `Z=13,425,049,620`. Parameters/sampler are newly local-read attachments, not new distributional qualifications. |

For `0<=a<t`, let `r=[q*a]_t`. The recovered encoder is congruent to
`-t^-1*r mod q`. Its canonical COEFFICIENT representative is `floor(q*a/t)`,
since `0<=floor(q*a/t)<q`. Thus `t*Enc_q(a)=q*a-r` is INTEGER equality only
for that canonical representative; its full-U residual is in `(-1,0]`.
Neither a ciphertext coefficient nor an arbitrary modular lift inherits that
integer equality without the stated representative choice.

The prior-reviewed §3 switcher/scaler attachments supply exact p-upscale and
three unsigned Garner digits `d_i=[c]_(q_i)` in `[0,q_i-1]`, with key rows
`p*g_i*sigma(s)+e_i`. With compatible component lifts and centered `r0,r1`,
`xi=(sum_i d_i*e_i-r0-r1*s)/p`; the COMBINED numerator is divisible by p,
not each summand. Restoring `sigma(c0)` gives incoming phase transformed by
sigma, plus xi modulo q. Compatible lifts remain a premise, not inferred from
unrelated component representatives.

The §5 coefficient analysis supplies the reviewed same-phase trace theorem:
31 switches per direction, suffix norm sum 469, no independence or error/64
shortcut; the candidate has `q=1 mod 64`. Only this structural trace transfers,
NOT old radix/ternary K or old moduli. Combining it with the existing wider K
and `t*H/q<1` proof above gives circular ±1 per direction and ±2 after assembly
for every mask and supported noise, CONDITIONAL on the stated source/premise
attachments and exact CRT/Down REPLACEMENT. It is not equality with upstream
approximate RnsScaler decoding.

Provenance is granular: archived primary encoder body; newly local-read
parameters/sampler; prior-reviewed upscale/Garner/drop attachments; same-phase
trace as a reviewed theorem, NOT an original-source execution oracle. No fresh
all-body/hash verification or HTTP acquisition is claimed. Distinct context
validity, compatible lifts and backend reproduction remain unproved; no generic
fully qualified source path or issuer/funding/implementation approval follows.

Actual prime existence/construction, successful native constructor/context checks
and optional backend qualification remain open. Independent 128-bit security,
Galois and joint-privacy review remain
required; neither `Q<2^215` nor a 218-bit table threshold certifies security.
Wider-RNS bytes, cumulative work, funding and owner contracts remain missing.
Beyond D67's design-family selection, no concrete parameter, security, funding,
profile, engine or implementation approval follows; no execution is claimed.

## 17. Selected exact CRT/Down component design — Untimed reuse / Residue classes

The user explicitly selects **Untimed reuse** and private **Residue classes**.
This consolidates the supplied Astra component designs against the local source
anchors below. It selects design only: no implementation, parameters, engine
activation, numerical tariff or broader protocol-security approval. Earlier
conditional findings remain intact. Independent Astra review
`ses_e7b19a6aaffewNezGWztr3p2dx` and parent read-only review completed with
design-only clearance of section 17 and reconciled introduction/sections 4, 6–9,
16, not runtime or security approval. Other privacy, genuine
issuer and original-ledger admission gates are unchanged.

### 17.1 Numerical contract and shared arithmetic

For three ordered public odd, pairwise-coprime limbs, the candidate arithmetic
domain is `3<=p_i<2^61`, `q=p0*p1*p2>=2^128`; the limb bound implies `q<2^183`.
It is **not an approved instantiated prime/context/security profile**. Coprime
composite limbs suffice for this arithmetic contract, not for BFV construction.
For every triple of u64 words, define

```text
x = CRT(a0 mod p0, a1 mod p1, a2 mod p2),  0<=x<q,
out = floor((2^64*x + floor(q/2))/q) mod 2^64.
```

Public setup reuses current `Rns3`, which stores three ordered limbs, q, p01
and two Garner inverses: 13 u64 payload words only, not a `sizeof`, allocation
or stack bound. There is no extra basis precompute; `h=floor(q/2)` is computed
by `half` inside Down. Its constructor, reconstruction and exact Down are
[`wide.rs::Rns3`](../../crates/pllm-core/src/coefficient_phase_reference/wide.rs)
(`new`, `reconstruct`, `down`); public clear `encode`/U behavior stays unchanged.
The existing four-word representation covers this domain: `2^64*x+h<2^247`.
No second paper-specific CRT is prescribed.

Selected private path first normalizes each word modulo its public limb, then
calls legacy canonical reconstruction and exact Down: three bounded u64
remainders per coefficient. Variable-time arithmetic is permitted **only under
this selected component design**, not inherited by COMMON or other secret paths.
An opaque wrapper is an access boundary, **not timing isolation**. Legitimate
canonical outputs are unchanged; existing public clear APIs still reject
`NonCanonical`. Private residue semantics introduce no hidden invalidity flag,
secret-validity status or conditional output suppression/release refusal.
Even a fixed schedule would leak through such status or refusal. Public-domain,
provenance and resource checks remain separate gates. Totalization proves no
phase authenticity, provenance or release entitlement; it grants no raw caller
import authority and changes no ciphertext validation rule.

Source-level totality argument, not an executed test proof: normalization makes
the residues canonical; Garner gives `x01<p0*p1` and final `x<q`, with four-word
operations fitting. For `B=2^64`, initial Down numerator `num=B*x+h<2^247`;
the wrap case has quotient B and correctly returns zero modulo B. Otherwise
`num<q*B`; after normalization, `D<2^192`, top divisor word `d2>=B/2` and
`N<D*B<2^256`. True quotient `k<B`; the floored top-word estimate, capped at
`B-1`, lies between k and `k+2`. Thus at most two decrements suffice and the
final remainder check succeeds. Under the validated immutable domain, normalized
inputs cannot secret-dependently return `Capacity` or `NonCanonical`; no output
validity gate is needed. Domain/provenance/resource refusal remains a separate
public, authorized gate.

### 17.2 Genuine issuer and exclusive consuming custody

No genuine secret-phase issuer exists in the inspected paths.
[`garble coefficient_phase_reference.rs`](../../crates/pllm-garble/src/coefficient_phase_reference.rs)
is an original-counter **clear** adapter with public `ClearRegion::{phases,
converted,matrices,inputs}` getters, not private issuance. Required future issuer
retains actual key authority and binds authenticated ciphertext, ordered domain,
session, recipient, replay state and original budget/private ledger. Lifecycle:

```text
PhaseOwner -> exclusive consuming permit -> Running -> authorized output / Burned
```

Guard identity, digests, caller labels or a clear owner cannot mint this authority.
[`material_wire.rs`](../../crates/pllm-garble/src/material_wire.rs) explicitly
provides corruption checks, not authentication, replay prevention or issuance.
Phase custody is local to the authorized decrypting party; entitlement to decoded
output is **not permission to disclose the phase**. No public raw-phase
constructor/getter, `Clone`, `Debug`, serialization or adoption callback belongs
on the private boundary. Genuine issuer and output entitlement remain unimplemented.

USER71483's COMMON original-issuer design tranche is recorded in
[secure HE sampling §7.1](he-secure-sampling-design.md#71-common-exact-ring-original-issuer--proposed--implementation-held):
**PROPOSED / implementation HELD**. Private custody names, generation/lineage,
authentication envelope, fail-burn and original-ledger obligations are proposed
contracts only; authentication/peer binding and all implementation, secret,
numeric, parameter and backend authority remain held. No genuine issuer is added.

### 17.3 Prepaid original-ledger admission, not a guessed allowance

Consume the exclusive permit, then debit work/reserve bytes **before covered
allocation or secret read**. Bootstrap accounting must precede construction of
the accounting owner itself. Keep original attempts and cumulative work through
retries and retained outputs: no fresh-ledger reset, successful-debit refund or
retry bypass. Failure after permit consumption burns it. Live backing stays
charged through retirement; payload zeroizes/retires before its charge releases.

Reuse shared semantics, not clear numeric caps or tariffs:
[`shared_allocation.rs`](../../crates/pllm-garble/src/shared_allocation.rs)
`OwnedExternalCharge::try_build` reserves inline storage before its builder;
`OwnedCharged<T>` declares payload before charge, enforcing destruction ordering,
not wiping. [`clear_input_owner.rs`](../../crates/pllm-compiler/src/clear_input_owner.rs)
keeps ledger/attempt state private, prepays construction and zeroizes admitted
numeric pool tails; its clear owners are not source/import authority.
[`row_fold::WorkLedger`](../../crates/pllm-core/src/row_fold.rs) supplies
nonrefundable cumulative debit semantics, including failed attempts/cleanup,
not an approved secret decoder fee schedule.

For public coefficient count `n_b`, complete quote must bound

```text
P = max_t[root + input + domain + scratch + output + metadata
          + construction/failure overlap],
Qwork = Qbootstrap/domain/issuer/init + n_b*Qdecode + Qcontrol/failure/erase.
```

Count actual retained backing capacity, full parent tails, input/output/cleanup
lifetimes and old/new overlap, not logical lengths. Packed input `24*n_b` bytes
is only a lower bound if retained capacity is larger. Work inventory is fixed
and public, conservatively covering bounded variable-time branches, initialization,
failure and owned-buffer erasure. No numerical tariff, inherited clear fees,
stack bound or complete backing quote is approved. A public bounded quote is not
payment; absence of finite original-ledger funding still refuses admission.

#### 17.3.1 Source-audited conservative arithmetic inventory — not a tariff

Astra inventory `ses_e7b6c8806ffetlxFy0iWZ4Vfwn` received completed independent
source review by `ses_e7b19a6aaffewNezGWztr3p2dx`. Anchors:
[`wide.rs` 122–245](../../crates/pllm-core/src/coefficient_phase_reference/wide.rs#L122-L245)
and [`work.rs` 1–49](../../crates/pllm-core/src/coefficient_phase_reference/work.rs#L1-L49).
The following are conservative source macros, **not instruction counts, latency,
machine-timing, stack/allocation proofs or fees**:

| Stage | Arithmetic macro inventory |
| --- | --- |
| Private normalization | Exactly 3 REM64; future wrapper loop/control/scratch writes unpriced. |
| CRT | At most 3 canonical comparisons, 1 REM64, 2 modular subtractions (each one comparison and at most 2 scalar subtractions), 2 u128 multiply/remainder pairs, 2 Wide-word MUL, 2 Wide ADD, 1 four-word REM. |
| Down | At most 4 Wide SHIFT, 1 HALF, 1 Wide ADD, 1 Wide-word MUL, 3 Wide SUB, 6 Wide CMP, 1 leading-zeros, 1 u128 division, 2 checked quotient decrements; top-construction shift/OR, top comparison and correction loop (at most 2 visits) counted separately. |
| Composed normalization + CRT + Down | At most 4 u64 remainders, 6 u128 remainders, 1 u128 division, 2 scalar u128 multiplications, 3 Wide-word MUL, 3 Wide ADD, 3 Wide SUB, plus shifts/half/control above. |

Fixed-wide helpers traverse at most four words/pairs; do not double-count their
expanded products or equate a macro with one CPU operation. Existing `work.rs`
constants price only the prior fixed clear campaign, not this private wrapper.

Public setup adds at most 3 limb checks, 2 Wide-word MUL, product-capacity
predicates, 1 u64 remainder, 1 four-word REM and 2 inverse calls. Each inverse
has an explicit 128-visit cap; inputs below `2^61` need at most 122 Euclidean
updates plus a terminal visit. Each update uses 1 signed division, 2 signed
multiplications, 2 signed subtractions and 4 state replacements, with
`rem_euclid` on success. Coefficient magnitudes are at most p and `d<=p`;
products at most `p^2` and subtractions at most `p^2+p` fit i128.

All loop, wrapper, control, initialization, owned-zeroization, output, attempt
and issuer-construction work remains unpriced. Genuine issuance, concrete owned
layout/full-copy envelope, bootstrap/control/cleanup/output tariffs and original-
ledger funding remain unresolved. Named `Wide` locals and optional 512-byte/
184-byte proposals do not bound transient copies. This source-level arithmetic
audit approves no implementation, parameters, security or fees.

#### 17.3.2 Concrete storage/lifetime proposal — UNSELECTED

Planning Astra `ses_e7b0dac99ffe4OS7FXYn1NANph`, independently reviewed by
`ses_e7b19a6aaffewNezGWztr3p2dx`, cleared viability with the corrections below.
No new security-policy decision is required; this remains an **unselected**
proposal, not authority, full-quote, tariff, implementation or security approval.
All named private types below are future non-Clone types, not existing APIs;
write n for the public coefficient count `n_b`.

| Proposed storage | Custody, layout and lifetime |
| --- | --- |
| `PrivateInput` | Fully initialized limb-major `Vec<u64>`, `words[l*n+i]`, `len=capacity=3*n`, held by the genuine original issuer owner. Original charge/input custody lasts at least through last output retirement; original may outlive outputs, never release early. Admission must not zero existing phase input. |
| `PublicDomain` | Inline validated `Rns3`, public n and one-use state in the real root; quote actual full parent size and retained bindings, not 13 payload words as `sizeof`. |
| `Workspace` | Owned inline `repr(C) { norm: [u64;3], reconstructed: [u64;4], output: [u64;1] }`; one exclusive workspace reused per coefficient and retained with opaque output. Eight payload words only; future actual workspace size is quotable, not a full stack bound. |
| `OutputStorage` | Initialized `Vec<u64>`, `len=capacity=n`, allocated only after payment. Output payload/workspace zeroizing RAII destroys before phase/charge and input-attachment loan; include actual future parent headers, padding, bindings and all retained capacities. |

`PrivatePhase`/output must keep a genuine **external** lifetime-bound loan or real
ownership transfer, not a self-referential borrow from a `try_build`-owned root.
Do not copy clear `BorrowedOutput::{values,original}` into this private boundary.
Stream three normalized words → CRT → owned reconstructed words → Down → owned
output slot → destination; total normalization adds no hidden validity flag and
partial output is inaccessible. Execution stays synchronous/exclusive, without
callbacks/suspension. Install zeroizing guards before secret population or any
subsequent fallible operation, including partial construction; nonpanicking
failure/unwind cleanup and cancellation ownership need explicit proof. Abort,
leak and `mem::forget` exclusions remain.

UNSELECTED construction candidate: inside existing `OwnedExternalCharge::try_build`,
reserve before builder entry, then install `Zeroizing<Vec<u64>>::new(Vec::new())`
BEFORE fallible `try_reserve_exact(n)`, check capacity policy, then `resize(n, 0)`;
install `Zeroizing<[u64;8]>` before any phase read. Follow
[`memory.rs` 275–287](../../crates/pllm-core/src/maverick_batch/memory.rs#L275-L287), not a new allocator.
Payload field cleanup order is output → workspace → genuine input-retention loan;
[`OwnedCharged` 550–555](../../crates/pllm-garble/src/shared_allocation.rs#L550-L555) drops payload before charge.
Public-size admission/work debit on the original ledger precedes backing allocation.
Never wipe borrowed input destructively: its original owner retains terminal wipe duty.
This covers proposed owned buffers only, not transient erasure or a complete-parent quote.

Reuse `shared_allocation::{OwnedCharged,ExternalCharge,Budget}` and
`row_fold::WorkLedger`; extract method-neutral private-attempt/lifetime machinery
from clear owners, not authority. Sources: `clear_input_owner.rs`
[98–107](../../crates/pllm-compiler/src/clear_input_owner.rs#L98-L107),
[124–130](../../crates/pllm-compiler/src/clear_input_owner.rs#L124-L130),
[212–227](../../crates/pllm-compiler/src/clear_input_owner.rs#L212-L227),
[580–655](../../crates/pllm-compiler/src/clear_input_owner.rs#L580-L655);
`shared_allocation.rs` [520–555](../../crates/pllm-garble/src/shared_allocation.rs#L520-L555),
[572–584](../../crates/pllm-garble/src/shared_allocation.rs#L572-L584).
`OwnedCharged` orders destruction, not wiping. No public raw-phase Vec adoption,
digest/caller ownership labels, fresh ledger or fake-issuer callback is authority.

Work/attempt debits are nonrefundable; byte reservations release after actual
retirement. Same-original-ledger reservation transfer must not double-charge
existing input/root. Quote original full parents/backing plus maximum-stage NEW
actual capacities, complete future parents, transitions and Copy-temporary storage;
count inline context/workspace once. Fund root bootstrap before constructing its
accounting owner. Consume authenticated permit before fallible admission, then
public checked-n quote and work debit/reservation before read/init/allocation.
Absent genuine issuance or a priced proof, refuse.

Payload inventories/success path only, not fees/full cleanup: existing input's
3n words are already initialized—do not overwrite. Fresh wrapper output/workspace
initialization and terminal wipe are each `n+8` words; input's terminal `3n` wipe
belongs to its actual original owner at its retirement. Whole-fresh construction/
root retirement is `4n+8` each only if that owner constructs/retires all three
stores; issuer-population writes are additional. Success adds `8n` scratch-wipe
writes, `3n` reductions, n CRT, n Down and n commits; workspace population,
index/control/moves, failure cuts and repeated cleanup remain unpriced.

Allocation-request correction (independent Astra `ses_e7a173872ffepdbQLq77RgJwxt`):
[`rust-toolchain.toml`](../../rust-toolchain.toml) pins Rust 1.85.1; actual compiler execution is unverified.
For an empty non-ZST `Vec<T>`, that version's `try_reserve_exact(n)` takes
`RawVec::grow_exact`: checked layout is exactly `n * size_of::<T>()`, overflow
fails before allocation, n=0 allocates nothing, and fresh storage uses allocate,
not grow. `MIN_NONZERO_CAP` rounding belongs only to the amortized path:
[`raw_vec.rs` 594–685](https://github.com/rust-lang/rust/blob/1.85.1/library/alloc/src/raw_vec.rs#L594-L685),
[760–781](https://github.com/rust-lang/rust/blob/1.85.1/library/alloc/src/raw_vec.rs#L760-L781).
The [Vec API](https://github.com/rust-lang/rust/blob/1.85.1/library/alloc/src/vec/mod.rs#L1342-L1383)
does not deliberately overallocate, but is no all-version byte-Layout promise.
A post-capacity check or caller-request-bound safe API alone cannot prove exact actual allocation;
the selected-version source path proves its Layout/capacity policy, not actual compiler execution or physical requests.
Neither bounds allocator usable blocks, RSS or internal overhead; `GlobalAlloc` may abort on OOM
([contract](https://github.com/rust-lang/rust/blob/1.85.1/library/core/src/alloc/global.rs#L136-L200)).
Custom `alloc_zeroed(Layout)` gives the same request bound, not physical bounds; no new unsafe allocator is needed.
Reuse fallible Vec/Zeroizing and charge composition, as in [`memory.rs` 230–303](../../crates/pllm-core/src/maverick_batch/memory.rs#L230-L303),
not a duplicate paper-specific owner: Maverick's u32 `Words` is not u64 phase authority;
any shared extraction must forbid core→garble dependencies and keep source budget/issuer separate.
This selects no layout or n+workspace full quote, grants no adoption/implementation approval,
and leaves wiping/failure guards, actual-capacity overlap, payload-before-charge, full-wrapper
quote, work tariff, funding and actual-toolchain qualification gates unchanged.
Complete allocation paths and target/compiler stack/frame/spill/move/
Result/failure Copy envelopes remain unproved. `Rns3`/`Wide` transient copies are
excluded from comprehensive erasure only, not storage/work quoting. Finite tariffs,
original-ledger funding and genuine authenticated issuer binding to key/ciphertext/
ordered domain/session/recipient/replay remain missing; section 17.6 stays held.

### 17.4 Owned-buffer erasure and explicit untimed exclusions

Owned phase, scratch and retained output buffers require concrete zeroizing
retirement/failure owners; outputs remain charged until authorized consumer
retirement. Their representation and complete layout/cleanup proof are still
missing. Existing `Wide` is `Copy`/`Debug`, with value-dependent comparisons,
division and correction exits: transient `Wide`, stack and register copies are
**not comprehensively erased**. The selected component explicitly excludes that
stronger guarantee; wrapping legacy arithmetic does not establish it.

Initial private execution is synchronous, without callbacks or suspension, under
an exclusive nonshared owner. Cancellation and unwind must specify ownership and
zeroizing retirement wherever scope permits; process abort, leaks and
`mem::forget` are explicit erasure exclusions, not successful cleanup paths.
Release is bounded to authorized decoded output. No machine constant-time,
timing/error masking or secure physical-memory isolation claim is made.
No secret-validity-dependent output/status is permitted. These component-scoped
exclusions do not weaken COMMON's distinct engine/seed/state/constructor-copy
requirements or establish whole-BumbleBee security.

The paper's static semi-honest source model does not itself impose constant-time
machine execution; choosing an untimed component does not prove that source model
matches this runtime. [`paper/SECURITY-AND-CORRECTNESS.md`](../../paper/SECURITY-AND-CORRECTNESS.md)
proves a conditional untimed **P2-B** theorem, not a BumbleBee theorem. Neither
that theorem nor this selection closes authentication, circuit privacy or the
other protocol gates above.

### 17.5 Optional fixed-schedule alternative — NOT SELECTED

Bit-Horner CRT/restoring division may share numerical proofs, not automatically
code. Its proposed 512-byte explicit workspace and 184-byte public-context
payload need layout proof; neither is the selected untimed allocation quote,
stack bound or tariff. Former secret-validity checks/release gates are withdrawn.
For reference only, proposed macro inventory is `257 ADD4, 512 SUB4, 512 SELECT4,
192 MASK4, 256 SHIFT1_4, 1 SHIFT64_4, 448 bit extractions, 256 quotient appends`.
These are abstract macros, not machine-operation counts or time guarantees; no
second architecture or fixed-schedule implementation is selected here.

### 17.6 Later qualification and remaining blockers

Tests **to author later**, not authored or run here: differential exact CRT/Down
against an independent integer oracle; all-u64 residue semantics, ordered limbs,
domain/rounding boundaries and wrap; unchanged legacy `NonCanonical` rejection;
provenance, replay, foreign backing and private-ledger refusal; cancellation,
construction/failure cuts, retained outputs, zeroizing destruction and
payload-before-charge ordering. Ordinary tests do not prove excluded timing or
comprehensive transient-copy erasure properties.

Missing genuine issuer/authentication, concrete owned-erasure representation and
layout, finite tariffs and complete backing quote, circuit privacy, security and
parameter construction remain blockers. This component selection excludes whole
BFV engine approval, four-limb CRT, component drop and secret-key multiplication.
No source code, tests, builds, experiments or native execution are authorized or
claimed; only this document is changed, with read-only design review completed.

## 18. COMMON candidate-bound consuming-decode admission — design only

[D67](../../paper/methods/DECISIONS.md#d67-bumblebee-wider-full-u-exact-down-design-target)
selects this candidate-bound COMMON issuer → consuming-decode admission DESIGN
tranche for §16's family, using §17's unchanged Untimed reuse / Residue classes
component. The states below are ROLE OBLIGATIONS, not implemented types/APIs or
a paper-specific issuer. No concrete primes, backend, §17.3.2 storage, trust/
disclosure policy, security, funding, source implementation or execution is
selected. D7 final-paper/evaluated-profile mapping remains open.

Source key (reuse evidence, not authority): **S** is the COMMON PROPOSAL in
[secure sampling §7.1](he-secure-sampling-design.md#71-common-exact-ring-original-issuer--proposed--implementation-held),
especially 518–558; **B** is this packet's §§16–17;
**R** is [`wide.rs` 150–232](../../crates/pllm-core/src/coefficient_phase_reference/wide.rs#L150-L232);
**A** is [`shared_allocation.rs` 538–573](../../crates/pllm-garble/src/shared_allocation.rs#L538-L573)
and [665–714](../../crates/pllm-garble/src/shared_allocation.rs#L665-L714);
**C** is [`clear_input_owner.rs` 98–115](../../crates/pllm-compiler/src/clear_input_owner.rs#L98-L115),
[195–227](../../crates/pllm-compiler/src/clear_input_owner.rs#L195-L227) and
[580–656](../../crates/pllm-compiler/src/clear_input_owner.rs#L580-L656);
**W** is [`row_fold.rs` 27–59](../../crates/pllm-core/src/row_fold.rs#L27-L59).
Clear arithmetic, accounting and destruction patterns supply no genuine issuer.

| Required transition | Binding, source correspondence and missing proof |
| --- | --- |
| 1. Outside → funded entry | Authentic original authority must cover delivery, lookup, authentication, rejection, bootstrap and terminal handling BEFORE those operations, with stable original owner/disposition destination retained outside the fallible call and finite invocation rights. C/W private cumulative accounting is NOT entry funding; neither a constructor nor a ledger retrofunds entry. Missing: authenticated ingress graph, actual funding and entry/cleanup quote. |
| 2. Entry → PhaseOwner eligibility | Bind actual sampled key birth/descendants, authenticated immutable two-component ciphertext, original purpose, direction, output ordinal/lineage, ordered levels/domain/representation, session/request/epoch/roles/recipient/replay, original budget/private ledger and fixed destination; keep keys party-local. Bind immutable PUBLIC ORDERED coefficient selection AND output mapping to the authentic original operation before secret access. S 518–545 is PROPOSED, not a genuine issuer; missing key generation, peer credentials, replay/purpose/release policy and evidenced selection rule. Preconsumption refusal uses only independently funded ingress/invocation resources: unauthenticated labels, replay claims or failed lookup cannot select, consume or burn an unrelated epoch/siblings. |
| 3. PhaseOwner → exclusive consuming permit | Public authentication checks precede secret access; consume one-use authority BEFORE fallible covered admission, retaining the original attempt, destination and sibling inventory. Immutable selection/output mapping includes direction/ordinal/recipient/fixed destination; COUNT alone grants no index/order/subset/release entitlement. An authenticated fixed rule may replace an index list only with evidence; D7 mapping stays OPEN. No active-caller substitution, phase callback, raw-phase factory/import, usable permit returned on failure or replacement ledger. S 547–558/C provide patterns, not exclusive/replay atomicity, sibling invalidation or refusal/unwind proof. Authentication → consumption atomicity is UNRESOLVED; preconsumption refusal cannot burn unrelated collections. |
| 4. Consumed → Running | Check public count/demand, debit newly covered work against REMAINING original allowance (including reserved failure/cleanup), and reserve uncovered bytes BEFORE allocation, initialization or secret read. End-to-end demand includes entry/bootstrap/authentication ONCE: incurred work must already have been funded before execution and remain original accounting history, never constructor/ledger retrofunding. Cover authority/control/phase/decoder/destination/failure/cleanup and total/control ceilings. W successful debit is nonrefundable; failed debit leaves state unchanged. A `try_build` reserves inline capacity before builder entry, NOT entry funding/full quote. Missing actual constructor/callee/layout/overlap, fees, original allowance and enforcement; controlled-capacity refusal must respect actual native ordering with observer exhaustion, whose binding remains UNIMPLEMENTED. |
| 5. Running → opaque Decoded | Genuine local phase/transform is at DATA C1=q. A proposed private wrapper normalizes three residues against ORDERED `[q1,q2,q3]`, then uses R's `Rns3::reconstruct`/exact `down`, for the public admitted coefficient count/selection. This is NOT four-limb/key C0=pq CRT; existing `reconstruct` rejects noncanonical inputs and does NOT perform wrapper normalization. Require exclusive synchronous execution, no callback/reentry/suspension, guards BEFORE secret population and no partial exposure. R implements math, not a private wrapper/genuine phase producer; immutable validated domain, valid-phase lineage and all guards remain missing. Key-birth continuity alone proves neither noise bounds nor honest evaluation. Untimed residue semantics add no hidden validity-release flag. |
| 6. Opaque → authorized Output/Spent | Release only the bound decoded output to its fixed entitled destination, NEVER phase material. Commit consuming release and externally retain disposition BEFORE any observable prefix; mark consumption/disclosure BEFORE exposure. Full-ring INTERNAL transform does NOT authorize full-ring decoded release. Retain original key/domain/input, full parent tails and charges through last dependent retirement. C 619–656 exposes clear getters: these MUST NOT transfer to the private boundary. A payload-before-charge drop is ordering, not custody/wiping proof. Missing release entitlement, opaque destination, transfer/lifetime proof and actual atomic/prefix/ACK boundary. |
| 7. Postconsumption failure → Burned | No retry/continuation/new ledger; successful debits remain. Burn only the genuinely bound collection/sibling inventory, with stable externally retained original terminal disposition even on unwind. Retire partial outputs/scratch BEFORE capacity release; borrowed input's original owner retains wipe duty. Later serialization/transport failure preserves committed entitlement, disclosed-prefix status, backing and charges: NEVER restore a permit, reset entitlement or redeliver by reset. Disclosure is irreversible; burn is not unpublication. Missing bounded nonpanicking refusal, unwind/cancel/Drop graph, terminal persistence and retirement proof. Host-only healthy refusal is NOT HE adoption. |
| 8. Unknown retirement → Burned + retained obligations → Retired ONLY on proof | Burn is not physical retirement or refund. Retain genuine backing, owner inventory, committed/prefix disposition and charges; no logical-last-use release, new ledger or double refund. A `split_owned_into` transfers capacity, NOT backing custody/whole-payload retention; ordinary charge Drop RELEASES bytes, so premature Drop must be prevented. Later transport failure cannot erase disclosed-prefix status or reinstate authority. Missing backing+charge coretention, bounded failure handling and authenticated final-release/retirement proof; a retained label alone is not actual custody. |

Do not expose or reproduce clear-owner getter/copy interfaces on the private boundary.
Untimed residue semantics are NOT timing isolation, COMMON privacy or constant-time
qualification. Any allowed partial release must ALREADY be bound by the authorized
mapping. Atomic release requires an actual all-or-nothing boundary; atomic/prefix/
acknowledgement semantics and bounded failure handling remain UNRESOLVED, not a
new disclosure selection. Public counts alone never establish release entitlement.

Symbolic end-to-end demand, separate capacity and work obligations:

```text
P = max_stage(original retained backing + key/ciphertext/domain
              + constructor/transform overlap + phase/scratch/output
              + control/cleanup)
Q = Qentry/bootstrap/auth/issuer/phase + n*Qdecode + Qrelease/failure/erase
```

P is TOTAL simultaneous LOGICAL capacity, NOT a fresh reservation amount.
Partition by actual original backing owner/accounting root, count shared backing
ONCE at FULL capacities/tails, and reserve NEW capacity only for demonstrably
UNCOVERED owners/overlap. Never subtract unretired backing or interchange party
budgets; the partition/transfer proof is MISSING. Q is separate LOGICAL work,
with entry/bootstrap/authentication counted once and incurred work retained in
the original history. Debit only newly covered work from remaining allowance,
including reserved failure/cleanup; do not pay historical work twice or pretend
it was prefunded by a later ledger. Here n is the public admitted decode count,
bound to the immutable selection/output mapping, not authority to choose it.

No numerical rates, CPU/RSS bound, payment, complete quote, funding or fit is
established. Authentication/key continuity/valid-phase provenance, capacity,
ownership, fees, retirement, protocol security/circuit privacy and actual funding
proofs remain missing. COMMON's proposal and B's conditional math cannot supply
them by analogy; D67 grants no implementation or execution and changes no caps/HOLDs.

### 18.1 Selected logical output mapping

The 5 November 2026 user follow-up selects this PUBLIC LOGICAL mapping under
[D67](../../paper/methods/DECISIONS.md#d67-bumblebee-wider-full-u-exact-down-design-target),
not physical storage, execution/wire order or extra disclosure. Two independent
source/role and conditional-index reviews passed; these are NOT test results.
The retained source correspondence is
[indexing corrections 45–53](bumblebee-indexing-corrections.md#L45-L53),
[80–88](bumblebee-indexing-corrections.md#L80-L88) and
[§3's original OLT/share roles](#3-original-olt--retained-shares-both-cross-directions).

Let public positive integers be `k,m,n,kw,mw,nw,N`, with `kw*mw*nw<=N`, and set
`K=ceil(k/kw)`, `C=ceil(n/nw)`, `G=ceil(K*C/mw)`. The abstract packed domain
has G groups of N coefficients EACH; that is NOT a physical-storage choice.
Here K is the row-block count, not §16's switch-error bound; n is the matrix
column count, not §18's symbolic decode-count placeholder. Valid output cells
are `0<=a<k`, `0<=b<n`, in row-major ordinal `a*n+b`. Their forward map is:

```text
(alpha,r) = divmod(a,kw)
(gamma,c) = divmod(b,nw)
(g,j) = divmod(alpha*C+gamma,mw)
ell = (r*nw+c)*mw+j
select packed(g,ell)
```

No padded output block or cropped cell is an output. Inner-block products
accumulate BEFORE parsing, as in the retained mapping; this logical rule does
NOT extend numerical lineage or approve more accumulation.

**Finite index lemma.** `alpha<K`, `gamma<C`, `r<kw`, `c<nw` imply
`block=alpha*C+gamma<K*C`, `g<G` and `0<=ell<kw*nw*mw<=N`;
the ordinal is `<k*n`. For a candidate `0<=g<G`, `0<=ell<N`, invert by:

```text
(h,j) = divmod(ell,mw)
(r,c) = divmod(h,nw)
block = g*mw+j
(alpha,gamma) = divmod(block,C)
a = alpha*kw+r
b = gamma*nw+c
accept iff r<kw and block<K*C and a<k and b<n
```

All quantities are nonnegative in this public domain; rejected addresses are
padding/unselected. The unique divmod decompositions recover every forward
image and reconstruct every accepted address, proving injectivity and bijection
ONTO SELECTED addresses, NOT the whole packed domain. This INDEX LEMMA needs
no power-of-two or divisibility condition; the producer's trace/interleave has
additional premises. Any future public counts, ceilings, products and indexes
require CHECKED arithmetic before use; no implementation is authorized here.

Direction descriptors are logically ordered `(0,1)` then `(1,0)`, NOT execution
or wire order. In `(i,j)`, Pi owns the key/lhs and retains selected `Parse(D_ij)`;
Pj evaluates/masks and retains selected `Parse(S_ij)`. The final local shares are:

```text
C0 = A0*B0 + Parse(D_01) + Parse(S_10)
C1 = A1*B1 + Parse(S_01) + Parse(D_10)
```

These remain separate PARTY-LOCAL shares, not reconstructed cross products.
Bind operation, direction, share kind, ordinal, logical role and destination
BEFORE secret access; a logical role is NOT a real principal, credential or
authenticated entitlement. No active-caller substitution is allowed. §18's
genuine-authority, consuming-release/no-reset and retained-charge requirements
remain unchanged; partial/atomic delivery and entitlement enforcement are NOT
selected by this map.

The generic [`coefficient_layout.rs` 228–245](../../crates/pllm-core/src/coefficient_layout.rs#L228-L245)
and [371–382](../../crates/pllm-core/src/coefficient_layout.rs#L371-L382) retain
degree `<=4096` and additional window gates, including power-of-two/bounded mw;
that source is UNCHANGED. N=8192/D7 producer correspondence remains UNQUALIFIED.
The ±1/±2 result applies ONLY under the recorded H/K, accumulation and trace
premises, not arbitrary shapes or more inner blocks. No actual issuer, phase
provenance, quote/payment, resource fit, backend, funding, security, source
implementation or execution qualification follows. §17.3.2 storage remains
UNSELECTED; all prior caps and HOLDs remain unchanged.

### 18.2 Selected-output source correspondence — NONNORMATIVE, conditional

This bounded equation attachment specializes §18.1 under the existing D67 DESIGN
grant; it does not revise that reviewed mapping or select a new parameter,
security, funding, disclosure or implementation policy. Candidate q and p remain
symbolic as in §16: `q=q1*q2*q3`, DATA `C1=q`, KEY `C0=p*q`, ordered
`[q1,q2,q3,p]`. Fix `N=8192`, `t=2^64`, shape `(3,65,129)` and window
`(1,64,128)` only. Here C0/C1 context labels are distinct from final party shares.

**Source boundary.** Local text anchors are
[`coefficient_phase_reference.rs` 283–402](../../crates/pllm-core/src/coefficient_phase_reference.rs#L283-L402),
[`wide.rs` 174–231](../../crates/pllm-core/src/coefficient_phase_reference/wide.rs#L174-L231),
[`coordinates.rs` 66–149](../../crates/pllm-core/src/coefficient_layout/coordinates.rs#L66-L149),
[`schedule.rs` 96–144](../../crates/pllm-core/src/coefficient_layout/schedule.rs#L96-L144)
and [`trace.rs` 47–91](../../crates/pllm-core/src/coefficient_layout/trace.rs#L47-L91).
The phase owner hardcodes the OLD limbs (24–27), requires
`CryptoError::ExplicitZero` (217–225), and computes clear products, not encrypted
phases. Generic layout's legacy maximum 4096 does not qualify N=8192; the explicit
clear path supplies its own 8192 bound (237, 388–402), not an encrypted capability.

**Packing through the accumulated phase.** For direction `(i,j)`, take canonical
matrices `A_i in [0,t)^(3*65)` and `B_j in [0,t)^(65*129)`. For row `a=0..2`,
inner block `beta=0..1` and column block `gamma=0..1`, in `R_t=Z_t[X]/(X^N+1)`:

```text
L_(a,beta) = sum_(h=0)^63 A_i[a,64*beta+h] X^(-h),
V_(beta,gamma) = sum_(h=0)^63 sum_(c=0)^127
                  B_j[64*beta+h,128*gamma+c] X^(64*c+h).
```

Out-of-range matrix entries are zero: beta=1 retains only h=0; gamma=1 only c=0.
Let barL/barV be coefficientwise canonical `[0,t)` representatives AFTER
negacyclic reduction. In particular h>0 puts `[-A_i[a,64*beta+h]]_t` at N-h;
apply this ring-t sign BEFORE U, not a signed integer encoder substitution.
Source 293–304 implements these addresses/signs and the unscaled RHS multiplier.
For every complete canonical lhs coefficient, define

```text
U(x)=floor(q*x/t)=q*x/t+eps(x),  0<=x<t,  -1<eps(x)<=0,
phase(Enc_i(barL_(a,beta))) = U(barL_(a,beta))+E_(a,beta) mod q,
F_(2*a+gamma) = sum_(beta=0)^1
                 (U(barL_(a,beta))+E_(a,beta))*barV_(beta,gamma) mod q.
```

U/eps act coefficientwise; products are negacyclic. The second line is REQUIRED
actual encrypted-phase identity, not something the clear source establishes.
Both inner blocks accumulate before any Down (source 287–321). No intermediate
Down, centered RHS, Delta substitution or multiplicativity of U is available.
To see the selected message, define the ring-t polynomial M and compatible
integer negacyclic lift W, with coefficientwise canonical representative barM:

```text
M_(a,gamma) = sum_beta L_(a,beta)*V_(beta,gamma) in R_t,
[M_(a,gamma)]_(64*c) = sum_(u=0)^64 A_i[a,u]*B_j[u,128*gamma+c] mod t,
W_(a,gamma) = sum_beta barL_(a,beta)*barV_(beta,gamma)
            = barM_(a,gamma)+t*Kcarry_(a,gamma),
F_(2*a+gamma) = (q/t)*barM_(a,gamma)
               + sum_beta (eps(barL_(a,beta))+E_(a,beta))*barV_(beta,gamma)
               + q*Kcarry_(a,gamma) mod q.
```

For the coefficient identity, exponents are `64*c+h'-h`, with `|h'-h|<=63`;
only h'=h reaches the selected multiple of 64, with no negacyclic alias there.
Kcarry is integer, so q*Kcarry vanishes modulo q; eps*barV and E*barV DO NOT.
The rational decomposition explains an integer phase; it is not modular division
by t or a claim `U(x*y)=U(x)*y`.

**Shared selector and actual switch schedule.** Pad F6=F7=0 structurally. Define
the ideal same-phase linear map over R_q, using inverse 64 modulo odd q:

```text
sigma_e(f)=f(X^e),
Pi64 = (1/64)*product_(h in {1,2,4,8,16,32}) (I+sigma_(N/h+1)),
T(F) = sum_(j=0)^5 X^j*Pi64(F_j),
[T(F)]_(64*c+j) = [F_j]_(64*c),  0<=c<128, 0<=j<6.
```

The implemented clear schedule normalizes first, then runs lane-major exponents
1025,513,257 on EACH of eight lanes: 24 automorphisms. Merge h=4,2,1 uses
2049,4097,8193 on respectively 4,2,1 pairs: `24+4+2+1=31`. An encrypted
lowering needs all corresponding real switches, including padded lanes; the
ideal selector alone is not their implementation or their error proof.
For selected logical `(a,b)`, §18.1 specializes exactly to

```text
gamma=floor(b/128), c=b mod 128, j=2*a+gamma, g=0, ell=64*c+j,
Selected = {64*c+2*a : 0<=c<128, 0<=a<3} union {1,3,5}.
```

These are 384+3=387 distinct addresses: residue modulo 64 fixes j and quotient
fixes c; odd lanes have c=0 only. Cropped odd lanes c>0 and other unselected
padding are NOT asserted zero after cryptographic errors. Source 341–343
substitutes Down(R) outside visible cells ONLY because its clear traced phase
is zero (335–338); that shortcut does not transfer to encrypted padding.

**Returned phase, exact decode and separate shares.** Write actual switch/drop
errors xi_nu, their suffix maps L_nu, and fresh-zero phase error zeta. Require

```text
P_ij = T(F) + sum_(nu=1)^31 L_nu(xi_nu) + zeta mod q,
Down(x)=floor((t*(x mod q)+floor(q/2))/q) mod t,
D_ij[a,b] = Down([P_ij+R_ij]_ell),
S_ij[a,b] = -Down([R_ij]_ell) mod t.
```

Each R_ij is a full canonical mask, including all unselected coefficients.
Before existing `Rns3::reconstruct`/exact `down`, the PROPOSED private wrapper
normalizes EACH incoming word modulo its ORDERED q_r; source reconstruct rejects
noncanonical words and does not implement that wrapper. This is data-level
three-limb CRT, not key-level four-limb decoding or upstream approximate scaling.
Source [`CROSS_TERMS` 395–396](../../crates/pllm-core/src/coefficient_layout.rs#L395-L396)
orders `(1,0),(0,1)`, OPPOSITE §18.1's logical descriptor order. Match by explicit
direction, NEVER array ordinal; source 375–378 assigns D to lhs owner and S to peer.
The retained selected party-local assembly (363–380) is therefore

```text
C0 = A0*B0 + D_01 + S_10 mod t,
C1 = A1*B1 + S_01 + D_10 mod t.
```

**H/K attachment, not an instantiated guarantee.** Quantify over all FOUR
canonical matrices, BOTH directions, all SIX blocks and TWO inner blocks, every
error coefficient, every canonical full mask, and all 387 selected cells. Require
the actual phase identities above and compatible integer error lifts with

```text
||E_(a,beta)||_infinity <= Z, ||zeta||_infinity <= Z,
||xi_nu||_infinity <= K, sum_(nu=1)^31 ||L_nu||_(infinity->infinity) <= 469,
H = 2*N*(t-1)*(Z+1) + 469*K + Z.
```

Factor 2 counts inner blocks. Selector preserves selected errors, with NO /64
reduction; 31-switch propagation includes padding, and cropping adds no gain.
Use ONLY §16's existing conditional derivation `Z=13,425,049,620`, `K<2^45`,
`t*H/q<1`; no new constants/profile are adopted. Exact rounding increments then
give circular ±1 ONLY for reconstructed directional `D_ij+S_ij` versus A_i*B_j,
and ±2 ONLY for final reconstructed C0+C1 versus (A0+A1)*(B0+B1), under these
premises. Individual D/S or C0/C1 shares need not be near any dense product.

Finite conditional mapping is closed here; actual candidate encrypted producer
establishing U(A)+E/Z and the COMPLETE switch schedule remains OPEN. Fixed
geometry is not arbitrary-shape, D7/evaluated-profile or whole-model proof.
Prebootstrap prior funding, genuine issuer/principal entitlement, authentication,
disclosure/control atomicity, admission, security and implementation remain OPEN
and unchanged; this NONNORMATIVE appendix grants no execution or adoption.
