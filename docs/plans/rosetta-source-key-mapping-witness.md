# ROSETTA source/key mapping: bounded refusal witness

**Astra67597 / ONEHE-R-MAP; 27 October 2026; baseline `23d691c6f0d353d988bcb405305bf4a9a1674479`.**
Root and `docs/AGENTS.md` read; HEAD matched at the original inspection. Contract only, pending parent's
independent Astra review before implementation. D62 permits reusable layers over
**qualified** general Rust crypto conditionally; current candidate is not qualified.
No backend/pin adoption, runtime adapter, encrypted execution, key generation,
parameter default, build/test result, or completed-method claim follows.

**4 November 2026 — D65:** separately labelled, independently parameterized
odd-modulus specification study approved, preserving paper algorithm, ownership
and security targets. Derive exact parameters and obtain independent review;
separate adoption remains required before implementation. No modulus/backend/
profile adoption or implementation/execution grant; original funding, custody,
source, admission, Native HOLDs and caps remain unchanged.

## 1. Historical local source boundary and reusable owners

Poulpy checkout `~/.cargo/git/checkouts/poulpy-66ed1d416543fd59/ca35325/`
has HEAD→master→`ca3532570ff62b96599a74f7adb85de7726b0446`.
At the original 27 October 2026 local-only inspection, API/reference files below
were read locally; checkout ref is not a new whole-tree integrity/security
attestation. In that inspection, immutable links were locators, not fetches.
HE3DB `1d8d7827baf32b4094b7547aeeaa9faf59253b26` and Lattigo
`37a34b4676e987795949a795f0ce9d0893763c96` source were **not found** in repository,
`target/`, workspace or approved OpenCode temporary-root filename searches;
Lattigo also absent from searched local Go module cache. No network was used in
that historical search. **SOURCE-MISSING-H / SOURCE-MISSING-L** described that
local-only scope, not current global source availability; §1.1 supersedes those
missing-file blockers for the recovered files listed below.

Existing [F6](../../paper/reproduction-contracts/encrypted-inference.md#f6--rosetta-requires-blind-rotation-of-encrypted-accumulators-not-generic-public-lut-pbs)
and `benchmarks/research/torus_comparison.py:3–28` remain secondary leads only.
The original missing-file list included the comparison and blind-rotation files
now recovered in §1.1. Their called TFHEpp parameter/key/bootstrapping and
ring/RLWE/RGSW/CKKS recovery dependencies are still not all recovered or qualified.
Research oracles are not
runtime dependencies/adoption and cannot certify source-evaluated parameters.

Reuse `crates/pllm-core/src/coefficient_layout.rs:1–11,29–74` (clear odd-ring
algebra, degree≤4096), coefficient conversion/repacking only within their actual
contracts, and compiler `encrypted_level_schedule.rs` / shared SSA and trace bridge.
None supplies encrypted ring conversion, sampling, custody or a noise proof.
Reuse native `staged_storage`, `staged_preparation`, `staged_identity` admission/
owner boundaries; do not create paper-specific parallel ledgers or samplers.

### 1.1. Source-recovery addendum — 27 October 2026, after local-only inspection

Original parent-authorized bounded read-only web research by **Astra12** recovered six
pinned source files: eight HTTP bodies, 53,255 bytes, six source-file HTTP 200
responses. This addendum records that review, not a new fetch by this author.
No copied upstream source/archive, backend/pin adoption, runtime, parameter,
RNG authority or cryptographic security qualification follows.

HE3DB comparison files at [the pinned directory][H] have reviewed blob identities:

| File | Blob identity |
| --- | --- |
| `extract_msb.cpp` | `48d0bbd8360f26988dc05db75fc44597dcdbe837` |
| `tfhepp_utils.cpp` | `74495a7b9b55f79826585bc05af58aac2992ea6f` |
| `tfhepp_utils.h` | `384bd91ab842052cdec0dc6e330a9b44d92956e5` |
| `HomCompare.h` | `1101ad25250a8f0e77cfec2f68002c232745e4b0` |

The source phase is `b-<a,s>`, requiring mask negation versus this proposal's
plus convention. `greater_than(A,B)=MSB(B-A)`; `>=` is `NOT(MSB(A-B))`, with
optional `LOG_to_ARI`. `HomMSB` supports ≤10 plaintext bits at lvl1 and ≤33
via lvl2→lvl1; comparison requires `plain_bits+1`. Actual paths include
lvl1→lvl0→lvl1, lvl2→lvl0→lvl2 and improved lvl2→lvl1, not a proof of the
proposed named L→same-L return. PBS offsets `2^(w-6)` / `2^(w-7)`,
`mu_polygen` filling `-mu`, and `MSBGateBootstrapping` arithmetic mode doubling `mu` plus adding after PBS do not
prove the proposed `RNE(q0/4096)` output encoding.

Lattigo [blind-rotation evaluator][LBR] and [RLWE automorphism evaluator][LA]
were recovered at the exact commit. The latter's actual filename is
`evaluator_automorphism.go` (HTTP 200); `automorphism.go` returned 404.
No Lattigo blob SHA was available; immutable commit URLs bind this provenance.
`Evaluate` resets to public LUT `(sigma_-g(f*X^b),0)` and documents
`f(X)X^(b+<a,s>)`; `BlindRotateCore` retains an encrypted accumulator through
keyed automorphisms and external products with `RGSW(X^s[j])`. The wrapper's
pretransform/key map for this proposal remains UNPROVEN. Modulus switching is
`round(2N*x/Q) mod 2N`, then nonzero even **mask** residues `r` become `r+1`;
the body receives no such odd adjustment. This is not whole-phase Quant4096.
Automorphism key switching occurs **before** `sigma_g`, with source-key direction
`S→sigma_g^-1(S)`: algebraically equivalent to post-action `sigma_g(S)→S`,
but with different error placement, not an interchangeable noise proof.

These source bodies are RECOVERED; remaining blockers are concrete conversion,
key-return, error and dependency contracts. TFHEpp parameter/key and recovery
bodies are not all recovered; the exact encrypted chain remains unqualified.

**Vendored dependency follow-up.** At the same HE3DB pin
`1d8d7827baf32b4094b7547aeeaa9faf59253b26`, Git tree metadata identifies
`thirdparty/TFHEpp` as mode `040000` TREE
`a8d22b5aa2463b719e49b5a98de85ebc1ad4254a`, not a gitlink; no separate
TFHEpp upstream commit is established. Additional recovered bodies are
`include/{params.hpp,cloudkey.hpp,gatebootstrapping.hpp,params/128bit.hpp}` and
`src/gatebootstrapping.cpp` beneath that tree, plus HE3DB
`src/HEDB/comparison/operators.h`. Immutable header locators:
[parameter dispatch](https://github.com/zhouzhangwalker/HE3DB/blob/1d8d7827baf32b4094b7547aeeaa9faf59253b26/thirdparty/TFHEpp/include/params.hpp),
[blind rotation](https://github.com/zhouzhangwalker/HE3DB/blob/1d8d7827baf32b4094b7547aeeaa9faf59253b26/thirdparty/TFHEpp/include/gatebootstrapping.hpp).
Dispatch selects CGGI16 for `USE_80BIT_SECURITY`, CGGI19 for `USE_CGGI19`,
concrete for `USE_CONCRETE`, otherwise `128bit.hpp`; no actual build configuration
is selected. In that DEFAULT UNADOPTED candidate, lvl0 is uint32, n672, k1;
lvl1 uint32, n1024, k1, `mu=2^29`, `alpha=2^-25`; lvl2 uint64, n2048,
k1, `mu=2^61`, `alpha=2^-52`, `l=4`, `Bgbit=9`. Key-switch lvl10/20/21
directions are 1→0/2→0/2→1; no silent 1024→2048 transport follows. Secret
generation, noise convention and full joint-key security remain UNKNOWN;
the filename `128bit.hpp` is not a security certificate.
The header’s `BlindRotate` FFT overload taking a const TRLWE test vector applies
`PolynomialMulByXai` to every component, rather than resetting the accumulator.
Its separate Polynomial overload zero-initializes `res` and populates only the
body component from the public LUT. The `.cpp` FFT scalar wrapper
takes a Polynomial and invokes blind rotation plus `SampleExtractIndex` at index 0,
NOT encrypted continuation. `bkfft[i]` indexes SECRET coordinates, not per-message
index-bit controls. The actual header body quantizer TRUNCATES, while the mask
quantizer adds a half-step; the nearest-only lemma below is NOT its failure law.
`HomNOT` negates all TLWE components. CMUX, multiplication, extraction, key
generation, actual `LOG_to_ARI` output encoding and key return remain unqualified.
No security, profile/backend/API, code, funding, execution or HOLD clearance.

### 1.2. Forward clear-reference adapter accounting — design only

Beside the reusable owners above, Astra author `ses_e7b0710b6ffeWZV3qwlTm8gEgw`
and independent auditor `ses_e7b05bcaaffekPm9KaEkpsYqCF` record necessary accounting,
not approved fees, adapter implementation, reverse conversion or security qualification.
The [conversion kernel](../../crates/pllm-core/src/coefficient_torus_conversion.rs#L115-L122)
quotes `64(n+1)+128`; [actual extraction admission](../../crates/pllm-compiler/src/coefficient_extraction_reference.rs#L175-L195)
costs `(8ℓ+64)n+2128`, including its own attempt32. Proposed forward adapter adds
one attempt32; preflight `P`, initialization `I`, cleanup `Z` and control `C` remain unpriced.
On the **same original ClearInputs ledger**, necessary combined work is
`(8ℓ+128)n+2352+construction+builder+other+P+I+Z+C ≤ L ≤ 65536`.
Ignoring all additional costs gives bare `n≤464` (ℓ=1), `n≤438` (ℓ=2), hence
power-of-two `n≤256`: **necessary bounds, not fit or adoption**.
Frozen encrypted degree2048 **cannot fit this proposed same-ClearInputs-ledger path**;
this is not engine-failure proof or permission to reduce encrypted degree, raise caps
or reset ledgers. Original consumed attempts burn; work never refunds.
Required future order is **begin/burn → debit32 → funded preflight P → validation
→ remaining debit → reservation → allocation**, never P before begin; actual
[attempt burn/debit](../../crates/pllm-compiler/src/clear_input_owner.rs#L529-L537) precedes fallible validation.
[Extraction storage](../../crates/pllm-compiler/src/coefficient_extraction_reference.rs#L189-L195)
retains scratch plus words: `16(n+1)+32768+wrappers`, including two relation words
inside [Buffers](../../crates/pllm-compiler/src/coefficient_extraction_reference.rs#L53-L77).
New output adds `8(n+1)`; complete wrapper/capacity-transition inventory remains unpriced.
Existing extraction double wipes are priced [there](../../crates/pllm-compiler/src/coefficient_extraction_reference.rs#L163-L168),
with [borrowed scratch cleanup then retained output](../../crates/pllm-compiler/src/coefficient_extraction_reference.rs#L224-L240);
that pricing cannot transfer to a hypothetical recursive wrapper wipe.
[Capacity mismatch before initialization](../../crates/pllm-compiler/src/coefficient_extraction_reference.rs#L87-L94)
frees allocated capacity on cleanup; it does **not** prove whole-capacity wiping.
No conversion wrapper implementation exists; future RAII, ordering and wipe contracts
remain **UNPROVEN**. This design/accounting note grants no execution or adoption.

## 2. Frozen proposal, not recovered author identity

D65 approves study, not selection of the proposal constants/scales/knobs below.
Independently derived exact parameters must not be labelled unavailable evaluated
constants or claimed to establish evaluated-source parity.

Scope: **one scalar, four segments ×2048 entries**, full/small degrees65536/2048,
extracted LWE2048 then keyed LWE1024. Endpoint proposal: CKKS level4/scale `2^36`
in and out, standard complex packing with only slot0 valid, all padding zero.
[Round-trip proposal §§2–5](rosetta-reciprocal-round-trip.md) owns exact grid/secants:
`x=X/2^20`, `2^20≤X<16·2^20`, `tau_s=2^s`, `h_s=2^(s-11)`,
`S=sum(j=1..3)[X≥2^20*tau_j]`, `I=floor((X-2^20*tau_S)/2^(9+S))`.
Equality selects right segment/interval; `x=16` rejects, never wraps/clamps.
`a=-1/(l*u)`, `b=1/l+1/u`; nearest-even coefficient scales24/44.
These are proposed numeric identity, not author constants or an adopted profile.

Notation: `R_H=Z_Q[T]/(T^65536+1)`, `R_S=Z_Q[Y]/(Y^2048+1)`, `Y=T^32`;
proposed phase `b+a*s=m+e`, LWE `b+<a,s>`. `Q_l=prod(q0..ql)`;
odd q0≈53 bits, others≈41 bits, all proposed `1 mod 131072`, **exact values unknown**.
H/S/L propose independent binary secrets; E proposes ternary weight192 full-ring
encapsulation. These assignments/distributions are unqualified proposals; do not
borrow upstream defaults. Errors, gadgets, auxiliary primes, joint/cyclic-key
security, complete-mask distributions and circuit privacy remain UNSUPPORTED.

## 3. Edge derivation register; no executable composition

Every row binds modulus, phase, scale/level, rounding/error, key direction,
rotation, tensor rank and relinearization. `U(name)` means concrete UNSUPPORTED
attachment, not zero work/error. Rank here means number of secret-mask polynomials;
tensor degree is separate. Linear operations stay degree1; no ciphertext-square
relinearization is needed for those ideal equations. Actual switch/external-product
decomposition and temporary layouts remain charged and unqualified.

| Edge | Required equation, owners and exact outstanding attachment |
| --- | --- |
| StC→H65536 | Proposed `4→1`, scale36, Q4→Q1; slot0→constant `2^16*X`, padding zero. Rank1/degree1 under H. DFT automorphisms/conjugation need client H Galois keys `sigma_g(H)→H`. U(STC): matrices, exact rotation set, each rescale rounding and full-polynomial error; no relin for ideal linear transform. Poulpy [P0] is torus DFT/recovery, not this RNS schedule or phase-sign adapter. |
| H→S2048 | At Q1/scale36: `m(T)=sum(r=0..31)T^r*m_r(Y)`. Splitting H into 32 cosets gives **rank32**, not independent rank1 S. Client switching `H-coset relation→S` must preserve selected/aligned constant; U(RING-DOWN): complete switching matrix, coset rotations, phase/sign, gadgets/error and output-mask law. Degree1; rank32→1 conversion is key switching, not tensor relinearization. Entry normalized trace follows this edge. |
| Entry trace in S | Q1/scale36/level1, rank1/degree1 under S. Ideal `Tr0(C)=2048^-1*sum(g odd mod4096)sigma_g(C)`. Execution order requires `z=2048^-1*C` on **BOTH ciphertext components FIRST**, before any automorphism/key-switch accumulation. Keys `sigma_g(S)→S`; g=-1 then `5^(2^k) mod4096`, k=0..9, each stage `z=z+KeySwitch_to_S(sigma_g(z))` using preceding z. The ideal sum alone does not prove equivalent execution order: post-dividing fresh key-switch errors differs. No relin. Modular inverse exact; switch-error propagation in this order is U(TRACE-ERROR), not real division by2048. **Direct torus substitution INCOMPATIBLE; stop below.** |
| S→extraction2048 | Companion Q1→q0 is restriction, not divide/rescale: level1→0, scale36 unchanged. Constant LWE vector `(a0,-a2047,…,-a1)` under `vec(S)` preserves `b+<a,s>` algebraically, no rounding/rotation/relin. Nonconstant selection first needs shift. Poulpy [P3] exposes conversion but its raw sample routine copies masks in its own convention; U(EXTRACT-SIGN) and odd-modulus transport, no invented adapter. |
| extraction2048→L1024 | At q0/scale36, client decomposed switch `vec(S)→L`, independent binary L1024. Same scalar phase plus U(LWE-KS-ERROR), including gadgets/rounding. Linear vector2048→1024, no polynomial tensor/relin/rotation. **Never truncate/pad a secret or ciphertext.** [P3] API is not proof of this parameterized switch. |
| L→HomComp/S | Three proposed comparisons of `2*C_X-2*2^16*K_j+2^16`, K_j=`2^20*tau_j`; proposed output phase `beta*bit`, beta=RNE(q0/4096), under named L, level0. Sum would yield S. Proposed decoded **INTEGER-X** error≤1/64 means ciphertext **PHASE** error≤`2**16/64`, **BEFORE comparison doubling**; this bound is not proven. U(HOMCOMP), after §1.1 source recovery: odd-q↔torus transport, mask-sign conversion and comparison complement, plaintext-bit domain, named-key return, PBS/rotation/rounding/noise and output-bit encoding remain unproved, including temporary rank/tensor and **every** internal relin. Recovered HE3DB paths do not establish L→same-L or this output encoding. Generic PBS is not this proof. |
| S→phase index I | Retained C_x at Q1: coefficients `A_s=RNE(q1*q0*(4096/B_s)/2^40)`, `Bidx_s=RNE(q1*q0*(4096/B_s)*(-2*K_s-B_s+1)/2^25)`; B_s=`2^(9+s)`. Affine plaintext multiply then centered nearest rescale q1 gives q0/level0 index phase. BR by S selects coefficient; extract2048 and KS→L1024. Rank1/degree1, no ct×ct relin. U(INDEX): BR keys/conversion/errors, whole-phase `Quant4096`, not component rounding. |
| encrypted affine BR/trace | `P_s=Rescale_q1(C_x*sum_i RNE(q1*2^8*aHat_si)Y^i + sum_i RNE(q1*2^44*bHat_si)Y^i)`; Q1/36→q0/44, rank1/degree1 under S. `BR(P_s,I)=Y^-I*P_s`, trace each, gather `sum_s Y^s*c_s`, BR by S, trace. Client BR material relates L→S; [P1/P2] instead takes **GGSW index bits under S**. U(PHASE-TO-BITS for P1/P2); a direct-phase attachment is a separate unqualified alternative below. Scale/error and gadget schedule remain unqualified; external products/CMux are not free tensor/relin steps. Trace keys as above at q0. Public LUT replacement [P4] invalid. |
| S→H | Embed Y=T^32 at q0/scale44/level0; client switch `S(T^32)→H`. Rank1/degree1, no ct×ct relin; U(RING-UP): rotation/switch schedule, error/rounding, complete H-mask distribution. Embedding occupies proper subspace; it does not restore uniform full-ring masks. |
| H→ModRaise/CtS/EvalMod/restoration | Client H→E at q0, raise0→17 under E, E→H at Q17; scale44. CtS17→13 then EvalMod13→5; `Rescale_q5(C*RNE(q5/256))` gives5→4/scale36, not metadata edit. Linear switches/DFT rank1/degree1; EvalMod tensor growth, exact relin points, rotation keys, coefficients/scales/rounding/errors U(RECOVERY). Period lambda=q0/2^44, target lambda*(t-RNE(t)), t=K+phi/lambda, proposed |K|≤256. [P0] torus bootstrap cannot instantiate this unknown odd-prime approximation. |

**First proven incompatible mathematical edge: entry normalized trace.**
Earlier StC/ring-down mappings are UNSUPPORTED, not asserted compatible. For odd q,
gcd(2048,q)=1; multiplication by2048 has inverse. For torus Z/(2^k), k≥11,
gcd(2048,2^k)=2048: `2048*u≡1 mod2^k` has no solution (left even, right odd).
The trace sum loses information; integer right shift/lift is not its inverse and
needs a new precision/noise/capacity construction. [P0] explicitly uses q=2^k.
Stop composition here; later rows describe obligations, not legal continuation.
Internal CRT convolution primes do not change ciphertext modulus to odd Q.
This excludes direct `Z/(2^k)` substitution in the modular-inverse proof only,
not every separately derived torus trace; no such alternative is supplied here.

Independent later blocker: [P1:6–15,41–59] requires encrypted **index bits**;
[P2:59–85,106–107] preserves/rotates encrypted accumulator and uses CMux but
clears noise metadata (`set_noise(None)`). Supported feature ≠ LWE-phase conversion
or zero noise. [P4:845–861] instead zeroes temporary and initializes public LUT.
Whole-phase Quant4096 needs strict error `<1/(2*B_s)`, uniformly `<1/8192`:
this is a **proposed rotation-cell budget, not achieved or adopted**;
componentwise rounding may incur `(1+||L||_1)/2` cells (≤512.5 for binary1024).
Neither bits interface nor a clear phase oracle supplies missing correction keys.

### D65 ordered normalized-trace error constraint (symbolic)

Primary derivation: Astra22 (83597); independent review: Astra34 PASS (83605).
This symbolic constraint does not qualify an encrypted implementation.
Let `R_Q=Z_Q[Y]/(Y^2048+1)`, `N=2048`, with odd `Q>1` (composite allowed),
and `u=N^-1 mod Q`. Write `P0(f)=f_0` embedded as a constant polynomial.
Use eleven ORDERED stages: `g_1=-1`, `g_(k+2)=5^(2^k) mod 4096`, `k=0..9`,
and `L_j=I+sigma_(g_j)`. Each proper key switch `sigma_(g_j)(S)→S` adds
phase error `eta_j` after applying the automorphism to the preceding ciphertext.
Premultiply BOTH ciphertext components by `u` before these stages:

```text
p_0 = u*(m+e)
p_j = L_j*p_(j-1) + eta_j                     (j=1..11)
prod_(j=1..11) L_j = sum_(g odd mod 4096) sigma_g = N*P0
p_11 - m_0 = e_0 + sum_(j=1..11) (prod_(r=j+1..11) L_r)*eta_j   in R_Q
```

The odd automorphism group is `C2 × C1024`; the listed generators' subset
products enumerate it, giving the trace identity above. The empty product is
identity. Input error PROJECTS to `e_0`; no divisibility assumption on `m` or
`e` is needed. Each `eta_j` is injected AFTER inverse premultiplication and
is NOT divided by 2048. These errors can retain nonconstant coefficients.

Choose integer coefficient lifts with `||eta_j||_infinity≤B_j` and
`|e_0|≤E_0`. Automorphisms are signed coefficient permutations, so each
remaining `L_r` multiplies this norm bound by at most two. Consequently

```text
B_out = E_0 + sum_(j=1..11) 2^(11-j)*B_j
```

bounds an integer representative of the ERROR polynomial. If `B_out<Q/2`,
that representative is the centered error. Message-plus-error wrap, scale and
decoding margins still need separate proof; the bound is not a decode guarantee.
Intermediate `u`-scaled phases need NOT be small: the modular identity remains
valid. A backend must bound `eta_j` for its actual decomposition, intermediate
ciphertexts and randomness, and compose the associated failure probabilities.
Later HomComp/rotation margins and the phase-to-encrypted-bits bridge remain
SEPARATE obligations. No `q`, scale, noise law or other knob is adopted here;
no 128-bit/joint-key qualification, code/execution permission or HOLD clearance.

#### Joint candidate-path constraints

Independently reviewed symbolic constraints. ALL constants remain UNADOPTED
existing proposals.
Write `D=2^16`, `T=4096`, `Q1=q0*q1`; trace phase is `D*X+e_T(Y)`,
with `||e_T||_infinity≤B_out`. Literal component restriction `Q1→q0` entails NO division,
scale change or rounding. Constant extraction uses `(a0,-a2047,…,-a1)`;
dimension-changing LWE key switching adds error bounded by `E_KS`, giving
`E_X=B_out+E_KS`. Ordinary centered interpretation requires
`|D*X|+E_X<q0/2`. A minus-phase interface needs explicit mask negation.

The candidate comparison has phase `D*[2*(X-K)+1]+2*e_X`. Correct ideal sign
requires `2*E_X+E_C<D` and `|D*[2*(X-K)+1]|+2*E_X+E_C<q0/2`, with
comparison-transport error `E_C` in ORIGINAL `q0` phase units. Neither condition
proves HomComp or its output encoding.

Index construction uses the RETAINED `Q1` ciphertext, not the restricted one.
For `B_s=2^(9+s)`, `K_s=2^20*tau_s`, unrounded operands are

```text
A_t* = q1*q0/(T*B_t*D)                       (t=0..3)
C_s* = q1*q0*(-2*K_s-B_s+1)/(2*T*B_s)
delta_A_t = A_encoded,t - A_t*
delta_C_s = C_encoded,s - C_s*
A_poly = sum_(t=0..3) (A_t*+delta_A_t)*Y^t
v = (X-K_s)/B_s - 1/2 + 1/(2*B_s)
```

Choose integer representatives of encoded operands to define these deviations.
After actual rescaling by `q1`, `v` is the ideal CELL phase. The existing
[floor proof in reciprocal §3](rosetta-reciprocal-round-trip.md#3-representations-homcomp-and-floor-without-a-roundingclamp-shortcut)
applies only to the SELECTED segment: `0≤X-K_s<2048*B_s`, integer grid,
`0≤n≤2047`, strict half-cell margin `1/(2*B_s)`, no ties. Other-segment
selection/masking remains UNPROVED. Rotation domain is modulo 4096, NOT 2048:
the latter loses negacyclic sign; `Y^-n` requires the correct sign convention.

CONDITION the bound below on HomComp producing correct encrypted `S` and BR-by-S
actually selecting that coefficient with a proved phase-preserving noise bound.
Wrong `S` is a branch failure, not automatically small `E_downstream` noise;
aggregate `E_cell` proves neither S-selection, HomComp nor the key interface.
Other-segment coefficients require actual selection/masking semantics.

```text
E_cell ≤ (T/q0)*[(|D*X|*|delta_A_s|
                   + B_out*sum_(t=0..3) |A_t*+delta_A_t|
                   + |delta_C_s|)/q1 + E_rescale + E_downstream] + E_mod
require E_cell < 1/(2*B_s)
```

The selected coefficient of `A_poly*e_T(Y)` includes all four SIGNED negacyclic
convolution terms; the triangle bound handles wrap signs without independence.
A scalar-only bound requires proven constant-only error or a single-coefficient
multiplier; neither is established here.

`E_rescale,E_downstream` are ABSOLUTE `q0` phase-unit bounds. Rescale includes
body AND secret-weighted mask rounding; the triangle bound assumes no independence.
Downstream includes actual BR-by-S, extraction and key-switch noise, EXCLUDING
the normalized modulus-switch term counted once in `E_mod` (CELL units).
Nearest componentwise LWE `q0→T` rounding alone gives
`E_mod≤(1+||s_L||_1)/2`. The recovered Lattigo helper ALSO odd-adjusts masks:
its additional actual secret-weighted error is NOT covered by that nearest-only
bound. If used, `E_mod` must bound its actual TOTAL, not just nearest rounding.

Larger `q0` reduces only fixed absolute errors normalized by `T/q0`; actual noise
may depend on `q0`. Normalized modulus-switch cells do not shrink that way.
A larger `q0` alone proves neither the required quantization nor a compatible encrypted-rotation key interface.
No manifest, `q`, scale, error law, backend/profile, 128-bit/security, code,
execution or funding adoption, nor HOLD clearance, follows.

#### Conditional rotation attachment families and precision transport

Two legitimate attachment families must be distinguished. Direct LWE-phase blind
rotation targets `Y^(b+sum_j a_j*s_j)*C` using preissued SECRET-coordinate
evaluation material (monomial encryptions in the inspected Lattigo form).
The `a_j,b` are public ciphertext data, not plaintext
index bits. A minus-phase interface requires negated mask actions; the desired
`Y^-I` orientation must also be explicit. This route does NOT mandate per-output
GGSW conversion or circuit bootstrapping. The bit-controlled route instead uses
`I=sum_k 2^k*i_k` and `CMUX(GGSW(i_k),Y^(-2^k)*C,C)`; it needs an actual
TLWE-message-bit-to-gadget-control conversion. The routes agree only for the same
index modulo 4096 with compatible keys and errors. Lattigo `Evaluate` resets a
public LUT, not a continuation; its lower core remains an unqualified candidate.
The coarse-switch obstruction below also applies to a direct route using that
coarse switch, under the lemma's assumptions, not the actual TFHEpp quantizers.

For a conditional precision-preserving candidate, retain the correct-segment
conditions above. Let `B_s=2^b`, `b=9..12`, `d=X-K_s=n*B_s+r`,
`0≤n<2048`, `0≤r<B_s`, `T=4096`, `F=8192`, and

```text
v = d/B_s - 1/2 + 1/(2*B_s)
M = F*(v+1/2) = 2^(13-b)*d + 2^(12-b)
0 < M < 2^24; floor(M/F) = n
```

Thus bits 13..23 of integer `M` encode `n`. Hypothetically switch components
nearest from `q0` to auxiliary `Q=2^64` with the SAME integer secret and
consistent representatives: a phase winding `k*q0` maps to `k*Q`. The lift error
obeys `E_lift≤(1+||s||_1)/2` in TARGET `Q` units. A changed key or ciphertext
type requires a separately qualified conversion. Add exact public
`Q/(2*T)=2^51`; the phase becomes `(Q/2^25)*M+e`.

Under the EXPLICIT GENERIC contract `Enc_p(m)=Q*m/2^p+noise`, with `MSB_p`
returning bit `p-1`, multiplying the input by `2^(24-j)` and taking `p=j+1`
would return bit `j` of `M`, `j=13..23`. Absolute errors multiply too; their
ratio to the encoding unit stays fixed. Actual HE3DB encoding, padding, offsets,
error acceptance, internal noise and output key are NOT established by this
generic contract or the API's ≤33-bit ceiling. Require the conditional bound

```text
delta_M ≤ F*E_pre_cell + (2^25/Q)*(E_lift+E_extra_key) < 1/2
```

Here `delta_M` is absolute input-phase error in `M` units. `E_pre_cell` is absolute
PRE-transport index-phase error in `T/q0` cell units, excluding the old coarse
`q0→T` `E_mod`; do not count it twice. `E_extra_key` is measured at the
PRE-multiplication `Q` scale; later events need their own normalized bounds.
This imposes the stronger nearest-`M` cell margin
`1/(2*F)`, not an achieved HomMSB performance or rotation-control construction.
The main-ring study remains odd-modulus; `Q=2^64` is hypothetical auxiliary
transport, not a whole-profile replacement. All constants and route choices here
remain UNADOPTED.

#### Recovered IdentityKeySwitch and conditional HomMSB staging

At HE3DB `1d8d7827baf32b4094b7547aeeaa9faf59253b26`, primary-source audit
recovered `thirdparty/TFHEpp/src/keyswitch.cpp::IdentityKeySwitch`;
`include/keyswitch.hpp` contains declarations only. Source-evidenced DEFAULT
UNADOPTED lvl21 parameters are $d=64,w=32,t=10,B=3$, dimension $2048\to1024$;
these are not security/profile qualifications. Output starts zero. With
$Q_{64}=2^{64},Q_{32}=2^{32}$, body rounding is unsigned half-up modulo $Q_{32}$:

$$
B'=((b+2^{31})\bmod Q_{64})\mathbin{\gg}32,\qquad
\bar a_i=(a_i+2^{33})\bmod Q_{64},\quad
v_{ij}=(\bar a_i\mathbin{\gg}(64-3(j+1)))\mathbin{\&}7,
\quad j=0,\ldots,9.
$$

Masks decompose DIRECTLY at $Q_{64}$, not after $Q_{32}$ rounding. Each nonzero
digit subtracts ALL components of `ksk[i][j][v_ij-1]`, without extra digit
multiplication. Define minus-phase $\phi_q=b-\langle a,s\rangle\pmod q$.
The generation message and phase sign are now source-evidenced at the same pin:
[`src/cloudkey.cpp::ikskgen<P>`](https://raw.githubusercontent.com/zhouzhangwalker/HE3DB/1d8d7827baf32b4094b7547aeeaa9faf59253b26/thirdparty/TFHEpp/src/cloudkey.cpp)
passes $s_i v_{ij}2^{32-3(j+1)}$ to
`tlweSymEncrypt<targetP>(..., P::α, targetsecret)`.
[`src/tlwe.cpp`](https://raw.githubusercontent.com/zhouzhangwalker/HE3DB/1d8d7827baf32b4094b7547aeeaa9faf59253b26/thirdparty/TFHEpp/src/tlwe.cpp)
forms the body as `ModularGaussian(message, α) + Σ mask*targetsecret`:
this is raw-torus encryption, not Δ-scaled `IntEncrypt`. Thus each selected
entry, conditional on defined sampler execution and the resulting entries, has
minus-phase $s_i v_{ij}2^{32-3(j+1)}+\epsilon_{ij}$ under $s'$
modulo $Q_{32}$, with $\epsilon_{ij}$ defined as the sampler residual; no law
or independence is asserted. Consequently,

$$
\widehat a_i=\sum_{j=0}^{9}v_{ij}2^{64-3(j+1)},\qquad
\phi_{32}\equiv\frac{\phi_{64}}{2^{32}}+\rho_b+
\sum_i s_i\rho_i-\sum_{(i,j):v_{ij}\ne0}\epsilon_{ij}\pmod{Q_{32}},
$$

using consistent cyclic lifts
$\rho_b=B'-b/2^{32}$, $-1/2<\rho_b\le1/2$, and
$\rho_i=(a_i-\widehat a_i)/2^{32}$, $-2\le\rho_i<2$.
All errors here are $Q_{32}$ word-phase units; old $e_{64}$ scales by $2^{-32}$.
Selected-entry errors enter with a MINUS sign. The default candidate's
$\alpha=2^{-25}$ gives nominal $Q_{32}$ word scale 128, not achieved noise or a
tail bound. At this same pin, [`include/utils.hpp`](https://raw.githubusercontent.com/zhouzhangwalker/HE3DB/1d8d7827baf32b4094b7547aeeaa9faf59253b26/thirdparty/TFHEpp/include/utils.hpp)
lines 35–38/50–64 recover `dtot32`/`ModularGaussian`: the $Q_{32}$ path draws
`std::normal_distribution<double>(0., stdev)` and converts with
`int32_t(int64_t((d-int64_t(d))*(1LL<<32)))`, then adds modulo the unsigned word.
Floating-to-integer conversion truncates toward zero, NOT nearest; floating-to-
`int64_t` requires representability, and an unrepresentable `int64_t` to `int32_t`
conversion is C++17 implementation-defined. The $Q_{64}$ path draws standard
normal `(0,1)`, forms `val = stdev * draw * 2^64`, then `static_cast<P::T>(val)`
for `uint64_t`. A negative TRUNCATED integer or other unrepresentable floating-to-
`uint64_t` result is undefined, not portable modular conversion; `-1 < val < 0`
instead truncates to zero. This is a source-definedness gap, not observed wrong
output, failure probability, attack or universal backend defect. It does not
automatically change the conditional $Q_{32}$ IKS sign algebra above.

`include/utils.hpp:17–22` declares header `static thread_local std::random_device
trng`; under `USE_RANDEN`, `randen::Randen<uint64_t> generator(trng())`, otherwise
the generator itself is `std::random_device`. These have internal translation-unit
linkage and thread-local state; no explicit fork reseed is shown. No OS-entropy,
IID or independence claim follows. [`CMakeLists.txt:3,8,21–23`](https://raw.githubusercontent.com/zhouzhangwalker/HE3DB/1d8d7827baf32b4094b7547aeeaa9faf59253b26/thirdparty/TFHEpp/CMakeLists.txt)
requests C++17 and defaults Randen ON; deployed configuration/version remains
unqualified. These findings and the key mechanism below inherit two independent
read-only reviews of four raw files each: eight GETs, 17,790 response bytes, all
HTTP 200, no redirects, retries or failed requests. Joint tails/security and
actual build behavior remain unqualified; no new acquisition or execution here.

For candidate phase $Q_{64}M/2^{25}+e_{64}$, target bit $j=13,\ldots,23$
uses $p=j+1$: multiply ALL ciphertext words by $2^{25-p}$ before `HomMSB(p)`.
This means left-shift **25-p bits**, not shift by the exponentiated value.
Original stage phase is $Q_{64}M_t/2^t+E_t\pmod{Q_{64}}$;
$E_t,\eta_S,\eta_R$ are $Q_{64}$ WORD-PHASE units, distinct from preceding
$Q_{32}$ key-switch errors. Initially $t=p$, $M_p=M\bmod2^p$,
$E_p=2^{25-p}e_{64}$.
At width $t$ with current message $M_t$ and error $E_t$, improved staging
multiplies by $2^{t-6}$ and requires arithmetic-MSB phase
$Q_{64}\operatorname{bit}_5(M_t)/2+\eta_S$.
After subtraction, identity-PBS scale $64-t$ MUST return
$Q_{64}(M_t\bmod32)/2^t+\eta_R$.
Subtracting that result from ORIGINAL stage input gives message
$\lfloor M_t/32\rfloor$, width $t-5$, error $E_t-\eta_R$.
Final $Q_{32}$ staging peels four low bits. First combined shift is
$(25-p)+(p-6)=19$ bits: factor $2^{19}$ also multiplies $e_{64}$.

Helpers add body-word offsets $2^{57}$ at $Q_{64}$ or $2^{26}$ at $Q_{32}$
(normalized torus offsets $2^{57}/Q_{64}$ and $2^{26}/Q_{32}$). LUT padding means
COEFFICIENT REPETITION, not dimension padding. All $\eta$ include intervening
operations; no independence is assumed. $\delta_M<1/2$ proves neither actual PBS
acceptance nor key-switch numerics. Generic bit ceiling proves no exact index,
control or key return. D65 permits conditional study only: no profile, keys,
noise law, funding, security, implementation, execution or HOLD clearance.

#### Conditional HomMSB block margins and helper-error propagation

Let $T=2N$ and use the phase-oriented index
$k=\lfloor Tb/Q\rfloor-\sum_i s_{0,i}\operatorname{HU}(Ta_i/Q)$,
where HU is half-up rounding. The raw rotation exponent has the opposite sign;
the LUT orientation must match that sign rather than silently identifying it
with $k$. CONDITIONALLY on a binary bootstrap secret of weight $h$, consistent
cyclic lifts give $\zeta=k-T\phi/Q$ with $-1-h/2<\zeta\le h/2$.
This is a deterministic quantizer bound, not an IID-mask assumption.

Let $u$ be input error AFTER the helper's input key switch, measured in $T$
cells; it excludes both public centering and the floor/HU error $\zeta$.
The $Q_{64}$ six-bit block has length $L=T_{64}/64$ and public offset
$Q_{64}/128$; the $Q_{32}$ five-bit block has $L=T_{32}/32$ and offset
$Q_{32}/64$. Conservative sufficient block conditions are

$$
1+h/2-T_{64}/128\le u_{64}<T_{64}/128-h/2,\qquad
1+h/2-T_{32}/64\le u_{32}<T_{32}/64-h/2.
$$

Default candidate $N_2=2048,N_1=1024$ makes both conditions
$h/2-31\le u<32-h/2$. At the same pin,
[`src/key.cpp:8–17`](https://raw.githubusercontent.com/zhouzhangwalker/HE3DB/1d8d7827baf32b4094b7547aeeaa9faf59253b26/thirdparty/TFHEpp/src/key.cpp)
separately zero-fills lvl0/1/2, then
[`include/key.hpp:38–49`](https://raw.githubusercontent.com/zhouzhangwalker/HE3DB/1d8d7827baf32b4094b7547aeeaa9faf59253b26/thirdparty/TFHEpp/include/key.hpp)
performs 512/512/1024 indexed draws WITH REPLACEMENT setting entries to 1, using
a fresh `std::random_device`-seeded `default_random_engine` per call. This recovers
a binary alphabet, NOT exact fixed weight or IID Bernoulli, nor a joint/level-
independence proof. For a successful UNCHANGED constructor with the retained
`n=672` candidate, the hard bound is $h_0\le512$. Its uniform constructor-wide
sufficient certificate substitutes $h=512$, giving EMPTY $[225,-224)$.
An upper bound does NOT establish actual weight, an empty actual interval or
protocol failure. Imported/deserialized/subsequently mutated keys require
independent provenance; recovered construction does not authenticate actual keys.

For a $Q_{64}$ stage of width $t$, propagate helper inputs and the residue as

$$
u_S=\frac{T_{64}}{Q_{64}}(2^{t-6}E_t+\kappa_S),\qquad
u_I=\frac{T_{64}}{Q_{64}}(2^{t-6}E_t-\eta_S+\kappa_I),\qquad
E_{t-5}=E_t-\eta_R.
$$

Here $\kappa_S,\kappa_I$ are input-referred switch errors in $Q_{64}$ word
units, not the switch-to-index error $\zeta$ counted again. $\eta_S$ is the
arithmetic-output total error; $\eta_R$ is the identity-output total error
relative to the ideal residue, including its internal operations, not added
again. The $Q_{32}$ analogue uses $T_{32}/Q_{32}$, $2^{r-5}$ and
$E_{r-4}=E_r-\eta_R$, with all errors in $Q_{32}$ word units.
The first caller amplifies $e_{64}$ by $2^{19}$. Despite the recovered binary
constructor mechanism, actual-key provenance, joint secret/noise laws and actual
input-key-switch/LUT/PBS error bounds are still needed; $\delta_M<1/2$ alone is
insufficient. No probability qualification, profile/backend adoption,
execution, security, funding or HOLD clearance follows from this D65 study.

#### Conditional coarse-switch failure lemma

Let `q` be positive odd, `T=4096`, and fix a binary secret of weight `1..1024`.
Assume masks `a_j` are IID uniform in `Z_q`, independent of fixed `m,s`, and
`b=sum_j a_j*s_j+m mod q`. CONDITION on an actually feasible lift
`v=T*m/q=n0-1/2+epsilon`, `0<epsilon≤1/1024`, `n0=0..2047`.
This does not assert that the proposal's exact dyadic `v_s` is realizable at odd
`q`. Write `D_int` for the unwrapped integer below (not the earlier `D=2^16`):

```text
d = RNE(T*b/q) - sum_j s_j*RNE(T*a_j/q) mod T
  = RNE(v-E) mod T; D_int = RNE(v-E)
E = sum_(j active) [RNE(T*a_j/q)-T*a_j/q]
Pr[d != RNE(v) mod T] ≥ 1/2 - epsilon - 1/(2*q)
```

Since `gcd(T,q)=1`, summand errors are IID uniform on
`{h/q: h=-(q-1)/2..(q-1)/2}`. Ties are absent for odd `q`; convolution keeps
`E` symmetric with maximum atom ≤`1/q`. Symmetry and counting lattice points
give `Pr[E>epsilon]≥1/2-epsilon-1/(2*q)`. On this event `D_int≤n0-1` and
`|D_int-n0|<513<T`, so modulo `T` cannot hide the mismatch. No LWE noise is needed.
These assumptions do NOT establish IID masks after actual extraction/key switching
or an attack on actual ROSETTA/HE3DB. The lemma rejects the shortcut, not all blind
rotation. No new manifest, parameter, backend/profile, security, funding, source-
authoring or execution permission, or HOLD clearance, follows from these constraints.

## 4. Custody and preconstruction admission

Client alone generates/holds secrets, samples keys and originates immutable export
epoch. Evaluator gets only evaluation material; no intermediate decrypt/re-encrypt.
Equal serialized bytes, digest, layout tag or reparsing cannot mint client custody.
`staged_identity.rs:440–462` refuses MissingTrustedClientExportEpoch;
`rejectMissingTrustedClientExportEpoch` remains required refusal, not an export seam.
D48 original Layout proves owning capacity/alignment **only**; D50/D53 sampling
designs are not adopted Main. No whole-backend crypto qualification from layout,
quarantine imports, deterministic seeds or successful clear-oracle arithmetic.

Retain original immutable evaluator backing for every raw/prepared key/context and
all dependent ciphertexts, views and in-flight operations. No reconstruction by
equal bytes. Before allocation, parse or factory call, one cumulative root ledger
must reserve work and complete owners: secrets/client generation/export, transport,
raw+prepared overlap, ring cosets, all four accumulators, LWE, scratch, DFT/NTT,
expanded recovery RNS buffers, allocator arenas and final decoding. Include retry/
attempt cost, cancellation, failed release and release verification. Unknown refuses;
new child/request ledger cannot reset root spent work or disguise retained backing.
Reserve output+scratch before retiring inputs; failed release stays charged.
`staged_preparation.rs:223–265` admits before prepare, but explicitly starts with
existing Module/raw/output/scratch: **full preconstruction bound remains open**.
Historical proposal scope was ≤8 GiB aggregate ceiling, host reserve≥4 GiB,
zero new swap; that 4 GiB reserve is NOT the current execution gate. D59 current
campaign authority requires ≥16 GiB host reserve, ≤8 GiB aggregate process-tree
sampled RSS with automatic stop, stop on swap growth, ≤15 minutes per command,
≤2 hours per campaign and ≤20 GiB temporary disk, one native job globally, and
all tighter existing guards. Fresh grants and preflight remain required. This
proposal is neither an execution grant nor hard containment; sampled monitoring
is not hard OS containment. Ceilings are not proof of fit. Complete work/owner
bounds are still UNKNOWN.

## 5. Regression definitions and acceptance (not executed)

| Witness | Required rejection/check |
| --- | --- |
| Wrong modulus | Even torus substituted for odd trace limb rejects before construction; verify no inverse2048 exists. Wrong exact prime manifest rejects even at equal bit width. |
| Wrong key/epoch | Swap H/S/L direction, raw/prepared family or client epoch: reject; equal bytes cannot upgrade untrusted export. |
| 2048→1024 truncation | Drop final1024 mask/secret coordinates: reject missing client KS; full original relation must remain named. |
| Public LUT overwrite | Encrypted x-dependent accumulator routed to [P4] rejects; retained input mask/message cannot become trivial LUT. |
| Index equality/neighbors | Define all 8192 local boundaries and segment boundaries with grid predecessor/equality/successor; right-side equality, valid I∈[0,2047], tau_end/off-grid reject. No noisy clamp. |
| Padding | Final client-only decode checks all zero-padding slots, not only slot0; whole-error-polynomial propagation required. |
| Scale-only recovery | Increase level/scale metadata or ModRaise alone: reject absent CtS/EvalMod and charged restoration. |
| Fresh-ledger bypass | Split identical work among new sessions/children/retries: root cumulative reservations persist; unknown attempt cost refuses. |
| Retain/release | Keep view or dependent operation alive while dropping original raw/prepared/context handle: backing stays charged/alive; failed release cannot restore budget. |

Acceptance: independent reviewer traces **each edge** to actual pinned primitive
and numeric/key/owner attachment, or records concrete UNSUPPORTED/INCOMPATIBLE
edge. After §1.1 recovery, current result remains a **bounded source-resolution
refusal witness, not HE execution**; no HomComp output encoding is adopted.
Exact primes, HomComp and recovery approximation are not recovered author identity.
D65 now commissions the separately labelled, independently parameterized odd-modulus
specification study; study authorization is no longer pending. Derive exact manifest
and obtain independent review, then separate adoption before implementation. Existing
endpoints/scales/knobs remain unselected; no new primes, fitting coefficients,
adaptation defaults or evaluated-source parity are adopted or established here.

Shared DAG: `secureSource + originalLayout/export + completePreconstructionBound`
→ independently qualified arithmetic/rotation/relin/refresh → NEXUS dual-key,
THOR deferred schedule, CacheMir KV state, ROSETTA conversion. Method-specific
security/numerics stay separate. BumbleBee BFV is **not CKKS-refresh-dependent**.
Next mathematically supported neutral unit, after review/separate authorization:
extend existing bounded coefficient-layout regression coverage for odd-limb
inverse2048/trace and explicit even-modulus refusal; no HE adapter or large ring.
Genuine blocker for next encrypted unit remains missing trace construction,
phase-to-index conversion, remaining pinned dependency/key/error attachments,
custody and complete bound. Source inspection is not whole-method execution.

[P0]: https://github.com/poulpy-fhe/poulpy/blob/ca3532570ff62b96599a74f7adb85de7726b0446/docs/bootstrapping.md#L33-L51
[P1]: https://github.com/poulpy-fhe/poulpy/blob/ca3532570ff62b96599a74f7adb85de7726b0446/poulpy-bin-fhe/src/api/bdd/glwe_blind_rotation.rs
[P2]: https://github.com/poulpy-fhe/poulpy/blob/ca3532570ff62b96599a74f7adb85de7726b0446/poulpy-bin-fhe/src/reference/bdd/glwe_blind_rotation.rs
[P3]: https://github.com/poulpy-fhe/poulpy/blob/ca3532570ff62b96599a74f7adb85de7726b0446/poulpy-core/src/reference/conversion.rs#L30-L96
[P4]: https://github.com/poulpy-fhe/poulpy/blob/ca3532570ff62b96599a74f7adb85de7726b0446/poulpy-bin-fhe/src/reference/blind_rotation/algorithm.rs#L845-L861
[H]: https://github.com/zhouzhangwalker/HE3DB/tree/1d8d7827baf32b4094b7547aeeaa9faf59253b26/src/HEDB/comparison
[LBR]: https://github.com/tuneinsight/lattigo/blob/37a34b4676e987795949a795f0ce9d0893763c96/core/rgsw/blindrot/evaluator.go
[LA]: https://github.com/tuneinsight/lattigo/blob/37a34b4676e987795949a795f0ce9d0893763c96/core/rlwe/evaluator_automorphism.go
