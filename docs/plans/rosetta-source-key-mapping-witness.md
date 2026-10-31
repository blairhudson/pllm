# ROSETTA source/key mapping: bounded refusal witness

**Astra67597 / ONEHE-R-MAP; 27 October 2026; baseline `23d691c6f0d353d988bcb405305bf4a9a1674479`.**
Root and `docs/AGENTS.md` read; HEAD matched at the original inspection. Contract only, pending parent's
independent Astra review before implementation. D62 permits reusable layers over
**qualified** general Rust crypto conditionally; current candidate is not qualified.
No backend/pin adoption, runtime adapter, encrypted execution, key generation,
parameter default, build/test result, or completed-method claim follows.

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
missing-file blockers for the six recovered files only.

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

Parent-authorized bounded read-only web research by **Astra12** recovered six
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
`mu_polygen` filling `-mu`, and ARI doubling `mu` plus adding after PBS do not
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
| encrypted affine BR/trace | `P_s=Rescale_q1(C_x*sum_i RNE(q1*2^8*aHat_si)Y^i + sum_i RNE(q1*2^44*bHat_si)Y^i)`; Q1/36→q0/44, rank1/degree1 under S. `BR(P_s,I)=Y^-I*P_s`, trace each, gather `sum_s Y^s*c_s`, BR by S, trace. Client BR material relates L→S; [P1/P2] instead takes **GGSW index bits under S**. U(PHASE-TO-BITS), scale/error and gadget schedule; external products/CMux are not free tensor/relin steps. Trace keys as above at q0. Public LUT replacement [P4] invalid. |
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
Small next user decision, if source cannot supply them: explicitly commission a
separately named alternative trace/index/recovery derivation with fixed endpoints,
or hold mapping. No new primes, fitting coefficients or adaptation defaults here.

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
