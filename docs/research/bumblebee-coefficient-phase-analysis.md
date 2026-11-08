# B-S/ANALYZE-1 — finite source-only design delta

**Disposition:** the twelve-case analysis is complete; B-S/Q and numerical/backend adoption remain closed. This is symbolic analysis of the unchanged proposal, not executed arithmetic, a backend qualification, an estimator result, or evaluated-author evidence. In particular, the proposed coefficient product has a large deterministic scaling-carry discrepancy: a universal one-bit reconstruction claim cannot follow from this graph.

## Scope, sources and unchanged identity

Authority: user-approved ANALYZE-1, `docs/plans/bumblebee-executable-profile-d37.md`, section D (D). Its Attachments A/B remain **PROPOSED**. Candidate remains `bumblebee.final-paper-derived-gpt2-mixed.v1`; primitive candidate remains `bumblebee.odd-rns-primitives.fhers-98a087f.v1`. No checkpoint, source mapping, sampler, modulus, numerical profile or security target is changed.

All arithmetic below is in `Z[X]/(X^8192+1)` unless stated scalar. `can_m` is coefficientwise canonical reduction; `rnd(z)=floor(z+1/2)`. Set:

```text
N = 8192; t = 2^64; B = 2^20
OLT data limbs = [562949952847873, 562949952798721, 562949952700417]
p = 562949952274433
q = 178405960980747204108026560376635421240770561
Q = q*p = 100433627219585994566577152443730065352832756315331459366913
bOLE data limbs = [1125899906826241, 562949952847873, 562949952798721]
q_b = 356811922413223917036199361012173016050532353
(t_1,t_2,t_3) = (35184371613697,35184371417089,35184371138561)
T = 43556140369735457451271451648667273920513
Galois indices = [257,513,1025,2049,4097,8193]
```

Ordered key limbs are OLT data limbs followed by p. Eight distinct secrets: two OLT owners, six bOLE owner/context keys. OLT public/evaluation keys use **196-bit Q**, with data public keys obtained by CRT restriction, not message rescaling. Twelve logical Galois keys contain eight rows each. Secrets/ephemerals remain iid ternary; errors remain CBD(20); flooding remains uniform on `[-2^97,2^97)`; rejection limit remains 128 attempts. Targets remain 128-bit classical computational security and total statistical distance at most `2^-40`, neither established.

Local evidence read:

- **D:** the D37 packet above, including B2.1/B3/B4/C1.
- **P:** `papers/r22-bumblebee-final.pdf`, all 18 pages, especially pp.3, 5–8, 16–18. Locked digest recorded by D: `184bbe5104f183d4df66db2413f89c96c6c814e0b0ec6297f19745a1fa6e203b`. No new digest verification was executed. The paper's functional/hybrid statements do not certify the proposed concrete samplers or tuples.
- **L:** `docs/plans/bumblebee-indexing-corrections.md:14–87`; `crates/pllm-core/src/coefficient_layout.rs:149–211`.
- **C:** `crates/pllm-core/src/coefficient_conversion.rs:11–149`; `paper/reproduction-contracts/shared_profile_oracle.py:108–182`.
- **S:** `crates/pllm-compiler/src/encrypted_level_schedule.rs`, public trace/owner declarations and validator locator.
- **F:** `paper/methods/rust-he-backend-sources.json:89–102` and `rust-he-backend-qualification.md` F1–F3 references.

**Cache limitation:** the existing cache at `.../opencode/rust-he-qualification-20261017/fhe-rs/crates/fhe/src/` is empty. No pinned backend source bytes were available there; no fetch or substitute version was used. F records `tlepoint/fhe.rs@98a087f1bf5f7fa95196ebe16b0e1b0eeb632442`, but its historical export/version observations cannot establish any exact API edge below. UNKNOWN means unverified here, not proven absent upstream.

Budget: one reviewer/pass; nine existing document/source/manifest files plus one 18-page PDF (ten files total), below 32 files/16 MiB text; twelve cases, sixteen API rows, eight OT rows, one Markdown artifact below 128 KiB. No executable experiment, tests, builds, dependency resolution, network, entropy, key generation, native imports/HE or model execution. No Main/index/source edits. The interrupted initial artifact tool call wrote nothing; this document is the sole deliverable.

## Twelve fixed symbolic cases

### 1. Raw public-key encryption phase and correlations

For the proposed raw public key `(b,a)` with `b+a*s=e_pk (mod m)`, encryption of coefficient message A is `(b*u+e0+Delta*A, a*u+e1)`, where `Delta=floor(m/plaintext_modulus)`. Its phase is exactly

`Delta*A + E_enc (mod m)`, with `E_enc=e_pk*u+e0+e1*s`.

The negacyclic convolution inequality gives

`||E_enc||inf <= 20||u||1 + 20 + 20||s||1 <= 40N+20 = 327700 = C_enc`.

This support bound requires no independence. Fresh u/e0/e1 do not make encryptions independent after retaining the same secret/public-key error. Reused ciphertexts retain the identical error polynomial; repeated key-switch rows retain identical row errors. **Outcome: exact phase and conditional sampler-support bound closed; backend realization UNKNOWN.** [D B3/B4; P p.3]

### 2. Full-share coefficient product, including x=y=t-1

Write the actual integer product of canonical packed message polynomials as `H=x*y=M+t*K`, `M=can_t(H)`. For `r_q=q-t*floor(q/t)`,

`Psi_product = Delta*M + E_enc*y - r_q*K (mod q)`.

The scalar endpoint gives `H=(t-1)^2`, `M=1`, `K=t-2`, hence carry error **`-r_q*(t-2)`**, not zero. This is a full-width share calculation, unrelated to a bound on the reconstructed Q18 activation.

The unchanged tuple also permits a useful exact bound on its remainder without a sweep. Put `a=(573439,622591,720895)`, so its three data primes are `2^49-a_i`. Their pair-product sum is 3 modulo `2^15`; consequently

`r_q = t + 3*2^49 - a_1*a_2*a_3`, and `15t/16 < r_q < t`.

Indeed each a_i is between `2^19` and `2^20`, and the negative remainder before adding t lies strictly between `-t` and zero. **Outcome: exact carry closed; large endpoint discrepancy is real, not an encryption-noise estimate.** [D B1/B3]

### 3. Complete inner blocks and additional carries

For the two inner blocks in each of C1's six output blocks, let `H_b=x_b*y_b=M_b+t*K_b`. If `sum_b M_b=M+t*K_add`, then

`Psi_sum = Delta*M + sum_b(E_b*y_b) - r_q*(sum_b K_b+K_add) (mod q)`.

Equivalently use `H_total=sum_b H_b=M+t*K_total`; this includes every block carry once. A support-only bound is

`||E_sum||inf <= sum_b C_enc*||y_b||1 + r_q*||K_total||inf`,

where `||y_b||1 <= N(t-1)` and `||K_total||inf <= ceil(2N(t-1)^2/t)` suffice for two generic canonical packed polynomials. Sparse actual packing can sharpen this only by its proved support, not by treating random shares as small. Negative monomial packing must retain the difference between a signed representative and its canonical ring-t representative; no unrecorded centered encoding is substituted. **Outcome: accumulation identity closed; no per-block error reset justified.** [D B2/B3/C1; L]

### 4. B2.1 automorphism, digits and special-prime switch

Use exactly `u_i=can_q(sigma_g(c_i))` and eight nonnegative radix-B digits of u1, with upper digit at most 127. Raw row phases are `b_j+a_j*s=p*B^j*sigma_g(s)+e_j (mod Q)`. Thus, with `U0=p*u0+sum d_j*b_j`, `U1=sum d_j*a_j`, centered representatives Z_i, rounded quotients D_i and `rho_i=p*D_i-Z_i`,

`Z0+Z1*s=p*(u0+u1*sigma_g(s))+E_KS+Q*L`,
`E_KS=sum d_j*e_j`,
`Psi_out = u0+u1*sigma_g(s)+D_KS (mod q)`,
`D_KS=(E_KS+rho0+rho1*s)/p`.

The numerator is divisible by p from this equation; none of its summands need be divisible separately. Odd p excludes half ties. With `||s||1<=N`,

`||D_KS||inf <= D_* = [N*20*(7*(B-1)+127)+(p-1)*(1+N)/2]/p`.

No fresh-noise or rounded-component independence is used. **Outcome: exact identity/single-switch support bound closed; raw row construction, canonical digit lift, full-Q centered reconstruction and this exact DropSpecialPrime API UNKNOWN.** [D B2.1]

### 5. D21 eight-input interleave/trace and all 31 automorphisms

The inspected shared schedule first multiplies each of eight padded inputs by `v=64^-1 mod q`, then applies h=8,16,32 trace steps to each, then merges at h=4,2,1. Counts are `8+8+8+4+2+1=31` per direction; indices in execution order are `1025,513,257,2049,4097,8193`. Two directions give 62 switches, reusing the 96 key rows, not generating fresh rows.

Propagate full phases, not separately divided errors:

- Initial: `P_j <- v*P_j (mod q)`. Because q is 1 modulo 64, `v=(1+63q)/64`; for any integer phase P, a canonical resulting representative W satisfies `64W=P+q*Z` for an integer carry Z. It is not ordinary P/64.
- Trace: `P <- P+sigma_(N/h+1)(P)+D_KS (mod q)`.
- Merge: `P <- A+X^h*B+sigma_(N/h+1)(A-X^h*B)+D_KS (mod q)`.

These equations include representative changes modulo q automatically. If rewritten as `Delta*M+E` at any intermediate message normalization `M_old=M_new+t*K`, E must change by `-r_q*K`.

For the exact noiseless linear operator L, the complete inverse/trace/merge composition is the coefficient selector: output coefficient `64k+j` is input j coefficient `64k` for j<8, and zero for j>=8. This follows the D21 trace/interleave invariant; it applies to the whole phase, including input errors. It does **not** divide incoming selected errors by 64. Any signed representative normalization remains part of the case-3 carry ledger.

Each injected switch error is propagated by its remaining signed monomial/permutation sums. A conservative suffix bound, with no cancellation or independence discount, is

`8*(32+16+8)D_* + 4*4D_* + 2*2D_* + D_* = 469D_*`.

Hence `P_out=L(P_inputs)+sum_e L_suffix,e(D_e) (mod q)`. The 469 bound charges even padded work; it is not a measured noise distribution. **Outcome: complete phase recurrence/count and conditional additive-switch bound closed. Source-size implementation remains UNKNOWN; current shared degree limit is 4096.** [L; D C1]

### 6. Coefficient H2A, fresh zero and Down

For one final coefficient let `P=Delta*M+E`, add fresh-zero noise E0 and uniform r in Z_q. The reconstructed ring-t output is

`z=Down_q(P+E0+r)-Down_q(r) (mod t)`.

Writing each nearest-rounding remainder explicitly gives

`z = M + [t*(E+E0)-r_q*M]/q + eta (mod t)`, with `|eta|<1`.

This equation uses real lifts of the two rounded integers; changing a phase representative changes the expression by a multiple of t. Thus a sufficient condition for an integer reconstruction error in {-1,0,1} is `|t*(E+E0)-r_q*M|/q < 1`; no such universal condition is established by the proposal.

There is a concrete noiseless counterexample to a universal product/H2A one-bit law: use case 2 and E0=0. Its reconstructed error is

`-A+eta`, where `A=r_q*(t-1)^2/q`.

From `15t/16<r_q<t`, `2^146<q<2^147`, and `(t-1)^2>t^2/2`, one obtains `2^43<A<2^46`. Therefore every mask r gives a circular ring-t error far larger than one; it cannot wrap back to a small error. This is a counterexample to the noiseless algebraic claim, not an executed sampler trial or a quantified probability of a complete backend failure. Adding fresh-zero encryption cannot be presumed to repair the carry.

**Outcome: exact masked relation and failure of a universal noiseless one-bit product claim closed. Actual whole-fixture error/probability and coefficient-H2A transcript simulation remain unresolved.** The paper's Appendix C H2A assertion cannot be transferred to this independently frozen Delta/full-share graph without resolving this discrepancy. [D B3; C; P p.16]

### 7. Global T Lift and canonical y

For scalar canonical x,y in [0,t), `L=Lift_T(x)=T*x/t+epsilon`, with `|epsilon|<=1/2` (ties upward). Multiplication gives `L*y=T*x*y/t+epsilon*y`, so

`|t*epsilon*y/T| <= t*(t-1)/(2T) < 1/2` because the fixed T exceeds t^2.

Consequently `Down_T(L*y mod T)=x*y mod t` in ideal exact arithmetic. Reducing L into the three plaintext primes preserves this same global epsilon; independently lifting into each prime does not. **Outcome: ideal global conversion identity/support closed, not a BFV-noise bound.** [D B3; P p.8; C lift/down]

### 8. Three-context bOLE return phases and decryption gap

For each context j, let A_j and Y_j be the actual canonical coefficient polynomials after its SIMD encoding, `H_j=A_j*Y_j`, and `n_j=can_tj(-r_j)` coefficientwise. Set

`H_j+n_j=M_j+t_j*K_j`, `M_j=can_tj(H_j+n_j)`,
`Delta_j=floor(q_b/t_j)`, `rho_j=q_b-t_j*Delta_j`.

Then the complete returned phase is

`Psi_j=Delta_j*M_j+E_j (mod q_b)`,
`E_j=E_enc,j*Y_j-rho_j*K_j+E_fresh,j+F_j`.

K_j includes both negacyclic multiplication and negative-mask addition carries; n_j is zero when its mask coefficient is zero, otherwise t_j-r_j. Fresh negative-mask encryption is not trivial zero. Ordinary coefficient BFV rounding would decode M_j if

`||t_j*E_j-rho_j*M_j||inf < q_b/2`.                  (gap-j)

Equivalently the expression inside this norm is `t_j*(E_enc,j*Y_j+E_fresh,j+F_j)-rho_j*(H_j+n_j)`. This is the actual q_b/t_j rounding gap, not q_b/2 applied directly to E_j. A support-only sufficient inequality for each unchanged context is:

`t_j*[C_enc*N*(t_j-1)+C_enc+2^97] + rho_j*[N*(t_j-1)^2+(t_j-1)] < q_b/2`.

| Context | Exact substitution in gap-j and the support inequality |
| --- | --- |
| 1 | t_j=35184371613697; rho_j=356811922413223917036199361012173016050532353 mod 35184371613697 |
| 2 | t_j=35184371417089; rho_j=356811922413223917036199361012173016050532353 mod 35184371417089 |
| 3 | t_j=35184371138561; rho_j=356811922413223917036199361012173016050532353 mod 35184371138561 |

These are exact conditional inequalities, not asserted satisfied bounds. No backend decoder/encoding source is available to establish that its convention is this one or that it implements the proposed canonical coefficient multiplication. Failure of a sufficient triangle inequality would not prove actual decryption failure. **Outcome: full return/gap identities closed; successful three-context BFV decryption remains UNKNOWN.** [D B3/B4]

### 9. SIMD, full CRT, single Down and padding

Conditional on all three exact decryptions in case 8, decode each context in the frozen lane order `zeta_j^(2i+1)`. Both parties then CRT-reconstruct their own canonical full-T share. Since coefficient masking is uniform over each plaintext ring and SIMD is invertible, its decoded mask is uniform; independent context masks give a uniform full-T mask R.

The key owner's lane is `can_T(Lift_T(x)*y-R)`; the peer's is R. Their separately rounded shares sum to `x*y+e (mod t)`, where

`e=t*epsilon*y/T+eta0+eta1`, `|eta0|,|eta1|<1/2`.

The right side is an integer with magnitude <3/2, so `e in {-1,0,1}`. This support statement needs no independence between rounding errors. A decoded BFV error is not covered. The square doubles this conversion error; the two general cross-term errors add without an independence claim.

8195 lanes require two full batches; the second has three useful lanes and 8189 padded lanes. Both parties still mask/decode/CRT-convert under the declared full-ring contract before public cropping. No per-context Down, per-limb truncation or unmasked tail is justified. **Outcome: conditional full-CRT conversion law closed; exact backend lane permutation/API UNKNOWN.** [D B2/B3/C1; P p.8]

### 10. Fixed-fixture flooding tally and correlated shifts

| Fixed work | Count |
| --- | --- |
| OLT directions; inputs; products | 2; 12 original input ciphertexts; 24 ct-plaintext products |
| OLT output groups / fresh-zero H2A returns | 2 / 2; separate coefficient masking, not bOLE flooding |
| OLT automorphism invocations | 62 total; 31 per direction |
| bOLE paths | 2 general cross directions + 1 square cross term |
| bOLE batches/context count | 2 batches per path, 3 contexts per batch |
| bOLE returned full rings | 3*2*3 = 18 |
| Flooded coefficients | 18*8192 = **147456** |

No additional return is required by the symbolic equations inspected. If an eventual API/circuit requires one, this tally/bound stops; it must not be reused. Doubling the square after conversion adds no return. No full-model or repeated-request inference is made.

For integer shifts d, independent uniform F of width W=2^98 has `TV(F,F+d)=min(1,|d|/W)` on integer supports; modular reduction cannot increase this distance. Conditional on a coupled pre-flood history, let D_i bound each required simulator/real shift. Fresh flooding must remain independent of that history; key/input/error correlations themselves need not disappear. The hybrid/union bound is

`delta_fixture <= min(1, sum_(i=1)^147456 min(1,D_i/2^98) + epsilon_bad)`.

For a common bound D and zero exceptional allowance, this method can certify 2^-40 only if `D <= 2^58/147456 = 2^44/9`. Actual epsilon_bad includes unsupported gap events, sampler/entropy failures and any other conditioning failure. Observable abort remains part of the transcript; conditioning on success is not free.

The support-only convolution allowance `C_enc*N*(t_j-1)` already exceeds 2^75 for these primes; it supplies no shift envelope as small as 2^44/9. A pairwise shift may need twice an individual residual bound, plus the carry terms. **This is an insufficient upper bound, not a demonstrated statistical attack.** The necessary conditioned shifts/simulator, exceptional probabilities and the other ciphertext component's computational argument are UNKNOWN. **Outcome: tally and conditional whole-fixture inequality closed; no 40-bit privacy result.** [D section D, Case10 fixed-fixture tally; B4]

### 11. Bounded rejection draws and finite exhaustion union

At the proposal level, there are 8 secret polynomials, 8 public-key raw phase pairs and 96 raw switch rows. OLT has 12 original encryptions and 2 fresh zeros; bOLE has 18 fresh negative-mask encryptions. Let n_b be its number of distinct original encryptions: at most 18 if there is one per declared context/batch/path, potentially fewer only with an explicitly identified immutable input reuse. C1 does not freeze those reuse identities, so the exact n_b is UNKNOWN, not a discount. Total public-key encryption calls are `32+n_b`, at most 50 under that graph.

Conditional on literal per-coefficient samplers with no extra backend expansion, the maximum unshared logical draw ledger is:

- Ternary coefficient samples: `(8+50)N=58N` (secrets plus ephemerals).
- CBD coefficient samples: `(8+96+2*50)N=204N`, each exactly 40 fair bits.
- Uniform public-key/switch-row RNS coefficient samples: `(2*4+6*3+96*4)N=410N`.
- Uniform masks: `2*3N` OLT limb samples plus `18N` bOLE plaintext samples.
- Uniform flood words: `18N` words of exactly 98 bits; no rejection.

Thus the explicit uniform-rejection envelope is `434N` coefficients; ternary is `58N`. If a uniform prime residue uses its minimal bit-width rejection sampler, its per-draw rejection probability is below 1/2; ternary rejection is 1/4. With fresh uniform proposal bits, 128 attempts and no independence assumption between failure events, the union bound is

`Pr[exhaustion] <= 434N*2^-128 + 58N*2^-256 < 2^-106 + 2^-237`.

This is a conditional logical upper envelope, not an audited backend RNG/work quote. A seeded uniform-a expansion, secret/default Gaussian generator, key preparation, retry/reseed policy or OT expansion may add draws and change the law. Entropy failure itself has no bound here. The approved finite sampler must burn the entire attempt on exhaustion; abort privacy still needs case 10. **Outcome: known logical calls and conditional exhaustion formula closed; actual backend expansion, original-reuse ledger and fallible/fork-safe RNG seam UNKNOWN.** [D B2/B4/C1]

### 12. Security and circuit-privacy boundary

Exact published relations needing review are, per OLT owner, `b+a*s=e_pk (mod Q)` and its restricted q relation; for each frozen g and j=0..7, `b_gj+a_gj*s=p*B^j*sigma_g(s)+e_gj (mod Q)`. Each of the six bOLE owner/context keys has `b+a*s=e_pk (mod q_b)`. All use the same N=8192 ring but distinct secrets as specified; row errors and uniform a values are fresh at generation, then immutable and reused. Publishing these related-secret evaluation-key encryptions needs the corresponding key-dependent/circular-security argument, not merely an ordinary independent-message RLWE label.

Inputs for a later estimator/review are therefore the exact ordered q/Q/q_b moduli, **196-bit OLT key modulus**, ternary secret distribution, CBD(20) errors (variance 10), public/row sample counts and reuse, automorphism relations, two-owner/context separation, threat model and estimator identity/settings/attack models. Reduction/security arguments must also cover CRT restriction, conditioned transcript shifts, accepted outputs and visible failures. Ferret/base-OT pin, circuits and application correlation proofs remain missing.

**Outcome: security-input/key-relation inventory closed; no estimator run, 128-bit security result, 40-bit transcript result or completed OT attachment.** [D B2/B4/B5; P pp.3–4,16–18]

## Bounded API attachment — sixteen rows

Backend column refers only to the exact F pin; no backend symbol is inferred from a functionality name. Shared symbols below are actual local contracts, not implementations of the missing backend edge.

| # | Edge | Exact existing symbol/locator or UNKNOWN; boundary |
| --- | --- | --- |
| 1 | Parameter construction | Backend **UNKNOWN**, cache missing; D B1 defines ordered values, not a constructor. |
| 2 | ring64 plaintext | Backend **UNKNOWN**; `CoefficientConversion::new`, `coefficient_conversion.rs:55–74`, explicitly rejects inadequate u128 arithmetic; no ring64 implementation claim. |
| 3 | Canonical/centered CRT | Backend **UNKNOWN**; no cached full-Q reconstruction API. `down`, `shared_profile_oracle.py:149–152`, is a clear integer reference only. |
| 4 | Component order | Backend **UNKNOWN**; `(c0,c1)` phase convention is D B2.1 proposal, not verified library representation. |
| 5 | Raw phase keys | Backend **UNKNOWN** for Q-level raw key/row generation and q restriction. |
| 6 | Gadget digits | Backend **UNKNOWN** for exactly eight canonical base-2^20 digits with upper digit <=127. |
| 7 | Automorphism direction | Backend **UNKNOWN** for sigma_g(s)→s key switch. `CoefficientRing::add_automorphism`, `coefficient_layout.rs:205–210`, is private clear coefficient permutation, not encrypted switching. |
| 8 | p-scaled c0 accumulation | Backend **UNKNOWN** for B2.1 raw full-Q component access/accumulation. |
| 9 | DropSpecialPrime | Backend **UNKNOWN** for centered nearest division by p of both components and checked RNS reconstruction. |
| 10 | Fresh public encryption | Backend **UNKNOWN** for sampler-controlled independent zero/negative-mask encryption; structural zero is insufficient. |
| 11 | Explicit samplers/flood | Backend **UNKNOWN** for fallible bounded ternary/CBD/rejection and F injection into component zero only. |
| 12 | SIMD lane adapter | Backend **UNKNOWN** for the exact declared zeta^(2i+1) lane order and inverse transform. |
| 13 | Complete H2A conversions | Backend **UNKNOWN**; `CoefficientConversion::{delta_encode,down,evaluate}`, `coefficient_conversion.rs:85–149`, and clear `coefficient_h2a`, oracle:155–182, do not supply wide/source-sized HE/freshness/privacy. |
| 14 | Original backing/temporaries | `LayoutStorage`, `coefficient_layout.rs:214–223`; `PublicCiphertextOwner`, schedule:493; actual backend key/context/NTT/preparation allocation capacities **UNKNOWN**. |
| 15 | Error/burn propagation | `ConversionError`, conversion:11–21, supplies local refusal only; backend sampler/HE/transport whole-attempt burn and cancellation integration **UNKNOWN**. |
| 16 | Shared transport/admission binding | `PublicCiphertextTraceCapacity`, schedule:599; `validate_public_ciphertext_trace`, schedule:888; generic typed owner/trace validation exists. Exact B5 two-party authenticated frame/setup/admission join **UNKNOWN**, not supplied by those symbols. |

## Bounded OT attachment — eight rows

For **every row**, concrete distributed circuit symbol/pin, base-OT/Ferret commit, message count, setup, work/storage quote and secure correlation allocation are **UNKNOWN**. The following are known functional laws only; they imply no gate/message counts.

| # | Function | Functional law and exact local/paper locator |
| --- | --- | --- |
| 1 | Wrap | `w=floor((a+b)/2^ell)` for canonical shares; D A2; clear `bumblebee_trunc`, oracle:108–119, computes it openly, not via OT. |
| 2 | MSB | MSB of reconstructed modular sum; local-share MSBs XOR carry of low ell-1 bits, P p.9. D A2's approximate comparison first drops each share locally; `bumblebee_less`, oracle:122–127, is only clear reference. |
| 3 | Exact comparison | Protected signed `x<y` under admitted no-wrap differences; D A2.4; concrete circuit **UNKNOWN**. |
| 4 | Boolean-to-arithmetic | Fresh additive sharing of the same Boolean value in the target ring, no opening; D A2.4; concrete circuit **UNKNOWN**. |
| 5 | Mux | `b*x+(1-b)*y`, protected bit b and fresh application correlations; D A2.4, P Table I; concrete circuit **UNKNOWN**. |
| 6 | One-hot exponent | Private one-hot for k=1+floor(log2 D), k in 1..61, and derived m in -8..22; all public candidates charged; D A5. Concrete bit/priority circuit **UNKNOWN**. |
| 7 | Truncation | D A2.1's biased share shift, protected wrap correction and party-0 parity correction; `bumblebee_trunc`, oracle:108–119. Its marginal probabilities are not cross-gate independence. |
| 8 | Positive extension | P1 adds M=2^30, protected wrap, subtracts lifted 2^32*w and M; range [-2^30,2^30), D A2.6/P Alg.5 p.18. `positive_extension`, oracle:130–140, uses a different clear rank/sign-flip formulation; not the exact distributed party schedule or a ready circuit. |

## What this closes; ranked implementation/adoption blockers

1. **Correctness first:** cases 2/6 demonstrate that the frozen Delta/full-share coefficient product cannot inherit a universal one-bit H2A reconstruction law. Cases 3–5 now give an explicit full-phase/trace recurrence instead of dividing error by 64. A probability/range contract sufficient for the proposed intended outputs remains missing; silently changing encoding/moduli is not authorized.
2. **bOLE/privacy:** cases 7/9 close the ideal global Lift/CRT/Down arithmetic conditional on exact BFV decryption. Cases 8/10 expose the exact three gap inequalities and 147456-coordinate transcript requirement. No successful error envelope, exceptional probability, conditioned simulator or security estimate follows from them.
3. **Executable ownership/API:** cache absence leaves backend edges unresolved; sampler expansion, complete original owner/phase admission, exact OT implementation and circuit costs remain blocking. Existing shared contracts are useful but neither source-sized nor an HE runtime grant.

No additional cases, replacement parameters, implementation proposal, new experiment or approval request follows. **ANALYZE-1 terminates with this artifact; B-S/Q remains NOT SELECTABLE.**

### Separately scoped public-arithmetic ownership — Astra's frozen custody proposal

**Independent design-only status.** The preceding terminal paragraph records historical ANALYZE-1 scope and findings; it neither authorizes nor adopts this later proposal. This subsection records the frozen shared-public-coefficient custody design only. It does not reopen the twelve cases, resolve their numerical/security blockers, qualify a backend, activate an adapter, change caps/profiles, or authorize implementation or execution. The 14 primitive composition tests are source-cleared but **UNCOMPILED/UNRUN**; this is not execution evidence.

#### Selected representation and trust boundary

- Extend `clear_input_owner` with proposed private coefficient state, **one original cumulative `WorkLedger`**, a shared attempt counter, and an explicitly enforced sticky-terminal protocol. Existing APIs do not already supply that protocol. No parallel paper-specific owner or ledger is selected.
- Selected input route is caller-declared **PUBLIC ORDINARY spectra**, copied into original-owned backing. It supplies no ciphertext, checkpoint, or encryption provenance. `original_source::PrimitiveAttempt` retains the genuine configuration/import root, not spectra. An authenticated coefficient issuer/transfer route remains unselected.
- A genuine existing-owner transfer must preserve backing **and original ledger**. Recopying into a fresh ledger cannot continue prior work. A genuinely new public root is new construction, not authorized continuation; bootstrap failure before a root exists cannot claim continuity.

**DESIGN labels only:** Pending owns three source and four target `Option<NegacyclicNtt>` slots; Ready owns complete `[NegacyclicNtt; 3]` and `[NegacyclicNtt; 4]` arrays, immutable thereafter. Preserve source order and targets `[q0,q1,q2,p]`. Complete arrays supply `RnsContraction::new` references; optional arrays cannot.

Completion requires **one exhaustive aggregate match**: all seven `Some` values move without cloning into complete arrays; the other arm retains every remaining value in cleanup custody and returns failure. No `unwrap`, indexing extraction, fallible call, allocation, or user callback may occur between extraction and destination ownership. Input pool and charge remain continuously owned. Count simultaneous pending/transfer/ready/result slots conservatively. Concrete transfer-owner layout, legal moves around destructors, and cleanup on each branch/unwind remain **OPEN**; the finite ordering requirement is **not a panic-safety proof**.

The exact initialized-`u64` input pool remains `27N`: source spectra `[0,3N)`, followed by ordered row/component/target rows `[3N,27N)`. Fixed ranges/descriptors remain under shared limits; output and scratch remain separate exact `6N` and `19N` vectors. No self-referential primitive is stored; borrow contexts for contraction only during admitted execution. The published result retains original root, input, contexts, buffers, and charges through the existing borrowed-output pattern.

#### Persistent-root state and failure contract

1. **Establish the persistent root before request processing.** Independently funded bootstrap or a genuine existing-owner issuer remains unresolved. Failure before a root exists establishes no continuity.
2. **Exclusive synchronous custody:** the proposed private route holds exclusive root custody during each synchronous operation. No exposed future, pending guard, external callback, or reentrant entry is selected; the entire call graph must exclude suspension. Mutable borrowing alone proves none of those call-graph properties. Trusted `Attempt::with_ledger` funding closures are permitted. No allocated `Busy` path is selected.
3. **Shared accounting:** construction and execution are distinct logical attempts, sharing the existing maximum of **32 total attempts**. Exactly one increment per phase must use a common private accounting seam, not a claim increment followed by an independently incrementing `begin_attempt`. Preserve existing ordering: attempt 33 refuses before debit; attempts 1–32 increment before the 32-unit entry debit. A refused `WorkLedger` debit leaves balances unchanged; prior successful debits and the increment remain burned, but a failed envelope is not described as spent. Fund each envelope before associated construction/query/scan/allocation; reserve full capacity before allocation.
4. **Proposed private sticky guard:** arm before fallible phase work. Success disarms only after complete publication; recoverable returned `Err` disarms only after partial storage retires before its charges and the same root is restored to its prior retryable state. Entry/envelope refusal is recoverable with actual prior burns only. Disarming never clears an independently terminal root.
5. **Abandonment/cancellation/unwind:** terminal marking remains sticky, including cleanup failure. Concrete nonpanicking marking/cleanup/disarm ordering remains **OPEN**. An unwound operation retains its root only if the caller holds that owner outside the unwound scope; abandoning the root destroys it rather than reopening it.
6. **Retirement:** live borrowed claims/results prevent retirement. Payload is destroyed before charge release; concrete context/descriptor/buffer/charge ordering still needs review. No abort rollback or physical-erasure guarantee follows.

Current `ClearInputBuilder::new` loses its local ledger on failure; rerunning it cannot implement same-root retry. Its numeric resize precedes `Data` cleanup ownership, so it is not a universal install-before-fill precedent. Preinstalled partial cleanup must be established for this route.

#### Storage census, exact capacities, and wipe multiplicity

| Logical storage | Initialized `u64` words | Contract |
| --- | --- | --- |
| Original-owned source and row pool | `27N = (3+24)N` | Exact capacity checked before numeric fill/copy; fixed ranges above. |
| Execution output | `6N` | Separate exact vector. |
| Execution scratch | `19N` | Separate exact vector. |
| Logical total | `52N`, or `416N` bytes | **Not complete physical peak.** |

`NegacyclicNtt` contains inline stage arrays, not nested heap allocations. Count actual optional-slot, complete-array, root/pending/transfer/result/phase layouts, descriptors, alignment, and simultaneous temporaries using actual `size_of` for selected types, not an arbitrary envelope. `Prepared::reserve` already includes `BorrowedOutput` and `Prepared`; `Budget::charge_external` adds guard metadata. Count each once.

Existing `u64_values <= 8192` gives the necessary condition `27N <= 8192`, hence power-of-two **`N <= 256`**. This is not a complete feasibility result and does not admit source `N=8192`. Exact-capacity checks precede numeric fill/copy. Each buffer must enter a cleanup owner before filling; track its initialized prefix and never read uninitialized spare capacity.

Keep lifecycle work separate from live storage:

- Root initialization and input copy each touch `27N`; root retirement wipes `27N` once.
- Execution initialization touches `25N = (6+19)N`. Failed partial execution wipes at most `25N`, according to initialized prefixes.
- Successful retirement currently includes both `BorrowedOutput::drop` and self-cleaning `Buffers::drop`: **two `25N` wipes (`2*25N`)**, not one. Removing that duplication requires a separately reviewed representation change.
- Context/descriptor destruction, vector frees, and charge release remain additional lifecycle events, not silently absorbed into word initialization/copy/wipe counts.

#### Separate funding and unchanged-cap approval gates

Inherited owner charges do not pay for new fields. Contraction **`C813 <= 1024`** and its existing 15 NTT fields remain unchanged. Composition `EXTRA` must preserve all **35 componentwise categories**, including inverse NTT, drop, constructors, and control; spare contraction-C capacity is not reusable funding.

The separately funded census must include:

- New claim/pre-entry query/control work; at most seven context constructions plus ring-constructor and refusal costs.
- Fixed ordering/binding checks, `27N` input copies, and **each** additional full scan.
- Two reservations, their exact-capacity checks, and `25N` execution initialization.
- Exactly one `contract_drop_into` per consuming ticket, then publication move/wrapper checks.
- All failure/retry/cancellation cleanup paths within the existing 32-attempt total, successful double execution wipe, root retirement, and context/descriptor/vector/charge lifecycle costs.

These remain distinct logical units. No arbitrary scalar conversion into `WorkLedger`, invented tariff, parallel counter, or spare-C reuse is selected. Before implementation, approval requires an original-ledger-compatible admission representation, complete concrete layout/query/branch/lifecycle census, and feasibility under unchanged caps. Returned-error atomicity alone establishes none of these.

`WorkLedger` subtracts clear plus control work from shared remaining work, and control work from control remaining work; control is **not an extra total budget**. No composition-to-ledger conversion is selected. NTT counters overlap: residue checks are included within checks, and exponent products within exponent steps; do not sum these as independent units. Preserve all 35 componentwise comparisons.

#### RNS-SOURCE-CALL-1 — USER73439 quotation design only

**Selected boundary, deliberately different scope.** This RNS subquote expands ordinary repository bodies (including macros and repository callbacks), but stops at explicitly named, typed stdlib/derived **CALL terminals** with symbolic NONZERO tariffs, values UNSET. It does not replace USER73302's later-recorded ledger-helper ordinary-library expansion or select the still-UNSELECTED selected-sum iterator extension. No free stdlib work, physical storage/CPU/time bound, implementation, funding, execution or cap change follows.

Public quotation domain is `N=2^k`, `0<=k<=13`, odd `3<=q<2^61`; lengths, residues and all ceiling fields may be arbitrary. Existing validated NTT descriptors retain their stronger prime/root contracts. Validate scalar domain and descriptor binding; reject wrong exact source/row/output lengths or insufficient scratch **before residue scans**. Arbitrary excess scratch is never traversed. Invalid residues and ceilings still require bounded refusal paths; quotation must not assume successful inputs. This domain is not owner admission of `N=8192` (the earlier `27N` cap condition remains).

`V` has separate coordinates for each typed arithmetic/conversion, comparison/branch, binding/store, borrow/index/slice, call/return, aggregate construction and Result/Option decision. Macro expansion contributes its actual statements, not fictitious macro calls. Repository closures/callbacks expand separately; a terminal iterator driver cannot swallow their work. Candidate terminal names are slice `len`, `fill`, `copy_from_slice`, `split_at_mut`, `swap`; `iter`, `iter_mut`, `IntoIterator::into_iter`, `Iterator::{next,any,enumerate,zip,step_by}`; integer `checked_{add,mul}`, `try_from`, `from`, `trailing_zeros`; `Option::{ok_or,and_then,is_some,is_some_and}`; `Result::{map_err,expect}`; instantiated `Default`, `Eq`/`PartialEq` and iterator retirement. Each requires exact receiver/argument/result types, operation and bounded length in its tag; this candidate list is **not exhaustive**. Syntax-level `?`, short circuit, bounds decisions and repository callbacks remain explicit events. Compiler-generated internals are not an invented Rust body.

Bulk CALL setup/return and word coordinates are disjoint: e.g. one `copy_from_slice<u64>(N)` boundary plus `N` copied-word coordinates, not a bulk terminal priced for those same words again. Iterator creation, each poll (including final exhaustion) and retirement are separate. Composition is sequential vector addition, alternative-arm componentwise maximum, and `init+(m+1)*poll+m*body+retire` for a loop bounded by `m`; early-exit loops use that conservative bound. Nested loops retain their source bounds. Existing 11 contraction and 35 EXTRA ceiling comparisons remain unchanged checks, not scalar prices. `V` replaces any overlapping source-work pricing; never add bundled legacy categories again, including residue/check and exponent/product subsets.

**Independently source-read anchors** (current paths under `crates/pllm-core/src/`; no build/test evidence):

| Expanded source sites | Conservative successful-path multiplicities; refusals are bounded separately |
| --- | --- |
| `coefficient_rns_contraction.rs:431–455,572–589,645–675` | 12 forward + 3 source inverse + 8 target inverse = 23 transforms; work query invokes `operations` 15 times and EXTRA 8 times = 23. All 23 query **bodies**, storage/work/EXTRA arithmetic and adapters require quotation, not only their call boundaries. External queries add separately bounded invocations; no unlimited caller-query allowance. |
| `coefficient_rns_contraction.rs:487–503`; `coefficient_ntt.rs:200–213`; `coefficient_layout/rns_drop.rs:51–52` | Distinct scans: `27N` core canonical comparisons, `23N` NTT validation, at most `12N` drop canonical comparisons. These are not three names for one scan; drop `any` can exit early. |
| `coefficient_ntt.rs:227–277`; shared `modular.rs:3–8` | `B=(N/2)k`: `23B` butterflies; NTT `mul` calls `46B+57N` when `N>1`, zero when `N=1`; NTT `sub` calls `23B`. Bit reversal inner loop <=k per outer iteration; swaps <=floor(N/2) per transform. Expand shared repository mul/sub bodies. |
| `coefficient_rns_contraction.rs:577–588`; `coefficient_layout/rns_drop.rs:55–64` | Lift `12N` unsigned remainders; contraction `24N` mul calls and corrected sums; drop `6N` remainders, `6N` mul calls and <=`12N` sub calls, six half computations. These exclude NTT arithmetic. |
| `coefficient_layout/rns_drop.rs:21–33`; `coefficient_layout/modular.rs:19–34` | Three inverse calls, three input remainders; each inverse <=128 iterations, two signed products/subtractions per active iteration, <=one terminal `rem_euclid`. Constructor failure and mapped errors remain included. |

Owner lifecycle stays outside this core: `19N` scratch, `52N` logical payload, one `27N` root wipe and two `25N` successful execution wipes retain their earlier meanings. Initialization/copy, cleanup, descriptor/vector retirement and all owner paths need separate quotation, not duplicate core scan/store charges.

**Finite enumeration proposal — table UNFROZEN, not a frozen census.** Source order is contraction file → NTT file → ring accessors/canonical → rounded drop → shared modular helpers. Expand macros in invocation order and repository callees in first-discovery order, retaining call edges (no repeated body definitions). Within each body use lexical expression preorder, left-to-right operands, then explicit entry/parameter/return and loop-create/poll/retire sites; distinguish branch arms and callback bodies. A tag is `(file,function,expanded-site-ordinal,event-kind,types,operation,length-bound)`. Sort by that order; assign dense indices only after exhaustive review. Each resulting coordinate is one event, not an aggregate-field alias or compiler-slot census.

Emit finite postorder term records `ZERO`, `EVENT`, `SEQ`, `ARM` and `LOOP`, with explicit bound and maximum expressions below. Bounds come from source: fixed 2/3/4/8/10/13/128, N, k, N/2, stage width/count, and their checked finite combinations, with Euclid <=128 and bit reversal <=13. Iterator callback work is a child term, not hidden within a CALL. Source terms include condition evaluation followed by ARM alternatives, short circuit and error propagation. Wrong-length refusal retains the storage and EXTRA queries performed **before** length checks in `contract_drop_into`; it excludes subsequent scans. Wrong descriptor binding remains a modeled refusal alternative. External queries require separate bounded invocations.

**FROZEN conditional interpreter rules ONLY; table/events UNFROZEN.** Counts, expression values and vector cells are `u128`; indices/extents are `u64`, opcode tags `u8`, metadata field tags `u16`. Immutable reviewed table has finite `D,E,T,root` and physical expression/term record lengths E/T. Trust in this immutable reviewed table is a premise, not a runtime authentication helper. Actual lengths, maxima and source coverage remain UNFROZEN, not an invented global event budget K. Fixed metadata has the following individually named fields, widened losslessly to u128; there is no caller-selected field collection, array lookup or callback. META is an exact switch from each listed numeric tag to its named scalar field (all other tags refuse):

| Tags, in order | Fixed metadata fields (existing Rust ceiling paths retained) |
| --- | --- |
| 0–8 | `N`, `Q0`, `Q1`, `Q2`, `P`, `source_length`, `row_length`, `output_length`, `scratch_length` |
| 9–18 | `ceiling.contraction.events.checks`, `.residue_checks`, `.moves`, `.swaps`, `.modular_additions`, `.modular_subtractions`, `.modular_multiplications`, `.exponentiations`, `.exponent_steps`, `.primality_rounds` |
| 19 | `ceiling.contraction.coefficient_reductions` |
| 20–29 | `ceiling.extra.inverse_ntts.checks`, `.residue_checks`, `.moves`, `.swaps`, `.modular_additions`, `.modular_subtractions`, `.modular_multiplications`, `.exponentiations`, `.exponent_steps`, `.primality_rounds` |
| 30–37 | `ceiling.extra.drop.checks`, `.residue_checks`, `.moves`, `.modular_subtractions`, `.modular_multiplications`, `.unsigned_reductions`, `.ordinary_subtractions`, `.ordinary_divisions` |
| 38–45 | `ceiling.extra.construction.checks`, `.descriptor_stores`, `.euclid_iterations`, `.signed_divisions`, `.signed_multiplications`, `.signed_subtractions`, `.signed_remainders`, `.unsigned_remainders` |
| 46–54 | `ceiling.extra.control.capacity_outcomes`, `.adapters`, `.result_inspections`, `.ceiling_comparisons`, `.length_comparisons`, `.dispatches`, `.selected_loop_polls`, `.slice_partitions`, `.selected_phase_calls` |

Expression record is `(opcode:u8,a:u64,b:u64,c:u64,imm:u128,field:u16)`. Term record is `(opcode:u8,a:u64,b:u64,c:u64,d:u64,e:u64,f:u64)`. The following patterns define **all** active fields; every inactive field must equal zero. Validate opcode/pattern and all active reference ranges BEFORE any operand access; unknown opcode refuses.

| Expression opcode | Active fields; prior-expression references |
| --- | --- |
| 0 LIT | imm; none |
| 1 META | field, exactly 0–54; none |
| 2 K | none |
| 3 ADD, 4 MUL, 5 SUB, 6 DIV, 7 EQ, 8 LT | a,b; both < current expression index |
| 9 SELECT | a,b,c; all < current expression index |

| Term opcode | Active fields and required ranges |
| --- | --- |
| 0 ZERO | none |
| 1 EVENT | a=event tag <D, b=count expression <E |
| 2 SEQ, 3 ARM | a,b=term references < current term index |
| 4 LOOP | a=init,b=poll,c=body,d=retire: terms < current term index; e=bound,f=reviewed source maximum: expressions <E |

Here `+!` and `*!` are selected primitive checked unsigned add/multiply at the stated operand width, refusing overflow; not opaque repository helpers. Ordinary subtraction below requires left>=right, division requires positive divisor and is unsigned floor. Every failed require/operation returns the sole payload-free `Refused`: no retry, recursion or partial V. Every loop uses a u64 index, condition `index<extent`, and `index := index +! 1`. Before any dereference, require index<extent and usize fit; flattened vector index is explicitly `(row *! D) +! column` in u64, with row<T and column<D checked first. Cell byte offset `16 *! index` and end `offset +! 16` must fit u64, usize and that region's checked byte extent. Table-record addresses likewise use checked u64 `index *! reviewed_record_stride` and `offset +! reviewed_record_stride`, within physical record extent and usize, before reading. Strides/layout/provision remain representation-review obligations, not implicit allocation authority.

```text
evaluate(fixed_metadata, immutable_reviewed_table, provisioned_workspace):
  # Original issuer prepaid evaluator/admission/refusal envelope BEFORE this call.
  require D>0 and T>0 and root<T and physical_record_lengths=(E,T)
  TD:u64 := T *! D; C:u64 := (E +! TD) +! D; bytes:u64 := 16 *! C
  require D,E,T,root,TD,C,bytes and all table extents/record byte extents fit usize
  # Fixed regions: expression[E], flat vectors[TD], output[D], all u128.
  # Check provisioned extents before addressing; no allocation/growth here.
  require workspace has these exact logical region extents and checked byte extents
  require 1<=N<=8192
  v:u128 := N; k:u128 := 0
  for step:u64 in 0..13:
    if v==1: break
    require v%2==0
    v := v/2; k := k +! 1
  require v==1
  require 3<=Q0<2^61 and Q0%2==1; require 3<=Q1<2^61 and Q1%2==1
  require 3<=Q2<2^61 and Q2%2==1; require 3<=P<2^61 and P%2==1
  zero exactly E expression cells, TD vector cells and D output cells (C total)
  for i:u64 in 0..E:
    read record after address checks; validate opcode, active fields, inactive zeros
    require all active expression references <i BEFORE reading their values
    switch opcode:
      LIT: value := imm
      META: require field in exact enum 0..54; value := named scalar selected above
      K: value := k
      ADD: value := expression[a] +! expression[b]
      MUL: value := expression[a] *! expression[b]
      SUB: require expression[a]>=expression[b]; value := expression[a]-expression[b]
      DIV: require expression[b]>0; value := expression[a]/expression[b]
      EQ: value := 1 if expression[a]==expression[b] else 0
      LT: value := 1 if expression[a]<expression[b] else 0
      SELECT: require expression[a]==0 or expression[a]==1
              value := expression[b] if expression[a]==1 else expression[c]
      otherwise: return Refused
    expression[i] := value
  # Eager postorder: prior errors in EITHER SELECT arm already refused.
  for r:u64 in 0..T:
    read record after address checks; validate opcode, active fields, inactive zeros
    require active term references <r, expression references <E, event tag <D
    # Only applicable references in validated pattern are checked/read.
    switch opcode:
      ZERO: leave zero row
      EVENT: vectors[r,a] := expression[b]
      SEQ or ARM:
        for j:u64 in 0..D:
          left := vectors[a,j]; right := vectors[b,j]
          vectors[r,j] := left +! right if SEQ else (left if left>=right else right)
      LOOP:
        m:u128 := expression[e]; require m<=expression[f]; p:u128 := m +! 1
        for j:u64 in 0..D:
          x:u128 := p *! vectors[b,j]; y:u128 := m *! vectors[c,j]
          z:u128 := x +! y; w:u128 := z +! vectors[a,j]
          vectors[r,j] := w +! vectors[d,j]
      otherwise: return Refused
  for j:u64 in 0..D: output[j] := vectors[root,j]
  return Vector(output)  # fixed output region, not an allocation or partial result
```

All validation, zeroing, loop polls, arithmetic, address checks, copies and returned refusals themselves consume events. Fixed `E,T,D` and <=13-step bounds give conditional determinism/termination under the trusted finite reviewed table, **not an exact helper-body event count**. Workspace persists after success/refusal; subsequent retirement needs separately prepaid evaluator-owner custody. No wipe/free, heap/stack choice, physical layout, cap fit or available storage is implied. Before this call, original issuer allowance must provision evaluator, admission and refusal envelopes; cleanup remains separate, all debits use the same original ledger. No self-funding recursion or implicitly available bootstrap K: existing startup/entry/cleanup notation proves neither prices nor available allowance. Original-unit admission tariff evidence and funding existence remain OPEN.

Named CALL terminal supplement: `i128::rem_euclid(self:i128,rhs:i128)->i128`, at `coefficient_layout/modular.rs:24`, has positive validated rhs. This uses the already selected stdlib CALL boundary, not intrinsic/body expansion; exact typed site and symbolic nonzero tariff still require table review.

**Limited closure:** selected RNS CALL boundary, source-checked anchors and conditional interpreter rules only. **OPEN:** actual table/maxima/event sites, exhaustive typed terminal list, every expanded body/macro/callback/iterator/query/validation/refusal source-coverage census, exact interpreter helper-body success/failure/retirement event counts, representation/layout/provision, independent review, tariffs, allowance, admission, cleanup and cap feasibility. Symbolic terminal `(a_i,b_i)` are nonnegative with `a_i+b_i>0` per typed terminal; no numeric prices selected. Dot-product conversion remains UNSELECTED; constructing V is not evaluating tariffs. Control consumes shared total; preserve all 11+35 ceiling checks and never double-price overlapping source events with legacy bundles. Prior numerical/security/native HOLD gates remain unchanged. This is neither implementation-ready, funded nor executable qualification, and representation is proposal only, not Rust adoption. Stop edits for independent Astra review.

##### bounded local source-event subtable — quotation only, unpriced/unfunded

This independent Astra53-plan subtable freezes **one canonical invocation only** of `modular::sub` at `crates/pllm-core/src/coefficient_layout/modular.rs:6–8`, under the already selected RNS quotation boundary. It is a literal review artifact, not a new verifier, archive, implementation, or execution permission. The helper body is fully expanded:

```text
if a >= b { a - b } else { p - (b - a) }
```

The local precondition is odd `3<=p<2^61`, `0<=a,b<p`, all three parameters `u64`. For the true arm, `0<=a-b<p`; for the false arm, `1<=b-a<p` and `1<=p-(b-a)<p`. Thus the selected primitive subtractions do not underflow and the selected return is canonical. **The helper performs no validation.** Establishing these preconditions and covering invalid calls/refusals remain outside this local subtable; this is not a quote for arbitrary helper inputs.

The following 17 distinct source-event sites use lexical expression preorder, then explicit invocation entry, parameters and selected return. Numbering does **not** imply dynamic evaluation order. Each tag retains the shared schema `(file,function,expanded-site-ordinal,event-kind,types,operation,length-bound)`, with the file/function above, ordinal below and scalar bound 1. Copies are value uses, not new bindings, temporary slots or result slots; each operator result feeds its enclosing expression directly.

| Local tag | Event kind, exact scalar types and operation |
| --- | --- |
| 0 | `if` decision on the `bool` result of tag 1; select exactly one arm. |
| 1 | Primitive `u64 >= u64 -> bool` comparison. |
| 2 | Copy `a:u64` into the comparison's left operand. |
| 3 | Copy `b:u64` into the comparison's right operand. |
| 4 | True-arm primitive `u64 - u64 -> u64`, `a-b`. |
| 5 | Copy `a:u64` into the true subtraction's left operand. |
| 6 | Copy `b:u64` into the true subtraction's right operand. |
| 7 | False-arm outer primitive `u64 - u64 -> u64`, `p-(b-a)`. |
| 8 | Copy `p:u64` into the outer subtraction's left operand. |
| 9 | False-arm inner primitive `u64 - u64 -> u64`, `b-a`. |
| 10 | Copy `b:u64` into the inner subtraction's left operand. |
| 11 | Copy `a:u64` into the inner subtraction's right operand. |
| 12 | One invocation boundary, `(a:u64,b:u64,p:u64)->u64`. |
| 13 | Parameter binding `a:u64`. |
| 14 | Parameter binding `b:u64`. |
| 15 | Parameter binding `p:u64`. |
| 16 | One selected `u64` return, from the chosen arm. |

Caller argument evaluation and caller result receipt are outside this subtable. The invocation boundary is charged **once here**, never again at the caller. There are no stdlib or repository calls inside this body, loops, references, indices, aggregates, `Result`/`Option` decisions or cleanup bodies. Primitive endpoints are the selected source-event boundary, **not** a claim that compiler work is free.

**Literal local records.** Local `D=17`, `E=1`, `T=33`, `root=32`; these are not whole-RNS dimensions. Expression record format remains `(opcode,a,b,c,imm,field)`:

```text
expression 0 = (0,0,0,0,1,0)
```

Term record format remains `(opcode,a,b,c,d,e,f)`. Every record is written explicitly; this is not runtime generation:

```text
term 0  = (1,0,0,0,0,0,0)
term 1  = (1,1,0,0,0,0,0)
term 2  = (1,2,0,0,0,0,0)
term 3  = (1,3,0,0,0,0,0)
term 4  = (1,4,0,0,0,0,0)
term 5  = (1,5,0,0,0,0,0)
term 6  = (1,6,0,0,0,0,0)
term 7  = (1,7,0,0,0,0,0)
term 8  = (1,8,0,0,0,0,0)
term 9  = (1,9,0,0,0,0,0)
term 10 = (1,10,0,0,0,0,0)
term 11 = (1,11,0,0,0,0,0)
term 12 = (1,12,0,0,0,0,0)
term 13 = (1,13,0,0,0,0,0)
term 14 = (1,14,0,0,0,0,0)
term 15 = (1,15,0,0,0,0,0)
term 16 = (1,16,0,0,0,0,0)
term 17 = (2,12,13,0,0,0,0)
term 18 = (2,17,14,0,0,0,0)
term 19 = (2,18,15,0,0,0,0)
term 20 = (2,19,2,0,0,0,0)
term 21 = (2,20,3,0,0,0,0)
term 22 = (2,21,1,0,0,0,0)
term 23 = (2,22,0,0,0,0,0)
term 24 = (2,5,6,0,0,0,0)
term 25 = (2,24,4,0,0,0,0)
term 26 = (2,8,10,0,0,0,0)
term 27 = (2,26,11,0,0,0,0)
term 28 = (2,27,9,0,0,0,0)
term 29 = (2,28,7,0,0,0,0)
term 30 = (3,25,29,0,0,0,0)
term 31 = (2,23,30,0,0,0,0)
term 32 = (2,31,16,0,0,0,0)
```

All inactive fields are zero. Every active term reference is strictly backward; every EVENT tag is below 17 and its expression reference is 0, below 1. Expression 0 is the literal 1 and has no references. Term 23 contains the eight common entry/parameter/condition events; term 32 adds the common selected return. Thus there are **nine common events**, with true-arm three events (12 total) or false-arm five events (14 total). Term 30 takes a **componentwise** ARM maximum over disjoint arm coordinates, so the root envelope is exactly 17 ones. Its coordinate sum 17 is conservative, **not an actual path length**; neither arm executes both branches.

**Conditional interpreter arithmetic, not a complete evaluator quote.** For these local dimensions, `TD=33*17=561`; `C=E+TD+D=1+561+17=579` `u128` cells, or `16*579=9264` logical cell bytes. Expression/term tables, fixed metadata, record layouts, padding and provisioning are additional obligations. This is not a storage-cap or physical-peak claim and selects no heap/stack representation.

On successful EVAL, partial anchors are 17 EVENT cell writes; `15*17=255` checked SEQ additions; `1*17=17` ARM comparisons; 579 initial zero writes; and 17 output copies. These are distinct interpreter operations, not helper source events or a full interpreter EVENT census. Validation, addressing, refusal and retirement work remain **UNCOUNTED** here; expression processing, loop/control and other interpreter work are not silently included in these anchors. Original-ledger evaluator/admission/refusal prepayment is required **before quotation**, with separately prepaid cleanup custody; this local arithmetic supplies neither tariffs nor allowance.

The existing outer successful-invocation bound `sub <=23B+12N` remains a **conditional multiplicity**, not complete outer-caller source coverage or proof. Integrating this local body must replace overlapping bundled old `sub` prices, never add this expansion on top; caller-only work remains separately quoted. Symbolic event tariffs `(a_i,b_i)` satisfy `a_i>=0`, `b_i>=0`, `a_i+b_i>0`; their values and debit conversion remain **UNSELECTED**. No dot-product price, independent counter, spare-C funding or cap increase follows.

**Closure remains local.** Only this canonical invocation's sites and literal subtable are frozen. Full-RNS fixed table, maxima, source coverage, exhaustive typed terminals, interpreter EVENT census, layout/provision, prices, funding, cleanup and cap feasibility remain **OPEN**. Do not install these local `D/E/T/root` as global dimensions or treat the existing 55 metadata fields as a frozen whole-RNS census; the local expression uses no META fields. Preserve all 62 existing rows, all 11+35 ceiling checks, original-ledger requirements and every numerical/security/native HOLD gate. No collection, import, execution, slot/index change or implementation permission follows. Stop edits for independent Astra review; parent independently checks this artifact.

##### Derived aggregate subbody table — bounded composition, quotation only

This is a **distinct derived aggregate**, not a replacement for the frozen one-call `D=17,E=1,T=33,root=32` table above and not a whole-RNS census. It uses the same 17 helper-body event coordinates and the already selected RNS CALL boundary and user quotation design. The Astra53 proposal and independent Astra31/67 clearance concern this bounded subcomposition only; no implementation, execution or funding follows.

**Successful-call multiplicity proof.** On an existing validated same-degree NTT descriptor family, let `N=2^k`, `0<=k<=13`, and `B=floor(N/2)*k`. At `coefficient_rns_contraction.rs:572–581,644–660`, there are three source inverse transforms, twelve forward transforms and eight target inverse transforms: 23 in total. Each transform has exactly B butterflies and one `modular::sub` invocation per butterfly (`coefficient_ntt.rs:255–265`), giving `23B` invocations. For `N=1`, there are no stages or butterflies, so `B=0`.

The six retained-limb projections (`coefficient_rns_contraction.rs:670–675`) each invoke the helper once per coefficient at `coefficient_layout/rns_drop.rs:64`, and once more at line 62 exactly when the removed coefficient is strictly greater than `(P-1)/2`. For component `c` in `{0,1}`, let `H_c` count coefficients satisfying that strict inequality in its removed-component vector. The same removed vector is reused for all three retained limbs, so the six projections contribute exactly `6N+3(H_0+H_1)` calls. **H is proof notation only:** no evaluator metadata, new scan or runtime counter is introduced. Since `0<=H_c<=N`, successful actual invocation count is

```text
23B + 6N + 3(H_0+H_1) <= 23B + 12N.
N=1: B=0; actual invocation counts are 6, 9 or 12.
```

Canonical helper arguments follow from the validated descriptors, whole-call preflight, reductions, canonical inverse outputs and corrected sums—not odd moduli alone. In NTT butterflies, reduced products and canonical inputs feed the helper; corrected sums preserve that domain for later stages. In drop projections, reduction establishes the canonical centered residue before the final subtraction, including when P exceeds a retained modulus. The signed minus operations in the Euclidean inverse constructor and the direct corrected-sum subtraction at `coefficient_rns_contraction.rs:588` are **not** invocations of this helper and are not counted by this aggregate.

For `RnsContraction::contract_drop_into` on an existing validated descriptor, every returned `Err` exits through lines 617–637 before any `modular::sub` invocation. This includes mismatched supplied descriptor bindings, invalid lengths/residues, insufficient ceilings and propagated query/constructor failures. Subsequent NTT/drop calls use `expect`; their validated premises establish successful execution. This statement does not count panics as returned refusals, prove panic rollback, or make validation/refusal work free.

**Literal derived aggregate records.** Local `D=17`, `E=11`, `T=33`, `root=32`. Expression format remains `(opcode,a,b,c,imm,field)`; all six fields of every record are explicit:

```text
expression 0  = (0,0,0,0,1,0)
expression 1  = (1,0,0,0,0,0)
expression 2  = (2,0,0,0,0,0)
expression 3  = (0,0,0,0,2,0)
expression 4  = (6,1,3,0,0,0)
expression 5  = (4,4,2,0,0,0)
expression 6  = (0,0,0,0,23,0)
expression 7  = (4,6,5,0,0,0)
expression 8  = (0,0,0,0,12,0)
expression 9  = (4,8,1,0,0,0)
expression 10 = (3,7,9,0,0,0)
```

Expression 1 reads existing META field 0 (N); expression 2 uses existing K validation, not new metadata. Expressions 4/5 compute `floor(N/2)` and B; 7/9 compute `23B` and `12N`; 10 is their checked sum. All active expression references are strictly backward; inactive fields are zero. No H expression or residue traversal is present.

Term format remains `(opcode,a,b,c,d,e,f)`. All 33 records are literal below. Relative to the one-call table, only EVENT records 0–16 change their count-expression reference from 0 to 10; terms 17–32 are unchanged:

```text
term 0  = (1,0,10,0,0,0,0)
term 1  = (1,1,10,0,0,0,0)
term 2  = (1,2,10,0,0,0,0)
term 3  = (1,3,10,0,0,0,0)
term 4  = (1,4,10,0,0,0,0)
term 5  = (1,5,10,0,0,0,0)
term 6  = (1,6,10,0,0,0,0)
term 7  = (1,7,10,0,0,0,0)
term 8  = (1,8,10,0,0,0,0)
term 9  = (1,9,10,0,0,0,0)
term 10 = (1,10,10,0,0,0,0)
term 11 = (1,11,10,0,0,0,0)
term 12 = (1,12,10,0,0,0,0)
term 13 = (1,13,10,0,0,0,0)
term 14 = (1,14,10,0,0,0,0)
term 15 = (1,15,10,0,0,0,0)
term 16 = (1,16,10,0,0,0,0)
term 17 = (2,12,13,0,0,0,0)
term 18 = (2,17,14,0,0,0,0)
term 19 = (2,18,15,0,0,0,0)
term 20 = (2,19,2,0,0,0,0)
term 21 = (2,20,3,0,0,0,0)
term 22 = (2,21,1,0,0,0,0)
term 23 = (2,22,0,0,0,0,0)
term 24 = (2,5,6,0,0,0,0)
term 25 = (2,24,4,0,0,0,0)
term 26 = (2,8,10,0,0,0,0)
term 27 = (2,26,11,0,0,0,0)
term 28 = (2,27,9,0,0,0,0)
term 29 = (2,28,7,0,0,0,0)
term 30 = (3,25,29,0,0,0,0)
term 31 = (2,23,30,0,0,0,0)
term 32 = (2,31,16,0,0,0,0)
```

EVENT tags remain below 17, expression reference 10 is below 11, and every active term reference remains strictly backward. The root is `(23B+12N)*(1,...,1)` over 17 coordinates: a **componentwise branch envelope**, not simultaneous execution of all 17 sites or an actual path-length count. Multiplying the one-call envelope bounds a sequence of at most `23B+12N` canonical helper invocations even when different invocations take different arms.

**Logical evaluator cells only:** `TD=33*17=561`; `C=E+TD+D=11+561+17=589` u128 cells; `16*589=9424` logical cell bytes. Expression/term records, metadata, layout, padding and provisioning are excluded from that cell-byte figure and remain separate obligations. It is neither physical storage/peak usage nor an allocation, stack/heap selection or cap-fit claim. Evaluating 11 expressions is not free; this section does not turn prior partial evaluator anchors into a complete evaluator quote.

Symbolic event tariffs remain nonnegative with combined value positive per terminal (`a_i>=0`, `b_i>=0`, `a_i+b_i>0`), values **UNSET**. Count the helper invocation boundary exactly once per modeled call; replace overlapping legacy helper fees rather than adding this expansion on top. Caller arguments/results, guards, loops, scans and other caller events remain outside this helper-only aggregate. Evaluator work, admission, refusal and cleanup also remain outside and **OPEN**, requiring prepaid authority on the same original ledger; this bound supplies no allowance or funding. Full census, layout/provision, prices, allowance, caps, all 11+35 ceiling checks, all 62 bootstrap rows, numerical contracts, prior library boundaries and every HOLD remain **UNCHANGED**. Stop edits for independent Astra review; no implementation or execution permission follows.

##### Bounded local multiplication table and exact multiplicity lemma — quotation only

**Review-cleared DOC-only scope.** Astra 31/80 authored this bounded addition; independent Astra 13 cleared the DOC-only scope. It uses the existing **RNS-SOURCE-CALL-1** boundary, not a new library boundary. The frozen subtraction one-call subtable and derived aggregate above remain unchanged. Only one canonical `modular::mul` invocation is tabulated below; the whole-call lemma supplies multiplicity, **not an aggregate interpreter table or whole-RNS quote**.

The expanded body at `crates/pllm-core/src/coefficient_layout/modular.rs:3–5` is:

```rust
(u128::from(a) * u128::from(b) % u128::from(p)) as u64
```

Parameters are `a,b,p:u64`, with odd `3<=p<2^61` and `0<=a,b<p`. Widening is lossless; the product is less than `2^122`, hence fits `u128`. The divisor is positive, the remainder lies in `[0,p)`, and its cast to `u64` is safe and canonical. **The helper performs no validation.** These premises exclude primitive overflow and zero-divisor failure for this invocation; they do not invent compiler-inserted check events or a physical instruction/time/storage bound.

The 14 tags use lexical expression preorder, then explicit invocation entry, parameters and return, **not execution order**. Each retains `(file,function,expanded-site-ordinal,event-kind,types,operation,length-bound)`, with the file/function above, ordinal below and scalar bound 1.

| Local tag | Event kind, exact scalar types and operation |
| --- | --- |
| 0 | Primitive cast `u128 as u64 -> u64` of the remainder. |
| 1 | Primitive `u128 % u128 -> u128` remainder. |
| 2 | Primitive `u128 * u128 -> u128` multiplication. |
| 3 | Typed stdlib CALL `<u128 as From<u64>>::from(a:u64)->u128`, including its return. |
| 4 | Copy `a:u64` into tag 3's argument. |
| 5 | Typed stdlib CALL `<u128 as From<u64>>::from(b:u64)->u128`, including its return. |
| 6 | Copy `b:u64` into tag 5's argument. |
| 7 | Typed stdlib CALL `<u128 as From<u64>>::from(p:u64)->u128`, including its return. |
| 8 | Copy `p:u64` into tag 7's argument. |
| 9 | One helper invocation boundary, `(a:u64,b:u64,p:u64)->u64`. |
| 10 | Parameter binding `a:u64`. |
| 11 | Parameter binding `b:u64`. |
| 12 | Parameter binding `p:u64`. |
| 13 | One `u64` helper return. |

The three `From` sites are distinct selected stdlib CALL terminals, with symbolic NONZERO tariffs, values **UNSET**; their return work is included in each CALL, not separately added. Argument copies are separate events. No extra conversion copy, binding, temporary or result slot is invented: operator and CALL results feed their enclosing expressions directly. Caller argument evaluation and caller result receipt are excluded. The helper invocation boundary is counted **once here**, never again at the caller. No ordinary `From` body expansion or compiler-generated implementation is asserted.

**Literal one-call records.** Local `D=14`, `E=1`, `T=27`, `root=26`; none is a global RNS dimension. Expression format is `(opcode,a,b,c,imm,field)`:

```text
expression 0 = (0,0,0,0,1,0)
```

Term format is `(opcode,a,b,c,d,e,f)`. EVENT uses opcode 1; **SEQ uses term opcode 2, not expression ADD**. Every record is explicit:

```text
term 0  = (1,0,0,0,0,0,0)
term 1  = (1,1,0,0,0,0,0)
term 2  = (1,2,0,0,0,0,0)
term 3  = (1,3,0,0,0,0,0)
term 4  = (1,4,0,0,0,0,0)
term 5  = (1,5,0,0,0,0,0)
term 6  = (1,6,0,0,0,0,0)
term 7  = (1,7,0,0,0,0,0)
term 8  = (1,8,0,0,0,0,0)
term 9  = (1,9,0,0,0,0,0)
term 10 = (1,10,0,0,0,0,0)
term 11 = (1,11,0,0,0,0,0)
term 12 = (1,12,0,0,0,0,0)
term 13 = (1,13,0,0,0,0,0)
term 14 = (2,9,10,0,0,0,0)
term 15 = (2,14,11,0,0,0,0)
term 16 = (2,15,12,0,0,0,0)
term 17 = (2,16,4,0,0,0,0)
term 18 = (2,17,3,0,0,0,0)
term 19 = (2,18,6,0,0,0,0)
term 20 = (2,19,5,0,0,0,0)
term 21 = (2,20,2,0,0,0,0)
term 22 = (2,21,8,0,0,0,0)
term 23 = (2,22,7,0,0,0,0)
term 24 = (2,23,1,0,0,0,0)
term 25 = (2,24,0,0,0,0,0)
term 26 = (2,25,13,0,0,0,0)
```

All inactive fields are zero. Every active term dependency is strictly backward; EVENT tags are below 14 and reference expression 0, below 1. Expression 0 has no references and equals 1. The SEQ chain follows entry/bindings, operand evaluation and selected return; root 26 is exactly 14 ones. There is no branch maximum or loop in this one-call table.

**Logical cells, not layout or evaluator work.** `TD=27*14=378`; `C=E+TD+D=1+378+14=393` u128 cells; `16*C=6288` logical cell bytes. This excludes expression/term records, metadata, padding and representation/provisioning obligations. It is neither a physical-peak/cap-fit claim nor an evaluator event census. Evaluator validation, addressing, zeroing, expression processing, SEQ arithmetic, output copies, refusal and retirement all remain separately chargeable and unquoted here.

**Exact successful whole-call multiplicity lemma (no aggregate table).** Consider one `RnsContraction::contract_drop_into` on an **existing validated descriptor**, with `N=2^k`, `0<=k<=13`, and `B=floor(N/2)*k`. Its composition calls at `coefficient_rns_contraction.rs:644–675` include twelve forward transforms, three source inverse transforms and eight target inverse transforms: twelve forward and eleven inverse total. At `coefficient_ntt.rs:231`, `N=1` returns before **all** modular multiplications. For `N>1`, lines 237–239 and 255–274 give exactly `2B+2N` helper calls per forward transform and `2B+3N` per inverse transform: two per butterfly, two N-sized twist/untwist contributions per transform, and the inverse-only N-sized scaling contribution. Thus NTT helper calls total `46B+57N` when `N>1`, and zero when `N=1`.

Pointwise contraction contributes exactly `24N` calls (`coefficient_rns_contraction.rs:572–588`). Six retained-limb drop projections contribute exactly `6N` more (`coefficient_layout/rns_drop.rs:56–64`), independent of the centering branch. Therefore the exact number of `modular::mul` invocations on successful execution is

```text
M(N) = 30                         if N=1;
M(N) = 46B + 87N                  if N>1.
```

This is a proof-level invocation count, not new metadata, a runtime counter, an aggregate expression/table proposal or a whole-call source-event vector. Each of these invocations has the one-call vector above under its canonical premises; caller events are not covered by that fact. Do not substitute M or this local table's dimensions for a global census.

**Reachability and domain qualifications.** Work/storage/EXTRA queries at `coefficient_rns_contraction.rs:256–456` and `coefficient_ntt.rs:170–198` use ordinary arithmetic, not this modular helper. Drop construction's Euclidean signed products are not `modular::mul` calls. NTT `new`, exponentiation/pow, primality and standalone pointwise APIs are not reached by this existing-descriptor execution; descriptor construction is excluded from M and remains chargeable elsewhere. None of those exclusions makes query, construction or caller work free.

Canonical arguments follow from validated roots/inverses, descriptor binding, whole-call scans, lift reductions, canonical transform outputs and corrected sums, **not odd p alone**. Reduced products and canonical inputs preserve butterfly invariants; lift/drop reductions and validated inverses establish the pointwise/drop premises. Every returned `Err` at `coefficient_rns_contraction.rs:617–637` precedes any modular multiplication, so those returned-error executions invoke this helper zero times. Preflight, query and refusal work still require quotation and funding. Later validated NTT/drop calls use `expect`; this returned-error statement is not a panic-rollback guarantee or a claim that panic paths are returned refusals.

**Local closure only; frozen STOP EDIT for Astra review.** Symbolic tariffs remain `a_i>=0`, `b_i>=0`, `a_i+b_i>0`, values **UNSET**. Replace overlapping legacy helper fees; never add this expansion on top. Caller/evaluator/refusal/admission/cleanup census, full source coverage, layout/provision, pricing, allowance and funding remain **OPEN** on the same original ledger. All 62 bootstrap rows, all 11+35 ceiling checks, existing caps, numerical contracts, prior library boundaries and numerical/security/native HOLD gates remain **UNCHANGED**. This literal one-call table and exact multiplicity lemma introduce no mechanism, numeric output, tests, implementation, prepayment proof or execution authorization. Parent independently checks this DOC-only artifact; no aggregate multiplication table is selected.

**Existing contract anchors (reuse/extension, not new APIs):**

- Compiler `clear_input_owner::{Payload, ClearInputBuilder::new, ClearInputs::begin_attempt, Attempt::with_ledger, Prepared::reserve, BorrowedOutput::drop}`.
- Compiler `coefficient_extraction_reference::{prepay, Prepayment::extract, Buffers::drop, initialize}` and `original_source::{PrimitiveAttempt, SourceAdmission}`.
- Garble `shared_allocation::{OwnedExternalCharge::try_build, OwnedCharged, exact_capacity}`.
- Core `coefficient_rns_contraction::{SourceSpectra, RowSpectra, CompositionExtraWork, contract_drop_into}` and `coefficient_ntt::{NegacyclicNtt, construction_operations, LogicalOperations}`.

#### Proposed bootstrap boundary — scoped logical storage approved; adoption HELD

User 72515 approves only the bounded **logical storage contract**: concrete owned backing plus explicitly selected simultaneous typed slots. Hidden compiler/library frames and allocator internals are excluded; this is not an ABI-stack, RSS, or physical-peak bound. Independent Astra review 72537 accepts the conditional closure-layout/helper reasoning below, not a complete quote. Candidate B remains **UNFROZEN**. It extends shared `clear_input_owner`, not a parallel root; current APIs do not supply this boundary. Numeric layout (uncompiled), event tariffs, funding, pre-ledger authority, mutable-cap/ordinary-phase changes, adapter implementation/execution and integration-attachment fit remain **HELD**. GKM source remains frozen/HOLD; no future campaign is a premise.

Proposed private layout inside the existing owner (not implemented); the attachment remains an unselected future representation:

```rust
Header { ledger: RefCell<WorkLedger>, attempts: Cell<usize>, work_limit: usize, phase: Cell<RootPhase> }
Payload { data: RefCell<Data>, coefficient: RefCell<Option<OwnedCharged<Box<CoefficientState>>>>, header: Header, caps: Cell<Caps>, poisoned: Cell<bool> }
EmptyRoot { owned: OwnedCharged<Payload> }
RootPhase { Empty, ClearBuilding, ClearReady, CoefficientReady, Constructing, Executing, Terminal }
```

`Payload` stays private; existing `ClearInputBuilder` and `ClearInputs` retain the same `OwnedCharged<Payload>` wrapper. Reuse `OwnedBudget`, opaque `OwnedCharged`, and `WorkLedger` unchanged: `payload()` returns only `&T`, with no mutable/extract API. Consequently validated caps need interior `Cell<Caps>` installation **before** ready publication, not an `OwnedCharged` API expansion. `RootPhase` is a new name because `Phase<'a, T>` already exists at `clear_input_owner.rs:606`. No spectra, callbacks, guard adoption, caller ledger, or caller-configurable work limits enter bootstrap; policy must be immutable, approved, and internal.

The empty payload needs a **new private initializer**: four empty `Data` vectors (`floats`, `words`, `tensors`, `ops`), four zero used counters, placeholder all-zero caps, `coefficient: None`, phase `Empty`, and zero attempts. These placeholders must never escape through ordinary `finish`: current `finish` checks only poison, so a future explicit phase gate is required. Ordinary construction would initialize `ClearBuilding` → `ClearReady`. Preserve current ordinary overlapping `begin_attempt` behavior; do **not** globally mark ordinary tickets `Executing`/`Terminal`. Proposed `CoefficientReady` must reject ordinary `begin_attempt`; that restriction needs separate approval.

Future coefficient entry requires exclusive `&mut EmptyRoot`, retaining the same ledger and shared 32-attempt counter, burning one attempt per phase: attempt 33 refuses before debit; attempts 1..32 increment before the 32-unit debit. Fresh builders/new ledgers cannot establish same-root continuity. A future sticky guard borrows **only the phase `Cell`**, armed before any `RefCell` borrows; cleanup must remain independent of terminal marking. Complete all fallible work and release all `RefMut`s → commit phase/disarm with no fallible gap → end guard scope → move the **same** `owned` wrapper. These are proposed enforcement obligations, not implemented guarantees.

Bootstrap belongs privately in the same module or a child module: independently funded fixed-policy/layout/postledger-claim checks → new original ledger → approved postledger debit **before reservation** → `budget.charge_external(B)` → `charge.try_build` closure moves that ledger into the **empty same `Payload`** → `EmptyRoot`, zero attempts. No numeric allocation, contexts, scans, or authentic issuer is created. Incoming budget-handle destruction is counted separately; retained charge keeps the accounting `Arc`. Owning a charge is **not work authority**.

There is **no sticky guard before `EmptyRoot` exists**. Unpublished bootstrap failure drops locals/new ledger; it cannot pretend to retain a failed root. `try_build` refusal before closure invocation drops the closure's captured ledger, then the charge. On later `Err`/unwind, closure locals drop before charge; success destroys `Payload` before charge (`Data` wipes empty numeric vectors). A repeat before publication is separately funded **new public input**, not continuation; after publication requests need the actual retained root. The helper cannot clean resources lacking their own cleanup custody.

Helper review is concrete, but complete call sites remain **SYMBOLIC**; no finite whole-boundary quote follows:

| Helper/path | Concrete census or remaining condition |
| --- | --- |
| `WorkLedger::new` | Two `min` operations, two field initializations; total clamps to `MAX_CLEAR_WORK`, control to `1 << 20`. |
| `WorkLedger::debit` | One checked add, one `ok_or`, one `?`, 0..2 comparisons; success has two updates. |
| `charge_external(B)` | Checked `E = B+G`, reservation, then `Arc` clone and guard construction. |
| `try_build` T1 | Checked subtraction `H-G` and refusal handling. |
| T2 | `payload_bytes()` subtraction, checked addition, refusal handling. |
| T3 | Resize checked addition/comparison; reserve growth; excluding nongrowth requires proof of `H>G` for actual `Payload`. |
| T4 | Budget access, closure `Result`/`?`, wrapper construction. |
| Reservation/lifecycle | A finite logical source-event census does not require physical CAS-retry/allocator-latency bounds; reserve/Arc/drop coverage still needs an approved original-unit envelope. Captured-ledger/charge, payload/charge, and incoming-handle drops remain distinct. |

Pre-ledger authority is absent: this boundary cannot retroactively pay for its creation or debit admission. Whole call-site policy, quote, capture/storage inventory, `Result`/return slots, and drop placement remain **UNFROZEN**; only the closure-type sizing candidate below is conditionally discoverable. Helper counts do not bound allocator CPU, panic, physical erasure, or whole-boundary work.

Astra independent review 48's **ledger-free closure sizing** witness is reaffirmed by independent review 72537, source-only and uncompiled. Partial candidate pseudocode (not the frozen bootstrap body; no implementation created):

```rust
use std::alloc::Layout;
use pllm_core::row_fold::WorkLedger;
use pllm_garble::shared_allocation::{Budget, Error as SharedError};
struct Capture { ledger: WorkLedger } // proposed private type
fn make(c: Capture) -> impl FnOnce(&Budget) -> Result<Payload, SharedError> {
    move |_| Ok(Payload::empty(c.ledger))
}
fn closure_layout<F>(_: fn(Capture) -> F) -> Layout { Layout::new::<F>() }
let layout = closure_layout(make);
```

Function-item coercion and the opaque return fix `F`; sizing its type invokes neither
`make` nor `WorkLedger::new`. A future actual builder MUST use this **same `make`**,
not a separately written closure. Precise capture may retain just `ledger`: measure
the actual closure `Layout::<F>`, never substitute `Layout::<Capture>` or field sums.
`Payload::empty`, `Header` and `RootPhase` remain unimplemented private proposals;
the `Ok`-only body is valid only if initialization genuinely has no fallible result,
not proof of panic safety. Non-Clone/non-Copy `WorkLedger` moves one original ledger.
Existing `try_build` (`shared_allocation.rs:524–540`) drops its builder parameter
before its receiver on pre-invocation refusal, hence captured ledger before charge;
wrapper field order (552–554) retires payload before charge.

Simultaneous incoming-budget/layout/policy/quote/`Result`/error/return storage and
actual ledger→capture/charge/wrapper overlap remain **UNFROZEN**: no `B` quote or
complete census follows. Count ownership once, but conservatively account for actual
storage overlap; `E=B+G` → `F=B+H` is unchanged. Fixed pre-ledger entry, including
`WorkLedger::new`, debit admission/refusal cleanup and incoming-budget destruction,
requires separate **UNAPPROVED** entry/refusal/unwind authority; future debit cannot
fund past work. Astra 21's source audit distinguishes logical funding from physical
execution: `WorkLedger` (`row_fold.rs:27–60`) only checks/subtracts supplied clear-plus-control
amounts, defining neither CPU units nor primitive tariffs. `Geometry::quote` (102–120)
and `ClearInputBuilder::resources` (175–180) are scoped envelopes, not transferable
reserve/Arc/drop fees; `shared_allocation::Budget::{reserve_bytes, release_bytes}`
updates byte accounting, and test `accounting_updates` counts completed updates,
not work debits. A finite logical source-event census can proceed **without**
physical CAS-retry or allocator-latency bounds; this adopts no one-atomic-call/one-unit
rule. The actual next boundary is approval of an original-unit logical envelope for
the exact empty-bootstrap call site and lifecycle, including reserve/Arc/drop coverage;
no numeric funded quote is established. CPU latency, panic safety and physical erasure
remain excluded, not prerequisites to that logical census. No generic tariff,
representation adoption, ordinary-attempt change or funding clearance follows;
cleanup custody, all 32 attempts and existing holds remain unchanged, and the existing
14 tests remain **UNCOMPILED/UNRUN**.

##### PARTIAL SOURCE CENSUS — approved scope, not an exact or funded quote

Astra 24's partial event census, reviewed by Astra 11, selects no representation.
Uninhabited `enum CoefficientState {}` is invalid as a future attachment layout;
proposed private inhabited `struct CoefficientState;` with attachment `None` is only
a bootstrap placeholder (no allocation), NOT the selected future representation.
Freeze the concrete inhabited attachment, `Payload`, `H`, `H>G` and outside inventory
BEFORE creating any root intended for future attachment: a placeholder quote proves
neither integration nor same-root layout continuity.
Candidate `HelperFrame` is a LOGICAL slot inventory, not the actual helper stack.
An exact quote requires frozen bootstrap pseudocode and quoting, validation,
initializer and error-conversion bodies, then closed typed occurrence IDs with
multiplicities, lifetimes and alias rules. Ownership is counted once; simultaneous
selected source/destination slots may both count. “Each call” expansion alone does
not close concrete `B`. Missing selected occurrences include helper parameters,
moves/results (including enclosing results), `charge_external` bytes, nested
`try_build`/`resize_payload` locals, `reserve_bytes` previous value, closure
parameters/results, debit total, initializer `Data`/`Header`/`Caps` intermediates and
the `RefCell::new(ledger)` boundary; also actual opaque closure type `F` occurrences,
validation/layout function-pointer and result slots, constructor/layout-check sums,
error conversions and unwind slots. Count actual `QuoteFrame` once, not both its
aggregate and aliased fields. These are selected source-visible slots, not a demand
to include excluded hidden compiler/library frames. Keeping all named moves live
cannot cover omitted selected occurrences.

By-value `IntegerPolicy`/`Quote` uses need valid `Copy`/`Clone` declarations or explicit
borrowed slots; declare the debited ledger `let mut`. The initializer must read the
same private policy constant without capturing `Policy` or changing closure `F`.
Validation must establish fixed policy identity/approved ceiling, clear-plus-control
total and control bounds, and consistent `Header.work_limit`. If both initial
balances are asserted equal to `limit`, require `limit <= min(MAX_CLEAR_WORK, 1 << 20)`;
`WorkLedger::new` otherwise clamps them. Unassigned symbolic policy constants are
not adopted numeric limits or fees. Validation cannot mint admission authority.
Define logical events R=`reserve_bytes` calls, C=Arc clones, I=builder invocations,
L=`release_bytes` calls, A=Arc-handle drops. Under checked-arithmetic and actual-growth
premises, excluding earlier refusals and partial unwind, including incoming-handle destruction:

| Path | R | C | I | L | A |
| --- | --- | --- | --- | --- | --- |
| Initial reservation failure | 1 | 0 | 0 | 1 | 1 |
| Growth reservation failure | 2 | 1 | 0 | 2 | 2 |
| Success THROUGH retirement | 2 | 1 | 1 | 2 | 2 |

Pre-ledger/arithmetic refusals are separate. These are EVENTS, NOT UNITS: no atomic-call/unit tariff.
`OwnedBudget::drop` refunds `size_of::<Self>()`, not zero; `OwnedExternalCharge::drop`
refunds before its Arc field drops. Payload precedes charge; pre-invocation ledger precedes charge.
`WorkLedger` has no drop glue; neither `WorkLedgerDrop` nor `FDrop` is an installed
callback or executed fee. Logical retirement references are census labels only.
Complete empty `Data::drop` wipes once (two empty-slice zeroize calls), then four Vec
drops; there is no `Payload::drop` double wipe. Partial initialization
drops ONLY initialized fields, not automatically a complete `Data`. Last-strong Arc drop destroys
`Budget`, not another release call; backing deallocation depends on weak links.
Anchors: `shared_allocation.rs:101–120,398–414,475–485,524–554`, `row_fold.rs:27–60`,
`clear_input_owner.rs:87–107`. Original-unit weights and pre-ledger entry/refusal/unwind
funding remain HELD; implementation and the 14 tests remain **UNCOMPILED/UNRUN**.

Let `G = size_of<OwnedExternalCharge>` and `H = size_of<OwnedCharged<Payload>>`, including padding. Subject to the still-unverified actual-layout premise **`H>G`**, review 72537 conditionally supports **`E=B+G`** and **`F=B+H`**, not `B+G+H`; preflight all checked sums, including **`B+(H-G)`**, before relying on the growth path/event table. Here `F` denotes final charged bytes, not the earlier closure type parameter. `B` is outside `H`; no complete concrete `B` is established. Freeze actual types/layout, simultaneous selected outside slots and drop placement before quoting it. Existing ordinary byte quote uses actual `size_of<OwnedCharged<Payload>>`; its work formula does **not** fund new initialization. A real allocated aggregate uses checked `Layout::extend` / `pad_to_align`, not field-size sums or invented stack alignment. Future boxed backing must be charged **before allocation**, with initialized partial self-cleanup and payload-before-charge retirement; count backing/wrapper/metadata once. Empty inhabited ZST/`None` establishes neither a future integration root nor `52N` attachment fit.

Selected scratch live before `B` is reserved requires independently admitted
**HELD ENTRY/EXTERNAL storage authority**, plus entry/refusal/unwind work authority.
Possessing incoming `OwnedBudget` alone admits neither those new slots nor their
work; later `B` reservation cannot fund them retroactively. Existing owned-budget
bootstrap covers only its own selected inventory (`shared_allocation.rs:327–357`).

`SourceAdmission::reserve_bootstrap_phase` for config `Ledger` / `PrimitiveAttempt` is sequencing precedent **only**, not clear-spectra authority or ledger conversion. Existing `OwnedBudget` bootstrap is separately inherited; fresh budgets cannot disguise retry. Clear plus control share the total, not extra credit.

**Gates before code or tariff:** frozen bodies/types/schedule and closed occurrence census; exact success/refusal/layout/lifecycle quote; original-ledger-unit funding under unchanged caps; unwind/root enforcement; explicit approval of new pre-ledger storage/work authority. Only scoped logical storage is approved, not those authorities or a numeric ABI quote. Same original ledger, shared 32-attempt cap, ordinary overlap and phase/finish enforcement remain design obligations. No full-census or root-safety claim is made; the 14 primitive tests remain **UNCOMPILED/UNRUN**, and adapter implementation/activation remains held.

##### Candidate decision packet — inhabited attachment and original entry units (PROPOSED)

**APPROVED DESIGN FOR NEXT QUOTATION STAGE ONLY:** the same-root representation/plumbing below, the safe one-State allocation route, and the private single-use OriginalAdmission/job/issuer design recorded below are chosen for quotation. This is not adopted Rust code, a current API, a security guarantee, a numeric tariff or original pre-entry funding. Earlier EmptyRoot/non-retained-handle and fresh-ledger bootstrap sketches remain historical proposals, not a second authority for this design. The earlier placeholder census does **not** already provide this attachment. Actual revised layout/all-path cleanup census remains UNFROZEN; concrete issuer/access paths, prices and funding remain OPEN. Only selected source-visible typed slots and owned backing are in scope: hidden compiler/library frames, allocator internals, ABI stack overlap, RSS, latency, and physical erasure are excluded. No size query, compilation, test, or execution was performed; implementation, execution and HOLD recovery remain unauthorized.

**Source reconciliation.** Core `coefficient_rns_contraction.rs:205–245,256–286` has `SourceSpectra<'a>`, `RowSpectra<'a>`, and the full **`RnsContraction<'a>`**, whose private fields borrow complete context arrays. It is not an owned plan and cannot safely be retained self-referentially inside this movable root. `NegacyclicNtt` (`coefficient_ntt.rs:59–67,140–164`) owns its ring/scalars and two inline `[u64; 13]` arrays; it is Clone, not Copy, and has no nested allocation. Retain seven actual descriptors, then construct the full borrowed plan locally during admitted validation/execution, never a substitute two-reference struct, fabricated lifetime, or stored optional plan. `clear_input_owner.rs:87–122,197–259,529–655` supplies the original owner, shared attempts and borrowed-output lifecycle; `shared_allocation.rs:398–414,475–554` supplies the unchanged owned transition. The garble fixed utility is source-cleared (its `work.rs:158` census has 13 predebit quote calls); this packet alleges no defect there and requires no new generic phase framework.

**Proposed private types, not existing APIs.** Resolve imports to the actual core types above, compiler `Data`, `Caps`, `Error`, and shared `OwnedCharged`, `OwnedBudget`, `OwnedExternalCharge`, `Budget`, `Error as SharedError`. `Range` is `std::ops::Range<usize>`. The following are candidate declarations, not compilable/adopted source:

```rust
struct Header {
    ledger: RefCell<WorkLedger>, attempts: Cell<usize>,
    work_limit: usize, phase: Cell<RootPhase>,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum RootPhase {
    Empty, ClearBuilding, ClearReady, CoefficientReady,
    Constructing, Executing, Terminal,
}
enum ContextSlots {
    Pending { sources: [Option<NegacyclicNtt>; 3],
              targets: [Option<NegacyclicNtt>; 4] },
    Ready { sources: [NegacyclicNtt; 3], targets: [NegacyclicNtt; 4] },
}
struct CoefficientState {
    n: usize,
    source: [Range<usize>; 3], rows: [Range<usize>; 24],
    composition: Option<CompositionStorage>,
    contexts: ContextSlots,
}
struct CoefficientBuffers { words: Vec<u64>, scratch: Vec<u64> }
struct Payload {
    data: RefCell<Data>,
    coefficient: RefCell<Option<OwnedCharged<Box<[CoefficientState; 1]>>>>,
    header: Header, caps: Cell<Caps>, poisoned: Cell<bool>,
    attachment_budget: OwnedBudget, // last: inherited handle, not a fresh budget
}
struct ContextSpec { ring: CoefficientRing, psi: u64 } // public request descriptor
enum PolicyIdentity { CoefficientBootstrapCandidate }
struct EntryEnvelopes {
    success: usize, refusal: usize, unwind: usize, retirement: usize,
}
struct InternalPolicy {
    identity: PolicyIdentity, limit: usize, source_word_ceiling: usize,
    postledger_control: usize, root_byte_ceiling: usize, entry: EntryEnvelopes,
}
```

`CoefficientState` is inhabited and non-ZST, with real pending/ready storage, not `struct CoefficientState;`, an uninhabited enum, or an unspecified future `None` attachment. The empty initializer starts the attachment at `None`, but its **actual optional wrapper/array-box type is fixed above before quoting the root**; access the one retained State as `&owned.payload()[0]`. Reuse existing `ClearInputBuilder { owned }`, then `ClearInputs { owned }` with the **same** `OwnedCharged<Payload>`; no private replacement root type. The new last Payload field retains the incoming OwnedBudget handle, already inherited/charged once; it creates no new Arc, byte counter, ledger or authority. Default field destruction retires Data and the attachment before this retained handle; each owned wrapper still drops payload before its charge. These field/body changes are chosen design for quotation only, not current APIs.

`ContextSlots` and `CoefficientState` have no custom Drop in this candidate; numeric cleanup belongs to `Data` and `CoefficientBuffers`, permitting legal moves of context fields. Pending is **local construction state inside attachment `try_build`**, never an installed mutable `OwnedCharged` payload. An exhaustive seven-`Some` aggregate match consumes that local Pending: success immediately builds Ready; refusal owns/drops the residual tuple. No cloning, indexing extraction, unwrap, fallible gap or callback between extraction and ownership. A local replacement/transfer, if needed, must be included in the new quote, not assumed free. All initialized context values drop on refusal; these public descriptors require no asserted secret wipe. Complete composition/validation locally before boxing/wrapping; there is no dependence on a nonexistent mutable-owned-payload API.

The source pool remains `Data.words`: exact capacity/length `27N`, source range `iN..(i+1)N` for `i=0..3`, row range `3N+jN..3N+(j+1)N` for `j=0..24`, ordered `[source][b,a][q0,q1,q2,p]`. No duplicated source pool in `CoefficientState`. `Data` owns wipes/frees; range descriptors only alias it. `CoefficientBuffers.words` is exact `6N`, scratch exact `19N`, each installed empty in that cleanup owner **before its first allocation**, capacity checked before fill. Length tracks initialized prefix; never read spare capacity. Its proposed Zeroize wipes both lengths; its Drop repeats Zeroize, preserving the existing `BorrowedOutput` plus buffer Drop **two `25N` wipes**. Output descriptors are six N-word ranges `kN..(k+1)N`; scratch descriptor is `0..19N`, with internal partitions supplied by the actual `CompositionStorage`/core operation, not access to its private fields. Those seven ranges are candidate execution-local `[Range<usize>; 6]` and `Range<usize>` slots, not hidden extra retained vectors.

The proposed root reservation includes a fixed internal **source-word capacity ceiling** before any request. At attachment, check `27N <= P.source_word_ceiling <= 8192`; allocate exactly `27N`, not a vector of the ceiling's length. Thus the earlier necessary `N<=256` remains, not full feasibility. Reserving the ceiling keeps unused byte allowance charged; it neither materializes nor initializes unused backing. This is necessary because `OwnedCharged` exposes no mutable charge/resize/extraction API: a later allocation in `Data.words` cannot silently grow its original guard. Obtain the separate attachment charge from private `payload.attachment_budget`; cover `Layout::new::<[CoefficientState; 1]>()` backing and actual `OwnedCharged<Box<[CoefficientState; 1]>>` inline metadata, **not** the already reserved source pool. Construct/validate a mutable local state inside that charge's actual `try_build` closure, then use the chosen fallible allocation route below. No installed Pending mutations. Output/scratch keep existing `Prepared::reserve::<CoefficientBuffers>` accounting, extended only by the proposed optional guard plumbing below; its actual new slots need re-quoting. Neither attachment nor execution charges replace `owned`, its original ledger, or its shared 32 attempts.

**Chosen safe one-State route, design for quotation only.** Reuse compiler `allocated::<CoefficientState>(1)` (`clear_input_owner.rs:124–131`), including its fallible reserve and exact `capacity()==1` check. Within the actual attachment `try_build`, finish all contexts/composition/validation **before** allocation; then `let mut v = allocated::<CoefficientState>(1)?; v.push(finished_state);` and, with `len()==capacity()==1`, `let slice = v.into_boxed_slice();` followed by checked `Box::<[CoefficientState; 1]>::try_from(slice)`. A sufficient-capacity push does not allocate; Vec guarantees len==capacity conversion to boxed slice without reallocating or moving elements. The safe Box slice-to-array TryFrom converts in place without allocation; stable since 1.43, compatible with Rust 1.85.1/MSRV 1.85. Its Err arm explicitly drops the returned slice **before returning Resource through the error adapter**; the initialized State/residual-context owners retire before the attachment charge on all refusal/unwind paths. This is source/design reasoning, not experimental validation. No unsafe/raw cast, Box::try_new, new allocator or replacement budget. One heap State backing is charged once; separately quote the actual Vec descriptor, slice fat-box and array-box move destinations, conversion Result/error arm and helper slots, not extra backing owners.

Official std contracts: [Vec guarantees](https://doc.rust-lang.org/std/vec/struct.Vec.html#guarantees), [Vec::into_boxed_slice](https://doc.rust-lang.org/std/vec/struct.Vec.html#method.into_boxed_slice), and [Box conversion implementations](https://doc.rust-lang.org/src/alloc/boxed/convert.rs.html). The route is chosen for the next logical quote, not permission to run it or a complete cleanup/layout proof.

Seven caller-declared PUBLIC `ContextSpec`s and public spectra enter only attachment, never bootstrap. Inside attachment `try_build`, construct at most seven `NegacyclicNtt`s under separately prepaid ring/NTT work; immediately put each result in local Pending, initially `composition: None`. This optional query metadata does not make the inhabited attachment a placeholder. Borrow local Ready arrays for the full `RnsContraction<'_>`; query actual composition storage/work and validate ordered roots, degree, lengths, canonical spectra and all componentwise ceilings. End the full plan borrow **before assigning** `Some(storage)` to the local state. Only a fully validated Ready state may be boxed/wrapped and published. `SourceSpectra` borrows `0..3N`; `RowSpectra` borrows `3N..27N`; immutable aliases own no backing. Output/scratch remain separate mutable borrows. No ciphertext/producer authenticity or authenticated clear-spectrum issuer follows from these checks or PUBLIC declarations.

**Proposed attachment construction/rollback/publication body, not installed plumbing.** Require exclusive `&mut ClearInputBuilder` in Empty phase with empty pool/attachment, original shared attempt increment and prepaid work/cleanup. Declare the construction phase guard **before** a pool rollback guard. The latter retains `RefMut<Data>`, prior caps/counters and cleanup status while the `27N` source pool is allocated, capacity checked against the root-reserved ceiling, initialized and filled. No private field access is fabricated through OwnedCharged; mutation is through existing interior ownership. Obtain the attachment charge from the retained private budget; complete local contexts/composition/validation in `try_build` as above, with immutable slices borrowed through that pool guard. On refusal/unwind, initialized local state drops before attachment charge release; caller's pool guard then wipes/frees initialized words and restores the empty vector/counters/caps **before** any recoverable phase reset. Phase-guard unwind marks sticky Terminal; uncertain cleanup cannot publish Ready or restore retryable state. All cleanup work is prepaid, not funded by the refusal.

For success, end all plan/slice/RefMut borrows, including the pool guard's borrow, before the nonfallible commit boundary. With exclusive builder custody, no callback/suspension and the private attachment slot proven unborrowed and None, install the completed owned wrapper, commit caps/counters, then disarm construction guard into CoefficientReady. No fallible allocation, validation, work debit or new callback may intervene after rollback disarm. A private RefCell replacement is nonfallible only under this proven-unborrowed invariant; this is a proposed proof obligation, not an assumption that arbitrary RefCell access cannot panic. Proposed rollback/phase guard types, disarm/move destinations and all partial-initialization paths must appear in the new quote. Ordinary construction/overlap behavior is unchanged.

**Candidate initializer/error bodies and historical bootstrap subquote.** The internal immutable policy `P` has named, **unassigned** fields: `identity`, `limit`, `source_word_ceiling`, `postledger_control`, `root_byte_ceiling`, and entry/lifecycle envelope ceilings requiring separate approval. No caller-supplied policy, ledger, callback or work-limit argument. Design approval does not assign `P`, its identity or any executable constants. Proposed bootstrap error `EntryError { Policy, Capacity, Work, Resource }` is private; `from_shared(_: SharedError) -> EntryError { EntryError::Resource }`, and `from_debit(_: row_fold::Error) -> EntryError { EntryError::Work }` deliberately retain no dynamic payload. Checked Option-to-Result/layout failures map to `Capacity`; validation refusals map to `Policy`. These names/bodies are proposals, not claimed installed conversions. The Capture/initializer sketch below remains quotation input for the same-ledger move, not an adopted bootstrap body.

```rust
struct Capture { ledger: WorkLedger, budget: OwnedBudget }
fn empty(c: Capture) -> Payload {
    let Capture { ledger, budget } = c;
    let data = Data {
        floats: Vec::new(), words: Vec::new(), tensors: Vec::new(), ops: Vec::new(),
        float_used: 0, word_used: 0, tensor_used: 0, op_used: 0,
    };
    let header = Header {
        ledger: RefCell::new(ledger), attempts: Cell::new(0),
        work_limit: P.limit, phase: Cell::new(RootPhase::Empty),
    };
    let caps = Caps { f64_values: 0, u64_values: 0, tensors: 0, operations: 0 };
    Payload { data: RefCell::new(data), coefficient: RefCell::new(None),
              header, caps: Cell::new(caps), poisoned: Cell::new(false),
              attachment_budget: budget }
}
fn make(c: Capture) -> impl FnOnce(&Budget) -> Result<Payload, SharedError> {
    move |_| Ok(empty(c))
}
fn closure_layout<F>(_: fn(Capture) -> F) -> Layout { Layout::new::<F>() }
```

**HISTORICAL/UNSELECTED validation subquote, not an adopted callable body.** The earlier scalar-only `validate` proposal (**quote UNFROZEN**) borrowed `&P`, compared its identity to the proposed private identity, and required `limit <= min(MAX_WORK, MAX_CLEAR_WORK, 1<<20)`, `postledger_control <= limit`, source-word ceiling within 8192, and each named envelope within its approved ceiling; otherwise `Policy`. It obtained `closure_layout(make)` without invoking `make` or constructing a ledger. Its still-unfrozen inventory required actual-type Layouts (errors map to `Capacity`), checked outside occurrences plus checked `P.source_word_ceiling * size_of::<u64>()` into B, and a ceiling check. With `G=size_of::<OwnedExternalCharge>()`, `H=size_of::<OwnedCharged<Payload>>()`, it required checked `delta=H-G` **and `delta>0`**, otherwise `Capacity`; checked `E=B+G`, `B+delta` and final `B+H` had to fit the root ceiling. The proposed `Quote { b: B, initial: E, grown_payload: B+delta, final_bytes: B+H }` has four `usize`s, but establishes no current B or complete quote. These numerical contracts remain quotation constraints; the selected issuer now constructs the sole ledger before quotation. No later debit pays these prior checks.

**HISTORICAL/UNSELECTED fresh-ledger bootstrap subquote, not the adopted callable body:** the earlier sequence was admission funding → `validate`/quote → `let mut ledger = WorkLedger::new(P.limit, P.limit)` → `ledger.debit(0, P.postledger_control).map_err(from_debit)?` → `budget.charge_external(q.b).map_err(from_shared)?` → `make(Capture { ledger, budget })` → `charge.try_build(builder).map_err(from_shared)?` → `Ok(ClearInputBuilder { owned })`. Its fresh post-validation ledger creation and local pre-capture retirement are superseded by the approved admission design below. Preserve this only as a historical subquote, not as complete B or a funded implementation. The selected move still uses the same `make` for the actual opaque F witness and call; `q` is borrowed, not implicitly Copy. No second closure, captured Policy/Quote, copied ledger, synthetic owner or borrowed guard substitutes for actual Capture/F. After successful initial reserve, the same ledger and incoming budget move Capture→F→Payload. Growth refusal retires captures before the root charge; partial initializer unwind retires only initialized fields and remaining captures before charge. On success the incoming handle retires as Payload's last field after Data/attachment, then the enclosing root charge drops. Every changed lifetime requires a new quote.

`empty` has no explicit fallible step: partial unwind drops only initialized fields, not a fictional complete Data; completed Data::drop invokes two empty numeric zeroizes and four Vec drops. This is no blanket nonpanic proof. Proposed ordinary construction sets ClearBuilding; ordinary `finish` accepts **ClearReady only**. Separate private `coefficient_finish` accepts **CoefficientReady only**, returning `ClearInputs { owned: self.owned }` with that same root; these bodies/gates are PROPOSED, not current APIs. Coefficient construction/execution use one shared increment each (33rd refuses before debit), existing 32-unit entry ordering and original ledger. Ordinary `begin_attempt` refuses coefficient state. Never alter ordinary overlapping attempts or globally mark their outputs Executing/Terminal.

Proposed private coefficient entry on ClearInputs constructs existing `Prepared { owner: self, ... }` with an **optional coefficient phase guard**; ordinary entries initialize that field to None. Transfer Some(guard) from Prepared into Phase when reserving CoefficientBuffers, using the same owner/ledger/attempts rather than a caller budget or fresh attempt. Phase declares the guard **after its charge field** so existing BorrowedOutput zeroize plus payload Drop retire buffers first, charge Drop releases storage next, and guard reset follows last. Normal successful/recoverable retirement restores the allowed prior phase only after cleanup; unwinding or uncertain cleanup marks sticky Terminal, never Ready. Prepared refusal must likewise retire any partial owner/charge before its guard resets. All new guard fields/transfers, normal/unwind detection and refusal bodies are PROPOSED and require actual-type/all-path quoting; they are not existing security or panic-safety guarantees. No replacement root, original ledger or caller-ledger authority is returned.

**Typed occurrence schedule — UNFROZEN after body revision.** Earlier B01–B15/A01–A04/X01 selected inventories described the prior non-retained-handle body and are **not a complete current quote**. Their exact multiplicities are withdrawn here rather than silently reused. The named replacement worklist below identifies actual types/owners and changed lifetimes; multiplicities, helper/result/guard move occurrences and all-path cleanup must be re-quoted before numeric B. It is not a new exact census. `s(T)` means conditional actual `size_of::<T>()`; aggregate fields alias resident storage, while explicitly selected move destinations need separate logical slots without duplicating ownership. Scope remains concrete owned backing and named logical slots, excluding hidden compiler/library stack and allocator internals.

| Revised inventory IDs | Actual types/slots to re-quote | Lifetime, aliases and partial-init/drop obligations |
| --- | --- | --- |
| B01–B04 | compiler-owned job/issue right, private issuer/OriginalAdmission custody and history, incoming `OwnedBudget`, policy/identity references, `fn(Capture)->F`, `Layout`, `Quote`, checked `usize`/Option/Result slots | Setup constructs the sole ledger before quotation; the layout witness constructs no additional ledger/F. Bootstrap borrows admission and burns entry before validation. Quote generation/validation and their own slots require prior funding. Early failure retains originals through cleanupTerminal; success moves the handle onward. Concrete setup/entry/helper/history slots remain unquoted. |
| B05–B06 | `Capture { ledger, budget }`, **actual revised F**, the same `WorkLedger`, debit/adaptation locals/results | Ledger originates in issuer setup, not bootstrap. Only after initial charge succeeds do ledger/budget move from admission into Capture/F; logical move slots are not new owners or backing. Growth refusal drops captures before charge; unwind tracks only initialized fields. Never replace s(F) by s(Capture). |
| B07–B09 | `OwnedExternalCharge`, helper borrows, checked resize/reserve slots, `Result<Payload, SharedError>`, revised `Payload` | Root charge/build transition; actual helper receiver/result/move slots need new counts. Result fields alias their own aggregate, not another resident root. |
| B10–B12 | revised `Header`, `Data`, `Caps`, RefCell/Cell field initializers, four typed Vec descriptors, last `attachment_budget: OwnedBudget` | Partial initializer and retained Payload; account handle move destinations, never duplicate inherited Arc backing. Retained last handle outlives Data/attachment; empty Data still has real descriptor/cleanup paths. |
| B13–B15 | `OwnedCharged<Payload>`, helper result, `ClearInputBuilder`, finish-result/`ClearInputs` destinations, error adapters and retirement borrows/scalars | Same owner through both finish paths; count selected overlap but exclude exactly one final resident H from outside B. Original-ledger and retained-handle destruction now occur with Payload retirement, not a separate successful entry-return release. |
| A01–A02 | seven `ContextSpec`s, serial NTT constructor results, local `ContextSlots`/`CoefficientState`, full plan/query results, `Vec<CoefficientState>`, `Box<[CoefficientState]>`, `Box<[CoefficientState; 1]>`, conversion `Result<Box<[CoefficientState; 1]>, Box<[CoefficientState]>>`, allocation/error-adapter helpers, attachment build closure/charge/results | Construct/validate locally under try_build before allocated(1); no mutable installed Pending. End full-plan borrow before local composition assignment. One array backing counted once; descriptor/result/move slots separately quoted. Err slice explicitly drops before Resource return, then attachment charge. Safe route selected; actual layouts/multiplicities/all-path cleanup remain UNFROZEN. |
| A03–A04 | construction phase guard, pool rollback guard with `RefMut<Data>`, saved caps/counters/status, input slices, local validated wrapper, commit/disarm slots | Guard declared before rollback owner; rollback retains pool borrow through fill and validation. Rollback wipes/frees/restores before recoverable phase reset; unwind/uncertainty Terminal. End RefMuts before nonfallible publication. No retained references in owned attachment. Guard bodies/types/multiplicities remain UNFROZEN. |
| X01 | amended `Prepared`, `Phase<'a,CoefficientBuffers>`, optional phase guard/move slots, `BorrowedOutput<'a,CoefficientBuffers>`, buffer/result locals, six output ranges and scratch range | Same original owner/ledger/attempt; ordinary optional guard None. Output zeroize/payload Drop then charge then guard; refusal/unwind paths included. Full 6N+19N backing once, both wipes funded. New phase/helper layouts are not the prior quote. |

After this re-quote only, proposed `B_out` is the checked sum of selected outside occurrences (not their aliased fields or one final resident H); proposed `B = checked_add(B_out, checked_mul(P.source_word_ceiling, s(u64)))`. Inherited OwnedBudget/Arc backing is charged once even though logical move slots are inventoried. Checked Layout::extend/pad_to_align must describe any actual backing aggregate; a conservative sum of named logical slots is not an allocated stack-frame or RSS bound. Conditional algebra **E=B+G → final=B+H** survives only if the revised actual layouts prove **H>G**; never B+G+H. It establishes no current B, complete bound or funded feasibility. Attachment/execution backing, constructor/query slots and all new rollback/guard/helper/retirement paths need their own complete prepaid quotes. Future reservation/debit cannot fund past quotation.

**Entry-unit funding PROPOSAL, NOT APPROVED, with event envelopes rather than invented tariffs.** Existing `WorkLedger::debit` checks/subtracts supplied clear+control and control balances; byte reserve/release and Arc/drop APIs define no work conversion. Propose administration classification `clear=0, control=w_event` for the *entry administration events only*, so `admin(0,w)` consumes **both shared total and control**, not additive/free control credit. Zero clear-work here does not mean no arithmetic/CPU; spectra operations and 35-category composition/NTT work remain independently classified, not zeroed. Leave weights `w_R,w_C,w_I,w_L,w_A,w_validate,w_layout,w_checked,w_new,w_debit,w_convert,w_init,w_retire` named and **unpriced**. Each weight needs an approved original-unit envelope, not a one-atomic/one-unit fiction; no spare contraction C or future debit pays prior quotation.

For the revised bootstrap sequence, define the candidate administrative control envelope schema as checked `R*w_R+C*w_C+I*w_I+L*w_L+A*w_A+V`, where V covers the listed bounded validation/layout/control/initializer/drop branch, not another credit. V remains qualitative until branch multiplicities and original-unit prices are frozen; this schema is not a complete or funded bound. H>G now requires proof for the revised layouts. Destruction of the retained incoming OwnedBudget handle moves to Payload retirement on success; it is not a second entry-return release. Admission debit refusal leaves balances unchanged by that refused debit, not restored to their initial values. The issuer-created ledger can already be consumed: debit refusal can be reachable despite `postledger_control <= limit`. A fresh-ledger unreachability proof does not transfer. Prior admitted spending and burned job/entry rights remain consumed.

| Revised path, through all owned cleanup | R/C/I/L/A and V status | Required ordering to re-quote |
| --- | --- | --- |
| Policy/layout/arithmetic or original-ledger debit refusal | UNFROZEN; prior-body event rows are not the current lifecycle quote | Admission retains original ledger/budget through cleanupTerminal; entry already burned. Include validation/debit adapters, history and terminal cleanup. |
| Initial reserve refusal | UNFROZEN | Borrowed admission retains originals through cleanupTerminal; no Capture or published Payload, no retry or renewed entry right. |
| Growth reserve refusal | UNFROZEN | Revised F owns incoming handle/ledger; captured handle drops before root charge. Include changed Capture/F/helper moves and destruction. |
| Bootstrap closure/partial-initializer unwind | UNFROZEN | Only initialized fields and remaining captures drop before charge; never assume complete Data/Payload. |
| Successful bootstrap through retained-payload retirement | UNFROZEN | No incoming-handle destruction at bootstrap return; Data/attachment retire before last Payload field, then root charge. Include both finish paths and changed lifecycle. |
| Attachment refusal/unwind/publication | UNFROZEN; separate original-ledger work/cleanup envelope | Local state before attachment charge; pool rollback before phase reset; uncertain/unwind Terminal; commit only after borrows end. |
| Coefficient Prepared/Phase/output retirement | UNFROZEN; separate original-ledger work/cleanup envelope | Same attempt/ledger; output wipes/payload Drop, charge Drop, then optional guard reset; include all refusal/transfer/unwind slots. |

R names logical reserve calls, not physical CAS retries; C names charge Arc clones, I builder invocations, L releases, A Arc-handle destructions. Their **prior numeric rows belonged to the prior body**; no exact replacement counts or tariffs are asserted. Independent review conditionally cleared only that previously selected bootstrap; its original S excluded validator/helper/retirement work and was incomplete. That clearance is not a complete quote for the new issuer/admission lifecycle. Last-strong Arc destruction belongs to its named lifecycle envelope, not an invented extra release. V and the typed inventories must be frozen for actual revised success/refusal/closure-unwind/retirement paths, including only live initialized owners and every validation/control call. There is no explicit closure Err in bootstrap make's Ok-only body; attachment closure can refuse while constructing/validating. Real 27N/25N buffer cleanup is separately prepaid, never funded by the former empty-initializer envelope. No nonpanicking, complete all-path bound or allocator-latency claim follows.

##### Approved admission decision — design/quotation only

The selected design is a **private, shared, method-neutral `OriginalAdmission` under a compiler-owned local job**. A private issuer consumes that job's sole issue right. There is no public Clone/adopt/reset/extract-ledger API. Each job has one bootstrap entry, burned **before validation**; refusal, abandonment or Drop never recreates either right. This is a local ownership/lifecycle contract, not cross-job identical-input deduplication or a ciphertext/producer authenticity claim. These are design obligations, not installed enforcement.

Issuer setup constructs the original `WorkLedger` **once, before quotation**, and retains the existing incoming `OwnedBudget`; no replacement budget, Arc authority or bootstrap-created ledger. Future bootstrap **borrows admission until successful initial reserve**. Earlier failure retains both originals through `cleanupTerminal`; only after reserve succeeds do that same ledger and budget move Capture→actual F→Payload. Later failure retires initialized owners and captures before charge and allows no retry. Successful publication leaves admission in terminal **Published** history; it cannot issue or bootstrap again. The root retains its original shared **32 construction/execution attempts**, unchanged; bootstrap entry is not a new root-attempt allowance. Ring/NTT/init/copy work after publication stays on that root's original ledger. The last Payload budget field and selected array-box attachment/plumbing remain unchanged.

`SourceAdmission` (`crates/pllm-compiler/src/original_source.rs`, struct at 236, impl at 332) is a **custody/sequencing pattern only**: do not reuse its native prices, ledger or units. Astra independent review 13 conditionally clears custody only, not concrete access paths, setup/entry/body/cleanup quotes or complete funding feasibility. Source supplies no implemented clear-spectrum issuer; the incoming byte handle, request identifier or caller ledger cannot substitute for admitted work authority.

**Approved logical-allowance origin — AUTHORITY DESIGN FOR QUOTATION ONLY (decision 72861):** each compiler-owned job originates a fixed logical allowance `L`; irrevocable startup `S` is consumed **before enrollment, issuer-right installation, ledger creation or checks**. The sole ledger is created with remaining `R=L-S`, while retaining `work_limit=L`, so `consumed_work` starts at `S` and adds each successful debit exactly once. Require `0 <= S <= L <= 65536`; applicable control ceilings remain unchanged. Failed debit leaves balances unchanged by that debit, **not a startup refund**. No limit renewal, retry credit, work refund or caller-ledger adoption is authorized. This selects a proposed origin for quotation, not actual approved funding, numeric S/L, prices, tariffs, a funded bound or feasibility. Earlier “no funding source authorized” wording must distinguish this approved proposed origin from still-unapproved actual funding.

Startup must cover issuer/setup/refusal/retirement administration. Separately funded policy establishment is explicitly **uncovered** by this selection. Bootstrap/root cleanup requires irrevocable original-ledger prepayment, without replenishment; no future debit pays past work, no prepaid event is charged twice, and unused cleanup or abandonment authorizes no fresh work. Proposed `admin(0,w)` consumes both shared total and control balances; cleanup earmarks remain nonspendable. **MAX_WORK=65536**, existing control limits, source-word ceiling **8192**, byte ceilings and all earlier numerical contracts remain unchanged. Review found no intrinsic accounting contradiction in the proposed origination; existing work clamps are not feasibility proof.

**Issuer failure-ownership CANDIDATE 22 — SELECTED FOR QUOTATION ONLY by user decision 73115.** This selects the consuming IssueGuard/EntryGuard/latched-cleanup corridor and setup below as quote design, not actual Rust enums/latches/APIs, prices, funding, feasibility, implementation or execution. Private `OriginalJob` holds `Option<IssueRight>` / `Option<OriginalAdmission>`; the right owns the budget. Consuming it immediately installs Terminal admission with budget Some, ledger None, history BeforeLedger and cleanup IssuerDue, before policy checks or fallible work. An armed `IssueGuard` covers setup; the sole ledger is created resident there. Ready is permitted only after full entry and cleanup debits succeed. Failed prepayment does not retrospectively fund setup or retirement: the issuer-fixed envelope remains an independent obligation.

On failure, mark Terminal before potentially panicking cleanup; take/drop the budget and mark it retired only after Drop returns. Retain the original ledger/history in the job: before-ledger failure has ledger None; after-ledger failure leaves exact original balances queryable. Successful root transfer records only a historical remaining/control snapshot, not final spending or a refund; RootOwned plus Published retains no entry right or extra attempt credit. Root-shared 32 attempts remain unchanged.

Partial takes must retire locals/captures before charge. The concrete proposed guard/declaration/scope corridor below is conditionally reviewed, not adopted; actual layouts and the full all-path quote remain OPEN. `try_build` inline growth before invoking F must include early reserve refusal and capture drops. No handle `try_clone`, extraction or adoption is introduced; the shared charge's existing internal Arc clone must be quoted, not prohibited. Separate new jobs may admit identical inputs.

**Invocation bound remains OPEN:** `FnOnce` alone is insufficient because a separate Drop caller exists; nonrenewable rights do not bound unlimited refusal calls. Astra 13's concrete proposed outer/Drop corridor below has independent Astra 11 conditional review; decision 73115 now selects that corridor and setup for quotation only. Full helper/callee census, complete quotation, enforcement and funding coverage remain OPEN. The proposed latch precedes potentially panicking cleanup; budget None is **not proof of retirement**. Historical decision 72861 did not itself approve Candidate 22 or any new Candidate 51/K62 table/cleanup enum; the later quote-design selection is not actual enum/latch adoption. Astra 31 independent review 16 conditionally freezes only the repaired empty-bootstrap selected-occurrence schema below, unpriced and unfunded; this is not new authority approval or implementation adoption. Actual original pre-entry funding remains required; no “single funding gate” establishes a complete quote. Any generic `new(P.limit, P.limit)` must preserve existing total/control ceilings and the selected `R=L-S` origination rather than silently restoring L. Current `ClearInputBuilder::new` uses `new(work_limit, work_limit)` at line 202 with MAX_WORK=65536; no new control field or actual-ceiling defect is inferred. Generic WorkLedger clamps are not original-unit admission proof.

**Candidate 22 — SELECTED FOR QUOTATION ONLY (user 73115); Astra 13 corridor/Astra 18 setup notation.** The previously unselected consuming-boundary/custody proposal, conditionally reviewed by Astra 73017 and independently by Astra 11, is now selected together with setup as quote design. This does not retroactively change those reviews into approval, select a new candidate/full K table, or adopt actual Rust enums/latches/APIs. The primitive corridor and setup are concrete quotation notation; whole-lifecycle invocation, full callee/occurrence census, all-path quotation, enforcement, prices, funding and feasibility remain **OPEN**. Preserve superseded bodies through Git history, not copies.

**Selected quote-design Rust-like body (verbatim quotation notation, not implementation):** `ResultRoot` denotes root/error; `Allowance`, `EntryError`/error and status names do not adopt actual types. `ORIGIN_FIXED_L_CONSUME_S_BEFORE_ENROLLMENT` denotes the approved logical origin, **not an actual funding API**: fixed L, irrevocable S consuming **both total and control**, `R=L-S`, retained `work_limit=L`, `0 <= S <= L <= 65536`, unchanged applicable control ceilings. Only this private outer boundary constructs jobs; every new job consumes its own S, with no cross-job deduplication or global finite-external-call promise.

```rust
struct IssueRight(OwnedBudget); // No custom Drop.
struct OriginalJob { right: Option<IssueRight>, admission: Option<OriginalAdmission> }
struct IssueGuard<'a> { job: &'a mut OriginalJob, armed: bool }
struct EntryGuard<'a> { a: &'a mut OriginalAdmission }
fn outer(p: &ValidatedPolicy, incoming: OwnedBudget, choice: Choice) -> ResultRoot {
    let r = ORIGIN_FIXED_L_CONSUME_S_BEFORE_ENROLLMENT; // R=L-S; work_limit=L.
    let job = OriginalJob { right: Some(IssueRight(incoming)), admission: None };
    let driver = move || { let driver_result = consume_job(job, p, r, choice); driver_result };
    let outer_result = driver(); return outer_result;
}
fn consume_job(mut job: OriginalJob, p: &ValidatedPolicy, r: Allowance, choice: Choice) -> ResultRoot {
    let mut issue = IssueGuard { job: &mut job, armed: true };
    let taken_right = issue.job.right.take();
    let IssueRight(incoming) = match taken_right { Some(right) => right, None => return Err(Resource) };
    let initial = OriginalAdmission { budget: Some(incoming), ledger: None,
        phase: Terminal, history: BeforeLedger, cleanup: IssuerDue, dispatch_latched: false };
    issue.job.admission = Some(initial); // Previous value is structurally None.
    let setup_result = setup_once(&mut *issue.job, p, r);
    match setup_result { Ok(()) => (), Err(error) => return Err(error) }
    match choice { Abandon => return Err(Resource), Enter => () }
    issue.armed = false; drop(issue);
    let a = match job.admission.as_mut() { Some(a) => a, None => return Err(Resource) };
    let entry_result = bootstrap(a, p); return entry_result;
}
impl Drop for IssueGuard<'_> { fn drop(&mut self) {
    if self.armed { match self.job.admission.as_mut() { Some(a) => dispatch(a), None => () } }
} }
impl Drop for OriginalJob { fn drop(&mut self) {
    match self.admission.as_mut() { Some(a) => dispatch(a), None => { let right = self.right.take(); drop(right); } }
} }
impl Drop for EntryGuard<'_> { fn drop(&mut self) { dispatch(&mut *self.a); } }
fn dispatch(a: &mut OriginalAdmission) {
    if a.dispatch_latched { return; } a.dispatch_latched = true;
    if matches!(a.history, Published) { return; }
    a.phase = Terminal;
    let pending = matches!(a.cleanup, TransferPending);
    let issuer = matches!(a.cleanup, IssuerDue);
    let taken_budget = a.budget.take();
    match taken_budget { Some(budget) => { drop(budget);
        if !pending { a.cleanup = if issuer { IssuerRetired } else { EntryRetired }; }
    }, None => () }
}
fn bootstrap(a: &mut OriginalAdmission, p: &ValidatedPolicy) -> ResultRoot {
    K62_CHECK_READY; a.phase = Entering; let entry = EntryGuard { a };
    K62_VALIDATE_AND_DECLARE_CHARGE_USING(entry.a, p);
    let owned = { K62_SNAPSHOT_AND_TRANSFER_PENDING_USING(entry.a);
        let ledger = match entry.a.ledger.take() { Some(x) => x, None => return Err(Resource) };
        let budget = match entry.a.budget.take() { Some(x) => x, None => return Err(Resource) };
        let capture = Capture { ledger, budget }; let f = make(capture); charge.try_build(f).map_err(from_shared)? };
    let root = ClearInputBuilder { owned }; entry.a.cleanup = RootOwned; entry.a.history = Published; return Ok(root);
}
```

**Inline-block and borrow interpretation.** `K62_*` stands for unchanged documented **inline blocks**, not new runtime helpers. The validation block declares `let charge = ...` in bootstrap's scope after `entry`; its validation/reserve and the snapshot/TransferPending block are the existing K62 operations, not additional calls or credits. Each use of `entry.a` for a callee means a mutable reborrow **`&mut *entry.a`**, never moving a field out of a Drop type. The immutable `entry` binding can mutate through its `&mut` field. `EntryGuard` dispatches only from Drop, with no explicit cleanup method. The inner transfer scope and original K62 initializer/capture field order retire initialized transferred locals/Capture/actual F/partial Payload before charge, then EntryGuard; early unwind before its installation does not establish K62 guard coverage.

**Astra 11 conditions (preserved, not blanket approval of later setup).** The job precedes its armed IssueGuard and remains exclusively borrowed through `issue.job` until disarm/drop. Taking/destructuring the no-custom-Drop IssueRight and installing the complete false-latch Terminal admission into structurally None form the proposed callback-free, allocation-free, nonfallible primitive corridor, with no hidden overwritten-owner Drop. The later selected method-neutral `setup_once` notation below constructs the sole original ledger with R and requires complete prepayment before Ready; its full callee quote remains OPEN. Admission Some implies right None. Job Drop borrows `admission.as_mut()`, falling back to `right.take()` only when absent. Custom Drop still automatically destroys fields; those and local destructors are real separately quoted work, not extra dispatcher calls.

Published requires resident **budget None AND ledger None**, with the actual root owning both originals and no job-to-root access path; the body takes both before publication. The latch marks an **attempt**, not retirement. Budget Drop must return normally before the plain retirement-state assignment, whose actual primitive must be nonpanicking or status remains unknown. TransferPending history/status never certifies transferred-owner retirement, including after residual resident cleanup. None, latch state, panic or double-panic abort supplies no retirement evidence; cleanup remains UNKNOWN absent normal retirement. Unmoved original ledger/history stays resident until job destruction.

**Common caller counts, conditional on no helper/reentrant dispatch:** the body has exactly three syntactic `dispatch` callers (the three Drops). Per job, pre-installation unwind dispatches zero while admission is absent; setup/abandon has IssueGuard+job <=2; Enter before EntryGuard installation has job <=1; entry/transfer/publication has EntryGuard+job <=2. Setup and bootstrap caller invocations are each <=1; substantive dispatcher cleanup is <=1. These do not extend to whole-helper invocation/termination bounds, implicit destruction or a complete quote; the full callee census remains open, and FnOnce is not receiver-slot coverage.

**Source-selected partial additional destinations — not a complete census or actual layouts.** Counts are conditional move/logical destinations, not duplicate authority/backing; nested Some payloads and guard fields alias their aggregates. K62 ledger/budget/Capture/F/owned/root routing and cleanup **replace** corresponding existing occurrences, never add duplicate cleanup credit.

| Proposed type/slot | Selected destinations/count | Qualification |
| --- | --- | --- |
| `&ValidatedPolicy` p / `Choice` choice / `Allowance` r | 3 each: outer parameter/local, closure capture, consume parameter | Origin result is notation, not a funded API. |
| `OriginalJob` | 3: outer local, closure capture, consume parameter | Same job moves; not three jobs. |
| actual driver closure value | 1 | Does not cover invocation receivers. |
| `IssueRight` construction / `Option<IssueRight>` taken_right / matched right | 1 each | No custom Drop; nested fields alias. |
| outer incoming / driver incoming `OwnedBudget` | 1 each | Same incoming handle, no new Arc authority. |
| initial `OriginalAdmission` / installed destination | 1 each | Complete false latch before installation into None. |
| `IssueGuard` | 1; fields alias | `EntryGuard` routing/receiver reconciles existing K62. |
| setup_result / error binding | <=1 each | Setup internals unquoted. |
| driver a / entry_result | <=1 each | Reborrows/return moves not exhaustive. |
| driver_result / outer_result / outer return destination | <=1 each | Partial return inventory only. |
| fallback right Option / drop destination | <=1 each | Automatic destruction separately quoted. |
| dispatcher receiver | <=2 | Only under the no-helper/reentrant-dispatch condition. |
| pending / issuer / taken_budget / matched budget / drop destination | <=1 each | Latch field aliases admission; only substantive dispatch counted here. |
| each Drop receiver | <=1 each | Field/local destruction is separate work. |

`shared_allocation.rs:524–540` establishes builder-before-charge cleanup and `:550–555` payload-before-charge; `OwnedBudget::drop` at `:410–414` includes release plus Arc destruction, and `OwnedExternalCharge::drop` is at `:544–547`. `clear_input_owner.rs:104–107` Data wipe is a separate explicit callee. These paths are neither zero work nor a nonpanic guarantee. Actual setup, destructor/callee/argument slots, take/drop-call parameters, reborrows, return moves and full multiplicities/layouts remain unquoted; no sizes, prices or complete bound follow.

**Selected setup quote design — user 73115, Astra 18 notation (not implementation).** Startup S consumes both total/control before enrollment; retained L and the sole `new(R,R)` represent initial S consumption once, not fresh work. `row_fold.rs:29–59` stores only remaining total/control with independent clamps; `clear_input_owner.rs:202,440–441` retains work_limit outside the ledger and computes consumed work as limit minus remaining. Preserve those actual clamps and unchanged ceilings; do not restore L as the remaining balance.

```text
ORIGIN(p): irreversibly consume fixed S from this job fixed L before enrollment/checks; r := L-S (invariant incoming validated policy); continue existing job/IssueGuard installation with r.
setup_once(job:&mut OriginalJob,p:&ValidatedPolicy,r:usize):
 a:=match job.admission.as_mut():Some(a)=>a;None=>Err(Resource)
 if job.right.is_some():returnErr(Resource)
 if a.phase!=Terminal or a.history!=BeforeLedger:returnErr(Resource)
 if a.cleanup!=IssuerDue or a.dispatch_latched:returnErr(Resource)
 if a.budget.is_none() or a.ledger.is_some():returnErr(Resource)
 a.ledger:=Some(WorkLedger::new(r,r)) // structurally None; sole construction
 a.history:=AfterLedger // proposed plain historical marker
 ledger:=match a.ledger.as_mut():Some(x)=>x;None=>Err(Resource)
 match ledger.debit(0,E):Ok(())=>();Err(_)=>returnErr(Work)
 match ledger.debit(0,K):Ok(())=>();Err(_)=>returnErr(Work)
 a.cleanup:=EntryDue
 a.phase:=Ready
 returnOk(())
```

E and K are fixed, **UNPRICED** entry/cleanup administrative envelopes, not caller values; r is private origin output. This finite setup body has no loops, allocation, callback, dispatch, retry or budget operation: constructor <=1, ordered debits <=2. Each successful `debit(0,...)` consumes both total/control; successful E stays burned if K fails, while a refused debit changes neither balance. Failure keeps the original ledger resident under the existing cleanup custody; Ready follows only both successful debits. These prepayment earmarks are **already spent**, not extra balances or authority to debit covered work again. Setup/failure administration belongs to S; refused prepayment cannot retroactively fund it. ORIGIN's logical event/subtraction is not actual funding realization; its implementation and the full setup/callee quote remain OPEN, including independent review before quotation completeness.

**Finite partial setup inventory delta — not a complete K table or layout quote:**

| Selected setup slot/destination | Partial count/obligation |
| --- | --- |
| job/p/r parameters and associated reborrows | One parameter each; callee/receiver reborrows still require full quotation. |
| `Option<&mut OriginalAdmission>` / matched admission borrow | <=1 each; resident admission fields alias their owner. |
| `WorkLedger` constructor return / install destination | <=1 each; same sole original ledger, not two owners. |
| `Option<&mut WorkLedger>` / matched ledger borrow | <=1 each; resident ledger storage not counted again. |
| debit Results | <=2, ordered E then K; branch payloads alias Results. |
| E/K arguments | <=1 each; fixed administrative envelopes, not new credit. |
| error bindings/adapters and return destinations | Finite branches as shown; complete caller/callee/move census still OPEN. |

This partial delta reconciles existing setup/caller and K62 destinations; no duplicate K62 occurrence or cleanup credit, actual sizes or complete K claim follows. Issuer startup/administrative retirement belongs to S; entry/root cleanup requires ordered irrevocable E/K prepayment on the same original ledger, without double payment, replenishment or refund credit. Incoming policy establishment/validation, budget acquisition and **argument delivery** remain outside the origin and unquoted/unfunded; S cannot retroactively pay pre-origin work. The corridor/setup are selected for quotation only; full quotation, original-unit prices, actual funding/feasibility and enforcement remain OPEN. Sole original ledger, root-shared **32 attempts**, source-word ceiling **8192**, all caps, native HOLD and cleanup UNKNOWN remain unchanged. No actual Rust enum/latch/API, original-origin implementation, execution or HOLD recovery is authorized.

**Retained-L representation and corrected partial census — USER73222 selects FOR QUOTATION ONLY.**
Astra review 10 identified a gap: the earlier selected initializer uses
`Header.work_limit := P.limit`, not a value proven to be the original L.
USER73222 selects the reviewed private L/R representation and CORRECTED partial
census below, following Astra 7 (`ses_e75de7369ffeS2SMqTvY8tKRlI`). Older sketches
remain historical quotation records; their affected initializer/setup/transfer
operations are superseded FOR QUOTE DESIGN by this selection. Conditional source
clearance is not a whole quote, funded bound, Rust/API adoption or execution.
The selected path realizes user 73115's retained-L quotation intent as follows:

Astra 16 conditionally source-cleared the document, not whole delivery/quotation.
Astra 20 found NO existing proof that P.limit meant the origin ceiling: in the
pre-decision snapshot, proposed fields (540–543), policy text (564), initializer
(576), and numeric validator (836) do not establish original L's provenance.
Actual `clear_input_owner` constructor work_limit is accounting, not proposed P.
USER73191 now approves an ADDITIONAL design rule FOR QUOTATION ONLY: after
nonrefundable startup S, reject private original `L > P.limit` before ledger
creation, preserving ALL post-start R/work/control policy constraints. This
rejects `R <= P.limit < L`; it is not merely notation or an existing proof.
No equality, new parameter/API, funding, implementation or execution is approved.

1. Refine the existing private `Allowance` to `{ original_l: L, remaining_r: R }`.
   The sole private origin irreversibly consumes S against both total/control
   before checks, with `R=L-S`; no caller-supplied or reconstructed pair.
   Preserve `0 <= S <= L <= 65536` and every applicable control/policy check.
2. Transport the SAME pair through existing outer local → actual driver capture
   → consume parameter, with the same job/IssueGuard right and incoming budget.
   No second origin, balance, issuer right or authority is created by transport.
3. The initial Terminal admission (`OriginalAdmission`) includes immutable
   `original_l: allowance.original_l` before fallible setup; `ledger: None` and
   original budget custody remain as before, including installation into None.
4. Existing setup job/p/r parameters receive `allowance.remaining_r` directly,
   with no selected argument temporary. After existing structural checks bind
   `l := a.original_l`, `ceiling := p.limit` by direct private-field reads;
   `if l > ceiling { return Err(Policy) }` precedes sole ledger creation.
   Preserve required post-start R/work/control constraints and the validated-policy
   input invariant; do not move the full validator before origin or equate P.limit
   with L. Refusal leaves the budget guarded and ledger None, with S nonrefundable.
   Sole `WorkLedger::new(r,r)`, `R=L-S`, then ordered `debit(0,E)`/`debit(0,K)`
   precede Ready; both balances are consumed, successful prepayments stay burned.
5. AFTER initial reserve, inside the existing transfer block copy `original_l := entry.a.original_l`,
   then use the existing ledger/budget takes to form
   `Capture { ledger, budget, original_l }`. `make` captures the WHOLE Capture
   in actual F, and F moves that Capture into `empty`; no parallel scalar route.
6. `empty` destructures all three fields. Header INSIDE Payload assigns its
   existing `work_limit := original_l`, not P.limit, Data or a new Header field.
   Published may retain scalar L, but no authority or job-to-root reentry path.

**Typed partial replacement delta — not a complete census, layout or price:**

| Affected occurrence | Re-quotation obligation |
| --- | --- |
| Existing Allowance outer local / driver capture / consume parameter | Replace all three aggregate destinations; fields alias them, no double backing. |
| Initial/installed OriginalAdmission | Both aggregate destinations gain original_l; fields alias containers, outside K62. |
| Setup checks, outside bootstrap K62 | Two usize bindings l/ceiling, one comparison-expression bool, one new Policy-branch Result CONSTRUCTION <=1; no adapter. |
| Setup return/caller destinations | Existing function RETURN <=1, setup_result and error binding unchanged; error payload aliases Result. Construction and return are distinct logical destinations, not peak stack. |
| Setup field reads | No extra operand bindings, getter, Policy copy or selected argument temporary; existing job/p/r parameters remain. |
| Transfer / `empty`, K62 row 14 | Two extra usize bindings: transfer L and destructured L; replace count 2 with 4 including existing remaining/control snapshots. |
| K62 row 21 / row 22 | Replace layout witnesses for three Capture / three actual-F destinations respectively. |
| K62 rows 24/26 | Existing one Header / two Payload destinations have new work_limit provenance; no added Header field. |

Whole K, helper/destructor census, layout and numeric funding remain unclosed;
future actual-F size witnesses must be revisited. No `+sizeof(usize)` shortcut or
assumption that K62 remains a complete total applies. No fresh budget, ledger,
Arc, balance or change to root-shared 32 attempts follows. S/E/K meanings, caps,
native HOLD and cleanup UNKNOWN stay unchanged. Complete issuer/Drop/argument/callee
and all-path census, origin API/return realization and pre-origin argument delivery
remain unproved; layouts, tariffs and actual funding remain OPEN. No code, tests,
execution, import, build, probe, slot access or recovery is authorized here.

**Selected-sum helper — Astra 46 body frozen for design only.** `checked_selected_sum<const K>(terms: [usize; K]) -> Result<usize, EntryError>` calls `into_iter` once, starts sum at 0 and loops on `next`: None returns Ok(sum); Some(value) binds term, then `added = sum.checked_add(term)`; Some(value) updates sum, None returns Err(Capacity). No allocation, multiplication, subtraction or ledger operation occurs.

Selected callee inventory: one parameter array, one IntoIter, two `Option<usize>`, four `usize` slots (sum/term/two value bindings), one Result. Caller inventory: one completed argument array and one received Result. Payload fields alias aggregates; do not count the iterator's internal array again. Normal path has K+1 next calls and K adds; overflow at j has j next calls/j adds and remaining K-j usize drops; iterator retires once.

Generic/full-adapter K remains UNFROZEN; the conditional empty-bootstrap schema below selects **K=62** only for its stated body and selected callees. The helper's own array/iterator-size expressions depend on K but add no further rows or numeric recursion. Review 16 conditionally froze the original schema's products, term construction, checked credits and refusal prefixes, not actual layout values, complete B, the whole lifecycle or funding; USER73222's revision is qualified separately below. Actual typed layouts, issuer/caller/Drop paths, attachments, prices and funding feasibility remain unfinished; H>G and B+H constraints and all work/control/source/byte caps are unchanged. No representation Rust adoption, pricing, execution or HOLD change follows.

**USER73222/Astra 7 scoped revision:** original review 16 froze row 14 at count 2;
the revised count 4 below is conditional on Astra 7's corrected census and this
selection, not retroactive review-16 clearance. Keep 62 GROUPED rows/products:
rows 49/50 remain array 2 / iterator 1; rows 54/55/56 remain 186/62/186.
Only for this selected grouping, sum success stays 63 next/62 adds and the stated
refusal-prefix formulas stay unchanged. Replace all three Capture and all three
actual-F layout witnesses; reevaluate affected layouts/products/byte inequalities,
not `+sizeof(usize)` or a new numeric B/K/full bound. Setup delta stays outside K62.

##### Astra 31 review 16 — conditional empty-bootstrap schema, with USER73222 scoped revision

**Scope/status.** Original review 16 conditionally froze the repaired bootstrap-only schema; USER73222/Astra 7 revise affected occurrences as qualified above, still **unpriced and unfunded**. It selects logical occurrences, not physical simultaneous stack occupancy, RSS or hidden compiler/library/interpreter frames or allocator internals. Incoming resident admission, original ledger, incoming budget/Arc backing, issuer/setup and incoming quotation authority remain separate obligations, not additional credit. Endpoint retirement is of an **empty root without construction**; attachment, execution and populated-root cleanup are outside this schema. No actual sizes were queried and no compilation occurred. This is neither a full quote nor new authority, enum, Rust implementation or cleanup-certification approval.

**Complete scoped body (quotation description, not adopted Rust).** Input is a borrowed Ready `OriginalAdmission`, resident original ledger/budget and already prepaid entry authority. Check Ready; set Entering and immediately install `EntryGuard`, with no intervening fallibility. The guard is declared before all transferred locals. Validate the Quote as below. An explicit `match` on `budget.as_ref()` binds Some or returns `EntryError::Resource` for None; call `charge_external(q.b).map_err(from_shared)?` **before any takes**. An explicit `match` on `ledger.as_ref()` likewise binds Some or returns Resource; obtain `remaining` and `control_remaining` snapshots through its two getter receivers. Record Transferred history and TransferPending cleanup; end those borrows. After initial reserve copy `original_l := entry.a.original_l` in the transfer block. Explicit `match` on `ledger.take()` and then `budget.take()` binds each Some local, with each None arm returning Resource; there are no `ok_or(...)?` Results outside the inventory. Construct the same Capture with ledger/budget/original_l; existing `make` captures the WHOLE Capture in actual F, which moves it into `empty`, then `charge.try_build(F).map_err(from_shared)?` completes. `empty` destructures all three fields and assigns existing Header.work_limit from original_l, not P.limit. Construct `ClearInputBuilder { owned }`, record RootOwned and Published, return Ok(root). These USER73222 operations supersede affected older initializer/transfer quotation operations, not actual Rust; no additional ledger, F, Capture, debit or root-attempt allowance is created.

**Guard/refusal body.** Successful RootOwned/Published cleanup is a no-op. Otherwise the guard's explicit admission reborrow sets Terminal before the explicit budget `take` match and potentially panicking drop: Some(budget) is dropped, and EntryRetired is recorded only after Drop returns; None alone proves no retirement. TransferPending remains unconfirmed, not EntryRetired by inference. Transferred locals/captures retire before the charge; partial initialization retires only initialized fields and remaining captures. The schema assumes a single guard cleanup-body call; `FnOnce` does not prove that bound. Concrete setup/caller/Drop cleanup-once bodies and their funding remain unknown; their single-invocation latch precedes potentially panicking cleanup.

**Validator body.** Borrow policy and identity; check identity, `limit <= min(65536, MAX_CLEAR_WORK, 1<<20)`, `postledger_control <= limit`, source-word ceiling `<=8192`, and all four entry envelopes against their fixed symbolic ceilings; refusal is Policy. These checks prove neither funding nor remaining balances. Obtain the actual F Layout using the `fn(Capture)->F` witness and `Layout::new::<F>()`, constructing no F or Capture. Let `G=s(OwnedExternalCharge)` and `H=s(OwnedCharged<Payload>)`; checked `delta=H-G` must be positive. In table order, construct 62 checked multiplicity products and pass the completed array to `checked_selected_sum`, producing raw. Compute checked `S_layout=raw-H`, `pool=P.source_word_ceiling*s(u64)`, `B=S_layout+pool`, `initial=B+G`, `grown_payload=B+delta`, `final=B+H`; require B, initial, grown_payload and final each within the root-byte ceiling. These are seven checked arithmetic operations including delta, seven Capacity-adapted Results and the received sum Result. The source candidate's actual four-field Quote selection carries **B, control, source ceiling and F Layout**; this selection is not the historical four-usize Quote above, nor Rust adoption or approval of executable field declarations. `q.b` names the B field already used by the source candidate; no additional field names are selected here. No actual layout values or H>G proof follow. **S_layout is a byte-layout subtotal, never approved startup S.**

**Product body.** Each `product(count: usize, size: usize) -> Result<usize, EntryError>` performs checked_mul, adapts None to Capacity with `?`, binds bytes and returns `Ok(bytes)`; no allocation. Each call selects three usize destinations, one Option and three Results (adapted, returned, caller-received). Product failure at row j retires the prior j-1 initialized usize array prefix and never invokes the sum. Successful sum performs 63 next calls and 62 adds; overflow at j performs j next/j adds and retires the remaining 62-j usize elements with the iterator once. K is the **number of rows**, not the sum of multiplicities; array/IntoIter sizes depend on K without another term or recursive census. Aggregate fields, match payloads and array elements alias their aggregate storage.

| Row | Count | Type | Selected destinations |
| --- | ---: | --- | --- |
| 01 | 2 | `&mut OriginalAdmission` | bootstrap parameter; cleanup receiver |
| 02 | 1 | `EntryGuard` | immediate entry guard |
| 03 | 1 | `&mut EntryGuard` | Drop receiver |
| 04 | 1 | `&InternalPolicy` | validator policy borrow |
| 05 | 1 | `&PolicyIdentity` | identity borrow |
| 06 | 1 | `fn(Capture)->F` | witness parameter |
| 07 | 2 | `Layout` | witness return; validator local |
| 08 | 1 | `Quote` | caller q |
| 09 | 1 | `Result<Quote,EntryError>` | validator return |
| 10 | 10 | `usize` | G, H, delta, raw, S_layout, pool, B, initial, grown_payload, final |
| 11 | 7 | `Option<usize>` | validator checked arithmetic |
| 12 | 8 | `Result<usize,EntryError>` | seven adaptations; received sum |
| 13 | 3 | `&WorkLedger` | checked reference; two getter receivers |
| 14 | 4 | `usize` | remaining snapshot; control_remaining snapshot; transfer original_l; empty-destructured original_l (USER73222/Astra 7 revision) |
| 15 | 2 | `&OwnedBudget` | checked reference; charge-call receiver |
| 16 | 1 | `Option<&OwnedBudget>` | checked budget reference |
| 17 | 1 | `Option<WorkLedger>` | ledger take |
| 18 | 2 | `Option<OwnedBudget>` | transfer take; cleanup take/drop |
| 19 | 2 | `WorkLedger` | taken local; empty destructure |
| 20 | 2 | `OwnedBudget` | taken local; empty destructure |
| 21 | 3 | `Capture` | construction; make parameter; empty parameter |
| 22 | 3 | actual `F` | make return; caller builder; try_build parameter |
| 23 | 1 | `Data` | empty initializer |
| 24 | 1 | `Header` | empty initializer |
| 25 | 1 | `Caps` | empty initializer |
| 26 | 2 | `Payload` | empty return; try_build local |
| 27 | 1 | `Result<Payload,SharedError>` | closure return |
| 28 | 2 | `OwnedExternalCharge` | caller; try_build receiver |
| 29 | 1 | `Result<OwnedExternalCharge,SharedError>` | initial charge |
| 30 | 1 | `Result<OwnedExternalCharge,EntryError>` | charge adaptation |
| 31 | 1 | `OwnedCharged<Payload>` | caller owned |
| 32 | 1 | `Result<OwnedCharged<Payload>,SharedError>` | build return |
| 33 | 1 | `Result<OwnedCharged<Payload>,EntryError>` | build adaptation |
| 34 | 1 | `ClearInputBuilder` | root |
| 35 | 1 | `Result<ClearInputBuilder,EntryError>` | endpoint; final resident H credited once |
| 36 | 2 | `SharedError` | adapter parameters |
| 37 | 8 | `usize` | four shared locals: charge bytes, inline bytes, payload bytes, resize bytes; four parameter destinations: charge_external payload_bytes, try_build payload_bytes, two reserve_bytes bytes |
| 38 | 4 | `Option<usize>` | shared arithmetic |
| 39 | 4 | `Result<usize,SharedError>` | shared adaptations |
| 40 | 3 | `Result<(),SharedError>` | two reserves; resize |
| 41 | 2 | `&mut OwnedExternalCharge` | resize; Drop |
| 42 | 2 | `&OwnedExternalCharge` | payload_bytes; budget receivers |
| 43 | 5 | `&Budget` | two reserves; two releases; builder argument |
| 44 | 1 | `&mut OwnedBudget` | Drop |
| 45 | 2 | `usize` | release terms |
| 46 | 2 | `&mut Data` | Drop; wipe |
| 47 | 1 | `&mut [f64]` | empty wipe |
| 48 | 1 | `&mut [u64]` | empty wipe |
| 49 | 2 | `[usize;62]` | caller array; sum parameter |
| 50 | 1 | `array::IntoIter<usize,62>` | sum iterator |
| 51 | 2 | `Option<usize>` | sum next; added |
| 52 | 4 | `usize` | sum; term; two value bindings |
| 53 | 1 | `Result<usize,EntryError>` | sum return |
| 54 | 186 | `usize` | 62 product count/size/bytes destinations |
| 55 | 62 | `Option<usize>` | product arithmetic |
| 56 | 186 | `Result<usize,EntryError>` | product adaptation/return/caller-received |
| 57 | 1 | `&mut Option<WorkLedger>` | take receiver |
| 58 | 2 | `&mut Option<OwnedBudget>` | two take receivers |
| 59 | 1 | `Option<&WorkLedger>` | checked ledger reference |
| 60 | 1 | `&Option<WorkLedger>` | as_ref receiver |
| 61 | 1 | `&Option<OwnedBudget>` | as_ref receiver |
| 62 | 1 | `&mut OriginalAdmission` | explicit guard-cleanup reborrow |

**Reviewed source correction and paths.** Source review at `shared_allocation.rs:398,475` and `reserve_bytes:101` accounts for the four parameter destinations in row 37: **4→8**, not four locals alone. Selected R/C/I counts are initial refusal **1/0/0**, growth refusal **2/1/0**, completed bootstrap **2/1/1**. Each created charge has one release; the incoming budget retires once, and Arc destruction is last-strong only when actually last. Positive H-G conditionally supports **B+G→B+H**, never B+G+H. Completed empty Data has two empty zeroizes and four Vec drops; Header and empty attachment retire before the last budget field, then the enclosing charge. Partial initialization includes only initialized owners. These counts and the selected occurrence union do not price administration, prove original-unit admission or cover issuer/job/caller/Drop cleanup-once funding. Full adapter/attachment/populated-root quotation remains open; native HOLD and cleanup UNKNOWN remain unchanged.

##### Astra 23 — pinned ledger-helper SOURCE-audit subquote (partial, outside K62)

This later source audit resolves the ordinary ledger-helper bodies below, not their
prices, a complete inventory or historical approval. Upstream Rust **1.85.1** annotated
tag `0035dbbca59afecf0b4e53d96d61e00680dca9de` resolves to commit
`4eb161250e340c8f48f66e2b929ef4a5bed7c181`. This is a public quotation oracle only,
NOT installed-binary or compiler-lowering provenance. All source links pin that commit.
The unapproved opaque-stdlib boundary proposal is discarded as unnecessary; the
specific intrinsic/language endpoint proposal below was **UNSELECTED at audit time**.

**Later decision USER73302 — SELECTED FOR QUOTATION ONLY:** expand the cited Rust
1.85.1 ordinary bodies; stop ONLY at the listed compiler intrinsics and primitive
language operations, including function-item invocation. Retain distinct selected
destinations; `?` delivers directly to existing total/residual parameter, with NO
extra extraction slot. Terminal work tariffs are symbolic NONZERO, values UNSET.
No ordinary-body exclusion or compiler-generated storage/instruction/RSS/time bound
is selected. This later user selection is not Astra 23 audit approval, does not close
K62/other helpers/iterator/layout or the full quote/funding, and authorizes no Rust
implementation, execution or HOLD recovery.

Let `c ∈ {0,1}` count sole-constructor reachability, `m=2c`, `n<=2` debit calls,
`o<=n` checked-add overflows and `s<=n-o` debit successes. Selected clear=0 gives
`o=0`. Successful E then refused K leaves consumption `S+E`, remaining `R-E`;
both succeed: `S+E+K`, remaining `R-E-K`. Refused debits leave balances unchanged.
These symbolic identities assign no S/E/K prices or funding.

The following supplements selected logical occurrences **OUTSIDE K62**; it adds no
K/layout values. Parameters, named bindings, constructed aggregates, callee returns
and caller receipts are separate selected destinations; fields alias aggregates.
Direct delivery counts a receipt at the selected destination ONLY once.

| Source-verified body | Multiplicity and partial selected occurrence/work facts |
| --- | --- |
| [`Ord::min`, cmp.rs:1001–1006](https://github.com/rust-lang/rust/blob/4eb161250e340c8f48f66e2b929ef4a5bed7c181/library/core/src/cmp.rs#L1001-L1006) | m: two usize parameters, min_by call, usize return; existing ledger fields are final receipts. |
| [`min_by`, cmp.rs:1443–1448](https://github.com/rust-lang/rust/blob/4eb161250e340c8f48f66e2b929ef4a5bed7c181/library/core/src/cmp.rs#L1443-L1448) | m: two usize parameters plus fixed usize::cmp function-item parameter; comparator invocation, Ordering receipt, match, usize return/received tail. |
| [`cmp`, cmp.rs:1711–1780](https://github.com/rust-lang/rust/blob/4eb161250e340c8f48f66e2b929ef4a5bed7c181/library/core/src/cmp.rs#L1711-L1780) | m: two &usize parameters; three_way_compare takes two usize inputs, returns intrinsic Ordering/receipt, then cmp returns; NOT two imagined bools. |
| [`checked_add`, num/uint_macros.rs:486–500](https://github.com/rust-lang/rust/blob/4eb161250e340c8f48f66e2b929ef4a5bed7c181/library/core/src/num/uint_macros.rs#L486-L500) | n: two usize parameters; add_with_overflow takes two usize inputs, returns/receives (usize,bool), whose fields alias; unlikely call/branch; one Option construction plus separate return/existing caller Option. |
| [`unlikely`, intrinsics/mod.rs:1528–1535](https://github.com/rust-lang/rust/blob/4eb161250e340c8f48f66e2b929ef4a5bed7c181/library/core/src/intrinsics/mod.rs#L1528-L1535) | n: bool parameter/return/receipt and branch; true invokes cold_path o times. Ordinary body is resolved, NOT opaque. |
| checked_add success arm, same pinned body | n-o: ADDITIONAL unchecked_add with two usize inputs and return DIRECT into Some payload; not overflowing_add and not a zero event. |
| [`ok_or`, option.rs:1252–1257](https://github.com/rust-lang/rust/blob/4eb161250e340c8f48f66e2b929ef4a5bed7c181/library/core/src/option.rs#L1252-L1257) | n: Option receiver, Error parameter, Some-arm usize binding n-o, match, Result construction/return; caller eager Error and Result stay separate. |
| [`Try::branch`, result.rs:2000–2016](https://github.com/rust-lang/rust/blob/4eb161250e340c8f48f66e2b929ef4a5bed7c181/library/core/src/result.rs#L2000-L2016) | n: Result receiver, usize v binding n-o / Error e binding o, match, ControlFlow construction/return/caller receipt; nested Break residual Err payload aliases ControlFlow. |
| `?` consumer | n ControlFlow decisions; as LATER SELECTED by USER73302, success delivers DIRECT to EXISTING total, failure DIRECT to residual parameter; NO extra extraction slot or invented compiler locals. |
| [`FromResidual`, result.rs](https://github.com/rust-lang/rust/blob/4eb161250e340c8f48f66e2b929ef4a5bed7c181/library/core/src/result.rs) | o: residual Result parameter, Error e binding, match, identity conversion call, final Err construction/return/receipt REPLACES existing overflow final Result construction; not another final Result. |
| [`From<Error> for Error`, convert/mod.rs:765–773](https://github.com/rust-lang/rust/blob/4eb161250e340c8f48f66e2b929ef4a5bed7c181/library/core/src/convert/mod.rs#L765-L773) | o: Error parameter and return DIRECT to Err payload; transfer is not automatically zero work. |

Language-call supplement: comparator function-item construction → compare parameter
→ FnOnce receiver; two-reference argument tuple → call boundary → cmp parameters;
Ordering return crosses invocation. [`FnOnce::call_once`, ops/function.rs:242–250](https://github.com/rust-lang/rust/blob/4eb161250e340c8f48f66e2b929ef4a5bed7c181/library/core/src/ops/function.rs#L242-L250)
is language-supplied: fabricate no Rust shim body. usize macro instantiation is
[`num/mod.rs:1212–1233`](https://github.com/rust-lang/rust/blob/4eb161250e340c8f48f66e2b929ef4a5bed7c181/library/core/src/num/mod.rs#L1212-L1233).

Terminal endpoints LATER SELECTED by USER73302 FOR QUOTATION ONLY: [`three_way_compare` / `add_with_overflow`, intrinsics/mod.rs:3242–3271](https://github.com/rust-lang/rust/blob/4eb161250e340c8f48f66e2b929ef4a5bed7c181/library/core/src/intrinsics/mod.rs#L3242-L3271),
[`unchecked_add`:3407–3413](https://github.com/rust-lang/rust/blob/4eb161250e340c8f48f66e2b929ef4a5bed7c181/library/core/src/intrinsics/mod.rs#L3407-L3413),
[`cold_path`:1484–1489](https://github.com/rust-lang/rust/blob/4eb161250e340c8f48f66e2b929ef4a5bed7c181/library/core/src/intrinsics/mod.rs#L1484-L1489),
and language calls/matches/copies/borrows/comparisons/subtractions. Intrinsic
unimplemented placeholders are compiler-overridden, NOT real panic paths.
USER73302 selects symbolic NONZERO terminal work tariffs, values UNSET, for this
quotation boundary only; no numeric prices, funding or Rust adoption follows.

Constructor installation, caller Option/Result, total, final debit return and setup
receipt count ONCE; each debit has exactly one final Result, even across generic
impossible branches. Selected clear=0 makes checked_add overflow impossible for
already-typed usize control; hence o=0 removes cold_path and residual conversion.
Independently, new(R,R), R<=65536 and identical successful decrements keep total
and control balances equal after identity clamps, making control refusal unreachable.
Ledger comparisons <=2n, subtractions 2s. Keep operation-work counts separate from
slot census and aggregate-field aliasing.
These ordinary helpers are now source-resolved, not unknown stdlib recursion;
other helpers (including selected-sum iterator) are NOT closed by this ledger audit.
Physical stack/RSS/time/hidden compiler state remain unproved; no blanket exclusion
of all std-internal storage follows. Full issuer/Drop/args/setup/origin/all-paths/
cleanup/other-helper census remains OPEN; root 32, caps, native HOLD and cleanup
UNKNOWN remain unchanged. No full quote, funding, execution or Rust adoption follows.

##### Selected-sum iterator — PARTIAL SOURCE AUDIT, not a selected complete boundary/census

This ordinary-source supplement concerns only the existing frozen
`checked_selected_sum` helper's `[usize;62]` IntoIterator internals. The public
Rust commit `4eb161250e340c8f48f66e2b929ef4a5bed7c181` remains a source oracle,
not a runtime dependency or installed-binary/lowering witness. The frozen outer
helper and all existing K62 counts remain unchanged. Do not re-count the original
iterator, helper Option receipts, sum, term or value bindings.

Let `(t,a)=(63,62)` on success and `(t,a)=(j,j)` on addition overflow at j:
t counts next attempts, a counts yielded elements/closure invocations. Entry
creates and retires the iterator once. These work-event multiplicities are not
byte counts or layout totals. The following are partial source occurrences,
not a newly selected complete iterator census.

| Pinned ordinary body | Partial occurrence/work facts |
| --- | --- |
| [`into_iter`, array/iter.rs:56–72](https://github.com/rust-lang/rust/blob/4eb161250e340c8f48f66e2b929ef4a5bed7c181/library/core/src/array/iter.rs#L56-L72) | Once: array receiver, `transmute_unchecked`, named `[MaybeUninit<usize>;62]` data binding, `zero_to(62)`, IntoIter construction and return into the existing iterator receipt. Fields alias their aggregate; distinct receiver/data bindings do not alias merely because fields do. No second caller iterator receipt. |
| [`IntoIter::next`, array/iter.rs:235–250](https://github.com/rust-lang/rust/blob/4eb161250e340c8f48f66e2b929ef4a5bed7c181/library/core/src/array/iter.rs#L235-L250) | t mutable receivers, t `IndexRange::next` calls, t closure constructions and t `Option::map` calls; closures invoked a times. Each invocation binds usize idx and returns usize into a Some payload through map. |
| [`zero_to` / `len` / `next_unchecked`, ops/index_range.rs:31–62](https://github.com/rust-lang/rust/blob/4eb161250e340c8f48f66e2b929ef4a5bed7c181/library/core/src/ops/index_range.rs#L31-L62) | `zero_to`: one usize parameter, aggregate construction/return into alive. `len`: t shared receivers, t usize returns/receipts and t intrinsic unchecked-sub calls. `next_unchecked`: a mutable receivers, a named usize value bindings, a usize `unchecked_add(1)` calls and a usize returns into range Some payloads. Its debug assertions are configuration-conditional, not absent/free by assumption. |
| [`IndexRange::next`, ops/index_range.rs:121–128](https://github.com/rust-lang/rust/blob/4eb161250e340c8f48f66e2b929ef4a5bed7c181/library/core/src/ops/index_range.rs#L121-L128) | t mutable receivers, t len calls and t `>0` decisions; a next_unchecked calls. Construct and return t Options; constructions, returns and caller receipts are distinct logical destinations, with payload fields aliasing aggregates. |
| [`Option::map`, option.rs:1114–1122](https://github.com/rust-lang/rust/blob/4eb161250e340c8f48f66e2b929ef4a5bed7c181/library/core/src/option.rs#L1114-L1122) | t Option receivers, t closure parameters, t matches, t Option constructions and t returns; a x bindings feed a closure idx parameters, with a closure usize returns into Some payloads. Map return → array-next receipt/return → existing helper Option receipt; preserve distinct boundaries without adding the existing receipt again. |

**Unclosed pointer/read/drop tail.** Each of a closure invocations calls
`get_unchecked` and [`assume_init_read`, mem/maybe_uninit.rs:721–727](https://github.com/rust-lang/rust/blob/4eb161250e340c8f48f66e2b929ef4a5bed7c181/library/core/src/mem/maybe_uninit.rs#L721-L727);
the latter calls `assert_inhabited`, `as_ptr` and `read`. Iterator
[`Drop`, array/iter.rs:350–358](https://github.com/rust-lang/rust/blob/4eb161250e340c8f48f66e2b929ef4a5bed7c181/library/core/src/array/iter.rs#L350-L358)
runs once through `as_mut_slice`/`drop_in_place`, with 0 remaining elements on
success or 62-j on overflow. Pointer/read/drop descendants, their destinations
and configuration-conditional checks remain unresolved, not zero work or free.

Extending the ledger-only terminal boundary to iterator transmute, unchecked-sub
or pointer endpoints is **PROPOSED/UNSELECTED**; USER73302 did not approve that
extension. This record neither invents a helper implementation nor closes iterator
quotation. No layout total, tariffs, prices, quote completeness, feasibility,
funding, implementation or execution claim follows. Native HOLD and cleanup
UNKNOWN remain unchanged.

**Quotation-stage decision status (no code adoption implied):**

1. **CHOSEN DESIGN FOR QUOTATION:** private method-neutral OriginalAdmission/compiler-owned local job/single-use issuer; one ledger constructed in issuer setup before quotation; borrowed admission until initial reserve, then same ledger/budget through Capture→F→Payload and terminal admission history. Retained incoming same-budget handle stays last in Payload; same ClearInputBuilder→ClearInputs root/shared 32 attempts; fully validated local attachment before wrapping, pool cleanup before phase reset, finish gates and optional Prepared/Phase guard ordering. Attachment is `OwnedCharged<Box<[CoefficientState; 1]>>`, via allocated(1)→push→boxed slice→checked array-box. Not current APIs or approved security guarantees.
2. **UNFROZEN full actual layout/cleanup quote; original conditional K62 schema reviewed by review 16, scoped revision selected by USER73222/Astra 7:** **user 73115 selects Candidate 22's consuming IssueGuard/EntryGuard/latched-cleanup corridor AND setup FOR QUOTATION ONLY**. Astra 13's corridor and independent Astra 11 conditional review retain their historical conditions; Astra 18 setup notation is now selected quote design, not blanket review clearance or implementation. Full setup/callee/destructor/argument/return-move census, ORIGIN realization, no-helper/reentrant-dispatch enforcement and actual layouts remain OPEN. Reconcile replaced K62 cleanup/routing without double credit; early unwind does not prove K62 guard installation. Attachment, allocation/conversion helpers/results/guards/rollback slots and full success/refusal/unwind/retirement quotation remain unfinished under unchanged caps. The limited K62 schema supersedes earlier bootstrap multiplicities only within its scope. Partial inventories and conditional algebra supply no numeric B, prices or feasible bound.
3. **APPROVED origin/setup design for quotation; OPEN actual prices/funding/feasibility:** decisions 72861/73115 select fixed L, irrevocable pre-enrollment startup S consuming both total/control, retained work_limit=L, sole `new(R,R)` with R=L-S, then ordered administrative entry E and cleanup K prepayments. E/K remain fixed and unpriced; successful E stays consumed if K fails, and spent earmarks are not extra balance or renewed debits for covered work. The selected notation is not an implemented ORIGIN or funded body; full original-unit administrative quotes, startup coverage and complete original-ledger cleanup prepayment still need quotation and actual funding approval, with independent setup review before completeness. Incoming policy establishment, budget acquisition and argument delivery remain uncovered. A later debit or retained byte handle cannot close past-work funding; no numeric S/L, tariffs, funded bound, feasibility, layout or cleanup certification follows.
4. **SELECTED retained-L representation and corrected partial census FOR QUOTATION ONLY; whole quote/funding OPEN:** USER73222 selects the reviewed private Allowance L/R path and Astra 7's corrected partial census, including row 14 count 4 and setup delta outside K62. USER73191's additional original-L ceiling after nonrefundable S remains, with all post-start constraints. Affected older P.limit initializer/setup/transfer sketches are superseded for quotation by the selected same-Capture→actual-F→empty→existing-Header path; this is not actual Rust/API adoption or proof of original funding/pre-origin argument delivery. Reevaluate affected aggregates/layouts/products/byte inequalities; partial Capture/F/Header replacement is not a complete method census. Astra 23 resolves only the pinned ledger-helper bodies above; later USER73302 selects this subquote's intrinsic/primitive-language quotation boundary, direct extraction delivery and symbolic nonzero terminal work tariffs with values UNSET, not K62/other-helper/iterator/layout closure. Full issuer/Drop/args/callee/all-path/destructor quotation, origin API/return realization, numeric tariffs and actual funding remain OPEN. No retries, refunds, new authority or replacement root follow.

**DESIGN chosen for next quotation stage only; implementation/actual funding/execution HELD.** Representation/plumbing, safe one-State allocation route, private single-ledger admission/job/issuer custody and decision 72861's proposed logical-allowance origin are selected for quotation, not adopted Rust code or actual approved funding. Actual types/access/caller/Drop bodies, setup/entry/body/query/all-path cleanup census, tariffs/original pre-entry funding, enforcement proofs, original-ledger unit admission, unchanged-cap feasibility and numerical/security gates remain OPEN. No numeric S/L, prices/constants, actual funding source, funded bound, feasibility/layout/cleanup certification, full quote, Rust construction/execution implementation, execution, campaign retry/recovery or canonical HOLD change is authorized. The 14 primitive tests remain UNCOMPILED/UNRUN; adapter implementation and activation remain unauthorized.
