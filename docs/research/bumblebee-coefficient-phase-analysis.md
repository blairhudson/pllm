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
