# Maverick explicit-code native batch capsule

Source-derived **algebra conformance**, not an admitted private/verifiable model.
Implements Maverick v1 Protocol 3 and the complete two-level Protocol 2 path.
Specification and independent vectors:
[paper companion](../../paper/reproduction-contracts/maverick/README.md).

## Fixed scope

| Parameter | Value |
| --- | --- |
| Field | BabyBear, `q=2,013,265,921`; canonical u32 residues |
| Matrices | Ordered pair of 64×64 public field matrices, including coefficients outside i8 |
| Codes | Explicit public RAA8 descriptions; two privacy codes, shared primary code, secondary code |
| Privacy | `Nx=512`, exact weight 165; fresh OS randomness for every issuance |
| Primary verification | `Ny=512`, exact weight 31, two independent checks |
| Secondary verification | Dimension 128, `N'=1024`, exact weight 31, two independent checks |
| Limits | One live query/window, 16 issuance attempts per client, scalar serial execution |
| Admission | Checked 16 MiB **per-capsule** allocation ceiling, including both role owners; not process-global |

The historical seeded/i8/41 RAA and BabyBear APIs are unchanged. The proposed
larger field is rejected. No FP8/BF16 conversion, integer no-wrap claim, clipping,
truncation, model import, persistent decode, transport or SDK registration is added.

## Files and use

New implementation lives entirely under `crates/pllm-core/src/maverick_batch/`:
`arithmetic.rs`, `code.rs`, `memory.rs`, `publication.rs`, `protocol.rs`, `mod.rs`;
`testing.rs` is compiled only under `cfg(test)`.

Parent integration now exports `pllm_core::maverick_batch` from core `lib.rs`.
Deterministic injection tests retain a `#[path]` test instance. A separate
production-export test imports public input arrays and runs both complete
protocols through the real exported types, without deterministic RNG injection.
Six exported-API compile-fail doctests reject cloning and reuse of Query, Batch
and ChallengedBatch.

```text
AllocationPlan::bounded()
  -> preprocess(plan, intended matrices + explicit code descriptions)
  -> UnvalidatedPublication::validate(intended source)
  -> ValidatedPublication::split()
  -> (Client, Server)
```

`UnvalidatedPublication::import` admits externally produced P/Q/Qaux only through
the same complete validation. Every entry is recomputed, including the last
entry of Qaux. Matching hashes or a few random spot checks do not qualify it.
Intended matrices and code descriptions are compared in full, not merely by hash.
Public-source dimensions, canonical residues and permutations are checked before
allocation. Native code descriptors use the companion's fixed little-endian
encoding and SHA-256 identity. Publication identity binds all matrices, code
descriptions, derived matrices and fixed parameter choices.

**Protocol 3:** `issue_query` returns a private non-cloneable `Query` and a full
masked `Request`. Server evaluates the complete field vector. `finish_query`
consumes the query on success **and every error**, copies the claimed response,
samples two fresh exact-weight challenges, checks both equations, then demasks.
Only `AcceptedOutput::values()` releases the canonical result.

**Protocol 2:** `issue_batch` creates both private claims. After both replies,
`begin_verification` consumes the batch, validates ordered IDs, copies **all**
original responses, then samples/discloses two primary challenges. Server returns
both stacked `Qstack*e` projections. `finish_batch` consumes `ChallengedBatch`,
copies the **complete** auxiliary reply, then samples secret secondary challenges.
It checks all four secondary and all four original equations before returning
either accepted result. No secondary challenge or privacy noise goes to Server.

Server owns separate public matrix/Q buffers and replay state; it has no P,
privacy code, mask correction or client controller. Only a resource-counter
meter is shared between the in-process owners. Replies bind publication, random
session identity, attempt, mode and ordered matrix slot. Server rejects replay
and auxiliary projection before both originals have been evaluated. Dropping any
private owner cancels it, erases owned buffers and releases the live-window lease.
Failed or busy issuance still consumes an attempt; the counter is an operational
bound, **not** a deployment cryptographic lifetime.

Only OS randomness is available through normal APIs. Unbiased rejection sampling
and partial Fisher-Yates produce exact support cardinality without duplicates;
nonzero coefficients are independently sampled. Geometry and allocation are
bounded; the rejection loop has **no fixed retry bound**. Its distribution is
unchanged. Entropy failure returns an error and retires private ownership.
The OS fill scratch is `Zeroizing<[u8;4]>`, including partial-fill failure paths.
Challenge-derived diagnostic arrays stay in zeroizing owners on success and every
error path; only deterministic `cfg(test)` helpers copy their values out.
All owned word buffers zeroize before deallocation.
This does not claim erasure of CPU registers, allocator metadata or caller copies.

### RAA generator boundary and conditional candidate accounting

Existing [`RaaNumericCode::from_public_seed`](../../crates/pllm-core/src/coded_linear.rs#L176-L216)
builds two full descending Fisher-Yates permutations and two nonzero scale arrays
using a [SHA-256 seed/counter stream](../../crates/pllm-core/src/coded_linear.rs#L326-L353).
This is numeric-only: no distance, security, source-provenance or verification
authority. Its finite seeded generator is not an independently authenticated
qualifying paper ensemble; this does not say all RAA generators are absent or
proven insecure. No production provenance-producing generator or native export
from this numeric generator exists.

Native [`CodeDescription`](../../crates/pllm-core/src/maverick_batch/code.rs#L7-L78)
(`NativeCodeDescription` when aliased) checks exact lengths, bijections and nonzero
canonical scales; its digest binds contents, not sampling history. Existing
[`import` / `validate`](../../crates/pllm-core/src/maverick_batch/publication.rs#L129-L219)
compare intended arrays and recompute algebra, not ensemble provenance.
Any future candidate must reuse [`RandomWords` / `OsRandom` / `below`](../../crates/pllm-core/src/maverick_batch/arithmetic.rs#L35-L67),
[`Words` ownership](../../crates/pllm-core/src/maverick_batch/memory.rs#L259-L303) and
[`SetupInputs`](../../crates/pllm-core/src/maverick_batch/publication.rs#L11-L28)
with existing import/validation, not a parallel registry or trust root.

**Candidate only if independently selected:** full descending Fisher-Yates,
two independent uniform permutations and independent uniform nonzero diagonals.
For `N=8d`: `2(N−1)` successful `below` calls/swaps plus `2N` diagonal calls
give `4N−2` successful calls; four retained u32 arrays give `4N` words / `16N` bytes.

| Candidate population | Successful `below` calls | Swaps | Scales | Retained words / payload bytes |
| --- | ---: | ---: | ---: | ---: |
| One code, `N=512` | 2,046 | 1,022 | 1,024 | 2,048 / 8,192 |
| One code, `N=1024` | 4,094 | 2,046 | 2,048 | 4,096 / 16,384 |
| Current `SetupInputs`: three `N=512`, one `N=1024` | 10,232 | 5,112 | 5,120 | 10,240 / 40,960 |

These are neither total cost nor peak allocation: initialization, validation,
copies, hashing, overlap and failure retirement add work/storage. If a future generator is
independently selected and reuses shared `arithmetic.rs::below`, it has the same source-authored
mechanical 32-attempt engineering cutoff per call, not user numerical-policy adoption
or future-generator authorization: at most `32 × 10,232 = 327,424`
candidate-word draws, with spent attempts burned and no retry reset.
The unbounded native `below` version is historical. Current source authors a fail-closed
32-draw engineering cutoff, returning `Error::Randomness` on exhaustion; it is **UNRUN here**,
not paper-required or a paper-selected sampler, distribution or provenance, and does not
complete whole Maverick. Arbitrary same-named numeric `PublicCodeSampler` implementations
are not covered.
Generator, caller, sampling law, schedule, new multiplicity, funding, tariff, security
and source authority remain separate requirements, not granted by helper reuse.
Funding-first source hold remains; this accounting grants no authoring authority and
changes no HOLD, cap, policy, tariff, adaptation or source/native grant.
The retained payload is neither peak allocation nor total cost.

Before generation, bind population roles/coverage, generation, entropy, failure,
retry and reuse rules; no cherry-picking or renewing authorization. Partial
generation refuses. Independent uniform entropy is an assumption; conditioning
on completion is distinct from later distance-check conditioning. Neither arrays
nor digests confer distance/joint-security acceptance or an operational issuer.

### Conditional finite-field native-ensemble distance bound

**DESIGN ONLY — mathematics and source inspection; evaluations UNRUN.** This
subsection describes a future **UNSELECTED** ensemble, not the current finite
public-seed generator, qualifying provenance, or an implemented generator.
It neither extends nor qualifies the paper Appendix's quoted distance claim.

The native [encoder](../../crates/pllm-core/src/maverick_batch/code.rs#L116-L144)
and [numeric encoder](../../crates/pllm-core/src/coded_linear.rs#L223-L247) implement
the column-vector map

\[
C=A D_2 P_2 A D_1 P_1 R,\qquad N=8k.
\]

Here \(k\ge1\) is an integer. The counting argument holds over any finite field
of prime-power order \(q\ge2\); the native specialization uses \(q=2013265921\).
The single-accumulator transition is defined for every integer \(N\ge0\); the
two-stage encoder uses \(N=8k\).

Here \(R\) repeats each message coordinate eight times, \(P_j\) permutes,
\(D_j\) scales each position **after** permutation, and \(A\) takes prefix sums
starting at zero. There is no termination constraint on the final prefix.
Define \((P_j x)_i=x_{\pi_j(i)}\), matching source gather indexing.
Assume \(\pi_1,\pi_2\) are independent uniform elements of \(S_N\), and all
\(2N\) diagonal entries are IID uniform in \(\mathbb F_q^*\), independent of
both permutations.
Uniform support alone is insufficient: the nonzero values must have the stated
independent uniform law. The current finite public-seed stream does not establish
these hypotheses or qualifying generation provenance.

For one accumulator, let \(Z_t(h,l)\) and \(U_t(h,l)\) count length-\(t\)
field input strings of input weight \(h\) and prefix-output weight \(l\),
ending with a zero or nonzero prefix, respectively. Initialize
\(Z_0(0,0)=1\); all other initial and out-of-range entries are zero. Exactly,

\[
\begin{aligned}
Z_{t+1}(h,l)&=Z_t(h,l)+U_t(h-1,l),\\
U_{t+1}(h,l)&=U_t(h,l-1)+(q-1)Z_t(h-1,l-1)\\
&\quad +(q-2)U_t(h-1,l-1).
\end{aligned}
\]

A zero next input preserves the prefix. From a zero prefix, each of the
\(q-1\) nonzero inputs gives a nonzero prefix. From a nonzero prefix, exactly
one nonzero input cancels it; the other \(q-2\) choices leave it nonzero.
For \(0\le h,l\le N\), the exact weight transition is

\[
T_N(h,l)=\frac{Z_N(h,l)+U_N(h,l)}{\binom Nh(q-1)^h}.
\]

The denominator counts all length-\(N\) field strings of input weight \(h\).
For a fixed message \(m\) of weight \(w\), repetition has weight \(8w\).
The first uniform permutation makes its support uniform; independent per-position
nonzero scales destroy the repeated-value correlations, giving IID uniform
nonzero entries on that support. Conditional on any first-stage output of weight
\(l\), the independent second permutation and scales give the same uniform
weight-\(l\) input law. Thus

\[
\Pr[\operatorname{wt}(Cm)=d]
=\sum_{l=0}^{N}T_N(8w,l)T_N(l,d).
\]

Every sampled encoder is linear and injective: repetition is injective and each
permutation, nonzero diagonal and prefix-sum map is invertible. Nonzero scalar
orbits therefore have \(q-1\) messages and identical output weight. There are
\(\binom kw(q-1)^{w-1}\) such classes of message weight \(w\). A union bound
over these classes gives, for every integer \(0\le D\le N\),

\[
\Pr[d_{\min}\le D]\le\min\!\left\{1,
\sum_{w=1}^{k}\binom kw(q-1)^{w-1}
\sum_{d=1}^{D}\sum_{l=0}^{N}T_N(8w,l)T_N(l,d)\right\}.
\]

This conditional ensemble bound needs no independence across messages and may
equal one. It is valid at finite \(q=2013265921\), without a \(q>2^{31}\)
premise. Hand checks: \(N=1,h=1\) gives output weight 1 certainly;
\(N=2,h=2\) gives weights 1 and 2 with probabilities \(1/(q-1)\) and
\((q-2)/(q-1)\). For \(h=0\), \(T_N(0,l)=\mathbf 1_{\{l=0\}}\);
\(N=2,h=1\) gives weights 1 and 2 with probability \(1/2\) each.
At \(q=2\), the \(q-2\) term vanishes and a nonzero input toggles the prefix.

Conditioning the current source-authored 32-draw helper on full completion preserves this law
only under IID u32 draws, fixed bounds and call count, and no accepted-value-
dependent filtering or retry. Each accepted residue is independent of its
first-success index, so completion does not bias the accepted-value law under
those assumptions. The unbounded helper is historical; the current fail-closed
engineering cutoff remains **UNRUN here**, not execution or paper-sampler qualification.
Later distance filtering, PRG/provider
assumptions, arrays versus generation provenance, material reuse and joint
security remain separate requirements.

No experimental numerical DP or full exact bound evaluation, per-code certificate, soundness target,
security claim or parameter acceptance is supplied. Generator selection and new ensemble
provenance remain unresolved; rates, funding-first source hold, helper execution/qualification
hold (not absence of the source-authored implementation), source authority, HOLDs
and all existing caps remain unchanged; no receipt machinery is added.

#### Exact run-count evaluation (design only)

The same conditional ensemble admits the following exact evaluation design;
numerical evaluations remain **UNRUN**, with no security target or parameter adoption.
Write \(a=q-1\), \(b=q-2\). For \(l>0\),

\[
\begin{aligned}
Z_N(h,l)&=\sum_{m=1}^{l}\binom{l-1}{m-1}\binom{N-l}{m}
 a^m\binom{l-m}{h-2m}b^{h-2m},\\
U_N(h,l)&=\sum_{m=1}^{l}\binom{l-1}{m-1}\binom{N-l}{m-1}
 a^m\binom{l-m}{h-2m+1}b^{h-2m+1}.
\end{aligned}
\]

Invalid binomials are zero; \(0^0=1\). In each summand, first guard
\(j=h-2m\) (for \(Z\)) or \(j=h-2m+1\) (for \(U\)):
if \(j\notin[0,l-m]\), skip it **before evaluating the power**.
Set \(Z_N(h,0)=\mathbf1_{\{h=0\}}\), \(U_N(h,0)=0\);
all out-of-range \(h,l\) entries are zero.

To derive these counts, partition the nonzero output positions into \(m\)
runs: their positive lengths have \(\binom{l-1}{m-1}\) compositions.
Zero gaps have \(\binom{N-l}{m}\) placements when ending at zero and
\(\binom{N-l}{m-1}\) when ending nonzero. Each run starts with \(a\)
choices; exits contribute one nonzero input each. Starts plus exits contribute
\(2m\) or \(2m-1\) input weight, respectively. Among \(l-m\) internal
positions, choose \(j\) changes, each with \(b\) choices; unchanged positions
contribute zero input. This also covers \(q=2\) through the guarded convention.

Let the scalar count \(A(h,l)=Z_N(h,l)+U_N(h,l)\) (not the prefix-sum matrix).
Its support satisfies \(h\le2l\). Thus a final weight \(d\) requires
\(4w\le l\le2d\), hence \(w\le\lfloor d/2\rfloor\). Define

\[
\begin{aligned}
M_0&=0,\\
M_l&=\sum_{w=1}^{\min(k,\lfloor l/4\rfloor)}
 \binom{k}{w}a^{w-1}\frac{A(8w,l)}{\binom{N}{8w}a^{8w}},\qquad l>0,\\
B(D)&=\sum_{d=1}^{D}\sum_{l=1}^{\min(N,2d)}
 M_l\frac{A(l,d)}{\binom{N}{l}a^l},\qquad 0\le D\le N.
\end{aligned}
\]

Here \(B(D)\) is the raw, unclipped class union sum; \(\min\{1,B(D)\}\)
is an ensemble union bound, **not** an equality with the actual probability.
Stream each \(M_l\) once, retaining only the needed \(O(D)\) rational entries,
and evaluate \(A\) on demand; no global \(A\) table or binomial cache.
Summing run candidates costs \(O(D^3)\) combinatorial candidates, not bit
operations. Entry count is not a word/byte bound, nor CPU, RSS or speed evidence;
arbitrary-precision arithmetic and library scratch require separate accounting.
Independently of ensemble randomness, uniformity or independence, write
\(y=Ax\), \(y_0=0\): \(x_i=y_i-y_{i-1}\) implies pointwise
\(\operatorname{wt}(x)\le2\operatorname{wt}(Ax)\), since each nonzero input
touches a nonzero current or prior output. Repetition gives weight \(8\operatorname{wt}(m)\);
genuine permutations and nonzero diagonals preserve weight. Thus every valid
\(C\) has \(\operatorname{wt}(Cm)\ge2\operatorname{wt}(m)\), so \(d_{\min}\ge2\).
This structural bound supplies no higher-distance, soundness or privacy claim,
actual-ensemble qualification, or adopted security target.
Hand checks, without numerical evaluation: \(B(0)=B(1)=0\), and the sole
\((w,l,d)=(1,4,2)\) contribution gives
\[
B(2)=\frac{420k}{\binom{N}{4}\binom{N}{2}a^6}.
\]

#### Conditional analytic D=8 brackets

Set \(q=2013265921\), \(r=8\), \(a=q-1\), \(b=q-2\), under the SAME
independent uniform \(P_1,P_2\) and ALL IID uniform nonzero \(D_1,D_2\)
assumptions above. For \(p=\Pr_{\mathrm{ensemble}}[d_{\min}\le8]\),
\[
\begin{array}{c|c}
(k,N)&\text{conditional analytic bracket}\\
(64,512)&2^{-109}<p\le B(8)<2^{-46}\\
(128,1024)&2^{-124}<p\le B(8)<2^{-51}.
\end{array}
\]
These are ensemble inequalities, NOT certificates for an actual sampled code.

For the upper bound, keep BOTH terminal cases of each full positive run summand
in \(A(8w,l)A(l,d)\), with all binomial/power guards above. Nonzero support
requires \(1\le w\le4\), \(4w\le l\le2d\le16\).
Let \(\epsilon_i\in\{0,1\}\) indicate a nonzero terminal prefix, \(m_i\)
be its run count, and \(g=(m_1-\epsilon_1)+(m_2-\epsilon_2)\).
Replacing \(b^j\le a^j\), the two composition/change-coefficient products
are at most \(2^{2l+2d-4}\); the two gap binomials are at most \(N^g\).
Falling products give
\[
\binom N{8w}\ge\frac{N^{8w}}{32(8w)!},\qquad
\binom Nl\ge\frac{N^l}{2l!}.
\]
Indeed \(\prod_i(1-i/N)\ge1-\sum_i i/N\); the worst sums are
\(496/512\) and \(120/512\), respectively, also valid at \(N=1024\).
Thus EACH full class/run contribution is bounded by
\[
F=64k^w(8w)!l!\,2^{2l+2d-4}
 N^{g-8w-l}a^{w-1-g}.
\]
The class count already supplied \(a^{w-1}\): there is NO extra projective division.
Use \(l!\le2^{3l}\), \((8w)!<2^{H_w}\) with
\((H_1,H_2,H_3,H_4)=(16,45,80,118)\), and \(2^{30}<a<2^{31}\).
Split the sign-sensitive power as
\(a^{w-1}a^{-g}\le2^{31(w-1)-30g}\), rather than bounding its exponent blindly.
At \((k,N)=(64,512)\),
\[
F<2^{H_w-35w-4l-13-21g},\qquad
g\ge\max(0,8w-l)+\max(0,l-d).
\]
For \(w=1\), \(g\ge|l-8|\), so \(4l+21g\ge32\).
For \(w=2,3,4\), use \(l\ge4w\), \(g\ge8w-8\).
The resulting exponent maxima for \(w=1,2,3,4\) are
\((-64,-238,-422,-603)\). At most
\(4\cdot8\cdot16\cdot16\cdot8\cdot4=2^{18}\) run/terminal candidates
therefore give \(B(8)<2^{-46}\). Doubling \(k,N\) multiplies each \(F\) by
\(2^{g-7w-l}\); since \(g\le4w+l/2\) and \(l\ge4w\), this is at most
\(2^{-5}\), yielding \(B(8)<2^{-51}\).

The LOWER bound uses actual events, not a lower bound on a union sum.
For basis line \(i\), let \(S_i\) be its eight repeated positions. Event
\(E_i\) places \(S_i\) in the last eight positions under \(P_1\), probability
\(\binom N8^{-1}\), and keeps every first tail prefix nonzero, probability
\((b/a)^7\). Independently \(P_2\) places the new eight-position support last,
again with probability \(\binom N8^{-1}\); independent \(D_2\) rerandomizes
its nonzero values, giving second-tail survival \((b/a)^7\).
Each \(E_i\) yields a basis codeword of weight eight. The \(E_i\) are mutually
exclusive: \(P_1\) cannot put two disjoint \(S_i\) into the same last eight slots.
Consequently
\[
p\ge k\binom N8^{-2}(b/a)^{14}.
\]
Use \(8!>2^{15}\), \(\binom N8<N^8/8!\), and Bernoulli's inequality
\((1-1/a)^{14}\ge1-14/a>1/2\) to obtain the strict lower endpoints above.
There is NO extra \(q-1\) factor. This is an event-probability statement,
not a 128-bit security/insecurity verdict, LPN or composed-security conclusion.

Native [`SetupInputs`](../../crates/pllm-core/src/maverick_batch/publication.rs#L11-L28)
has FOUR slots: two privacy codes, primary and secondary; dimensions are
\(64,64,64,128\), lengths \(512,512,512,1024\) ([constants](../../crates/pllm-core/src/maverick_batch/mod.rs#L28-L33)).
The [fixture](../../crates/pllm-core/src/maverick_batch/testing.rs#L165-L193)
([assembly](../../crates/pllm-core/src/maverick_batch/testing.rs#L267-L282)) makes four DETERMINISTIC
array instances, not the seeded `RaaNumericCode` route or the ideal ensemble.
[Publication copies](../../crates/pllm-core/src/maverick_batch/publication.rs#L142-L147) through
[`Code::copy`](../../crates/pllm-core/src/maverick_batch/code.rs#L81-L98) make four owner copies of
four-`Words` array descriptions, NOT new random samples; borrowed descriptions can alias,
so four slots do not imply four independent external generators/roots. Under EACH slot's ideal
ensemble MARGINAL, \(\Pr[\text{any slot }d_{\min}\le8]<3\cdot2^{-46}+2^{-51}=97\cdot2^{-51}<2^{-44}\),
without cross-code independence. Immutable reuse adds no sampled-code event; regeneration needs a new census.
The earlier census remains 10,232 hypothetical accepted Fisher-Yates/scale draws and 40,960 payload bytes,
excluding raw attempts, overhead, copies and peak. Its conditional 32-draw-cap inheritance and separate
generator/caller/law/multiplicity/funding/tariff/security/source-authority requirements remain unchanged.

#### Conditional bounded-generation lemma (unselected candidate)

Current [arithmetic.rs:55–68](../../crates/pllm-core/src/maverick_batch/arithmetic.rs#L55-L68)
accepts \(x<2^{32}-(2^{32}\bmod n)\), returns \(x\bmod n\), and authors a fail-closed
32-draw engineering cutoff returning `Error::Randomness` on exhaustion (**UNRUN here**).
The UNCAPPED version is historical; this source change selects no paper sampler,
distribution or provenance and does not complete whole Maverick.
A FUTURE independently selected generator using the source-authored 32-draw helper
has the same mechanical engineering bound as specified above; this implies no user
numerical-policy adoption, generator authorization or source authority.
Model ONLY an IID uniform raw-u32 stream, deterministic Fisher-Yates/scale bounds \(n_i\),
and no provider error/stall/other conditioning. Let \(\rho_i=(2^{32}\bmod n_i)/2^{32}\),
\(J_i\in\{1,\ldots,32\}\) be first-accept length, and \(Y_i\) its residue.
For fixed \(j\), consecutive ACTUALLY consumed blocks are disjoint IID blocks, so
\[
\Pr[J=j,Y=y]=\prod_i\rho_i^{j_i-1}(1-\rho_i)/n_i.
\]
Writing \(S\) for every call accepting within 32 draws, summing \(j\) gives
\[
\Pr[S,Y=y]=\prod_i(1-\rho_i^{32})/n_i,\qquad
\Pr[S]=\prod_i(1-\rho_i^{32}).
\]
Conditioning ONLY on \(S\), residues are independent uniform; deterministic descending
Fisher-Yates yields two independent uniform permutations and IID nonzero scales per code.
Thus the ideal marginal D=8 bound applies, with no extra field factor or fixed-32-block claim.
For 5,120 scale calls, \(n=a=15\cdot2^{27}\) has remainder \(2^{28}\), hence
\(\rho=1/16\): scale-abort probability is at most \(5120\cdot2^{-128}<2^{-115}\).
For 5,112 permutation calls, \(n\le1024\) gives \(\rho<2^{-22}\), so their
abort probability is \(<5112\cdot2^{-704}<2^{-691}\). A first-failure union remains
valid when generation halts at its first abort, giving \(\Pr[S^c]<2^{-114}\).
Let Bad mean any of the four slots has distance at most eight. Then
\(\Pr[S^c\cup(S\cap\mathrm{Bad})]<2^{-114}+2^{-44}<2^{-43}\);
the distance union itself needs no cross-code independence assumption.
This is NOT a full runtime-failure bound, OsRandom/provenance qualification, physical
termination/funding/security/parameter acceptance or authoring/execution grant. Extra provider,
admission/import or value-dependent-filter success conditioning is not covered and can bias the law;
neither historical seeds nor arbitrary PublicCodeSampler implementations are thereby qualified.

Full exact \(B(D)\) remains unevaluated by the experiment; these conservative
hand-derived inequalities are analytic only. The seeded generator is NOT qualified,
and no parameter-selection, source-authoring, execution or HOLD change follows.

This is evaluation mathematics only, not experiment authoring or execution
authority; all ensemble assumptions, provenance limits and holds above remain.

Separately user-approved standalone authoring delivered the
[experiment](../../scripts/maverick_raa_distance_experiment.py) and [tests](../../tests/test_maverick_raa_distance_experiment.py).
Both passed source review only; all execution remains **UNRUN**.
The script specifies exact hex JSON and a 1,000,000-candidate/assembly cap per profile, not CPU/RSS bounds.

## Allocation receipt

`AllocationPlan` is immutable outside the module. Unsupported dimensions,
checked-size overflow, undersized ceilings and ceilings above 16 MiB reject before
allocation or hashing. Every variable-sized module buffer also reserves its
capacity plus 128 bytes before calling the allocator.

| Simultaneous charge | Bytes |
| --- | ---: |
| Client public matrices/codes/P/Q/Qaux | 2,695,168 |
| Borrowed source plus incoming publication overlap | 2,695,168 |
| Separate server matrix/Q copies | 294,912 |
| Preparation/validation/online/message transient allowance | 524,288 |
| Control, stack and allocator allowance | 262,144 |
| All 16 successful two-result outputs retained | 8,192 |
| **Planned peak** | **6,479,872** |
| **Hard allocation ceiling** | **16,777,216** |

Qaux alone is 2,097,152 bytes. Validation streams rows without another full Qaux
or dense G allocation. Runtime meters include the external/control reservation;
the dedicated allocator test independently measures requested allocation layouts,
checks release after the complete 16-window lifetime, and observes private-buffer
zeroization immediately before free. This is a bounded allocation receipt with
explicit allowances, not whole-process RSS, allocator-residency or model admission.
Arbitrary caller-created copies and multiple independent capsules require their
own aggregate owner budget.

Dynamic word-buffer exhaustion uses `try_reserve_exact` and returns
`Error::Allocation` with consuming-owner retirement. **Fixed-size `Arc::new`
allocations for the meter and client control remain an exception:** they follow
Rust's abort-on-OOM policy, even though their sizes are covered by the control
allowance. This capsule does not promise universal recoverable OOM behavior.

The allocation-failure binary returns null from the actual `GlobalAlloc` call
at each dynamic allocation index across import, full P/Q/Qaux validation, split,
query/batch issuance, evaluation, both fixation boundaries, projection and final
verification. Fixed Arc allocations are deliberately excluded. A nonallocating
ledger observes real buffer pointers and live reservation counters: admission
must precede allocation, bytes must be zero before physical free, and refunds
must follow free. It also checks exact retained-message charges after private
owners retire, monotonically consumed attempts, new-owner recovery and complete
cleanup. Observer hooks exist only under `cfg(test)` and never simulate failure.

## Targeted checks

```sh
CARGO_BUILD_JOBS=2 CARGO_PROFILE_TEST_OPT_LEVEL=2 cargo test -p pllm-core \
  --test maverick_batch_reference --test maverick_batch_allocation \
  --test maverick_batch_export --test maverick_batch_failures \
  -- --test-threads=1 --nocapture
CARGO_BUILD_JOBS=2 cargo test -p pllm-core --doc maverick_batch -- --test-threads=1
```

Coverage includes frozen Python-vector parity for complete 64×64 Protocols 3/2;
literal-matrix RAA basis/transpose checks; full P/Q/Qaux corruption and intended
identity rejection; fresh-noise round trips; exact support draw enumeration;
forged original and auxiliary replies; an auxiliary lie orthogonal to the original
input; immutable response copies; replay, malformed/omitted replies, cancellation,
attempt exhaustion and entropy failures at all private sampling phases, including
partial OS-fill rejection. The failure sweep is finite but exhaustive over the
observed dynamic allocation indices; full validation includes its final Qaux row.
Use the test-only optimization setting above for this sweep: repeating complete
validation at every allocation index exceeded a 600-second debug-run limit after
all 1,539 validation failure indices had passed. It changes no source parameters.
Deterministic noise/challenge injection exists only in `cfg(test)` helpers.

Audited 17 October 2026 targeted run: **15 native tests and six ownership
compile-fail doctests passed**. The optimized failure sweep completed in 35.69 s:

| Failure stage | Actual allocation indices exhausted |
| --- | ---: |
| Import | 23 |
| Complete P/Q/Qaux validation | 1,539 |
| Party split | 4 |
| Query / batch issuance | 8 / 16 |
| Server evaluation | 1 |
| Protocol 3 verification | 10 |
| Original fixation / primary reveal | 12 |
| Server auxiliary response | 3 |
| Auxiliary fixation / Protocol 2 verification | 13 |
| **Total injected allocation failures** | **1,629** |

The ledger checked **1,341,622 buffer destructions** for erasure-before-free and
refund-after-free, plus exact retained-message charges, private-owner retirement
and recovery under a new attempt ID. Real production-export Protocols 3/2 round
trips and the unchanged frozen vectors passed. Observed tracked peak including reservations was
**5,984,208 bytes**; independent allocator-layout peak plus external/control
reserves was **5,974,040 bytes**. Both fit the 6,479,872-byte receipt and 16 MiB
ceiling. All component allocations were released at teardown; the private-buffer
erasure probe passed. These are conformance observations, not model benchmarks.

## Unchanged qualification gates

Explicit code arrays do not prove uniform code selection, distance, or dual-LPN
security. No tiny-code certificate is extrapolated. Finite-distance evidence,
actual-code/noise security and complete query-horizon accounting remain required.
The secondary parameters are the approved bounded source-compatible capsule
instantiation, not recovered settings from the paper's evaluated model system.

Exact model FP8/activation/BF16 semantics, original eight-token Qwen3-4B validation,
accepted-state decoder ownership, complete resource measurements and ordinary
Pipeline/Experiment/SDK/gateway integration remain separate gates. Next integration
step is bounded role transport using the now-exported consuming owners;
no complete-method capability or performance result follows from this capsule.
