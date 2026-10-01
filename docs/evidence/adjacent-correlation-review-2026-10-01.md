# HE-assisted function-specific joint correlations — 2026-10-01

## Verdict

**Stop these layouts as a tenfold Qwen decoder route. Keep the algebra as a
conditional incremental optimization.** Joint preprocessing really can remove
one repeated vector opening; compressing the generator alone cannot remove the
remaining norm-opening floor. A projected quadratic region has much smaller
material than five independently shared coefficient arrays, but Qwen SiLU is
not quadratic and its rounded numeric edges cannot be fused for free.

This report supplies concrete distributions, consuming equations, a dealerless
semi-honest AHE generator specification, a bounded BFV numeric/cost probe, and a
complete-layer **screen**. It does not supply executable protected RMSNorm,
attention, SiLU, or a complete decoder. No training, TEE, weight changes,
provider plaintext activations, or client reconstruction of body states is used
to rescue the estimates. Two online workers each hold one share; this differs
from the currently executable prepared topology.

### Saved artifacts and exact checks

- Report: `docs/evidence/adjacent-correlation-review-2026-10-01.md`.
- Probe: `scripts/probe_structured_correlations.py` (stdout JSON; no downloads,
  output files, dependency installation, or full-scale material).
- Tests: `tests/test_structured_correlations.py`.

```sh
uv run --no-sync python scripts/probe_structured_correlations.py
uv run --no-sync python scripts/probe_structured_correlations.py --bfv
uv run --no-sync pytest -q tests/test_structured_correlations.py
```

Executed: default screen; BFV batches 1/4/16; **17 tests passed in 0.08 s**.
`uv run --no-sync ruff check scripts/probe_structured_correlations.py
tests/test_structured_correlations.py` passed. Per-file
`git diff --no-index --check /dev/null <new-file>` passed for all three artifacts.
Checks cover exact joint consumption over F5, F257, F65537, Z/2^24 and Z/2^32;
projected cubic expansion; finite-field rank; small-field output-pad marginal
support; reused-mask difference leakage; source/scalar replay, cancellation,
invalid share, wrong session/party/residue, and truncated-body burn. BFV checks
exact generation and subsequent consumption separately, rejects generation
replay, and asserts the evaluator context has no secret key. These are algebra,
state-machine and cost checks, not a security proof or checkpoint-quality test.

## 1. Locked workload and complete consuming region

Controls come from the existing compiler-region evidence, SHA-256
`2cc3e18916d4f95b19fd8951bdaea41078f3db9981f536cb9540873ac1e5b78e`:
[`compiler-region-contract-qwen25-2026-09-30.json`](compiler-region-contract-qwen25-2026-09-30.json).
Checkpoint: `Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`.
Body fingerprint:
`5d631be30158b3ea2a72cae355ce09a3b304bbfddde48b5758c339c346a34974`.
Composition:
`bd91e1659773faabcbb998a350893cbbc6df313a830da7cb7bfb87463c8f0b36`.

Pinned [public configuration](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/raw/7ae557604adf67be50417f59c2c2f167def9a775/config.json)
confirms 24 layers, hidden width n=896, intermediate width m=4864, 14 query
heads, two KV heads, head width 64, SiLU and RMSNorm epsilon 1e-6. The probe
uses these explicit dimensions and the saved compiler cohorts; it does not
load pretrained tensors or rerun compilation.

| 39-input cohort | +8 | +32 |
|---|---:|---:|
| Executed body rows, 39 + outputs − 1 | 46 | 70 |
| Covered control all-link bytes | 116,843,966 | 178,970,558 |
| Covered control online bytes | 73,831,744 | 113,545,024 |
| Strict-tenfold integer all-link budget | 11,684,396 | 17,897,055 |
| Strict-tenfold integer online budget | 7,383,174 | 11,354,502 |
| Body norm rows, 2 × 24 × executed rows | 2,208 | 3,360 |
| Existing resident token ingress/egress floor | 304,640 | 605,696 |

The boundary comparator retains separate 24-bit ingress shares to both workers
and separate 32-bit final-hidden egress shares to the client. Client selects
tokens and feeds back embeddings; there is no invented private vocabulary
argmax. Boundary bits are the earlier placement assumptions, not validated
numeric widths for the proposed protocol.

### Every operator has a consuming obligation

| Layer region, in causal order | Potential structured correlation | What is still required |
|---|---|---|
| Attention RMSNorm | Joint statistic + vector/scalar block below | Private inverse square root, epsilon/division, weighted scale and exact rounded edges |
| Q/K/V projections | Public maps on local shares; reuse fixed maps | Full accumulator domain, dynamic activation quantization and signed share conversion |
| Rotary, KV append/read | Public rotations/maps and local shared KV | Numeric parity, past-mask lifetime, capacity and cancellation state |
| QK scores | Bilinear tensor/matrix correlations, or same-source quadratic forms on a stacked source | Fresh-query/past-KV mask cross-correlations; full causal/GQA layout |
| Causal softmax | Function-dependent nonlinear preprocessing | Private max, exponential, sum, division and exact baseline behavior; no specified cost |
| Probabilities × V | Matrix/bilinear block | Probability source is produced after softmax; it cannot reuse the Q source opening automatically |
| Attention output, residual | Public maps/local additions | Existing float/quantized scale transitions |
| MLP RMSNorm | Another joint norm block | New post-attention source; same private inverse/rounding gaps |
| Gate/up projections | Public maps on local shares | Exact quantization before applying a gate correlation |
| SiLU × up | Quadratic only for an affine gate; cubic numerator construction in section 4 | Actual pretrained SiLU and its separately rounded composite, not a surrogate numerator |
| MLP down, residual, next layer | Project outputs before sharing only if arithmetic edges permit | Public linearity does not commute through intermediate rounding/quantization |

Unknown entries have **unknown** cost. The norm floor alone already vetoes
32/24-bit variants, so constructing all missing attention material is unnecessary.
This screens the complete consuming region without calling a small quadratic
island a complete layer.

## 2. Distribution: shared source, public maps, quadratic functions

Let R be a finite commutative ring; the BFV experiment uses R=F65537. Choose
independent uniform party-local source masks r0,r1 in R^n and scalar masks
b0,b1 in R. Put r=r0+r1 and b=b0+b1. Public maps G,U,D are fixed to the
function/plan. Define

```
q(r) = <r,r>
c(r,b) = r b
k(r) = D ((G r) ⊙ (U r))
F(r,b) = (q(r), c(r,b), k(r)) ∈ R^(1+n+h).
```

For the ideal distribution, choose independent uniform pads C0 in R^(1+n+h),
independent of every source mask; set C1=F(r,b)−C0. Party i receives only
`(ri, bi, Ci)` plus public maps/binding. Conditional on its own ri,bi, each
party's Ci is jointly uniform over the whole output space. A test dealer can
sample this distribution; a deployment needs the distributed generator below.
The source masks are independent across regions/rows and never reused for a
different logical value.

The concrete mass function is zero unless `C0+C1=F(r0+r1,b0+b1)`; otherwise
it is `|R|^(−(2n+2+1+n+h))` for a complete tuple with the given public maps.
This describes a polynomial correlation, not independent multiplication triples
with a matching shape. The component totals q,c,k are dependent functions of
the same hidden r,b even though each party's output pads are independent.

### Linear-image correlation and rank, rather than misleading real covariance

For uniform r in Fp^n, A r is uniform on im(A), with entropy
`rank_Fp(A) log2(p)`. For stacked maps A=[G;U;...], projected masks must obey
the same joint linear relations. Independent gate/up masks would break these
relations and lose source-opening reuse. The exact Fourier characterization is

```
E[exp(2πi tᵀ A r / p)] = 1 if Aᵀt=0, otherwise 0.
```

Thus rank-deficient image coordinates can be omitted in a suitable basis,
but rank-full n-dimensional sources retain n field elements of entropy.
Ordinary covariance of integer representatives of modular values is not the
security-relevant covariance of this distribution. Over Z/2^k use image size
and Smith-normal-form invariant factors, not a finite-field rank argument.
The tiny synthetic stacked G/U map has rank 16 for n=16. **No real checkpoint
matrix rank or exact module factorization was measured.**

If the entire consuming function depends only on a rank-k public map Ax,
workers could open a masked k-coordinate image representation instead of x.
This requires exact factorization and invariance to ker(A); using a low-rank
mask for an otherwise full-dimensional x opening leaks the unmasked kernel
component. RMSNorm's source statistic prevents applying a gate-map compression
to the whole norm/gate region without a separate proof/protocol.

For quadratic constant outputs, a basis of the public quadratic forms can
reduce stored constants to the dimension of their polynomial span, at most
`min(output_count,n(n+1)/2)` over an odd field. This is an achievable algebraic
representation bound, not a universal cryptographic storage lower bound.
Symmetric-matrix formulas using division by two do not transfer to Z/2^k;
use explicit ordered monomial coefficients there.

## 3. Consuming protocol: actual openings removed

Workers initially hold input shares x0,x1. Each sends `di=xi−ri` to the other;
both reconstruct only d=x−r. Each party computes

```
[<x,x>]i = [q(r)]i + 2<d,ri> + (i==0)<d,d>
[k(x)]i = [k(r)]i
          + D((Gd)⊙(Uri) + (Gri)⊙(Ud))
          + (i==0)D((Gd)⊙(Ud)).
```

Cross terms are **linear in ri**, so workers derive them locally. No separate
shares of every first derivative are needed for a quadratic function. If only
D-projected outputs are consumed, issue h constants, not m intermediate
product constants. This removes a vector opening for a same-source product
and cuts coefficient transport; it is more than renaming triples.

A protected nonlinear protocol would next turn the statistic shares into
shares s0,s1 of the normalization scalar. The probe instead supplies shares
of a **synthetic scalar**; it does not implement that private inverse. Workers
exchange `ei=si−bi` and reconstruct e=s−b, then compute

```
[x s]i = [r b]i + d bi + ri e + (i==0)d e.
```

The original d serves both the statistic and multiplication. Online arithmetic
is `2(n+1)` ring elements, rather than `4n+2` for independently masked
sum-of-squares plus broadcast. Explicit party-local norm material is
`(ri,bi,[q(r)]i,[rb]i)`: `2n+2` elements per party, rather than `3n+2`.
The extra k constants cost h per party when that same-source quadratic output
is needed. These counts exclude headers and the unimplemented private inverse.

**Safe reuse:** one source mask, one opening, multiple fixed public-map outputs
of the same logical source within one joint block. **Unsafe reuse:** opening
x−r and y−r for different logical values reveals x−y. Likewise, one scalar
mask cannot protect two different scalars. A VOLE's repeated scalar across one
vector correlation is intentional; this does not authorize replaying material.

Normalized y=x s is a different source. One cannot claim that the above opening
also evaluates the real post-normalization MLP quadratic. That requires a new
mask/opening or a higher-degree correlation in x,s. Attention's softmax output
likewise establishes another source. Public linear maps on shared y are local,
but do not themselves eliminate later nonlinear openings.

### Full-response norm floor, even after this reuse

All numbers below are arithmetic bytes across both directed peer links;
explicit material covers both recipients. They describe this construction,
not a lower bound on every possible norm protocol.

| Assumed width / quantity | +8 | +32 |
|---|---:|---:|
| 32-bit joint norm openings | 15,844,608 | 24,111,360 |
| 32-bit norms + token boundary, online | **16,149,248** | **24,717,056** |
| 32-bit explicit norm material / dealer issuance if transmitted | 31,689,216 | 48,222,720 |
| 32-bit dealer-layout known all-link floor | **47,838,464** | **72,939,776** |
| 24-bit norms + token boundary, online | **12,188,096** | **18,689,216** |
| Hypothetical 12-bit norms + boundary, online | 6,246,368 | 9,647,456 |
| Hypothetical 12-bit explicit norm material | 11,883,456 | 18,083,520 |
| Hypothetical 12-bit dealer-layout all-link floor | **18,129,824** | **27,730,976** |

32/24-bit online floors exceed **both** corresponding budgets before SiLU,
attention, inverse or preprocessing traffic. This improves the previous
48.20 MB 32-bit online norm comparator to 24.11 MB for +32, but still fails.
Even free or perfectly compressed generation cannot fix those online bytes.

The hypothetical 12-bit online floor leaves only 1,136,806 / 1,707,046 bytes
for all other online arithmetic. Its explicit-dealer all-link floor still
fails. Local generation could avoid transmitting expanded arrays, but its
transcript must be charged and the arrays still need one-use storage. Twelve-bit
shares also lack a Qwen-valid signed lift/carry, inverse and exact rescaling;
this is **not** an admitted numeric option.

## 4. Cubic gate: structured material improves constants, not fidelity

The existing `shared_gate_polynomial.py` evaluates the degree-three numerator
`(256 g + g²)u`, not a degree-two complete gate. It independently pads five
shifted coefficients per intermediate channel. For a public factored cubic

```
F(x) = D((α Gx + β (Gx)²) ⊙ Ux),
a=Gr, b=Ur, v=Gd, w=Ud,
```

the following identity avoids separately sharing linear-mask coefficients:

```
F(r+d) = F(r)
       + D((αv+βv²)⊙w)
       + D((αv+βv²)⊙b + (αw+2βv⊙w)⊙a)
       + D(βw⊙a² + 2βv⊙(a⊙b)).
```

Issue ri plus shares of the **joint quadratic features** `(a²,a⊙b)` and the
projected constant F(r). The a,b linear-mask terms derive from ri locally.
Material is `n+2m+h` elements per party, versus `n+5m` in the current
unprojected numerator construction. The shared quadratic-feature basis might
be smaller if its public forms have exact dependencies; no such checkpoint
reduction is demonstrated. A generic Jacobian matrix would cost h×n instead,
which can destroy this factored advantage.

At n=h=896,m=4864, explicit 32-bit cubic material projects to **101,744,640 /
154,828,800 bytes** for +8/+32. The simpler projected quadratic comparator is
15,826,944 / 24,084,480 bytes, but it would require replacing SiLU with an
affine gate and is therefore not a fixed-model implementation. These are
separate layouts, not costs to add blindly to the norm table. Projection before
intermediate rounding can also change the model. The exact modular expansion
test grants no pretrained numeric quality claim.

For attention, causal edges per head are `39×40/2 + sum(40..38+outputs)`.
Across 24 layers and 14 heads this gives 363,216 / 834,960 score outputs.
Explicit two-party 32-bit quadratic score constants alone are **2,905,728 /
6,679,680 bytes**, before masks or probabilities×V. This is a storage/issuance
comparator for an explicit score-correlation representation, not an information
theoretic lower bound or a complete attention generator. GQA does not reduce
14 query-head scores to two heads. Residing KV avoids KV return traffic but
does not synthesize the needed cross-row correlations.

## 5. A secure dealerless generator specification

### Concrete AHE construction for these quadratic functions

This section specifies a semi-honest generator under Paillier's composite
residuosity assumption [S6], with statistical masking of the decryptor's view.
It is **not implemented or timed** in the BFV probe. It produces the particular
joint quadratic functions above directly, rather than manufacturing generic
triples first. It works for R=Z/M, including a prime field or M=2^32, without
incorrectly treating Paillier's plaintext modulus N as a multiple of M.

1. A holds a Paillier secret key; B receives only public N (g=N+1). They sample
   independent uniform local vectors `aA=(rA,bA)`, `aB=(rB,bB)` in R^K,
   K=n+1. A sends fresh encryptions of each canonical aA coordinate. Neither
   worker ever knows the joint source mask. Public weights/functions are bound
   before issuance.
2. For each homogeneous quadratic output Qj, bilinearity gives
   `Qj(aA+aB)=Qj(aA)+Qj(aB)+sum_l cj,l(aB) aA,l mod M`.
   B computes each cj,l as a canonical residue in [0,M). Coefficients are
   derived jointly from the same aB for all outputs, not chosen independently.
3. Set `L=K(M−1)^2`. Choose T with `M T >= 2^κ L`; use κ large enough to
   cover a union bound over all outputs in the admitted response, e.g. κ=160
   for these bounded counts. Require `N > M T + L`. For each output B samples
   independent σj uniform in [0,M), tj uniform in [0,T). It returns
   `Enc(sum_l cj,l aA,l + σj + M tj)` using additive HE and a **fresh uniform
   Paillier randomizer**. No modular wrap in N can occur under the public bound.
4. A decrypts Yj and retains `CA,j=Qj(aA)+Yj mod M`. B retains
   `CB,j=Qj(aB)−σj mod M`. The totals are exactly Qj(aA+aB), and each output
   pad is uniform modulo M. Both local mask vectors remain local; there is no
   additional dealer-to-worker array shipment in this placement.

For A, the decrypted integer is a bounded shift of a uniform interval of
length M T; statistical distance is at most `L/(M T) <= 2^−κ` per output,
then summed over outputs. Fresh Paillier randomization makes response
ciphertexts fresh encryptions conditional on their plaintext, even for a
decryptor who retained the original request randomness. B's view hides A's
coordinates by semantic security. This addresses the **circuit-privacy** gap
that ordinary evaluated BFV ciphertexts do not resolve.

A uniform pad solely modulo M is insufficient: the decryptor could see high
bits of the integer cross term. A pad uniform modulo N followed by reduction
modulo M is also an invalid shortcut: wrap subtraction by N changes the
answer when M does not divide N. The public no-wrap bound and wide mask are
essential. Malicious-key/ciphertext behavior is outside the specified
honest-but-curious model; no active-security admission is claimed.

Unpacked 3072-bit N gives fixed 768-byte ciphertexts. For norm-only joint
`(q,rb)`, K=n+1 requests and n+1 replies cost at least
`2(n+1)×768 = 1,377,792 bytes per norm row`, before metadata/setup. Across
all norms: **3,042,164,736 / 4,629,381,120 bytes** for +8/+32. Public N itself
is 384 bytes; packing, public weight distribution, key generation and all
exponentiation CPU are unmeasured. This explicit secure generator is therefore
cost-vetoed without generating full-scale material. It is a concrete existence
construction, not an efficient candidate or reviewed production implementation.

For the factored cubic in section 4, A can additionally encrypt its local
`a=GrA,b=UrA,a²,a⊙b` features. B's cross terms for each channel use coefficients
`αbB+2βaB bB`, `αaB+βaB²`, `βbB`, `2βaB`, respectively. Their linear
combination plus local F(rA),F(rB) reconstructs the cubic constant. Apply the
same independent wide-mask technique to every required feature/constant.
This is a function-specific extension with more ciphertexts, not a claim that
one OLE or one Ring PCG call produces a cubic tensor correlation.

### Lifetime and views

Bind every block to checkpoint/body, numeric domain, public maps, tensor shape,
phase, session, operation/row and parties. Reserve identifiers at both workers
before the first encrypted request/opening. Cancellation, malformed result,
timeout, partial delivery and replay permanently burn the **entire joint block**;
never resample/reissue the same operation or restore seeds from a cache. Only
fresh operation identifiers with fresh masks can run again. Any stored KV mask
cross-correlations must survive exactly as long as the corresponding KV state
and remain bound to the request.

The probe exercises logical consume/cancel/binding and zeroes its arrays on
burn; it has no durable distributed burn ledger, authenticated transport,
asynchronous generation-cancellation API, or guaranteed physical memory erasure.
Those remain implementation obligations. A test harness may reconstruct both
shares for an oracle after execution; no evaluator method is given both local
shares or the HE secret key.

Party-private seeds may expand **their own** randomness computationally. A
provider-expandable common seed revealing r or b destroys input masking. Even
party-private PRG seeds do not generate the required cross-function products.
Compressing one party's output pad does not make the complementary correction
free: account for its secure generation and delivery. Explicit material
counts here are representation costs; they are not lower bounds against PCGs
or computational compression.

## 6. Executed BFV joint-block cost probe

The actual encrypted inputs are party A's own rA, repeated local bA, GrA and
UrA. B adds only its own corresponding quantities, evaluates q,rb and
D-projected gate products, and independently masks **every** scalar/vector
output. A decrypts only the masked results. B holds a public-only context;
the private key never enters its context. The oracle reconstructs generated
coefficients only outside the evaluator to check parity. Consuming workers
then evaluate norm statistics, same-source projected quadratic outputs and
vector×synthetic-scalar shares with one source opening and one scalar opening.

BFV is F65537, **not** Z/2^32 or a certified Qwen accumulator. G/U/D are public
synthetic integer matrices (16→24→2), not checkpoint weights. Each sample uses
independent bounded rows and fresh masks. There is no approximation or decrypt
fallback; parity failures stop execution.

Measured environment: Python 3.13.15, TenSEAL 0.3.17, macOS 26.5.2 arm64.
Degree 8192, standard backend coefficient chain, n_threads=1 per context;
public serialization includes public, full rotation and relinearization keys.

- A→B public context: **54,750,082 bytes**, once per tested key/parameter set.
- A private-context serialization: **812,401 bytes**, local, never transmitted.
  This is one serializer's private representation, not full resident key memory.
- Context generation plus evaluator import: **0.662512 CPU seconds**.
- Observed whole-process peak RSS: **735,363,072 bytes**, below 1 GiB.
  Work is dimension-bounded; the script checks observed RSS rather than claiming
  an OS-enforced allocation cap. No full-scale material was created.

| Independent rows | 1 | 4 | 16 |
|---|---:|---:|---:|
| A→B request envelope bytes, four ciphertexts | 1,729,976 | 1,729,975 | 1,729,796 |
| B→A result envelope bytes, 1+3×rows ciphertexts | 1,730,074 | 5,621,758 | 21,189,380 |
| Warm generator bodies, every direction | **3,460,050** | **7,351,733** | **22,919,176** |
| Cold generator bodies, including context | 58,210,132 | 62,101,815 | 77,669,258 |
| Warm generator bytes per row | 3,460,050 | 1,837,933.25 | 1,432,448.5 |
| B local output-mask u32 bytes, no plaintext transmission | 76 | 304 | 1,216 |
| Each party's local r,b u32 bytes | 68 | 272 | 1,088 |
| One-use complete material u32 bytes, both parties | 288 | 1,152 | 4,608 |
| Online source-opening envelope bytes, both directions | 460 | 846 | 2,382 |
| Online scalar-opening envelope bytes, both directions | 340 | 364 | 460 |
| Warm generation + consuming envelopes, all-link | **3,460,850** | **7,352,943** | **22,922,018** |
| Cold generation + consuming envelopes, all-link | **58,210,932** | **62,103,025** | **77,672,100** |
| Generation CPU seconds | 0.123869 | 0.414266 | 2.222991 |

Individual ciphertext bodies are roughly 432 kB; the probe prints every body
length, both context serializations, all envelope lengths and all local mask
storage. Request/reply counts include redundant encrypted projection inputs
and scalar-return ciphertexts. Nothing sent is excluded as a "mask" or "setup"
body. Material is created locally from returned ciphertexts, so it is stored
but not transmitted again as plaintext dealer arrays. Maps are preinstalled
public synthetic fixtures; real cold model distribution and full wire remain
unmeasured. Sizes vary slightly with randomized compressed serialization.

**Failure evidence:** degree 4096 default BFV parameters failed exact statistic
parity (one observed result 59,473 versus oracle 49,186). Returning a larger
8192 context restored exact parity for the tested block. `BFVVector.pack_vectors`
rejects mixed vector sizes with `ValueError: vectors sizes are different`;
packing equal-size post-evaluation scalar ciphertexts also failed parity in the
8192 parameter set. Final probe returns separately checked ciphertexts. It
does not assume free post-evaluation repacking or investigate larger contexts.
These failures are not evidence of a universal BFV bound.

**Security limit:** TenSEAL's ordinary evaluated ciphertexts are not sanitized
for circuit privacy against their secret-key holder. Output plaintext masks
alone do not establish a simulatable view of B's masks; ciphertext noise can
depend on them. Therefore this BFV path is explicitly marked
`cryptographically_secure_generator_admitted=false`. Its role separation,
parity and measured bodies are real; its deployment privacy is not established.
The AHE construction above supplies a separate secure specification with its
own much worse known cost. No BFV number is promoted as a secure end-to-end win.

## 7. Primary literature and adaptation boundary

Primary author/publication pages below were fetched on 2026-10-01. Abstract
and bibliographic review is distinguished from full-paper reproduction; the
equations/protocol adaptation in this report are our explicit derivation.
No upstream code or additional dependencies were installed.

- **[S1] Donald Beaver, _Efficient Multiparty Protocols Using Circuit
  Randomization_, CRYPTO 1991.**
  [IACR record](https://www.iacr.org/cryptodb/data/paper.php?pubkey=1013),
  DOI `10.1007/3-540-46766-1_34`. Circuit-dependent preprocessing is the
  foundation; a public bilinear map can consume a generalized `(a,b,B(a,b))`
  correlation through `B(d,b_i)+B(a_i,e)+B(d,e)`. Tensor/matrix correlations
  specialize this identity; generic scalar-triple counts are not mandatory.
- **[S2] Boyle, Couteau, Gilboa, Ishai, _Compressing Vector OLE_, CCS 2018.**
  [Primary ePrint 2019/273](https://eprint.iacr.org/2019/273). Short correlated
  seeds expand VOLE via FSS and decoding-hard codes. A VOLE schema can be
  written sender `(u,v)`, receiver `(Δ,w=Δu+v)`. It captures a common scalar
  across vector coordinates; it does not also supply `<r,r>` or arbitrary
  projected cubic features from the same additive source. Programming inputs,
  orientation, cross-share conversions and adjustments require their own cost.
- **[S3] Boyle et al., _Efficient Pseudorandom Correlation Generators: Silent
  OT Extension and More_, CRYPTO 2019.**
  [Primary ePrint 2019/448](https://eprint.iacr.org/2019/448). PCG security is
  correlation-specific; secure seed sampling precedes silent local expansion,
  and composition has a stronger protocol-security requirement. An ordinary
  independent PRG is not a nonlinear correlation generator.
- **[S4] Li, Xing, Yao, Yuan, _Efficient Pseudorandom Correlation Generators
  over Z/p^k Z_, CRYPTO 2025.**
  [Primary ePrint 2025/1223](https://eprint.iacr.org/2025/1223). The abstract
  explicitly covers Galois-ring OLE, authenticated triples, matrix triples and
  circuit-dependent preprocessing. That is relevant, but **does not instantiate
  this report's joint norm/inverse/SiLU/attention distribution**. The local
  `ring_pcg_reference.py` checks tiny sparse cross-term and trace algebra;
  it has neither reviewed hardness parameters nor distributed seed setup nor
  arbitrary nonlinear coefficient expansion. The existing R15 note records
  the cached full source and QA-SD attack caveats. The Springer chapter endpoint
  returned a JavaScript challenge; the ePrint abstract was accessible.
- **[S5] Hasler et al., _Overdrive LowGear 2.0: Reduced-Bandwidth MPC without
  Sacrifice_, ASIA CCS 2023.**
  [Primary ePrint 2023/462](https://eprint.iacr.org/2023/462). Explicitly studies
  matrix triple generation and HE-assisted preprocessing. Its reported offline
  improvement is not an online vector-opening removal or a PLLM cost result.
  Also relevant: Rathee, Schneider, Shukla,
  [_Improved Multiplication Triple Generation over Rings via RLWE-based AHE_](https://eprint.iacr.org/2019/577),
  CANS 2019, semi-honest two-party Z/2^ell. Its ring protocol is not reproduced
  merely by substituting the prime-modulus TenSEAL probe.
- **[S6] Pascal Paillier, _Public-Key Cryptosystems Based on Composite Degree
  Residuosity Classes_, EUROCRYPT 1999.**
  [IACR record](https://www.iacr.org/cryptodb/data/paper.php?pubkey=2681),
  DOI `10.1007/3-540-48910-X_16`. Additive encryption/rerandomization enables
  the explicit cross-term generator in section 5; the wide-mask and no-wrap
  adaptation is specified here rather than attributed as a reproduced paper.
- **[S7] Couteau et al., _On Compressing Linearly Shared Correlations_, 2026.**
  [Primary ePrint 2026/255](https://eprint.iacr.org/2026/255), revision Sept 7.
  Discusses HSS-based general linear-output correlations and threshold-shared
  constant-degree constructions under additional assumptions. This provides
  an adjacent research direction, not a practical arbitrary-layer PCG or a
  compatible implementation in this repository.

## 8. Decision gate for further work

**Stop:** scaling explicit norm material; the measured BFV generator; treating
the cubic numerator as pretrained SiLU; reusing masks across dependent sources;
or claiming seed compression alone gives a tenfold response.

**Pursue only if a new complete-region design does all of the following:**

1. Eliminates substantially more norm/source openings, rather than only
   material bytes; uses exact numeric domains, protected signed lifting and
   every rounded edge of the fixed checkpoint.
2. Gives a concrete compatible generator for the entire dependent attention
   and MLP region, including inverse, softmax, KV cross-correlations and
   probabilities×V. Document both worker views and circuit privacy.
3. Accounts for every generator/setup/opening/feedback/output body and one-use
   storage; passes both cohorts' online and all-link budgets before executing
   a whole layer. Streaming can reduce peak storage, not transmitted bytes.
4. Passes whole-region held-out prefill, same-token decode and free-generation
   fidelity with most body computation remote. Public map reuse should amortize
   weights and computation without treating fresh secret masks as reusable.

Known result: real algebraic opening reduction, explicit cost veto, bounded
encrypted generation parity. Unknown: a viable secure compact full-layer
generator, exact checkpoint numeric behavior, complete wire and full CPU costs.
**No protected runtime, topology or tenfold model claim is admitted.**
