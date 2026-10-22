# NEXUS numeric graph and decoder contract audit

Source inspection only, 22 October 2026. No upstream runtime, checkpoint, tests,
builds or encrypted execution performed. This evidence does not qualify a trained
decoder, backend, numerical profile or security claim.

## Immutable sources and existing approvals

- Final paper: *Secure Transformer Inference Made Non-interactive*, NDSS 2025,
  DOI `10.14722/ndss.2025.230868`; `papers/r21-nexus.pdf`, SHA-256
  `c809182345f27f76e21d983d97dffa0d086dbcd17dabc265f0df9fb6ca9b70f2`.
  Lock: `paper/methods/source-lock.v1.json:71–78`. Page references below use
  printed PDF pages.
- Artifact: `zju-abclab/NEXUS` commit
  `e15058c7f8f58fcf037c4709211109cb1be7d1e0` (20 September 2024, before final
  paper). Artifact paths/lines below refer to immutable raw files at
  `https://raw.githubusercontent.com/zju-abclab/NEXUS/e15058c7f8f58fcf037c4709211109cb1be7d1e0/`.
  Existing selected-file hashes: `paper/reproduction-contracts/encrypted-inference.md:248–259`.
- Governing contracts: `paper/methods/encrypted-computation-contracts.md:66–156`
  and `paper/methods/DECISIONS.md`. D1 approves CPU-first development; D3 selects
  common Qwen while preserving method contracts; D4 permits bounded Poulpy
  qualification, **not** NEXUS representation/backend adoption. D5 requires ordinary
  SDK integration. D9's fixed-count private feedback applies to four named shared-
  computation methods, **not NEXUS**. No other method's constants or range policy
  transfer automatically.

## Numeric graph: paper versus artifact versus current PLLM

### Encoding, scale and rescale

Paper pp.9–12 uses approximate RNS-CKKS, message scale approximately `2^50`.
It does not establish a fixed-point Q18/Q12 contract. Exact external encoding
rounding, coefficient precision, ordered rescale primes, per-node scale
normalization and accumulated error remain unresolved.

Artifact `src/ckks_evaluator.cpp:366–475,524–546` explicitly encodes constants,
modulus-switches operands, relinearizes ciphertext products and rescales; some
branches assign configured scale directly to ciphertext metadata. These are
artifact choices, not proof of final-paper numerical identity.

Current PLLM `python/pllm/runtime/ckks_reference.py:1–5,39–77` separates client
decryptor from provider public evaluation material. Its arithmetic uses `2^40`
scale and permits relative rescale drift `0.0001` before metadata normalization.
Framing is not authenticated transport. This bounded operator backend is not
the published NEXUS context.

### Softmax and recovered reciprocal

Paper p.8, Eq.(4) and following unnumbered expression; Algorithm 4 p.9:

```text
z_i = a_i - c                 # public constant replacing maximum
e_i = (1 + z_i / 256)^256     # eight squarings, intended z_i <= 0
y_i = e_i / sum_j(e_j)
```

Missing: actual `c`, lower shifted bound, denominator scaling/domain,
Goldschmidt seed/count and underflow/error margin. Reported average exponential
error `1e-5` is not a uniform domain certificate. No clipping follows from paper.

Artifact differences:

- `src/ckks_evaluator.cpp:534–547`: divisor **128**, **seven** squarings.
- `src/softmax.cpp:23–39`: scale denominator by **0.01**, invert, then multiply
  reciprocal by **0.01**. Comment claims scaled denominator in `[0,1]`; zero
  cannot be admitted, and comment supplies no checkpoint-wide proof.
- `src/ckks_evaluator.h:100`: default `inverse(..., iter=4)`.
  `src/ckks_evaluator.cpp:487–508`: initialize `e=1-u`, `r=1+e`; perform four
  additional updates `e=e*e; r=r*(1+e)`. Thus exact-real algebra gives
  `u*r = 1-(1-u)^32`. Convergence requires `0<u<2`; approximation and CKKS
  errors still need separate bounds. This is not four total factors or PLLM's
  three-stage schedule.

Current PLLM `python/pllm/runtime/ckks_softmax.py:9,18–25,34–48,68–73` uses
shift `1/8`, eight squares, width-normalized denominator, three reciprocal stages
and sequential row rotations. File explicitly labels public-range choices as
adaptations. They do not resolve source-profile gaps.

### LayerNorm and recovered inverse-square-root schedule

Paper p.9 Eq.(5):

```text
z_i = n*a_i - sum_j(a_j)
y_i = gamma*sqrt(n)*z_i / sqrt(sum_j(z_j*z_j)) + beta
```

Missing epsilon, zero-variance behavior, inverse-square-root operand domain,
seed/count and checkpoint-equivalence contract.

Artifact `src/layer_norm.cpp:6–32` squares/sums supplied input, multiplies by
**1/768**, invokes `invert_sqrt(...,4,2)`, then multiplies original input.
Helper contains no centering, gamma, beta or epsilon. Caller-side mapping remains
necessary; it cannot be treated as complete paper LayerNorm.

Recovered artifact schedule (`src/ckks_evaluator.cpp:366–484,517–531`):

```text
y = -1.29054537e-04*v + 1.29054537e-01
repeat 4 times: y = 1.5*y - 0.5*v*y^3
x = v*y; h = y/2
repeat 2 times:
    r = 0.5 - x*h
    x = x + x*r
    h = h + h*r
return 2*h
```

These recovered constants do not supply a certified domain or final-paper mapping.

### GELU and sign composition conflicts

Paper p.8 Eq.(3): zero for `x<=-4`; cubic `P` for `-4<x<=-1.95`;
degree-six `Q` for `-1.95<x<=3`; identity above `3`. Sign-derived selectors
combine segments. Observed `[-8,8]` and average approximation error `1e-4`
are empirical evidence, not public range proofs. The cited primary papers supply
explicit `P/Q` polynomials (below); NEXUS's exact coefficient precision and CKKS
interpretation remain unresolved. A new fit is not needed to recover those
printed polynomials.

Artifact `src/gelu.cpp:10–23,64–110` instead uses breakpoints **±3.5**,
comparison scale **1/8.5**, and degree-12 central polynomial. Coefficients in
power order `[0,1,2,4,6,8,10,12]`:

```text
[2.25775755e-04, 0.5, 3.96880960e-01, -6.37042698e-02,
 8.38841647e-03, -7.17830961e-04, 3.49617829e-05, -7.26059653e-07]
```

Inspected `src/gelu.cpp` SHA-256:
`37d6afb3db39db65c6cd255cb809b865f723ae1deca8d606fcf148477a5aef20`.
These are **not** missing paper cubic/degree-six coefficients.

Paper p.4 uses comparison normalization into `[-1,1]` by a public magnitude
bound, composition `f(f(g(g(x))))`, and uncertainty band `±2^-20`.
Artifact `src/ckks_evaluator.h:21–26` supplies:

```text
f(x) = (315*x - 420*x^3 + 378*x^5 - 180*x^7 + 35*x^9)/128
g(x) = (5850*x - 34974*x^3 + 97015*x^5 - 113492*x^7 + 46623*x^9)/1024
```

However, default `sgn_factor=0.5` (`:98`) scales final polynomial in **each**
g/f loop (`src/ckks_evaluator.cpp:181–211`). For `(d_g,d_f)=(2,2)`, actual
expression is `0.5*f(f(0.5*g(g(x))))`, not merely half of paper composition.
Resolve this before adopting selectors. Four degree-nine compositions have
formal degree 6561; paper p.12's degree-36 discussion cannot establish their
certified error bound.

### Recovered primary-source polynomials (not an adopted profile)

PUMA v3, `papers/puma.pdf`, §4.2, p.5 Eq.(3), supplies the following literals in
ascending ordinary powers of **unshifted, unnormalized** `x`:

```text
P = [-0.5054031199708174,
     -0.42226581151983866,
     -0.11807612951181953,
     -0.011034134030615728]
Q = [0.008526321541038084,
     0.5,
     0.3603292692789629,
     0,
     -0.037688200365904236,
     0,
     0.0018067462606141187]
```

PUMA attributes the coefficients to `numpy.polyfit`; it does not give the sampling
grid, weights or underlying higher-precision fit. PUMA Eq.(2)/Algorithm 1 assigns
`x=-4` to `P` and `x=-1.95` to `Q`. **NEXUS's own Eq.(3) endpoints govern NEXUS**:
those points belong to zero and `P`, respectively.

BumbleBee final, `papers/r22-bumblebee-final.pdf`, Appendix D, printed p.17,
gives the same construction with shorter decimal tails:

```text
P = [-0.5054031199708174, -0.4222658115198386,
     -0.1180761295118195, -0.0110341340306157]
Q = [0.0085263215410380, 0.5, 0.3603292692789629, 0,
     -0.037688200365904, 0, 0.0018067462606141]
```

BumbleBee's own p.8 Eq.(2)/p.9 Algorithm 3 instead uses thresholds
`-5,-1.97,3` and an epsilon-shifted construction; those semantics do not transfer
to NEXUS. The two citations strongly identify concrete polynomials, but NEXUS
does not identify which printed precision it adopts. PUMA's Q18 or BumbleBee's
parameterized fixed-point floor encoding does not specify CKKS coefficient
conversion, evaluation order or rescaling. Their stated target is tanh GELU, not
an automatically transferable error guarantee against exact `x*Phi(x)`.
PUMA's reported mean `<0.00168`, median `<4.41e-05`, maximum `<0.01403` errors
do not certify NEXUS's reported average `1e-4` on an unspecified distribution.

The sign polynomials also have an independent primary source: Cheon–Kim–Kim,
*Efficient Homomorphic Comparison Methods with Optimal Complexity*, ASIACRYPT
2020, DOI `10.1007/978-3-030-64834-3_8`, printed p.232 §3.1 and p.241 §3.5:

```text
f4(x) = (315*x - 420*x^3 + 378*x^5 - 180*x^7 + 35*x^9)/128
g4(x) = (5850*x - 34974*x^3 + 97015*x^5 - 113492*x^7 + 46623*x^9)/1024
```

These are exact printed dyadic polynomials. `g4` denotes the published
finite-precision `g_(4,1/4)` choice, not every member of the paper's `g_(n,tau)`
family. The source mentions `1e-4` fitting stopping precision separately from its
`2^-10` coefficient precision. Algorithm 3, p.242, applies `g` repetitions before
`f` repetitions, then converts the final sign to a comparator with `(s+1)/2`.
It does not introduce the artifact's intermediate factor `0.5`. NEXUS's degree
and repetition counts do not explicitly identify `tau` or coefficient rounding.
Source Corollary 3 relates repetition counts to both input separation and output
accuracy; it does not prove NEXUS's `(2,2)` choice for every claimed input.

NEXUS p.8 does state segment indicators, the weighted output
`b0*0 + b1*P(x) + b2*Q(x) + b3*x`, and adjacent-selector differences in its
near-boundary example. These are not a complete numerical selector contract.
For a public input domain `[L,U]`, comparison at threshold `t` needs a bound on
the **difference**, `Delta_t=max(abs(L-t),abs(U-t))`. Normalized separation
`2^-20` then represents `Delta_t*2^-20` in original units, not uniformly
`2^-20`. Odd sign polynomials yield zero at an exact threshold; comparator
conversion gives `1/2`, not an exact Boolean. The near-boundary blending
discussion supplies neither a complete CKKS rounding contract nor a quantified
boundary-error certificate.

Remaining concrete decisions, before implementing a numerical profile:

1. Adopt the full PUMA printed literals as an explicitly source-derived NEXUS
   adaptation, or obtain author-confirmed NEXUS literals; no new refitting.
2. Specify coefficient conversion, evaluation order, precision and CKKS rescaling.
3. Explicitly select `f4,g_(4,1/4)`, public threshold-difference domains, and
   tie/blending and approximation-error acceptance contracts.

None of these choices is approved or implemented by this audit.

Primary-source SHA-256 values, verified locally:

| Source | SHA-256 |
| --- | --- |
| PUMA v3 | `84c82810a16be95b41200ba8e0071c39a8cadb15a45008ff8e75e295e0960973` |
| BumbleBee final | `184bbe5104f183d4df66db2413f89c96c6c814e0b0ec6297f19745a1fa6e203b` |
| Cheon–Kim–Kim publisher PDF | `3cebc1b089066707780fd54a4b16c71fec56ec56e6b7be72824a0a4570519146` |

The publisher PDF is an ignored 1,039,467-byte source-inspection input at
`target/nexus-source-review/cheon-kim-kim-asiacrypt2020.pdf`, fetched from
`https://link.springer.com/content/pdf/10.1007/978-3-030-64834-3_8.pdf`.
It is not vendored, linked, executed or required by a runtime.

## Approved bounded GELU numerical experiment

D24 approves this research experiment only, not a NEXUS numeric profile or CKKS
adoption. The reusable experiment is
`examples/benchmarks/nexus_gelu_numerics.py`; its ordinary output is
`docs/evidence/nexus-gelu-numerics.json`. With optional `mpmath==1.3.0` installed,
the executed command is:

```sh
uv run --no-sync python examples/benchmarks/nexus_gelu_numerics.py --output docs/evidence/nexus-gelu-numerics.json
```

The frozen candidate uses exact-rational PUMA printed coefficients, NEXUS's
thresholds and endpoint oracle, and `f4(f4(g4(g4(z))))`. The public research domain
is `[-8,8]`, not a measured model-layer domain. Threshold-difference normalizers
are `12,199/20,11`; normalized separation `2^-20` therefore has distinct
original-unit widths. No refit, clipping, extra repetitions, acceptance tolerance,
encrypted arithmetic or source-model execution is introduced.

At 80 decimal digits, the finite samples find:

| Diagnostic | Sampled maximum absolute error |
| --- | ---: |
| Blended output versus exact piecewise polynomial | 0.00793177 |
| Blended output versus tanh GELU | 0.01180188 |
| Blended output versus `x*Phi(x)` | 0.01221802 |
| Full versus shorter printed coefficient precision | 5.47e-15 |

These are sampled estimates, not uniform error bounds. At normalized distance
`2^-20`, the sign composition's error is approximately **0.99981150** on both
sides of every threshold. The fixed repetition count therefore does not provide
an accurate sign estimate at that separation. Exact thresholds produce comparator
`1/2` and blend adjacent branches, not the endpoint oracle's hard branch.

Directed-rounding interval evaluation covers the entire research domain, but
reaches the 128-cell cap with all 128 stage-domain enclosures unresolved. Its
loose bounds supply neither a useful uniform error certificate nor evidence that
actual values leave the stage domain. No error tolerance has been approved.
Independent signed-point rational compositions and all printed coefficient
literals are covered by the 27 passing normal regression tests. The outcome is a
bounded negative selector diagnostic; executable NEXUS and its open numerical,
backend and decoder contracts remain unqualified.

## Missing HE, publication and complete-request contracts

Paper pp.9–12 specifies degree `65536`, slots `32768`, 1763-bit modulus,
claimed 128-bit security, total depth `35`, bootstrap depth `14`, nominal usable
depth `21`, `q0≈2^60`, other primes `≈2^50`, Gaussian `sigma=3.2`, secret
weight `192`. Exact Q/P chains, evaluation-key/security treatment, refresh
parameters/errors and operation-level scale/layout ledger remain missing.
Table IV's multiplication `20→21` cannot itself restore levels. Artifact
`src/bootstrapping.cpp:33–75` instead uses degree `32768`, scale `2^46`,
16 remaining plus 14 bootstrap levels and disabled library security checking;
this is neither the final-paper context nor proof of insecurity by itself.

Paper §III.C and Appendices B,D (p.16) require two key domains: compressed
server-encrypted weights, client-computed encrypted `UW`, commutable addition of
client RLWE term, then removal of server term. Online masked input combines with
`Enc_client(UW)`. Uniform one-time pads in `R_Q` are essential; bounded real
masks do not inherit proof. Model-bound publication, client preprocessing costs,
mask freshness/retirement and composition with already-encrypted hidden layers
must be explicit. Column/diagonal/coefficient layout transitions and attention-
value multiplication also remain unresolved.

Paper Algorithm 2 p.7 returns encrypted **selection vectors**, nominally multi-hot
on exact ties, not full logits. Its `2m<S` condition cannot accommodate either
full source vocabulary (`30522`, `128256`) in `S=32768` slots. Cross-ciphertext
selection, padding, near-tie separation, tie policy and decoding thresholds need
specification; reduced vocabulary does not qualify.

Table IV p.12 establishes one-word BERT/Llama workloads, not persistent private
generation. Missing complete transaction:

- Initial embedding custody; private selection-to-embedding feedback and append.
- Per-token versus final-only disclosure, stopping rule and complete head.
- Actual common-Qwen causal attention, RMSNorm/SiLU/RoPE/GQA mapping; table's
  GELU/LayerNorm labels do not establish Llama architecture fidelity.
- Full-prefix versus persistent encrypted KV, including layout, capacity, level
  recovery, refresh, request binding, cancellation and retirement.
- Source/checkpoint-bound numeric identity, public range certificates, complete
  quality evidence and D5 ordinary SDK/role/gateway/benchmark availability.

No blanket approval packet is supported by current evidence. Artifact constants
narrow investigation but do not close final-paper or decoder-contract gates.

## One smallest proposed next slice

**Paper Eq.(5) centered-second-moment prefix**, using Algorithm 3 QuickSum:

```text
a -> s = QuickSum(a) -> z = n*a - s -> v = QuickSum(z*z)
```

Propose reusable masked row-folding/reduction and this arithmetic prefix, with
explicit non-power-of-two padding/row masks for BERT width **768**, independent
dense algebra oracle and existing arithmetic/owner/admission interfaces. It
advances normalization graph without inventing reciprocal seed, epsilon,
clipping, refresh or output policy. Intermediate `v` stays within arithmetic
ownership; this proposal does not introduce a reveal boundary.

**This is a proposed dependency slice, NOT approved implementation, encrypted
execution or backend qualification.** Source-profile execution still requires
numerical/backend mapping and resource admission; even a completed prefix would
not qualify full LayerNorm or trained decoder.

Existing `crates/pllm-core/src/coefficient_compression.rs:1–13` already implements
bounded clear coefficient compression/decompression algebra and explicitly
excludes CKKS encoding, rounding, keys, masks and noise. **Do not repeat existing
coefficient algebra as another NEXUS milestone.** Reuse it where applicable;
do not relabel generic BFV coefficient repacking or bounded CKKS softmax as
complete NEXUS protocol execution.
