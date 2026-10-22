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
are empirical evidence, not public range proofs. Exact paper `P/Q` coefficients
and precision remain to be resolved from cited constructions.

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
