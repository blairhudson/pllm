# NEXUS numeric graph and decoder contract audit

Source inspection and the bounded clear GELU experiment below, 22 October 2026.
No upstream runtime/checkpoint execution, upstream tests, builds or encrypted
execution performed; local numerical regression tests are recorded below. This
evidence does not qualify a trained decoder, backend, numerical profile or security
claim.

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
`examples/benchmarks/nexus_gelu_numerics.py`. The current ordinary output, including
the added numerical blockers, is `target/nexus-gelu-numerics.json` (ignored build/
diagnostic space); `docs/evidence/nexus-gelu-numerics.json` is the earlier output
and was not regenerated for this run. With the existing optional
`mpmath==1.3.0` and pytest installed, the commands actually executed were:

```sh
uv run --no-sync python examples/benchmarks/nexus_gelu_numerics.py --output target/nexus-gelu-numerics.json
uv run --no-sync python -m pytest tests/test_nexus_gelu_numerics.py -q
```

The frozen candidate uses exact-rational PUMA printed coefficients, NEXUS's
thresholds and endpoint oracle, and `f4(f4(g4(g4(z))))`. The public research domain
is `[-8,8]`, not a measured model-layer domain. Threshold-difference normalizers
are `12,199/20,11`; normalized separation `2^-20` therefore has distinct
original-unit widths. No refit, clipping, extra repetitions, acceptance tolerance,
encrypted arithmetic or source-model execution is introduced.

### Sampled observations, not uniform certificates

At 80 decimal digits, 381 distinct original-unit inputs (257-point uniform grid
plus thresholds and in-domain normalized dyadic offsets) give:

| Diagnostic | Sampled maximum absolute error |
| --- | ---: |
| Blended output versus exact piecewise polynomial | 0.00793177 |
| Blended output versus tanh GELU | 0.01180188 |
| Blended output versus `x*Phi(x)` | 0.01221802 |
| Full versus shorter printed coefficient precision | 5.47e-15 |

These are sampled estimates, not uniform error bounds. The shorter printed
coefficients do not explain the observed selector problem. Exact thresholds
produce comparator `1/2` and blend adjacent branches, not the endpoint oracle's
hard branch. Endpoint assignments remain **zero at -4, P at -1.95, Q at 3**.

### Rigorous point enclosures and exact oracle jump lower bounds

The output's `numeric_blockers.sign_point_witnesses` uses outward interval
arithmetic, including rational coefficient conversion. The following numbers are
rounded displays of those enclosures; exact binary endpoints are in the JSON.
They hold on both signs and map to `x=t+Delta_t*z` for each threshold:

| Normalized point z | Sign error | Comparator error | Band relation |
| --- | ---: | ---: | --- |
| `+/-2^-20` | 0.999811499255490484 | 0.499905749627745242 | Band boundary |
| `+/-2^-19` | 0.999622998521496399 | 0.499811499260748200 | Strictly outside |

These point errors lower-bound any uniform error budget covering those points;
they are not upper bounds over a domain. The strict-outside witnesses avoid any
ambiguity about whether the nominal band includes its boundary. Independent
exact-rational compositions test both signs at both separations.

Outward point enclosures also prove negative adjacent-difference weights:
`b1(-85/64)` is approximately `-1.3252989633e-11`, and `b2(83/16)` approximately
`-1.3221357765e-11`. Their upper endpoints are negative, while the individual
comparators at those points remain in `[0,1]`. This is not floating-point noise
and does not imply that all weights are negative or that an input left the domain.

The exact P/Q literals make the piecewise oracle discontinuous. For any continuous
approximation Y, continuity and the triangle inequality on the two one-sided
limits imply `sup_x abs(Y(x)-F(x)) >= abs(F(t+)-F(t-))/2`. The experiment's
`numeric_blockers.oracle_jumps` evaluates these differences with exact rationals:

| Threshold | Exact half absolute jump |
| --- | ---: |
| `-4` | 0.000313315939415676 |
| `-1.95` | 0.00361628446467038256759398984375 |
| `3` | 0.00793176970057680015 |

Thus **0.00793176970057680015 is a rigorous uniform-error lower bound**, not an
achieved uniform upper bound, for any continuous approximation to this fixed
endpoint oracle on `[-8,8]`. It does not apply as a lower bound against tanh GELU
or `x*Phi(x)`, and does not depend on encrypted-arithmetic errors. Changing sign
repetitions alone cannot remove this oracle-jump floor.

### Bounded interval result and qualification status

Directed-rounding interval evaluation covers the entire research domain, but
reaches the 128-cell cap after 246 cell evaluations; maximum leaf depth is 7
(configured caps: 128 cells, depth 12, 256 evaluations). All 128 stage-domain
enclosures remain unresolved. These loose bounds supply neither a useful uniform
error certificate nor evidence that actual values leave the stage domain.

The original D24 regression pass covered **34 tests**, including all printed literals,
independent signed rational compositions, exact and one-sided endpoint semantics,
closed-cell enclosures, negative selector witnesses, normalizer units and bounded
refinement. No error tolerance has been approved. The fixed selector is
**unqualified**: it is not an exact endpoint oracle and is inaccurate as a sign
estimate even strictly outside the stated band. Acceptance as a looser blended
GELU approximation is neither disproved for every possible budget nor established
by these samples. Executable NEXUS and its numerical/backend/decoder contracts
remain unqualified.

The user subsequently approved the separate four-schedule clear-arithmetic study
in `docs/plans/nexus-gelu-numeric-experiment.md`. Its executed result follows.
D24's frozen candidate is unchanged, and neither study adopts a numerical contract.

## Approved four-schedule clear-arithmetic follow-up

The user answered **"Approve bounded study"**. Approval is for this experiment
only: no CKKS, runtime, backend, numerical acceptance policy or executable profile
is adopted. Only the pinned printed Cheon–Kim–Kim `f4,g_(4,1/4)` coefficients above
are used; the new repetition counts are research adaptations, not claimed NEXUS
schedules. PUMA P/Q, NEXUS thresholds and endpoints, adjacent selectors and public
difference normalizers are unchanged. No upstream runtime was executed.

Actual commands:

```sh
uv run --no-sync python examples/benchmarks/nexus_gelu_numerics.py --follow-up --output target/nexus-gelu-follow-up.json
uv run --no-sync python -m pytest tests/test_nexus_gelu_numerics.py -q
uv run --no-sync ruff check examples/benchmarks/nexus_gelu_numerics.py tests/test_nexus_gelu_numerics.py
```

The ordinary JSON records `completed_bounded_study_not_certified`. Study work
took **3.374 seconds**, with **1,936 attempted interval cell evaluations** (484
per candidate); no candidate timed out. Each candidate used **291 normalized
sample locations and 381 original-unit locations**, fewer than the approved
307+381 caps. Standalone normalized grids are classified separately into
`D_epsilon=[-1,-epsilon] union [epsilon,1]`. Original-unit samples are separately
classified into `E_epsilon` (all three normalized differences separated by epsilon)
and its within-band complement, for epsilon `2^-4,2^-10,2^-20`.

### Sampled comparisons — not uniform upper bounds

| (r,s) in `f4^s(g4^r(z))` | Max sign error on sampled D_(2^-4) | D_(2^-10) | D_(2^-20) |
| --- | ---: | ---: | ---: |
| (2,2) | 2.64587e-11 | 0.8088406552 | 0.9998114993 |
| (6,2) | 2.64415e-11 | 2.64415e-11 | 0.8013106704 |
| (10,2) | 2.64263e-11 | 2.64263e-11 | 2.64263e-11 |
| (10,3) | 1.01493e-52 | 1.01493e-52 | 1.01493e-52 |

The sampled counts on these three separated sets are 242, 262 and 282,
respectively; points at zero/inside the relevant gap are not sign-domain samples.
Corresponding comparator errors are half the sign errors. These tables do not
certify any entire D_epsilon, nor behavior arbitrarily near zero.

| (r,s) | Sampled max error vs piecewise oracle | vs tanh GELU | vs `x*Phi(x)` | Largest sampled negative-weight magnitude |
| --- | ---: | ---: | ---: | ---: |
| (2,2) | 0.00793176970058 | 0.01180187834 | 0.01221801982 | 1.32530e-11 |
| (6,2) | 0.00793176970058 | 0.01946455556 | 0.01987691824 | 1.32382e-11 |
| (10,2) | 0.00793176970158 | 0.01950079097 | 0.01991309323 | 1.32606e-11 |
| (10,3) | 0.00793176970058 | 0.01950079098 | 0.01991309323 | 5.16633e-53 |

Sharper sampled comparison is **not** improved end-to-end GELU accuracy: the
sharp candidates approach the fixed polynomial branches immediately below 3,
where the oracle itself has appreciable tanh/`x*Phi(x)` error. On sampled
E_(2^-20), the piecewise error drops from about `0.00793146` for (2,2) to
`3.82384e-9` for (10,2) and `1.32318e-50` for (10,3), but this separated subset
excludes the exact thresholds. It must not replace the full-domain result.

Maximum sampled selector leakage remains about **0.5**, including exact ties.
Negative-weight entries above are high-precision observations, not sign proofs
over a domain. In particular, (10,3) also has outer-weight estimates around
`2.1e-81`, at the 80-digit arithmetic noise floor; those do not prove exact
negativity. Original D24's rigorously negative point witnesses remain as above.
Printed-coefficient sensitivity remains about `5.5e-15` in all four candidates.
Every sampled intermediate stage, individual selector, inactive-branch error term
and separated/within-band group is retained in the ordinary output.

### Rigorous point diagnostics and unchanged jump floor

Outward interval point checks use only the listed normalized sample locations,
with exact binary endpoints separate from decimal display estimates. At positive
normalized separation `2^-20`, (10,2)'s sign error is enclosed near
`1.8048427129e-15`; at `2^-19`, near `9.1192182873e-39`. For (10,3), those
point-error enclosures contain zero with upper endpoints approximately
`4.18e-73` and `9.79e-73`, respectively. Containing zero does **not** establish
an exact zero error. These facts concern specific points, not whole separated
domains. Both signs are included in the JSON and independent staged-polynomial
regressions.

The exact oracle jump calculation above is independent of r and s. Its
**0.00793176970057680015 uniform-error lower bound** still applies to every
continuous candidate against the fixed piecewise oracle. It is not an upper
certificate and is not a lower bound against the two smooth GELU targets.

### Clear-arithmetic cost, unresolved intervals and remaining gate

The evaluated DAG uses ordinary Horner seeded with the highest coefficient,
without zero-coefficient folding or expanded composed polynomials. One degree-nine
stage uses one constant product, eight variable products and nine additions.

| (r,s) | Composed stages | Exact formal sign degree | Sign constant/variable products | Sign additions / multiplication-dependency depth |
| --- | ---: | ---: | ---: | ---: |
| (2,2) | 4 | 6,561 | 4 / 32 | 36 / 36 |
| (6,2) | 8 | 43,046,721 | 8 / 64 | 72 / 72 |
| (10,2) | 12 | 282,429,536,481 | 12 / 96 | 108 / 108 |
| (10,3) | 13 | 2,541,865,828,329 | 13 / 104 | 117 / 117 |

For N=r+s, the full scalar blend DAG uses `3N+2` constant products, `24N+10`
variable products, `27N+20` additions/subtractions and six public-constant
divisions; multiplication-dependency depth is `9N+1`. This excludes diagnostic
targets and sensitivity controls. An operation-counting correctness test checks
the actual evaluated DAG. These are **not CKKS level, latency, memory or security
claims**; no HE evaluation strategy is selected.

All 16 interval passes hit their 64-leaf cap: **all 1,024 leaf stage-domain
checks remain unresolved**. Each sign pass used 126 evaluations (maximum depth
7), and each GELU pass used 106 (maximum depth 6), within the 128/pass, 2,048-total
and depth-12 limits. Every full-domain partition cuts all thresholds and the three
original-unit separation widths; E_epsilon bounds select wholly separated cells
from that covering partition without another sweep. Outward bounds are retained,
but supply no useful uniform upper-error certificate or evidence of an actual
stage-domain escape.

The dedicated suite passes **48 tests** and Ruff passes. Coverage includes the
four source-polynomial compositions against independent higher-precision staged
powers, original endpoint behavior, sample/group units, actual arithmetic cost,
work/depth caps, real nested alarm interruption/restoration, incomplete/nonzero-
exit handling and rejection of nonfinite endpoint serialization. Numeric work is
hard-stopped at 120 seconds per candidate and 480 seconds total, including CLI
setup/report output; a total deadline saves only compact incomplete status after
stopping study work. No retry or budget increase was run.

**Outcome: bounded study completed, no candidate adopted or numerically qualified.**
The next user decision is the accepted target, error budget/statistic,
boundary/blending and negative-weight acceptance, and public-range/failure policy.
Useful uniform upper-error certification remains absent; encrypted conversion,
rounding, Q/P/rescale/refresh and full-profile admission remain independent gates.

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

## Separate unapproved normalization dependency proposal

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
