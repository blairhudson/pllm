# Projective / rational delayed normalization: fixed-checkpoint review

Date: 2026-10-01. Track-local research; no inference component is admitted.

## Decision

**No exact W8A8 tenfold decoder follows from carrying `(numerator, denominator)`.
Reject unrestricted cancellation across RMSNorm, softmax/value accumulation,
dynamic activation quantization, and residuals.** Real-number identities are
useful for designing protected circuits, but do not preserve the baseline's
rounded intermediate values automatically. The bounded probe produces concrete
float32, fixed-Q7, and dynamic-W8A8 counterexamples.

Two useful narrower directions survive:

1. **Stable attention summaries** `(maximum, normalizer, weighted numerator)`
   compose exactly over real arithmetic. This can avoid materializing a full
   probability matrix in a *new* protected algorithm, but requires secret max,
   exp, value products and a rounded-output contract. Plaintext FlashAttention's
   HBM/SRAM savings are not private-network savings.
2. **Rounding-aware fused predicates:** cross-multiplication can implement exact
   rational rounding without a reciprocal. This is proved against an independent
   `Fraction` oracle in the probe. It is not yet exact emulation of the existing
   float32 program. The strongest next gate is a bounded, bit-exact
   norm→dynamic-quantization certificate/predicate, including epsilon and every
   rounded edge, followed by complete-layer directed-link costing.

Naively maintaining independent secret residual denominators with the existing
32-bit vector/scalar correlation costs **31,689,216 / 48,222,720 projected online
bytes** for 39+8 / 39+32 *just for numerator alignment*. Both exceed the entire
matched online and all-link budgets before denominator products, norms,
attention, SiLU, quantization, material framing or token boundaries. This vetoes
that construction, not all possible projective protocols.

## Scope, evidence classes, and privacy

- Existing fixed checkpoint; no training, model replacement, TEE or
  provider-visible plaintext activations. Most dense body MACs stay remote.
- Existing prepared baseline retains Client / non-colluding Preparation and
  Inference roles. Proposed resident projective arithmetic instead needs an
  explicitly distinct **two-online-worker, honest-but-curious non-collusion**
  contract. Workers retain separate activation/KV/denominator shares; neither
  gets a common expansion seed or reconstruction of an intermediate.
- Client may reconstruct final hidden state, perform final norm/head and select
  each next token. Persistent KV remains protected under the resident contract.
  No private full-vocabulary argmax is added to the current baseline.
- Secret denominators are private state, not safe public metadata. Revealing
  RMS magnitudes, attention normalizers, max scores, signs, zero tests or
  correction outcomes requires its own leakage contract and is not admitted here.
- Charge Client↔each worker, worker↔worker, dealer→each worker, offline one-use
  material, setup, cold checkpoint distribution, keys and full wire separately.
  Cancellation/replay must burn correlations. Unknown protected inverse,
  square root, exponent or comparison costs remain **unknown, not zero**.

**Measured here:** small synthetic numeric/symbolic probes and their scoped tests.
**Compiler-derived here:** dimensions, semantic coverage, dependency depth and
tensor/MAC counts, matched to existing locked schedule/composition digests.
**Projected here:** arithmetic body costs for specified 32-bit correlations.
**Not measured:** protected layer execution, independent server transport,
real-checkpoint activation ranges/quality, latency, full wire or cold transfer.
No giant weight loading or network execution occurs.

## Reproduction and source locks

New files:

- `scripts/probe_projective_feasibility.py`
- `tests/test_projective_feasibility.py`
- This report.

```sh
uv run --no-sync python scripts/probe_projective_feasibility.py --with-pinned-config
uv run --no-sync pytest -q tests/test_projective_feasibility.py
uv run --no-sync ruff check scripts/probe_projective_feasibility.py tests/test_projective_feasibility.py
uv run --no-sync ruff format --check scripts/probe_projective_feasibility.py tests/test_projective_feasibility.py
```

`--with-pinned-config` reads only cached official `config.json`, with
`local_files_only=True`; omitting it runs only synthetic probes. Output is JSON
on stdout, with no additional evidence file written. Seed: `20261001`.
Observed environment: Python 3.13.15, NumPy 2.5.3,
`macOS-26.5.2-arm64-arm-64bit-Mach-O`.

Pinned checkpoint:
`Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`.

| Lock | Value |
| --- | --- |
| Official config SHA-256 | `18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45` |
| Existing control body fingerprint; weights not reloaded here | `5d631be30158b3ea2a72cae355ce09a3b304bbfddde48b5758c339c346a34974` |
| W8A8 composition digest | `bd91e1659773faabcbb998a350893cbbc6df313a830da7cb7bfb87463c8f0b36` |
| 39+8 plan | `d292d42933e35239655e2a2d09e7bb8ac3c0d29ceb3b96ddd5512836148c82b1` |
| 39+8 schedule | `19d45c42975b80997c85b9c7bf4d594905e485588e7d8dce13b39de3bf478327` |
| 39+32 plan | `62381dbd11470bd150723cfb0d7bab096c52f331ec8e9f5dd23e178df28a0a4d` |
| 39+32 schedule | `bf3bc2e91b6e95a8d0b9ac7eebfb2866a3b3b8c2fe4c2ff401dc625f130a0d5c` |

The script rejects mismatch against
`docs/evidence/latent-response-network-qwen25-2026-09-28.json`. Those locks also
agree with `compiler-region-contract-qwen25-2026-09-30.json`. Existing unrelated
worktree edits are retained. Root `NETWORK_IO_10X_RESEARCH.md` was read and is
not changed by this track.

## Compiler-derived semantic coverage and counts

The probe enumerates `plan.to_dict()[phase]["operations"]`, declared `operator`,
`layer`, `inputs`, and `output_shape`; it does **not** parse operation IDs or
model weight/stage names to infer semantics. It verifies exact once-only
operation coverage by the runtime schedule, uses `scheduled_stage_specs`, and
calls the existing public `compiler_region_contract_cost` and
`token_boundary_he_layer_gate` validators for complete lineage and causal
attention→MLP dependency coverage. The HE validator is used only for its
semantic/dependency audit; no HE context or ciphertext is created.

Each of 24 layers has seven linear operators, two RMSNorms, two rotary
operators, two KV appends, two cache views, one QK score, attention scale,
causal mask, softmax and value accumulation, two residual adds, one SiLU and
one gate×up product. Four reshape operators per layer complete the layouts.
Per phase: 168 body linears, 48 body norms, 48 rotary, 48 KV append, 48 cache
views, 96 reshape, 24 each attention/SiLU/multiply, and 48 residual operations.
Outside layers: token lookup, final RMSNorm, last-token extraction, output
head, greedy selection and token feedback, once each.

Compiler dimensions: hidden 896; 14 query heads; head dimension 64; 2 KV heads
(GQA group 7); intermediate 4,864. Executed rows are `39 + generated - 1`.

| Response quantity | 39+8 | 39+32 |
| --- | ---: | ---: |
| Executed body rows | 46 | 70 |
| Body norm rows / residual-add rows | 2,208 / 2,208 | 3,360 / 3,360 |
| Attention query-head normalizers | 15,456 | 23,520 |
| SiLU elements / gate×up elements | 5,369,856 each | 8,171,520 each |
| Residual-add elements | 1,978,368 | 3,010,560 |
| Body quantized stage-row input boundaries, including fused QKV and gate/up | 4,416 | 6,720 |
| Visible causal query-key pairs over heads/layers | 363,216 | 834,960 |
| Dense runtime score/value pairs, including masked prefill positions | 612,192 | 1,083,936 |
| QK MACs / probability×V MACs | 39,180,288 each | 69,371,904 each |
| Body public projection integer MACs per resident worker | 16,460,021,760 | 25,047,859,200 |
| Capacity-sized 32-bit KV share storage per worker | 1,130,496 B | 1,720,320 B |

Visible pairs use `24×14×(39×40/2 + sum(40..last_executed_position_count))`;
dense prefill instead uses `39²`. Decode uses actual valid-prefix lengths,
not the maximum-capacity shape for every step. MACs are arithmetic counts, not
measured CPU, latency or a complete protected-operator count. KV storage is a
representation estimate, not encrypted size or traffic.

The existing non-affine polynomial schedule has a **96-product serial floor**:
QK → probability×V → non-affine SiLU → gate×up, across 24 layers. This is the
existing validator's optimistic dependency count, not a universal HE/MPC lower
bound. A rational SiLU numerator requires more than the one counted product;
norm/exp/inverse, carries, comparisons and exact float emulation add work.

## Algebra inventory: what can be carried, where it breaks

Let `x=n/d`, with positive secret `d`. Vector numerator and denominator shares
must have explicit ranges/precision. A homogeneous representation describes
the *represented value*, not an authorization to skip rounding nodes.

### Affine maps, residuals, and heterogeneous denominators

Over reals, `W(n/d)+b = (Wn+bd)/d`. Public weights/biases do not require secret
products to form this expression if represented exactly in a compatible domain.
The actual W8A8 linear path first computes dynamic signed-i8 activation codes,
then an exact integer dot, converts the accumulator to float32 and computes
`float32(float32(accumulator×activation_scale)×weight_scale)`, followed by bias
addition. Changing the order or skipping an intermediate scale is not exact.

Residual addition is
`n1/d1+n2/d2=(n1*d2+n2*d1)/(d1*d2)`: two secret vector/scalar products plus
one secret scalar denominator product. A public unit denominator saves one
broadcast; equal denominators save both, but equality/common-source provenance
must be proved. Forcing both branches to share a new denominator requires
scaling one branch, so does not make alignment free. Final float32 addition
also rounds, unlike carrying an unrounded rational sum.

After multi-head attention each head has a different normalizer `l_h`.
Concatenating heads preserves **14 denominators per query row**, not one.
An output projection mixes them. A common denominator can be
`D=product_h l_h`, with complement products `D/l_h` computed without division
using product trees/prefix-suffix products; then each head numerator must be
multiplied by its secret complement. This generally introduces 896 feature
alignment products per row before projection, or 14×896 products per row if
separately projected contributions are aligned afterward. Product `D` has
degree 14 in normalizers and tree depth at least 4 using binary products;
complement construction, alignment, rounding and norm increase depth further.
This is one layout choice, not a protocol lower bound.

### RMSNorm: epsilon, sign, radical, and dynamic quantization

For positive `d`, exact real algebra gives

```text
RMSNorm(n/d) = gamma*n / sqrt(mean(n*n) + epsilon*d*d).
```

The denominator does **not** cancel when epsilon is nonzero. For negative `d`
there is an additional sign; zero is invalid. The new denominator is an
algebraic radical, not generally a rational. Replacing it with a rational
approximation changes the numeric model. Keeping a formal radical does not
permit existing finite-ring multiplication/division to evaluate it for free.
Squaring a general rational denominator doubles its algebraic degree; squaring
an already stored radical instead recovers its radicand. Adding epsilon prevents
the scale cancellation commonly asserted for epsilon zero.

**Non-obvious legal representation:** carry `x=n/sqrt(s)` with positive secret
`s`, rather than materializing `d=sqrt(s)`. Ideal RMSNorm then becomes

```text
n_next = gamma*n
s_next = mean(n*n) + epsilon*s
x_next = n_next/sqrt(s_next).
```

Epsilon is retained, and no new square root is needed merely to update `s`.
This avoids a nested radical along chains of zero-bias public linears and norms;
the radical can be deferred to a real consumer of absolute values. The probe
independently compares squared normalized components via rational variances on
85 cases, with nonzero epsilon. Squared values plus known numerator signs prove
the ideal normalized outputs coincide. This is a stronger identity than dropping
epsilon, but still not an exact rounded-runtime optimization. An affine bias
adds `b*sqrt(s)` to the numerator; residuals between different radicands need
radical cross terms; QK/exp and SiLU need absolute magnitudes. These are the
compiler-declared blockers to carrying this representation through a whole layer.

An attractive narrower identity is dynamic quantization. In ideal arithmetic,
for nonzero numerator row and positive common `d`,

```text
q_i = round_even(127*n_i/max_j(abs(n_j)))
activation_scale = max_j(abs(n_j))/(127*d).
```

Hence codes of `n/d` can be formed before division, while the absolute row scale
still needs `d`. Similarly ideal RMSNorm codes depend on `gamma*n` but the
absolute scale depends on its radical denominator. Zero rows need the baseline's
scale-1 special case. Denominators must be common across the *whole quantized
row*, not only inside each attention head. A signed scale needs sign correction.

This identity survives mathematically with epsilon retained, but **fails as
an unconditional float32 W8A8 rewrite**: division, learned-scale multiplication,
max/127, and division by that rounded scale have separate rounded edges. It also
does not erase the absolute scale before SiLU, QK, bias or residual paths.
Moving normalization to projection outputs can require scaling 1,152 QKV
outputs or 9,728 gate/up outputs rather than the original 896 hidden inputs.
Dense MACs remain remote, but broadcast/rounding costs can get larger.

### Softmax numerator, normalizer, stable composition, and shifts

For each visible query/head, with `m=max(s)`, define

```text
l = sum_t exp(s_t-m)
n = sum_t exp(s_t-m)*v_t
attention = n/l.
```

For disjoint summaries `(m_a,l_a,n_a)`, `(m_b,l_b,n_b)`:

```text
m = max(m_a,m_b)
a = exp(m_a-m); b = exp(m_b-m)
l = a*l_a+b*l_b
n = a*n_a+b*n_b.
```

Independent proof: each term equals `exp(s_t-m)` times its scalar/value, so
merged sums match the union. Therefore composition is associative and
commutative over exact real arithmetic. Masked terms contribute zero; an empty
block needs a tagged identity and must not evaluate `-infinity-(-infinity)`.
A fully masked query remains invalid. The probe independently verifies exact
composition with rational powers of two, and float64 exponential composition
against directly summed weights. These are algebra/numeric checks, not a secure
protocol proof or exact float32 associativity.

Stable range: for a nonempty block of length `T`, `1≤l≤T`, each exponential
lies in `[0,1]`, and `|n_i|≤l*max_t|v_ti|`. With `T≤70`, one head normalizer is
small; however a 14-head common product may be as large as `70^14` (about
86 integer magnitude bits), before fixed-point fractional bits. Unshifted
float32 exponentials overflow for sufficiently large positive scores and
underflow on large negatives. Public offset selection requires a score bound;
secret max/exp/select and secret rescaling of summaries still need protection.

`softmax(s+c*1)=softmax(s)` is an exact additive-shift identity. If the same
mask is used, it can absorb a truly common scalar bias before exp. It cannot
absorb **multiplicative** query/key denominator factors: those change attention
temperature. A row-constant relative-key shift could remove a query-dependent
offset over reals, but creates its own secret centering products and changes
rounded dot/subtraction edges. Baseline already subtracts its maximum.

Delaying division from probabilities to `n/l` removes `T` quotient evaluations
in favor of a value-width quotient, but a protected implementation normally
shares one reciprocal of `l` across probabilities anyway. Compare complete
circuits, not `T` fictitious independent inverse calls. The float32 baseline
rounds *each probability before* its value product and sum; final `n/l` is
different. A protected stable-summary protocol must either preserve those nodes
or be declared approximate and evaluated on held-out checkpoint inference.

### SiLU, rational approximations, and down-projection denominator explosion

Exact `SiLU(x)=x/(1+exp(-x))` is not homogeneous:
`SiLU(n/d)=n/[d*(1+exp(-n/d))]`. Even a quadratic changes the powers of `d`;
dividing a polynomial numerator by the original `d` is generally wrong.
The gate×up product adds another denominator. Absolute gate input scale cannot
be dropped; `SiLU(c*x)` is not `c*SiLU(x)`.

The probe uses a no-training illustrative Padé exponential approximation:

```text
exp(x) approximately (x*x+6*x+12)/(x*x-6*x+12)
SiLU(x) approximately x*(x*x+6*x+12)/(2*x*x+24)
SiLU(n/d) approximately
  (n^3+6*n^2*d+12*n*d^2)/(2*n^2*d+24*d^3).
```

The approximation has degree 3/2; homogenized numerator/denominator degrees
are 3/3. The explicit denominator is positive (at least 24 for scalar `x`),
but tends to the wrong large-|x| behavior (`x/2+3/2` instead of the SiLU tails).
A numerator cubic requires at least two sequential binary multiplication
levels; forming denominator products, gate×up and branch alignment adds levels.
For common up denominator `d`, gate×up is degree 4/4. These are expression
counts, not prices for inverses or secure comparisons.

Each of 4,864 gate features now has its own denominator. Down projection sums
4,864 different fractions. A common product has degree 4,864 in those
denominators (binary tree depth at least 13), plus expensive complement
products and feature rescaling. On `x∈[-4,4]`, scalar denominators are in
`[24,56]`; their raw product alone requires more than 22,000 magnitude bits,
unless a public common rescaling is introduced. At Q7 `x=q/128`, the un-reduced
integer denominator `2*q*q*128+24*128^3` is already at least 26 bits; naively
forming 4,864 such factors exceeds 121,000 magnitude bits. Public rescaling
only moves the precision problem; secret GCD cancellation is neither free nor
guaranteed. Existing float32 learned weight scales and per-feature bias make
simple common-factor assertions even less credible.

Keeping featurewise fractions until down projection avoids that giant product
only by paying for private per-feature normalization (or a different specialized
sum-of-ratios protocol). Rational approximation may help an admitted approximate
protected model; it does not confer fixed W8A8 checkpoint parity.

### KV, rotary, and output boundary

- A linear RoPE over reals carries the same denominator. Baseline coefficient
  generation and float32 product/add boundaries still have to be reproduced;
  homogeneity alone does not preserve the rounded rotated keys/queries.
- KV denominator belongs to its layer/token/head or feature. It cannot be
  silently replaced by the newest query's denominator. QK uses denominators
  `d_query*d_key_token`; these are usually token-dependent and enter inside exp.
  Value denominators add further heterogeneous fractions to the numerator sum.
  A resident implementation needs capacity-bound protected numerator and
  denominator state, share renewal and consistent prefill/decode scales.
- Holding one hypothetical 32-bit normalizer share per attention query/head
  costs only 123,648 / 188,160 B across both workers over the response. This is
  hypothetical scalar storage, **not** network bodies, privacy or arithmetic
  admission. Secret broadcast products dwarf that metadata.
- Final positive common scalar would not change an ideal real argmax, but
  learned per-feature final norm, rounded activation codes/scales and head
  accumulation can change ties/order. Baseline already selects from logits
  without vocabulary softmax; no extra softmax can be credited as eliminated.
  Client final reconstruction/feedback remains once per generated token.

## Bounded measured results and counterexamples

The scalar baseline operators call actual `SemanticDecoderRuntime._local`
for RMSNorm, stable SiLU, softmax and grouped attention-values; quantization
uses actual `quantize_activation_per_row(bits=8)`, including installed native
dispatch. No baseline function is replaced at runtime.

| Probe | Observed result |
| --- | --- |
| Rational clipped ties-even via cross-comparison vs exact `Fraction` oracle | 8,811 cases passed in script; tests independently check 20,485 cases |
| Homogeneous Padé/SiLU and residual identities | 85 numerator/denominator inputs passed |
| Squared-denominator RMSNorm update vs independent exact rational variance | 85 cases passed, epsilon `1e-6` retained |
| Exact base-two exponential summary union/order checks | Passed; direct weighted value `23/49` |
| Float64 stable summary vs direct weights, 256 eight-key/four-feature examples | max absolute error `4.440892098500626e-16` |
| Delayed attention division vs runtime probability-before-value float32 | 251/256 rows bitwise different |
| Q7 half-threshold attention stress, scale `1/128`, signed codes clipped to ±127 | 66/256 rows changed a code |
| Dynamic W8A8 numerator/common-denominator cancellation, 127 threshold rows each | denominators 3 / 10 / 0.001 changed 9 / 15 / 19 rows; denominator 128 changed 0 |
| Nonzero-epsilon invalid RMSNorm denominator cancellation | max absolute error `0.19349411129951477` |
| Illustrative rational SiLU, 2,049 points on [-8,8] | max absolute error `1.4710016250610352`; 837 fixed-Q7 elements changed |

These mismatch rates are synthetic adversarial/threshold diagnostics, **not**
rates on real Qwen activations. Q7 denotes public fixed scale `1/128` here;
it is separate from the baseline's dynamic W8A8 per-row scaling. Zero mismatches
for the tested power-of-two denominator is not a universal proof (underflow,
overflow, max/scale rounding and additional operators still need bounds).

Concrete dynamic-W8A8 counterexample: float32 numerator
`[1, 0.035433072596788406]` quantizes to `[127,4]`. Dividing the row by 3 first
and then quantizing produces `[127,5]`, despite positive common scaling.

Concrete attention counterexample: the probe's first threshold case has rounded
output feature `0.00390625` when probabilities are divided before accumulation,
versus `0.003906250465661287` with final division. At scale `1/128`, ties-to-even
returns code 0 versus 1. Full synthetic scores/outputs are printed by the script.

With numerator `[1e-4,-2e-4,3e-4]`, denominator 3 and epsilon `1e-6`, baseline
RMSNorm is approximately `[0.03324725,-0.06649449,0.09974175]`; dropping `d`
without updating epsilon gives `[0.09774528,-0.19549055,0.29323587]`.
The legal real identity with `epsilon*d*d` is close but itself differs in
float32 low bits (`0.0332472436` versus `0.0332472473` in the first feature).

Softmax shift counterexample: `[0,1,2]+float32(2^25)` becomes three identical
float32 scores. Probabilities change from approximately `[.0900,.2447,.6652]`
to `[1/3,1/3,1/3]`. This falsifies unconditional finite-precision shift
invariance; it does not invalidate exact algebraic shift invariance.

## Comparison-based exact rational quantization: surviving legal rewrite

For integer numerator `N`, positive denominator `D`, public positive scale `s`,
rounding `N/(D*s)` can use comparisons with thresholds instead of division:

```text
2*abs(N) > (2*k+1)*D*s,
with equality resolved by parity of k (ties to even), then restore sign.
```

The bounded reference uses eight fixed binary-search iterations for clipped
codes ±127 and handles ties, sign and saturation. All denominators, threshold
products and comparisons must be protected; Python control flow in the probe
is **not** a private implementation. A fixed-size Boolean/arithmetic selection
schedule can hide branch results. At arbitrary secret scale, multiplication by
`s` adds another secret operand and range requirement.

For an ideal norm quotient `a/sqrt(R)` and public fixed output scale `s`, a
nonnegative threshold comparison can instead compare
`4*a*a` with `(2*k+1)^2*s*s*R`, preserving epsilon inside `R`. This avoids an
explicit sqrt only for the ideal rounded quotient, not the actual rounded
float32 sequence. Signs, ties, zero/range handling and all products need their
own numeric and protected contracts. That squared-threshold extension is an
analytic candidate, not implemented or measured by this probe.

**Exact-checkpoint gate:** translate the actual rounded arithmetic trace into
predicates, or prove a public-domain interval certificate showing its result
cannot cross a quantization threshold. Exact rational semantics alone do not
meet this gate. Private-input-dependent fallback/message lengths would leak
and require a separately admitted fixed-size protocol.

## Directed costs and crossings

Matched existing covered-body controls, not full wire:

| Workload | Control all-link / online B | Strict-tenfold all-link / online budget B |
| --- | ---: | ---: |
| 39+8 | 116,843,966 / 73,831,744 | 11,684,396 / 7,383,174 |
| 39+32 | 178,970,558 / 113,545,024 | 17,897,055 / 11,354,502 |

Current prepared schedule still has 96 remote body stages; a projective tuple
alone eliminates **zero demonstrated stage messages or semantic crossings**.
Quantized inputs, returned integer accumulators, absolute scales, nonlinear
outputs, KV updates and residual edges remain part of the trace. A new resident
protocol can change placement, but needs complete protected implementations.

Using the existing `BroadcastMaterial` distribution as a comparator, one
width-896 vector×secret-scalar costs `2*(896+1)*4 = 7,176 B` on the two peer
directions, and `2*(2*896+1)*4 = 14,344 B` dealer→both. The 32-bit modular
product is a distribution/body comparator, not proof that 32 bits suffice for
real projective values or exact float emulation. For two independently scaled
residual branches, charge two such broadcasts per residual row:

| Specified construction | 39+8 | 39+32 |
| --- | ---: | ---: |
| Secret broadcast products | 4,416 | 6,720 |
| Worker A→B alignment openings | 15,844,608 B | 24,111,360 B |
| Worker B→A alignment openings | 15,844,608 B | 24,111,360 B |
| Total online alignment bodies | **31,689,216 B** | **48,222,720 B** |
| Dealer→A alignment material | 31,671,552 B | 48,195,840 B |
| Dealer→B alignment material | 31,671,552 B | 48,195,840 B |
| Alignment-only all-link online+dealer | **95,032,320 B** | **144,614,400 B** |

Denominator product triples, widening, signed carries, exact rescaling,
sqrt/inverse/exp, comparisons, masks, correlations for gate/value products,
KV share upkeep and full wire are excluded **unpriced obligations**, not zero
cost assumptions. Even the favorable one-public-denominator special case halves
the online figures to 15,844,608 / 24,111,360 B, still over both budgets.
Equal-denominator provenance could eliminate these particular alignment
products; its other arithmetic costs would still need a complete accounting.

The existing hypothetical resident 12-bit **two-source** placement costs
6,239,744 / 9,637,376 B known online, including token boundaries, leaving
1,143,430 / 1,717,126 B online and 5,444,652 / 8,259,679 B all-link for
everything omitted. These small rings lack exact share-lift/carry admission.
If projective alignment retains that placement, its vector openings are
additional unless a joint protocol explicitly absorbs/replaces them. Do not
double-count common sources, but do not delete their openings by notation.

Secret denominator scalar storage or scalar wire fields look cheap; denominator
**broadcast multiplies and rounded conversions**, not tuple metadata, dominate.
No protected inverse/exp cost or denominator degree is hidden inside an
unpriced "homogeneous operator".

## Failure gates and strongest next direction

1. **Semantic coverage:** retain every compiler-declared norm, attention score,
   mask, softmax, value accumulation, projection, rotary/KV update, SiLU,
   residual, rounded quantization and token boundary. Unsupported/omitted work
   blocks admission. Current stage messages do not disappear from an identity.
2. **Numeric parity:** exact claims require equality of activation codes,
   integer accumulators, float32 rounded scales/edges and held-out prefill,
   same-token decode and free generation. Current unconditional cancellation
   candidates fail bounded counterexamples already. Approximate candidates
   must be named as such and pass fixed-checkpoint quality evaluation.
3. **Range/degree/depth:** positive/nonzero denominators, epsilon, stable exp,
   head/feature denominator heterogeneity, carry and scale precision must be
   bounded publicly. Reject denominator products that need unbounded widening,
   refresh or precision. Current Padé SiLU quality and featurewise denominator
   explosion fail this screen; no whole-decoder substitution is justified.
4. **Directed whole-layer cost:** price secret comparisons, exp/inverse/sqrt,
   broadcast products, share renewal and correlation expansion. Reject the
   tested independent-residual alignment on known bodies. Passing a scalar
   byte estimate or a sub-budget token boundary is not passing a layer.
5. **Protocol evidence:** only after the above, execute one complete layer
   using party-local shares, independent material and burn-on-cancel/replay;
   measure each link and client/worker compute, cold bytes, memory and wire.

**Recommended next bounded gate:** exact norm→dynamic-quantization predicate
fusion. Start from the runtime's real float32 operation order and tiny public
W8A8-domain examples, derive threshold predicates or uniform interval
certificates that include epsilon/learned scale/max/127 rounding, and falsify
them on adversarial ties and underflow cases. Then cost those predicates across
both independent hidden sources and all norm rows, fitting the remaining
1.14 / 1.72 MB online room if retaining the 12-bit source-opening comparator.
Failure to produce bit-exact rounded codes or a compact full-layer correlation
closes that candidate before loading giant weights or activating a topology.

Stable-summary attention is the strongest **approximate/numerically reassociated**
alternative in this track: bounded normalizer range and composability are real,
but retain masked private attention, max/exp and protected values, count merge
products and output-projection denominator alignment, and validate complete
checkpoint quality. It cannot yet claim either exact W8A8 or tenfold traffic.

## Primary sources and local contracts

Retrieved via `webfetch` on 2026-10-01:

- Milakov & Gimelshein, *Online normalizer calculation for softmax*,
  [arXiv:1805.02867v2](https://arxiv.org/html/1805.02867v2), Algorithms 2–3 and
  §3.1 equations 3–4: stable online normalizer and composable max/sum summaries.
  Claimed gains concern memory accesses on ordinary GPU execution.
- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with
  IO-Awareness*, [arXiv:2205.14135v2](https://arxiv.org/abs/2205.14135v2): tiled
  attention reduces GPU HBM/SRAM traffic. Its "exact attention" is not evidence
  of bitwise equality to this float32/W8A8 trace, secure sharing, or network gain.
- Zhang & Sennrich, *Root Mean Square Layer Normalization*,
  [arXiv:1910.07467](https://arxiv.org/abs/1910.07467): RMSNorm's rescaling idea.
  Nonzero-epsilon checkpoint behavior is governed by the local compiled/runtime
  contract, rather than inferred from an ideal invariance description.

Local inspected contracts: `NETWORK_IO_10X_RESEARCH.md`,
`python/pllm/runtime/semantic_executor.py` (actual scalar float32 operators),
`semantic_numeric.py` (distinct BF16 boundaries, not substituted for Qwen W8A8),
`semantic_attention.py`, `semantic_stages.py`, `quantization.py`,
`region_contract_cost.py`, `he_layer_feasibility.py`, and
`interaction_reduction_reference.py` (broadcast-body comparator).

The report makes no literature novelty claim and no measured protected-network
improvement claim. Scope verification passed: probe completed both pinned
cohorts; **4 scoped tests passed**; scoped Ruff lint and format checks passed.
