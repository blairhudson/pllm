# Design notes: resident private inference after the projection-masking architecture

## 1. Scope, security and accounting

The architecture uses two non-colluding inference workers with public weights and additive shares of private state. The Client supplies inputs and reconstructs outputs, not intermediate activations. A trusted, non-colluding dealer supplies input-independent correlations and never sees their online openings. Public weights make matrix multiplication by those weights local; private/private attention products, nonlinear operations, rescaling and selection are not free.

The code is a single-process reference, so its driver necessarily sees both shares to check correctness. That is not deployment isolation. Correctness tests are not security tests. Protocol claims are conditional on standard semi-honest secret-sharing/Beaver/FSS assumptions, fresh randomness, fixed public shapes, protected channels, and correct composition. Malicious preprocessing, authenticated shares, model authentication, rollback/cloning, and implementation side channels are not solved here. A production migration cannot discard the previous paper's integrity requirements.

`core.py` counts ring words as four-byte values in Z_(2^32), even though the NumPy backing arrays use uint64. Boolean messages are counted with byte-padded bit packing. Online bytes sum both directions; offline bytes sum the explicit shares delivered to both workers. Counts omit transport, identifiers, authentication and initial input sharing. Round counts are logical dependency rounds, not wall time. The fused cache schedule is modeled explicitly. No packet compression, overlap, network bandwidth, or GPU performance is inferred from these counts.

The separate spectral gate uses Z_(2^64). Its byte counts cannot be compared with 32-bit gates as though their functionality and numerical representation were identical.

## 2. Candidate A: lifetime-aware correlations for immutable KV operands

### Construction

For a fixed shared matrix K, the dealer supplies a uniform mask B and its shares. The workers open E = K - B once. For each new query q, the dealer supplies a fresh independent A and shares of C = A B. The workers open D = q - A, then locally form:

    [q K] = [C] + D [B] + [A] E + D E.

Only one worker adds the public D E term. Reconstruction gives (A+D)(B+E)=qK. For scores, use K transposed. The same mechanism applies to a fixed V cache and fresh attention-weight vectors.

**Reuse is permitted only for the exact same immutable operand.** This is not reuse of a pad for changing activations. A cache append gets fresh row identities and masks. Replacing, requantizing or moving a value into an existing logical identity requires retiring that identity and using a fresh one. Restore/clone protection is a separate runtime requirement.

The dealer needs B and fresh A to prepare A B, but no K, model matrix or query. Fresh left masks and their products are still required for every multiplication. This work has moved, not disappeared.

### Avoid the extra round

A naive implementation opens each new K/V row separately and loses the synchronization advantage. The reference instead reserves each row before exposure and batches both new masked rows with the next query-mask opening. The masked V row is ready before the later weighted-value product. Pending rows cannot be rebound.

### Executed comparison

Eight synthetic caches start with 32 rows and append 24 rows, with width 16. Both elementary full-matrix Beaver multiplication and the cached version produce the exact modular answer for all 384 products. All 192 attempted row rebindings are rejected.

| Counter | Full fresh matrix triples | Cached immutable operand |
|---|---:|---:|
| Online payload, both directions | 2,280,192 B | 207,616 B |
| Explicit offline shares | 2,373,120 B | 300,544 B |
| Logical online rounds | 384 | 384 |

The online reduction is **90.895%** for these tested products and initial-cache accounting. It is not a whole-transformer traffic reduction. A small exhaustive experiment confirms only the uniform marginal distribution of public masked openings; it is not a complete simulation proof. A negative control explicitly recovers the difference between changed operands when they are masked with the same B.

### Longer-context analytical budget

Using the official Qwen2.5-0.5B configuration (24 layers, 14 query heads, 2 KV heads, head width 64) [5], the reference formula at context 4,096 gives:

- 212,508,672 bytes for the elementary fresh-matrix opening baseline.
- 11,231,232 bytes/step for cached steady-state openings, including new cache rows.
- 201,326,592 bytes of initial cache openings.
- 22,364,160 bytes/step of fresh, explicitly delivered A/C shares, excluding recurring B storage and every other operator.

These are **formulas**, not Qwen executions. The baseline already amortizes the opened KV operand across grouped-query heads. At an ideal aggregate serialization rate of 100 Gbit/s, 11.23 MB alone represents 0.8985 ms. This excludes synchronization, exponentiation, normalization, truncation, integrity and computation. At 32,768 context the same steady-state formula is 88.30 MB, or 7.06 ms under that illustrative rate.

CryptoGen already studies encrypted KV-cache reuse in a different HE/MPC architecture [1]; cache persistence is not novel. The candidate contribution here is immutable-operand correlation lifetime plus append/opening fusion and measured total-cost accounting. Priority for this precise composition has not been established.

## 3. Candidate B: postpone attention normalization

For exponentials e, sum z, and value matrix V:

    (e / z) V = (e V) / z.

Compute the unnormalized numerator and reciprocal concurrently where the graph allows, then multiply the reciprocal into the d output coordinates rather than L attention probabilities. At L=4,096 and d=64, this reduces the normalization branch from 4,096 secret multiplications to 64. It does not eliminate the L-dimensional exponentiation or either attention product.

All 60 float64 test cases agree to maximum absolute difference 4.663e-15. Another 100 finite-field cases agree exactly. With explicit fixed-point rounding, all 60 tested outputs change: moving rounding is not a bit-preserving rewrite. In the selected 4,096-context fixtures the mean RMSE versus floating reference drops from 0.0025551 to 0.0001431, but these synthetic error numbers do not predict model quality.

The unnormalized accumulator needs an explicit range/precision policy. The reciprocal is still private. Delayed normalization is established attention algebra and appears in FlashAttention-2 [2]. The implementation contribution would be its secure scheduling and numeric specification, not the identity.

## 4. Candidate C: fused polynomial and piecewise gates

### One masked-input opening

For F(x,u)=u P(x), write x=d+r and u=e+s with fresh full-ring masks. The dealer expands:

    P(d+r) = sum_j a_j(r) d^j.

It shares both a_j(r) and s a_j(r). After opening d and e together, each worker evaluates:

    sum_j (e [a_j] + [s a_j]) d^j.

No secret online Horner multiplications or separate secret gating multiply are needed for this **specified integer polynomial**. Four hundred random full-ring polynomial cases pass at degrees 2, 3, 5 and 7. The dealer uses public gate coefficients and masks, not model weights.

Piecewise polynomials additionally need private interval-dependent coefficient selection. The code implements this, first using a dense table and then actual SHAKE-based DPF/DCF keys. The compact implementation translates the signed intervals, incorporates wrap-dependent polynomial offsets, pads the number of boundaries, and FSS-shares the coefficient step function. The shared-tree DCF adds output corrections at each prefix. These are standard constructions [4], not oracle calls or new DPFs.

### Measured key payloads

Four intervals, degree three, 32-bit result:

| Input bits | Dense table, both shares | Independent-prefix DCF keys | Shared-tree DCF keys | Online masked inputs, both directions |
|---|---:|---:|---:|---:|
| 8 | 16,384 B | 7,560 B | 3,432 B | 10 B |
| 16 | 4,194,304 B | 23,656 B | 6,568 B | 12 B |
| 24 | 1,073,741,824 B | 48,456 B | 9,704 B | 14 B |
| 32 | 274,877,906,944 B | 81,960 B | 12,840 B | 16 B |

Dense wider-domain sizes are calculated, not allocated. Compact sizes are actual serialized payloads. There are 5,461 DPF queries, 1,364 independent-prefix DCF queries, 5,460 shared-tree DCF queries, and 1,264 fused gate queries, all correct. Exhaustively evaluating one key over many inputs is a functional test only: deployment must not reuse the mask pair across distinct private activations.

FuseFSS explicitly describes private polynomial Horner multiplication after coefficient selection [3]. This experiment evaluates shifted coefficients on public masked coordinates instead. However, offset-function FSS and secret coefficients of public basis functions are established [4]. A broad novelty claim would be incorrect.

### Why this is still not a complete SiLU implementation

The exact polynomial tests use integer fixtures, not a trained model's activation approximation. Fixed-point truncation is not automatically included in the one-round claim. A single global polynomial fit is also unsafe outside its range: the degree-nine fit on [-8,8] has about 0.072 maximum error there and enormous errors at the tested tails. Piecewise approximation, authenticated public ranges and quality validation remain necessary.

Even 6,568 bytes per scalar is substantial expendable material. At 24 layers and intermediate width 4,864, 116,736 such records would mean approximately 766.7 MB per generated token for this branch alone. This is an illustrative dimension-based sum, not an implemented Qwen gate graph or throughput result.

## 5. Candidate D: boundary-matched polynomial-plus-Fourier gate

This additional experiment addresses that key-size problem without pretending that approximation is exact inference.

### Representation

SiLU has the identity:

    SiLU(x) = x/2 + g(x),     g(x) = (x/2) tanh(x/2).

The even function g has matching endpoint values on [-R,R], but its periodic extension has a derivative jump. Choose:

    a = g'(R)/(2R),
    h(x) = g(x) - a x^2.

Now h is even and its endpoint first derivatives are zero. Approximate h by a short cosine series. The quadratic term removes the dominant boundary cusp; it is not cosmetic. In the initial float grid at R=8, eight cosine modes without the quadratic subtraction have about 0.0957 maximum error, versus about 5.09e-5 after subtraction.

Thus:

    SiLU(x) approximately x/2 + a x^2 + c0 + sum_j c_j cos(j*pi*x/R).

Translation of each cosine is closed in a sine/cosine pair. With x=d+r:

    c_j cos(omega_j*(d+r))
      = (c_j cos(omega_j*r)) cos(omega_j*d)
        - (c_j sin(omega_j*r)) sin(omega_j*d).

The dealer can share the shifted polynomial and trigonometric coefficients. The workers evaluate only public powers, sine and cosine of the **masked** coordinate. Sharing the coefficients multiplied by an additional mask s fuses multiplication by private u exactly as in Section 4. There is no private interval lookup.

### Real masking, not an orthogonal-transform privacy heuristic

The implementation opens d=x-r and e=u-s modulo 2^64 with fresh uniform masks. It does not disclose real-valued x plus a small mask. The integer trigonometric period is 16,384, which divides 2^64; reducing the public d modulo that period is compatible with modular addition. Each party's coefficient and mask shares are independently randomized additive shares. Conditional one-party privacy follows the usual fresh full-mask and hidden-coefficient argument; no production proof or cryptographic audit is claimed.

Public sine/cosine values and secret translated coefficients are fixed-point integers. Their multiplication and masking are exact ring operations. Their approximation to real trigonometric values introduces a small **mask-dependent numerical error**. The result is not bit-identical to a deterministic SiLU lookup table.

### Executed results

Input scale: 2^10. Intermediate SiLU numerator scale: 2^44. Trigonometric coefficient/basis scales: 2^22 each. Declared |x|,|u| <= 8. Mask and arithmetic ring: 64 bits. The gate returns an unrescaled integer numerator; secure final rescaling remains additional work.

For each profile, the numerical sweep covers every one of 16,384 quantized x values at 32 mask phases: 524,288 evaluations. Separately, 256 **fresh-record two-party** fused gates are checked against an independent integer reference.

| Cosine modes | Fresh record bytes, both workers | Maximum SiLU error on tested grid/phases | Maximum observed fused-gate error in fresh-record cases |
|---|---:|---:|---:|
| 4 | 384 | 0.0072750 | 0.0550375 |
| 8 | 640 | 5.128e-5 | 3.694e-4 |
| 12 | 896 | 5.394e-6 | 2.866e-5 |
| 16 | 1,152 | 4.183e-6 | 2.133e-5 |

All 1,024 fresh-record modular reconstructions are exact for their generated integer coefficients. All profiles use one joint opening of two 64-bit inputs: 32 bytes total across both directions. No secret output is opened to either worker.

The code separately derives a uniform-in-mask coefficient/basis rounding bound under ideal real trigonometric evaluation, and combines it with the exhaustive discrete-input approximation error. For the 12-mode profile this gives a fused-gate bound of about 5.39e-5 for |u|<=8, excluding uncertified numerical library error. It is not a formally machine-checked interval bound.

The 12-mode explicit record sum at 116,736 gates is 104,595,456 bytes, or **99.75 MiB per token**, before other gates. That is smaller than the particular piecewise-key fixture, but not a matched accuracy/security/performance comparison: the fixtures use different rings and functions. It is still a significant preprocessing rate. Local Python evaluation is not near-native inference; vectorized/GPU implementations would need actual measurement.

### Boundaries and novelty

At x=32 the approximation returns about 48.15 rather than 32. The declared range cannot be extended casually. It must be proved from the graph, enforced privately, or explicitly accepted as an approximation constraint. A data-dependent plaintext fallback is not a privacy-preserving repair.

Public-basis coefficient sharing and offset preprocessing are already FSS techniques [4]. Fourier-based secure computation also has prior art [8], and complex-activation approximation is studied in Compact [9]. The candidate to investigate is the **boundary-matched periodic decomposition plus translated-coefficient fusion for bounded nonlinear gates**, with a proof of its numeric envelope and cost. Priority for that combination has not been established by this search.

## 6. Division-free sampling and resident token feedback

Given nonnegative secret integer weights w_i, let S=sum(w_i) and C_i=sum_{j<=i} w_j. Draw an independent public U uniformly on {0,...,2^b-1}. Compute all secret predicates:

    2^b C_i > U S.

Differencing their monotone Boolean indicators yields an arithmetic secret one-hot vector. Multiply it locally by the public embedding matrix. No normalized probability vector, opened token index at either worker, client selection, or client round trip is needed for this feedback path.

The implementation uses actual Boolean Beaver AND gates, a parallel-prefix arithmetic-to-Boolean conversion and Boolean-to-arithmetic conversion—not ideal comparison calls. All 128 integer-weight sampling/embedding cases pass. The 32-bit comparator implementation takes seven online rounds. Independent conversion tests cover 5,124 values.

The finite random grid approximates the categorical distribution of **integer weights**. Its total-variation error is at most (V-1)/2^b. The tested eight-bit grids deliberately expose noticeable discretization error; the maximum observed value is 0.0500. The illustrative bound at V=151,936 and b=48 is 5.40e-10, but that wider comparator was not executed and excludes exponent approximation and weight quantization.

Remaining costs: private exponentials, suitable wide comparisons, unbiased public randomness, sampling policies such as top-p/top-k, and a full embedding scan on each worker. The dense scan is a real memory/computation trade-off. The identity is ordinary inverse-CDF sampling; only the resident protocol composition is a design candidate.

## 7. Exact share kernels and rescaling

Uniform 32-bit shares cannot generally be multiplied as ordinary FP32 activations without losing ring exactness. The reference splits shares into centered byte limbs, evaluates bounded 512-term dot-product chunks, and recombines them modulo the target ring. All 180 matvec cases pass across 16-, 24- and 32-bit rings. Naive FP32 conversion produces 5,390 wrong coordinates out of 5,760 tested coordinates.

This is not a GPU Tensor Core benchmark. It requires two, three or four limb passes per worker at the respective widths. GPU secure-computation engineering has substantial prior art, including Piranha [6]. Native low-bit model kernels cannot simply be assumed to have the same cost on wide random shares.

Local share truncation is also not exact. Exhaustive 8-bit-ring tests with a four-bit right shift fail in 30,720 of 65,536 share pairs (46.875%) if the low-bit carry is omitted. Adding the carry gives the exact narrowed-ring answer in every pair. A separate secure carry implementation passes 4,096 cases in four Boolean rounds. Widening, sign extension and preserving a larger output ring require additional specified protocols.

## 8. Rejected or model-changing shortcuts

**Warm-start normalization.** Reusing the previous variance's inverse square root as a Newton seed looks excellent on smooth synthetic sequences. Two iterations have maximum relative error about 4.3e-6 there. But a 64x scale jump generates wrong-sign and divergent iterates. Even a jump from variance 1 to 16 turns the first estimate into -6.5 instead of 0.25. A guard adds secure work; a visible adaptive fallback may leak.

**Recurrent quadratic attention.** Replacing exp(q.k) with (c+q.k)^2 gives a finite feature map and context-independent aggregate state. All 216 recurrence checks agree with the replacement kernel. It is not softmax: observed relative output errors range roughly 14–96%, and logits [-12,0] with c=5 reverse the correct ranking. MPC-friendly model modification/distillation is an established separate research route [7]. This candidate is not a drop-in optimization of public pretrained weights.

**Weak masks and rotations.** Orthogonal random rotations preserve norms, enabling perfect distinction in the explicit norm-1 versus norm-2 fixture. Uniform bounded additive masks on [-127,127] with secret shift 64 have total variation 64/255. A visible branch requesting more exact work can disclose a value-dependent event. These are not substitutes for full-ring masking or a stated alternative privacy definition.

## 9. Recommended implementation sequence and stopping criteria

First implement immutable cache lifetimes and append/opening fusion in the real runtime, with crash/retry tests and unchanged-output comparisons. Then specify late normalization and its accumulator/rounding rules. Evaluate the spectral gate against a strong piecewise-FSS baseline on real activation traces and full-model quality, including tails, rather than just comparing scalar packets. Keep the output selection and embedding entirely within the resident worker loop.

The next combined benchmark should include attention, normalization, activation/gating, all rescaling, sampling, fresh preprocessing replenishment and authenticated execution. Match the native model/quantization, batch, context, output length and hardware. Report per-user output TPS, GPU-seconds/token, bytes by phase/direction, critical-path rounds, warm/cold latency, preprocessing rate and client footprint.

For an illustrative native baseline of 50 TPS, retaining 90% leaves only 2.22 ms/token of extra time. Even the 4,096-context cached attention openings consume about 0.90 ms at the hypothetical aggregate rate above, before all the missing work. No experiment here demonstrates fitting that budget.

The strongest possible paper is therefore a measured **lifetime- and basis-aware secure decode compiler/runtime**, not a claim to have invented Beaver triples, FSS, Fourier approximation or private sampling. The new spectral gate is a promising narrower hypothesis. Authenticated model binding and actively secure preprocessing still require their own design and costs; these experiments did not resolve them.

## Primary sources consulted

These sources inform comparisons, not the numerical results generated by this package.

1. H. Zhang et al., *CryptoGen: Secure Transformer Generation with Encrypted KV-Cache Reuse*, February 2026. https://arxiv.org/abs/2602.08798 — cache reuse in a different HE/MPC setting.
2. T. Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, 2023. https://arxiv.org/abs/2307.08691 — delayed normalization and attention scheduling.
3. Y. Ma, Y. Li, S. Schmid, *FuseFSS: Efficient Secure LLM Inference with Function Secret Sharing*, June 2026. https://arxiv.org/abs/2606.09551 — compiled private helpers, coefficient selection and Horner costs.
4. E. Boyle, N. Gilboa, Y. Ishai, P. Scholl, *Distributed Point Functions and Function Secret Sharing*, July 2026. https://arxiv.org/abs/2607.27696 — tree DPF/DCF, offset-function sharing, public-basis sharing and correlation generators. The prototype follows the described standard tree constructions.
5. Qwen, official Qwen2.5-0.5B-Instruct configuration. https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/raw/main/config.json — dimensions only; no model weights were used.
6. J. Watson et al., *Piranha: A GPU Platform for Secure Computation*, USENIX Security 2022. https://www.usenix.org/conference/usenixsecurity22/presentation/watson
7. D. Li et al., *MPCFormer: fast, performant and private Transformer inference with MPC*, 2022/2023. https://arxiv.org/abs/2211.01452 — model approximation and distillation as a different trade-off.
8. A. Sonnino, *FMPC: Secure Multiparty Computation from Fourier Series and Parseval's Identity*, 2019. https://arxiv.org/abs/1912.02583 — Fourier-based secure computation precedent, not the same gate construction.
9. M. Islam et al., *Compact: Approximating Complex Activation Functions for Secure Computation*, 2023/2024. https://arxiv.org/abs/2309.04664 — approximating SiLU/GELU and related nonlinearities for MPC.

Research snapshot: 13 September 2026. No first-in-literature or complete-SOTA coverage claim is made.
