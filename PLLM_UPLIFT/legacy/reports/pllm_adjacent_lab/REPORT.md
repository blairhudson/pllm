# PLLM: structured preprocessing, translation rank, and quantization-aware gates

**Research and executable reference experiments — 13 September 2026**

## Decision

The highest-upside literature lead is **pseudorandom correlation generation over Galois rings**: stop treating explicit delivery of every preprocessing share as an unavoidable communication cost. The strongest additional compiler hypothesis is **minimizing the translated function representation jointly with its numeric error and local evaluation cost**.

These are different claims. Published PCGs already address the first. This package does not implement one. The second produced a concrete scalar-gate improvement in the executed experiments, but is not a new general cryptographic primitive or a demonstration of full-model quality or near-native TPS.

“Overlooked” here means overlooked in our earlier PLLM exploration, not neglected by the research community.

The target remains two non-colluding resident inference workers holding public model weights and additive shares of the private state. The Client need not retain weights or intermediate activations. A preprocessing role needs public gate specifications and randomness, not model matrices. Public-weight matrix multiplication stays local to each worker. Attention products, nonlinear functions, rescaling, selection, integrity and replenishment still have to be implemented and paid for.

This is an extension of the resident-share research design, not a retroactive claim about the original paper's projection-masking implementation, which kept attention and nonlinear computation at the Client.

## 1. Galois rings and coding theory: reduce preprocessing communication at its source

### Published foundation

Li, Xing, Yao and Yuan's CRYPTO 2025 work constructs pseudorandom correlation generators (PCGs) over Galois rings, including integer rings modulo powers of two [1]. It uses Galois theory, Hensel lifting and coding-based assumptions to support OLE correlations and authenticated multiplication triples. The paper also discusses circuit-dependent preprocessing and matrix multiplication triples.

A PCG gives each party a different correlated seed, from which it locally expands its own share of many correlated values. It is not the same thing as handing both parties an ordinary shared PRG seed. The paper obtains sublinear communication with substantial local expansion work; concrete batch size and security parameters matter.

**Proposal:** use an established ring PCG backend for the ordinary multiplication correlations required by the resident graph, with a separate pool for each arithmetic domain and correlation type. Generate and expand ahead of consumption, but measure sustained replenishment alongside inference rather than hide it behind an initially full buffer.

Generic multiplication triples do not automatically generate the specialized shifted sine/cosine coefficients from our gate. Nor do generic independent triples automatically implement the repeated-right-operand correlation schedule for an immutable KV cache. Those correlations require a specified conversion or specialized generator and a composition argument.

### Why the arithmetic domain matters

The 2026 follow-on *Faster Pseudorandom Correlation Generators via Walsh-Hadamard Transform* replaces costly transform multiplications with additions/subtractions in large-prime-field PCGs [2]. That is a useful separate backend comparison, not an automatic optimization of the power-of-two-ring construction.

For a length-N Walsh matrix H, H^2 = N I. When N is a power of two, N is invertible in an odd-prime field but is not a unit modulo 2^k. Our length-eight example has a nonzero vector in H's kernel modulo 256. Consequently, the field construction cannot simply be relabelled as a ring construction.

### Security qualification

Practical cryptanalysis has broken concrete parameter sets of earlier quasi-abelian PCGs [3]. The retrieved ring-PCG paper explicitly updates parameters against that attack and labels some faster comparison rows as insecure and unsuitable for deployment. We have not independently certified any parameter set. Use a current, reviewed parameter selection and identify the exact assumption, rather than select the largest throughput number in a table.

PCG expansion trades network traffic for local arithmetic, memory and setup complexity. Its cost can contend with inference. Authenticated triples are not, by themselves, an authenticated model or a complete maliciously secure inference protocol.

### Executed adjacent-math check — not a PCG

`galois_ring_checks.py` implements the degree-two extension

    GR(2^k,2) = Z_(2^k)[w] / (w^2+w+1).

Writing an element as a+bw, the lifted automorphism and trace are

    sigma(a+bw) = (a-b)-bw,
    Tr(a+bw) = 2a-b.

A dual basis is

    beta0 = (1-w)/3,
    beta1 = (-1-2w)/3,

where 3 is invertible modulo 2^k. Then Tr(x beta0) and Tr(x beta1) recover the two base-ring coordinates. The program checks multiplicativity of the automorphism and this coordinate extraction on **73,536 pairs**, including an exhaustive 4-bit-base-ring case and sampled 8-, 16-, 32- and 64-bit cases. All pass.

The program also shows why “Frobenius equals squaring” cannot be copied naively from characteristic two into characteristic 2^k. These checks explain an algebraic ingredient of the literature; they neither construct correlated seeds nor test their security.

## 2. Levi-Civita's equation: optimize translation rank, not just polynomial degree

A masked public-basis gate has the form

    f(d+r) = sum_{j=1}^J a_j(r) b_j(d).

The masked opening d is public; each worker holds shares of a_j(r). J governs the amount of coefficient material and public-basis evaluation required. Multiplying by a second private input may require a second set of correlated coefficients.

This is a version of the Levi-Civita functional equation and finite-dimensional translation-invariant function spaces [4]. Under suitable regularity assumptions over the reals, finite-dimensional shift spaces lead to exponential-polynomial structure. That explains why polynomial and trigonometric bases fit this architecture particularly well.

This is **not** a general lower bound on MPC, FSS, HE, or all protocols with one round. It constrains this particular linear public-basis representation. Ordinary polynomial closure under real addition must also be distinguished from a signed lookup table under cyclic modular wraparound.

### Executed finite-domain diagnostic

Over F_65537, the 256-point discrete Fourier transform exists and is invertible. The cyclic translation matrix of a function is circulant, so its exact rank equals the number of nonzero Fourier coefficients. The implementation uses exact modular arithmetic, not a floating-point singular-value threshold.

| Function on the specified 256-element domain | Exact translation rank |
|---|---:|
| Constant | 1 |
| One finite-field character pair | 2 |
| Four character pairs plus a constant | 9 |
| The tested integer ReLU table | 256 |
| The tested quantized SiLU table | 256 |
| The tested floor-by-16 table | 241 |

The character-pair examples are finite-field functions, not claims that fixed-point real cosines satisfy exact finite-field identities.

**Compiler hypothesis:** select a function approximation whose translated representation has low dimension, while constraining approximation error, fixed-point rounding, range and local runtime. Optimizing unmasked approximation error alone is insufficient.

## 3. Krylov-Lanczos boundary correction plus minimax fitting

### Correcting the novelty framing

Our previous quadratic subtraction before Fourier approximation has classical precedent. Krylov-Lanczos methods subtract a polynomial to make endpoint derivatives match, accelerating the Fourier series. Bernoulli polynomials provide a systematic formulation; Tasche's 1991 paper treats this explicitly [5]. We should credit that numerical technique rather than imply that the boundary correction itself was invented here.

The potentially additional contribution is a private-gate compiler that uses this structure while optimizing translated coefficient count and fixed-point error.

### New approximation search

For the even part of SiLU,

    g(x) = (x/2) tanh(x/2),

we jointly fit the quadratic coefficient and the cosine coefficients:

    g(x) approximately a*x^2 + c0 + sum_j c_j*cos(2*pi*j*x/T).

The screen compares periods 16, 32 and 64 and 4, 6, 8, 10, 12 and 16 modes. It includes least-squares and discretized minimax linear-program fits. Fitting uses 4,097 points on [0,8]. Verification uses every point of the declared 16,384-point, scale-1024 signed input grid.

The linear program minimizes maximum error on the fitting grid, not a certified continuous minimax norm. Only successful solver results are retained. It is not a global search over all possible bases or parameter choices.

The larger-period Fourier extensions considered here often require large cancelling coefficients. For example, a tested period-32, ten-mode least-squares fit has a coefficient near 8.82e5 and an error around 2.23e-4. The selected period-16 ten-mode fit keeps its largest coefficient around 1.23. Fourier extension conditioning has a substantial literature, including methods that remain stable despite ill-conditioned coefficient recovery [6]. These experiments reject the tested parameter choices, not Fourier extensions generally.

### Seeded delivery baseline

Before invoking an elaborate PCG, use a simpler delivery optimization. For each intended vector v of correlated preprocessing values, the dealer gives Worker A a fresh seed and Worker B the vector

    v - PRG(seed) mod 2^64.

Worker A regenerates its additive share. The seed is for one party's random shares, **not** a seed revealing the complete input masks. Sharing the same complete-mask seed with both workers would be a privacy failure.

This is a standard seeded-share representation, not a cryptographic novelty claim. It removes one vector transmission, not both. The code uses SHAKE-256 with domain separation and fresh 32-byte seeds; identifiers, transport protection and authentication tags are outside the reported payloads. A production batch can amortize seed overhead further.

### Matched executed comparison

All profiles use full 64-bit modular masks, the same bounded inputs |x|,|u| <= 8, and one opening of the two masked inputs. The previous baseline is re-evaluated on the same 64-phase grid as the new profiles.

| Profile | Explicit two-share record | Seeded record | Maximum observed SiLU error on matched grid |
|---|---:|---:|---:|
| Previous 12-mode profile, 44 fraction bits | 896 B | 480 B | 5.098e-6 |
| New 10-mode profile, 46 fraction bits | 768 B | 416 B | 3.392e-6 |
| New 12-mode profile, 46 fraction bits | 896 B | 480 B | 1.313e-6 |

The overall 896-to-416-byte improvement is **53.57%**, but most is the seeded-share baseline. Comparing both with seeded delivery gives **480 to 416 bytes, or 13.33%** incremental reduction. Without seeded delivery, basis reduction is **896 to 768 bytes, or 14.29%**. The improvements also change numerator/basis precision within the existing 64-bit ring; the JSON includes all intermediate profiles to separate those effects.

The tests comprise **7,340,032 numerical grid/phase evaluations** across seven profiles and **1,792 fresh-record fused-gate executions**. Every fresh execution exactly reconstructs its specified modular integer computation. This does not make it exactly equal to native SiLU or validate language-model quality.

The online payload remains **32 bytes across both directions per scalar**, with **one masked-input opening before secure rescaling**. This experiment primarily reduces preprocessing traffic; it does not reduce online rounds or establish a latency improvement.

### What the scalar result does not solve

The code returns an unrescaled integer numerator. Secure final rescaling, output-ring choice and any later widening still need a protocol. The compiler must enforce the declared input envelope. Increasing polynomial precision consumes accumulator headroom; this example fits within the 64-bit ring for the declared ranges, but that is not a universal setting.

Fixed-point sine/cosine coefficient rounding makes the result depend slightly on the mask phase. The reported maximum is across the executed phase grid, not every 64-bit mask. A separate analytical rounding bound excludes uncertified numerical-library error. A complete security treatment must define the permissible randomized approximate functionality and its composition, not infer privacy from a correct ring identity or uniform openings alone.

## 4. Dither, stochastic rounding and the chain of power-of-two rings

A local rescaling identity can avoid a carry exchange when both a changed rounding rule and a smaller result ring are accepted.

Let q=2^k, S=2^f, and q'=q/S. For x=a+b mod q with canonical shares, define

    alpha = ceil(a/S) mod q',
    beta  = floor(b/S) mod q'.

A direct calculation gives

    alpha+beta = floor(x/S) + indicator(0 < a mod S <= x mod S) mod q'.

For a fresh uniform a, the result rounds up with probability (x mod S)/S. Away from the signed/modular boundary, that gives the familiar unbiased stochastic rounding marginal and variance t(1-t), t=(x mod S)/S.

The program exhaustively checks **1,118,208 share pairs** and the exact output distributions for **1,344 input values**. All match the identity.

However, this is not a free drop-in deterministic shift:

- The result ring is **2^(k-f)**, not 2^k.
- Simply reinterpreting those shares in the old ring is wrong: **241,640 of 279,552** tested naive lifts fail.
- The rounding randomness depends on the input sharing. A marginal distribution calculation is not proof of standard or composable secure truncation.

The truncation literature explicitly distinguishes these security definitions. The 2025 *Truncation Untangled* SoK discusses the issue and independent-randomness alternatives [7]; Curl proves related probabilistic truncation security in a specified stand-alone model [8]. Our algebra test supplies no replacement proof.

**Research use:** the compiler can consider runs of operations in successively smaller rings and place explicit secure widening only at genuine boundaries. Evaluate the entire schedule, including altered rounding and authentication. Do not announce zero-cost rescaling merely because one local identity works.

## 5. Wavelets are a comparator, not an unexplored discovery

Curl already applies discrete wavelet transforms to compress nonlinear lookup tables and evaluates the resulting system on language models [8]. It is a necessary baseline for difficult functions and tails, especially when a globally smooth low-rank translated basis is unattractive.

**Untested hybrid hypothesis:** choose a low-dimensional global translated basis for smooth regions, and a fixed-schedule private wavelet/FSS correction where it lowers total cost. Selection cannot be based on visible private-input branches. Both paths, or their selector, must be costed and secured. A sparse wavelet table does not automatically have low cyclic translation rank, and a high translation rank does not rule out a compact cryptographic lookup key.

No new wavelet implementation or wavelet performance comparison was executed in this package.

## 6. Negative controls: attractive algebra that does not meet the objective

**Projective or modular inversion is not ordinary fixed-point division.** With p=65537 and scale S=256, the ring expression S^2 / (3S) mod p returns residue 21931; the desired fixed-point rounded value is 85. Number-field or projective representations need an explicit decoding and rounding protocol, not relabelled division.

**Multiplicative masks from the odd units of Z_256 leak the 2-adic valuation.** Secret values 2 and 4 produce disjoint masked supports. The useful group action is not transitive on the whole private domain.

**Public low-rank additive masks leak a projection.** If masks are U*r with public U of rank smaller than the activation dimension, a public left-null vector v gives v^T(x-U*r)=v^T*x. The reference constructs that disclosure exactly. Coding-based pseudorandom generators need their actual hardness assumptions and noise structure; a bare low-rank linear map is not a substitute.

These counterexamples reject the naive constructions, not every conceivable protocol using projective coordinates, Galois rings, group actions or structured randomness.

## 7. Prioritized next work

| Priority | Work | Decision criterion |
|---|---|---|
| 1 | Benchmark a reviewed Z_(2^k) PCG against explicit/seeded dealer shares and an established OT/OLE baseline | Matched security; total bytes; seed setup; expansion rate; peak memory; contention with inference; sustained usable correlations, not raw generator outputs |
| 2 | Make the gate compiler optimize translated basis size, coefficient magnitude, precision and numeric envelope jointly | Better error/traffic/runtime trade-off than seeded baseline and strong FSS/wavelet implementations on real activation traces |
| 3 | Specify ring narrowing/widening and randomized rounding functionality through a complete decoder block | No hidden conversion rounds; bounded accumulated error; appropriate simulation/composition argument; deterministic schedule |
| 4 | Integrate immutable KV correlation lifetimes, resident sampling and authenticated model execution | Matched end-to-end output TPS, GPU-seconds/token, total network bytes and sustainable preprocessing |

The highest research upside is **less material generated and transferred per useful gate**, not weaker masks. No experiment in this package establishes near-native TPS, full-transformer fidelity, malicious security or first-in-literature novelty.

## 8. Reproduction and provenance

Run instructions are in README.md. The package is self-contained apart from NumPy and SciPy. It reuses the earlier `baseline_spectral.py` as an explicit comparison. No trained weights are downloaded or used. The reference driver sees both shares; production role isolation is not demonstrated.

`results.json` contains the public approximation coefficients, numerical envelopes, exact test counts and negative controls. `basis_screen.json` contains successful search results; solver failures are not represented as successful candidates. `galois_results.json` contains extension-ring and Walsh-domain checks. Fresh protocol records use operating-system randomness; observed fused-gate sample errors can vary slightly between runs. Numerical grids and algebra fixtures are reproducible.

## Primary references

[1] Z. Li, C. Xing, Y. Yao, C. Yuan. **Efficient Pseudorandom Correlation Generators over Z/p^k Z.** CRYPTO 2025; retrieved ePrint revision. https://eprint.iacr.org/2025/1223 . Read the threat model, parameter updates, and the warning against insecure benchmark rows in Section 7.

[2] Z. Li, H. Liu, C. Xing, Y. Yao, C. Yuan. **Faster Pseudorandom Correlation Generators via Walsh-Hadamard Transform.** CRYPTO 2026. https://eprint.iacr.org/2026/196 . This review uses the official abstract/publisher description, not an independent reproduction or a claim of applicability to powers-of-two rings.

[3] C. Bouillaguet, C. Delaplace, M. Hamdad, D. Vergnaud. **Practical cryptanalysis of pseudorandom correlation generators based on quasi-Abelian syndrome decoding.** 2025. https://eprint.iacr.org/2025/892 . Concrete attacks on earlier parameter sets; not a claim that every QA-SD parameterization or the entire ring-PCG construction is broken.

[4] J. M. Almira, E. V. Shulman. **On certain generalizations of the Levi-Civita and Wilson functional equations.** 2017 version. https://arxiv.org/abs/1612.03756 . Finite-dimensional shifted representations under specified regularity assumptions, not a general communication lower bound.

[5] M. Tasche. **Accelerating Convergence of Univariate and Bivariate Fourier Approximations.** Zeitschrift für Analysis und ihre Anwendungen 10(2), 239–250, 1991. https://ems.press/content/serial-article-files/34604 . Krylov-Lanczos polynomial boundary correction and Bernoulli formulations.

[6] B. Adcock, D. Huybrechs, J. Martin-Vaquero. **On the numerical stability of Fourier extensions.** https://arxiv.org/abs/1206.4111 . Coefficient conditioning versus stable approximation; the paper does not justify rejecting Fourier extensions universally.

[7] C. Harth-Kitzerow, A. Suresh, G. Carle. **SoK: Truncation Untangled: Scaling Fixed-Point Arithmetic for Privacy-Preserving Machine Learning to Large Models and Datasets.** PoPETs 2025(4), 369–391. https://crysp.petsymposium.org/popets/2025/popets-2025-0135.pdf . See the discussion of stochastic-truncation security and arithmetic domains.

[8] M. B. Santos et al. **Curl: Private LLMs through Wavelet-Encoded Look-Up Tables.** CAMLIS 2024. https://eprint.iacr.org/2024/1127 . Wavelet lookup compression and specified stand-alone probabilistic truncation security; the performance claims were not reproduced here.
