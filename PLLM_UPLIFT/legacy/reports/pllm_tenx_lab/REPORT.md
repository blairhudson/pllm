# PLLM: weighted garbled transducers and a search for 10× component improvements

**Experimental research note — 14 September 2026**

## Decision

This round found a 10×-class **component** trade-off, not a 10× private-LLM speedup or a proven new cryptographic primitive.

The strongest candidate combines additive-edge decision diagrams, secret telescoping edge potentials, and arithmetic/Boolean label conversion. It evaluates an exact discrete nonlinear function by following a fixed 16-step encrypted path and returns an arithmetic label. Preparation is offline, holds the public model, and builds fresh material. One Inference evaluator performs the online work. The Client does not hold weights or execute intermediate gates. No HE or second online inference worker is used.

For the tested SiLU table, the complete pipeline used **302,468 bytes**, versus **4,113,408 bytes** for the matched flat hashed arithmetic table: **13.60× less preparation payload**. Against our stronger synthesized Boolean implementation, it was **15.29× faster online** in interleaved Python-reference timings, but used **43.6% more preparation bytes**. The flat table remained approximately 50 times faster online than the new path. These comparisons describe different trade-offs and must not be multiplied or advertised as a single across-the-board improvement.

The candidate is a custom, unaudited garbling composition. The tests establish exact arithmetic and functional consistency, not a complete privacy, malicious-security, or composability proof. The application-level privacy target has not been experimentally proven.

## 1. What changed from the previous round

The source report, *PLLM: model-aware offline preparation, bounded garbling and packed encoded arithmetic*, established that Preparation can compute model-dependent label bases locally when it has W. It also demonstrated bounded hashed arithmetic lookup tables and identified residue/label expansion as a major cost. This round retains those role and arithmetic boundaries; it does not reintroduce interactive secret sharing or HE.

The original September 11 PLLM manuscript delegated masked matrix products and kept nonlinear operations at the Client. This experiment is a later architectural branch: nonlinear evaluation remains at the single online evaluator using prepared cryptographic encodings. It is not a measured optimization of the original nine-run Qwen implementation. All matrices here are synthetic; no model weights or production PLLM runtime were executed.

The baseline arithmetic label is

    L(x) = B + x Delta mod p.

The experiment uses two primes, 251 and 241, with 18 components per prime. Each Delta has 17 independently random coordinates and a final point-and-permute coordinate equal to one. A complete arithmetic label occupies 36 bytes. These are research parameters, not a claimed end-to-end security level. The integer reconstruction domain is the signed range of P = 251*241 = 60,491, namely [-30,245, 30,245].

## 2. Adjacent foundations and novelty boundary

**Edge-valued decision diagrams.** Hardware-verification work in the 1990s represented integer functions with additive weights on decision edges. Subfunctions that differ by a constant can share one residual graph. Factored variants also normalize multiplicative factors [1,2]. This compression method is not new.

**Garbled decision programs.** Private evaluation of trees and branching programs has substantial precedent, including schemes whose evaluator follows a short encrypted path [3]. This is not the invention of garbled branching programs or the first noninteractive private lookup.

**Arithmetic and mixed garbling.** Dash supplies the affine-label, public-matrix and tensorized evaluation foundations; ReDASH emphasizes scaling and representation costs [4,5]. Half-gates supply the Boolean baseline, and projective conversions are an active research topic [6,7]. These established primitives must be credited.

**Candidate contribution:** an additive-edge, secret-potential transducer that returns compatible arithmetic labels, plus a compiler that selects among weighted paths, Boolean synthesis and flat tables after accounting for conversions. Priority for this precise composition has not been established. A production backend would require a formal security analysis or a reduction to reviewed constructions.

## 3. Exact public-function compression

### 3.1 Canonicalize subfunctions modulo constants

For a residual integer table v, subtract its first value:

    offset = v[0]
    residual = v - offset.

Tables with identical residual arrays at the same decision level share a node; the offset is stored on the incoming edge. Evaluation sums edge weights along the path. This produces the exact original integer table, with no fitted polynomial or additional numerical approximation.

Every path is padded to the same public 16 levels. A constant subfunction still follows a padded chain. This avoids making the path length a data-dependent output.

The functions are fixed tables on all 65,536 two's-complement 16-bit inputs. Let s be the signed integer and x=s/4096:

| Name | Exact table definition used by the reference |
|---|---|
| SiLU | `rint(256 * x / (1 + exp(-x)))` |
| Clipped ReLU | `min(255, max(0, s) // 32)` |
| Sigmoid | `rint(4096 / (1 + exp(-x)))` |
| Exponential | `rint(256 * exp(x/2))` |

NumPy binary64 and ties-to-even `rint` generate these fixed integer tables. Their hashes are recorded. Exactness means equality to those tables, not arbitrary-precision transcendental evaluation or full-model floating-point fidelity. The protected CRT interface uses its smaller 60,491-input signed domain; compiling the full 65,536-input table is conservative rather than a range shortcut.

### 3.2 Executed state counts

| Function | Most-significant-bit first | Least-significant-bit first | Alternating ends |
|---|---:|---:|---:|
| SiLU | **1,291** | 11,170 | 2,441 |
| Clipped ReLU | **26** | 1,028 | 31 |
| Sigmoid | **1,863** | 15,882 | 4,092 |
| Exponential | **2,827** | 39,054 | 5,776 |

All **786,432** evaluations across four functions and three variable orders reproduced the original table. Node count is not a secure byte count or a runtime result. Variable ordering is a major public compilation choice; it does not depend on the secret input.

## 4. The proposed garbled transducer

### 4.1 Secret edge potentials

Assign each node v an independently secret arithmetic vector B_v. Let Delta be the arithmetic-label offset. If edge v→u has public semantic weight w, its encrypted payload includes

    E(v,u) = B_v - B_u + w*Delta.

A fresh secret state key identifies each node. The edge is encrypted under the concatenation of that state key and the valid garbled label of its input bit. Its payload also contains the next state key. Rows are stored under hashed state-key addresses, sorted by random addresses rather than semantic node numbers.

Set the terminal potential to zero and the root potential to

    B_root = B_output + table_offset*Delta.

The evaluator adds the contributions on the selected path:

    sum E(v,u) = B_output + f(x)*Delta.

The node potentials telescope. No plaintext edge weight or intermediate value is intentionally released. The output is an ordinary arithmetic label, ready for the next public matrix product.

The method is not a scalar function that the evaluator may freely call on x+t. It requires compatible cryptographic input-bit labels. Arbitrary label corruption is rejected in the tested cases. That functional observation is not a full proof against an adaptive evaluator.

### 4.2 Actual conversion, not an ideal oracle

The implemented input conversion is:

1. A full-domain, label-keyed table for each input residue.
2. A 98-AND garbled circuit reconstructing the signed CRT value as 16 encoded bits.
3. The weighted transducer's 16 encrypted transitions.
4. Direct arithmetic-label output from the telescoping sum; no final Boolean conversion is needed on this path.

For residues a mod251 and b mod241, the circuit uses

    t = (217*b - 217*a) mod241,
    positive = a + 251*t,

and subtracts 60,491 when positive≥30,246. Since 217 is the inverse of 251 modulo241, this gives the signed representative. The residue tables precompute the small modular terms; the Boolean circuit handles combination, carries and signed interpretation.

The bridge is **106,168 bytes**, including its actual two residue tables and 3,152-byte garbled Boolean core. All **60,491** clear CRT cases and **64** fresh garbled CRT cases passed. The evaluator object contains no garbler zero-label array or Boolean Delta.

A weighted node uses 152 serialized bytes: a 16-byte hashed address plus two 68-byte encrypted records. Each record carries a 16-byte next-state key, a 36-byte arithmetic contribution and a 16-byte consistency tag. The program header is 68 bytes.

### 4.3 Fresh functional executions

The harness evaluated **48 fresh transducer gates**, all exactly. All **48** tested wrong bit labels were rejected. It also executed **four fresh protected public-matrix blocks**, with **16** checked matrix-to-SiLU outputs. The matrix computation genuinely operates on the incoming arithmetic labels; the plaintext reference is only in the external test driver.

There is no online Preparation call or intermediate network exchange in these reference graphs. One scalar has 36 bytes of encoded input and 36 bytes of encoded output. This is not a claim about a complete LLM request: embedding access, attention, protected state and sampling are absent.

## 5. Match stronger baselines, including all conversions

### 5.1 Flat arithmetic table

The flat reference constructs one encrypted row for every integer in [-30,245,30,245]. It uses the same 36-byte arithmetic input/output encodings and exact table function. It is the earlier hashed-lookup format extended to this wide integer domain, not a claim that the previous small-field LUT was four megabytes.

Each row is 68 bytes; the header is 20 bytes. The actual serialized table is **4,113,408 bytes**. One full fresh flat table for each function was built and checked, rather than estimating an unexecuted ideal object.

### 5.2 Boolean synthesis

The stronger baseline synthesizes each output bit with shared Shannon or positive-Davio expansions, complement normalization, free XOR, and memoized ANDs. Both MSB-first and LSB-first orders are tried. It selects the lowest-AND circuit among these four candidates.

| Function | Best core AND count |
|---|---:|
| SiLU | 3,202 |
| Clipped ReLU | **10** |
| Sigmoid | 4,672 |
| Exponential | 7,282 |

All **262,144** clear-table evaluations and **32** fresh isolated garbled executions passed. These are inspectable reference synthesis methods, not a claim to match a leading industrial logic optimizer.

For the matched pipeline, the CRT circuit is composed directly into the Boolean function circuit. An implemented 16-bit-to-arithmetic return gadget produces the same output label as the weighted path. All input and output conversions are included in both payloads and timings. The resulting comparison uses **32 fresh matched pairs**, eight per function, with two warmups and seven interleaved same-input repetitions per pair.

### 5.3 Measured trade-off

All byte figures are serialized prepared material. Times are scalar Python-reference medians in milliseconds.

| Function | Flat bytes | Boolean bytes | Weighted bytes | Boolean online ms | Weighted online ms | Speedup vs Boolean |
|---|---:|---:|---:|---:|---:|---:|
| SiLU | 4,113,408 | 210,616 | 302,468 | 10.8221 | 0.7076 | **15.29×** |
| Clipped ReLU | 4,113,408 | 108,472 | 110,188 | 0.6359 | 0.6787 | **0.94×** |
| Sigmoid | 4,113,408 | 257,656 | 389,412 | 17.3838 | 0.7152 | **24.31×** |
| Exponential | 4,113,408 | 341,176 | 535,940 | 21.7806 | 0.7049 | **30.90×** |

Weighted preparation is **13.60×, 37.33×, 10.56× and 7.68× smaller** than the flat table, respectively. However, it is larger than the executed Boolean baseline in every row. Flat online lookup took about **0.013 ms** per scalar in the separate direct comparison and was much faster than the weighted path.

The result is a three-way trade-off:

- Flat table: smallest online work, largest disposable material.
- Boolean synthesis: smallest prepared payload among the executed pipelines; particularly strong for clipped ReLU.
- Weighted path: much less material than flat lookup and much less reference evaluation work than Boolean synthesis for the three curved functions.

These are CPU Python SHA-256/SHAKE constructions, not a production fixed-key AES garbling engine, GPU benchmark, model TPS, or universal cryptographic comparison. Differences in compiler quality, vectorization and hashing backends may materially change the ratios. Raw interleaved samples are included, with no outlier deletion.

### 5.4 A stronger published baseline remains unimplemented

EUROCRYPT 2024 **logrow** reports `(n-1)*kappa + n*m*kappa + N*m` bits for an N-row, m-bit-output LUT, where n=ceil(log2 N) [8]. This is not logarithmic total traffic.

For N=65,536, m=16, kappa=128, the formula is **135,408 bytes before our arithmetic-label conversions**. This analytical number already shows why beating a 4.1 MB flat reference does not establish a 10× advantage over the best literature. No logrow runtime or full conversion composition was implemented here. This is a necessary next matched baseline, not a result to extrapolate.

## 6. Other rare or speculative approaches tested

### 6.1 Factored diagrams: normalize scales as well as offsets

The factored diagram compiler also divides a residual by its common integer factor, requiring that factor to be a unit in both arithmetic fields. This is related to the established FEVBDD representation [2].

All **524,288** function evaluations across four functions and two variable orders were exact. For MSB-first SiLU, the state count fell from 1,291 to **1,213**, only **6.0%**. Clipped ReLU and sigmoid did not shrink, and exponential fell by just two nodes. This did not provide a 10× opportunity.

Only the public factored compiler was implemented. Secure propagation of multiplicative edge scales would require a separate construction and proof; it is not silently substituted into the additive transducer.

### 6.2 Four Russians / subset-sum matrix compilation

Hypothesis: because the weights are public and small integers, replace much of W times the encoded columns with precomputed subset sums and public lookups. LUT-GEMM applies related lookup ideas successfully to GPU weight-only quantization [9].

The executed C++ reference builds the current-label subset-sum tables online and uses precompiled public weight-bit patterns. Group widths 4, 8 and 10 were compared with an AVX-512 VNNI signed-weight/unsigned-label dot-product kernel. Both compute the same exact residue products, and table construction is included.

| Matrix | Direct VNNI ms | Group-4 LUT ms | Group-8 LUT ms | Group-10 LUT ms |
|---|---:|---:|---:|---:|
| 256×1,024 | 0.1294 | 1.6865 | 1.9479 | 3.6204 |
| 512×2,048 | 0.9151 | 6.7294 | 8.0693 | 12.4536 |
| 1,024×4,096 | 3.7936 | 29.0460 | 33.9923 | 51.0442 |

Every tested adaptation was slower: roughly **7–28×**. Table storage and gathers outweighed saved arithmetic. This rejects this CPU encoded-label layout, not LUT-GEMM's published GPU result. Static public-pattern compilation is excluded from both online accounting and timings; its actual stored byte count is recorded.

### 6.3 A larger Mersenne field to reduce component count

Hypothesis: replace 36 small-field components with six components modulo 2^31−1. Five random coordinates plus a fixed permutation coordinate could provide a larger raw offset entropy, while reducing lane count. This does not establish an equivalent complete security level.

The raw exact matrix kernels improved **2.77×, 1.26× and 1.36×** at widths 512, 1,024 and 2,048. They use FP64 versus FP32 BLAS with four threads and explicit modular reduction. Weight storage doubles in the wider-field reference, and the large-field nonlinear conversions were not implemented. The two matrix representations were checked against independent int64 products, not run as a full mixed-field cryptographic graph.

Across the subset-sum and wider-field experiments, **408,576** output coordinates were exact. These are integer arithmetic checks, not security samples. Neither direction achieved the targeted 10× kernel improvement.

## 7. Attacks on attractive shortcuts

### 7.1 Stop traversing when the function becomes constant

This would save work in clipped ReLU's tails. The unpadded execution depth revealed the input sign in **65,536/65,536** tested cases: depth one identified the negative half of the domain. The retained construction always performs 16 transitions. Publicly small residual graphs are useful; secret-dependent path lengths are not free.

### 7.2 Derive node masks from evaluator-visible state keys

This would make preparation look much more compact. But if B_v is computable from the state key, the evaluator subtracts it from each decrypted contribution:

    E(v,u) - B_v + B_u = w*Delta.

The final point-and-permute coordinate of Delta is one. CRT reconstruction then reveals w, and summing the edge weights reveals f(x). The explicit negative control recovered **256/256 outputs and 4,096/4,096 edge weights**.

The retained construction uses independently secret potentials. Secret seeds held only by Preparation may generate its local randomness; handing a seed that reproduces those potentials to Inference is the unsafe step.

### 7.3 Reduce the input modulus merely because the output has few bits

A small output alphabet does not make f a function of x modulo a small p. Tests found explicit conflicting same-residue inputs for **all 16 function/modulus combinations** tested: four functions, moduli 251, 241, 256 and 4,096. This shortcut changes the computation or requires additional hidden information. It does not follow from output clipping.

These are attacks on deliberately weakened alternatives, not a proof that the retained custom design is secure.

## 8. What this says about the 10× objective

This result improves one kind of nonlinear operation under the requested architecture. It does not reduce the number of arithmetic-label components in the matrix path or establish a cheap full attention/normalization implementation.

For illustration, if the affected operation consumes 20% of total runtime, accelerating it by 15.29× yields only

    1 / (0.8 + 0.2/15.29) = 1.23×

end-to-end, before other overhead. The 20% share is hypothetical, not measured. A 10× full-system claim requires profiling the whole system, not multiplying component ratios.

Fresh material remains a hard constraint. At an illustrative 100,000 SiLU-like gates per token, **302,468 bytes per gate would require 30.25 GB per token**, before other operators. The example is not a measured model or graph count. Large per-gate relative reductions do not make that absolute preparation rate acceptable. The original paper correctly recorded preparation as real work and traffic; moving it offline does not make it free.

The decision is to pursue **backend selection and conversion-aware compilation**, not to replace every nonlinear operator with this structure:

1. Boolean synthesis wins for the tested clipped ReLU.
2. Weighted paths offer a useful point for the tested curved functions when online work matters more than minimum prepared bytes.
3. Flat lookup may win when latency dominates and setup is genuinely amortized or affordable.
4. Reviewed compact LUTs such as logrow must be benchmarked before any state-of-the-art claim.

The most promising specific research artifact is an exact nonlinear compiler that selects variable order, residual normalization, output encoding and conversion boundaries together. A proof for secret-potential transducers and a matched native implementation come before production use.

## 9. Reproducibility and boundaries

The scripts are single-process references with role-separated evaluator objects. External test drivers hold plaintext references and garbler state for checking. This does not demonstrate process isolation, authenticated deployment, rollback protection, malicious preprocessing resistance or secure erasure. Consistency tags and corruption tests are not substitutes for those properties.

Fresh cryptographic material uses operating-system randomness. Numeric fixtures use fixed NumPy seeds. Repeated evaluations during timing use the same semantic input, not multiple different inputs under a reusable wire encoding. No reuse mechanism for one-time garbled material is claimed.

No GPU, pretrained model, full transformer, attention, KV cache, sampling, private embedding lookup or sustained-preparation benchmark was run. Public bounds, shapes, table identities and fixed schedule are visible. Implementation side channels and the complete simulation argument remain open design tasks.

The captured environment is in `environment.json`; all raw measurements are in JSON. The provided C++ kernel experiment requires an x86-64 CPU with AVX-512 VNNI. The Python arithmetic/garbling experiments do not require that extension.

## Primary sources

[1] Y.-T. Lai and S. Sastry. *Edge-valued binary decision diagrams for multi-level hierarchical verification.* DAC 1992. https://dl.acm.org/doi/10.5555/113938.149642 . Public function representation, not a cryptographic construction.

[2] P. Tafertshofer and M. Pedram. *Factored Edge-Valued Binary Decision Diagrams.* Formal Methods in System Design 10, 243–270 (1997). https://link.springer.com/article/10.1023/A:1008691605584 . Additive/multiplicative edge representations; publisher abstract and bibliographic references consulted.

[3] S. Niksefat, B. Sadeghiyan, P. Mohassel. *Oblivious decision program evaluation.* IET Information Security 7(2), 155–163 (2013). https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-ifs.2012.0032 . Garbled path evaluation precedent, not the retained custom transducer's proof.

[4] J. Sander et al. *Dash: Accelerating Distributed Private Convolutional Neural Network Inference with Arithmetic Garbled Circuits.* https://arxiv.org/html/2302.06361v2 . Public arithmetic labels and offline/online foundations.

[5] Maurer, Sander, Eisenbarth. *ReDASH: Fast and efficient Scaling in Arithmetic Garbled Circuits for Secure Outsourced Inference.* https://arxiv.org/html/2506.14489v1 . Representation/scaling baseline, not reproduced here.

[6] S. Zahur, M. Rosulek, D. Evans. *Two Halves Make a Whole: Reducing Data Transfer in Garbled Circuits using Half Gates.* https://eprint.iacr.org/2014/756 . Two-ciphertext AND and free-XOR-compatible foundation; research SHA-256 instantiation used here.

[7] N. Khambhati, A. Bhattacharya, D. Heath. *Duty-Free Bits: Projectivizing Garbling Schemes.* https://eprint.iacr.org/2026/476 . Official abstract consulted; does not make the reverse arithmetic-to-bit conversion free.

[8] D. Heath, V. Kolesnikov, L. K. L. Ng. *Garbled Circuit Lookup Tables with Logarithmic Number of Ciphertexts.* EUROCRYPT 2024. https://eprint.iacr.org/2024/369 . Communication formula from the official abstract; not implemented.

[9] G. Park et al. *LUT-GEMM: Quantized Matrix Multiplication based on LUTs for Efficient Inference in Large-Scale Generative Language Models.* ICLR 2024. https://arxiv.org/html/2206.09557v4 . Published GPU lookup-multiplication approach; our CPU adaptation is a different experiment.

Research snapshot: 14 September 2026. No first-in-literature or full-SOTA coverage claim.
