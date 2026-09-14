# PLLM: model-aware offline preparation, bounded garbling and packed encoded arithmetic

**Research round: 14 September 2026**

## Decision

Allowing Preparation to hold the public model removes the previous model-free OT matrix-product setup entirely. It does not remove the need for cryptographic input binding, one-time garbling or encoded arithmetic. The strongest result in this round is a model-aware compilation path: prove tighter intermediate bounds, select smaller arithmetic representations, fuse unary operations, and prepare only the proven domain using a lookup representation that does not expose semantic indices.

The desired architecture remains a minimal-work Client, an offline-only Preparation service and one online Inference service. Both remote services hold W. Neither HE nor a second online inference worker appears in the experiments. Preparation must remain trusted for correct preparation and must not obtain the online encoded inputs; it must not collude with Inference. Loading W at Preparation is not the same as making Preparation an online worker.

**Executed:** 108 exact bounded fixed-point graphs, 504 public linear stages, 131,200 tests of integer product bounds, 633,600 checked CRT label coordinates and 69,120 packed-weight label coordinates. Four deliberately unsafe optimizations were broken. These experiments are distinct and should not be pooled into a cryptographic-security sample size.

**Not established:** near-native TPS, production security of the custom lookup construction, complete transformer inference, or low sustained preprocessing cost.

## 1. What changes when Preparation has W

The previous report computes model-dependent label bases using oblivious transfer. For an affine arithmetic label

    L(x) = B + x Delta mod p,

public matrix evaluation gives

    W L(x) = W B + (W x) Delta mod p.

Preparation needs W B to construct the next garbled operation. With W local, it calculates W B directly. There is no OT, HE, two-party OLE, or network message for this matrix product. The model-dependent base W B stays at Preparation; publishing it would not preserve the intended protection.

The previous experiment's 288mn-byte OT vector term consequently disappears. For the earlier 8192-square illustration, this removes the approximately 19.3 GB term; that number was an analytical extrapolation of the old message format, not a measured large-model setup. It is not a new 19.3 GB transfer measured and removed here.

This saves communication and cryptographic setup operations, not all preparation computation. W B is still a matrix product with encoded components. Fresh nonlinear tables are still delivered to Inference. Those costs can be batched, but are not free or reusable for unlimited requests.

`graph_results.json` also includes a like-for-like byte accounting of the previous width-four, full-field gate layout using local setup. That calculation is kept separate from the new width-eight fixed-point experiments.

## 2. A complete small fixed-point graph, not just modular ReLU

Each new graph uses integer matrices W with entries in [-2,2], width eight, and depth two, four or eight. The initial input is an integer vector in [-4,4]^8. Each stage computes exactly

    t = W x
    x_next = min(7, floor(max(0,t)/4)).

This is a specified quantized/clipped ReLU graph. It is not SiLU or a pretrained language model. The compiler propagates public integer intervals using the actual matrices. It rejects a stage if its signed result could exceed the representable small-field interval [-125,125]. No empirical input trace determines the accepted domain.

Arithmetic labels use p=251 and 18 components: 17 secret offset coordinates and one fixed point-and-permute coordinate. The approximately 135-bit entropy of those offset coordinates describes the reference parameters, not the security level of the full implementation. Labels and gates follow the established arithmetic-garbling family [1,2]; the SHAKE/SHA-based reference has no independent security proof.

Three configurations implement the SAME graph:

1. Full-domain separate gates: one ReLU table and one rescale/clip table per scalar.
2. Full-domain fused gates: a single table for the composed unary operation.
3. Public-bound fused gates: one table only for the compiler-proved interval, indexed by a hash of the complete input label.

Every configuration uses a fresh root and fresh gates. The Client receives a 32-byte root from which it derives input encoding and final decoding state. It receives no matrices and does not execute intermediate model layers. This is ordinary domain-separated seeded randomness, not an evaluated PCG or a novelty claim.

All 108 graph executions passed, covering 504 linear stages. All 108 second input assignments were rejected by the client object, and all 108 tested altered labels were rejected. A separate test changed a model coefficient after preparation in 48 fresh circuits; all 48 were rejected. These negative controls do not prove malicious security, snapshot rollback resistance, or authentic model selection.

### Measured serialized setup payload

Values below are means over 12 matrices/inputs at each depth. Fresh random cryptographic state changes table contents, but not the public-domain sizes. Every table's byte counter was checked against its actual serialization.

| Depth | Separate full tables | Fused full tables | Fused bounded hash-indexed tables |
|---:|---:|---:|---:|
| 2 | 273,792 B | 136,912 B | 58,184.67 B |
| 4 | 547,552 B | 273,792 B | 104,408.17 B |
| 8 | 1,095,072 B | 547,552 B | 181,334.33 B |

Fusion removes approximately half the baseline table traffic. Domain specialization saves another 57.50%, 61.87% and 66.88% against the already fused full-domain baseline. Combined savings against separate full-domain tables are 78.75%, 80.93% and 83.44%. These are improvements against this reference's simple full-field tables, not against logrow or a tuned arithmetic-garbling framework.

Online input is 144 bytes and online output is 144 bytes in every graph. There are no intermediate online messages or Preparation calls. The client seed is 32 bytes. This fixed online size follows from the fixed width and one graph execution; it is not an LLM prompt/token bandwidth claim.

Counts include gate identifiers, row counts, format flags, ciphertexts, explicit hash addresses where used, and the client root. They exclude model distribution, transport framing, an authenticated public graph manifest, inventory metadata, and production authentication. Both services are assumed already to have the model. The count is serialized protocol payload, not observed network-interface traffic.

Timing samples are retained, but the scalar Python table implementation is not a performance target. Bounded tables improved preparation markedly; online lookup timings were close to full fused tables. Smaller offline packets do not imply proportionally faster online evaluation.

## 3. Why sparse tables need different addressing

In the earlier full-domain arithmetic table, a label's final coordinate provides its encrypted row index. Suppose we simply remove rows outside a public domain D. The exposed remaining indices are

    {b + x mod p : x in D},

where b is the secret shift in the label's final coordinate. For a nonempty proper cyclic interval over a prime field, this set has a unique shift. Inference can recover b and subtract it from the observed input label's final coordinate.

The explicit attack recovers 192/192 inputs across six interval widths. This is why the reference refuses sparse point-and-permute tables.

The tested alternative is

    address = H("address", gate_id, full_input_label),

with a 128-bit address, domain separation from the encryption pad, and a fresh gate identifier. Each row encrypts the output label under the full input label. Only an encoded row is queried; no plaintext interval test is performed by Inference. The row count and public domain may be visible because they depend on public weights and the declared input envelope, not the actual input.

This fixes the demonstrated index-set attack, but that fact is not a complete proof of the sparse construction. A proof must cover correlated affine labels, random-oracle/hash assumptions, malformed queries, dictionary access patterns, selective failures and composition. The table's small tag is a functional consistency check, not a standalone malicious-verification theorem. A production design should integrate a reviewed garbling backend rather than deploy this research encryption format.

Logrow [4] is an important stronger comparison. Its communication formula has a logarithmic number of security-parameter-sized ciphertexts plus a linear table-data term. It does not give logarithmic total communication. It uses a different bit-label interface; conversions to/from the arithmetic labels must be included in any performance comparison. It was researched, not implemented in this round.

## 4. Geometry can reduce encoded arithmetic itself

Independent coordinate bounds discard relationships between coordinates. For normalized inputs, the ball or ellipsoid described by those relationships can be much tighter than a box. RMSNorm provides a concrete case [3].

Let

    z = v / sqrt(||v||_2^2/d + epsilon),  epsilon >= 0,

so ||z||_2 <= sqrt(d), with a defined nonzero denominator. Suppose x is coordinatewise rounding of S z to integers. Then x=S z+e with |e_j| <= 1/2. For an integer public matrix W,

    |(W x)_i| <= S sqrt(d) ||W_i||_2 + 0.5 ||W_i||_1.

The implementation computes a conservative integer ceiling using integer square roots, not an unchecked floating-point estimate. For row squared norm v and L1 norm u, it uses

    B_i = (ceil_sqrt(4 S^2 d v) + u + 1) // 2.

The coordinate-box alternative uses |x_j| <= ceil(S sqrt(d)+0.5) and multiplies that by the row L1 norm. With learned gamma, one can apply the corresponding ellipsoidal support bound, e.g. ||W_i diag(gamma)||_2 before input rounding, plus the appropriate rounding and weight-quantization error. The executed matrices use gamma=1. Arbitrary fixed-point RMSNorm approximation does not automatically inherit the exact real norm bound; its error must be incorporated.

This is Cauchy-Schwarz and abstract-interpretation-style bound propagation [5], not new normalization mathematics. The proposal is to use a certified bound to remove unnecessary encoded arithmetic and prepare smaller valid domains.

### Executed bound checks

The test uses 256 synthetic integer rows per dimension, W entries in [-7,7], S=4, 128 normalized random input vectors, and 32 row-aligned stress inputs per dimension. There are 131,200 checked integer products, all within the bounds.

| Input width | Median box bound | Median geometric bound | Median improvement | Residue counts |
|---:|---:|---:|---:|---|
| 64 | 7,953 | 1,229 | 6.45x | 2 to 2 |
| 256 | 62,400 | 4,911.5 | 12.66x | 3 to 2 |
| 896 | 405,713 | 17,179.5 | 23.62x | 3 to 2 |
| 4096 | 3,929,530 | 78,412.5 | 50.10x | 3 to 3 |

The chosen prime basis begins 251,241,239,233,... and covers an integer bound when its product exceeds twice the bound. Better numeric bounds do not always remove a whole prime. At widths 256 and 896 all 256 rows drop from three residues to two; at widths 64 and 4096 none do.

The row-aligned inputs reach roughly 86-97% of their respective geometric bounds. This prevents treating typical Gaussian activation values as proof that the range can be made arbitrarily small.

No pretrained weight file was read. An attempted range download from the official Qwen repository failed because the container could not resolve the host. All numeric weight results are therefore explicitly synthetic. The cited Qwen configuration was used only as contextual reference for dimension 896, not as evidence about actual Qwen weights or quality.

## 5. A real CRT-label arithmetic benchmark

A separate experiment actually encodes the same integer input in two and three prime residues, performs the public matrix product on its labels, validates the reconstructed label in every component, then reconstructs the integer result with CRT.

It uses widths 896,2048,4096, S=1 normalized/rounded synthetic inputs and W entries in [-7,7]. The computed worst-case bounds fit the two-prime product. Both representations therefore compute the same integer function; the experiment is not comparing a lower-precision approximate output against a higher-precision one.

Labels have 18 components per prime, hence 36 versus 54 arithmetic columns. All 633,600 evaluated label coordinates and the resulting integer products passed independent int64 checks.

Four CPU threads, three warmups and 15 interleaved timing samples are used. FP32 accumulation is exact for the integer ranges in this fixture; each product includes modular reduction.

| Square width | Scalar integer-in-FP32 baseline | 2 residues | 3 residues | 3-to-2 speedup |
|---:|---:|---:|---:|---:|
| 896 | 0.02688 ms | 0.83782 ms | 1.20872 ms | 1.443x |
| 2048 | 0.17069 ms | 2.96636 ms | 4.32043 ms | 1.456x |
| 4096 | 1.62680 ms | 10.27705 ms | 14.80438 ms | 1.441x |

The largest two-residue case is still 6.32x the scalar reference. Smaller shapes have larger relative overhead. This is not a native INT4/FP8 model-engine comparison. No GPU, garbled nonlinearity, CRT base conversion between layers, preprocessing replenishment, private attention or token sampling is timed here. The small-field fixed-point graphs in Section 2 and this wide-integer linear experiment are separate, not a complete integrated wide-integer decoder.

## 6. Preserve INT4 weight storage and decode tiles once

The previous FP32 microbenchmark stored weights at four bytes each. That would surrender much of a modern native quantized model's bandwidth advantage. I tested a representation that stores two signed 4-bit weights per byte, unpacks a tile once, and applies it to all encoded columns with centered signed-8-bit residue values and exact integer accumulators.

All 32 fixtures passed, covering 69,120 output-label coordinates. Stored weight bytes are exactly mn/2: the encoded components do not create another weight replica at Inference. Preparation separately has its permitted copy. Matrix shapes range from 32x128 to 32x4096, with two or three residues. This verifies the layout and arithmetic, not a fused GPU implementation or a speedup. The NumPy reference unpacks into a temporary tile and is not optimized for timing.

### Explicitly speculative hardware budget

For 36 encoded columns, each weight contributes about 72 multiply/add operations. With ideal 0.5-byte INT4 weight traffic, that is 144 operations per weight-byte. NVIDIA lists H100 SXM bandwidth of 3.35 TB/s and 3,958 sparse INT8 TOPS [6]. Using 1,979 dense TOPS, saturating the advertised weight bandwidth at 144 operations/byte would require about 24.4% of dense compute peak. Padding the encoded columns to 64 raises that to 43.3%.

This arithmetic does not prove achievable performance. It excludes activation/label traffic, hash tables, tile metadata, output writes, modular reduction, unpacking, instruction scheduling, launch latency and other operators. The useful hypothesis is only that a fused INT4-storage/INT8-label kernel could remain bandwidth-limited on a sufficiently capable GPU, despite doing more arithmetic. The measured CPU overhead does not establish that GPU result, and the calculation does not establish whole-model TPS.

## 7. Further hypotheses tested and rejected or restricted

### Reuse the prepared graph across requests

For two different values on the same affine wire,

    L(x)-L(y) = (x-y) Delta.

Because Delta's permutation coordinate is one, the last coordinate of the difference reveals x-y, and dividing the full vector by that nonzero value recovers Delta. All 256 tests recover the entire offset, without knowing either original semantic value. This attack is on reusing the SAME wire base for changing values. Fanout of one immutable label inside one fixed garbled execution is not that reuse.

Model compilation decisions can be reused. Ordinary one-time labels and their garbled instance cannot be reused for distinct inputs. A fresh seed also requires a compatible fresh circuit; giving a new input seed with old tables does not solve this.

### Collapse all encoded columns into one online scalar

Suppose Preparation supplies public slopes sufficient to recover every label coordinate from one of them. Those slopes are Delta_j/Delta_k. The fixed permutation component exposes 1/Delta_k, hence Delta_k and the complete offset. All 256 fixtures recover Delta exactly. This rejects the tested affine lane-reconstruction shortcut, not every possible succinct computational encoding.

### Secretly permute the copied model

Preparation can now create W'=P_out W P_in^-1. For pure row and column permutations of a known public matrix, sorted row signatures recover the row permutation, after which column matching recovers the input permutation. All 64 randomly generated matrices at widths 8,16,32,64 were recovered. This does not analyze arbitrary dense bases or scaling schemes; it breaks this particularly attractive native-arithmetic construction.

### Exploit holes in the exact reachable integer set

Set-valued convolution computes reachable outputs of a small integer row without interval overapproximation. For dense random weights and a four-value input alphabet, tested output domains become essentially contiguous: mean support density is 99.91% at width 16 and 100% at widths 64 and 128. There is no general huge hidden sparsity to exploit.

Deliberately multiplying every weight by four yields support density near 25%. The compiler can factor out that public gcd and fold it into a following gate. This is useful where the structure genuinely exists, not a reason to assume arbitrary trained weights have it. These are synthetic additive-combinatorics checks, not observed model sparsity.

## 8. Research positioning and next decisions

The contribution cannot be 'Preparation has weights' or 'one online evaluator': arithmetic garbling already supports that division [1,2]. Gate composition and range propagation also have precedent. A more defensible candidate is a compiler that connects model-derived numeric certificates to encoded representation and gate-domain choices, while preserving secret-index hiding and accounting for all lifetime-dependent material.

The implementation order suggested by this evidence is:

1. Use local model-aware garbling setup. Remove OT/OLE/HE from this path rather than optimize an unnecessary protocol.
2. Bring the actual quantized model into a sound bound/precision pass. Include RMSNorm approximation error, learned scale vectors, biases, residual correlations and attention bounds. Select residues only when a public proof covers all permitted inputs.
3. Fuse scalar operations and compare reviewed LUT/Boolean/arithmetic backends, including logrow and ReDASH conversion/rescaling. A simple hash dictionary is a research reference, not the final backend.
4. Implement the packed-weight, multi-component GPU kernel. Compare matched integer semantics and native packed-weight storage, then measure a complete decoder block.
5. Count sustained preparation, complete output feedback, private attention, KV state and sampling. Do not infer LLM throughput by multiplying the separate component reductions.

The data support a better constrained architecture and measurable component improvements. They do not yet support near-native output TPS or economical end-to-end private LLM inference. The highest-upside measured direction is eliminating unnecessary encoded precision; the highest-upside unmeasured direction is retaining native packed-weight traffic while reusing each unpacked weight tile across the remaining components.

## 9. Reproducibility and security scope

All programs execute role-separated objects in one process. The driver sees plaintext references to check correctness. Fresh graph secrets use operating-system randomness; the arithmetic microbenchmarks deliberately use reproducible synthetic randomness and are not deployment key generation. No model weights, runtime credentials, or private user inputs are included.

The client enforces one assignment in memory. Cloning, snapshot rollback, authenticated persistent allocation, secure erasure and cancellation semantics are not implemented. All artifacts are fresh per graph test. Same-input reevaluation is used for adversarial negative controls; no distinct semantic input is legitimately encoded twice for one prepared circuit.

Public dimensions, model choice, compiled bounds, shape and timing remain visible. No constant-time implementation or complete adaptive/malicious proof is claimed. Preparation must not gain access to the online transcript. Tags and the output-label equality checks demonstrate functional error rejection, not a general malicious-security theorem.

Run the scripts listed in README.md. Captured JSONs include timings, environment metadata, all experiment counts, and limitations. The reference is intentionally small enough to inspect; it is not the production PLLM codebase.

## Primary research sources

[1] Sander et al. **Dash: Accelerating Distributed Private Convolutional Neural Network Inference with Arithmetic Garbled Circuits.** https://arxiv.org/html/2302.06361v2 . Arithmetic labels, tensorized public-weight evaluation and offline/online role separation.

[2] Maurer, Sander, Eisenbarth. **ReDASH: Fast and efficient Scaling in Arithmetic Garbled Circuits for Secure Outsourced Inference.** https://arxiv.org/html/2506.14489v1 . RNS scaling/base conversion and representation-dependent performance. Researched as a baseline, not reproduced here.

[3] Zhang and Sennrich. **Root Mean Square Layer Normalization.** https://arxiv.org/abs/1910.07467 . Normalization definition. The bound and quantized experiments in this report are derived and executed here.

[4] Heath, Kolesnikov, Ng. **Garbled Circuit Lookup Tables with Logarithmic Number of Ciphertexts.** EUROCRYPT 2024. https://eprint.iacr.org/2024/369 . The official abstract provides the communication formula (n-1)kappa + n m kappa + N m bits, where n=ceil(log2 N). Not implemented.

[5] Mirman, Gehr, Vechev. **Differentiable Abstract Interpretation for Provably Robust Neural Networks.** ICML 2018. https://proceedings.mlr.press/v80/mirman18b.html . Adjacent foundation for sound correlated-domain bounds, not a garbling result.

[6] NVIDIA. **H100 GPU product specifications.** https://www.nvidia.com/en-au/data-center/h100/ . Accessed 14 September 2026. Peak specifications are not measured performance.

[7] Qwen. **Qwen2.5-0.5B-Instruct configuration and model repository.** https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/raw/main/config.json . Model content was not downloaded into this experiment; do not attribute synthetic results to Qwen.

[8] Khambhati, Bhattacharya, Heath. **Duty-Free Bits: Projectivizing Garbling Schemes.** https://eprint.iacr.org/2026/476 . Relevant representation-conversion lead previously identified. Its abstraction is not treated as a free conversion from arbitrary writable masked scalars.
