# Encrypted-computation reproduction contracts

Review date: 17 October 2026. Status: **source-qualified implementation specification; implementation and execution not authorized by this document**.

Scope: independent PLLM implementations of NEXUS, THOR, CacheMir, and ROSETTA. This document supplements [TEN-METHOD-COVERAGE.md](../TEN-METHOD-COVERAGE.md); it neither changes its evidence nor supplies a system ranking. Only this document was added. Review used primary PDFs, static PLLM source, and source-file hashing. No upstream research code was executed, linked, or vendored; no build, model workload, benchmark, runtime change, or manuscript change was performed.

## 1. Decisions needed first

1. **Approve a bootstrapping-capable cryptographic backend and its key/parameter contract.** The current SEAL/TenSEAL path is leveled CKKS; it has no CKKS bootstrap, programmable bootstrap, or CKKS–TFHE conversion implementation. Every complete method here needs capabilities beyond that path. Increasing the existing fourteen-level bound does not implement them.
2. **Resolve each source's missing numerical and representation details before implementation.** Paper formulas determine algorithms, but do not always determine executable coefficient tables, prime chains, scales, key-switch decomposition, or interval endpoint behavior. Missing details below are blocking specification questions, not permission to invent a convenient substitute.
3. **Approve functionality extensions separately.** THOR's evaluated task is BERT encoder classification. NEXUS has BERT evidence and a one-output Llama-3-8B extension, but no specified persistent-KV serving lifecycle. CacheMir and ROSETTA target autoregressive decoding, yet do not fully specify lookup, final output selection, token feedback, and service failure semantics. A common complete decoder needs explicit decisions at those boundaries.
4. **Preserve the source numerical/security class.** Default to approximate HE, source model/task, source packing, server-held weights, and the paper's trust assumptions. Do not force these methods through PLLM's exact W8A8/ring32 contract, local clear nonlinearities, or clear KV state to obtain executable rows.

Recommended first implementation dependency, after approval: a parameterized CKKS backend boundary with **real server-side refresh, public evaluation-key import, explicit layout/scale/level types, and client-only final decryption**. CacheMir's CPU backend family, Lattigo, is the strongest common starting candidate for CacheMir/ROSETTA; its exact version and required APIs still require qualification. NEXUS and THOR may need separate source-specific adapters. A unified library is useful only if it preserves their contracts.

The independent [source-fidelity audit](../reproduction-contracts/encrypted-inference.md)
pins supplemental author/library sources and identifies additional source conflicts.
Its original canonical snapshot remains identified by hash. The source-resolution
gates below include those findings; no library or adaptation is approved here.

## 2. Frozen evidence and locator convention

All page numbers below are **one-based PDF pages**, not text-extraction line numbers. Hashes were recomputed from the cached PDFs and matched [prior-art-sources.json](../prior-art-sources.json). The exact bytes, not an unversioned URL, define the reproduction target.

| ID | Frozen source | Identity and primary locator |
| --- | --- | --- |
| N | [NEXUS PDF](../../papers/r21-nexus.pdf), *Secure Transformer Inference Made Non-interactive*, NDSS 2025, 17 pages | [Publisher PDF](https://www.ndss-symposium.org/wp-content/uploads/2025-868-paper.pdf). DOI printed on p.1: `10.14722/ndss.2025.230868`. SHA-256 `c809182345f27f76e21d983d97dffa0d086dbcd17dabc265f0df9fb6ca9b70f2`. |
| T | [THOR PDF](../../papers/thor.pdf), *THOR: Secure Transformer Inference with Homomorphic Encryption*, 15 pages | [ePrint 2024/1881](https://eprint.iacr.org/2024/1881). SHA-256 `ddb57004b56ee8e00beffc4f563fd0921242753fda6c704065d4900b792439cd`. Cached metadata says created/modified 12 September 2025; this supports a dated snapshot, not proof of equality to the current ePrint revision. The prior audit reports a CCS 2025 primary record. |
| C | [CacheMir PDF](../../papers/cachemir.pdf), *Cachemir: Fully Homomorphic Encrypted Inference of Generative Large Language Model with KV Cache*, 16 pages | [arXiv:2602.11470v1](https://arxiv.org/pdf/2602.11470v1), stamped 12 February 2026 on p.1. SHA-256 `67db77545b62a4ec0f6906706b947492e6d1bcd9a37b84e5afc32445b41c98e7`. |
| R | [ROSETTA PDF](../../papers/rosetta.pdf), *ROSETTA: Efficient and Accurate Privacy-Preserving LLM Decoding via Hybrid CKKS/TFHE Evaluation*, 15 pages | [arXiv:2609.16915v1](https://arxiv.org/pdf/2609.16915v1), stamped 15 September 2026 on p.1. SHA-256 `b4c7195d32808881b2da8b3a6876b571d8215bbc02aaafaa78ef327aa27fb4e8`. Cite this preprint even though its running header says CCS '26. |

### Locator index

| Source | Ownership/security | Algorithms/numerics | Execution/resource evidence |
| --- | --- | --- | --- |
| N | §II.A p.2; §III.C p.6; Appendices B/D pp.16–17 | §II.D p.4; §III.B/Algorithm 1 p.5; Algorithm 2 p.7; §IV.B/Equations (3)–(5), Algorithms 3–4 pp.8–9; §V/Fig.7 p.9; Appendix C/Algorithm 5 p.16 | §VI.A–B pp.9–10; Table III p.10; §VI.D/Fig.9 p.11; Tables IV–VI p.12 |
| T | §6.1–6.2 p.10 | §§3–4 pp.3–9; Algorithms 1–2 pp.5,8; §4.4 pp.8–9; §5/Algorithm 3 pp.9–10; §6.2.2 p.11; Appendices B/C pp.14–15 | §7.1 pp.11–12; Table 5/Table 6 p.12; Table 7/§7.3 p.13 |
| C | §2.3 p.4; §2.1 p.3 | §§4–5 pp.4–8; §6/Equation (1)/Fig.7 pp.8–9; Appendices C/D/Algorithm 1 p.15; Appendix E/Table 8 p.16 | §7.1/Table 3–4 p.10; Table 5–6/Fig.9 p.11; Table 7/§7.5/Fig.10 p.12; Appendix F/Table 9 p.16 |
| R | §2.4 p.5 | §2.2.3 p.4; §§3.1–3.3 pp.5–8; Algorithm 1 p.8; §4/Table 2/Figs.8–9 pp.8–10 | §5.1 pp.10–11; Tables 4–5 p.11; Tables 6–8 p.12; Tables 9–10/§§5.3–5.4 p.13 |

Static PLLM baseline: `06b5d6fc0b768be8460e056e2c3c775825e90738`. The working tree contained other work before this review; none was modified. Appendix A pins the six inspected implementation files most relevant to reuse.

## 3. Common contract vocabulary

Use `R` for polynomial ring degree and `S=R/2` for complex CKKS slots in implementation records. NEXUS/CacheMir use `N'` for degree and `N` for slots; THOR/ROSETTA use other conventions. Do not interchange a paper's slot count and ring degree.

Every compiled encrypted value must bind:

```text
source_id, method_revision, model_digest, semantic_operation
scheme, backend_build, parameter_set_id, key_domain_id
shape, physical_slot_or_coefficient_map, padding_and_validity_mask
level, scale, coefficient_moduli, representation_form (NTT/coefficient/SIMD)
declared_input_interval, approximation_id, accumulated_error_budget
owner, lifetime, permitted_next_operations
```

Every approximation must separately freeze input transformation, interval, coefficients and their precision, evaluation order, rescale order, iteration count, normalization/denormalization, and error criterion. A decimal polynomial and its evaluation schedule are distinct parts of the contract. A fitted range observed on calibration data is not a proof that all subsequent private inputs satisfy that range.

Every key manifest must distinguish public encryption keys, relinearization keys, rotation/conjugation keys, bootstrap keys, ring-switch keys, LWE key-switch keys, and blind-rotation keys. Include secret distributions, key relationships, decomposition bases, auxiliary moduli, and allowed recipients. `128-bit security` in a paper is a target, not a complete parameter recipe; evaluate the actual composite key/modulus graph at that target. Do not disable security validation to accept a desired depth.

Separate the papers' claimed model privacy from an implementation security argument. Semantic security of input encryption does not by itself establish circuit privacy of returned evaluated ciphertexts. For NEXUS/CacheMir/ROSETTA, map the client-visible output, evaluation material, decryption error and model-dependent ciphertext distributions to the claimed functionality; determine whether circuit-privacy treatment is required. The current local framing supplies no such argument. This is a qualification gate, not a finding that the original protocols are insecure.

The requested implementations are independent PLLM algorithm implementations over approved general-purpose cryptographic dependencies. Installing a paper's inference package or linking its custom research backend is not independent reproduction. No such dependency has been approved or loaded by this review.

## 4. NEXUS contract

### 4.1 Original functionality and ownership

**Primary evaluated path:** BERT-base, 12 layers, hidden dimension 768, 12 heads, sequence length 128, FFN width 3072, vocabulary 30,522. This is encoder-based inference, not evidence of a sustained decoder service. N p.12 Table IV additionally reports **Llama-3-8B, 32 layers, 8 input tokens, one-word output**, with vocabulary 128,256; Table V reports RTE, SST-2, QNLI accuracy for both models. Preserve that extension rather than claiming NEXUS only evaluated BERT.

The Llama table retains labels `GELU` and `LayerNorm` and displays three full-width Q/K/V projections. These labels alone do not resolve Llama's actual activation, normalization, grouped-query attention, checkpoint, or token-loop implementation. Obtain a source-specific architecture mapping before treating this table as an implementation specification for an unchanged Llama checkpoint.

N §II.A and Appendix D claim privacy of both input and model against a static, computationally bounded semi-honest adversary corrupting either client or server. The server holds weights; the client holds its private input and final decryption capability. This is not malicious-result verification.

**Two key domains matter.** In §III.C, the server first sends compressed weights encrypted under its own key. The client decompresses them, chooses a fresh random matrix `U`, computes encrypted `U W`, and applies the commutable encryption step under the client's key. The server removes its encryption layer to retain `Enc_C(U W)`. Online, the client sends `A-U`; the server evaluates `(A-U)W + Enc_C(UW)`. Appendix B p.16 specifies the RLWE transformation `(A, A s_S + m + e)` to `(A, A s_S + A s_C + m + e + e')`, followed by removal of `A s_S`.

This preprocessing is not ordinary client-key CKKS evaluation and cannot be replaced by public weights or a single generic key-switch without a protocol change. Its storage, client rotations, initial weight transfer, model-change invalidation, and fresh single-use masks must be charged. Appendix D uses uniform masks in the plaintext polynomial ring; sampling arbitrary bounded real masks is not that proof. Resolve representation/scaling of masks and composition with already-encrypted intermediate activations before implementing the offline-online construction throughout a network.

### 4.2 HE parameters, levels, and refresh

N §VI.A pp.9–10 specifies RNS-CKKS with:

| Field | Source value |
| --- | --- |
| Ring/slots | `R=2^16`, `S=2^15` |
| Ciphertext modulus | 1,763 bits; stated 128-bit security |
| Total depth | `L=35` |
| Bootstrap circuit depth | `K=14` |
| Normal computation depth | `L-K=21` |
| Prime sizes | `q0≈2^60`, subsequent `qi≈2^50` |
| Message scale/error-analysis parameters | `Δ≈2^50`, Gaussian `σ=3.2`, secret Hamming weight `h=192` (p.12) |
| Noise management | Scale propagation; bootstrap error analysis references parameter II of paper reference [10] |

The rounded bit counts are not an enumerated RNS chain. Resolve exact main/special/bootstrap primes, sparse-secret security treatment, scale propagation, and refresh keys from a pinned specification. Do not assume ordinary SEAL TC128 defaults reproduce this distribution or bootstrap.

The audited pre-publication NEXUS code uses a different degree/scale context and
disables its library security check. Its exponential uses seven squarings and
divisor 128, unlike the final paper's eight squarings and divisor 256. Preserve
the pinned-paper identity; a code-compatible variant needs a separate decision.
Exact source pins and inspected functions are in audit findings F1–F2.

N §V/Fig.7 chooses refresh where ciphertext count is small, notably after FFN down-projection rather than at the 3072-wide GELU output. Table IV p.12 records softmax `19→3`, GELU `16→2`, LayerNorm `17→1`, refreshes `1→17` and `1→21`. The same table prints a matrix-multiply transition `20→21`; a multiplication cannot itself restore levels. This is a schedule-resolution item, not permission to treat multiplication as refresh. Build an operation-level ledger that accounts for each restoration, packing transition, and omitted operation.

### 4.3 Exact algorithm and unresolved numerical fields

**Linear algebra.** Implement §III.B coefficient packing/compression and Algorithm 1 decompression with monomial shifts, substitutions, and final division by ring degree. Implement §III.C preprocessing separately. Appendix C/Algorithm 5 uses column-packed encrypted `Q,K`, elementwise column products/sums, and successive rotations of `K` to produce diagonals of `QK^T`. The complete column/diagonal/row/coefficient conversion chain, including the attention-value product and transitions between transformer blocks, requires explicit derivation. PLLM's diagonal-slot matvec is not a substitute for these layouts.

**Sign/max/argmax.** N §II.D p.4 normalizes a comparison input to `[-1,1]` with public `Δ=max(|amin|,|amax|)`. It applies `f` and `g`, both degree 9, as `f^df(g^dg(x/Δ))`, with `df=dg=2`, `α=20`. Here powers mean composition. The stated sign approximation excludes the uncertainty band `(-2^-α,2^-α)` except at zero. Exact polynomial coefficients and their precision must be obtained from the cited construction; the PDF does not list them.

Four composed degree-nine polynomials have formal degree 6561, not 36. The
paper's p.12 error discussion uses 36. Independently validate errors through the
actual composition; that discussion cannot supply a certified implementation
tolerance. This finding does not authorize changing the sign approximation.

Algorithms 2–3 use logarithmic slot folding and

```text
max(a,b) = (a+b+(a-b)·SGN(a-b))/2
b_i = SGN(a_i-max(a)) + 1
```

The output is an encrypted selection vector, not full vocabulary logits. Exact ties nominally yield multiple selected positions; near ties interact with approximation error. Freeze tie handling, required separation, padding, and decoding thresholds. The algorithm's `2m<S` packing precondition cannot cover vocabulary 30,522 or 128,256 in one `S=32768` ciphertext. A multi-ciphertext tournament/selection layout is mandatory and must be specified; do not silently reduce vocabulary.

**Softmax.** N p.8 Equation (4) defines normalized exponentials. The following **unnumbered formula** defines the approximation:

```text
z_i = a_i - c, where c is a fixed public replacement for amax
e_i = (1 + z_i/2^8)^(2^8), z_i ≤ 0
y_i = e_i · inverse(sum_j e_j)
```

Use eight squarings as stated by the formula. Algorithm 4's printed inclusive-looking loop and inconsistent variable names must be reconciled against that expression. The source reports average exponential error within `10^-5`; this is not a uniform bound for every non-positive real. The PDF does not freeze `c`, a lower bound on `z`, denominator scaling, Goldschmidt initialization/iteration count, or layer-specific underflow margin. Those fields block a faithful executable profile. Neither `c=1/8` nor three reciprocal iterations follows from the paper.

**GELU.** Equation (3) p.8 uses four segments: zero for `x≤-4`; cubic `P` for `-4<x≤-1.95`; degree-six `Q` for `-1.95<x≤3`; identity for `x>3`. Encrypted sign-derived selectors combine segments. The paper says inputs observed in `[-8,8]`, with average approximation error within `10^-4`; Table III separately reports measured HE GELU error `7.7e-4`. The two errors are different evidence. `P/Q` coefficients are delegated to references, not printed here. Resolve coefficients and boundary behavior rather than substituting SiLU or a generic polynomial.

**LayerNorm.** Equation (5) p.9 sets `z_i=n a_i-sum_j a_j` and computes `γ sqrt(n) z_i / sqrt(sum_j z_j^2)+β`, using QuickSum and Newton inverse-square-root with a suitable initial value. The printed formula omits the checkpoint epsilon term. Resolve epsilon, normalization of the inverse-square-root operand, seed, iteration count, and zero-variance behavior before claiming model equivalence. Do not silently fix it using THOR's different approximation.

### 4.4 State and output boundary

No specified persistent encrypted KV append/read/capacity contract accompanies the one-output measurements. The amortization variable in §III.C is a number of matrix inputs/responses, not proof of an implemented encrypted token-feedback loop. Initial token lookup/embedding ownership also needs an explicit source mapping for the selected task.

Faithful default: first reproduce one original BERT request and its encrypted Argmax output. A repeated autoregressive service must separately approve how encrypted selection feeds private embedding, whether the client participates per token, which output is revealed, and where KV lives. Returning logits for arbitrary client sampling changes NEXUS's intended output leakage. Client decryption/re-encryption at every layer changes non-interactivity.

### 4.5 Source hardware and benchmark geometry

N §VI.A uses C++/SEAL plus **FHE-MP-CNN for bootstrapping**, HEXL on Intel CPUs, and Phantom on GPU. The research bootstrap dependency may not be linked under the present instruction; independently implement its required algorithm or qualify an approved general-purpose implementation.

- CPU: two 3.70 GHz Intel Xeon instances, 128 GB RAM, 32 threads for system measurements; matmul microbenchmark uses one thread.
- GPU: four Tesla A100 GPUs, each 40 GB. Ten-run averages.
- LAN: 3 Gbps, 0.8 ms RTT. WAN: 100/200 Mbps crossed with 40/80 ms RTT.
- Matmul: `128×768` by `768×768`; preprocessing included in the reported amortization, with input counts including 256 and 1024.
- Fig.9 p.11: 128-token BERT inputs; end-to-end performance amortized over **128 inputs**. Table IV p.12 instead says **32 batched inputs**, BERT 128 tokens/Llama 8 tokens. Keep these cohorts distinct.
- Table IV reports 37.34 s GPU/BERT and 51.84 s GPU/Llama per amortized input. These are not unbatched admission-to-first-token latencies or sustained KV-cache throughput.

**Minimal faithful slice:** after numerical/source-resolution gates, reproduce coefficient compression/decompression and dual-key preprocessing for one source-shaped linear projection; then a complete BERT attention-plus-FFN block with real layout conversions and all required bootstraps; finally the original full BERT one-output path. The first operator is a dependency test, not the NEXUS reproduction. Quality validation must include numeric outputs at encrypted boundaries, full Argmax behavior, and the named task data.

## 5. THOR contract

### 5.1 Original task, ownership, and HE parameters

T §2.2 p.3 focuses on the **encoder**. §6.2 and §7 evaluate BERT-base with `n=128`, `d=768`, 12 encoder blocks, 12 heads of width 64, FFN width `4d`, then first-token pooler, Tanh, and classifier. The client computes token embeddings using embedding tables, encrypts them, and receives encrypted inference results. There is no evaluated autoregressive KV-cache loop.

T §6.1 p.10 states input privacy against an honest-but-curious server and requires decrypted results to remain with the secret-key owner. Do not infer model confidentiality from this statement, particularly when embedding tables reside at the client. Do not expose decrypted numeric diagnostics to the server as part of the protocol. No malicious-result verification is claimed.

T §7.1 pp.11–12 uses RNS-CKKS in **Liberate.FHE**, `R=2^16`, `S=2^15`, a modulus selected for 128-bit security, and **13 usable multiplicative levels before bootstrap**. It does not enumerate an exact modulus chain, CKKS scale, secret/error distributions, bootstrap depth, or evaluation-key decomposition. Do not import CacheMir's `1763/53/41/K=15` configuration merely because it cites THOR. Those are separate frozen sources.

The pinned author repository describes a PyArmor-protected bootstrap and separate
external model/key artifacts. Its readable nonlinear code supplies task/layer
profiles that still require reconciliation with the PDF. Importing that bundled
bootstrap would not meet the independent implementation requirement. Audit F3
records the commit, functions and unresolved profile relationships.

### 5.2 Packing and kernel contract

THOR's contribution is a complete matrix-layout strategy, not a softmax-only variant:

1. §§3–4 define upper/lower diagonal vectors and **multi-diagonal batched encoding**. Interlace heads/matrices, then concatenate `c` diagonals into a ciphertext.
2. Algorithm 1 p.5 partitions plaintext weights into blocks, computes PC-MM in parallel, and aligns/sums output blocks with masked internal rotations. The inference implementation uses transposed products such as `Q^T=W_Q^T X^T` to rotate plaintext weights instead of encrypted inputs where possible.
3. Algorithm 2 p.8 implements lower-diagonal CC-MM with BSGS, replicated diagonals, masks, and delayed rotations of partial sums.
4. §4.4 pp.8–9 includes **lazy rescaling, lazy relinearization, and complexification/conjugation**. The current PLLM arithmetic wrapper eagerly rescales/relinearizes every product and only handles its bounded real-slot schedule; it cannot express these kernel promises unchanged.
5. Appendix B pp.14–15 supplies replication and transpose procedures. §6.2.2 p.11 carries layouts through Q/K/V, scores, softmax, values, output projection, normalization, FFN, and classifier.

For the source BERT geometry, §4.5.2 p.9 chooses `c=16`, the largest power of two no greater than `S/(nH)`. Preserve actual padding, head ordering, and validity masks; do not assume all 32,768 slots are useful. Plain diagonal matvec plus ad hoc conversion is insufficient to reproduce the proposed operation counts.

### 5.3 Numerical contract

**Softmax, §5.1/Algorithm 3 pp.9–10:** for values in `[a,b]`, subtract the public midpoint `(a+b)/2`; let `M=b-a`. Choose scaling factors `δ0,δ1,δ2` so the initial exponential sees approximately `[-δ0/2,δ0/2]`.

```text
z_i = AExp(x_i/(δ1·δ2))^δ1
y_i = z_i · AInv(sum_j z_j)
repeat log2(δ2) times:
    z_i = y_i^2
    y_i = z_i · AInv(sum_j z_j)
```

T p.12 uses actual MRPC attention scores in `[-44.19,50.25]`, so midpoint `3.03`, centered half-range `47.22`, with `δ0=11.805`, `δ1=2`, `δ2=4`. Initial AExp interval is consequently approximately `[-5.9025,5.9025]`. This is a published profile, not a guarantee for arbitrary checkpoints. Factors were selected by greedy search on training data. Three normalizations require three bootstraps in Table 5; removing them changes the evaluated algorithm.

Appendix C p.15 gives the degree-15 AExp coefficients, in ascending powers:

```text
0.99999, 0.99999, 0.50002, 0.16667,
0.041653, 0.0083316, 0.0013920, 0.00019871,
0.000024471, 0.0000027286, 0.00000029466, 0.000000026438,
0.0000000014896, 0.00000000012104, 0.000000000020777,
0.0000000000013376
```

Freeze the printed coefficients as one candidate specification. Determine whether more precise author coefficients are necessary before asserting evaluated-build equivalence; regenerating a fit is a numerical adaptation.

**Reciprocal:** Appendix C starts `a0=x∈(0,1), b0=1`, then

```text
a[i+1] = k[i]·a[i]·(2-k[i]·a[i])
b[i+1] = k[i]·b[i]·(2-k[i]·a[i])
inverse(x) ≈ b[T]
k = (2.00, 2.00, 1.98, 1.94, 1.79, 1.46, 1.12,
     1.98, 1.94, 1.79, 1.46, 1.12, 1.01)
```

Preserve the printed repeated factors rather than silently replacing them. §5.1 describes adaptive successive over-relaxation with `k_i=2/(1+ε_i)` for the current minimum. Resolve how the printed sequence, lower bound, per-normalization scaling, and stopping count are selected for each layer. In particular, a sum of exponentials is not automatically in `(0,1)`.

**Inverse square root:** Appendix C starts `a0=x∈(0,1), b0=1` and uses

```text
a[i+1] = k[i]·a[i]·(3-k[i]·a[i])^2/4
b[i+1] = sqrt(k[i])·b[i]·(3-k[i]·a[i])/2
k = (2.6374, 2.1722, 1.5135, 1.0907)
```

LayerNorm includes mean, variance, checkpoint epsilon, gamma and beta (§5.2 p.10). Freeze the input normalization and inverse normalization factors; the listed iteration coefficients alone do not define LayerNorm.

**GELU/Tanh:** use the composition `f2(f1(x))`, degrees 31 and 27 for GELU, evaluated with Paterson–Stockmeyer at reported depth 11 and 39 multiplications; Tanh uses two degree-15 polynomials. Appendix C p.15 prints all four coefficient lists. Preserve those lists, their ordering, input/output scaling, and evaluation schedule. The reviewed text does not clearly freeze all fitting domains or normalization transforms; this must be resolved before code. Polynomial degree alone is not the contract.

### 5.4 Source hardware, geometry, and slice

- Single NVIDIA A100 GPU, Liberate.FHE; GPU memory size and host CPU/RAM are not fixed in the inspected setup text.
- PC-MM comparison: `768×768` by `768×128`; CC-MM: 12 parallel `64×128` by `128×128` products. Ciphertext level is varied; comparisons cannot be transferred between levels.
- Softmax microbenchmark: 256 rows of width 128 per ciphertext. Table 5 reports 38 multiplications, 21 rotations, 3 bootstraps, 3.704 s total/0.0145 s amortized; the operation comparison excludes initial exponentiation. Its negative signed precision entry must not be confused with ROSETTA's positive `-log2(RMSE)` convention.
- Full BERT result: Table 6 total 602.26 s, including 337.86 s bootstrapping. This is the paper's one-input, 128-token result, not an autoregressive token rate.
- Fine-tuning setup: batch 16, learning rate `5e-5`, five epochs. Quality tasks: MRPC, RTE, SST-2; Table 7 distinguishes unchanged plaintext, successive approximations, and encrypted evaluation.

**Minimal faithful slice:** first compose THOR PC-MM → transpose/replication → CC-MM → source-range softmax with actual bootstrap → attention-value CC-MM → output projection, preserving multi-diagonal layout. Then one entire BERT encoder block and finally all 12 blocks plus pooler/classifier. A standalone square-normalize operator does not qualify. Reproduce encoder quality before requesting an autoregressive extension. Causal masking, RoPE, RMSNorm, SiLU/SwiGLU, GQA and encrypted KV are substantial additions for a modern decoder and need approval.

An approved GPT-2 extension instead needs that checkpoint's learned positions,
pre-norm/final norm, activation, causal attention, head and persistent state; it
does not inherently need RoPE, RMSNorm or GQA. The independently pinned standard
GPT-2 configuration has 1024 positions. ROSETTA's 2048/8192-position GPT2-base cost
shapes do not establish unchanged standard-model execution. Keep any common
standard GPT-2 workload within its actual position bound, or request a separately
specified model extension. Audit F7 supplies the configuration identity.

## 6. CacheMir contract

### 6.1 Task, threat model, and crypto profile

CacheMir directly targets **decoder-only autoregressive inference with encrypted KV cache**. C Table 3 p.10 lists GPT2-base `(heads=12,d=768,layers=12)`, TinyLlama-1.1B `(16,2048,22)`, and Llama-3-8B `(32,4096,32)`. Those are the printed geometry values; exact model revisions and query-head/KV-head mapping still need checkpoint-level binding. Do not silently expand grouped-query KV into a different architecture or assume the table fully specifies GQA.

C §2.3 p.4 assumes a semi-honest client and server, server-held proprietary plaintext weights, private client input, and confidentiality of each party's inputs. Server state remains encrypted under client-controlled key material. §2.1 p.3 places encryption/decryption at protocol boundaries and treats bootstrap as a black box. No active-integrity mechanism is specified.

C §7.1 p.10 specifies:

| Field | Value |
| --- | --- |
| Backend | Lattigo CPU; customized Phantom GPU |
| Ring/slots/security | `R=2^16`, `S=2^15`, 1,763-bit ciphertext modulus, stated 128-bit security |
| Usable maximum level | `L=13` |
| Bootstrap | `K=15`: CtS-first, CtS depth 4, StC depth 3, approximate modular reduction depth 8 |
| Sparse secret | Hamming weight 192, with **sparse secret encapsulation** to preserve the stated security target |
| Prime sizes | `log2(q0)≈53`; `log2(qi)≈41` for `i≥1` |

Here `L` is usable level and is not NEXUS's total `L=35`. Resolve exact main/auxiliary prime lists, message scale, encapsulation keys, bootstrap input range/error, and backend version. Rounded bit counts and a sparse Hamming weight alone cannot recreate the source security profile.

### 6.2 Packing and persistent state

**VMM, §§4.1–4.2 pp.5–6:** interleaved replicated packing fills idle slots with permuted copies of a single activation vector. With square width `d`, set `t=S/d` when integral. Preprocess using `log2(t)` rotate/add steps, multiply packed weight diagonals, reduce interleaved results, then extract valid slots. BSGS reduces rotations to `O(2 log(S/d)+2 sqrt(d²/S))`. Fuse output extraction into the next operation to preserve one multiplication level for VMM.

For `(S,d)=(32768,4096)`, Table 2 p.6 reports 512 plaintext–ciphertext multiplications, 52 rotations, depth one. This is a concrete fidelity target for the layout/cost trace. Appendix C/Algorithm 1 p.15 extends rectangular up/down projections with padding and reordered weight diagonals. Its printed variable/index definitions are not fully consistent: `r_o` is defined in the precondition and reassigned in line 13, while the explanatory product relation also constrains it. Resolve the algebra and indexing against the figure and dense multiplication; do not transliterate undefined indices into production code.

**K cache, §5.1 pp.6–7:** pack `t` tokens per ciphertext, interleaving corresponding K coordinates. For one head, ciphertext count is `ceil(n'/t)`. Replicate Q into the matching interleaved layout, multiply with each cache ciphertext, rotate/reduce feature coordinates, mask duplicates, and compact attention scores. On append:

- If `n' mod t != 0`, generate the new K directly at the remaining position in the last ciphertext by changing the VMM output's rotation/mask placement; add into empty slots.
- At `n' mod t == 0`, append a new ciphertext. Do not repack or decrypt all earlier keys.

**V cache, §5.2 pp.7–8:** use the paper's distributed interleaved layout compatible with contiguous attention scores and an interleaved attention output. For one head, every S-token group needs `d` ciphertexts; multi-head packing uses `d/H` ciphertexts per `S/H` tokens. Append uses customized V placement and masked plaintext–ciphertext products, with the validity mask fused into V generation. Attention-value multiplication has its own BSGS/reduction schedule; it is not K's layout with the labels exchanged.

**Multi-head and position, §5.3 p.8/Appendix D p.15:** interleave heads in blocks of `t` elements; reorder `W_Q/W_K/W_O` accordingly. One attention ciphertext can hold `S/H` token scores: 1024 for the printed Llama geometry. RoPE uses three masked trigonometric plaintexts, two rotations and depth one, fusing slot extraction. The physical meaning of Appendix D's `±1` rotations must be reconciled with the interleaved stride; freeze logical and physical rotations explicitly.

**Prefill, §5.4 p.8:** generate compatible cache with the same layouts, additional within-group rotations for cross-token QK interactions, and SIMD masks for V. The paper calls this prefill suboptimal and amortizes it across generated tokens. Replacing it with a clear host-generated cache or importing another method's prefill requires approval.

An executable state descriptor must include layer, position, active token count, allocated token groups, validity masks, K/V layouts, key domain, scale, and separate level metadata for each retained ciphertext. Queries should consume derived copies at compatible levels; repeatedly modulus-switching retained cache in place would exhaust it across tokens. The PDF does not fully freeze retained-cache level/refresh policy over an arbitrarily long session. Prove append alignment, nonmutation of old tokens, and refresh needs before a repeated-token claim.

### 6.3 Approximation and bootstrap-placement contract

**Do not equate CacheMir's nonlinearities with unchanged THOR.** C Appendix E p.16 explicitly says THOR's original parameter configurations do not apply to its larger models. It profiles inputs, selects an initial scaled exponential `c·exp(kx)`, normalizes inverse operands into `(0,1)`, and uses **standard Goldschmidt reciprocal rather than THOR's aSOR** for predictable depth. SiLU/GELU use composite polynomials; normalization uses inverse-square-root; §6 p.9 describes decomposition of these operators.

Exact per-layer `c,k`, shift, inverse/normalization scales, coefficients, iteration counts, clipping policy, error tolerances, and sampling/calibration corpus are not printed as a complete executable specification. Preserve these as source-resolution gates. Adopting the current `[-1/8,1/8]` profile, or reusing THOR's printed BERT parameters, is not faithful.

C Table 8 p.16 gives illustrative Llama ranges, not complete all-layer bounds:

| Module | Layer/index 1 | 6 | 31 |
| --- | --- | --- | --- |
| Softmax input min/max | `[-24.38,23.12]` | `[-18.5,10.5]` | `[-17.63,67.50]` |
| Norm input min/max | `[-0.39,0.37]` | `[-31.75,8.31]` | `[-48.25,15.81]` |
| SiLU input min/max | `[-2.94,3.97]` | `[-7.22,6.13]` | `[-31.63,24.50]` |

C Table 6 p.11 reports approximate depths/precision for its THOR-derived configuration: softmax depth 43, precision 14.99 bits; norm depth 16, precision 10.81 bits; SiLU depth 10, precision 27.54 bits. Precision is `-log2(RMSE)`. Depth 43 with usable level 13 explicitly requires refresh within the computation; it does not justify a no-bootstrap substitute. Table 7 p.12 evaluates the **approximated model**, not proof of encrypted complete-request accuracy.

**Placement, §6 pp.8–9:** construct level vertices `v[i,x]`, feasible edges only when `x≥ell(i)`, and costs

```text
w[i,x,y] = t_i(x) + 1[x-ell(i)<y] · t_boot(y)
```

The source assumes level drop cost negligible, computation before optional bootstrap/drop, monotonic costs, and bootstrap cost insensitive to input level. Prune dominated transitions, solve repeated blocks hierarchically, handle residual shortcuts recursively, and expose intra-module boundaries when the live ciphertext count stays small. Group polynomial/Goldschmidt subgraphs whose many temporaries make internal refresh costly. A greedy "refresh when needed to maximum level" policy is an ablation, not CacheMir's optimizer.

Freeze the measured cost table and its backend/hardware identity as compiler inputs. Prove any retained pruning assumptions for the selected backend. Do not transplant the source's GPU costs into a CPU schedule and describe the result as its optimum.

The printed pruning condition needs a correction before use. At depth one,
input level one and output level two, compute-then-refresh is feasible, but the
printed rule can remove it in favor of an infeasible input-level-zero predecessor.
Use the complete feasible DAG as an oracle and independently prove each guarded
dominance reduction. Test nonmonotone measured costs as well as the source's
monotonicity assumption. Audit F4 records the exact counterexample; it is not a
benchmark result or permission to claim the printed optimizer was reproduced.

### 6.4 Output gaps, hardware, and minimum slice

The paper describes KV append and decoder blocks but does not fully specify initial private token lookup, complete output head placement, private argmax/sampler, token re-embedding, EOS, cancellation, continuation, or replay. Its statement that the client encrypts/decrypts only at boundaries cannot be silently replaced with per-layer client refresh. Decide whether repeated-token functionality remains entirely encrypted or uses an approved per-token client boundary; freeze output leakage before implementation.

Source execution scope:

- Intel Xeon Platinum 8558P, maximum frequency 4 GHz, 192 threads; NVIDIA A100 80 GB. Host RAM capacity and precise customized-library commits are not stated in §7.1.
- Module latencies are measured and aggregated to **estimate** end-to-end runtime. Baselines are estimated from theoretical operation counts and measured primitive costs. This is not a matched observed full-request experiment.
- Method batch size one. VMM widths 768/2048/4096, levels 1/4/13; attention comparisons fix module input level four.
- Table 5 p.11: one Llama decoder block at `(n0,n')=(64,512)`, CPU 477.81 s/GPU 3.02 s, including amortized prefill and bootstrap rows. The 1.61-minute token claim scales across the model; it is not a new observed serving trace.
- Other geometry: `(128,256)`, `(64,512)`, `(16,1024)` and the three listed architectures. Quality: Llama-3-8B, MRPC and WikiText-2, without downstream fine-tuning.
- §7.5/Fig.10 p.12: roughly 17 GB encrypted KV at 1024 Llama tokens; encoded weight plaintexts reach hundreds of GB (the plot labels Llama weights 694 GB). These are source footprint figures, not whole-process RSS or proof that an 80 GB GPU holds all data simultaneously. Streaming/offload and host/device residency must be specified and charged.

**Minimal faithful slice:** one original-model decoder block, encrypted prefill → cache initialization → repeated attention/FFN updates using source VMM, nonlinear schedules, and optimized refresh. Exercise both partially filled and newly allocated K groups and the `S/H` attention/V-group boundary. A small synthetic shape can test indexing, but must be followed by full source geometry and real levels. Full-method completion then needs all blocks plus the separately resolved lookup/head/token loop. Do not substitute speculative or multi-token prediction; that is not this paper's contribution.

## 7. ROSETTA contract

### 7.1 Task, ownership, and parameters

R §§2.3–2.4 pp.4–5 targets **one-token decode with retained KV**, private input, confidential server-held weights, public architecture, and honest-but-curious client/server. It uses CKKS for arithmetic and TFHE-style scalar/LUT operations selectively. All required secret keys must remain with the client; the server receives evaluation/conversion material. The paper does not fully enumerate the cross-domain key graph, which is a mandatory implementation gate. No malicious-result verification is established.

R §5.1 pp.10–11 freezes CKKS `R=2^16`, `S=2^15`, 1,763-bit modulus, usable `L=13`, bootstrap depth 15, CtS/StC depths 4/3, depth-eight modular reduction, sparse secret Hamming weight 192 with encapsulation, and approximate RNS prime sizes 53 then 41 bits. TFHE uses **LWE dimension 1024, blind-rotation ring degree 2048, and 53-bit ciphertext modulus**. All are claimed 128-bit secure. Exact primes, errors, gadget decomposition, key distributions/relationships, failure probabilities and fixed-point scales are still required.

In particular, extracting from the 2048-dimensional small ring does not by itself explain the final 1024-dimensional LWE key domain. Freeze the necessary key switching/secret relation explicitly; do not treat raw RLWE coefficient extraction as complete CKKS–TFHE conversion.

### 7.2 Segmented LUT algorithm: preserve the encrypted accumulator

R §§3.1–3.2 pp.5–7 and Algorithm 1 p.8 define:

1. Divide `[tau0,tauNseg]` into nonuniform segments; each has `Nbr` equally spaced local intervals. Fit a linear function `a_si x+b_si` in each interval.
2. Compute encrypted segment index `Sidx=sum_{j=1}^{Nseg-1} 1[x≥tau_j]` using the referenced **HomComp** high-precision comparison (multiple PBS operations internally).
3. Construct coefficient-domain encrypted candidates for the local interval mapping, then select by encrypted `Sidx`; perform the specified modulus-switch/rounding to obtain `I=floor((x-tau_Sidx)·Nbr/(tau_{Sidx+1}-tau_Sidx))`.
4. For every segment, construct **encrypted test-polynomial coefficients** `a_si x+b_si` using plaintext–RLWE multiplication, blind-rotate by encrypted `I`, and trace to retain only its constant term.
5. Shift these constants by monomials `X^s`, sum, blind-rotate by encrypted `Sidx`, and trace again. The result remains an RLWE ciphertext holding the selected linear evaluation.

This is **online construction followed by encrypted selection**. It avoids encrypted fixed-point LWE×LWE multiplication. A conventional PBS that only accepts a public plaintext LUT cannot substitute for blind rotation of the encrypted accumulator. Nor can a garbled lookup with an interactive client substitute for TFHE.

The paper's complete lookup count is `Nseg-1` HomComp calls, `Nseg+1` plaintext–ciphertext multiplies, `Nseg+2` blind rotations, and `Nseg+1` traces, excluding blind rotations internal to HomComp and conversion. At `Nseg=4`, record 3/5/6/5 respectively. Trace means Galois automorphisms/additions plus modular division by ring degree, not CKKS real-number division.

### 7.3 Conversion pipeline and key dependencies

R §3.3 pp.7–8 specifies the following round trip:

```text
CKKS SIMD
  → SlotToCoeff
  → RingSwitch from degree 65536 to 2048
  → coefficient-domain RLWE x in constant term
  → ModSwitch and ExtractLWE for the companion scalar input
  → segmented lookup using both RLWE and LWE forms
  → RingSwitch back to degree 65536
  → ModRecover = ModRaise + CoeffToSlot + EvalMod
  → CKKS SIMD at a chosen target level
```

For multiple values, perform StC once; split into small-ring ciphertexts; move each selected coefficient to constant position by a monomial and trace; evaluate; shift results back and recombine; perform one final full-ring recovery. Sharing StC/recovery across heads is central to the cost model.

Native dependencies include raw coefficient-domain RLWE arithmetic, bidirectional ring switching, correct coefficient↔slot transforms, scalar extraction/key switching, PBS/BlindRot accepting encrypted accumulators, high-precision HomComp, trace, ModRaise, EvalMod, and target-level recovery. They require compatible moduli, message encodings, key domains, NTT conventions, and noise analysis. Separate SEAL and TFHE ciphertext serialization is not an interoperable conversion protocol.

Lattigo v6.1.1 exposes a lower-level `BlindRotateCore` with a ciphertext
accumulator; its higher-level `Evaluate` starts from a plaintext polynomial.
Only the former is a candidate primitive for independent composition. Its
modulus-switch conventions, key graph, ring switching and recovery still need
qualification. Audit F6 records the exact commit and API evidence.

R Fig.7 shows level-one RLWE and level-zero LWE stages; the optimizer represents a TFHE candidate by a level-zero node. Do not interpret this abstraction as StC having zero depth or being legal from any physical CKKS state. The backend must establish legal entry levels and transforms, then the compiler must enforce them.

### 7.4 Approximation, scaling, and selection

**Source defaults:** `Nseg=4`, `Nbr=2048`; logarithmically spaced segment boundaries for reciprocal/inverse-square-root, uniformly spaced for SiLU/GELU (§5.1 p.11). Each segment contains a fine linear LUT; four segments do not mean four total interpolation intervals.

The paper gives the mathematical floor operation and worked example `[0.1,2.0]`, two segments, four entries, `x=0.22`, producing `I=1` and approximately `4.571`. It does not fully specify a quantized, noise-tolerant floor at interval edges, endpoint handling at `tau_end` (which would nominally give `I=Nbr`), or all fixed-point coefficient scales. Resolve these before code. A modulus switch usually entails rounding and noise, so the displayed real-valued floor is a functional target, not automatically the behavior of an arbitrary library API.

Freeze exact boundaries and coefficient-generation convention (e.g. which interpolation/fitting points), coefficient precision, comparison encoding, interval guard policy, signed negacyclic/torus encoding, modulus-switch rounding, final reconstruction scale, and uncertainty budget for both comparisons and table selection. Near-boundary values must receive defined behavior; do not introduce an unapproved clamp. The text's "linear interpolation" and rounded worked coefficients are insufficient to bind an evaluated-build numeric identity without those details.

**Softmax:** R p.5 follows max-free normalization/squaring: start with a scaled exponential and repeatedly normalize squared probabilities using `Softmax(2x)_i=y_i²/sum_j y_j²`. Keep vector exponentials in CKKS when selected; scalar reciprocals per head can go through TFHE. Exact exponential coefficients, rescaling factors, iteration counts, and per-layer ranges still require freezing. R §5.1 says its CacheMir baseline uses THOR nonlinearities "without modification"; C Appendix E explicitly changes reciprocal strategy and calibration. Bind the actual version-specific schedule rather than reconciling those statements by assumption.

**Scheme selection, §4 pp.8–10:** keep add/multiply/scalar affine/reduction in CKKS. Decompose composites such as RMSNorm; only exp, inv, invsqrt, SiLU/GELU/ReLU are candidate non-arithmetic primitives. Inject TFHE nodes into CacheMir's level DAG and optimize jointly with the CKKS level assignment:

```text
w_T[i,0,y] = t_StC + t_TFHE(number_of_values, input_range)
             + t_ModRecover(y)
```

The paper equates StC plus recovery cost to CKKS bootstrap cost. Verify that accounting includes actual ring switches, key switches, extraction, traces, recombination, transfers, and allocation overhead on the chosen backend. Charge each once; do not set conversion cost to zero or assume the paper's approximation holds for another implementation. Freeze output-level choices and a new calibrated cost graph per hardware/backend profile.

A hard-coded all-TFHE nonlinear layer or a manually selected reciprocal does not reproduce the scheme-aware compiler, although either can be a dependency test. The core reproduction needs automatic placement with a recorded optimal path and equivalent arithmetic across that path.

### 7.5 Source benchmark geometry and output scope

- CPU: Intel Xeon Gold 6240R, 2.40 GHz, 48 threads, 384 GB RAM; Go 1.24, Lattigo v6. GPU: customized Phantom, CUDA 12.4, NVIDIA A100 80 GB.
- Lattigo does not expose a direct LWE API in the authors' implementation; they manage LWE with RLWE ciphertexts and build on blind rotation. This strongly favors qualifying low-level Lattigo primitives rather than assuming an off-the-shelf high-level TFHE API is sufficient.
- Source measures modules to estimate end-to-end runtime (§5.1). Table 7/§5.3 concerns **one decode iteration, excluding prefill**, using identical linear implementations across methods. Models: the same printed GPT2/TinyLlama/Llama geometries as CacheMir; context lengths 512, 2048, 8192. Reported GPU estimates span 1.31–6.79 minutes/token. These are not complete request latency or observed long-run throughput.
- Softmax: lengths 512/1024/2048/4096/8192; truncated normal inputs on `[-M,M]`, variance `M²/3`, `M∈{8,16,32}`; separate wide-range experiment `[-128,128]`.
- Norm: dimensions 768/2048/4096; per-group variance ranges `[0.2,150]` and `[1,8192]`.
- Primitive tests: reciprocal/inverse-square-root on `[0.01,10]` and `[0.01,100]`; SiLU/GELU on `[-20,20]`. Table 9 reciprocal precision is around 12.5–12.7 bits, not a universal 17–21-bit guarantee. Record exact profile-specific results and metric `-log2(RMSE)`.
- Calibration uses a **fixed public prefix** on Llama-3-8B across five datasets (§5.2 p.13). Its exact bytes/positioning are not specified here. Attention spans below 64 cover 31/32 layers; the last reaches 105.5–109.2. Norm variance bound 8192 covers only about 72% of profiled operations. These ranges are not complete model-wide admission bounds.
- §5.4/Table 8 evaluates **ciphertext-aligned plaintext** PPL on Llama-3-8B, Qwen2-7B, Mistral-7B and DeepSeek-R1-Distill-LLaMA-8B; WikiText-2/103, LAMBADA, GSM8K, ShareGPT. It is not encrypted whole-model PPL. The reported maximum relative increase of 4.3% cannot automatically become a PLLM acceptance threshold.

KV and token state follow the decode setting; the paper does not fully bind a complete private lookup/head/sampler/feedback service or a distinct persistent-cache implementation. Importing independently implemented CacheMir linear/cache algorithms is reasonable only as an explicitly recorded composition, with the inherited state/numeric gates resolved.

**Minimal faithful slice:** first an actual CKKS → small RLWE/LWE → segmented reciprocal → recovered CKKS round trip, with four segments, all boundaries, declared scales, and client-only decryption. Then multiple head reciprocals sharing StC/recovery; then complete source-range softmax and RMSNorm selected by the level DAG; then one full decode block using source-compatible encrypted KV. Whole-method completion requires all blocks and the approved full request boundary.

## 8. Reusable PLLM primitives and missing implementation layers

### 8.1 Reuse boundary

| Existing path | Reusable material | Limits that prevent method equivalence |
| --- | --- | --- |
| [ckks_layout.rs](../../crates/pllm-core/src/ckks_layout.rs), [Python binding](../../crates/pllm-python/src/ckks.rs) | Shape/parameter admission pattern, bounded diagonal generation, allocation accounting structure | Ordinary width ≤32, depth ≤8, 512 MiB estimated allocation cap. Softmax separately admits width ≤32, depth 14, `R=32768`, 660 coefficient bits, scale bits 40, 32 live ciphertexts, 2 GiB estimate. None is a source profile; estimates are not RSS. |
| [ckks_reference.py](../../python/pllm/runtime/ckks_reference.py) | Client-only decryptor pattern; public provider evaluator; parameter/frame binding; sequence checks; rotation/add/multiply primitives; counters | Fixed scale `2^40`; eager rescale/relinearize; input bounds `[-1,1]`; bounded keys/queries; no bootstrap, complexification, raw cross-scheme conversion or persistent encrypted state. Local contexts are not distributed authenticated roles. |
| [ckks_softmax.py](../../python/pllm/runtime/ckks_softmax.py) | THOR's printed exponential coefficient list; arithmetic-expression testing scaffolding | Narrow `±0.125` input; NEXUS shift `0.125` and three unrelaxed reciprocal steps; THOR `δ1=1,δ2=2` and two unrelaxed steps. Sequential one-step row rotations differ from source folding/packing. These are existing adaptations, not source-ready schedules. |
| [coefficient_repacking.rs](../../crates/pllm-core/src/coefficient_repacking.rs), [coefficient_repacking.py](../../python/pllm/runtime/coefficient_repacking.py) | Independent negacyclic sample-extraction/embedding algebra, modular power-of-two division, trace/interleave reasoning and test oracles | BFV degree 4096, ≤2 RNS limbs, size-two coefficient ciphertexts; Python reconstruction is bound to particular SEAL BFV frames and scale one. Not CKKS StC/CtS, ring switching, TFHE PBS, key-domain switching or ModRecover. Port algebra with new proof/tests; do not reuse serialized BFV frames as CKKS. |
| [he_layer_feasibility.py](../../python/pllm/runtime/he_layer_feasibility.py) | Semantic dependency/coverage checks and explicit unknown-work reporting | Deliberately non-executable optimistic depth gate. It excludes real norms/softmax/refresh/KV costs and retains client-side head/token selection. It cannot admit these methods. |
| [planned_methods.json](../../python/pllm/components/planned_methods.json), [protocols](../../python/pllm/protocols/__init__.py), [state](../../python/pllm/state/__init__.py), [passes](../../python/pllm/passes/__init__.py) | Names/interfaces for source-specific protocols, EncryptedKvState, NexusCkksBootstrap, RosettaSchemeConversion and RosettaCkksTfheAssignment | Pending declarations, not executable components. |

The HE extra currently pins `tenseal==0.3.17` in [pyproject.toml](../../pyproject.toml). No inspected native/compiler implementation supplies full HE bootstrap or the ROSETTA bridge. Current generic garbled/FSS nonlinear mechanisms change interaction, representations, trust, and ownership and are not interchangeable with these FHE protocols.

### 8.2 Required native/backend contracts

1. A verified source parameter loader, including auxiliary primes, sparse-secret encapsulation, per-operation scales, precision targets, and evaluation-key inventory.
2. `bootstrap(ciphertext, target_level, target_scale, precision_profile)` implemented entirely at the evaluator, with legal entry bounds, actual restored depth and measured error. Server-side decryption is prohibited by the source ownership contract.
3. Deferred multiply/rescale/relinearize, conjugation, rotations with hoisting where promised, coefficient arithmetic, and memory-managed temporaries. Source kernel traces must distinguish tensor products, relinearizations, key switches and rescale operations.
4. Layout transforms covering each paper's physical slots, head ordering, partial groups, padding, extraction masks, transpose, replication and complex packing.
5. For NEXUS, dual-key commutable encryption and independently implemented coefficient compression/decompression, including uniform-ring masking and mask-use retirement.
6. For ROSETTA, all primitives in §7.3 plus encrypted-accumulator blind rotation and HomComp. A generic `convert_scheme` placeholder is not an implementation.

### 8.3 Required compiler/runtime contracts

- **Compiler:** source model graph (encoder and decoder are distinct); scheme/key/layout/scale/level typed values; range/approximation registry; real transform costs; residual branch alignment; source-specific bootstrap placement; ROSETTA joint scheme/level DAG; serialized schedule digest. No implicit decrypt fallback or automatic clipping.
- **Runtime:** separate client/provider key custody; parameter/evaluation-key delivery; actual network framing; server encrypted model-state sessions; cancellation/failure retirement; persistent cache ownership; approved output/feedback boundary; honest-but-curious security scope. Transport protection is needed for the selected deployment but does not add malicious-computation verification.
- **Admission:** checked dimensions and exact key/weight/cache/scratch counts before allocation; declared host/device placement; bounded lifetime and concurrency; no reuse of another session's cache or preprocessing mask. Source geometry must remain visible when a local resource limit rejects it.

## 9. Backend/library decisions requiring approval

| Decision | Faithful recommendation | Qualification required before implementation |
| --- | --- | --- |
| B1: leveled SEAL versus full HE | Retain bounded SEAL operators as historical components; add a genuinely bootstrap-capable backend boundary for full methods. | No current SEAL bootstrap. Choose an independently implemented refresh or an approved general-purpose library primitive; do not link FHE-MP-CNN or authors' modified inference trees. |
| B2: CacheMir/ROSETTA CPU | Qualify a pinned **Lattigo v6** release as the initial common CPU candidate, with independent PLLM packing, lookup, scheduling and state code. | Exact APIs for sparse-secret encapsulation, CtS-first refresh, raw RLWE/blind rotation, ring switching and target-level recovery; supported parameters; license; native ownership/FFI or separate-process boundary. CacheMir does not pin its original Lattigo version. |
| B3: NEXUS/THOR algorithm port | Preserve each method's parameters, layouts, approximation schedule and output semantics even if backend adapters differ. | NEXUS's dual-key operations, scale propagation and 21 usable levels; THOR's deferred kernels/complexification and source bootstrap. A different generic backend is a recorded execution adaptation requiring approval. |
| B4: GPU | Implement and qualify CPU algorithms first; treat source GPU reproduction as a separately budgeted target. | Do not call a CPU implementation a reproduction of four-A100 NEXUS, single-A100 THOR, or customized-Phantom speedups. Qualify general GPU dependencies separately; independently implement required kernels/customizations. |
| B5: alternative full-HE library | Consider only if it can demonstrate the same primitive/key/layout semantics and precision at the intended security target. | A library advertising CKKS bootstrapping or CKKS–TFHE switching does not establish ROSETTA's encrypted-accumulator and ring-conversion contract. No silent switch to a fixed low-precision LUT or client refresh. |

A Go/Lattigo implementation behind Rust/C ABI or a provider process is an architectural/execution decision, not an already accepted change. Include copying, serialization, Go runtime/GC, host-device transfer and lifetime costs. A backend change may preserve algorithm reproduction while changing the hardware-performance claim; record both identities.

## 10. Approval register: concrete unavoidable choices

Each approval must name source evidence, proposed change, necessity, affected security/numeric/state/execution claims, and acceptance test. Approval of one field does not waive the others. Items marked source resolution can be resolved faithfully without changing the method; if not resolved, an explicit adaptation approval is required **before code**.

| ID | Choice to resolve | Faithful default |
| --- | --- | --- |
| A1 | Approximate source HE versus the common exact W8A8/ring32 contract | Implement original floating/approximate model contract; use predeclared approximation and task-quality criteria. Do not quantize to fit existing operators. |
| A2 | Backend, native bridge, secret distributions, full prime/key graph and refresh implementation | Approve B1–B4 with source-specific profiles; preserve 128-bit target and ciphertext-only evaluation. No shallow no-bootstrap proxy. |
| A3 | THOR encoder→autoregressive task; NEXUS single-output→persistent decoder | Reproduce original evaluated functionality first. Separate named decoder extension needs explicit approval for new attention mask, norm/activation, positional encoding, GQA, KV and token semantics. |
| A4 | Lookup, output disclosure and token feedback for all methods; especially incomplete CacheMir/ROSETTA boundaries | Preserve private input/server weight assumptions. NEXUS keeps Argmax output; THOR keeps documented client embeddings. Resolve private head/embedding and per-token versus fully encrypted feedback; never silently return full logits or put head/KV at client. |
| A5 | Missing coefficients, scales, ranges, iteration counts, bootstrap precision and tie/endpoint rules | Recover pinned primary specifications or obtain static source clarification. Any regenerated fit, changed threshold, truncation, clamp, epsilon, relaxed iteration count or new public prefix needs approval. |
| A6 | CacheMir/ROSETTA prefill, persistent levels, append/capacity and GQA mapping | Implement paper-compatible encrypted prefill/cache. Specify copy-versus-retained-state modulus switching and refresh. No clear prefill, clear retained KV, cache compression/eviction, or reduced context as a hidden substitute. |
| A7 | Source underspecification/contradictions: NEXUS layout composition/level table; THOR domains; CacheMir Appendix C/D; ROSETTA extraction dimensions and rounding | Document an algebraically justified resolution, validate it independently, and approve any changed behavior. Preserve unresolved status until then. |
| A8 | Weight/key/cache residency at source scale; CPU/GPU hardware and cost calibration | Budget the source shapes and full encrypted objects. Approve streaming/offload or alternate hardware explicitly and account for all transfers. Resource failure of a local candidate is not failure of the method. |
| A9 | Shared-private-model versus public-weight common comparison, and integrity expectations | Keep paper security classes separate. Public-weight deployment or added verification is a distinct approved adaptation, not an implicit strengthening/equivalence claim. |

Private input-dependent range selection, degree selection, overflow disclosure, or early termination can change leakage. Prefer public, frozen profiles and fixed-shape behavior. If complete input coverage cannot be established, state the supported task/domain and obtain approval for its admission rule; do not make an unchecked global claim.

## 11. Validation and resource gates

These are **future execution gates**, not work performed by this review. No build or workload is authorized here.

### G0 — Evidence and specification lock

**Ready:** four PDF hashes/page locators and local static inventory. **Still required:** source-resolution fields above; exact checkpoint/tokenizer/config and dataset revisions; approximation tables; bootstrap/key manifest; approved deviations. Static author-code review, if later used, must remain a quarantined specification oracle and bind selected source bytes; author inference code must not be run or linked.

### G1 — Backend/key feasibility, before allocating a model

Produce a capability matrix at source ring degree, scale and security parameters. Show real refresh and key ownership; for ROSETTA, show compatible full/small ring and 1024-dimensional LWE domains, encrypted-accumulator blind rotation, and bidirectional conversion. Freeze parameter validation evidence and all evaluation-key counts/sizes. Missing bootstrap or conversion blocks full-method implementation, rather than triggering a weaker fallback.

### G2 — Independent mathematical conformance

- Check packing/unpacking and dense algebra with an independent clear oracle, including non-square weights, head boundaries, invalid slots and full vocabulary fragmentation.
- Verify source polynomial expressions with a high-precision oracle over declared domains; include interval endpoints, selector uncertainty bands, near ties, tiny denominators and zero/near-zero variance. Separate approximation error, CKKS error, bootstrap error, and conversion/index error.
- Resolve source pseudocode ambiguities before ciphertext tests. An oracle that calls the same implementation is not independent.
- Validate level/scale ledgers symbolically, including every layout mask, scalar multiplication, residual join, StC/CtS and recovery. Count actual extended ciphertext lifetimes for lazy kernels.

### G3 — Native primitive validation under approved small execution budget

Use real ciphertexts and source cryptographic parameters. A reduced tensor tests indexing but is not the source workload. Check encrypt→evaluate→decrypt, multi-refresh chains, bootstrap entry-domain limits, and target-level output. ROSETTA must additionally check comparison, interval selection, ring conversions and round-trip error for single/multiple values, including both sides of every boundary. Secret keys/decryptors remain absent from provider code paths.

Freeze numeric tolerances before tests. Average RMSE alone cannot validate Argmax, table-index correctness, or zero-padding invariants. Record worst-case error in the tested domain and failure counts as well as the paper's reported metric. Security/decryption-failure targets need their own parameter analysis, not a few successful decryptions.

### G4 — Source-shaped vertical slices

| Method | First meaningful composed slice | Required exit evidence |
| --- | --- | --- |
| NEXUS | Original BERT attention/FFN block, coefficient/slot transitions, real refresh; separately full-vocabulary encrypted Argmax | Complete numeric/level trace; source preprocessing accounted; no client intermediate reveal; exact selected output policy. |
| THOR | BERT attention/FFN block in multi-diagonal layout, source-range aSOR nonlinearities and real refresh | PC-MM/CC-MM algebra and kernel counts; lazy operations; source input/output scales; encrypted-versus-clear block error. |
| CacheMir | One source decoder block with encrypted prefill and repeated encrypted cache updates | Append at `t-1,t,t+1`, partial V groups, attention/V boundary `S/H-1,S/H,S/H+1`; old-token invariance; real bootstrap schedule; state level/scale trace. |
| ROSETTA | Multi-value segmented inverse/invsqrt in a complete softmax/norm pipeline, then decode block | Both conversion directions, correct encrypted selection, automatic scheme/level path, complete conversion costs, inherited KV invariants. |

Reject only the candidate whose declared contract fails. Do not turn the existing Qwen range rejection into a NEXUS/THOR system failure, or another paper's drop-in baseline divergence into failure of the original system under its own task/profile.

### G5 — Full functionality and quality

For original encoder methods, complete lookup/embeddings through classifier or Argmax. For approved decoder methods, execute prefill plus multiple generated tokens, with full head/selection/feedback, active KV, positions/history, pending token, stopping behavior and sampling randomness bound. Include continuation, cancellation and fail-retirement behavior in the runtime contract; do not infer these from a successful isolated decode iteration.

Compare exact state only where the algorithm promises exactness; otherwise compare source-compatible numeric outputs and predeclared task quality. Preserve a clear approximation-only diagnostic as a separate result from encrypted execution. Source calibration prefixes, model transformations, and fine-tuning are part of the model identity.

### G6 — Resource admission and complete measurement

Before running any full shape, estimate separately:

```text
raw CKKS payload ≈ ciphertext_component_count · R · active_RNS_limbs · 8
total live memory = weights + K/V + evaluation keys + contexts
                  + ciphertext temporaries + bootstrap/conversion scratch
                  + serialized copies + language/runtime overhead
```

Account for size-three products before lazy relinearization, hoisted decomposition buffers, bootstrap keys, ring-switch/PBS keys, small-ring batches, encoded weight diagonals and host/device duplication. Bit-packed modulus size is not allocated RNS memory. Disk-mapped/streamed data still incurs I/O and transient residency. Estimate K and V separately using their actual layouts, including padding/capacity; neither is ordinary float KV.

Freeze CPU threads, host RAM, GPU model/count/VRAM, model/cache/key placement, maximum concurrency and a maximum admitted execution budget. Do not launch a model to discover that its encoded weights need hundreds of GB. A source-shape admission rejection should report missing capacity or unsupported dependency, without reporting an invented timing or declaring the original method failed.

Once admitted, measure cold-client setup, warm-public setup, private preparation, key delivery, prefill, each decode, output processing and cleanup. Capture client and provider/helper CPU, full directional/all-link traffic, host RSS and device peaks, and synchronized simultaneous aggregate memory. Apply the comparison contract's client tiers only to observed whole-client peaks; keep source hardware replication separate from common-hardware comparisons. Amortization, module estimates and observed full-request measurements must remain separately labeled.

## 12. Dependency order and completion criteria

```text
PDF/specification freeze + approval register
    → approved full-HE backend, parameter/key graph, ownership boundary
    → source bootstrap and layout primitives
        → NEXUS dual-key/compression + source nonlinearities → original request
        → THOR diagonal/lazy kernels + aSOR → original BERT request
        → CacheMir VMM + encrypted KV + calibrated nonlinearities
            → source placement DAG → encrypted decoder block → approved token loop
        → ROSETTA ring/LWE bridge + HomComp + encrypted-accumulator LUT
            → joint scheme/level DAG + qualified decoder state → approved token loop
    → source-shaped numerical/quality qualification
    → budgeted full lifecycle measurement
```

**First blockers:** backend/bootstrap approval for all four; missing source numerical schedules; NEXUS's preprocessing/layout composition; THOR's encoder-versus-decoder decision; CacheMir's retained-state level and output loop contract; ROSETTA's cross-domain keys/conversion/rounding contract.

Algorithm reproduction is achieved only when the independently implemented protocol, approximations, layouts, state and output functionality match the frozen source or explicitly approved adaptation. Hardware reproduction additionally needs the source execution platform and relevant kernel behavior. A CPU algorithm reproduction can be valid without reproducing GPU latency; a fast GPU primitive cannot establish the whole algorithm. None of these four currently qualifies as a complete PLLM source-system reproduction.

## Appendix A. Static implementation fingerprints

SHA-256 of the reviewed checkout bytes:

| Path | SHA-256 |
| --- | --- |
| `crates/pllm-core/src/ckks_layout.rs` | `23a8aa356780e4ab99fb1a880f16b868346f54e49145fff36586c0b2736e43c4` |
| `crates/pllm-core/src/coefficient_repacking.rs` | `0fb2fb216de9563a8fca9f3469999cd7cc518c330dd336f189aed767c0d2f24c` |
| `python/pllm/runtime/ckks_reference.py` | `a21e3c62e1a19c62343ad25ee6bc7d9f68783b7e7590f59b539025c79a65219a` |
| `python/pllm/runtime/ckks_softmax.py` | `2b1d9cbe8a16363d3d893cf4b639d6bf62a00fffcf0fa9805b0ff10f12bfc457` |
| `python/pllm/runtime/coefficient_repacking.py` | `2ec860af6bc1d84db7ec28ce82b912264242dd2a53e1b2c492ef842a54a07547` |
| `python/pllm/runtime/he_layer_feasibility.py` | `7910f5dfe538fe7a2ab6843d100f9221a7259d4d47fda5dd405a27f6177232bb` |

These hashes identify static inspection only, not a build or an execution receipt.

Document validation: all 18 local link occurrences resolve; all four PDF hashes match the source manifest; all six implementation fingerprints match the reviewed files. Fenced blocks are balanced. No model, cryptographic workload, or performance test was run.
