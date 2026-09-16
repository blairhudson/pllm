# PLLM: a fresh implementation audit and a smaller research direction

## Result

The most useful change is not another language model architecture. It is a different way to prepare private matrix operations. Packing many future masks by coordinate turns the server's encrypted calculation into a linear combination of ciphertexts. For the existing integer weight format, additions suffice. A second implementation performs the same calculation with ordinary integer matrix multiplication on ciphertext coefficients.

The complete preparation experiment is 33.5 to 36.0 times faster than a strengthened, cached version of the previous method on the two larger matched stages. The coefficient matrix kernel is a further 12.4 times faster than the addition implementation on one stage, but that number excludes conversion into and out of ciphertext objects. These factors must not be multiplied into a claimed language model token rate.

A fresh source audit also found defects that the preceding reviews did not catch. Plaintext activation scales identify almost every token in a synthetic public embedding. The optional integrity checker uses publicly reproducible default challenges. Cryptographic masks use a simulation generator. A database rollback resurrects consumed correlation identifiers. The authenticated arithmetic code keeps both participants' secrets in one process and is not a demonstrated distributed protocol for malicious participants.

This archive therefore contains a research reset, security corrections, executable experiments, and their results. It is not another production release, a rewritten dissertation, or a claim of complete proprietary model protection.

## 1. Evidence and source provenance

The executable baseline is the persisted PLLM 0.14.0 source archive. The previous reproduction bundle explicitly states that the claimed 0.15 binaries and revised manuscripts did not persist. No reconstructed 0.15 claim is treated as evidence that its implementation existed or passed tests.

The baseline is preserved without changes under `reference/`. `patches/pllm-input-privacy-fixes.patch` contains the actual corrections made in this run. SHA-256 hashes identify the source and every delivered file.

Experiments use synthetic public integer matrices and, for the scale attack, a synthetic public embedding. No trained Gemma, Muse, or other large checkpoint was evaluated. No GPU was used. All primary comparisons rerun the baseline in the same current environment rather than importing timing numbers from older machines.

Environment: Intel Xeon Platinum 8573C, five exposed logical CPUs, four CPU equivalents of cgroup quota, 4 GiB cgroup memory limit, Python 3.13.5, NumPy 2.3.5, PyTorch 2.10.0+cpu, and TenSEAL 0.3.17. Ordinary comparisons use one thread. The coefficient matrix experiment separately measures one and four PyTorch threads.

## 2. Security findings and corrections

### 2.1 Activation scales reveal information about the input

The old `MaskedStageRequest`, `BlindedStageRequest`, and `DirectFHEStageRequest` encoders send the per-row activation quantization scale as a plaintext float32 field. The remote integer matrix calculation does not need this value. The client already retains it for dequantization.

The attack builds a public embedding containing 4,096 token rows of width 128, applies local RMS normalization, and runs the actual old quantizer and request encoder. A server that knows this embedding can calculate a dictionary from float32 scale to token ID. The transmitted scale uniquely identifies 4,092 of the 4,096 tokens: 99.9023 percent.

This is an executed counterexample on a synthetic embedding, not a trained Gemma attack. It disproves the general inference that an audit containing no literal prompt or token ID necessarily hides token content.

Correction: all three encoders retain the old field for wire compatibility but fill it with the public constant one. The true scale remains on the client. The existing model output tests still pass. A new wire version should remove the field entirely. Static weight scales are model metadata and are a different issue.

### 2.2 The optional result checker uses predictable challenges

`create_linear_check_key` defaults to `seed=1`. An evaluator can reproduce the two check vectors, construct a nonzero error in their common nullspace, and change the result without failing the check.

The executed attack changed three coordinates of an output from a 64 by 64 matrix. The old checker accepted the wrong result. The random product check theorem is not the defect; its secret challenge assumption is absent in the default implementation.

Correction: the default now uses fresh operating system randomness with rejection sampling. An explicit seed remains only for deterministic test vectors. This fixes predictability; it does not by itself establish a complete malicious-server protocol. The number of checks, field, key reuse policy, trusted source of the input projection, feedback leakage, and total failure probability still need a deployment policy. In particular, two checks over a roughly 16-bit field are not a 128-bit authentication claim.

### 2.3 Client cryptographic masks use a simulation PRNG

The live client initializes NumPy's default generator from `secrets.randbits(128)`. A strong seed does not turn a statistical generator into a cryptographic generator. NumPy explicitly excludes security use for these generators [1].

Correction: masks in the live client use `secrets.token_bytes` and unbiased rejection into the arithmetic modulus. Random model fixtures and token sampling may still use deterministic generators; they are not cryptographic masks. No PRNG state-recovery attack was executed in this run. This finding is an implementation audit, not such an attack.

### 2.4 SQLite crash recovery is not rollback protection

The ledger reserves before use and burns reservations after ordinary crashes. Those tests address interrupted writes. They do not address restoration of an earlier database image.

The executed rollback test created a ledger, saved a copy, consumed correlation ID 1, restored the old copy, and reserved ID 1 again. Thus the ledger can reissue a consumed ID under rollback.

A simpler initial design should discard all preparation on process restart, bind it to a fresh session epoch, and refuse old sessions. Durable recovery requires freshness outside the restored database, such as an external monotonic service. A full virtual machine snapshot can also restore an in-memory epoch; ordinary restart handling does not solve that attack. The focused patch does not claim to repair durable rollback protection.

### 2.5 Arithmetic simulation is not distributed malicious security

The stored authenticated arithmetic prototype keeps the client share, server share, and authentication key in one Python process. Internal methods reconstruct both shares. Its default field has 65,537 elements and 65,536 possible nonzero authentication keys, yielding only 16 bits in the stated single-key forgery bound.

These experiments test arithmetic identities and tamper detection under their simulator. They do not demonstrate isolated participant execution, malicious input validation, maliciously secure preprocessing, or a compiler that prevents modified clients from injecting intermediate inputs. Documentation in the focused patch calls this a simulator explicitly.

For confidential weights, the client must not obtain arbitrary layer outputs. Encrypting chosen layer inputs does not remove that oracle. Standard HE data confidentiality also does not imply circuit privacy [2]. Full model protection requires a protocol with an explicit output policy and execution enforcement.

## 3. First simplification: transpose the preparation batch

### 3.1 Existing workload

For public weights W, each future invocation needs a fresh independent mask r and its matching result Wr. The online client sends x+r and subtracts Wr from the server result. These correlations are consumed once. Preparation can happen before x is known because W and the future stage shape are known.

The old factory packs complete mask vectors into segments of a ciphertext. Guard regions prevent unwanted cyclic overlap. On the 768 to 256 stage, this leaves capacity for only four masks. It then uses a diagonal transform with baby-step/giant-step rotations.

### 3.2 New representation

Take B independent masks, arranged as R[b,i]. Encode one polynomial per input coordinate:

    P_i(X) = R[0,i] + R[1,i] X + ... + R[B-1,i] X^(B-1).

For each output coordinate j, evaluate:

    Q_j = sum_i W[j,i] Enc(P_i).

Coefficient b of Dec(Q_j) is exactly `(R[b] W^T)[j] mod p`.

Because B is at most the ring degree, additions never mix different coefficients. The model weights remain unchanged signed W4 integers in [-7,7]. The reference builds required multiples 1 through 7 by ciphertext doubling and addition, then combines positive and negative terms.

There are no rotations, Galois keys, relinearization keys, ciphertext multiplications, plaintext multiplications, or bootstraps in this evaluation path. All-zero model rows use an encryption of zero rather than an invalid transparent ciphertext. The server holds public weights and HE parameters, not a client decryption key.

This is a layout and implementation contribution, not a new cryptosystem or a claim that encrypted integer linear algebra is new. Cheetah is an important prior example of designing HE linear protocols to eliminate rotations [3].

### 3.3 Measured comparison

Unit: complete stage correlations per second. One correlation covers one input mask vector and its complete output mask vector, not one scalar and not a generated language token.

| Stage | Old factory | Cached BSGS control | Coordinate packing | Gain over cached control |
|---|---:|---:|---:|---:|
| 64 to 64 | 163.3 | 596.3 | 6,951.0 | 11.66x |
| 768 to 256 | 2.30 | 14.48 | 484.57 | 33.47x |
| 1,536 to 512 | 0.718 | 4.605 | 165.60 | 35.96x |

The cached control was implemented in this run. It compiles plaintext diagonals once, stores them in NTT form, reuses the compiled weights, and performs rotations using the old layout. Reporting only the original slow control would overstate the packing result.

The old and cached layouts use batches 32, 4, and 2 for these shapes. The new layout uses 2,048 future masks under one key in each case. This batch size difference is a defining property of the layout, not evidence of lower first-request latency.

The new median batch completion times are 0.295, 4.226, and 12.367 seconds. Peak process RSS is approximately 129, 439, and 787 MiB respectively. These are substantial working buffers. Preparation is suitable for sustained sessions or a trusted customer runtime with enough memory, not automatically a tiny mobile client.

The full 768 to 3,072 expansion was also executed: 69.58 correlations per second, with a 29.435-second median batch time. No corresponding old-layout speedup is asserted for that shape.

### 3.4 Arithmetic and cryptographic parameters

The matched tables use exact arithmetic modulo 65,537. SEAL accepts the selected contexts at its TC128 security level: the old control uses degree 4,096; coordinate packing uses degree 2,048. Minimum measured invariant noise budgets for the three coordinate rows are 27, 26, and 25 bits. This is library parameter validation and measured decryption margin, not an independent security audit.

Modular exactness does not automatically establish a signed W4A4 dot product without wraparound. A worst-case width 1,536 dot can reach 49 times 1,536 = 75,264, which exceeds half of 65,537. A separate degree-2,048 run uses modulus 786,433. It produces 160.41 correlations per second on the 1,536 to 512 stage, with at least 21 bits of measured noise budget. A complete masking/unmasking test with all +7 and -7 inputs/weights recovers the exact signed extrema +75,264 and -75,264.

A production planner must derive the modulus from each stage's actual weight bounds and activation range. The smaller-modulus benchmark must not be applied blindly to larger signed dot products.

### 3.5 What timing includes

Coordinate timings include OS mask generation, plaintext formatting, encryption, ciphertext serialization, server deserialization, server evaluation, output serialization, client deserialization, and decryption/decoding. The low-level binding serializes through temporary files; this overhead is included. Key generation and clear reference verification are recorded separately and excluded from steady preparation timing.

No physical network was timed for these tables. For 768 to 256, payload volume is about 15.47 KB per correlation. Adding byte-transfer time for a sequential 1 Gbit/s link reduces the projected rate from 484.6 to 457.2 correlations/s; at 100 Mbit/s it becomes 302.9. These two figures are bandwidth projections, not measurements. One additional round trip per entire preparation batch is negligible at this scale but should be added for a specific deployment.

The same key must protect all masks packed into a ciphertext. Unrelated customers' HE keys cannot simply be merged. Ordinary online modular matrix batches can mix independently masked rows, but HE preparation must still respect key boundaries.

## 4. Second simplification: ciphertext evaluation is an integer GEMM

Once the operation is only a public linear combination of ciphertexts, each ciphertext coefficient follows the same integer linear map. With the tested BFV context there are two ciphertext polynomials, each with 2,048 coefficients, and one coefficient modulus q. Treat the input ciphertexts as an n by 4,096 matrix A of residues modulo q.

The server needs only:

    C = W A mod q.

The prototype breaks each residue into eight-bit digits. Each digit matrix is shifted into signed int8 form and multiplied by W with PyTorch's integer matrix kernel. A public row-sum correction removes the digit shift. Horner accumulation modulo q reassembles the limbs using int64 arithmetic. With q below 2^54, multiplication of a reduced accumulator by 256 fits int64; the input-width check separately bounds the int32 dot products.

No floating-point matrix multiplication is used. Tests compare against unbounded Python integer arithmetic, including values that overflow a naive int64 dot product. The real ciphertext experiment compares every output coefficient against SEAL's evaluator, not merely rounded decrypted outputs.

For a 768 to 256 stage with 2,048 masks:

| Server evaluator | Median time |
|---|---:|
| SEAL additions | 2.203 s |
| Integer GEMM, one CPU thread | 0.253 s |
| Integer GEMM, four CPU threads | 0.178 s |

The four-thread coefficient kernel is 12.40x faster than the addition evaluator on that experiment. Input extraction through Python accessors takes another 0.826 s. This backend does not yet reconstruct output SEAL objects or provide a production wire format. It is therefore a server kernel result, not a replacement measured total for the preceding table.

The practical next implementation is a small native bridge that moves coefficient arrays directly between SEAL and an integer GEMM kernel, followed by GPU integer GEMM. GPU performance is unmeasured. The promising point is architectural: this avoids writing a general accelerator for rotations, key switching, bootstrapping, and arbitrary encrypted programs merely to prepare linear correlations.

## 5. A simple direct alternative

A separate direct BFV experiment reverses the activation coefficients and lays weight rows consecutively into a polynomial. An ordinary polynomial product contains each required dot product at a known coefficient. Preencoded NTT weights avoid repeated preparation.

| Direct stage | Full local encrypt/serialize/evaluate/deserialize/decrypt | Server evaluation |
|---|---:|---:|
| 64 to 64 | 3.411 ms | about 0.181 ms |
| 768 to 256 | 96.101 ms | about 9.666 ms |

The larger result is dominated by representation and transport: its returned ciphertext payload is approximately 3.96 MB. This is useful as a latency alternative for small public stages, not a universal winner. The client can decrypt unused polynomial coefficients as well as the desired outputs, so this layout provides no model-confidentiality claim.

A cache-blocking variation of coordinate additions produced a smaller gain. On 768 to 256, blocks of 256 inputs reduced median evaluator time from 1.901 to 1.592 seconds, while blocks of 32 made it worse at 2.515 seconds. The run contains three samples per choice and noticeable variance. It supports testing cache sizes, not a large general speedup claim.

## 6. What to retain and what to stop doing

Retain the SDK and local Responses facade, the quantized model reference, the checkpoint reader, the client nonlinear graph, and a narrow remote tensor interface. Make the compiled stage specify matrix dimensions, integer range, modulus, cryptographic profile, key identity, model hash, and permitted operations. Cryptographic choices belong in a checked execution plan, not a growing list of user-facing flags.

Pause the provider market, the eight-paper expansion, and new architecture families as product dependencies. Archive rather than delete the research. They do not resolve the current confidentiality defects or demonstrate sustained private generation. A market admitting adversarial providers would make the missing malicious-server guarantees more important, not less.

Stop using an online-only number as the primary speed result. For a token requiring stages s, the steady preparation compute requirement is the sum of each stage's batch cost divided by its useful batch size, multiplied by its invocation count. Add discarded preparation, transfer volume, startup, and the online graph. A prepared response can be fast even when the service drains its inventory faster than it fills it.

Keep public-weight input privacy and confidential-weight model privacy as separate backend contracts. The latter must not become available merely because rate limits or output blinding are enabled. MP-SPDZ provides concrete malicious-security protocol implementations and exposes the distinction between secure protocols and stripped semi-honest variants; it is a more credible integration target than expanding the current arithmetic simulator [4]. It still requires application-specific validation and review.

## 7. The remaining constraints

### Interaction latency

A computation with 212 sequential exchanges has at least 212 times RTT of network latency. At 20 ms RTT, this alone is 4.24 seconds per token, a ceiling of 0.236 tokens/s before any computation. Batching other users does not remove one user's causal dependencies.

The model-preserving path is to keep intermediate values shared through reviewed secure nonlinear operations and communicate once per fused program rather than once per Python module. A persistent socket removes connection overhead but not this dependency bound. Slalom illustrates the longstanding importance of partitioning and verifying outsourced linear work [5]; its trusted-hardware setting is not interchangeable with the current client/server threat model.

### State and lookup

Dense weights are only one client cost. The attention/KV state and temporary cryptographic buffers still matter. Dedicated private embedding lookup avoids spending a large dense linear transform on a one-hot query. Keeping encrypted KV state on the server is a separate research branch; Cachemir specifically addresses that problem with decoding-aware packing and bootstrap placement [6]. These are not implemented by the new preparation kernel.

### Malicious execution

Public model weights remove model IP secrecy, not the need to protect client inputs against an actively malicious server. Ciphertext validation, decryption feedback, authenticated preprocessing, output correctness, and adaptive failures need an integrated protocol argument. The focused fixes remove concrete bugs; they do not certify this broader property.

## 8. Rebuild plan

The next coherent system is a small compiler and runtime around prepared matrix operations:

    application and tokenizer
        -> local private model state
        -> audited binary stage protocol
        -> remote quantized matrix engine

    same client key + known future stage
        -> coordinate-packed mask preparation
        -> native integer ciphertext GEMM
        -> short-lived correlation inventory

The compiler should choose between cached BSGS, direct coefficient convolution, and coordinate preparation according to input width, output width, available future masks, memory, bandwidth, and security policy. It should refuse an arithmetic profile that cannot preserve the signed result, or a privacy profile whose protocol is unavailable.

One full benchmark should decide the next release: a trained small checkpoint, exactly the same W4A4 reference graph, separate client and server processes, real prompt prefill, long autoregressive decode, no initial inventory hiding, measured replenishment, and long enough operation to expose depletion. Report total generated tokens, first-token latency, inter-token latency, useful and discarded masks, client/server CPU and memory, bytes, numerical parity, and model quality relative to the floating checkpoint.

The new measurements justify this narrower programme. They do not establish a new full-model token rate. The immediate research result is faster, simpler exact HE preparation and a concrete path from ciphertext arithmetic to existing integer matrix hardware, alongside corrections to the security boundary.

## 9. Validation performed

The focused application corrections passed all 177 tests in the copied 0.14 tree, with real TenSEAL available. The new experiment suite passed 36 tests; 11 are shared security regressions already counted in the application run. New tests cover full-capacity coefficient packing, negative and zero weights, polynomial boundary cases, separate spawned server execution without a client key, exact mask/unmask composition, signed wraparound bounds, cached BSGS, arbitrary-precision limb arithmetic, and matching real BFV ciphertext coefficients.

Three samples per principal coordinate and old-factory experiment and four samples for the cached control were used. The coefficient GEMM used three samples per thread setting. All samples, including first-use costs, remain in the raw JSON. Small sample counts and shared virtual hardware limit conclusions about small differences. The large matched gains are the conclusions emphasized here.

No real WAN, GPU, trained checkpoint generation, MPC backend integration, or updated manuscript submission is claimed.

## References

[1] NumPy. Random sampling. The security warning explicitly excludes its generators from cryptographic uses. https://numpy.org/doc/stable/reference/random/index.html

[2] Nico Döttling and Jesko Dujmovic. Maliciously Circuit-Private FHE from Information-Theoretic Principles. ITC 2022 / Cryptology ePrint 2022/495. https://eprint.iacr.org/2022/495

[3] Zhicong Huang, Wen-jie Lu, Cheng Hong, and Jiansheng Ding. Cheetah: Lean and Fast Secure Two-Party Deep Neural Network Inference. USENIX Security 2022. https://www.usenix.org/conference/usenixsecurity22/presentation/huang-zhicong

[4] MP-SPDZ project. Protocol and implementation documentation. https://mp-spdz.readthedocs.io/en/latest/readme.html

[5] Florian Tramer and Dan Boneh. Slalom: Fast, Verifiable and Private Execution of Neural Networks in Trusted Hardware. ICLR 2019. https://arxiv.org/abs/1806.03287

[6] Ye Yu et al. Cachemir: Fully Homomorphic Encrypted Inference of Generative Large Language Model with KV Cache. arXiv:2602.11470, 2026. https://arxiv.org/abs/2602.11470
