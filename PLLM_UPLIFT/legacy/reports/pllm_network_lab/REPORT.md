# PLLM: compressed corrections and inference without local model weights

**Research experiment, 13 September 2026**

## Decision

The most promising candidate tested here combines three changes:

1. Encode both offline correction tensors and online responses using the same low-bit-plus-syndrome representation.
2. Evaluate directly on the encoded corrections at Inference; do not reconstruct them there.
3. Generate correlations through offline, packed homomorphic evaluation so that only Inference needs the weight matrices.

A fourth option uses a client-only prediction as additional decoding side information, without sending the prediction to Inference. It worked on a deliberately correlated synthetic workload, but its usefulness for real autoregressive transformer activations is unestablished.

The implementation demonstrates exact algebra and an all-weights-remote toy autoregressive graph, not a fast, secure, production LLM implementation. A simple public-bound bit-packing baseline is competitive and should precede the more elaborate codec in production. The model-free preprocessing experiment adds substantial offline traffic and computation; this report does not claim an end-to-end communication improvement from the combined system at LLM scale.

## What was executed

The package contains actual executable implementations, not HE placeholders:

| Experiment | Observed result | Evidence |
|---|---|---|
| Bounded-overflow lossless codec | 600/600 synthetic vectors, 663,960 coordinates recovered exactly across 16-, 24-, and 32-bit rings | `codec_results.json` |
| Main 4,096-coordinate profile | 12,288 bytes becomes 6,456 bytes; 47.46% response reduction | `codec_results.json` |
| Ordinary compression control | zlib produced 12,299 bytes from the 12,288-byte masked payload | `codec_results.json` |
| Independent compiled codec | 330/330 cases exact and byte-compatible with reference encoder | `fast_codec_results.json` |
| Synthetic 4,096-by-128 integer products | 30/30 exact with 15 low bits and an outlier budget of 16; all 30 rejected with 14 low bits | `codec_results.json` |
| Public-bound baseline on the same products | 8,734 bytes, versus the syndrome codec's 7,914 bytes | `codec_results.json` |
| Computation with compressed corrections | 120/120 products exact; response packets identical to compressing the ordinary full result | `compressed_execution_results.json` |
| Model-free preparation using actual Paillier | 101/101 matrix-vector records exact, across four packing sizes | `weightless_results.json` |
| All-weights-remote toy graph | 32/32 stage products exact across eight autoregressive steps; includes tied embedding/head; neither Client nor Preparation is passed any model matrix | `toy_graph_results.json` |
| Client-only prediction | 40/40 synthetic random-walk steps exact at 5,484 bytes instead of 12,288 | `compressed_execution_results.json` |
| Competing masked stochastic requantization | 57,600 enumerated cases followed the derived rounding relation; not selected for the exact protocol | `codec_results.json` |

The correctness counts are distinct experiments, not samples from a cryptographic security test. Randomness for synthetic test data uses a reproducible NumPy generator. The HE role experiment uses system-generated secret seeds and real randomized ciphertexts. No claim of production constant-time implementation, malicious security, process isolation, or model quality follows from these checks.

## 1. First establish a stronger simple baseline

The earlier design chooses 16-, 24-, or 32-bit arithmetic rings. That does **not** require transmitting that many bits per output coordinate when a smaller public bound is available.

For an authenticated integer matrix W and declared activation limit |x_j| <= A, compute at the model compiler:

    bound_i = A * sum_j abs(W_ij)
    width_i = floor(log2(bound_i)) + 2   # for positive bound_i

Choose width_i so that bound_i < 2^(width_i - 1). Send only the low width_i bits of the masked output. The client adds the corresponding low mask bits and interprets the result as signed. This is exact for every activation satisfying the public input bound. Public groups of equal widths avoid per-response width metadata. Inference can likewise use low-width corrections without needing their discarded bits.

The Client needs an authenticated width manifest, not W. These widths must not be chosen from the actual private output. Incorrect or unauthenticated bounds also undermine correct interpretation, so verification and the signed model manifest remain important.

On the synthetic 4,096-by-128 matrix, the per-coordinate safe widths were 17 or 18 bits. Actual grouped packing used **8,734 bytes**, compared with **12,288 bytes** at 24 bits. This baseline is exact throughout its declared input range and requires no empirical sparse-outlier assumption. The more ambitious codec used **7,914 bytes** on the same products: **9.39% less than the stronger baseline**, not merely 35.60% less than the original 24-bit representation.

Implementation: `bound_codec.py`. This comparison is important: do not attribute all savings from moving away from byte-multiple wire widths to the syndrome construction.

## 2. Candidate codec: low bits plus sparse-overflow syndromes

### Arithmetic and public profile

Let q = 2^k, B = 2^b, with b < k. Let y be an n-coordinate signed result in the canonical k-bit range. Inference holds only

    z = (y - s) mod q

while the Client knows s. Public parameters n, k, b, and t are fixed independently of the private activation. The additional domain condition is:

> At most t coordinates of y lie outside the signed b-bit interval [-B/2, B/2).

Let H be a binary parity-check matrix for a shortened primitive binary BCH code with distance at least 2t+1. We interpret its entries as ordinary 0/1 integers when working modulo q. The implementation uses field degree m = max(3, ceil(log2(n+1))) and r = tm binary rows from odd BCH syndromes. The even syndromes follow by squaring in the binary extension field.

### Encoder

The encoder knows neither y nor s. It transmits:

    low = z mod B
    high = floor((H z mod q) / B)

These fields use n*b and r*(k-b) bits, respectively, with byte padding on each field. Sending all k bits of H z would be wasteful: its low b bits can be reconstructed from H low mod B.

For n=4096, k=24, b=12, t=16, m=13 and r=208:

    low field:       4096 * 12 / 8 = 6144 bytes
    syndrome field:   208 * 12 / 8 =  312 bytes
    total:                            6456 bytes

The full vector is 12,288 bytes. The saving is 47.4609375% on this response payload.

### Decoder

The Client reconstructs:

    y0 = signed_b((low + s) mod B)
    a  = high * B + (H low mod B)
    T  = ((a + H s - H y0) mod q) / B

The division is exact because the numerator is divisible by B. There exists h over Z_(2^(k-b)) such that

    y = y0 + B h mod q
    T = H h mod 2^(k-b).

Only overflowing coordinates contribute nonzero h, so h has support at most t. Recover h one binary digit at a time. The low bit of T is a binary BCH syndrome. Decode that bit vector, subtract its integer H image, divide by two, and repeat. Every bit plane has support at most t. Finally, reconstruct y modulo q and convert to its canonical signed representation.

The optimized implementation uses the equivalent carry form. If

    v = (s + low - y0) / B,

then

    T = high + H v - floor(H low / B) mod 2^(k-b).

That avoids separately forming three parity products. The optimized code agrees byte-for-byte with the reference encoding.

### Conditional correctness argument

The BCH distance condition makes every binary error vector of weight <=t uniquely decodable from its syndrome. Reducing Hh modulo two therefore recovers h's lowest bit vector. Subtracting its integer image and dividing by two produces a syndrome for the next digit. Induction recovers all k-b bits. Negative signed overflow values are represented modulo 2^(k-b), so they are covered by the same argument.

An equivalent uniqueness argument uses the fact that any collection of at most 2t columns of H is linearly independent over GF(2). If two t-sparse ring vectors share a syndrome, subtract them and divide their nonzero difference by the largest common power of two. Reduction modulo two would give a nonzero dependence on at most 2t columns, a contradiction.

This is an exact codec **on the declared domain**, not universal lossless compression of arbitrary masked ring vectors. The public no-wrap bound for the original integer product remains independently necessary.

### Security boundary

For fixed public profiles and a fixed permitted interaction schedule, each compressed message is a deterministic function of the corresponding baseline message. Therefore this transformation does not reveal more through the message values than the baseline protocol. It does not reveal outlier indices in an explicit list. The Client's private mask is decoding side information, so the savings do not contradict the apparent randomness of z. Compression after encryption with decoder side information has longstanding precedent [1].

This statement does not cover data-dependent profile changes, failures, retransmission decisions, or response timing. The decoder's control flow depends on recovered values. A production implementation needs the same careful metadata treatment as the original protocol, plus the new decoding/failure boundary.

### Failure is not reliably detectable

Random out-of-budget tests with 17, 20, 32, and 80 overflow coordinates were all rejected in the executed samples. That does not establish universal detection.

The harness explicitly constructs a binary codeword v in H's nullspace with 121 nonzero positions. The two distinct outputs

    y = 0
    y' = -2^23 * v

produce identical low-bit and syndrome packets under the same mask for the 24/12/16 profile. The decoder returns zero for both without an exception. The second output violates the domain condition.

Consequently:

- Successful decoding is not an integrity or range certificate.
- An independent authenticated result check is mandatory before advancing model state.
- A secret hash of a value supplied by the same adversarial provider does not establish that it equals W x.
- An adaptive request for more data after failure reveals a data-dependent event unless explicitly protected or included in the leakage model.
- Empirical calibration estimates suitability; it does not establish correctness for every admissible model input.

No unrestricted, fixed shorter encoding can be injective on all q^n possible outputs for a fixed mask. A deployed system must use a proven restricted envelope, preserve an explicit failure/availability policy, or add an appropriate secure mechanism for handling exceptional outputs. Merely increasing t until a test corpus passes is not a proof.

## 3. Compress the offline corrections too

This is the most useful composition result in the experiments.

Define the algebraic representation

    F(z) = (z mod B, H z mod q).

Its two components are additive in their respective rings. The packed wire representation removes redundant low syndrome bits but retains everything needed to perform this addition.

Preparation does not send c=Wr-s in full. It sends the packed F(c). Inference does not decode c. Instead, it computes

    low_result = (W mod B) * (u mod B) + (c mod B) mod B
    syndrome   = (H W) * u + H c mod q,

where u=x-r mod q. It then transmits the packed F(Wu+c). The Client decodes it using s.

The full correction is arbitrary and almost certainly not sparse or narrow. That does not matter: nobody at Inference tries to reconstruct it. Its encoded image is sufficient to produce the encoded final result, whose unmasked value satisfies the codec's domain condition.

The server may precompute H W once per model stage and public codec profile. This is model-derived state stored **only at Inference**. It is not a second model copy at Client or Preparation. It adds storage and setup computation; it does not establish a speedup. In the reference, the 4,096-by-128 matrix's H W summary takes 212,992 bytes as int64 data and setup took about 0.52 seconds in that run.

All 120 executed products returned byte-identical packets to the simpler path of evaluating the full masked product and compressing afterward.

For one tested 4,096-by-128 stage with k=24, b=15, t=16:

| Traffic component | Full representation | Encoded representation |
|---|---:|---:|
| Preparation correction | 12,288 bytes | 7,914 bytes |
| Client upload | 384 bytes | 384 bytes |
| Server response | 12,288 bytes | 7,914 bytes |
| Total of these three components | 24,960 bytes | 16,212 bytes |

This is a 35.05% reduction in those payloads. It **excludes the additional HE exchanges required by model-free Preparation**. It also excludes transport framing, identifiers, model-summary setup, key exchange, and integrity setup.

Implementation: `compressed_execution.py`.

## 4. Model-free Preparation through packed offline HE

### Protocol tested

Only Inference receives W. Client and Preparation share fresh mask-generation seeds. Preparation generates an additive HE key and sends encryptions of independent mask records to Inference. Inference homomorphically evaluates the public-to-it matrix and returns encrypted products. Preparation decrypts the input-independent W r values, subtracts s, and installs either ordinary or encoded corrections. Online execution uses only the original masked ring operations or the encoded variant.

    Preparation -> Inference: Enc(r)
    Inference   -> Preparation: Enc(W r)
    Preparation -> Inference: F(W r - s)

    Client -> Inference: x - r
    Inference -> Client: F(W x - s)
    Client: decode using s, then verify using authentic verification data

The last verification step is a requirement, not implemented by this experiment. The test driver instead compares with an independent plaintext oracle.

The prototype implements actual 2048-bit Paillier encryption, decryption, scalar homomorphic multiplication, and addition [5]. It uses balanced radix packing of independent preprocessing records. W's signed weights multiply each packed record in parallel. Public dimension and weight-magnitude bounds set a radix large enough to prevent cross-slot carry and plaintext-modulus overflow.

This is ordinary HE-based oblivious evaluation, not a claimed new cryptographic primitive. Packing independently consumed records is also an engineering instantiation, not a first.

Preparation receives W r values but does not need or receive W as a matrix. Since the weights are public in the intended application, preventing Preparation from learning them is not a goal. Sufficient linear observations could reveal W; this is not a model-confidentiality construction. Neither service is given the user's online plaintext. Non-collusion remains essential.

### Measured trade-off

The test matrix has 16 outputs and 32 inputs, signed weights in [-7,7], and a 24-bit ring. Ciphertexts are 512 bytes. A conventional full correction is only 48 bytes per record.

| Records per encrypted batch | Encrypted request | Encrypted response | Full correction push | Total preparation bytes / record |
|---:|---:|---:|---:|---:|
| 1 | 16,384 B | 8,192 B | 48 B | 24,624 B |
| 8 | 16,384 B | 8,192 B | 384 B | 3,120 B |
| 32 | 16,384 B | 8,192 B | 1,536 B | 816 B |
| 60 | 16,384 B | 8,192 B | 2,880 B | 457.6 B |

Packing reduces this experiment's HE overhead substantially, but the 60-record case is still **9.53 times the original 48-byte correction traffic**. It takes about 5.54 seconds for the batch in the captured single run, or about 92 ms per record when divided by 60. Neither this division nor the batch's best timing is a general throughput claim. These are small, unoptimized, CPU big-integer measurements, not BFV/BGV/GPU results or WAN timings.

The standalone packing table uses full corrections to isolate the HE trade-off. The separate toy graph uses encoded corrections end to end. All key/ciphertext bounds and serialized ciphertext sizes are actual, not symbolic placeholder estimates. Key generation and fresh seed distribution are outside the table's payload sum. Mask reuse is not permitted across distinct records.

### High-value follow-on experiment

A production candidate should evaluate packed BFV/BGV or another established oblivious linear-evaluation backend, with actual key sizes, ciphertext sizes, noise bounds, and replenishment rates. An additional candidate is to produce the *coded correlation* directly, rather than decrypt a full W r and compress afterward. That requires correctly evaluating both W mod B and H W mod q, not casually reducing ciphertext moduli. It could change the optimum batch layout, but is **not implemented or benchmarked here**.

Do not extrapolate the Paillier proof of concept to a whole LLM or advertise its offline work as free.

## 5. Removing the Client's embeddings and output head

Removing W from Preparation does not remove the Client's original vocabulary matrix. The original manuscript explicitly keeps token lookup and the output head local. The revised manuscript also requires a full authenticated matrix during secret verifier setup.

The executed toy graph addresses the *functional* part rather than hiding it. It has a 32-token vocabulary, 16-dimensional embeddings, two hidden linear maps, tied embedding/head weights, local clipped ReLU, and greedy next-token selection. The input token is represented by a masked one-hot vector and embedded remotely. All hidden products and the output-head product are also remote. Every stage uses model-free Paillier preparation and encoded corrections.

It produced 32 exact stage results over eight steps with no weight arrays passed to Client or Preparation. The total online payload was 3,728 bytes, compared with 4,608 bytes for the same toy partition without the response codec: 19.10% lower. Offline preparation was 99,728 bytes and dominates the experiment. The repetitive generated token IDs are arbitrary outputs of random weights, not a language-quality result.

This is not yet an efficient full-size token boundary. A masked one-hot embedding query is O(vocabulary) coefficients. At Qwen2.5-0.5B's 151,936-token vocabulary, that baseline query alone would be 455,808 bytes at 24 bits. Returning the entire head similarly produces 455,808 uncompressed bytes per output vector, before codec assumptions. Qwen dimensions are from the official configuration [9].

A practical next implementation should benchmark **single-server PIR** for embedding retrieval so that Preparation need not hold a replica. Existing single-server PIR systems have explicit query, hint, setup, and response costs, and some trade smaller online queries for substantial client hints [7]. A second-server PIR requiring a replicated embedding table would violate the stated Preparation storage objective.

For the head, return the whole coded logit vector or implement a specified private sampling/selection protocol. Running ordinary top-k on independently masked logits is incorrect because masking destroys their order. Private selection can alter the computation partition, trust assumptions, or online role of Preparation; it must not be inserted as an unexplained zero-cost box.

For a transformer, the model compiler must also address normalization scale vectors, biases, routing weights, and all other learned parameters—not just large dense matrices. Small authenticated public configuration can remain local without constituting a full model copy. A stricter zero-learned-parameter Client requires explicitly folding or outsourcing those operations under a defined quantized graph.

## 6. Verifier setup is a separate unresolved requirement

The previous revision proposed secret a and b=W^T a computed from an authenticated signed matrix. That is incompatible with a Client that never reads W unless setup is changed.

Encrypting a and asking Inference to return encrypted W^T a keeps a secret but does **not** prove that the correct W was used. A malicious service could initialize the verifier and execute inference consistently under a substituted W'. A model hash without a proof connecting the computation to it does not repair this.

A complete extension needs authenticated model-owner verification metadata or a verifiable setup protocol/proof bound to the exact model representation. Gupta, Katz and Miers' September 2026 abstract explicitly uses information posted by the model owner for private verifiable open-weight inference [6]. It is directly relevant, but its full protocol was not accessible in this research session, so no implementation claim is made from the abstract.

Streaming the entire model through the Client once would avoid persistent storage, but not the download and setup burden the user wants to remove. It should not be presented as a weightless setup solution.

The prototype's oracle assertions are tests, not this missing proof. Model-free confidentiality and model-authenticated integrity must remain separate claims.

## 7. Prediction without a local model

The encoder can stay unchanged while the Client uses a predictor p based only on its previous local state. Replace the mask side information by

    s' = (s - p) mod q.

Then z=(y-p)-s' mod q. Decode y-p under the same bounded-overflow condition and add p back modulo q. Neither p nor the residual is sent to Inference. A previous output vector is not a copy of the model.

In a deliberately correlated synthetic input random walk, this reduced the tested packet to **5,484 bytes**, a **55.37% response reduction**, with 40/40 exact steps. None of the absolute outputs met the same 10-bit profile; all residuals did. The extra previous-output state is 32,768 bytes in the int64 reference.

The first previous output was obtained by the test oracle. An actual session must bootstrap it using a separately budgeted safe profile. Subsequent autoregressive transformer activations may not be sufficiently correlated, so these data cannot justify a model-level ratio. The correct experiment is a fixed, non-adaptive predictor and profile tested on held-out real-model traces, with adversarial or extreme inputs included.

## 8. Latency, not just bytes

The reference main-profile encoder took about 0.71 ms median and decoder 6.24 ms median in the final captured run. The Numba variant, timed separately after compilation/warmups, took **0.216 ms to encode and 2.016 ms to decode**. Its public uint8 parity matrix is 851,968 bytes; the reference's int64 matrix is 6,815,744 bytes. These are generic coding matrices, not model weights. Both implementations remain research code.

Saving 5,832 bytes saves only 0.467 ms of payload transfer at 100 Mbit/s. Counting the measured encoder and decoder times in full, a simple serial break-even estimate is approximately **20.9 Mbit/s**, not a promised speedup on faster networks. That estimate excludes baseline serialization costs, overlap, batching, device differences, setup, integrity checking, and service scheduling. A native implementation may improve it, but it must be measured.

The code reduces bytes, not the number of sequential inference exchanges. It does not solve WAN round-trip latency.

An architecture-only Qwen body calculation, assuming all four fused stages per layer use 24-bit input/output coefficients and every stage satisfies b=12,t=16, gives:

    client upload per row:             543,744 bytes  (unchanged)
    ordinary body download per row:    912,384 bytes
    coded body download per row:       482,112 bytes
    ordinary online body payload:   1,456,128 bytes
    coded online body payload:      1,025,856 bytes

That is 29.55% online-body reduction, not 47.46% whole-inference reduction. It is arithmetic planning only; actual per-stage widths, outlier budgets, and real model traces have not been measured. It excludes embeddings, head, preprocessing HE, public metadata, verification setup, and framing. It must not be compared directly to the original paper's phase-ambiguous aggregate traffic table.

## 9. Novelty position and rejected alternatives

Compression after encryption using decoder side information dates at least to Johnson, Wagner and Ramchandran's TCC 2004 work [1]. Fleischhacker, Larsen and Simkin provide oblivious sparse encrypted-vector compression [2]. Giorgi, Grenet and Simkin's EUROCRYPT 2026 paper gives deterministic, perfectly correct linear-code constructions with efficient syndrome decoding [3]. The latter is especially close conceptually. Its official abstract and publication information were available; the full ePrint PDF could not be retrieved, so its complete applicability to power-of-two rings is not resolved here. Related small-field work also exists [4].

Accordingly, do **not** claim a new general encrypted-compression primitive, novel BCH decoding, or first HE-based model-free preprocessing. A candidate contribution worth investigating is the concrete composition:

> Dense integer activations reduced to sparse modular overflows, a power-of-two-ring-compatible low-bit/syndrome format, direct execution on encoded offline corrections, client-only predictive side information, and a model-free preprocessing interface.

Priority is not established. The proof and experiment details need external cryptographic review and direct comparisons with the closest constructions before a novelty claim.

The following alternatives were not selected:

- Ordinary compression of pseudorandom masked bytes: zlib enlarged the test payload.
- Lossy masked requantization: the enumerated construction is unbiased stochastic rounding, but changes the function and invalidates the unchanged exact-product verifier. Secure truncation has substantial existing literature and subtle trade-offs [8].
- Low-entropy or low-rank masks selected merely to make traffic compressible: not an acceptable privacy shortcut without a separate security argument.
- Omitting embeddings/head or moving a full model copy to a renamed Preparation worker: does not satisfy the storage objective.
- Reporting online savings while excluding new HE setup and exhausted-inventory costs: not a full cost comparison.

## 10. Recommended order of work

**First: implement and measure public-bound wire widths.** This is the strongest simple baseline and does not depend on empirical outlier rates.

**Second: integrate the encoded-correction interface behind an experimental feature flag.** Test exact comparison with the existing quantized graph, secret-verifier rejection, mask allocation/retry rules, malformed payloads, and out-of-domain behavior. Fix profiles independently of live private activations.

**Third: collect real-model traces and test the codec envelope.** Include prefill/decode separately, every stage, multiple prompt types, extreme inputs, and held-out data. Report recovery rate and the complete failure policy, not only average outlier count. Evaluate the previous-output predictor separately from the zero predictor.

**Fourth: replace the HE microprototype with an optimized backend and authenticated setup.** Measure bytes per usable prepared record, replenishment rate, batch occupancy, cancelled/expired material, state size, and model verification. Evaluate direct coded-correlation generation as a distinct variant, not an assumed improvement.

**Fifth: implement full token boundaries and compare total cost.** Single-server embedding PIR, whole coded logits versus secure sampling, all model-derived local state, client CPU/RAM, preprocessing amortization, and WAN round count belong in the same evaluation.

The appropriate next paper claim is an experimentally supported, conditional network representation and weightless-role construction—not that global private LLM inference has already become bandwidth-efficient.

## Reproduction and evidence limits

All experiments ran in a Linux x86-64 container using Python 3.13.5, NumPy 2.3.5, cryptography 46.0.4, and optionally Numba 0.65.1. No pretrained checkpoint was available, and model downloads were unavailable in the execution environment. Consequently all new vectors, weights, and graph outputs are synthetic. The original paper's Qwen benchmark was not rerun. The experiments use role-separated objects in one process, not network or operating-system isolation.

Run `python run_all.py` after installing `requirements.txt`. Results change in timings and fresh HE randomness. Seeds for synthetic numerical workloads are fixed. `SHA256SUMS.txt` authenticates the captured package files against accidental changes; it is not a protocol security certificate. The original paper files are not modified.

## References

[1] Mark Johnson, David Wagner, Kannan Ramchandran. *On Compressing Encrypted Data without the Encryption Key*. TCC 2004. https://doi.org/10.1007/978-3-540-24638-1_27

[2] Nils Fleischhacker, Kasper Green Larsen, Mark Simkin. *How to Compress Encrypted Data*. EUROCRYPT 2023. https://eprint.iacr.org/2022/1413

[3] Pascal Giorgi, Bruno Grenet, Mark Simkin. *Oblivious Ciphertext Compression via Linear Codes*. EUROCRYPT 2026. https://eprint.iacr.org/2026/329 ; https://doi.org/10.1007/978-3-032-25330-9_13

[4] *Compressing Encrypted Data Over Small Fields*. https://eprint.iacr.org/2023/946 ; https://cs.au.dk/~larsen/papers/SmallFieldCompress.pdf

[5] Pascal Paillier. *Public-Key Cryptosystems Based on Composite Degree Residuosity Classes*. EUROCRYPT 1999. https://doi.org/10.1007/3-540-48910-X_16

[6] Kanav Gupta, Jonathan Katz, Ian Miers. *Private and Verifiable Outsourcing of Open-Weight LLM Inference*. ePrint 2026/1849. Official abstract approved 3 September 2026. https://eprint.iacr.org/2026/1849

[7] Alexandra Henzinger et al. *One Server for the Price of Two: Simple and Fast Single-Server Private Information Retrieval*. USENIX Security 2023. https://www.usenix.org/conference/usenixsecurity23/presentation/henzinger

[8] Christopher Harth-Kitzerow, Ajith Suresh, Georg Carle. *Truncation Untangled: Scaling Fixed-Point Arithmetic for Privacy-Preserving Machine Learning to Large Models and Datasets*. PETS 2025. https://eprint.iacr.org/2024/1953

[9] Qwen Team. Qwen2.5-0.5B-Instruct official configuration. https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/raw/main/config.json

[10] Numba official guide; compilation excluded from warm timing. https://numba.readthedocs.io/en/stable/user/5minguide.html

Source context: Blair Hudson's original PLLM manuscript (11 September 2026) and the supplied revised `PLLM_revised.md` (13 September 2026). Their reported production behavior is not evidence that the changes in this laboratory package are implemented in production.
