# PLLM: one online evaluator, offline model-free preparation, and attacks on cheaper encodings

**Research and executable experiments — 14 September 2026**

## 1. Reset the target

The desired system has a minimal-work Client, a Preparation service that is completely absent during online execution, and one Inference service holding the public model. There should be no second online worker performing another copy of model inference, no homomorphic-encryption backend, and minimal total network traffic. The previous two-worker experiments do not satisfy that architecture. Their privately shared coefficients cannot simply be handed to the remaining worker.

This round implements a functional one-evaluator alternative: arithmetic garbling, with model-dependent setup performed through offline oblivious transfer. It also investigates whether much cheaper encodings could preserve ordinary linear kernels and nonlinear evaluation without these labels. Several such hypotheses fail by explicit attacks.

**Result:** the toy topology works, but the near-native throughput and low total-cost requirements are not met. A matrix-only CPU experiment shows useful amortization of cryptographic label components, not native performance. Offline traffic is currently prohibitive. This is evidence about where to concentrate research, not a private-LLM release.

## 2. What was run

| Experiment | Captured outcome |
|---|---|
| Real Ristretto255 base OT | 512/512 transfers correct |
| Model-free offline matrix products | 4/4 products; 233 exact coordinates |
| Single-evaluator, linear/ReLU chains | 12/12 fresh graphs; 56 exact stages |
| Private multiplication with arithmetic half-gates | 128/128 fresh gates exact |
| Malformed-label and client-reuse controls | 48 malformed labels and 12 second assignments rejected |
| Batched label-component matrix arithmetic | 9,432 independent integer-reference coordinates exact |
| Kronecker packing of two components per wide dot | 1,728 independent integer-reference coordinates exact; slower |
| Exhaustive scalar-bridge lemma | 3,412 functions checked; exactly the 54 affine functions had input-independent difference signatures |
| Published shifted cubic coefficients | 500/500 input masks recovered |
| Exposed fixed-point Fourier phase coefficients | 16,384/16,384 mask phases recovered |
| Copyable scalar ReLU bridges | 256/256 masks recovered by enumeration |
| Deliberately exposed label offset | 24/24 private inputs recovered via forged shifted labels |
| Common commuting basis | All 80 sampled matrix-pair systems had only a one-dimensional commutant |
| Characteristic-two Frobenius square | Additive in all 65,536 pairs; therefore not the missing Boolean nonlinearity |

Full data are in `protocol_results.json`, `differential_results.json`, `benchmark_results.json`, `kronecker_results.json` and `summary_results.json`. Counts of correct samples are not estimates of cryptographic security.

## 3. Candidate A: label-based execution with only one online evaluator

### Published foundation

Arithmetic garbling already provides an appropriate separation of roles. In particular, Dash uses affine labels, public-weight arithmetic and garbled nonlinear gates, with an offline/online split and tensorized label evaluation [1]. ReDASH improves an important missing component: secure scaling in the residue representation [2]. Neither reference supplies a near-native autoregressive LLM result for this experiment.

Here a scalar x in F_p is encoded as a vector:

    L(x) = B + x Delta mod p.

B is a secret per-wire base. Delta is a secret circuit-wide offset. The last component of Delta is one, providing a point-and-permute index whose shift is hidden by the base. Our reference uses p=251 and 18 components: 17 random offset coordinates plus the fixed permutation coordinate. Their approximately 135-bit entropy is a parameter description, not a complete implementation security claim.

For a public matrix W:

    W L(x) = W B + (W x) Delta.

Inference evaluates this directly, using its original public matrix and a narrow batch of label components. It learns neither B nor Delta. Preparation builds encrypted tables that translate the appropriate input labels into fresh labels for f(x). The evaluator needs no intermediate Client or Preparation response.

The implementation uses SHAKE-based label-keyed encrypted rows, random gate identifiers, and a 16-byte consistency tag. These are research instantiations of the label/table structure, not an independently proved garbling construction. A deployed design should use a reviewed implementation and its precise security assumptions.

### No model matrix at Preparation, without HE

Preparation needs W B, not necessarily W. The implementation obtains it using actual oblivious transfer, following the prime-order-group OT approach of Chou and Orlandi [3], instantiated with libsodium Ristretto255 [4].

For each base coefficient bit b, Inference provides one of two vectors through OT:

    t             or             t + 2^b W[:,i].

Preparation's private choice is the corresponding bit of B. It sums the selected vectors; Inference supplies the accumulated random pads so Preparation can subtract them. This yields W B. Neither W arrays nor an ideal matrix-product oracle are supplied to the Preparation objects.

This is ordinary OT-based linear evaluation, not a new primitive. It is deliberately a simple base-OT implementation, not OT extension. The outer test driver initializes both role objects in one process and sees all data for reference comparisons. It does not demonstrate isolated network processes.

Preparation sees products involving public W and could infer W given sufficient information. Model confidentiality is not a goal: the property demonstrated is not requiring a model matrix as a Preparation input or stored replica.

### Online execution and costs

The tested graph is a width-four sequence of public matrix products and signed finite-field ReLUs. It is not a transformer or fixed-point implementation. Twelve independent fresh graph instances use depths two, four and eight.

| Graph depth | Total counted setup bytes | Online Client upload | Online return | Intermediate online messages |
|---:|---:|---:|---:|---:|
| 2 | 114,914 | 72 | 72 | 0 |
| 4 | 229,666 | 72 | 72 | 0 |
| 8 | 459,170 | 72 | 72 | 0 |

Setup counts include OT request/response payloads, garbled tables and client encoding/decoding material. They exclude public matrix delivery to Inference, transport framing, protocol identifiers beyond the implemented gate/session identifiers, and production authentication. All weights already reside at Inference. There are no online Preparation calls and no HE calls in any phase.

The 144 online bytes are fixed because graph input/output width is fixed, not because full-LLM traffic has been shown negligible. Online intermediate state never leaves the evaluator.

A p=251 unary gate occupies 8,550 bytes. The implemented two-half-gate private multiplication uses 17,100 bytes, two keyed row decryptions and no online messages. All 128 fresh multiplication cases reconstruct exactly. This demonstrates a private/private arithmetic building block; it is not a private-attention implementation.

### Offline cost is not small

In the deliberately simple OT construction, an m-by-n matrix with 18 label components requires n*18*8 base transfers, each carrying two m-byte encrypted vectors. Ciphertext payload alone is therefore 288mn bytes.

For an 8,192-square matrix, that is **19,327,352,832 bytes** of encrypted vectors and another **37,748,736 bytes** of OT query points, before final pad sums and nonlinear tables. This is a formula, not an executed setup at that size. It rules out presenting the current setup backend as economical. OT extension could reduce group-operation cost; by itself it does not eliminate this expanded vector payload. A better OLE/correlation construction is required.

## 4. Candidate B: make label redundancy look like a small GEMM batch

### Hypothesis

A decoding matrix-vector product may be dominated by reading W. Evaluating several label components together could reuse that read and cost much less than executing separate matrix-vector products. This is the same broad direction as Dash's LabelTensors [1]; the experiment does not establish novelty for batching.

### Executed CPU test

The same matrix was evaluated on one scalar column, 18 batched label columns, 72 columns, and 18 separate vector calls. Matrices were 2,048, 4,096 and 8,192 square. Weights are small signed integers stored in FP32, with components in [0,250]. Bounds ensure exact integer accumulation: n*7*250 < 2^24. Every timed variant includes a final reduction modulo 251.

The implementation uses four CPU threads, warmups and nine interleaved repetitions. There is no GPU. Independent int64 checks cover 9,432 sampled output coordinates.

At dimension 8,192:

| Variant | Median time |
|---|---:|
| Best of scalar matrix/vector paths | 5.146 ms |
| 18 separate vector calls | 93.222 ms |
| 18 batched components | 13.621 ms |
| 72 batched components | 37.979 ms |

Batching 18 components is 6.84 times faster than separate calls, but still 2.65 times the scalar baseline. Across all three dimensions, the 18-component factor is approximately 2.65–2.97. The 72-component path is approximately 7.38 times the scalar baseline at the largest dimension.

These measurements are not a native INT4/FP8 LLM-engine comparison. The FP32 weight storage, one small field, and absence of nonlinear gates favor a much simpler workload. Real integer semantics may require several CRT residues, secure sign/base extension and rescaling. The 72-column experiment measures more lanes; it does not execute a correct four-residue CRT circuit.

The successful hypothesis is only that label expansion can be amortized considerably. Near-native per-user TPS and GPU-seconds per token still need a matched full-model benchmark.

## 5. Candidate C: Kronecker substitution to pack two label components

### Hypothesis

Pack two component values into a wider exact number:

    packed = a + B b.

Evaluate one wide dot product and extract the two results, choosing a power-of-two base B with enough guard bits to separate signed sums. This might halve the number of matrix lanes without changing the cryptographic encoding.

### Test and outcome

The implementation packs 18 FP32 component columns into nine FP64 columns. Public bounds guarantee that each component sum is smaller than B/2 and the packed absolute accumulation is below 2^53. Balanced extraction recovers both signed components before reduction modulo 251.

All 1,728 independently checked coordinates were exact. However, total evaluation plus extraction was **1.36x, 1.23x and 1.55x slower** at the three matrix sizes. The experiment excludes input-packing cost, so it is already favorable to the hypothesis. FP64 doubles weight storage in this implementation.

Kronecker substitution is valid algebra. It did not provide a performance improvement on the tested CPU backend, and it is not a reason to claim more useful arithmetic fits into a native low-precision operation for free.

## 6. Candidate D: compile the nonlinear function into a scalar remasking program

### Attractive proposal

Give the evaluator an offline-compiled object implementing

    T(u) = f(u+r) - s,

while letting the online activation remain the single native-sized scalar d=x-r. This would preserve the original native linear kernel and remove client-side nonlinear work.

### A conditional obstruction: finite-difference signatures

An evaluator that can copy or freely query T can compute, for every chosen t,

    T(d+t) - T(d) = f(x+t) - f(x).

Both masks disappear. The right-hand side is the input's finite-difference signature under the public function.

For finite cyclic groups, all inputs have the same complete signature if and only if f is affine. To see necessity, write h(t)=f(t)-f(0). Equality with the signature at zero gives f(x+t)-f(x)=h(t); consequently h(x+t)=h(x)+h(t). Conversely an affine function has an input-independent difference signature. For uniform r,s, the full affine translator and masked input are independent of x.

The exhaustive check covered all 3,412 functions Z_q -> Z_q for q=2,3,4,5. Exactly 54 passed, matching the q^2 affine functions in each domain. ReLU had unique signatures on all tested domains. Square over an odd prime had unique signatures; square modulo powers of two retained only a two-way ambiguity.

**Scope:** this is a statement about unrestricted scalar-to-scalar translators and perfect privacy. It is not a lower bound for arbitrary garbling, FSS, HE, hardware-backed execution, or every computational privacy definition. Efficient recovery must be demonstrated separately for a large domain. Our small ReLU bridge is attacked by exhaustive querying; the coefficient attacks below are direct.

A promise to delete T after use does not prevent a software evaluator from copying it. The one-time-program literature explains why enforceable one-time functionality needs more than such a promise [5]. Ordinary garbling does not promise uncopyable execution: it binds the evaluator to the semantic input labels it possesses.

### Direct coefficient attacks

- Publishing the coefficients of (d+r)^3-s reveals 3r as the coefficient of d^2. Since 3 is invertible modulo 2^k, all 500 sampled masks were recovered exactly across 8/16/24/32/64-bit rings.
- For square, the coefficient 2r reveals all but one bit. Exhaustive tests at k=4,6,8,10 leave exactly two candidates per mask.
- Publishing sine/cosine coefficients of a shifted mode reveals its phase. At 30-bit coefficient precision, all 16,384 phases in the tested period were recovered. This exposes the mask modulo that period, not every bit of a 64-bit mask. It is enough to defeat a bounded activation represented over that period.
- A fully executable scalar ReLU translator over Z_256 revealed all 256 masks after its 256 possible queries.

These attacks explain why the preceding two-worker spectral coefficients cannot just become a single evaluator's plaintext coefficient table.

### Negative control on labels

Standard affine garbling keeps Delta secret. I deliberately exposed it and generated shifted labels L(x)+t Delta. The evaluator could now evaluate valid shifted gates, observe their difference signature and recover all 24 tested inputs. This is an attack on the intentionally weakened encoding, not on secret-offset arithmetic garbling.

The useful requirement is: a fast nonlinear interface must prevent an evaluator from obtaining valid evaluations for arbitrary shifted semantics. A one-use identifier checked only by that evaluator does not supply this protection.

## 7. Candidate E: a secret symmetry that commutes with the public weights

### Hypothesis

Let the server operate on Sx, keeping every public W unchanged. If S W = W S, then W(Sx)=S(Wx), with no extra matrix arithmetic. Perhaps a large family of secret S could hide inputs at native speed.

### Test and scope

I solved the joint commutant equations for 80 random matrix pairs over F_257 at dimensions 3,4,5,6. Every joint commutant had dimension one: only scalar multiples of the identity remained. Scalar masks preserve projective coordinate ratios; a separate public-codebook test identified all 256 masked inputs from these invariants.

This rejects the tested same-basis, unchanged-weight proposal. It does not prove that every transformed-model encoding or every particular pretrained matrix has a trivial commutant. Changing the weights and nonlinear operators creates a different scheme whose security and setup must be analyzed separately.

## 8. Candidate F: use characteristic-two Frobenius as a free nonlinearity

### Hypothesis

In a binary extension field, squaring is an inexpensive local operation. Could a Frobenius-based activation replace the expensive private nonlinear path?

### Test and outcome

In GF(256), using the irreducible polynomial x^8+x^4+x^3+x+1, every one of the 65,536 pairs satisfies

    (a+b)^2 = a^2+b^2,

where addition is XOR. Squaring is F_2-linear. A graph made only of such maps and public F_2-linear arithmetic remains F_2-linear, rather than reproducing the required Boolean nonlinear model computation. The tested cubic fails additivity on 64,770 pairs, and the signed-ReLU table on 48,768 pairs.

This is a valid rare-algebra trick for linear maps; it does not supply a free replacement for SiLU, comparisons or ordinary fixed-point multiplication. A retrained model using different operations would be a different research objective.

## 9. The strongest adjacent literature leads

### Projective encodings and Duty-Free Bits

The March 2026 *Duty-Free Bits* preprint studies converting Yao bit labels into affine encodings used by arithmetic garbling over large prime fields, using symmetric-key techniques [6]. It also gives a vector-OLE connection. This is a relevant lead for reducing representation-conversion overhead without HE.

The source comparison here is based on the official abstract, not an implementation or proof reproduction. It does not establish that arbitrary writable masked native scalars can safely become garbled inputs: that would evade the input-binding issue rather than solve it.

**Next experiment:** construct a matched large-field/bit-label conversion and compare its complete online arithmetic, conversion bytes, garbled nonlinear costs and rescaling with the small-prime/CRT path. Keep the input-label binding and count every component.

### LPN trapdoored matrices

Braverman and Newman's TCC 2025 work studies pseudorandom matrices with trapdoors for fast multiplication, and uses them for low-overhead secure delegated linear algebra [7]. Its protocols retain client-side computation and communication per delegated input; it is not a full remote nonlinear inference solution.

This provides an additional research direction for the dominant offline linear-evaluation cost. Integrating it into the model-free label-base setup is unimplemented. In particular, a construction whose client receives both operands is not automatically an OLE protocol for our separate model holder and mask holder. The required two-party transformation, assumptions and model/no-model boundary need to be specified.

### ReDASH and wide integer semantics

ReDASH addresses secure scaling and conversion costs in arithmetic garbled inference [2]. These are not incidental: a p=251 ReLU table is not a valid independent ReLU on each limb of a multi-prime encoded integer. Global sign, wrap and rounding have to be handled under the stated integer representation.

**Next experiment:** one complete fixed-point decoder block with reviewed nonlinear and rescaling gadgets, plus its one-use preparation inventory. Do not extrapolate matrix-lane microbenchmarks directly to sustained token throughput.

## 10. Decision

The topology to investigate is now:

    Offline:
      Preparation <-> Inference: oblivious label-base setup
      Preparation -> Inference: one-use garbled nonlinear material
      Preparation -> Client: small input/output encoding material

    Online:
      Client -> Inference: encoded request
      Inference: public weights over protected labels, local garbled gates
      Inference -> Client: encoded output

This contains one online inference worker, no HE, and no online Preparation. A complete autoregressive execution would need protected embedding access, sampling, KV state, private/private attention products and a bounded one-use execution schedule. Those are not demonstrated by this package.

The two priority bottlenecks are **the expansion of online encoded arithmetic** and **the rate/cost of replenishing offline nonlinear/setup material**. Batched labels show useful amortization; model-free OT proves a role separation but is far too expensive as written. No evaluated configuration achieves the complete ideal or establishes first-in-literature novelty.

The most useful conceptual result is the finite-difference attack criterion. It makes the search more disciplined: an elegant scalar remasking function that is freely executable on altered inputs can expose precisely the information it was intended to hide. Future fast schemes must preserve semantic input binding, rather than just hide a scalar behind an additive pad.

## 11. Security and reproducibility boundaries

All protocols execute in one process with role-separated objects and independent plaintext references. Fresh cryptographic masks and OT randomness use operating-system randomness; synthetic fixtures use reproducible PRNG seeds. Timing samples are included. This is not an audited deployment.

The exact checks prove implementation identities on their tested data. They do not establish malicious security, model authenticity, composability, correctness of the chosen cryptographic parameterization, rollback protection or constant-time behavior. The consistency tag rejects tested corruptions, but does not turn the experiment into a verified-inference theorem. Preparation and Inference must not combine their views.

No LLM weights are downloaded. No GPU is used. Tests omit a complete fixed-point transformer, native quality comparison, private embedding/head, KV-cache lifecycle, authentic public-model commitment, secure sampling, transport overhead and sustained replenishment. Client “minimal work” is demonstrated only for the toy graph boundary. Removing Preparation entirely is not demonstrated; shifting its heavy work to the Client would violate the stated goal.

## Primary references

[1] J. Sander, S. Berndt, I. Bruhns, T. Eisenbarth. **Dash: Accelerating Distributed Private Convolutional Neural Network Inference with Arithmetic Garbled Circuits.** https://arxiv.org/html/2302.06361v2 . Arithmetic labels, half-gates, CRT, LabelTensors and offline/online topology are established here and in its cited foundations.

[2] **ReDASH: Fast and efficient Scaling in Arithmetic Garbled Circuits for Secure Outsourced Inference.** https://arxiv.org/html/2506.14489v1 . Scaling and representation work; not a near-native LLM result.

[3] T. Chou, C. Orlandi. **The Simplest Protocol for Oblivious Transfer.** https://eprint.iacr.org/2015/267 . Our implementation is an unaudited prime-order-group functional reference, not a claim to instantiate every security property in the literature.

[4] Libsodium. **Ristretto.** https://doc.libsodium.org/advanced/point-arithmetic/ristretto . Official API documentation for the underlying group operations used in the actual OT implementation.

[5] S. Goldwasser, Y. Kalai, G. Rothblum. **One-Time Programs.** CRYPTO 2008. https://www.microsoft.com/en-us/research/publication/one-time-programs-2/ . Supports the distinction between software promises and enforced one-time functionality; not an impossibility statement about all private inference.

[6] N. Khambhati, A. Bhattacharya, D. Heath. **Duty-Free Bits: Projectivizing Garbling Schemes.** March 2026 preprint. https://eprint.iacr.org/2026/476 . Comparison based on its official abstract; not implemented here.

[7] M. Braverman, S. Newman. **Practical Secure Delegated Linear Algebra with Trapdoored Matrices.** TCC 2025. https://arxiv.org/html/2502.13060v3 . A relevant LPN-based direction with a different operand/interaction boundary; not treated as a drop-in garbled nonlinear protocol.
