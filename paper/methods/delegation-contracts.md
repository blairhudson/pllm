# Fidelity contracts: Gupta–Katz–Miers and Maverick

Review date: **17 October 2026**. Status: **implementation specification; approval gates unresolved**.

The default is an independent implementation of each paper's **private and
verifiable construction**, original numerical representation, and original
evaluated model family. Existing PLLM components are building blocks, not already
qualified reproductions. In particular, this contract does not adopt the older
A/C-first recommendation in [COMPETITOR-QUALIFICATION.md](../COMPETITOR-QUALIFICATION.md)
as permission to replace GKM's original computation.

Only this document is changed. Review covered both complete cached PDFs, the
[ten-method audit](../TEN-METHOD-COVERAGE.md), the qualification card, quarantined
GKM source extracts, and the current local sources listed below. No model,
operator, benchmark, build, or upstream implementation was executed. Proposed
interfaces, tests, and admission rules below are requirements for a later
authorized implementation, not evidence that they exist.

## 1. Frozen sources and citation convention

Page numbers mean **one-based PDF pages**, selected with `pdftotext -f/-l`.
`G p.N` refers to the GKM PDF; `M p.N` refers to the Maverick PDF. Section,
figure, protocol, and table numbers disambiguate the claims. Both PDF hashes were
recomputed in this review.

| Source | Frozen identity | Primary locators |
| --- | --- | --- |
| **G** — Gupta, Katz, Miers, *Private and Verifiable Outsourcing of Open-Weight LLM Inference* | [ePrint 2026/1849](https://eprint.iacr.org/2026/1849); [cached PDF](../../papers/r19-open-weight-verifiable.pdf), 17 pages; SHA-256 `e1863980621e0bbd2d446b43f2d9415318ebeec400b0e6c818685752f6448e1d`. Exact revision correspondence of the unversioned URL remains unspecified; the earlier fresh-PDF request failed with HTTP 403. | §3/threat model and §4/encoding, p.6; §4.1/Fig.1, pp.6–7; §4.2/Fig.2 and batching, pp.7–9; §4.3/Figs.3–4, pp.9–10; §4.4/Fig.5, pp.10–11; §5, pp.12–14; §6/MoE discussion, pp.14–15. |
| **M** — Merbaum, Raeisi, Wang, Papamanthou, Sotiraki, Zhang, *Maverick: Private and Verifiable LLM Inference Made Practical via Matrix–Vector Multiplication Delegation* | [arXiv:2609.10264v1](https://arxiv.org/abs/2609.10264v1), 9 September 2026; [cached PDF](../../papers/r24-maverick.pdf), 26 pages; SHA-256 `c9570fdeb51a26f9d707bcc7bf485f2001e7d3e15ebdf34815123935a1e8e2a0`. | §4/definitions and transparent preprocessing, pp.5–6; Protocol 1/Lemma 5.1/Theorem 5.2, pp.6–8; §5.4/Protocol 2, pp.8–9; §6/Protocol 3/Eq.(1), pp.10–11; §7.1, pp.11–12; §7.2/Figs.1–2/Tables 6–9, pp.12–16; §7.3/Table 10, pp.15–17; Appendices A/B, pp.22–24; Appendix C, pp.24–25; D/E, pp.25–26. |

### 1.1 Author code is documentary input

GKM author code is pinned to
[`kanav99/dvllm@271637be7196d26a5433ca1e62b1a5357e769a5b`](https://github.com/kanav99/dvllm/tree/271637be7196d26a5433ca1e62b1a5357e769a5b),
tree `e742e67ed278aba39a440a3d78c782134fe8732c`, commit dated 1 October 2026.
`U path:lines` below denotes that commit, not PLLM. The
[code lock](../artifacts/source-review/2026-10-16-gkm-qualification-v1/code-lock.json)
contains full-file hashes and bounded extracts of twelve selected files;
the [archive README](../artifacts/source-review/2026-10-16-gkm-qualification-v1/README.md)
defines the quarantine. Identity with the authors' evaluated build is unknown.
Its README's introductory single-server/local-embedding description does not
override the two-server protocol or the inspected private-lookup path.

No Maverick author-code commit is established by the reviewed cache. Claims below
about Maverick's implementation are either explicitly attributed paper statements
or observations about **PLLM's independent reference**. This is not a finding
that Maverick author code is unavailable. Obtaining an immutable author-code
snapshot would help resolve underspecified numerical and batch parameters;
upstream remains text/oracle input, never a runtime dependency.

Integrity pins for the existing GKM review:

| Record under `paper/artifacts/source-review/2026-10-16-gkm-qualification-v1/` | SHA-256 |
| --- | --- |
| `code-lock.json` | `0575902701e9eae051965a2b1cd651e0ef30dcb70349248739cfe5e2b46c04ab` |
| `primary-locators.json` | `57b9c3807d9542b89d292dc6906cdcd6d333d55568373297ecbf32d3b7e1b6a2` |
| `local-locators.json` | `88ced8251bb25a6cc09e7dbc5ace838e34b0a3acefb2357f56eec3fc6513e62e` |
| `repository-record.json` | `9d1eb8ba3ed5057303f2b53260c5771fb96e12a15786c9a63026e3911011473c` |

## 2. Gupta–Katz–Miers contract

### 2.1 Roles, assets, adversaries, and leakage

**Source contract — G §3, p.6; Fig.5, p.11.** There are an honest model owner,
a client C, and two public-weight servers S0/S1. The same two servers execute
embedding lookup, body linear operations, and the output classifier. Each holds
the public matrices/table; the client keeps its tokens, reconstructed activations,
nonlinear computation, outputs, and verification state. The model owner publishes
verification material once, independently of client inputs.

- Input privacy requires S0 and S1 **not to collude**, including their combined
  retained lookup/body/head transcripts. Either server may be malicious. Both
  malicious, colluding servers are allowed for the paper's **integrity** claim,
  under the honest-owner/discrete-log/hash assumptions. Collusion does not retain
  input privacy. A malicious client is immaterial to server confidentiality
  because the model is public and the paper imposes no such server requirement.
- The model-owner setup scalars must remain unavailable to adversarial servers;
  published group elements are public. Owner/source authentication and binding
  commitments to the intended model are necessary deployment obligations.
- Uniform Fig.1/Fig.3 shares give the stated single-server privacy algebra.
  The §5 PRG-seeded-share optimization changes this to computational privacy
  under the PRG assumption (G p.12); its entropy and stream lifetime must be bound.
- Public model identity, geometry, stage order, request/prompt lengths, message
  sizes, timing, termination and failure traffic are explicit deployment leakage.
  The paper does not give a complete network/side-channel leakage theorem.
  Request-dependent expert choices additionally leak in MoE unless all experts
  are queried (G §6, p.15). No MoE routing extension is part of the initial scope.
- Fig.5 queries embeddings without a cache optimization. The inspected author
  cache can reveal query omission/cache hits. Default to Fig.5's query schedule;
  enabling a source-code cache requires an approved leakage/cost variant.
- Confidential, authenticated, separately credentialed client-to-worker channels
  and authenticated public artifacts must be specified before remote admission.
  An authenticated HTTP primitive is not evidence of confidential deployment or
  independent operators. Two local processes suffice for conformance only.

An implementation that drops verification must be named a **semi-honest
restriction**, with no malicious-result integrity guarantee. A prepared-body
dealer or four-provider topology is not the paper's two-server construction.

### 2.2 Normative computation and numeric boundary

The following specifies G §§4.1–4.4 rather than PLLM's current W8A8 boundary.
Section 5 separates source-compatible completions, missing evidence, and changes
that need approval.

1. **Import and quantization.** For each real linear coefficient/input `v`, use
   `Q(v) = nearest_integer(2048 * v)`. Weights are signed int16; linear inputs
   and reconstructed outputs use signed int64 representatives of `Z/(2^64)`.
   G §3/§4, p.6 gives nearest-integer encoding; G §5, p.12 specifies the types.
   The paper does not define halfway ties or all floating-point evaluation details.
   Explicit completion: ties-to-even, exact binary scaling of finite imported
   values, reject unrepresentable weights/inputs rather than clip. This defines
   the PLLM source-derived profile, not the inspected author's truncating casts.
2. **Share and evaluate.** Sample fresh uniform `a` in `(Z/(2^64))^n`, set
   `b = Q(x) - a mod 2^64`, send one share to each server, and reconstruct
   `z = Mq*a + Mq*b mod 2^64` (Fig.1, G pp.6–7). Every multiplication/addition
   on shares uses defined unsigned modular arithmetic. Negative int16 weights
   are lifted into that ring. Signed-language overflow is never an implementation
   of modular arithmetic.
3. **Signed lift and rescale.** Interpret `z` in `[-2^63,2^63-1]` and convert
   it to float32 before division by `2048^2`, preserving the inspected client's
   conversion order. The no-final-overflow condition must be established per matrix and
   admitted input bounds: a sufficient condition is
   `sum_j abs(Mq[i,j]) * input_bound[j] < 2^63` for every row. Partial share
   products may wrap; the true unmasked integer product must not. The exact
   int64-to-float conversion and operation order belong in the numeric manifest.
4. **Embedding.** For token `w`, construct a dense V-bit one-hot selector `e_w`.
   Sample uniform V-bit `r`; send `r` and `r XOR e_w`. Servers XOR selected
   **bit-string** rows, not numeric sums, and the client XORs the two responses.
   Fig.3 step 2 has a self-reference typo; the prose on G p.10 explicitly gives
   `x(1) = x XOR x(0)`. Adopt that prose and record the correction. No DPF, HE,
   garbling, quantized row/scales, or local clear token disclosure is equivalent
   source execution (G pp.9–10).
5. **Row bytes.** The paper permits floating-point table rows without fixed-point
   conversion (G p.10). The inspected source uses `dim*sizeof(float)` float32
   bits (`U include/client.hpp:288–340`); the quantizer stores float embeddings
   and a separately quantized int16 classifier even for tied original weights
   (`U src/quantize.cpp:123–145`). Freeze float32 byte order and hash those exact
   bytes. Do not canonicalize signed zero or reserialize through a different
   floating type. Any non-finite source-value policy must be explicit.
6. **Remaining graph.** Client executes the original model's local operations,
   including attention, normalization, rotary position operations, activations,
   residuals and sampling; full vocabulary logits return through delegated
   linear computation. Biases, if present in the selected model, need explicit
   source-compatible placement and order. No existing Qwen-specific coefficients,
   RMSNorm epsilon, clipping, or fused-stage schedule may be inherited by name.

**Paper/code discrepancy.** G p.4's overview says “scale and truncate,” while
the formal definition on p.6 says closest-integer rounding. The pinned source
casts scaled values to integer types, which truncates toward zero when
representable (`U src/quantize.cpp:123–145`, `include/client.hpp:211–239`).
Neither paper nor reviewed paths establish safe out-of-range cast behavior.
Default to the formal nearest-integer paper construction; an author-code-numeric
replica is a separately identified track. Neither equals per-row W8A8/ring32.

### 2.3 Verification and acceptance

**G §4.2/Fig.2, pp.7–9; §4.3/Fig.4, p.10; Fig.5, p.11; §5, p.12.**

- Use the source's secp256k1 prime-order group. Let `p` be its scalar order,
  with `p >> 2^64`. The honest owner samples `r[0:m_max]` uniformly in `F_p`,
  publishes `R_i = g^(r_i)`, and for each matrix publishes
  `S_j = g^((M^T*r)_j)`. The owner may delete secret `r` after publication.
  Clients authenticate/bind dimensions, numeric matrix digest, global R, per-stage
  S and embedding hash table to the selected checkpoint and graph.
- Verify the **reconstructed integer relation** using
  `product_j S_j^(x_j) == product_i R_i^(y_i)`.
  Negative integers must map consistently into `F_p`. Do not take unsigned
  ring64 residues directly as prime-field exponents: `2^64-1` and `-1` are
  equal in ring64 but not in `F_p`. Verify individual uniform-share products
  only with a separately justified wrap-correction relation; the proposed
  minimal path checks reconstructed products under the no-overflow bound.
- For each fixed response, draw fresh secret uniform `alpha in F_p` and update
  per-matrix input/output accumulators modulo p. Combine repeated calls and then
  different matrices as described on G pp.8–9. Bind every included claim, count,
  matrix identifier and acceptance window. No predictable scalar, replayed
  challenge, omitted claim, or unbounded integer accumulator substitutes for this.
- Owner publishes `SHA-256(row_bytes)` for each embedding row. Compare recovered
  bytes to the authenticated hash indexed by the **client-private** token before
  using the row. A bad row causes immediate abort (G pp.10–11).
- Fig.5 permits dependent local computation before the final batched linear
  check, but outputs only after acceptance. Proposed implementation keeps a
  tentative state/output transaction: failed verification retires it, queued
  responses and all associated private material. No accepted text, reusable KV,
  committed sampler state or continuation is released from a failed window.
  A smaller verified streaming window changes batching/latency and needs approval.
  Syntactically invalid, oversized or non-finite dependent values fail closed
  without data-dependent diagnostic replies to workers.

**Inspected-path departures, not findings about every author entrypoint:**

| Pinned source evidence | Required treatment |
| --- | --- |
| `U src/llama3.2q.cpp:627–640`, `include/verification.hpp:59–64`: seeds derived from `rand()`; verifier draws an int64 coefficient rather than the paper's uniform `F_p` coefficient. AES counter expansion is visible at `ext/simdcrypt/src/PRNG.cpp:43–75`. | Independently implement reviewed entropy acquisition and unbiased scalar sampling. Separate privacy, lookup, verification, sampler and session streams. Record correction to inspected code; do not inherit its entropy assumptions. |
| `U include/client.hpp:377–399`: embedding mismatch prints a diagnostic, then caches the row. `include/verification.hpp:147–165`: failed matrix checks print diagnostics. | Implement Fig.5's rejection contract, propagate failure to the request owner, and prevent cache/state/output acceptance. This is fidelity to the paper, not a claim that all author wrappers fail. |
| `U src/llama3.2q.cpp:662–695`: generation precedes final `Verify()`; queue draining and commitment loading precede its printed verification timer. | Distinguish tentative generation from accepted output. Charge the complete verifier lifecycle, including publication/delivery, queue copies, accumulation, loading and final checks. |

PRG seed compression and batched prefill are source-present optimizations, not
optional invisible advantages. First conformance may use full shares; a qualified
source-performance implementation must implement the G p.12 seeded-share path
and declared batching or disclose and approve their omission. A seed is shared
only with its designated worker; complementary-share recipients must never get
that seed. Freeze AES mode/counter packing, key length, domain separation,
consumption order, restart behavior and exhaustion limits in a versioned format.

### 2.4 Original models, output, and persistent state

G Table 2, p.13 evaluates **Llama-3.2-1B/3B, Llama-2-7B/13B/70B,
Llama-3.1-8B, Gemma-3-1B/4B/12B/27B**, with costs per token amortized over
100 generated tokens. Its verification column covers all 100 tokens, not one.
G Table 1, p.13 reports Wikitext quality for Llama-3.2-1B/3B,
Llama-2-7B/13B and Gemma-3-1B/4B/12B; it does not give quality results for
all Table 2 models. G Table 3, p.14 separately evaluates 32/64-token prefill on
Llama-2-7B. The reviewed paper does not pin checkpoint revisions, tokenizer
bytes, precise Wikitext split/processing or sampler configuration.

The pinned code establishes full client float32 KV and full vocabulary logits
(`U include/client.hpp:56–76`), last-position logits from batched prefill
(`682–689`), and client-local attention plus remote classifier (`403–538`).
The inspected sampler supports greedy, temperature, multinomial and top-p;
top-p includes the threshold-crossing token and stochastic sampling mutates
logits in place (`U src/llama3.2q.cpp:313–436`). RMSNorm epsilon `1e-5` is
hardcoded in the inspected Llama path (`include/client.hpp:142–175`). These
observations do not qualify Gemma support or all checkpoint configurations.

**Default model anchor:** independently imported Llama-3.2-1B, the smallest
paper-evaluated Llama model, after revision/tokenizer/config pins are approved.
The formal nearest-integer oracle remains distinct from the author-code cast
oracle. Full study coverage must eventually include the intended matched workload;
starting here does not replace it with a toy or an easier model.

Freeze state transitions explicitly: consumed input tokens, position, active
K/V entries and lengths, masks/rotary positions, full pre-sampling logits,
selected-but-not-yet-executed token, sampler state, EOS state, continuation owner,
and tentative/accepted verification windows. The paper and inspected code do not
specify PLLM's cancellation, replay or continuation API. Proposed semantics must
be approved and then tested against the same clear numerical oracle.

## 3. Maverick contract

The [Maverick executable companion](../reproduction-contracts/maverick/README.md)
now freezes explicit code/field/capsule parameters and independent scalar vectors
for Protocols 2/3, with 19 runnable checks. Its
[source lock](../reproduction-contracts/maverick/source-lock.json) rechecks the
paper and existing Rust snapshots. This advances bounded arithmetic conformance;
model numerics, model-scale code certificates and persistent decode remain gated.

### 3.1 Roles, adversary, preprocessing, and leakage

**M §4, pp.5–6; §6, pp.10–11; §7.2/Fig.1, pp.12–13.** One client delegates
public-matrix linear operations to **one untrusted server**. The client retains
its input, local nonlinear values, privacy noise/corrections and verification
transcript; the server gets masked inputs and returns field products. Base
Maverick has no model-confidentiality requirement.

- Verification is information-theoretic **conditional on the actual code's
  distance and fresh challenge distribution**. Privacy is computational under
  dual-LPN for the actual privacy-code/noise distribution and polynomially many
  fresh samples. M Remark 4.3, p.6 gives a multi-query distinguishing-advantage
  factor equal to the number of queries; this is not an unlimited-use certificate.
- Preprocessing is transparent: `Gx`, `Gy`, `P = M*Gx` and `Q = Gy^T*M` are
  public and contain no secret trapdoor or client-specific verification key.
  Server or third party may publish them. An untrusted publication must be
  validated against **the intended M**, at preprocessing-scale cost, or have a
  separately verified proof (M §4.3, p.6). A hash of an unvalidated P/Q pair
  cannot establish correct preprocessing. Code-distance/distribution admission
  is additional to checking these algebraic equalities.
- Public matrix/architecture, dimensions, schedule, lengths, batch shapes,
  traffic sizes, timing and termination are permitted deployment leakage. The
  source's formal single-query view is `(pp, M, masked_x)` (M p.6); validate
  the whole interactive decoder/batch/abort view separately. No side-channel,
  transport-authentication or availability guarantee follows from that definition.
- Appendix C's private-client-owned-matrix extension uses **private, client-run
  preprocessing and a trapdoored matrix** (M pp.24–25). It is a different
  construction and is not the evaluated public-model default.
- Verification-only mode discloses the input and lets the server execute the
  nonlinear graph; client replays nonlinear operations and checks the transcript
  (M pp.13–14, Fig.2). It cannot fill a private-inference comparison cell.

### 3.2 Exact field protocol and code distribution

For `M in F_q^(m x n)`, choose independent codes
`Gx in F_q^(n x Nx)` and `Gy in F_q^(m x Ny)`. The evaluated configuration is
**BabyBear** `q = 15*2^27 + 1 = 2,013,265,921`, `Nx = 8n`, `Ny = 8m`,
target distances `delta_x = delta_y = 1/2`, `lambda = 128`, `kappa = 40`
(M §7.1, p.11). These are prime-field operations, not arithmetic modulo `2^32`.

Protocol 3 (M p.10) requires:

1. **Preprocess:** compute/publicly validate P and Q as above and bind their
   field, dimensions, code descriptions, source matrix digest and parameter set.
2. **Encrypt:** independently sample exact-weight `e_x` uniformly from all
   `t_x`-sparse vectors of length Nx, with uniform support without replacement
   and independent uniform **nonzero** field coefficients. Compute
   `x_prime = Gx*e_x`, `masked_x = x + x_prime`. Keep `e_x` private.
3. **Evaluate:** server returns `masked_y = M*masked_x` over `F_q`.
4. **Verify:** after the complete claimed response is fixed, compute
   `z = Gy^T*masked_y`. For each independent challenge `e_y` of exact weight
   `t_y`, test `(e_y^T*Q)*masked_x == e_y^T*z`. Reject if any check fails.
5. **Decrypt:** `y = masked_y - P*e_x`. Computing and retaining `P*e_x` early
   is algebraically equivalent; its lifetime and memory must be charged. Use a
   one-use owner; malformed replies, cancellation and failed checks retire it.

Appendix A (M pp.22–23) distinguishes exact-weight noise from Bernoulli or
structured regular noise. Do not use fixed supports, public noise seeds, zero
coefficients, or reused masks. Appendix E's subspace-leakage example (M p.26)
is a concrete reason not to replace dual-LPN masks with a small public mask span.

**RAA construction — M Appendix B, pp.23–24:**
`G = R * Pi1 * V1 * A * Pi2 * V2 * A`.
R repeats each entry eight times; each Pi is an independent uniform permutation;
each V has independent uniform nonzero diagonal entries; A is the upper-triangular
prefix-accumulation matrix. Implement `G^T*w` by repeat, permute, scale,
prefix-accumulate, permute, scale, prefix-accumulate. Implement `G*e` by the
actual transpose, including suffix accumulations and inverse data movement.
Use explicit code descriptions and an independent dense-matrix oracle in tiny
tests; never construct full dense G at model scale.

### 3.3 Parameter and security qualification blockers

These are **source applicability gaps**, not measured attacks on Maverick.

1. **Field versus cited bound.** Appendix B, M p.24 quotes a failure bound for
   **`q > 2^31`**, target distance 0.5 and repeat factor r, with a polynomial-time
   distance-testing refinement. Evaluated BabyBear is **less** than `2^31`.
   That cited bound therefore does not directly certify the evaluated field.
   A valid actual-field/actual-dimension distance bound or independently reviewed
   certificate is needed. The companion specifies **`q=2^31+11=2,147,483,659`**,
   the smallest qualifying prime, as a concrete **unapproved separate profile**.
   Its primality and u64 reduced multiply-add bounds are independently tested.
   It closes the printed field hypothesis only: asymptotic failure bounds still
   lack finite constants/dimension thresholds. BabyBear stays the source default;
   existing native modular kernels also cap their modulus below `2^31`.
   The current ePrint 2026/1075 record states a broader `r>=4`,
   `q-1>=gamma*N` scaled-RAA result that could preserve BabyBear. Its full PDF
   remains unverified. The newly retrieved author mirror is an **older** version
   requiring `r>=9` and `q-1>=(eN)^2`; it does not certify Maverick's r=8.
2. **31 versus 41.** M §7.1, p.11 chooses `t_y=31`, `ell=2` and attributes
   this to Theorem 6.1. Substituting `kappa=40`, `delta_y=1/2` and BabyBear
   into the theorem's printed sufficient formula (M pp.10–11; also Theorem 5.2,
   p.7) yields `t_y=41`, `ell=2`. This is an internal parameter-description
   discrepancy, not by itself a soundness break: Lemma 5.1 gives the conditional
   bound `epsilon = ((1-delta_y)^t_y + 1/(q-1))^ell`; with t=31 it is about
   `2^-59.905`, already below `2^-40`, assuming distance 1/2. With t=41 it is
   about `2^-61.811`. Neither calculation supplies the missing code-failure
   probability. Default evaluation-faithful parameters are 31/two checks,
   explicitly justified by the lemma rather than misquoting the theorem formula.
3. **Privacy parameter is heuristic.** Eq.(1), M p.10 chooses
   `t_x >= ln(2)*(lambda-log2(Nx))/delta_x`. For dimensions
   `n={1024,2048,4096,8192,16384}`, the evaluated values are
   `{160,159,157,156,154}` (M p.11). Appendix A, pp.22–23 describes the
   linear-test framework and its limitations. This is not a universal concrete
   128-bit security proof. Non-square decoder dimensions and the total query
   horizon need a reviewed actual-code/noise parameter record. The concrete
   source-derived 64-by-64 capsule uses `t_x=165`, Nx=512; infeasible small
   dimensions must reject rather than clip the calculated weight. Exact-weight
   sampling and Eq.(1) are independently checked in the companion, without
   equating the heuristic to a general `2^-128` per-sample advantage.
4. **Ensemble matters.** The source construction samples full uniform
   permutations and field scales. PLLM derives its code from public SHA-256
   seeds; no reviewed reduction transfers the quoted ensemble bounds to this
   seeded distribution. Default to the explicitly specified source ensemble.
   Freezing a sampled code makes it reproducible but does not prove its distance.
   Do not promote sparse-vector or corruption tests to a distance certificate.
   Untrusted publication of algebraically correct P/Q does not establish honest
   code selection. Bind independently controlled sampling or an instance proof.
   The companion's two `[16,2]` examples have **exact distance 15**, proved over
   all nonzero messages by projective-column counting; these tiny instance
   certificates do not transfer to any 64-dimensional or model code or dual-LPN.
5. **Whole-execution budget.** M Remarks 5.3/6.2, pp.8/11 add code-distance
   failure probability to conditional soundness. For each code instance, record
   its admitted failure bound and reuse horizon. Bound the union across distinct
   code instances and verification windows, plus multi-query privacy loss;
   count malformed/aborted attempts as applicable. A per-object counter reset
   cannot reset a deployment security budget. For W same-parameter Protocol-2
   batches, the conditional contribution is `W*2*((1/2)^31+1/(q-1))^2`.
   At W=65,536 and BabyBear it is about `2^-42.9054`, before any code-failure
   term; at W=`2^19` it already exceeds `2^-40`. W counts every shape batch,
   not merely complete model requests. This is illustrative accounting, not
   admission of the historical per-object attempt cap as a deployment lifetime.

Source-faithful implementation can proceed as an explicitly **uncertified
algebra-conformance artifact**. The companion supplies that independent scalar
artifact and exact sample-error arithmetic now. Its tests do not turn missing
proof evidence into a certified private/verifiable result.

### 3.4 Batch verification, modes, and acceptance

The source system uses **Protocol 2**, not just Protocol 3 independently at every
layer (M §§5.4/7.2, pp.8–9/12–13):

- Group B claims with the same `(m,n)`; use shared primary verification code G.
  For each matrix, `Q_b = G^T*M_b`. Stack
  `Q_tilde = [Q_1 | ... | Q_B]^T`, shape `(B*n) x N`.
  Choose secondary code `G_prime`, shape `(B*n) x N_prime`, and preprocess
  `Q_aux = Q_tilde^T * G_prime`, shape `N x N_prime`.
- **First fix every original response.** Sample independent primary challenges
  `e^(1)..e^(ell)` and send them to the server. The server returns every
  `u_tilde^(q) = Q_tilde * e^(q)`. These public post-response challenges are not
  the secret privacy noise.
- **Then fix every auxiliary response.** Sample independent secondary challenges
  `e_prime^(j)`. Check
  `(e^(q))^T * Q_aux * e_prime^(j) == (G_prime*e_prime^(j))^T * u_tilde^(q)`
  for each pair `(q,j)`, followed by every original claim
  `(u_b^(q))^T*x_b == (e^(q))^T*G^T*y_b`.
  Both checks are required. Shared primary challenges across B fixed claims
  introduce no B factor in that batch's theorem bound; changing response order
  invalidates the argument (M pp.8–9).
- The conditional batch error bound is `epsilon^ell + epsilon_prime^ell_prime`,
  plus the applicable code-failure probabilities. Freeze secondary code family,
  rate, sparsity and repetition parameters: the reviewed evaluation does not
  supply a complete operational manifest for them. The companion freezes a
  concrete B=2, m=n=64 capsule: secondary dimension 128, independent RAA8,
  N_prime=1024, target distance 1/2, t'=31 and ell'=2. This is a
  source-compatible Protocol-2 instantiation for conformance, not recovered
  evaluated secondary settings; model batching/evidence still need closure.
  A systematic-code storage optimization is described on M p.9 but is not
  permission to silently change
  the evaluated construction or its resource attribution.
- Batch verification adds a challenge/response round, server projection work,
  B*ell*n reply field elements, and Q_aux storage. Calling it “zero server
  overhead” from the base single-MVM description omits this cost (M p.9).

**Standard mode:** masks generated during inference; pipeline the next mask
while the server handles the current step (M p.13). **Boost mode:** fresh masks
and their corrections precomputed; charge generation, validation, private
inventory storage, unused material and retirement separately (M pp.12–13).
The prose says masks are independent of model/input, but `P*e_x` corrections
are visibly **matrix-dependent** in Protocol 3/Fig.1; they cannot be shared
across different matrix bindings.

M §7.2.1, p.12 and Fig.1, p.13 define optimistic verification: compute local
dependent steps, retain masked input/output claims, and **accept next-token
logits only if every check accepts**; otherwise discard the execution. Proposed
PLLM ownership must preserve tentative local state until that boundary. An
autoregressive extension must define when verification, state commit and token
feedback occur; it cannot silently treat tentative logits as accepted output.
The companion's state oracle enforces both immutable response barriers and
terminal acceptance/retirement. It includes an auxiliary forgery that passes
the original scalar checks but fails the mandatory secondary check. Python
conformance does not establish native secret erasure or a persistent decoder.

### 3.5 Original numerics, evaluated model, and missing state contract

**M §7.2.1, pp.13–14** specifies model parameters stored in **FP8**, followed by
`Wq = round(2^s * Wreal)`, **s=8**. Signed integers embed into BabyBear. After
each delegated linear operation the client demasks, interprets the result as a
centered integer, divides by **`2^16`**, converts to **BF16**, then runs the next
nonlinear operation. The paper reports no observed modular overflow in its
experiments. It does not establish a universal range bound.

This is **not signed-i8 W8A8 quantization**: an FP8 value scaled by 256 can
require substantially more than eight integer bits. Local `i8` kernels and
per-row scales cannot implement it unchanged. Nor does the sentence “all
arithmetic operations are done in the BabyBear field” on M p.13 justify
inventing a field-only nonlinear graph: p.14 explicitly specifies conversion
to BF16 before the nonlinear operations.

Required unresolved numerical fields:

- exact FP8 format, checkpoint/revision and any FP8 block/tensor scales;
- how the client encodes activations into the field (rescaling by `2^16`
  suggests equal input/weight scales, but the reviewed paragraph explicitly
  defines only the weight quantizer);
- nearest-integer halfway rule, finite/range policy and signed conversion;
- precise BF16 conversion/rounding, nonlinear accumulation types and operation
  order, normalization constants, rotary scheme, attention masking and biases;
- source-compatible embedding and output-head ownership/precision, and all
  other original Qwen3 operations beyond the seven listed projections.

Recommended completion, **pending approval/evidence**: equal scale 256 for
weights/activations, nearest ties-to-even, centered decoding in
`[-(q-1)/2,(q-1)/2]`, followed by specified BF16 rounding. This is an explicit
interpretation of incomplete source detail, not a verified fact about author
code. Require mathematical integer dot-product bounds below `(q-1)/2` before
claiming equivalence to integer arithmetic. Detect out-of-range inputs before
masking; modular verification alone cannot detect numerically wrapped products.

**Original end-to-end evaluation:** Qwen3-4B, **eight input tokens to the next
token's logits** (M §7.2.2, p.14). Table 6 gives these per-block shapes:

| Projection | Input | Output |
| --- | ---: | ---: |
| q | 2560 | 4096 |
| k | 2560 | 1024 |
| v | 2560 | 1024 |
| o | 4096 | 2560 |
| gate / up, each | 2560 | 9728 |
| down | 9728 | 2560 |

Table 7's 8.35 tokens/s is eight **prompt tokens** divided by the reported
0.9583-second execution; it is not 8.35 generated tokens/s. M p.15 likewise
defines steady-state throughput as processed prompt tokens, using microbenchmarks
with simulated latency and negligible server time. M §7.3, pp.15–17 times
online input arrival through accepted next-token logits. These measurements
neither demonstrate persistent autoregressive decoding nor include complete
source-cold publication/preprocessing costs.

The reviewed full text supplies no reproducible KV layout/dtype, append/capacity
policy, sampler, EOS, continuation, cancellation or repeated token-feedback
contract. Local nonlinear attention necessarily requires its working values;
that does not establish a published persistent-KV implementation. The first
source-model slice is the original **eight-token prefill to verified full
logits**, with checkpoint pins and numerical gaps resolved. Full ten-system
serving comparison additionally needs a user-approved autoregressive extension
and a new state/quality contract. Defaulting to existing Qwen2.5-0.5B W8A8
generation would change both the model and the method's numerical experiment.
The [concrete extension proposal](../reproduction-contracts/maverick/README.md#7-exact-residual-decisions-including-persistent-decode)
uses one verified next-logit window before client sampling/feedback, client-owned
tentative causal state, atomic selected-token/sampler commitment, and no KV append
for a selected-but-unexecuted token. Exact KV precision/layout/capacity and
full-prefix-versus-cache semantics remain unresolved numerical/state fields;
this proposal is not an approval or evidence of source decode support.

## 4. Reuse, implementation slices, and qualification

### 4.1 Current PLLM seams and gaps

Local review base: `06b5d6fc0b768be8460e056e2c3c775825e90738`.
Line locators below refer to the hashed working-tree files in §7.

| Current primitive | Reusable mechanism | Required source-faithful work |
| --- | --- | --- |
| `python/pllm/runtime/native.py:72–178` | Immutable matrix ownership; exact modular and wrap32/**wrap64** entrypoints. | Matrix weights are i8. GKM needs int16-by-ring64; Maverick needs arbitrary admitted field-encoded quantized coefficients. `modular` admits only `2 <= q < 2^31`; a larger-field change also needs new arithmetic and admission. |
| `python/pllm/runtime/offset_worker.py:31–37,92–163` | Bound roles, authenticated session framing, stage bindings and one-use bookkeeping. | Existing decoder numerical/body assumptions are not GKM. Add explicit two-worker lookup/body/head ownership and protocol descriptors; limits currently include 32 output tokens and four million tensor elements. |
| `python/pllm/runtime/private_boundary_transport.py:307–383` | Failure retirement, paired exchange ownership, complete head shape checks. | Lookup uses compact DPF over i8 row/scale bytes; head uses W8A8/ring32. Implement a separate dense-XOR/float-row/ring64 GKM boundary. Do not relabel this transport. |
| `crates/pllm-compiler/src/decoder_boundary.rs:160–195`; `python/pllm/runtime/model_binding.py:368–409` | Existing hard composition checks. | Current coupled boundary requires prepared-body roles, CPU W8A8, and no verification; runtime requires a `PreparedRemoteLinear` owner. Add a reviewed method-specific admission path; never bypass guards. |
| `python/pllm/runtime/semantic_executor.py:1107–1110,1131–1188` | Generic token selection/feedback and complete output-head hooks. | Method-specific numeric converters, full model operator/state support and verified acceptance must be introduced explicitly. Current float32 expectations do not establish BF16 Maverick or GKM cast/round ordering. |
| `crates/pllm-core/src/coded_linear.rs:176–354` | Linear-time RAA encode/transpose arithmetic; unbiased bounded random sampling at lines 160–173. | Current public-seed ensemble has no distance certificate. Add explicit source-ensemble code input/generation and actual-field parameter admission; keep tiny Walsh reference separate. |
| `crates/pllm-core/src/coded_delegation_reference.rs:21–69,155–307` | Protocol-3 P/Q algebra, exact-weight secret sampling, one-use non-cloneable query, binding/retirement, checked payload estimate. | Current inputs/weights are i8, t_y=41, in-process only. Need source numerics, t_y profile, authenticated transport, public preprocessing validation, Protocol 2, and decoder lifecycle. No certified privacy/soundness follows from current pass results. |
| `crates/pllm-python/src/coded_delegation.rs:33–65,76–143` | Pre-allocation byte/product checks and bounded operator wrapper. | Wrapper executes native synthetic operator in-process; no persistent distributed handle or model importer. Budget omits Python/OS/allocator peaks. Preserve limits until separately qualified. |
| `python/pllm/protocols/__init__.py:178–218` | Named descriptor entrypoints. | GKM remains pending; Maverick is a bounded operator, not an admitted decoder. Registry labels cannot supply missing guarantees. |

No source-faithful GKM secp256k1 commitment verifier is established by these
reviewed runtime paths. Existing prepared verification/Freivalds facilities or a
row-width certificate do not implement Fig.2/Fig.4. A vetted curve library may
provide group arithmetic; independent PLLM code must implement and bind the
source protocol, not import the author's implementation.

### 4.2 Smallest first code slices, after approvals

**Recommended first action: GKM bounded verified delegation capsule.** It has
fewer unresolved source parameters than Maverick's complete model path.

1. Add a distinct numeric descriptor for nearest/2048/int16/ring64, byte-exact
   float32 embeddings, signed bounds and secp256k1/SHA-256 publication identity.
2. Independently implement a scalar int16-by-u64 wrapping matrix operator and
   dense XOR row operator. Use a tiny synthetic matrix/table, including negative
   weights, non-byte-aligned vocabulary and non-power-of-two dimensions. Avoid
   compiler or checkpoint integration in this first slice.
3. Implement owner publication, reconstructed-MVM verification and immediate
   embedding verification. Exercise two logically separate worker views with
   fresh shares; use a small client-local deterministic operation between two
   delegated products, then final verified output. Include batched accumulators
   and the failed-window retirement path.
4. Pass the independent integer/row/group oracle and adversarial tests below.
   This qualifies only a bounded **protocol capsule**, not a decoder, security
   deployment or throughput result. Then add authenticated two-worker transport
   and source PRG compression with their own transcript/lifecycle conformance.
5. Next integrate independently imported Llama-3.2-1B: private lookup, every
   body stage, full head, prefill plus at least two dependent decode selections,
   state transaction and sampling. Retain original source model and numerics;
   enlarge to the eventual approved workload only through explicit admission.

**Maverick first slice: a source-parameter Protocol-3 capsule, then Protocol 2.**

The independent [scalar companion](../reproduction-contracts/maverick/README.md)
and [machine-readable profile](../reproduction-contracts/maverick/profile.json)
now supply the arithmetic expectations for this native slice. No native/model
execution follows from the scalar checks.

1. Separate field-generic algebra from the current i8 wrapper. Accept explicit
   independent RAA code descriptions and a manifest of q/Nx/Ny/t_x/t_y/ell,
   source matrix and preprocessing validation. Use evaluated BabyBear, rate 1/8
   and t_y=31/ell=2 for the source-evaluation profile; distinguish uncertified
   algebra from any later certified profile.
2. Use a bounded synthetic 64-by-64 field matrix, including coefficients outside
   i8, and exact-weight private noise. Independently verify dense G versus fast
   encode/transpose, P/Q identities, fresh masks, fixed-response checking and
   demasking. Freeze Nx=Ny=512, t_x=165, t_y=31, ell=2. The companion includes
   this source-parameter vector plus tiny literal dense G/P/Q oracle vectors.
   No FP8/model import is needed to establish this algebra slice.
3. Add a **two-claim same-shape Protocol-2** fixture, with explicit secondary
   parameters and separate original-response and auxiliary-response freeze
   barriers. Use the companion's RAA8 secondary dimension 128/N_prime=1024,
   t'=31/ell'=2 capsule instantiation; Q_aux payload is 2,097,152 bytes.
   One thread, one live batch and at most 16 conformance attempts fit only after
   a complete checked allocation receipt under the 16 MiB capsule ceiling.
   Verify corrupted original and auxiliary outputs independently, including
   auxiliary errors orthogonal to original inputs. These are conformance limits,
   not a security lifetime or model geometry approval.
4. After code/security and numeric approvals, implement the original Qwen3-4B
   eight-token-to-logits graph with optimistic batch acceptance and measured
   preprocessing ownership. Only then add the separately approved persistent
   decode/state extension needed by the complete study.

These slices are intentionally small conformance boundaries. Replacing their
eventual model bodies with stubs, clear provider computation on secret inputs,
or an unchanged local head cannot qualify a full independent reimplementation.

### 4.3 Complete conformance oracle

Use **two independent references**: an unprotected execution of the precisely
specified candidate numerical graph, and a separately written arithmetic/semantic
oracle that does not call candidate kernels. Upstream code remains documentary.
Exact protocol correctness is compared to that method's own numerical oracle;
quality relative to the original FP/BF16 checkpoint is a different result.

| Boundary | Required observations and pass rule |
| --- | --- |
| Identity/import | Exact checkpoint/tokenizer/config/artifact hashes; table bytes, quantized matrices, small coefficients, graph, code parameters and publication bindings. Reject dimension-compatible wrong artifacts. GKM ties/rounding and Maverick FP8/BF16 choices must be explicit. |
| GKM arithmetic | Big-integer oracle computes signed `Mq*xq`, share residues modulo `2^64`, signed reconstruction and rescaling. Test half-step positive/negative values, int16 boundaries, values beyond i8, int64 reconstruction endpoints, zero rows and range rejection. Individual shares must exercise wraparound while the true product remains in range. |
| GKM lookup/verification | XOR byte oracle over every tiny-table row, including repeated tokens and final partial selector byte; exact SHA-256 match. Independent group equations and per-call/per-matrix batch accumulators must agree with unreduced integer products lifted into `F_p`. Check bytes before float use. |
| Maverick algebra | Independent arbitrary-integer modular oracle; materialize tiny G matrices only in tests. Compare encode, transpose, P/Q/Q_aux, mask correction and original/auxiliary verification equations. Exercise zero, q-1, negative representatives and coefficients outside i8. Check both valid modular wrap and invalid centered-integer interpretation. |
| Noise/code | Exact support cardinality, no duplicates, nonzero canonical coefficients, uniform-without-replacement algorithm review, separate Gx/Gy and streams; deterministic test vectors remain test-only. Statistical spot checks and freshness tests are diagnostics, not privacy or code-distance proofs. |
| Numeric graph | At every delegated boundary, compare encoded inputs, exact integer/field products, rescaling, biases and client-local operator results. Compare full pre-sampling vocabulary logits. Establish source-compatible reduction order or approve bounded floating discrepancies before quality/cost comparisons. |
| Prefill/decode state | Compare all active K/V, positions and lengths, masks, rotary indices, executed token history, selected pending token, logits, sampler/RNG state and verification-window status. Cover empty/one/multiple-token legal requests, chunked versus unchunked prefill where supported, at least two dependent decode steps, context-capacity boundary and EOS. Maverick repeated decode is an approved extension, not source-evaluated fact. |
| Sampling | Greedy ties and frozen stochastic trajectories (temperature/top-p/seed). Preserve GKM threshold-crossing behavior if using the inspected source sampler. Snapshot logits before in-place sampler mutation. Do not confuse sampling PRNG reproducibility with cryptographic mask entropy. |
| Acceptance/lifecycle | No accepted text/state before all required checks. Compare continuation to uninterrupted clear execution. Cancel at publication, issuance, partial response, demasking, queued accumulation and final verification; confirm one-use retirement, discarded tentative KV/sampler state and no cross-request reuse. |
| Security evidence | Review complete per-worker/per-server transcript and artifact ownership, proof assumptions, code/distance certificates, entropy and retry horizon. Correct outputs plus corruption tests do not prove privacy, negligible soundness error or independent operators. |

For source versus source-code rounding variants, maintain separate expected
vectors and method identifiers; a disagreement is recorded, not averaged away.
For task quality, freeze datasets/splits, prompt formatting, scoring, stochastic
policy and non-inferiority margins **before** runs, consistent with
[COMPARISON-CONTRACT.md](../COMPARISON-CONTRACT.md) §§1.1–1.2. GKM's Wikitext
table is author evidence, not a PLLM acceptance margin. Maverick's reviewed
evaluation provides no held-out quality table establishing equivalence of its
FP8/scale-256/BF16 computation to another checkpoint precision.

### 4.4 Mandatory failure tests

These are specifications for later tests; none was executed for this document.

- **Artifact/role mismatch:** wrong checkpoint, stage, matrix, vocabulary/table,
  code, field, quantizer, commitment set, session, role or response sequence;
  stale publication, valid dimensions with wrong digest, cross-request replay.
- **Malformed input/response:** wrong lengths, duplicate fields/IDs, truncated
  frames, oversized allocation declarations, noncanonical field/group encoding,
  invalid curve point, out-of-range token, invalid selector padding, non-finite
  values and unrepresentable quantized inputs. Check admission before allocation.
- **GKM malicious replies:** flip every output coordinate in bounded fixtures;
  corrupt either/both workers, embedding bits, row hashes and matrix commitments;
  substitute another matrix or omit a batched claim. Test errors that cancel
  under coefficient-one batching and confirm fresh scalar binding catches them.
  Malformed replies must retire the pair even if only one worker replied.
- **Maverick malicious replies:** alter sparse and dense original outputs,
  transpose a code incorrectly, tamper P or Q, return corrupted auxiliary
  projections or Q_aux, omit a claim, change an original response after primary
  challenges or an auxiliary response after secondary challenges. Batch barriers
  must prevent adaptive replacement. Test a deliberately low-distance code to
  ensure it cannot receive certified admission merely by passing happy paths.
- **Randomness/one-use:** injected RNG failure, counter exhaustion, restarted
  session with old seed/material, duplicate mask, privacy/verifier stream
  collision, boost inventory replay and wrong-matrix corrections. Never fall back
  to public/deterministic secret randomness. Discard unconsumed material on abort
  according to the frozen ownership policy.
- **State/abort:** cancellation during each phase, partial prefill, EOS immediately
  after selection, continuation across accepted windows, failed verification
  after tentative KV updates, queue failure and worker timeout. No later successful
  request may observe failed-request state or accept its token/output.
- **Resources/transport:** impossible dimensions, checked-size integer overflow,
  exhausted sessions/calls/claims/queues, disk quota, publication validation peak,
  dropped/duplicated/reordered messages and interrupted cleanup. Failed pressure
  or capacity admission is an unresolved cell with its evidence retained.

Challenge injection and reduced-field exhaustive fixtures must remain isolated
test interfaces. Production secrets must not be logged to make these tests pass.

## 5. Source requirements, evidence gates, and approvals

**No substantial protocol, numeric, state, security or execution change below is
authorized by this document.** The full-study confirmation is not blanket
permission to choose convenient substitutes. The independent
[source-fidelity audit](../reproduction-contracts/delegation.md) distinguishes
mandatory fidelity from actual adaptation decisions. Secure entropy, correct
field arithmetic, and rejection before acceptance are requirements, not optional
changes for the author to approve. Missing source facts must first be researched.

| ID | Concrete decision requiring approval | Recommended default / blocking evidence |
| --- | --- | --- |
| G1 | Source requirement and numeric specification completion; approval only for a changed numerical profile. | Preserve formal nearest/2048/int16/ring64 rather than truncating casts. Specify deterministic ties-to-even, signed bounds, and float32 conversion before division by `2048²`; do not attribute those completions to the evaluated author build. |
| G2 | Dense XOR over float32 bytes versus compact DPF or quantized row/scales; ring64 versus W8A8/ring32. | Preserve Fig.3 dense XOR, float32 rows and original fixed point. DPF/W8A8 would be separately named adaptations with their own controls. |
| G3 | Verified two-server construction versus verification-off restriction or prepared-body/four-role architecture. | Preserve two servers plus honest publisher and complete Fig.2/Fig.4 checks. No inherited four-provider comparator or weakened integrity claim. |
| G4 | Approval required for a changed acceptance window, observable cache-hit omissions, or deployment/trust contract. | Paper-style final acceptance, immediate row checks, no lookup-cache omission; source AES compression with securely initialized, separated streams. Secure entropy, correct challenges and fail-closed acceptance are mandatory. Smaller verified streaming windows require separate approval. |
| G5 | Exact original model/checkpoint/tokenizer and missing model/sampler/state details; change to common Qwen workload. | Llama-3.2-1B source-family anchor first, pinned before import. A common-model port remains explicit and needs quality/state approval. |
| M1 | Source requirement plus evidence gate; approval required for a changed field, code, or ensemble. | Retain BabyBear/full source ensemble. Concrete separate proposal: `q=2^31+11`, independently checked prime/u64 arithmetic; it meets only the quoted field threshold and still needs finite code/security evidence. Preserve historical BabyBear/seeded/i8/41 identity. Tiny exact certificates do not certify model codes. |
| M2 | Source requirement plus missing evaluated batch/security evidence. | Retain evaluated `t_y=31` and two checks. Companion fixes B=2 RAA8 secondary dimension 128/N_prime=1024, t'=31/ell'=2 for bounded conformance; evaluated secondary settings and complete-model batch manifests remain unverified. Charge code-selection events once and every shape-batch attempt across the whole lifetime. |
| M3 | Source-resolution gate; approval required if a complete proposed fallback changes unresolved numerics. | Preserve FP8 weights, scale 256 and post-linear BF16. Recover the exact activation/weight format, scales and operation order. Do not ask for approval of an unspecified FP8/BF16 profile or silently substitute i8 W8A8. |
| M4 | Original eight-token-to-logits workload versus persistent autoregressive generation; embedding/head placement, KV, sampling, EOS and continuation semantics. | Reproduce original Qwen3-4B prefill first. Obtain explicit approval for the required full-study token-loop/state extension; no invented source decode support. |
| M5 | Source requirement; approval only for a changed protocol or acceptance mode. | Standard private/verifiable mode includes Protocol 2 and public-preprocessing validation. Boost is a separately metered source variant; verification-only cannot fill privacy cells. |
| M6 | Resident P/Q/Q_aux versus streaming/offload/recomputation, batching by different shapes, systematic secondary-code storage, new provider roles or changed client ownership. | Admit source ownership only where resources fit; otherwise report blocked and request a priced storage/lifecycle adaptation. Do not move private work to an unreviewed helper. |
| C1 | Matched model/quality contract, request envelope, security class, authenticated confidential channels, CPU/GPU backend and resource/network envelopes for the complete study. | Preserve each original source track, then approve comparison adaptations explicitly. CPU conformance is not reproduction of GKM CUDA performance. No ranking until complete coverage and measured lifecycle costs. |

**Immediate source-correct work:** freeze the explicit GKM numeric/material
manifest and implement bounded arithmetic, dense lookup, and original acceptance
oracles. No substitute-protocol approval is needed to retain G2/G3. Maverick
source-parameter algebra can likewise proceed with its certification status
explicit. Whole-model work waits on the missing numerical and batch facts.
The actual adaptation queue is common-model integration, persistent Maverick
generation, changed GKM verification windows, CPU/GPU execution profiles, and
Maverick storage changes. A code-distance certificate and a complete numeric
manifest are evidence gates; user approval cannot create either.

## 6. Resource admission and later measurement

Admission must use public geometry and checked arithmetic **before** weights,
codes, preprocessing or request buffers are allocated. Freeze request length,
output cap, live queries, batch sizes, verifier queues, context capacity,
publication owner, disk cache, process/thread count and hardware. Preserve failed
admissions and distinguish estimates, payload sizes and observed process peaks.

### 6.1 GKM ownership and lower bounds

- Each server owns int16 linear weights (including the head), float32 embedding
  table, any explicit replicated/tied representations, share/result buffers and
  execution scratch. With `W` linear weight elements and V-by-d float table,
  raw payload is at least `2W + 4Vd` bytes **per server**, before metadata/device
  copies. Deduplicate only allocations actually shared with identical encoding;
  float embedding and int16 head cannot be one unchanged byte representation.
- Client owns tokenizer, public small coefficients, all float32 KV
  (`2*layers*context*kv_width*4` bytes when allocated in that layout), logits,
  nonlinear scratch, secret shares/PRG state, pending output/state and verifier
  buffers. Embedding hashes cost `32V` bytes. With compressed secp256k1 points,
  the point payload estimate is `33*(m_max + sum_stage input_width)` bytes;
  the source does not fix that serialization, so the actual encoded/decoded
  point sizes and scratch must be bound separately.
- Per-matrix batched accumulators need both input/output scalar vectors; count
  32-byte field-scalar payloads plus native representation overhead. Bound
  asynchronous queues of copied `(x,y)` claims, commitment loading/MSM scratch,
  and temporary concurrent verification owners. A streaming accumulator can
  avoid retaining all token claims, but its memory/performance must be measured.
- Uncompressed raw traffic per matrix-vector call is `16n` client-upload bytes
  and `16m` client-download bytes across two servers; source PRG compression
  changes the upload to one explicit share plus seed/setup overhead. Dense PIR
  sends `ceil(V/8)` bytes per worker and receives `4d` bytes per worker per row
  before source seed compression/framing. Account bit padding and batching.
- G p.12 evaluates each server on two A100-80GB GPUs, using only one at a time
  for computation; client is a two-vCPU/4GB VM. The abstract's 179MB storage
  is not complete-client peak RSS, and the VM size is not observed usage.

### 6.2 Maverick ownership and lower bounds

- At rate 1/8 and four-byte field elements, P uses `32mn` bytes and Q uses
  `32mn` bytes: **`64mn` bytes total**, before codes, weights, validation,
  transcript, scratch or auxiliary batching. This applies to the specified
  resident representation, not every possible storage design.
- Protocol 2 additionally needs `4*N*N_prime` bytes for Q_aux, code G_prime,
  B claim inputs/outputs or lossless equivalents, encoded outputs, challenges,
  and auxiliary responses. The server must have the Q_b projections used by
  Protocol 2. Charge publication and distribution as well as storage. Repeated
  matrices/codes can be shared only with explicit batch-layout semantics.
- Public preprocessing can be produced once, but source-cold clients may pay
  full download and validation. Validation may itself require temporarily
  holding the public source matrix. Count its entire lifecycle even if the
  online client later drops weights. Public state is not free because it is public.
- Public matrices do not make query addresses public. Fetching only columns
  indexed by the private noise support from the evaluator reveals that support
  and can expose a linear function of the input. Any proposed paging must keep
  those accesses inside the trusted client boundary, or supply a separately
  approved oblivious-access protocol. Charge complete acquisition, validation,
  local disk reads and resident windows; verification accesses also retain their
  prescribed disclosure timing.
- Private query material includes sparse e_x or retained correction, masked
  input, pending response and transcript. Boost inventory additionally stores
  fresh input/output mask pairs, at least `4*(n+m)` bytes per precomputed pair
  in that representation; all unused and retired inventory is charged.
- Current Rust reference admits dimensions **64..16384**, at most **8 live
  queries**, **65,536 issuance attempts per object**, and a **256MiB payload
  ceiling**. Its SDK wrapper adds copies/scratch and a **2^26 product** bound
  per sample. These are reference limits, not source cryptographic parameters
  or full-process memory guarantees. Large vocabulary heads need separate
  geometry admission; no silent splitting or local fallback.
- The historical [Qwen2.5-shaped storage bound](../../docs/evidence/maverick-coded-storage-lower-bound-2026-09-27.json)
  is **22,900,899,840 bytes** of P+Q across 96 body stages, excluding batch
  auxiliaries. It rejects that resident layout against the recorded 2GiB
  client/6GiB cache budgets; it is not a measured Qwen3-4B source-system failure.
  The [bounded reference card](../../docs/research/maverick-reference.md)
  records admitted persistent payloads of 51.72/66.45MB and two larger-shape
  rejections; these are synthetic operators, not full-model RSS or quality.

### 6.3 Common admission receipt

Before any later run, require a source/method/numeric/security/graph manifest,
all role/artifact owners, pre-allocation lower bounds and conservative peak
estimates, approved storage policy, query/security budget and host-pressure
reserve. Run only the approved bounded slice. The full study additionally needs:

1. Complete-client process-group peak through final verification and cleanup;
   256MiB, 2GiB and 16GiB tiers use exact bytes and the largest accepted observed
   replicate. Include tokenizer, SDK, state audit, helpers and transient owners.
2. Every provider/publisher's phase and lifecycle peak, CPU and device memory;
   synchronized simultaneous host aggregates, not sums of unrelated role peaks.
3. Source-present cold, warm-public/fresh-private and source-cold costs;
   provisioning/publication, preprocessing validation, private preparation,
   prefill, decode, verification, retirement and unused inventory charged.
4. Full directed client and all-link traffic, including extra verification
   exchanges, seeds, public artifacts and framing. Freeze shared 100/40Mbps
   client/provider access constraints, RTT, aggregate provider capacity and
   concurrency under the later comparison manifest.
5. Admission-to-**accepted** first output, accepted full-request latency and
   actual repeated-decode throughput. Report tentative/verified output times
   separately if an approved interface exposes both. Prompt throughput and
   single-next-logit measurements cannot populate decode-throughput cells.

No expensive execution is authorized or needed to accept this documentary slice.
Unknown resource or security cells remain unknown; no original system is declared
ineligible merely because the present PLLM component rejects its geometry.

## 7. Local source identity and documentary checks

The following hashes pin the reviewed implementation seams, independently of any
installed binary. New source changes require rechecking their locators.

| Local path | SHA-256 |
| --- | --- |
| `crates/pllm-core/src/coded_delegation_reference.rs` | `1537445a3ba0a8c19842852f91d5b97bd6ae08ddf187eefab77138fcad41d9cd` |
| `crates/pllm-core/src/coded_linear.rs` | `9a774a772029e84a55a89f56bdf2d92b02cdef71da16d9f3b0b985d752142df9` |
| `crates/pllm-python/src/coded_delegation.rs` | `134ab5b9f61d4c2bbd5572ad6016d9d4f0ac18422e05bfa44b5190bcf7344424` |
| `python/pllm/runtime/native.py` | `4429341014da3db57badb879091afc0281961afa28379f6e09094eabd43cabf6` |
| `python/pllm/runtime/offset_worker.py` | `49bad6b0f651f6815604517303588b2c3ce9f89ec702e8cc69fd3b0dce071dbd` |
| `python/pllm/runtime/private_boundary_transport.py` | `d66afdad0794325b76b7e8ec57a2463a0281e5047c80b8f44adad1601dffe4e0` |
| `python/pllm/runtime/semantic_executor.py` | `0b9b4223e5c193530f40760fdcf930d4040f5fce5eb67f3ec03b27c56fa42852` |
| `python/pllm/runtime/model_binding.py` | `cfc078ba592764212c2dcbab9b28fd5f41cee01abe761587eef8a2a7932d1e7a` |
| `crates/pllm-compiler/src/decoder_boundary.rs` | `555a80f191039413f86ced4298a5b4f3b5b31c2471f93a114f0cc30b61ca7a91` |
| `python/pllm/protocols/__init__.py` | `4be9bbbe50883696ebb8a58652356dbffc5ee1eb294a73ee44910552ed2267ff` |

Read-only GKM archive check performed:

```text
python3 -B paper/artifacts/source-review/2026-10-16-gkm-qualification-v1/check.py --pdf-text
status: passed
JSON documents: 6; selected upstream files: 12; local files: 15
code citations: 41; replayed cached PDF page extracts: 8
frozen primary records unchanged: true
fresh full upstream bytes retrieved: false; upstream code executed: false
```

Both full primary PDFs were separately read, including Maverick's numerical
evaluation and security appendices; not merely their abstracts or cached notes.
The t_y conditional-error calculations in §3.3 are small arithmetic checks of
the printed lemma, not protocol execution, concrete code certification or a new
security theorem. Subsequent work must resolve the approval ledger and evidence
gates before any complete-system qualification.
