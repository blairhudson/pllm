---
title: "PLLM: One-Time Preprocessing and Local Verification for Private LLM Inference"
subtitle: "A runtime design and prototype study"
author:
  - "Blair Hudson — deployscience labs"
date: "13 September 2026"
lang: en-AU
bibliography: references.bib
link-citations: true
colorlinks: true
fontsize: 10pt
geometry:
  - margin=24mm
papersize: a4
header-includes:
  - \usepackage{microtype}
  - \usepackage{fvextra}
  - \DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,commandchars=\\\{\}}
  - \setlength{\emergencystretch}{3em}
  - \widowpenalty=10000
  - \clubpenalty=10000
---

## Abstract {.unnumbered}

PLLM delegates a language model's weight–activation matrix multiplications while keeping tokenization, attention, nonlinear operations, model state, and output generation at the client. A non-colluding preprocessing service prepares one-time masked corrections before online execution. This paper specifies the arithmetic and lifecycle requirements of that architecture and proposes a locally verified extension. The extension retains compact power-of-two rings for masked computation, but checks reconstructed integer results in a larger prime field using private, model-specific verification vectors. Its lifecycle assigns each preprocessing record to at most one distinct activation at the client, allowing identical-message retries without treating a server acknowledgement as evidence of freshness. We give conditional correctness, privacy, and verification arguments, and executable checks of the arithmetic and an abstract failure-state model. Separately, we retain the reported evaluation of the earlier, unverified prototype: nine warm Qwen2.5-0.5B-Instruct runs on one loopback host, with median online completion times of 2.618–7.442 seconds and reported client traffic of 63.33–432.71 MB. Those measurements do not evaluate the proposed verifier or establish wide-area performance, model-quality preservation, or a speed advantage over local inference. The resulting design makes explicit which costs are reusable, which preprocessing resources are expendable, and which deployment claims still require measurement.

# Introduction

Private inference should make the trust required to obtain computation explicit. In the setting considered here, the model weights are public, but the user's inputs, intermediate activations, and generated outputs are private. The client can execute the model's control flow and stateful operations, yet seeks to delegate its dense weight–activation matrix multiplications. The objective is not to hide the model or to eliminate all client computation.

PLLM divides this work among three roles. The **client** owns plaintext and the local model state. **Preparation**, a separate preprocessing service, generates input-independent correction tensors. **Inference**, the online service, uses those corrections to evaluate masked matrix products. Preparation and Inference must not combine their protocol views. This assumption concerns the parties that can access the data, not merely the number of processes or credentials.

The underlying masking identity and local verification pattern have substantial precedent. Slalom delegates masked linear computation and uses preprocessed verification; Carnival studies outsourcing its preprocessing [@slalom; @carnival]. Recent work also addresses private, verifiable open-weight language-model inference [@gkm; @maverick]. PLLM therefore does not claim a new additive masking primitive or priority for private linear-layer outsourcing.

The research question is narrower: **how should an autoregressive inference runtime combine compact arithmetic, locally checked results, and failure-safe consumption of one-time preprocessing material, and under what measured conditions does that combination make outsourcing worthwhile?**

The design separates three resources with different lifetimes. Model-specific verification data can be reused within a bounded verification epoch. Privacy masks and their correction tensors are consumed by distinct activations. Online execution consumes those resources while advancing the client's model state. Conflating these lifetimes can turn a retry or state rollback into mask reuse, or present an initially full preprocessing pool as sustainable throughput.

This revision contributes a specification of those boundaries, a proposed mixed-arithmetic verifier, and a client-enforced freshness invariant with explicit recovery assumptions. It also preserves the earlier prototype's measurements without attributing the proposed changes to that implementation [@hudson2026draft]. The accompanying reference program checks mathematical examples and a small abstract state machine; it is not the production runtime. No performance result in this paper measures the proposed verified extension.

# Relation to prior work

**Preprocessed linear outsourcing.** Slalom already combines local nonlinear operations, one-time additive blinding, and private preprocessed matrix-product verification. Carnival extends this line by outsourcing preprocessing using subset-sum-based masks [@slalom; @carnival]. PLLM inherits this architectural family rather than displacing it. A separate preprocessing role and a ticket identifier are not, by themselves, cryptographic novelty.

**Secure transformer computation.** Sigma uses trusted-dealer preprocessing and function secret sharing for secure GPT inference, including operations that PLLM retains at the client [@sigma]. Its computational partition and model-privacy requirements differ. A comparison must identify the trusted parties, protected values, client work, and offline costs before comparing latency.

**Contemporary open-weight delegation.** Gupta, Katz, and Miers describe outsourcing to two malicious, non-colluding servers, with client verification using information posted by the model owner [@gkm]. This comparison is limited to their official abstract; no claim about their detailed preprocessing requirements follows from it. Maverick provides private and verifiable matrix–vector delegation with transparent preprocessing and a precomputed-mask mode [@maverick]. Together, these works rule out a broad claim that combining non-collusion, public weights, or verification establishes PLLM's novelty.

**A different privacy boundary.** MOSAIC also protects weights and permits approximation error in masked outsourced computation [@mosaic]. PLLM instead targets exact reconstruction of a specified quantized integer product with public weights. Exactness relative to that product is not fidelity to the original floating-point model.

**Speculative execution.** Speculative decoding is an established way to verify several locally drafted tokens in a target-model batch [@speculative]. POST applies client-side drafting and batched verification to private inference [@post]. Any extension of PLLM in this direction must identify an additional contribution, such as the accounting and scheduling of expendable preprocessing material; private speculation itself is not new.

The mixed-arithmetic check below is an instantiation of established private matrix verification, not a new general verification primitive. Its purpose is to preserve the prototype's compact wire representation while obtaining an explicit field-based error bound. Whether this choice and the lifecycle design constitute a useful systems contribution depends on implementation and comparative measurement.

# Execution model and baseline protocol

## Roles, trust, and leakage

The client is trusted with plaintext, quantization scales, sampling state, verification secrets, and its allocation ledger. It executes token lookup, attention, normalization, nonlinearities, residual operations, the output projection, and decoding. Preparation and Inference hold the public matrices for delegated operations. Here, *public* means available to the protocol participants; it is not a licensing claim.

For the baseline privacy argument, Preparation and Inference follow the specified protocol and do not collude. Each may retain its own complete view. Inference must not obtain the masks or their generating seeds; Preparation must not obtain the corresponding online masked activations. Authenticated confidential channels protect these separations against outsiders but do not establish administrative independence.

The permitted observations include the public model and stage configuration, record allocation and delivery events, tensor dimensions, traffic volume, and execution timing. Unpadded row counts can disclose exact processed lengths, not just approximate lengths. This paper does not prove protection against implementation timing side channels or inference from the permitted metadata. Neither anonymity nor model confidentiality is a goal.

Retaining masks alone does not reconstruct inputs. Combining an input mask with its matching online masked activation does. Erasure can reduce exposure to a later compromise, but is not required for the stated static, non-colluding privacy argument and is not established by releasing Python objects. Retained root seeds, backups, or snapshots may regenerate supposedly erased masks.

The proposed verifier separately checks arithmetic integrity against arbitrary returned values. Its argument does not, by itself, establish a complete actively secure protocol against all malicious participants or remove the non-collusion assumption for privacy.

## Integer representation

For one delegated stage, let

$$
W\in\mathbb Z^{m\times n},\qquad x\in\mathbb Z^n,
\qquad q=2^k.
$$

The matrix is the authenticated, signed integer representation of a specified quantized operator. All masked computation occurs in the ring $R_q=\mathbb Z/q\mathbb Z$. The prototype supports $k\in\{16,24,32\}$. Signed integers are mapped to residues modulo $q$; returned residues are mapped back to the canonical interval $[-q/2,q/2)$.

To recover the intended integer product, the compiler must establish a public bound

$$
\|Wx\|_\infty\le B<q/2
$$

for every activation permitted by the stage's quantization policy. For example, if $\|x\|_\infty\le A$, a sufficient bound is

$$
B=A\max_i\sum_{j=1}^{n}|W_{ij}|.
$$

The bound is computed from public weights and the declared activation range, not the observed private activation. Otherwise ring selection itself becomes additional input-dependent leakage. An unsupported bound must cause compilation to fail or select an explicitly supported wider representation; silent saturation is not an alternative.

The bound concerns the **unmasked result**, not intermediate products involving random masks. Masked operations must implement modular arithmetic correctly even when their ordinary integer intermediates exceed $B$. Wider accumulators with periodic reduction or correctly defined wrapping arithmetic are suitable; accidental overflow, saturation, and inexact floating-point accumulation are not interchangeable with reduction modulo $q$.

Biases, private scales, and dequantization are applied locally after the integer product is recovered and, in the extension, verified. A quantization scheme with multiple scaled partial products must specify each integer relation being checked. It cannot be treated as one unscaled matrix product without justification.

## One-time preprocessing and online evaluation

For each stage-row identifier, the client and Preparation derive independent masks

$$
r\in R_q^n,\qquad s\in R_q^m.
$$

Preparation computes and sends Inference the corresponding correction

$$
c=Wr-s\pmod q.
$$

It does so before that record becomes available for online use. The client sends the record identifier and

$$
u=x-r\pmod q.
$$

Inference returns

$$
v=Wu+c\pmod q.
$$

The client reconstructs $z=v+s\pmod q$ and converts $z$ to its canonical signed representation $y$.

**Correctness.** Ring arithmetic gives

$$
z=W(x-r)+(Wr-s)+s=Wx\pmod q.
$$

The public no-wrap bound makes the canonical representative equal to the intended integer product: $y=Wx$ over $\mathbb Z$.

**Privacy for one operation.** With truly independent uniform masks, Inference's pair $(u,c)$ is uniform independently of $x$. For any fixed observed pair, exactly one mask pair produces it:

$$
r=x-u,\qquad s=W(x-u)-c\pmod q.
$$

This argument does not require $W$ to be invertible. The response is a deterministic function of the observed pair and public matrix. Preparation's masks and correction are independent of the activation. Fresh independent mask pairs extend this argument to a fixed sequence of stages, with the declared execution metadata treated separately.

The implementation replaces independent randomness with pseudorandom expansion from secret roots, making confidentiality computational rather than information-theoretic. A complete implementation argument must cover the joint outputs for input masks, output masks, and identifiers across all records, including collisions and domain separation.

## Derivation and configuration binding

A session begins with fresh client-generated entropy. A canonical public context binds the protocol version, session epoch, model digest, signed matrix representation, quantization policy and public bounds, stage dimensions, ring, wire encoding, and allocation budget. Each stage root is scoped to that session and stage. The stage-row index and distinct input-mask, output-mask, and identifier labels are included in expansion.

The baseline reports SHAKE-256 expansion. The revised specification requires unambiguous length-encoded inputs and disjoint derivation domains. KMACXOF256 is one standard keyed construction suitable for such an implementation; its adoption would require explicit parameters and test vectors, not a new security claim [@nist185]. Verification seeds are independent and are never supplied to Preparation.

The configuration digest binds a record to an interpretation. It is not proof that a remote service used those weights or performed any computation. The client must authenticate the model and the local execution graph through its own trusted configuration.

# Proposed extension: local verification across arithmetic domains

## Keep the compact ring; verify the reconstructed integer

A direct transplantation of a field-based verification bound into $\mathbb Z_{2^k}$ is unsound. For example, a nonzero error $2^{k-1}$ multiplied by a uniform ring element vanishes whenever that element is even, giving acceptance probability $1/2$ for that check.

Instead, retain $R_q$ for masking, transport, and remote computation, and use a larger prime field solely at the client. Choose

$$
p=2^{61}-1>q.
$$

For each immutable stage and verification epoch, the client privately samples $h$ independent uniform vectors $a_j\in\mathbb F_p^m$ and computes

$$
b_j=W^\top a_j\pmod p,\qquad j=1,\ldots,h.
$$

The client computes these vectors from the authenticated **signed** matrix. Setup processes the full matrix with $h$ field multiply–accumulates per weight; all vectors can be formed in one streaming pass. Giving the verification secrets to Preparation would invalidate the stated trust separation for verification.

After unmasking and canonical signed decoding, the client accepts $y$ only when

$$
a_j^\top y=b_j^\top x\pmod p
\quad\text{for every }j.
$$

The check uses the original signed activation and reconstructed signed result. It does **not** reinterpret the masked ring equation in $\mathbb F_p$: carries make that a different equation. Biases, dequantization, local state updates, and output release occur only after acceptance.

Precomputing a secret vector's product with a fixed matrix follows Slalom's verification pattern [@slalom]. The design choice here is to separate the efficient masking ring from the verification field through a bounded signed-integer interpretation.

## Conditional soundness

Assume the public no-wrap bound is correct, the matrix and verification state are authentic, and a returned result is fixed independently of the client's secret check vectors. If a canonical result $y$ is incorrect, let $e=y-Wx$ over the integers. Both $y$ and $Wx$ lie in $[-q/2,q/2)$, so each coordinate satisfies $|e_i|<q<p$. At least one coordinate is nonzero in $\mathbb F_p$.

A uniform $a_j$ therefore satisfies $a_j^\top e=0$ with probability exactly $1/p$. Independent vectors give false acceptance probability $p^{-h}$. For packed input columns $X$ and returned columns $Y$, check $a_j^\top Y=b_j^\top X$ for every column. A wrong batch contains a nonzero error column, so the same $p^{-h}$ upper bound applies to accepting that batch.

Verification state may be reused only under a bounded-lifetime argument. Until the first false acceptance, correct checks reveal no algebraic constraint on the secret vectors. The service must receive no residuals, check values, or differentiated failure diagnostics. It must fix each returned value before feedback. The client fails closed and retires the affected verification epoch on rejection. Under these conditions, a first-error argument bounds the probability of any false acceptance over at most $N$ checked batches by

$$
\Pr[\text{any false acceptance}]\le Np^{-h}.
$$

With $h=3$ and $N\le2^{32}$, this ideal bound is less than $2^{-150}$. Pseudorandomly generated vectors add the corresponding computational distinguishing term; that number is not a claim about the security of the complete implementation. Timing leakage from check processing, compromised verification state, model substitution at setup, and state rollback lie outside this calculation.

Checking each delegated result before it influences subsequent computation supports an inductive integrity argument for the trusted local graph. It proves agreement with that graph's quantized operators, not agreement with a different floating-point model, a public proof of correct execution, or availability against a malicious service.

## Costs and verification state

The extension adds no protocol messages or remote matrix multiplications. It adds $O(hmn)$ client setup work per stage and verification epoch, $O(h(m+n))$ client work per activation, and $O(h(m+n))$ stored field elements if both vectors are retained. Keeping seeds instead of expanded $a_j$ trades memory for regeneration work. These costs must be measured rather than described as negligible.

For many activations, verifier setup is reusable. The privacy corrections are not. This distinction is essential: reusing $b_j$ under the stated verification conditions does not authorize reusing $r$, $s$, or a correction record for a different activation.

# Proposed lifecycle: client-enforced assignment before transmission

## The security invariant

The required invariant is:

> A preprocessing record is assigned to at most one distinct activation, across all transmissions, retries, concurrent requests, cancellations, and supported recovery paths.

It is not “exactly one network transmission.” Repeating the identical masked request reveals no second independent masked activation. Conversely, a server saying that it has not consumed a record cannot make that record safe for a different input: the first message may already have been observed.

For instance, the unsafe sequence

```
assign record to x0 -> send -> timeout -> recycle -> assign to x1 -> send
```

reveals $u_0-u_1=x_0-x_1\pmod q$. An honest server's atomic consumption operation does not protect a client that can be induced to make this second assignment.

## State transitions

The client is the authoritative allocator. Before any masked request bytes can leave the client, it binds the record identifier to one activation and freezes the corresponding request bytes. The allocation state must be committed atomically with respect to every client writer that can use the same inventory.

| State | Meaning | Permitted next action |
|:--|:--|:--|
| Ready | Prepared and not assigned | Bind to one activation |
| Bound | Assignment and frozen request recorded before transmission | Send; or retire without sending |
| Exposed | The request may have left the client | Retry identical bytes; verify a response; or retire |
| Verified | A response passed all required checks | Advance local model state once; retire record |
| Retired | No further assignment is permitted | No transition back to Ready |

Retransmission is permitted only for the identical frozen application payload and execution identity; transport encryption still uses its normal fresh records. A runtime may instead choose the simpler conservative policy of retiring the record on every ambiguous timeout. It must not recompute a changed activation under the old record identifier. Duplicate responses cannot advance the local graph or emit a token twice.

Preparation delivery, server execution, and client acceptance are distinct events. A READY acknowledgement is a claim that a service has loaded the required records, not a proof of remote state. Server-side deduplication and atomic consumption remain useful operational safeguards, but the client's privacy invariant does not depend on a malicious service enforcing them.

## Failures, concurrency, and rollback

Cancellation, partial batch failure, and uncertain transmission never return a bound or exposed record to the free pool. A delayed correction delivery or acknowledgement must not recreate an invalidated allocation. Packed prefill binds every constituent row; accepting a batch also requires the expected ordered row identifiers and dimensions.

For ordinary process recovery, an implementation can use a crash-consistent client ledger, or invalidate the entire old session and obtain new entropy and preprocessing. All processes using a session must share one allocator. Two clones independently continuing the same saved session violate that requirement.

A journal alone does not protect against restoring the journal and seeds together from an old snapshot. Arbitrary rollback resistance requires a trusted monotonic state source, an appropriate external coordinator, or an explicit exclusion from the threat model. Neither TLS nor a server-issued ticket solves client rollback. The abstract checks in this paper exclude cloning and rollback rather than claiming to solve them.

Verification state has a separate epoch and check budget. A model-state rollback cannot roll back privacy allocation or reinstate a retired verifier epoch. This separation also matters for future speculative execution: discarded candidate tokens may roll back the KV cache, but their transmitted masked activations have still consumed preprocessing material.

# Prototype and specification status

The original prototype uses a Python model graph and a Rust extension through PyO3. Rust implements integer matrices, arithmetic, codecs, and parallel execution. Python implements model import, quantization metadata, scheduling, preparation inventory, transport, and the response interface. Matrices are quantized and imported once. The reported implementation includes packed prefill, a persistent decode WebSocket, local embeddings and output-head evaluation, and three separately authenticated role-to-role channels [@hudson2026draft].

The draft also reports stage-bound configuration digests, acknowledged preparation readiness, server-side atomic reservations, and retirement of unused reservations. It does not provide a production demonstration of the client allocation protocol or verifier specified above. Configuration binding should replace the draft's broader “commitment” terminology unless an actual commitment construction is specified.

| Component | Evidence available in this revision |
|:--|:--|
| Masked ring evaluation, preparation service, packed transport | Described and benchmarked in the supplied prototype report |
| Field checks of bounded signed products | Specified here; independent arithmetic checks supplied |
| Client assignment before send and identical-message retries | Specified here; a small abstract state model checked |
| Full verified LLM execution and fault recovery | Not benchmarked or demonstrated in the supplied production implementation |
| WAN performance, sustained replenishment, model quality | Not established by the reported experiment |

The source report is the basis for implementation statements. This revision has not inspected or reproduced production revision `277d19f`; the separate reference program must not be mistaken for that code.

# Evaluation

## Reported baseline experiment

The earlier study reports Qwen2.5-0.5B-Instruct model revision `7ae557604adf67be50417f59c2c2f167def9a775`, PLLM revision `277d19f`, Python 3.13.15, macOS 26.5.2, and an Apple M5 with 32 GiB memory. Client, Preparation, and Inference were separate processes on one loopback host. One excluded warmup preceded three repetitions for each of three input lengths. Generation was capped at 16 tokens [@hudson2026draft]. These are author-reported conditions, not independently verified environment observations.

**Table 1. Reported baseline latency in seconds.** Values are medians; the TTFT range is the observed minimum–maximum across three runs.

| Input tokens | Output tokens | TTFT | TTFT range | Online completion | Full completion |
|--:|--:|--:|:--|--:|--:|
| 30 | 9 | 0.978 | 0.935–0.982 | 2.618 | 4.674 |
| 63 | 16 | 1.374 | 1.353–1.379 | 3.194 | 5.347 |
| 255 | 16 | 5.158 | 4.961–5.289 | 7.442 | 13.059 |

The source defines online completion from inventory readiness and full completion as including preparation and transition overhead. It does not unambiguously specify the TTFT clock origin; the values retain their original labels rather than silently assigning a new origin. Different output counts preclude a direct generation-throughput comparison between rows. Three repetitions do not establish tail latency.

**Table 2. Reported baseline traffic and preparation allocation.** MB and the traffic categories retain the source's units and accounting labels.

| Input tokens | Client I/O, MB | Correction push, MB | Stage rows |
|--:|--:|--:|--:|
| 30 | 63.33 | 59.80 | 6,144 |
| 63 | 126.29 | 72.88 | 7,488 |
| 255 | 432.71 | 252.18 | 25,920 |

The source identifies 96 delegated stages and counts stage rows across all of them. It does not supply per-direction, per-phase client traffic or a per-run retirement count. Those quantities cannot be reconstructed uniquely from the aggregates. Nor does 96 stages establish 96 sequential network round trips: the scheduled dependency graph must be measured.

All nine records reportedly contained zero online Preparation operations and zero plaintext prompt or token bytes in the instrumented audit counters. The latter is evidence about those instrumented paths, not a confidentiality proof. The source reports 400 Python tests under native and scalar kernel configurations and 11 Rust tests; these are separate from the reference checks below [@hudson2026draft].

## New reference checks, separate from the prototype

The accompanying `protocol_checks.py` uses deterministic test data and arbitrary-precision Python arithmetic. It implements neither an LLM nor production randomness, networking, key management, or durable storage. Its purpose is to expose arithmetic mistakes and illustrate the lifecycle invariant with reproducible counterexamples.

For the one-dimensional ring $\mathbb Z_8$, it enumerates 4,096 mask assignments across 64 weight/input cases. The joint distribution of the masked input and correction is uniform for every case, including non-invertible weights.

For a verification counterexample in $\mathbb Z_{16}$, a nonzero error vector $(8,0)$ passes 128 of 256 single-vector ring checks. Lifting to $\mathbb F_{17}$ yields 17 accepting checks out of 289; two independent checks yield 289 out of 83,521. A deliberately undersized verification field also demonstrates aliasing. These exhaustive examples illustrate the need for the field and range conditions; they do not estimate the cryptographic false-acceptance probability of the proposed production parameters.

Across 1,500 generated integer products, with 500 cases at each supported wire width, masked reconstruction agreed with the integer reference. The program rejected 4,500 deliberately corrupted results and 1,500 noncanonical verifier inputs. Negative controls exercise wraparound and mask reuse. These tests cover vectors, not a production packed-prefill path.

Finally, a finite one-record, two-input state exploration examines 12 reachable states and 20 transitions for binding before transmission, with no violation of its freshness invariant. A deliberately unsafe timeout-recycling variant produces the counterexample `bind(0), send, timeout, bind(1), send`. This is bounded exploration of an abstract model, not formal verification of PLLM's implementation.

## What these results establish

The supplied prototype report establishes that its described partition executed the small evaluated model on loopback. The separate checks support the stated arithmetic identities and find the intended negative controls. Neither result establishes numerical fidelity to the floating-point model, sustained decode speed, verified-inference overhead, client memory savings, wide-area usefulness, or production security.

# Deployment costs and required comparisons

## Preprocessing is expendable work

Each consumed record in the baseline construction requires a preprocessing multiplication $Wr$ and an online multiplication $W(x-r)$. Unused prepared rows also incur their preparation cost. Batching may make preparation cheaper per row, and scheduling it early may reduce request latency, but neither changes its one-time nature.

For stage $j$, let $C_j$ be the number of distinct activation rows actually sent, and $P_j$ the number prepared. Without replenishment, $P_j\ge C_j$. Useful preprocessing utilisation is measured from consumed and prepared records, with separate counts for reserved, invalidated, expired, and still-ready records. A benchmark must not call every unconsumed row “waste” while the inventory remains usable.

A viable service also needs sustained preparation throughput at least as large as its long-run record demand, at every required stage. An initially full pool can conceal a replenishment bottleneck. Report the time and cost to build the pool, its capacity, and the latency effects of depletion, in addition to online timings.

The deployment justification must answer why the trusted preprocessing capacity is preferable to trusted online inference capacity. Candidate advantages are predictable batches, flexible scheduling, and keeping that service off the interactive path. Those are hypotheses for measurement, not established economic savings.

## Communication and local work

For a batch of $b$ activation columns at a stage with dimensions $m\times n$ and $k$-bit wire coefficients, the ideal coefficient payload is

$$
\text{upload}=bnk/8,\qquad
\text{download}=bmk/8,\qquad
\text{correction delivery}=bmk/8
$$

bytes, before identifiers, framing, authentication, and padding. The correction delivery is a separate Preparation-to-Inference cost. These formulas should be compared against instrumented bytes, not used to silently reinterpret the existing aggregate traffic counters.

Latency depends on the critical path through delegated operations and local computation. A persistent connection avoids repeated connection setup but does not remove dependencies. Fusing independent matrix products can reduce exchanges, but the original 96-stage manifest is not provided, so this revision does not assume those fusions are missing. A measured event trace must determine the sequential exchange count.

Client work is also material. The public configuration of the evaluated Qwen model specifies a vocabulary of 151,936, hidden size 896, and tied embeddings [@qwenconfig]. The shared embedding/output matrix therefore contains 136,134,656 parameters. Its actual representation and memory cost in PLLM must be measured. The client additionally retains the KV cache, nonlinear computation, masks, transport buffers, and—in the extension—verification data. The design does not establish a smartphone-sized client merely by outsourcing transformer matrices.

## Comparative evaluation plan

The next evaluation should compare the identical integer graph locally, the identical remote partition without masking, the masked baseline, and the locally verified extension. An optimised local inference engine is a separate practical baseline. These distinguish graph cost, networking, privacy overhead, verification overhead, and the user's alternative to outsourcing.

Report exact per-stage agreement against an integer reference, followed by logits, perplexity, and task quality against the original model and the selected quantized model. Include a materially larger public model, longer contexts, and sustained generation. Compare equal workloads and output budgets; do not divide prefill tokens by a next-token-logit latency and label it sustained decode throughput.

Measure cold and ready-inventory requests separately, with explicit clocks from request arrival and from readiness. Separate uploads, downloads, correction pushes, setup, replenishment, and unused allocation. Exercise controlled RTT and bandwidth, a physically separated deployment, concurrent requests, and depletion. The appropriate result is a crossover region in client capability, link quality, model size, and demand—not a single universal acceleration claim.

Cross-paper performance comparisons require the same care. Maverick's evaluated prompt-to-next-logits workload, aggregate prompt-token metrics, and separately simulated network effects are not interchangeable with sustained autoregressive decoding [@maverick]. Security and client-work boundaries must also match before headline speedups are comparable.

# Extension to speculative execution

A local drafter could propose a block of tokens that the target graph verifies with packed evaluation. Correct rejection sampling can preserve the target distribution [@speculative], and POST already studies private inference with client-side drafting [@post]. This is a plausible performance extension, not the novelty claim of the present design.

The additional PLLM-specific accounting problem is that rejected drafts still spend every preprocessing record used in their transmitted target evaluation. Restoring a KV-cache checkpoint cannot restore those records. A scheduling policy should therefore measure time, bytes, and prepared rows per accepted token, alongside replenishment pressure, rather than optimise acceptance rate alone.

Adaptive block sizes and traffic patterns can reveal information about acceptance and execution. An oblivious policy would require a public schedule and appropriate padding; otherwise those observations belong in the leakage description. The reference implementation and baseline measurements do not evaluate this extension.

# Conclusion

PLLM's strongest supported direction is a runtime for private linear outsourcing with explicit arithmetic, verification, and resource-lifecycle boundaries. The baseline masking construction follows established preprocessing-assisted delegation. The proposed extension retains compact ring computation while checking bounded signed results in a larger prime field, and makes the client responsible for assigning each privacy record before transmission.

The existing loopback measurements remain evidence about the earlier unverified prototype. The new executable checks validate examples of the arithmetic and abstract lifecycle model, not production integration or performance. The next decisive result is a verified implementation that demonstrates exactness, failure safety, model fidelity, and a useful deployment crossover after accounting for client work, communication, and preprocessing replenishment.

# References {.unnumbered}
