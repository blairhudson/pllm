# Maverick: executable source-derived algebra and admission profile

17 October 2026. Companion to [canonical Maverick contract](../../methods/delegation-contracts.md#3-maverick-contract).
This directory supplies an implementable **Protocol 3 / Protocol 2 arithmetic
profile**, independent bounded vectors, exact sampling checks, and a narrow
decision record. It does not qualify a complete model or cryptographic deployment.
No production file, historical BabyBear artifact, model, or native build is changed.

## 1. What is resolved

| Question | Concrete result |
| --- | --- |
| Which field matches the paper's printed `q>2^31` hypothesis? | `q=2,147,483,659=2^31+11`, the smallest such prime. Exact primality and machine-width checks run below. It is an **unapproved, separately named proposal**, not the evaluation field. BabyBear remains the source default. |
| Does this certify the code? | No. The printed asymptotic bound lacks finite constants/dimension thresholds; the proposed field only closes that one hypothesis. The current broader primary theorem could instead preserve BabyBear, but its full text remains unverified. |
| Exact RAA implementation? | Explicit zero-based matrix conventions, full permutation/nonzero-scale ensemble, both linear maps, serialized descriptors, and independent literal dense-matrix oracle below. |
| 31 versus 41? | Source `31×2` already meets conditional `2^-40` by Lemma 5.1. Existing PLLM `41×2` keeps its own historical identity. |
| Sparse noise? | Exact-weight uniform support without replacement, independent uniform nonzero coefficients; exhaustive draw-path tests establish the sampler's finite distribution. Eq.(1) is a heuristic, not a concrete 128-bit proof. |
| Batch parameters? | Freeze a concrete two-claim RAA8/31×2 secondary instantiation for the bounded capsule. It implements Protocol 2; the paper does not establish that these were its evaluated secondary parameters. |
| Small distance evidence? | Two explicit `[16,2]` fixtures have **exact minimum distance 15**, over BabyBear and the proposed field, using all-message projective-direction counting. This does not certify any 64-dimensional or model code or dual-LPN security. |
| Decode ownership? | Source window ends at verified next-token logits. Persistent decode is an explicit unresolved extension, with minimal proposed ownership semantics in §7. |

### Files and replay

- [`source-lock.json`](source-lock.json): pinned PDF/Rust bytes and equation locators;
  new primary-record retrieval outcomes, including a misleading older author mirror.
- [`profile.json`](profile.json): exact source-derived capsule parameters, field
  proposal, representation, role/state and admission boundaries.
- [`oracle.py`](oracle.py): independent integer arithmetic, literal RAA matrices,
  Protocols 2/3, full P/Q validation, exact sparse-error probability, small exact
  code-distance certificate, and two-barrier state oracle.
- [`fixtures.py`](fixtures.py): deterministic **test-only** fixture descriptions;
  `--emit` emits expanded small inputs; `--emit64` emits both full capsule inputs
  and all three verification/privacy code roles for another implementation.
- [`vectors.json`](vectors.json): frozen field products, correction/verification
  values and canonical-JSON digests, including `64×64` Protocol-3 and full B=2
  `512×1024` Qaux source-derived Protocol-2 cases.
- [`test_oracle.py`](test_oracle.py): scalar conformance and adversarial checks.

From repository root:

```sh
python3 -B -m unittest discover -s paper/reproduction-contracts/maverick -v
python3 -B paper/reproduction-contracts/maverick/fixtures.py --emit
python3 -B paper/reproduction-contracts/maverick/fixtures.py --emit64
```

Only Python standard library is required. Tests import neither PLLM nor upstream
code and perform no network request. Pinned local specification files are hashed,
not executed. The golden-vector replay compares every recorded integer and digest.
The floating `minus_log2_diagnostic` is descriptive; exact fractions govern bounds.
Fixture digests use ASCII JSON with sorted keys and separators `(',', ':')`.
Integers in JSON may exceed 53 bits; native consumers must not parse them via f64.

## 2. Source identity and field decision

**M** is cached `papers/r24-maverick.pdf`, arXiv **2609.10264v1**, SHA-256
`c9570fdeb51a26f9d707bcc7bf485f2001e7d3e15ebdf34815123935a1e8e2a0`.
Locators are physical one-based PDF pages. M §7.1 p.11 selects
`q=15*2^27+1=2,013,265,921`, RAA repetition 8, target distance 1/2,
`lambda=128`, `kappa=40`, `t_y=31`, `ell=2`. M Appendix B p.24 quotes
`q>2^31`, `delta=0.5`, and a soft-O failure tail. BabyBear is below that threshold.

**New primary-source resolution:** ePrint 2026/524's 29 July record refers to a
31-bit prime; ePrint 2026/1075's 20 August record states the scaled-RAA result for
`r>=4`, `q-1>=gamma*N`, and `0<delta<=1/2`, with parameter-dependent constants.
Direct PDFs returned HTTP 403. An accessible [author mirror](https://sites.ualberta.ca/~mkhabbaz/assets/pdf/RAA.pdf)
was retrieved: SHA-256 `ccb327d58c6b6f1f5491a45a9f8c9278ce62b5adfb3902c4c4bf8d79a2ba19f2`.
Its title and pp.1/3 are **older**: `r>=9`, `q-1>=(eN)^2`, soft-O tail.
Its p.5 confirms the scaled ensemble, but its hypotheses do not admit `r=8`.
It cannot stand in for the current revision. No finite model-scale failure
probability or distance-testing algorithm is verified by these records.

Do not replace a missing certificate with `N^-6`, `N^-7`, or a sampled minimum
over selected messages. Hidden constants, small-N conditions, and conditioning
on a distance test matter. Neither `q>2^31` nor `r=8` alone fixes this.

### Exact larger-field proposal, not adoption

`maverick-q31plus-proposed-algebra` uses `q=2^31+11` solely to make a concrete
decision available. Trial division through `floor(sqrt(q))=46340` proves primality.
Smaller odd candidates `2^31+{1,3,5,7,9}` have divisors `{3,83,7,3,11}`.

Both fields use canonical u32 residues and four-byte wire elements. For the new
field, `2(q-1)=4,294,967,316` **overflows u32**; add in u64. The maximum single
multiply-add with a reduced accumulator is
`(q-1)^2+(q-1)=4,611,686,063,524,544,622 < 2^63`.
Use `u64(acc)+u64(a)*u64(b)` and reduce **each term**. Accumulating an arbitrary
dot product in u64 without intermediate reduction is invalid. No signed i32
reinterpretation of incoming residues is allowed. Signed source coefficients
are lifted by Euclidean remainder. Endian/shape/canonical-value checks precede use.

Centered bounds change from `±1,006,632,960` to `±1,073,741,829`; at the source's
`2^-16` rescaling, from `±15360` to `±(16384+5/65536)`. Quantizers and rescaling
are not otherwise changed. If all intended integer dots fit both ranges,
arithmetic decoding agrees; otherwise the field change affects numerics.
It requires fresh codes/P/Q/Qaux, bindings, arithmetic admission, and model
validation; source BabyBear performance/quality cannot be inherited.

**Minimal decision M-F:** if a larger-field development track is desired, approve
this exact separate field/profile. Otherwise continue BabyBear algebra and seek
actual-field finite-distance evidence. Approval supplies no missing theorem.
The tests here exercise both mathematical fields without selecting a new native
or model protocol.

## 3. Exact code description and certification requirements

For message dimension `d`, let `N=8d`. Descriptor contains `q,d,r=8`, **two full
permutation arrays** `pi1,pi2`, and two nonzero field arrays `v1,v2`, each length N.
All indices are zero-based:

```text
R[i,j]  = 1 iff floor(j/8)=i
Pi[i,j] = 1 iff pi[j]=i
V[i,j]  = v[i] if i=j, otherwise 0
A[i,j]  = 1 iff i<=j
G       = R Pi1 V1 A Pi2 V2 A                  # d × N
```

**Encoding `G^T*w`** (M Appendix B p.23): repeat each coordinate contiguously
eight times; set `b[j]=v1[j]*a[pi1[j]]`; prefix-sum b; gather by pi2, scale by
v2, prefix-sum again. **Syndrome `G*e`** reverses the maps: suffix-sum e;
scatter `out[pi2[j]]=v2[j]*suffix[j]`; suffix-sum; scatter with pi1/v1;
sum each contiguous block of eight. Scatter is the transpose of gather, not
another gather using the same array. All arithmetic is modulo the bound q.

`Raa.dense()` independently multiplies seven literal matrices, including both
permutation matrices and both triangular matrices, for `N<=64`. Basis-vector
checks establish encoder and syndrome equality for every vector of each tested
fixed code. This checks the Rust reference's **orientation** without importing it.
The existing `RaaNumericCode` implements the same gather/prefix and suffix/scatter
orientation, but derives public codes from a 256-bit SHA-256 seed. Its ensemble,
i8 wrapper, and `t_y=41` are not silently relabeled as this source profile.

### Full source ensemble

Draw each permutation uniformly by Fisher-Yates using unbiased bounded draws.
Draw every scale independently uniformly from `1..q-1`. Draw Gx, primary Gy,
and secondary Gprime independently. Public descriptors may be frozen after
sampling; their public status does not justify biased, adversarial, or fixed-size
seed selection. Entropy failures reject. A finite CSPRNG realization needs its
own computational sampling interpretation; the information-theoretic ensemble
statement is about genuinely uniform independent draws.

An untrusted publisher can supply perfectly valid P/Q for a deliberately bad
code. Full P/Q validation therefore does **not** establish the code-selection
distribution. A probabilistic admission record must bind an independently
controlled sampling procedure; client sampling of public descriptors is one
source-compatible way. Alternatively use a validated instance certificate.
Rejection/grinding counts and any distance-test conditioning must be explicit;
conditioned dual-LPN security must also be justified. Do not introduce an
unexamined trusted global code publisher to preserve amortization.

### Required certificate / probabilistic-admission record

For each distinct code role and instance, bind:

1. q, characteristic, dimension, N, r, target **integer** distance
   `ceil(N*delta)`, full descriptor digest, ensemble and sampling provenance.
2. Exact theorem/version, finite constants and dimension/field hypotheses,
   evaluated upper bound `p_fail`, or independently checkable instance proof.
3. For a distance tester: complete algorithm, test order w, acceptance condition,
   randomness, actual transcript, all rejected samples, finite residual-failure
   bound and preprocessing cost. M's `O(N^w)` assertion does not specify these.
4. Code reuse horizon, total distinct instances, all verification windows, and
   privacy-sample count. Privacy-code minimum distance is evidence for the
   source heuristic, not proof of dual-LPN for the admitted distribution.

The included exact certificate is deliberately stronger than corruption tests
but deliberately smaller than a model certificate. For a two-row G, normalize
each nonzero column `(a,b)` to `(1,b/a)` or `(0,1)`. A nonzero message annihilates
exactly one such column direction; zero columns vanish for every message. Thus

`d_min = N - number_of_zero_columns - maximum_direction_multiplicity`.

Both pinned `[16,2]` cases give `16-0-1=15`; small-field exhaustive enumeration
checks the method. It covers **all q²−1 messages**, not a sample. It proves
nothing about the separately generated 64-dimensional codes and is not a
privacy-friendly substitute code for the full method.

## 4. Sparse distributions and sample soundness

M Lemma 5.1 p.7 and Appendix A pp.22–23 require exactly t nonzeros. Sample an
ordered support using partial Fisher-Yates, then independent uniform values in
`1..q-1`; ignore ordering or sort position/value pairs together. Every unordered
support has exactly `t!` equiprobable ordered preimages. Uniform integers in
`[0,u)` use a fresh 32-bit word z and reject `z >= 2^32-(2^32 mod u)`, then
return `z mod u`. Never replace this with raw modulo, Bernoulli noise, fixed
blocks, duplicate support positions, zero coefficients, or public privacy seeds.

Each privacy query gets fresh e_x, including repeated input values, repeated
matrices, cancelled attempts and retries. Gx can be public and reused; e_x,
`Gx*e_x` and `P*e_x` cannot. Boost binds each pair to its exact matrix, field,
codes and one-use inventory slot. Sharing one mask across distinct matrices or
requests is not authorized.

**Concrete Eq.(1) profile:**

`t_x = ceil(2*(128*ln(2)-ln(8*n)))` for target `delta_x=1/2`.

This gives 165 for n=64 and `{160,159,157,156,154}` for the paper's
`n={1024,2048,4096,8192,16384}`. Reject infeasible `t_x>Nx` instead of clipping.
Small 4×3 algebra fixtures use weight 5 expressly as reduced **test inputs**;
the separate 64×64 fixture uses full 165/31/two-check source-derived parameters.
All fixture code arrays and challenges are deterministic/public and carry no
privacy claim.

The paper's claim is computational **dual-LPN for the actual code and exact-noise
distribution with polynomially many fresh samples**, not information-theoretic
input hiding. Eq.(1) controls a linear-test heuristic; it neither estimates all
attacks nor proves a per-query advantage of `2^-128`. M Remark 4.3 gives a hybrid
factor H for H samples. Freeze H across all stages, token positions, attempts,
requests and reuses; do not reset it by reconstructing a runtime object. A desired
whole-execution target may require a new reviewed parameter choice, not an
automatic reinterpretation of source `lambda=128`.

### Exact, independently checked verification probabilities

For a fixed encoded error z of weight w and a uniform exact-weight t challenge,
let K be support overlap. Its law is hypergeometric:

```text
Pr[K=k] = C(w,k) C(N-w,t-k) / C(N,t)
a_0 = 1; a_(k+1) = (1-a_k)/(q-1)
Pr[e^T z=0] = sum_k Pr[K=k] a_k
```

The recurrence counts cancellation of k independent uniform nonzero field terms.
Multiplying them by fixed nonzero z coordinates preserves that distribution.
It is checked by exhaustive support/coefficient enumeration over F5 and refines
the source's upper bound `(1-delta)^t+1/(q-1)`. No such calculation establishes
the minimum possible w for a model code.

With `delta=1/2,t=31,ell=2`, BabyBear's conditional single-relation bound is
exactly `961/1037629354146162278400 ≈ 2^-59.9053885704`. Two-level batch
verification with the same primary/secondary choices costs **twice** that bound,
approximately `2^-58.9053885704`. The q31-plus candidate is approximately
`2^-60.0000000067` per relation, `2^-59.0000000067` per batch.

For W fixed batch attempts and distinct verification codes C, use

`total_integrity_error <= sum_(c in C) p_fail(c) + sum_(w=1..W) [eps_w^ell_w + eps'_w^ell'_w]`.

Count a reused code's bad-selection event once. Count each shape-batch/window
with fresh checks separately. **No B factor within one correctly frozen batch,
no ell factor for the shared secondary challenge** (M Theorem 5.4 pp.8–9).
For illustration, W=65,536 BabyBear batches give a conditional sum
`961/7916483719987200 ≈ 2^-42.9054`; this leaves less than `2^-40` for all code
failure terms combined. At W=`2^19`, conditional error alone exceeds `2^-40`.
These are budget examples, not admission of 65,536 model requests or of the old
per-object reference counter. Aborted/malformed attempts consume the configured
attempt budget conservatively; retries do not get free challenges.

## 5. Concrete Protocol 3 / Protocol 2 capsule

Use raw canonical field matrices/inputs, no model quantizer. Initial native
geometry is **m=n=64**, Nx=Ny=512, `t_x=165`, `t_y=31`, ell=2.
P=`M*Gx` has shape 64×512; Q=`Gy^T*M` has shape 512×64.
Validate **all** P/Q entries against the intended matrix/code descriptors;
authentication/hash binding alone cannot establish these equalities.

Client computes `mask=Gx*e_x`, `correction=P*e_x`, uploads `xhat=x+mask`;
server returns `yhat=M*xhat`. Bind the response before drawing challenges.
Protocol 3 checks `(e_y^T Q)*xhat=e_y^T*(Gy^T*yhat)` for two independent
weight-31 challenges. On acceptance output `yhat-correction`; on every failure,
timeout or cancellation retire that query. Optimistic downstream use is tentative.

**Second native capsule:** B=2 same-shape claims, shared primary Gy, independent
secondary RAA with dimension `B*n=128`, `Nprime=1024`, r'=8,
`t'=31`, ell'=2, target delta'=1/2. This is one concrete instantiation permitted
by Protocol 2's parameter inputs, **not recovered evaluated batch settings**.

```text
Qstack = vertical_stack(Q_1^T, Q_2^T)        # (B*n) × N
Qaux   = Qstack^T * Gprime                 # N × Nprime
u_q    = Qstack * e_q                      # B*n, for q=0,1
```

The client retains every masked `(xhat_b,yhat_b)` and code/matrix/claim binding.
After all original responses are fixed, reveal primary challenges e_0,e_1.
After **both** auxiliary u responses are fixed, sample two secret e'_j. Check
every pair `e_q^T Qaux e'_j = (Gprime e'_j)^T u_q` and every original
`u_(q,b)^T xhat_b = e_q^T Gy^T yhat_b`. Only all-check success accepts.
No client secret or fresh secondary challenge is sent to the server.
The tests include an auxiliary lie orthogonal to xhat: it passes all original
checks yet fails the mandatory secondary check.

The extra round sends two sparse primary challenges and receives `B*ell*n=256`
field elements (1024 bytes before framing). With u32 indices and coefficients,
two weight-31 challenges use 496 bytes before framing. Each original claim
sends 256 and receives 256 field-payload bytes. Qaux alone uses **2,097,152
bytes**. It is not zero-overhead server work.

Do not replace Gprime with a systematic code to claim storage savings; RAA is
not declared systematic. Do not silently deduplicate repeated claim positions or
change B after publication: the ordered matrix list determines Qstack/Gprime/
Qaux. At model scale, group by `(m,n,field,primary-code)` and bind a fixed
ordered list of `(window,layer,stage,token-position,matrix-digest)` identifiers.
All shape batches must accept before that inference window accepts.

## 6. Ownership, serialization and resource gate

The full immutable manifest binds field/arithmetic version, source matrix and
model/graph identity when applicable, dimensions, explicit code descriptions,
P/Q/Qaux digests, validation result, parameters, ordered batch list, code-distance
evidence and security-budget scope. Use fixed-width little-endian u32 elements
and indices; counters/lengths are checked u64. The native code descriptor's
canonical bytes are `ASCII("pllm.maverick.raa.v1") || NUL || LE32(q) || LE64(d) ||
LE32(r) || LE64(N) || LE32(pi1[*]) || LE32(pi2[*]) || LE32(v1[*]) || LE32(v2[*])`,
where NUL denotes one zero byte. Code identity is
SHA-256 of those bytes. Matrix identity analogously hashes
`ASCII("pllm.maverick.matrix.v1") || NUL || LE32(q) || LE64(m) || LE64(n) ||
LE32(row_major_elements[*])`. These are native binding encodings, distinct from
the test JSON digests. The enclosing publication also binds numeric-profile,
model/graph and ordered claim IDs. A digest binds bytes but proves no distance.
Reject duplicated/omitted IDs, noncanonical fields, wrong shapes and excess
allocation declarations before accepting payloads.

| Owner | Required material / transitions |
| --- | --- |
| Public publication | Matrices, explicit codes, P/Q/Qaux and validation/certification metadata; no trapdoor. Source-cold acquisition and validation are charged. |
| Server | Public matrices and Q_b, framed xhat/yhat, later public primary challenges and auxiliary products. It never receives e_x, correction, local attention/KV, sampler or e'_j. |
| Client query | Fresh e_x or retained `(mask,correction)`; unique query/session/stage binding. Reserve inventory atomically; issue at most once; erase retired private native buffers. |
| Client tentative window | Frozen original claims, immutable ordered IDs, primary/secondary RNG ownership, auxiliary replies, tentative dependent computation/logits/state. |
| Accepted owner | Receives logits/state only after all shape batches accept. Query/batch handles become terminal. Failed or cancelled material cannot enter a later window. |

Source sampling uses independent randomness domains for public code generation,
privacy noise, primary challenges, secondary challenges and token sampling.
Fix responses into client-owned immutable buffers **before** requesting the next
challenge. Merely storing a reference to a mutable transport buffer is not fixation.
The scalar state oracle copies tuples and tests both replacement attempts.
Native private erasure, cryptographic entropy, transport authentication and replay
binding remain implementation requirements, not established by a Python test.

For the two-claim native capsule, four-byte code indices give a stored client
subset of **2,695,168 bytes**: two matrices 32,768; two P 262,144; two Q 262,144;
Qaux 2,097,152; two Gx, one Gy and one Gprime descriptions 40,960. Add server
copies, preprocessing/validation overlap, Qstack if materialized, queries,
transcripts, challenge support pools, allocator and scratch. Freeze a checked
complete allocation receipt beneath the profile's **16 MiB allocation ceiling**,
one thread, one live batch, at most 16 conformance attempts, before native work.
This is not a whole-process RSS claim or a model security lifetime.

P/Q/Qaux are public but their sparse access indices need not be. Fetching only
P columns at secret e_x support from the evaluator leaks the support; when
`t_x<n`, annihilating those Gx columns reveals a linear function of xhat's clear
input. Native capsule keeps publications client-local. Model-scale file-backed
client storage needs priced ownership/admission; secret-index remote paging is
not an approved substitute.

## 7. Exact residual decisions, including persistent decode

The original source model is **Qwen3-4B, eight prompt tokens to verified full
next-token logits**, with client nonlinear computation and optimistic verification
(M pp.12–14). The paper does not specify a persistent KV/cache/token-feedback
implementation. D3's approved common Qwen2.5 model does not approve invented
Maverick numerics or persistent-state behavior.

| Gate | Minimal missing choice or evidence | Consequence |
| --- | --- | --- |
| M-F: larger field | Approve exact q31-plus profile above only if requested; otherwise retain BabyBear and obtain actual-field distance evidence. | New field changes bindings, kernels, preprocessed artifacts, centered range and source-performance identity. It does not remove certification work. |
| M-C: distance/privacy evidence | Current primary theorem with finite constants for each actual dimension/ensemble, or independently verified instance certificates; explicit dual-LPN/noise/query-horizon assessment. | Source algebra/native capsule can proceed uncertified. No certified private/verifiable model claim until evidence exists. |
| M-B: evaluated batch manifest | Obtain B/grouping and secondary code/rate/distance/sparsity/repetitions from an immutable author artifact; if unavailable, explicitly select `RAA8,31,2` for the defined two-claim capsule and later freeze every model batch. | Capsule is implementable now as source-compatible conformance; evaluated-system parity and model storage/soundness cannot be inferred. |
| M-N: exact model numerics | Recover FP8 variant/scales/checkpoint, activation encoding, tie rule, BF16 rounding, nonlinear reduction/order, embeddings/head/biases. | No scalar protocol test selects an FP8 fallback, truncation rule, model port or fixed-point rescale. A complete proposed numeric manifest, not an unspecified “FP8/BF16” approval, is required if source facts cannot be recovered. |
| M-D: persistent token loop | Approve the concrete per-next-logit acceptance/feedback ownership below; separately freeze numerical KV representation and capacity from the completed model profile. | New autoregressive experiment and latency/verification schedule; cannot be labeled the source's measured eight-token workload. |
| M-S: model public-state storage | Price actual P/Q/Qaux acquisition, validation and residency; obtain storage adaptation approval if resident bounds fail. | No secret-index remote fetching, unmetered helpers or reduced-code shortcut. |

**Minimal proposed M-D semantics:** one verified next-logit window per selection.
Client owns all causal attention state, positions, masks, input history, logits,
sampler state, EOS and selected-but-unexecuted token. Read only previously
accepted state; make tentative changes under the current window. Verify every
linear claim, including output head, before releasing logits or sampling.
Sampling uses only accepted full logits. Commit selected token/sampler transition
atomically; feed the selected token once into the next window. Failed/cancelled
windows commit nothing and use new private masks on retry. A selected token is
not yet an executed token or a KV append. Persistent-KV layout/dtype, capacity,
append/continuation behavior and full-prefix-versus-cache numerical equivalence
must be fixed and tested before implementing this extension. No persistent-state
approval is asserted here.

For malicious private inference, the whole interactive view also contains
abort/timing behavior of dependent local operations. Source single-query
dual-LPN plus fresh masks alone does not certify all numeric-failure side effects.
Use fixed public schedules, bounded local admission and nonsecret failure framing;
retain the complete interactive composition argument as a proof obligation.

## 8. Next executable native slice

1. Add a **separate field-generic capsule** consuming explicit code arrays and
   canonical field matrices, initially source BabyBear. Keep the historical
   seeded/i8/41 wrapper's identity intact. Implement u64 reduced multiply-add,
   encode/syndrome, full publication validation and typed one-use mask owner.
2. Match small frozen dense P/Q/transpose vectors and the 64×64,
   Nx=Ny=512, tx=165, ty=31, ell=2 vector. Inject fixed randomness only through
   test interfaces; secure sampling is a separate native path. Enforce range,
   entropy failure, replay, wrong binding, cancellation and private retirement.
3. Add B=2 Protocol 2 with the exact profile above and two immutable fixation
   barriers. Match small Qaux/auxiliary vectors and the orthogonal forgery case;
   match the frozen 64×64 native batch's Qaux/response digests and every scalar
   equality against this independent oracle.
   Keep admission explicitly algebra-only until M-C closes.
4. Only then compose through the ordinary SDK contracts required by D5. Original
   eight-token source-model work requires M-N/M-B/M-C and resource closure;
   persistent/common-model comparison additionally requires M-D. Operator
   conformance is not complete-method, quality or performance evidence.

**Checks performed:** 19 independent Python tests pass, including golden replay,
exact source snapshot hashes, both fields, basis-vector transpose equivalence,
uniform-support/rejection distribution enumeration, exact small-code certificates,
exhaustive sparse-check probabilities, both protocol levels, malicious auxiliary
and original replies, omitted/malformed claims, fixation copies, cancellation
and one-use retirement. No production/native/model execution or commit occurred.
