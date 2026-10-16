# Two-worker lookup/body/head composition: conditional preimplementation review

Review date: 16 October 2026. Inspected local source baseline:
`99e10e2c1e0e3dbeddc5f56dcc830e549cb1f2a5`, with the current working-tree
publication documents. **A complete two-worker boundary composition is not
currently admitted.** This document specifies sufficient conditions and gives
an untimed, static, semi-honest single-worker simulation and exactness argument.
It is a preimplementation review, not cryptographic certification or evidence
that an installed binary implements the proposed protocol.

The comparator labels follow
[COMPETITOR-QUALIFICATION.md](../../paper/COMPETITOR-QUALIFICATION.md):

- **A:** independently implemented GKM-derived exact-W8A8 adaptation, using
  seeded **dense Boolean XOR-PIR** and exact additive linear delegation.
- **C:** the same complete two-worker functionality with **packed128 XOR-DPF**
  substituted for dense lookup queries. This is an architecture control, not
  the original GKM implementation and not a stronger security claim.

Both workers hold the public body matrices and the public lookup/head matrix.
The trusted client retains nonlinear computation, attention, full KV, logits,
tokenization and sampling. Each worker's *combined* lookup, body and head view
is the corruption unit. Separate endpoints within one worker are not separate
adversaries. Both workers together are outside the guarantee.

The GKM paper specifies dense XOR-PIR, not DPF lookup. Its commitments are
published offline, but verification of linear outputs and recovered embeddings
is online, possibly delayed/batched, with rejection before accepted output.
A/C here explicitly omit verification and make no malicious-result-integrity
claim. The pinned upstream source and extracts remain documentary input only:
no upstream code was executed, imported, built or linked for this review.
See the qualification card for the original fixed-point/ring64 numerics,
paper/source rounding disagreement, seeding and inspected abort-path concerns.

## 1. Fixed functionality and actual numerical contracts

Let `M` fix the source/checkpoint/configuration/tokenizer identities, compiler
plan, semantic schedule, quantizer and scale bytes, all local coefficients and
biases, kernels and float operation ordering, workload bounds, output codecs,
party mapping, sampling/termination policy and initial-state contract. Distinct
topologies have distinct composition/session digests; equality is required for
their declared numerical computation, not for topology-specific digest bytes.

Let `A_s` be each signed-byte body matrix and `W` the tied `V × H` head matrix.
The admitted boundary subset has one primary lookup, one bias-free tied full
head, batch one, ordinary client-owned full KV and a final-position-only head
at each sampling decision. “Full head” means **all V logits** at that position,
not all prefill-position logits. Body biases remain client-local and are added
exactly once. Untied embeddings, auxiliary lookup projections, nonlocal
nonlinear regions and different state representations require another contract.

### 1.1 W8A8 is not the complete numeric identity

The actual contracts that an adapter must preserve are:

| Item | Inspected contract and integration consequence |
| --- | --- |
| Default precision | `SymmetricPerRow` defaults to weight/activation bits `(8,8)` and `causal_reduction=None`; `RuntimeComposition` likewise defaults to W8A8. The low-level Python quantization helpers default to **4 bits**. All adapter calls must pass admitted bits explicitly. Low-level boundary A4 support does not qualify a W8A8 comparator. |
| Quantization | Each finite float32 row uses `scale=max(abs(row))/127` in float32, or `1.0f32` for a zero row, then float32 division, ties-to-even rounding and clipping to `[-127,127]`. Scales must be finite and positive; underflow/overflow is rejected. Weights use the same per-row scheme. Rust `codec::quantize` and Python `quantization.py` define these operations. |
| Integer outputs | The target is the ordinary signed integer dot product recovered exactly, then dequantized. Body profiles currently choose the first sufficient ring from `u16/u24/u32` using the actual public row-L1 bound. Boundary head inputs/products use **u32**, even when outputs are narrowed. A comparator described as ring32 must explicitly state whether sufficient smaller body rings are an admitted encoding optimization. |
| Float dequantization | With exact signed accumulator `z`, compute `fl32(fl32(float32(z) * activation_scale) * weight_scale)`, then the separate local float32 bias addition if present. Do not multiply the scales first, retain a float64 accumulator through scaling, fuse the bias, or change integer-to-float rounding. |
| Lookup | Recover `i8[H]` followed by `little_endian_f32(scale[j])`, then compute `float32(W[j,:]) * scale[j]`. Return the original scale bytes, not a newly rounded or recomputed scale. Subsequent embedding multipliers remain the compiled client's operation. |
| Default attention | With `causal_reduction=None`, `SemanticDecoderRuntime` uses its declared float32 attention paths, including `einsum` or layout-specific helpers and softmax reductions over the allocated axis. W8A8 alone does not make those reductions invariant under changed prefill partitioning or state layout. |
| Canonical attention option | Explicit `causal_reduction="prefix_f32"` dispatches to `causal_reduction.py`: for each absolute query position, contiguous products and float32 sums over exactly its causal prefix; softmax over that prefix; weighted values over the same prefix. This is an optional **different numeric contract**, not the default or a synonym for W8A8. Select it identically in all comparison arms and the clear oracle before using its prefix invariance. |

Relevant source: `python/pllm/quantization/__init__.py:18-60`,
`python/pllm/runtime/quantization.py:77-162,179-202`,
`crates/pllm-core/src/codec.rs:142-174`,
`python/pllm/runtime/semantic_executor.py:954-1045`,
`python/pllm/runtime/causal_reduction.py`, and
`python/pllm/runtime/model_binding.py:494-505`.

The canonical option does not itself authorize prefix reuse or continuation.
For the default contract, keep the same partitioning, layout and reductions in
the exactness comparison. For either option, pin the actual runtime/kernel
environment and compare against an independent numerical reference. Algebra
does not establish equivalence across arbitrary BLAS, NumPy, native builds or
floating-point implementations. Neither option establishes equivalence to the
original FP/BF16 checkpoint or a task-quality result.

### 1.2 Source-bound table and scale identity

The public lookup table is

`T[j] = bytes_i8(W[j,:]) || bytes_le_f32(scale[j])`.

Both workers must validate the same source lock, checkpoint, configuration,
compiled semantic schedule, weight digest, scale digest and interleaved table
digest before admitting a request. The source-selected quantizer and tied row
orientation must agree with the clear decoder. The client holds the public
scale vector and its admitted digest, but not W; after lookup it checks the
returned scale is finite, positive and **byte-equal** to its scale for private
index j. This comparison stays client-local. A public scale vector does not
authorize disclosing which entry the client selected.

`PrivateBoundaryTransport.token_lookup` performs this check and the required
float32 multiplication (`private_boundary_transport.py:307-345`).
`private_boundary_loader.py:223-251` and
`private_boundary_worker.py:90-115` bind the actual weights/scales/table.
`private_pages.rs::Table::Vocabulary` and `Server::from_matrix` retain the
immutable matrix owner and the original scale bytes rather than separately
quantizing the embedding and head. Sharing that *public* owner across roles
does not permit sharing any private pad or seed. It also does not prove an
allocation/RSS bound without a separate lifetime measurement.

An honest source authority and correct worker admission are premises. A thin
client cannot rehash absent W. A worker-supplied digest alone is not independent
source approval; a scale comparison alone is not authentication of the weight
bytes against a malicious worker.

## 2. Parties, joint view and leakage

There are exactly three online principals: client `Cl` and workers `S0,S1`.
Each `Si` executes lookup, every delegated body stage and the full head. The
client receives all reconstructed outputs; a worker has no authorized clear
token, activation/activation-scale, KV, logit or sampler output. Public weights
and public weight scales are not confidential.

The static adversary selects `S0` or `S1` before execution, follows the protocol,
and retains all of that worker's messages, received seeds/keys, expanded masks,
local arithmetic and state across **all three roles and the entire history**.
The permitted coalitions are `∅,{S0},{S1}`. The statement does not cover later
corruption of the other worker, even if the first worker has shut down. It does
not rely on deletion or secure erasure. A common host/operator, TLS terminator,
logger or secret store with access to both sides counts as both workers.

Assume private authenticated client–worker channels. Public/session identifiers
may link all operations of the request; no unlinkability or anonymity is
claimed. Credential differences and distinct ports do not establish independent
operators. Body and boundary subservices on the same worker may share public
metadata and storage, but neither may receive the other worker's private share.

### 2.1 The precise untimed abstraction

Let `z` denote private input/initial client state, `σ` independent sampler coins,
and `η` externally specified application controls. Define semantic leakage
`L0(M,z,σ,η)` to include:

- public source/configuration, protocol/codec choices, geometry, bounds,
  scales/layouts/certificates, stage identities and worker-role mapping;
- executed prompt/feedback row counts, head count, requested limits, public
  batch boundaries and logical operation order/parallel-group membership;
- directions and **structural** message lengths, and semantic termination
  (cap, natural EOS, or an exogenous logical cancellation cut).

For every fixed semantic run, L0 is independent of cryptographic coins `ρ`.
Fresh random public identifiers are sampled consistently by the simulator,
not supplied as arbitrary extra functions of hidden pads. Index values,
activations, selected-token equality tags and private cache keys are not
authorized leakage. The base proposal has no private-token cache/query omission.
If omission, deduplication or prefix reuse is later introduced, its additional
leakage and exactness must be explicitly modeled; it cannot inherit this contract.

This is an honest abstract execution with no clocks, timing fields, timeout,
backpressure, retry, rejection or implementation-failure observations. Logical
cancellation cuts must be independent of cryptographic coins. Race-dependent
completion order and wall-clock-triggered cancellation are excluded. This is
**not** a real execution conditioned on observed success or a favorable timing
trace. `L0` may reveal semantic information about z; the theorem protects only
what is not implied by that leakage.

The reference functionality here is restricted to supported, finite numerical
executions satisfying the quantizer, accumulator and local-operator
preconditions. Arithmetic/domain rejection behavior is a separate contract;
excluding it from this abstract functionality is not conditioning a measured
runtime transcript on having passed its checks.

The offset runtime serializes measured `server_ns` in both raw and row-residue
responses (`transformer_engine.py:1625-1651`, `residue_codec.py:91-118`). Those
fields, their variable-length encodings, packet timing and error observations
are absent from the abstract view, not asserted safe by labeling them leakage.
For example, in `Z/4Z`, `l=q-a` and `T=a mod 2` separately have input-independent
distributions for uniform a, but `(l,T)` reveals q's parity. Thus the theorem
below is not a theorem about the full current wire transcript. See
[SECURITY-AND-CORRECTNESS.md](../../paper/SECURITY-AND-CORRECTNESS.md), §§5–7,
for the same scope distinction in the prepared-body composition.

## 3. Proposed protocol A

All operations have an authenticated public binding containing protocol/version,
source/table or stage identity, plan/numeric contract, paired request, worker
side, epoch/incarnation, operation kind, sequence/ticket and public dimensions.
The precise new wire schema is integration work. The formulas below do not
authorize bypassing existing compiler or transport admission.

### 3.1 Seeded dense XOR-PIR

For a causally chosen in-range token index `j`:

1. Cl draws a fresh secret seed `k_L` independently of all other operation coins
   and a fresh public query identifier. It expands
   `b = G_L(k_L, lookup_context)` into V Boolean selector bits.
2. Cl sends `d=e_j XOR b` to S0 and `k_L` plus its public context to S1. S1
   regenerates b. The seed is never sent to S0 or a cross-worker shared logger.
3. Each worker scans the public table and returns
   `r0 = XOR_v d[v]·T[v]` or `r1 = XOR_v b[v]·T[v]`.
4. Cl validates ownership/framing, XORs the two records, checks exact scale bits
   and dequantizes as in §1.2.

Here multiplication means inclusion of a bit string under a Boolean selector.
The complement is dense: its unframed size is `ceil(V/8)` bytes. Bit order and
unused final-byte padding must be fixed, with identical projection to the V
real bits on both sides; for example, LSB-first bits and zero unused high bits.
Pseudorandomness is needed only over the V used bits. Do not introduce
index-dependent compression or omit queries for repeats. A prefill batch uses
fresh independent seeds per lookup, or an independently reviewed disjoint-stream
construction; one mask reused across batch rows is not batching.

The initial proposed implementation should use fresh seeds per query. This
specification assumes a reviewed domain-separated PRG; **the inspected current
boundary transport has no dense-PIR branch or dense-seed frame**. The present
`packed128_xor_dpf_v2` identity must not label this new protocol.

### 3.2 Exact additive linear delegation

For every body stage input, Cl performs the admitted per-row quantization
`(x,α)=Q8(h)`. For a public integer matrix A and ring `R_m=Z/(2^m)`, Cl draws
a fresh seed `k_s`, expands `r=G_lin(k_s,stage_context)` over the full ring input
space and forms `u=x-r mod 2^m`. S0 receives u; S1 receives k_s and the context,
regenerates r, and each returns its local product:

`y0=A u mod 2^m`, `y1=A r mod 2^m`.

Cl reconstructs `y0+y1`, centers the admitted residues and applies private α,
the public weight scales and the local bias. No activation scale is sent.
The existing offset request uses an all-ones `activation_scales` placeholder;
it is not the actual α (`offset_reference.py:181-199`).

For the full head, set `A=W`, `x=q=Q8(final_hidden).values`, `m=32`; use the
same additive formula and reconstruct all V outputs. The public contract must
fix the head-share encoding:

- **Seeded head:** use a fresh, separately domain-bound head seed as above.
  This retains seed compression for the full linear delegation too; it requires
  new admission because the installed boundary head accepts raw words only.
- **Raw head:** draw a fresh uniform `a ∈ R_32^H`, send `q-a` to S0 and a to S1.
  This is the installed boundary head's choice. It has the same correctness
  and ideal-mask privacy, but its full upload cost must be charged explicitly
  rather than describing it as a seed-compressed head.

Either is a declared A/C variant, fixed before execution. Comparing A with C
to attribute a lookup-DPF effect requires the same linear/head encoding and
batch/dispatch policies. A default constructor is not evidence that seeded
sharing has been enabled: `TwoOnlineOffsetLinear` currently defaults to raw
input, raw output and sequential dispatch; the transport's seeded body branch
is selected by `input_encoding="seeded"`.

Seed-first dispatch may send a fresh independent seed to S1 before S0's
complement is ready. A seeded batch has distinct coordinate/row stream offsets;
separate calls get fresh seeds. A seed's expansion must never restart at an
already used coordinate. Responses are raw ring residues or the fixed narrowed
output residues of §5.2. Individual shares and products may wrap arbitrarily;
they are not quantized i8 inputs and must not be clipped or requantized.

There is **no preparation correction** in this two-online construction. In
particular, neither worker is sent the other worker's seed or product. Do not
import the prepared-body `Ar-s` message or its role custody into this protocol
under a relabeling. Its simulator and leakage model are different.

## 4. Protocol C: packed-DPF substitution

Replace only §3.1 with fresh key generation
`(K0,K1)=Gen(j,public_lookup_context)`. Send Ki to Si; each evaluates its local
selector vector fi and returns `XOR_v fi[v]·T[v]`. Reconstruct and check the
same record. Linear/head sharing, client computation, state and public semantic
schedule remain those declared for A.

The inspected local construction is `crates/pllm-garble/src/private_pages.rs`,
whose module explicitly calls itself **unreviewed**. It uses:

- `G=ceil(V/128)` groups, depth `d=ceil(log2 G)` (including depth zero),
  addressing `j=128·a+b`;
- independent 128-bit roots, initial controls `(0,1)`, shared 17-byte
  correction words, and one shared 128-bit final word;
- a fresh 128-bit query ID and the descriptor/context-bound AES domain;
  variable-key AES-128 expansion of distinct tree/level/counter blocks and a
  separately tagged 128-bit leaf expansion;
- a `65+16+17d+16 = 97+17d` byte key and a `65+H+4` byte reply, independent
  of j for fixed geometry. All real records are scanned; nonexistent trailing
  records in the last group/tree are ignored. Current batches admit at most
  64 keys (`issue_layout`, `domain`, `stretch`, `leaf`, `evaluate_batch`).

Its BGI/Gilboa–Ishai lineage is documentary attribution, not a transfer of a
published security reduction to this exact encoding, packing and AES use.
The privacy assumption in §6 must cover serialized roots, shared correction
words/final word, query IDs, all evaluated selector bits and retained state.
Neither exhaustive algebra checks nor uniform-looking masks prove that
assumption. AES-128 roots also rule out asserting 256-bit composition security
merely because body seeds are 256 bits. Full scans are not a machine
constant-time proof.

## 5. Exactness argument

### 5.1 Exact record recovery

For A, `d XOR b=e_j` for **every** expansion b, including a deterministic or
insecure one. By XOR linearity,

`r0 XOR r1 = XOR_v (d[v] XOR b[v])·T[v] = T[j]`.

Thus seed expansion agreement gives unconditional algebraic correctness; PRG
security is needed for privacy, not this equation.

For C, the actual tree construction has the following invariant. Off the
active path, the two corrected seeds and controls agree. On the active path,
the controls differ. At each level, the common losing-branch seed correction
cancels the seed difference there, while the two control corrections make
controls equal off-path and opposite on-path. Equal off-path states stay equal
under deterministic expansion. At the target leaf, the common final word is
`F(s0) XOR F(s1) XOR e_b`; exactly one differing control applies it. Therefore
`f0 XOR f1=e_j`. This also covers depth zero. Truncating nonexistent records
preserves equality because j is in range. The same XOR equation recovers T[j].

This is an algebraic argument for the indicated algorithm, conditional on its
implementation/serialization agreement, not its cryptographic privacy proof.
It recovers the entire original record, including scale *bits*. Applying §1.2
then produces the same float32 embedding as the tied clear lookup.

### 5.2 Integer products and narrowed output residues

For every row, distributivity gives

`A(x-r)+Ar = Ax mod 2^m`.

Let the actual public row bound be
`B_k=127·sum_i abs(A[k,i])`. It remains valid for signed-i8 weight byte -128;
weights from the admitted symmetric quantizer normally lie in `[-127,127]`.
Require `B_k < 2^(m-1)` for exact centered reconstruction of the ordinary dot
product. This strict bound is about the **unmasked sum**, not its masked
terms. Bias/scales are applied after reconstruction, not independently on
worker outputs.

If worker i sends just `y_i[k] mod 2^b_k`, require `1≤b_k≤m` and
`B_k < 2^(b_k-1)`. Reduction is a ring homomorphism because `2^b_k` divides
`2^m`, hence

`center_bk((y0[k]+y1[k]) mod 2^b_k) = sum_i A[k,i] x[i]`.

The center map subtracts `2^b_k` at residues **greater than or equal to**
`2^(b_k-1)`. A sufficient minimal symmetric width is
`max(1, bit_length(B_k)+1)`; a zero row needs one bit. Positive `2^(b_k-1)`
cannot be admitted: it aliases to a negative value. Native packing is LSB-first
with fixed public row order and canonical zero tail padding
(`offset_transport.rs::{row_residue_bits,pack_row_residues,reconstruct_rows}`).

For the boundary head:

- raw `u32` uses width 32;
- uniform residues use `B*=H·128·127` and `b=bit_length(2B*)`, deliberately
  conservative even for low-level A4 descriptors;
- source-certified row residues use the actual B_j widths, bound to the
  source/weight/scale/table identity. Workers rescan their immutable W; the
  thin client needs an independently approved publisher pin. A certificate or
  checksum is not a proof against a malicious worker/publisher.

These output codecs keep **head input shares at 32 bits**. A post-decode range
check cannot repair an undersized layout: true `z=6`, encoded in three bits,
centers to `-2`, which still passes `abs(value)≤6`. Width soundness must precede
execution. Similarly, a peer-selected row width, or an unverified digest of it,
is not a substitute for source-bound admission.

If a later adapter additionally narrows inputs to t bits, it needs the separate
condition `t≥max_k b_k` and uniform/pseudorandom masks in that input ring; then
discarded multiples of `2^t` vanish in every output ring. Current boundary
head code does not do this. Small t need not encode the entire quantized input
injectively; do not claim full x recovery from colluding shares for every such
extension. The installed body widths 16/24/32 and head width 32 do encode
`[-127,127]` injectively.

### 5.3 Float, state and sampler exactness

Couple the clear reference and A/C on identical admitted source/numerics,
initial client state, prompt tokens, sampler RNG state and semantic controls.
Cryptographic coins remain independent and need not match between arms.
Induct over the same compiled operation order:

1. §5.1 gives identical embedding bytes and float32 values.
2. §5.2 gives identical signed body accumulators. Source-bound scale bytes,
   private activation scales and the ordered float32 operations of §1 give
   identical body outputs, including local biases.
3. The same client-local RMSNorm, RoPE, attention, softmax, SwiGLU, residual,
   coefficient and full-KV operations receive identical values and perform
   the same reductions. They produce identical subsequent activations, active
   KV tensors, positions and lengths under the selected default/canonical
   contract. This claim does not compare different prefill partitions under
   default arithmetic.
4. The head gives identical **complete logits before sampling**. Any declared
   post-head transform is the same local operation. Greedy sampling uses the
   same `argmax` and tie behavior. Stochastic sampling uses the exact installed
   float64 temperature/probability calculation, sorting/cumulative top-p rule
   and `rng.choice`, with identical RNG state, so both choice and next RNG
   state agree. The top-p mask uses cumulative mass `≤top_p` with the first
   entry forcibly kept; substituting a sampler that always includes the
   threshold-crossing token is not exact. Greedy does not consume this RNG.
5. Equal selected tokens feed the next lookup and operation sequence. These
   causally chosen inputs remain the clear semantic trajectory, independent
   of mask/seed/DPF coins by the preceding correctness induction.

The sampler implementation is
`transformer_client.py::MaskedTransformerClientRuntime.sample:1828-1846`.
Sampler seeding is not a source of protocol randomness. Capture full logits
before any alternate sampler could mutate them; matching text or argmax alone
does not establish logit or state equality.

For a fresh prompt of n tokens and N non-EOS selections reaching the cap, the
usual final-row schedule executes `n+N-1` lookup rows and N head calls. The
last selected token is **pending**, present in output history but not yet in
executed KV/history. Natural EOS consumes a head decision, emits no EOS token
and performs no EOS feedback lookup; if K non-EOS tokens were fed back before
EOS, there are `n+K` executed lookup rows and `K+1` heads. Body stage row counts
come from the actual compiled schedule, including terminal-row optimizations,
not a blanket multiple of lookup count.

`client.py:4895-4944` checks cancellation after yielding text and before feedback,
and records the final pending token at the cap; `client.py:5140-5143` checks
position against executed versus pending history. Exactness evidence must
compare full logits, active KV, position/length, executed history, pending
token(s), emitted selections and RNG state at declared completed boundaries.
Uninitialized KV capacity, process-local seals, random request IDs and transport
counters need not be bit-identical semantic state.

Continuation must carry the exact execution basis, active KV/position, history,
pending token and sampler/RNG policy into newly admitted owners. Do not execute
the pending token twice or treat it as already cached. If the API deliberately
reseeds sampling on a new request, the clear reference must do so identically;
do not silently assume preservation. Existing boundary SDK execution rejects
`previous_response_id` (`client.py:3745-3747`) and state/cache-reuse composition.
Generic offset continuation support does **not** admit combined boundary
continuation. The theorem extends inductively to an explicitly admitted
continuation protocol satisfying these premises; current support and its
conformance evidence remain unresolved.

## 6. Conditional single-worker privacy

### 6.1 Assumptions, including concrete primitive review obligations

**P1 — Honest fixed computation and channels.** The client, source/configuration
authority and any approved layout publisher are correct. Providers are static
semi-honest, channels are private/authenticated, and the complete public
contract is enforced. The adversary never acquires the other worker's view.

**P2 — Fresh independent entropy.** Seeds, raw head masks, DPF roots and public
IDs use reviewed cryptographic entropy, independent of sampler coins and all
other purposes. No cloned/restored RNG state or deterministic sampler-derived
seed is permitted. Raw linear masks are uniform over every full ring coordinate,
not small i8 values. Distinct seeds are drawn per operation; batched coordinates
are disjoint. Independent draws permit only negligible accidental collisions.
New IDs alone do not repair reused pads.

**P3 — Concrete seeded-stream pseudorandomness.** The joint outputs of the
chosen dense-selector and linear PRGs are indistinguishable from independent
uniform vectors to a party that does not receive their seeds, for the admitted
polynomial number of calls and public contexts, with independently generated
auxiliary state. The assumption includes key derivation, expansion, truncation,
bit/word ordering and counter bounds, not merely hash collision resistance.

The existing offset implementation is **AES-256**, not the prepared-body SHAKE
stream: `offset_transport.rs::seeded_share` derives its key by SHA-256 over a
versioned domain, 32-byte seed, 32-byte context, count and ring bits, and encrypts
distinct counter blocks; low bits form 16/24/32-bit words. Python's context
hash commits model, stage, body, weight, plan, composition, recipient session,
ticket, rows, columns and bits (`offset_codec.py:24-93`). No dense-selector
PRG is instantiated by those facts. Its separate domain and construction need
specification/review. A SHA-256 identity/KDF plus AES name is not itself the
needed multi-instance PRG proof.

**P4 — Actual packed-DPF single-key privacy, for C only.** For either party and
fixed admitted public context, its *serialized actual packed128 key* for an
in-range j is computationally indistinguishable from a fresh key for fixed
dummy index 0. This holds jointly for polynomially many causally chosen queries
and auxiliary information independent of each query's fresh coins, including
that worker's retained body/head seeds, shares and state. Public IDs have the
same distribution and bindings in both experiments. This is a primitive
assumption requiring independent cryptographic review, not established here.
DPF key privacy implies privacy of the deterministic local evaluation/reply.

**P5 — Domain/custody separation and identity integrity.** Lookup, body stage
and head use separate versioned purpose domains and independent fresh secrets.
Bind protocol variant, source/table/weight/scales, numeric ring/layout,
paired request, worker side, epoch/incarnation, operation/stage/sequence and
row/coordinate allocation, directly or through an unambiguous admitted digest.
Retain intended correlations within one DPF pair (query ID, correction words,
final word), and within one additive pair (same mask expansion at Cl/S1).
Do not independently salt the two reproductions of one mask: they must agree.
Likewise, a DPF pair uses the same expansion domain on both sides; bind the
party tag in the authenticated key/envelope rather than changing the common
DPF expansion domain separately for each party. Correctness needs those
intentional correlations even though different queries/purposes are independent.
Bind actual worker mapping and every service admission to the same request.
Collision resistance supplies public identity binding, not secrecy or remote
attestation. Fresh independent secrets are still required even with good domains.

**P6 — Coin-independent semantic leakage.** The execution and abstract view are
exactly those of §2.1. Honest reconstruction gives the coin-independent semantic
trajectory of §5. Later tokens/activations may depend on all prior *clear*
outputs and sampler coins, but never on the current fresh protocol pad. Timing,
errors, retries and arbitrary pad-dependent control signals are not appended
to L0. This excludes the invalid inference “each message is uniform, so it is
safe under any extra leakage.”

### 6.2 Theorem and simulator

**Conditional theorem.** Fix M and an admitted honest abstract semantic run
with leakage L0. Under P1–P3 and P5–P6, protocol A's complete joint abstract
view at either statically selected single worker has a polynomial-time
simulator given only M, L0 and that worker's prescribed public/setup inputs.
Under P4 as well, the same holds for C. In particular, two such semantic runs
with the same L0 have computationally indistinguishable single-worker views.
The simulator receives no private token, activation scale/vector, KV, logits
or sampler state beyond what L0 reveals.

**Proof.** First generate correctly related public sessions, tickets/query IDs,
admissions and operation counters. Follow L0's logical order and batch groups,
including any declared exogenous logical cutoff. Public immutable weights,
scales, source records, layouts and the worker's own channel/setup state are
provided or generated consistently. Consider each fixed worker separately:

| Operation | Simulator for S0 | Simulator for S1 |
| --- | --- | --- |
| A dense lookup | Sample a uniform V-bit vector d with canonical public padding; compute its XOR table reply. | Sample the real fresh seed k_L, expand b under the actual public domain, and compute the reply. Include the seed and expansion in the view. |
| Seeded body/head linear | Sample a uniform input vector u in the admitted ring and compute A·u plus the chosen output encoding. | Sample the real fresh seed, expand r and compute A·r and encoding. Include all seed/local state. |
| Raw body/head linear | Sample the real uniform input share, compute its product/encoding. | Sample the real uniform input share, compute its product/encoding. |
| C lookup | Generate a fresh real DPF pair for dummy index 0, keep K0 only and compute its evaluation/reply. | Generate a fresh real DPF pair for dummy index 0, keep K1 only and compute its evaluation/reply. |

For S0, P3 replaces each unseen seeded selector/linear mask by an independent
uniform vector. The translations `b↦e_j XOR b` and `r↦x-r` are bijections for
every fixed j/x. Thus S0's new input is uniform conditional on the history and
the next clear semantic input; the simulator need not know that input. Raw
head masks give this step statistically without a PRG replacement.

For S1, a seed sent to it is **not hidden and cannot be replaced by an unrelated
uniform expansion while retaining the seed**. The simulator samples and expands
the actual seed, exactly as the protocol does. Its distribution is independent
of j/x, and the worker receives no complement. Seed-first dispatch is simulated
at its original logical point; later activations depend on the clear execution,
not that seed. The resulting product and all retained intermediate values are
deterministic functions of this simulated input and public A/T.

For C, P4 replaces each fresh real key with a dummy-index key for the corrupted
worker. Do not replace a DPF key with random bytes or assert that its evaluated
selector is uniformly distributed. The common correction words and final word
are preserved by honest dummy key generation. Evaluation, table replies and
internal expansion state are deterministic postprocessing. Known body seeds
on S1 are permissible auxiliary state because the DPF coins are independent.

Apply these replacements in causal order, one operation or disjoint batch
stream at a time. Exact reconstruction makes all subsequent secret inputs
functions of the same clear semantic run and σ, rather than fresh protocol
coins. Hence even tokens selected from prior full logits and inputs computed
from prior body/lookup results satisfy the conditional fresh-coin premise.
This is causal input selection, **not adaptive corruption**. It does not
extend the theorem to arbitrary applications that choose x as a function of
the very pad being used to hide it.

P2/P5 make lookup, body and head replacements valid in the worker's **joint
history**, including every seed it knows. Combining independent per-purpose
fresh coins is essential; the conclusion does not follow from separate
uniform marginals. Reusing the same public W across roles adds no private
random variable. Structural framing, public checksums, honest acknowledgements
and logical replay ledgers are simulated from the resulting inputs and public
bindings. No ranks, invertibility or entropy assumptions on public matrices
or secret activations are used.

Polynomially many hybrids give a sum of relevant PRG/DPF distinguishing
advantages and negligible identity/randomness-collision terms, under the stated
channel/identity assumptions. No numerical concrete security bound follows
without primitive reductions, parameters and usage limits. In an ideal
uniform-mask A variant the masking argument is statistical; the seeded A
instantiation is computational. ∎

### 6.3 Why the joint conditions matter

- If S0 receives `x-r` for a body input and `q-r` for the head using the same
  coordinates, it learns `q-x`; separate single-call uniformity does not hide
  that relation. If a co-located subservice also receives the seed for r, it
  recovers x outright. Role alignment must be consistent across all services.
- Reusing a dense selector mask gives
  `(e_j XOR b) XOR (e_k XOR b)=e_j XOR e_k`, disclosing equality and, when
  unequal, the pair of changed indices. Renaming query IDs does not help.
- Reusing a linear mask gives the activation difference. Reusing DPF roots or
  complete keys is outside P4; changing public IDs does not automatically
  reestablish independent key-generation coins.
- S0+S1 reconstruct the dense/DPF one-hot selector and the quantized linear
  inputs in the transmitted rings. For current 16/24/32-bit inputs this reveals
  the W8A8 vector exactly, though its private float scale need not follow.
  Saved historical views suffice; the theorem never rehabilitates this
  coalition by calling prior owners cancelled or erased.

## 7. Cancellation, replay and state ownership requirements

The abstract proof covers a declared logical cancellation prefix independent
of cryptographic coins. It does not claim confidentiality of real cancellation
timing or errors. Exactness covers completed semantic boundaries/prefixes, not
delivery of a complete response after cancellation or a failed partial stage.
Already emitted text cannot be retracted.

The integrated owner must implement these concrete invariants:

1. **Joint admission.** Bind both worker identities and all lookup/body/head
   subservices to one source/plan/numeric/workload attempt. Do not execute until
   both complete admissions agree. Burn all contacted owners on partial failure.
2. **One-use allocation.** Reserve operation sequence, query/ticket and stream
   coordinate ranges before dispatch/arithmetic. Duplicate, invalid or
   exhausted operations are terminal; uncertain delivery never refunds coins.
   Seed-first work belongs to that same operation even if its complement was
   never delivered. A failed batch does not release its issued keys for reuse.
3. **Joint retirement.** A lookup/body/head failure or cancellation closes the
   entire client attempt and both workers' corresponding service owners. Reject
   delayed cross-session/role/epoch replies. Prevent output delivery or further
   token feedback after cancellation; drain outstanding tasks before recycling
   capacity or closing transports. A failed cancellation request may leave remote
   state, but must never reopen or reuse the closed client attempt.
4. **Replay scope.** Honest clients generate fresh attempt IDs and secret coins
   after restart. Existing ledgers are process-local; persistent anti-replay
   across restarts requires an independently designed epoch/incarnation/storage
   mechanism. Do not claim it from random IDs, TLS or an in-memory set.
5. **State commit.** Publish reusable state only at a complete admitted semantic
   boundary. Partial KV writes on a failed operation must not qualify as a
   completed snapshot. Coupled continuation needs fresh worker owners and
   material, source/numeric/position/history binding and exact pending-token/RNG
   semantics; reject it until those are implemented and tested.

Existing offset client sessions use two worker-generated distinct session IDs,
ticket/per-stage budgets, a call lock and seeded-task executor. Boundary sessions
use a client-generated paired ID, epoch, one global operation sequence and a
native query-consumption ledger. They have independent cancellation/completion
paths today (`offset_reference.py:399-598`, `offset_worker.py:499-608`,
`private_boundary_transport.py:207-305,393-445`,
`private_boundary_worker.py:162-178,184-294`). Neither alone proves the required
joint lifetime. In particular, simply calling two `complete()` methods does
not establish one accepted complete-decoder result after a partial failure.

Array zeroing, dropping native decoder ownership and terminal ledger entries
are not end-to-end erasure: serialized seeds/keys, immutable Python/HTTP bytes,
logs, backups and a semi-honest worker's retained transcript may survive.
No erasure, adaptive/mobile-corruption or malicious-verification guarantee is
made here. Checksums and authenticated channels also cannot reject every
in-range wrong answer from an authorized malicious worker.

## 8. Concrete source conflicts and integration gates

These are inspected admission/contract conflicts, not results of executing an
A/C decoder. The existing prepared-boundary proof does not automatically make
the following combinations executable.

| Gate | Current source evidence | Required integration invariant |
| --- | --- | --- |
| Topology/compiler | `decoder_boundary.rs::validate_composition:160-195` requires prepared boundary topology and CPU W8A8, rejecting verification/state/cache/regions/epochs. `pllm-compiler/src/lib.rs::validate_bounded_linear_composition:2123-2206` excludes an extra boundary slot from the offset composition. Python `profiles/__init__.py:690-712` independently requires masked-linear preparation/inference roles. | Introduce an explicit complete two-worker topology and A/C protocol identities, with matching compiler/profile/runtime/source admission. Keep existing guards effective until that contract exists; no digest rewriting or bypass. |
| Runtime owner types | `model_binding.py:368-409` requires `PreparedRemoteLinear` and `PreparedInventoryLease` for a private boundary; `:456-466` separately requires exact `TwoOnlineOffsetTransport` type for an offset plan. `semantic_executor.py:90-113` repeats compiled owner checks. | Bind one new coordinated offset-plus-boundary owner to the full compiled schedule; arbitrary callback injection is not admission. |
| Endpoint/custody mapping | `client.py:1109-1122` demands four credentials/origins and preparation for the current boundary. Its offset path `:1126-1159` requires two worker connections and rejects preparation. | Admit exactly two operator principals, each receiving only its consistently aligned lookup/body/head shares. Multiple services on one principal share the same corruption budget. |
| A lookup support | Compiler/loader contract hardcodes `packed128_xor_dpf_v2`; descriptor requires `leaf_bits=128`; boundary worker rejects nonpacked query magic (`private_boundary_worker.py:265-267`). | Add independently implemented dense seeded PIR, its PRG/domain and canonical dense-bit codec, distinct descriptors/costs, exact source row/scales binding and equal logical lookup policy. A cannot be obtained by relabeling C. |
| Seeded full head | Current boundary `linear` always sends two `H*4`-byte input shares; the worker admits raw u32 words. The offset seeded-share branch does not change this boundary protocol. | If retaining seed compression for every linear call, including the classifier, add and bind a separate seeded-head request. Otherwise identify and charge the raw-head variant explicitly in both A and C. |
| Thin vocabulary bundle | `offset_worker.py:157-163` exports `placement="offset"`; `model_binding.py:1928-1960` requires local lookup/head payloads except in explicitly admitted boundary/remote-head branches. | Remove client vocabulary matrices only under the new bound complete contract. Retain source identities, public scales and small coefficients; ensure complete remote head dispatch without a local fallback. |
| Body output layout | `residue_codec.py::compiled_row_layout` excludes boundary stages only for namespace `prepared`; offset worker's row-layout digest excludes token lookup/head. | New offset boundary bundle must explicitly separate body layout commitments from lookup/head contracts. Reusing the prepared skip rule or assuming missing boundary row widths are body widths is invalid. |
| Ring field discrepancy | `decoder_boundary.rs:337-343,379`, `model_binding.py:1267-1276` and `private_boundary_loader.py:178-213` encode a **minimum sufficient** `head_ring_bits` from the worst-case `H·127²` bound. Yet `BoundaryDescriptor:108-145`, transport and worker always use ring32 head shares/products. For H=8 the compiled field is 24 while the actual head wire remains 32. | Distinguish minimum sufficient reconstruction ring, actual arithmetic/input wire ring and per-output residue widths in the new contract. Do not shrink existing head shares based on that field or price them as 24-bit. This is a real source-level contract-description mismatch, not evidence of incorrect reconstructed logits. |
| Numeric defaults | High-level default W8A8 differs from low-level quantizer defaults; default attention differs from explicit `prefix_f32`; offset input defaults to raw; installed head is raw u32. | Pin explicit numeric/encoding choices in all arms and the clear/independent oracles. Do not claim canonical attention, seeded head or narrowed input shares from a component name. |
| Source bounds | Current workers rescan actual row-L1 bounds; head row certificates need independent caller pins. Low-level i8 accepts -128 whereas compiler's source-quantizer bound assumes [-127,127]. | Preserve the source-quantizer admission or conservatively recompute bounds for all admitted signed bytes. Validate strict width bounds before execution; post-decode range checks are insufficient. |
| Continuation/lifetime | Current boundary rejects previous-response continuation and optional cache/state composition; offset and boundary replay/cancellation owners are separate. | Implement the joint lifecycle of §7 and explicit continuation binding, or retain the rejection and report that conformance cell unresolved. |
| Geometry/deployment | Offset worker caps generation at 32 tokens and tensors at 4,000,000 elements (`offset_worker.py:31-37`); boundary has its own query/frame/session bounds. `network_placement.rs:233-241` rejects live private-boundary placement. | Admit only the intersection of proved and implemented shapes/limits until independently extended. Derive total call/row budgets from the complete compiled schedule, including EOS/early cancellation; unsupported cells remain unresolved. |
| Runtime leakage | `server_ns` survives in raw/residue offset payloads; admission/execute/cancel paths expose errors and wall-clock effects. | Keep this theorem explicitly untimed. A deployment/timing/error security claim requires a separate joint leakage argument and implementation assessment. |

The ring discrepancy above requires no operator witness: it follows directly
from the three independently matching compiled-contract formulas and the fixed
u32 descriptor/transport/worker contract. No witness or test file was added by
this review. The algebraic reuse/undersized-residue examples explain premises;
they are not demonstrated attacks on an admitted A/C runtime, which does not
yet exist.

## 9. Evidence needed before qualification

The API-design and implementation work must discharge the gates above, then
produce complete conformance evidence against the pinned clear decoder **and
an independent numerical oracle**. Required cases include exact embedding
row/scale bits; seeded/raw linear equivalence; all complete head logits;
sound uniform/row residues and tails/zero rows; repeated-token lookups with
fresh randomness; declared prefill/decode schedules; greedy and stochastic
top-p trajectories; active full KV/positions/history/pending tokens/RNG; EOS;
fresh-owner continuation; cancellation during each service, seed-first and
parallel work; partial admission/finalization failures; replay and failure
retirement. Fault/lifetime tests are not a timing/error-confidentiality proof.

Independent review remains necessary for dense/linear PRG instantiations and
the actual packed128 DPF, including domain binding, auxiliary-state and
multi-query security. Build provenance, source/publisher approval and operator
channel custody must be established for any deployed claim. Full-client memory,
all-link communication and matched lifecycle measurements belong to the later
parent-owned measurement gate; no measurements or runtime/build execution were
performed here.

**Result:** A and C have the conditional algebraic and static untimed composition
argument above when its concrete premises hold. Current source does not admit
either complete two-worker comparator; cryptographic certification, complete
numerical/state conformance and the closest-comparator performance comparison
remain open.
