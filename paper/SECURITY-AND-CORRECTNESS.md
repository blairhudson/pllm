# P2-B: security and correctness of the coupled boundary

Audit date: 16 October 2026. Source baseline: `99e10e2c`. This document
analyzes the implemented `PrivateLookupAndFullHead` / `PreparedBoundaryRoles`
composition. It supplies algebraic correctness arguments and a **conditional,
static, semi-honest privacy theorem for an untimed abstract protocol with
cryptographic-coin-independent semantic leakage**. It is not independent
cryptographic review, a malicious-security proof, a timing-security proof, or
proof that the installed native binary was built from this source revision.
The independent-review revision below narrows the leakage model and corrects
conditional-mask and packed-input reconstruction statements.

## 1. Audit result and publication scope

The declared conservative family of admissible static provider coalitions
consists of the subsets containing
neither both Preparation and Inference nor both boundary workers. The four
maximal admissible coalitions are **{P,B0}, {P,B1}, {I,B0}, {I,B1}**. This is a
composition result conditional on the assumptions below, not a consequence of
writing two pairwise non-collusion statements. In particular, the body input
and output masks must be jointly pseudorandom, and all boundary randomness must
be independent of body material.

No concrete break of that static, untimed abstract model was found in this audit.
That statement is not a security certification. The native private-page module
itself labels its clean-room DPF **unreviewed**. Four important limitations are
demonstrated rather than deferred to generic caveats:

1. **Retirement is not secure erasure.** A cancelled `PreparedStageRows` still
   contains its seed, which regenerates its consumed masks. Saved serialized
   DPF keys survive decoder cancellation and can be evaluated by a fresh server
   owner. Replay ledgers are process-local. This is a medium assurance/claim
   gap; it becomes a high-impact privacy defect if advertised as protection
   against later acquisition of both roles' historical state. It is not an
   attack within the stated static non-collusion model.
2. **A decoded-value range check cannot validate an undersized residue layout.**
   A true result 6 reduced to three bits centers to -2, which still passes the
   global bound 6. Honest-worker rescanning and an independently approved
   publisher pin supply the missing condition; checksums do not.
3. **Cross-role mask reuse would invalidate mixed coalitions.** If a boundary
   right share reused a body input mask, I+B1 would recover the transmitted body
   activation residues (full W8A8 values at sufficient width), and P+B0 would
   recover the head activation. Actual head code draws fresh OS bytes
   independently; this counterexample explains why that implementation detail
   is a theorem premise.
4. **Declaring timing as leakage does not justify joint simulation.** In Z/4Z,
   `l=q−a` is uniform for uniform a, and `T=a mod 2` has the same distribution for
   every q; nevertheless `(l,T)` reveals q's parity. This is a medium theorem-
   scope defect in the pre-review argument, corrected here by restricting the
   theorem to coin-independent semantic leakage. The runtime serializes measured
   `server_ns`; its joint leakage with messages is outside this theorem. The
   toy witness does not establish an actual runtime timing exploit.

Evidence: `tests/test_paper_boundary_security.py` and
`paper/artifacts/security/boundary-security-checks.json`. Twelve bounded checks
passed, including 57,600 head-sharing cases and exhaustive joint distributions
for all 16 coalitions in a small **ideal-XOR** lookup model. Three review-added
checks cover coin-dependent leakage, causal body simulation/conditional masks,
and packed-input ambiguity. These tests validate finite algebra and
counterexamples, never the computational DPF assumption. The initial nine-check
record is retained as `boundary-security-checks.pre-review.json` and is superseded.

## 2. Functionality, parties, and adversary

### 2.1 Fixed functionality

Fix public checkpoint/source identity, model configuration, quantizer, compiled
plan and schedule, tied vocabulary matrix, public weight scales, numerical
kernels, workload limits, output codec and stopping/sampling policy. Let the
immutable signed-byte matrix be `W` of shape `V × H`. Lookup record `T[j]` is
exactly `W[j,:]` followed by the original four little-endian float32 scale
bytes. The admitted SDK composition is CPU W8A8, bias-free tied head, one
primary lookup and one complete head, ordinary client-owned full KV, and
final-row-only network head evaluation.

The low-level descriptor also accepts A4; that does **not** make the compiled
five-role SDK path A4-capable. The compiler and source loader require W8A8.
Verification, state/cache reuse, placement/region variants and decode epochs
are rejected by this composition. The broader library's other protocols are
not covered by this theorem.

The ideal computation returns the pinned quantized decoder's complete logits,
selected tokens and client state to C. No provider receives a clear token,
activation scale, activation vector, KV tensor, reconstructed logits or sampler
state as an authorized output. Public weights are not confidential. Correctness
means exact execution of this numerical contract, not BF16/FP32 checkpoint
equivalence, task quality, unbiased sampling beyond the specified sampler, or
remote verification of the result.

### 2.2 Roles and channels

- **C (Client)** is trusted, including its tokenizer, Python/native code,
  sampler/RNG, local nonlinear/attention execution, full KV, response handling,
  artifact admission and any client-side supervisors/relays.
- **P (Preparation)** receives fresh body seeds from C and sends one-use
  corrections to I. It has public body matrices and source metadata.
- **I (Inference)** owns public body matrices, receives corrections from P and
  masked body activations from C, and returns masked body products to C.
- **B0/B1 (boundary_a/boundary_b)** each owns the public tied matrix. Each
  receives only its DPF key and its additive head share and returns its local
  lookup/head result.
- A **source/configuration authority** fixes the intended public computation.
  Optional row-codec **publisher** asserts bounds/layout under a caller-approved
  artifact digest. These are public-integrity trust roles, not additional online
  holders of the client's private randomness.

The theorem assumes private, authenticated point-to-point channels for C–P,
C–I, P–I and C–Bi. A provider does not observe another link's plaintext merely
because it knows the endpoints or shares public session metadata. Compromise of
a TLS terminator, host, tracing collector, shared secret store or relay that
exposes multiple roles must be counted as corruption of all exposed roles.
Provider API credentials are not cryptographic non-collusion mechanisms.

The SDK currently launches loopback children and rejects a network deployment
for this composition. Distinct URLs, ports and bearer keys establish separate
software endpoints, not independent operators. In a common-host deployment an
operator able to inspect all children is outside the allowed coalition family.

### 2.3 Exact coalition family

Let `R = {P,I,B0,B1}`. A static adversary chooses `S ⊆ R` before execution,
follows the protocol, and retains and pools all input/output messages, local
randomness and internal state of roles in S. Define

`A = { S ⊆ R : {P,I} ⊄ S and {B0,B1} ⊄ S }`.

| Size | Admissible | Forbidden |
| --- | --- | --- |
| 0 | ∅ | — |
| 1 | P; I; B0; B1 | — |
| 2 | P+B0; P+B1; I+B0; I+B1 | P+I; B0+B1 |
| 3 | none | P+I+B0; P+I+B1; P+B0+B1; I+B0+B1 |
| 4 | none | P+I+B0+B1 |

For a body input transmitted in t bits, P+I recovers `x mod 2^t` from r and
`u=(x−r) mod 2^t`. This determines full W8A8 x when `t≥8`, since reduction is
injective on [-127,127], including the ordinary 16/24/32-bit paths. Smaller
packed widths need not identify x: a zero matrix can admit one-bit widths,
under which x=1 and x=3 coincide. The P+I prohibition is conservative and
applies to the complete composition; it is not a claim of full input recovery
at every stage or every degenerate geometry.
B0+B1 reconstructs the DPF selector and each quantized head input. Neither
argument requires inversion of a public weight matrix or guessing a prompt.
The boundary pair does not automatically learn the secret float32 head scale;
its direct disclosure is the quantized vector and lookup indices. Every larger
forbidden coalition contains one of these two bad pairs. Client corruption
immediately exposes the protected computation and is excluded.

## 3. Actual messages and per-role views

Notation: matrices act on column vectors; code stores batches row-major. For a
body stage let `A` be its public integer matrix and `R_m = Z/(2^m)`,
`m ∈ {16,24,32}`. Let `x` denote the quantized activation; its float scale stays
at C. P and C expand input mask `r` and output mask `s` from a 32-byte seed.

This section inventories implementation observations. Section 6 simulates only
the untimed abstract messages/state defined by A6; duration fields, timing,
errors and other implementation side channels listed here are outside that
theorem, even when serialized inside a message body.

| Link | Private payload and relevant public binding |
| --- | --- |
| C→P | seed, 128-bit attempt ID, inventory/session ID, row count; stage/model/weight/ring context independently admitted |
| P→I | `c = A r − s (mod 2^m)`, or its admitted public row residues; attempt/session/stage, source/weight and shape/ring metadata |
| C→I | `u = x − r (mod 2^m)`, possibly reduced to the admitted packed-input ring; one-use row tickets and batch IDs |
| I→C | `v = A u + c = A x − s`, in the admitted output ring, plus public framing/timing |
| C→Bi (lookup) | its DPF key, public descriptor/context, common random query ID, party tag; no token ID field |
| Bi→C (lookup) | XOR of table records selected by its local DPF evaluation, descriptor/query/party header |
| C→B0 (head) | `l = q − a (mod 2^32)` |
| C→B1 (head) | `a`, fresh uniform `R_32^H` bytes; `q` is the quantized final hidden vector |
| Bi→C (head) | `W l` or `W a`, full ring32 or specified row residues; no activation scale |

The body fallback request's `activation_scales` field is an all-ones placeholder,
not the private quantization scale. Packed requests omit that field. The boundary
head request contains only one `H × 4` byte share. Returned complete logits are
assembled only by C.

**P's view** includes seeds, both mask streams, their public domains, corrections,
authorization metadata, acknowledgements, schedule/count/timing and errors. It
does not include `u`, boundary keys, boundary head shares or client nonlinear
state. P can always remember a seed in a static semi-honest model: forbidding
P+I must therefore apply to retained historical information, not only to overlap
of live processes.

**I's view** includes public bundle/source delivery, authorization and identifiers,
all `(c,u,v)`, per-stage shapes/packed layouts, scheduling, counts and timing.
`v` is derivable from `(c,u,A)` and supplies no extra algebraic information.
Correction masks cannot be omitted simply because I already holds public A.

**Bi's view** includes W and scales, one key per lookup (root, shared correction
words/final word, query ID), its local DPF selection bits and XOR reply, one head
input share and its public linear image, peer credential *digest*, session,
sequence, admission budgets and timing. Knowing its lookup output and its head
product adds no information beyond its received key/share and public W. Sharing
one immutable W between these operations does not share private randomness.

**Mixed views must be considered jointly.** P+Bi knows body masks, while I+Bi
knows correlated body corrections and masked activations. Neither is safely
handled by a claim that each standalone message is uniform. Sections 5–6 give
the joint argument and counterexamples to shortcuts.

## 4. Correctness

### 4.1 Packed DPF lookup: unconditional algebra

The implemented construction is a clean-room binary-tree XOR DPF. The tree
addresses `G = ceil(V/128)` groups, with depth `d = ceil(log2 G)` (zero when
G=1), and produces 128 selection bits per leaf. Write the requested index as
`j = 128 α + β`. Key generation uses two 128-bit root seeds and initial control
bits `(0,1)`. At each level its common correction seed is the XOR of the two
uncorrected seeds on the branch *away* from α. Its two correction control bits
force equal controls off the active branch and opposite controls on it.

Inductively, after each level:

1. At every off-target node the parties' corrected seeds and controls are equal.
2. At the unique active node their controls XOR to one.

For an off-target child, the seed XOR becomes zero because exactly one active
parent control applies the common losing-branch correction. The control-word
formula similarly sets control XOR to zero there and one on the continuing
active child. Equal off-target parent states remain equal under deterministic
expansion and equal corrections. This proves the induction independently of PRG
security or the particular random seeds.

The common final word is `F(s0) XOR F(s1) XOR e_β`, where `F` is the leaf
expansion and `e_β` is a one-hot 128-bit word. Each evaluated leaf is
`F(sb) XOR tb·final_word`. At the active leaf the two outputs XOR to `e_β`;
elsewhere equal seeds/controls give zero. Truncating nonexistent records at the
public end of the table preserves this result because j is admitted in range.
Thus local selection vectors obey `f0 XOR f1 = e_j`. Distributivity yields

`(XOR_k f0[k]·T[k]) XOR (XOR_k f1[k]·T[k]) = T[j]`.

This proves exact record recovery, including scale *bytes*. C verifies finite
positive scale and exact equality with its public per-token scale, then computes
`float32(W[j,:]) * scale[j]` in the same order as local tied lookup. That scale
check alone does not authenticate the recovered weight bytes against a malicious
worker; correct worker evaluation is a premise.

### 4.2 Body correction

For each one-use row, in its admitted ring,

`A(x−r) + (Ar−s) + s = Ax`.

For output row k, assume `|Σ_i A[k,i] x[i]| ≤ B_k < 2^(m−1)`.
The center map takes residues `[0,2^(m−1))` to themselves and subtracts `2^m`
otherwise. It is inverse to reduction on `[-B_k,B_k]`, so reconstructed signed
integers equal the ordinary quantized matrix product. The client applies the
same float scales and any local body bias as the reference implementation.

Output row residues of width `b_k ≤ m` remain correct when
`B_k < 2^(b_k−1)`. Packed inputs may use `t = max_k b_k`: any difference from
discarding high input bits is a multiple of `2^t`, hence its integer weighted
contribution vanishes modulo every `2^b_k`. The proof does not require a bound
on individual masked products. Public layout admission is essential.

### 4.3 Head additive sharing and exact residues

Let `q ∈ [-127,127]^H` be C's W8A8 head quantization and let weights be signed
i8, including the conservative value -128. For every output row j,

`z_j = Σ_i W[j,i] q_i`, with `|z_j| ≤ B_j = 127 Σ_i |W[j,i]|`.

For fresh `a ∈ R_32^H` and `l=q−a`, distributivity gives

`y0 + y1 = Wl + Wa = Wq (mod 2^32)`.

If `B_j < 2^31`, centered ring32 reconstruction is exactly z_j. If workers send
only the low `b_j` bits of their own outputs, reduction from `R_32` to
`Z/(2^b_j)` is a ring homomorphism because `2^b_j` divides `2^32`. Consequently,

`center_bj((y0 mod 2^b_j + y1 mod 2^b_j) mod 2^b_j) = z_j`

whenever `B_j < 2^(b_j−1)`. The strict inequality matters: positive `2^(b−1)`
would center as negative `−2^(b−1)`. Individual shares/products may fill the
whole ring; no small-share or no-wrap assumption is used. Input shares remain
ring32 in all installed boundary output codecs.

- **u32:** width 32; the transport checks the descriptor's signed bound after
  combining replies. `HeadOutputCodec.decode`'s standalone u32 branch does not
  itself impose that bound; the caller does.
- **Uniform residues:** `B*=H·128·127`, `b=bit_length(2B*)`.
  Hence `2B* < 2^b`. This deliberately uses the A8 ceiling even for low-level
  A4 descriptors. The original quantizer emits [-127,127] weights, but using
  -128 in the bound makes native signed-byte admission conservative.
- **Row residues:** the publisher scans actual immutable rows and emits
  `b_j=max(1,bit_length(B_j)+1)`. This is the minimal positive signed width for
  the symmetric interval [-B_j,B_j], including width 1 for a zero row. Workers
  independently recompute widths and a digest of the exact bounds.

The head float result is precisely
`fl32(fl32(float32(z_j) * activation_scale) * weight_scale[j])`, with separate
multiplies, not a reassociated real-number expression. Exact integer recovery
plus the same scale bits, quantizer and local operation order yields the same
float32 logits, including integer-to-float rounding for large accumulators.
Serial versus parallel HTTP changes scheduling, not that expression.

### 4.4 Whole-decoder and sampler equivalence

Couple the reference and boundary executions on identical initial local state,
input tokens, model/numeric contract, sampler RNG state and application controls.
Induct over the compiled operations. Section 4.1 supplies identical embedding
bytes/floats. Section 4.2 supplies identical body linear outputs. Local RMSNorm,
rotary operations, attention, softmax, residuals and full-KV updates then receive
identical inputs and execute the same code. Section 4.3 supplies identical full
logits. Thus every completed operation and active KV/position agree.

The ordinary sampler is client-owned: greedy `argmax`, or its existing float64
temperature/probability/top-p calculation and `rng.choice`. Equal logits and
coupled RNG imply identical choices and RNG evolution. This is pathwise
equivalence to that sampler, not a new sampler theorem. Cryptographic randomness
is independent of sampler seeding and never needs to match between arms.

For n input tokens and N selected non-EOS tokens reaching the requested cap,
there are `n+N−1` executed lookup rows and N head calls. Only the first N−1
selected tokens feed the decoder. The final selection remains pending, outside
executed KV/history; generated output history can include that pending token.
At natural EOS, the EOS decision uses a head but emits no token and triggers no
feedback lookup. Cancellation after yielding text is checked before feedback.
Admitted counts are upper bounds; early EOS may consume fewer operations.

The installed composition rejects previous-response state reuse. A claim that
the final pending token can be resumed through this same boundary owner would
therefore exceed its current SDK contract. Any future continuation requires a
separately admitted fresh owner/material protocol. Failure/cancellation supplies
no guarantee of a complete response; already-delivered output is a prefix and
cannot be retracted. API retirement prevents successful continuation of that
owner, not adversarial editing of previously delivered text.

## 5. Primitive assumptions and randomness

### 5.1 Assumptions needed by the privacy theorem

**A1 — Honest trusted computation and channels.** C and its admitted source and
publisher configuration are correct; providers are semi-honest. Channels have
confidentiality and endpoint/message integrity. RNG state is not cloned,
snapshot-restored or seeded from public sampler configuration.

**A2 — Body stream pseudorandomness.** For secret uniform 256-bit stage seeds,
the joint SHAKE256 outputs under the distinct `input-r` and `output-s` domains
are computationally indistinguishable from independent uniform streams to an
observer not holding the seed. Domains commit purpose, session, model/body,
attempt, stage, weight, rows/geometry, precision and ring/bound. Different rows
use disjoint stream slices. This is a domain-separated keyed-XOF/PRG assumption;
SHAKE collision resistance alone is insufficient.

**A3 — DPF single-key privacy.** For fixed public geometry/context and either
party, its serialized key for any in-range index is indistinguishable from a
key for a fixed dummy index, under polynomially many fresh, adaptively selected
queries and auxiliary information independent of their fresh coins. This must
cover the *actual packed128 construction*, its common correction words, final
word and public query ID, not merely scalar point functions in an unrelated
implementation. The intended primitive basis is a secure domain-separated PRG
from 128-bit seeds, instantiated here by variable-key AES-128. Applying a
published DPF reduction to this exact instantiation remains review work.

**A4 — Independent fresh coins.** Body stage seeds, DPF roots/query IDs, head
mask vectors and session identifiers come from independent OS-random draws.
All head coordinates are uniform full-width words, not small signed values.
Each prepared row and DPF query is used at most once by honest C. A stage seed
intentionally supplies many distinct input/output row slices; calling that
"one seed per token" would be inaccurate.

**A5 — Public identity integrity.** Source and schedule commitments and optional
certificate pins refer to the agreed computation; SHA-256 bindings are assumed
collision resistant. These digests supply identity checking, not privacy,
remote attestation, or proof of faithful execution by a malicious provider.

**A6 — Untimed protocol and coin-independent semantic leakage.** Let M be the
public contract, z the private input, σ the independent client sampler coins and
η any externally fixed application controls. The simulator receives only
`L0=L0(M,z,σ,η)`: public geometry/codec, operation order and structural message
lengths, input/output counts, and semantic stopping (including natural EOS,
token limits, or an exogenous cancellation independent of cryptographic coins).
Operation order means logical schedule order and parallel-group membership;
race-dependent completion order is excluded. L0 may reveal information about z,
but is independent of the mask/DPF/seed coins ρ for each fixed semantic run.
The honest clear trajectory is itself independent
of ρ by correctness. Fresh public random IDs are simulated consistently as
protocol fields, not supplied as arbitrary leakage of hidden pads.

This is an admitted, honest abstract execution with no clocks, timing fields,
backpressure, timeout/retry/rejection observations or implementation failures.
Clock-derived fields such as `server_ns` and their encoding-dependent lengths
are absent, rather than included in L0 and asserted safe. The abstraction is a
model premise, not conditioning a real run on its observed success, timing or
errors. Extending the theorem to such observations needs a separate joint
simulation or leakage-security argument; their equal marginal distributions
would not suffice.

Freshness is stronger than new identifiers: reusing the same pad under new IDs
still leaks differences. Conversely, protocol metadata and keys' shared public
correction words are intentional correlations; they are covered by A3.

### 5.2 Exact DPF instantiation

`crates/pllm-garble/src/private_pages.rs` uses 128-bit roots and 128-bit random
query IDs from `getrandom`. The AES domain is the first 128 bits of SHA-256 over
a versioned domain string, descriptor digest, query ID and table geometry.
Each stretch encrypts three distinct domain/level/counter blocks using the seed
as AES-128 key: two blocks become child seeds, and two low bits of the third
become child controls. Leaf expansion uses a distinct domain tag and yields
128 bits. The AES domains are not secret keys shared between roles.

A packed query has a 65-byte public header, 16-byte root, d correction words
of 17 bytes, and 16-byte final word: `97+17d` bytes. A reply has 65 header bytes
plus `H+4` record bytes. Query lengths depend on public V/H, not the index.
Both parties scan every real row; batch evaluation supports at most 64 keys.
It computes masks for all rows and uses masked XOR even for a zero selector.
This is not a proof of constant-time machine execution. Branches on a worker's
own DPF control bits are functions of the key already in that worker's view.

The relevant antecedents are Gilboa and Ishai, *Distributed Point Functions and
Their Applications*, EUROCRYPT 2014; Boyle, Gilboa and Ishai, *Function Secret
Sharing*, EUROCRYPT 2015,
<https://doi.org/10.1007/978-3-662-46803-6_12>; and their *Function Secret Sharing:
Improvements and Extensions*, CCS 2016,
<https://doi.org/10.1145/2976749.2978429>, revised full version
<https://eprint.iacr.org/2018/707>. These references identify the primitive
lineage; this audit does not assert exact algorithmic equivalence to a reviewed
library or inherit its concrete security bounds. AES-128 roots also preclude
describing the whole composition as having 256-bit security merely because the
body uses 32-byte seeds.

## 6. Conditional privacy composition

**Theorem (static privacy of the untimed abstract protocol).** Fix the admitted
public contract and two honest semantic executions with the same L0 as defined
in A6. Under A1–A6, for each `S ∈ A` there is a polynomial-time simulator for
S's joint **abstract** protocol message/state view given the public contract,
L0 and S's prescribed public/setup inputs. It does not receive C's token IDs,
quantized inputs, activation scales, full logits or KV beyond what L0 reveals.
The two abstract views are computationally indistinguishable. The statement
is conditional on A3, not a newly established proof of the native DPF, and is
not a theorem about the full implemented payload transcript: that transcript
contains clock-derived fields. No joint timing/error simulation is claimed.

**Proof.** It suffices to handle the four maximal coalitions; projecting a
simulated view proves all their subsets.

For **P+Bi**, simulate P by honestly drawing stage seeds and expanding r,s and
corrections. These depend on public schedule and random coins, never on x.
Simulate each boundary key using a fresh DPF key for a fixed dummy index and
its associated public header, and compute its local lookup evaluation. A3
permits this hybrid even for repeated/adaptively selected actual tokens. Draw
Bi's head input share uniformly and compute its product/residue. For B1 this
is exactly a; for B0, `q−a` is uniform for every q. Independence from all seeds
known to P follows from A4. Deterministic local replies and public framing can
be generated from these simulated inputs.

For **I+Bi**, first replace the honest P's distinct mask streams by independent
uniform r,s under A2. For any fixed x, the mapping

`(r,s) ↦ (u=x−r, c=Ar−s)`

is a bijection on `R_m^input × R_m^output`: its inverse is
`r=x−u`, `s=Ar−c`. Thus **the joint pair (u,c)** is uniform and independent of
x, even though the messages share r and I sees both. Respecting actual causal
order, the simulator first samples c uniformly **offline**, then samples u
uniformly when that row is used **online**, and derives `v=Au+c`. After c is
revealed, r remains uniform, since each r has exactly one compatible
`s=Ar−c`. Their conditional joint distribution is constrained by this equation,
not the product of two independent uniform masks. In particular, A=0 makes
`s=−c` completely known. Privacy uses the conditional uniformity of r, not a
claim that every pad remains hidden or uniform after the correction. For
multiple rows, disjoint independent slices give this argument for each unused
r conditional on all already released corrections. Public packing/reduction
is a deterministic projection of the same abstract view and preserves
indistinguishability. Simulate Bi's keys and head shares as above. A4 makes
them independent of body coins, so the *joint* mixed-role distribution is
simulated, not only its individual marginals.

Apply these replacements in causal order over fresh row slices, lookup queries
and head calls. The body simulator uses the conditional-r argument above,
including when corrections precede their online activations; fresh independent
head a remains uniform, and each fresh DPF key admits the A3 hybrid. Under A6,
semantic lengths/stopping do not additionally condition cryptographic coins.
Correctness makes token feedback, including tokens selected from earlier logits,
independent of these coins, so it does not break the induction. L0 reveals
which abstract operations occurred; no cache hit or value-based remote routing
is hidden by this argument. Polynomially many primitive hybrids give the usual
sum of their distinguishing advantages, plus negligible randomness/identifier
collision and channel/commitment failure terms. No numerical end-to-end
security advantage is asserted without a primitive reduction and usage bound.
∎

This proof explains why no rank, invertibility or entropy assumption on W is
needed. It also explains why output-s is essential: if c were merely Ar, I
could compute `Au+c=Ax`. Reusing a head mask on two calls reveals `q2−q1` to
B0. Reusing body r as boundary a breaks the two mixed coalitions described in
Section 1. The bounded tests include all three counterexamples.

## 7. Abstract leakage versus implementation observations

### 7.1 Leakage covered by the untimed theorem

L0 exposes the fixed source/checkpoint/configuration, model and stage identities,
precision, public scales/layout/certificate, V/H, codec/kernel/dispatch and
budgets; actual prompt row counts, requested output allowance, actual head and
feedback counts, and semantic completion/stopping. It also exposes public
operation order, batch sizes, directions and **structural** message lengths
under A6. An upper-bound session field does not pad away actual counts. Party
labels and public context relationships are preserved; random attempt/query/
session IDs are sampled consistently by the simulator. L0's counts may depend
on private inputs and independent sampler coins, but not cryptographic coins.

### 7.2 Observations outside the theorem

The runtime additionally exposes wall-clock scheduling, network pauses and
backpressure, tokenization/sampling gaps, output-stream pacing, handler errors,
timeouts/retries/rejections and cancellation triggered by these events. Such
observations can depend on secret state or cryptographic coins and are outside
A6. Calling them "declared leakage" does not prove that masking remains private
when jointly observed with the messages.

This issue is present in the transcript format, not only in packet arrival
times: `transformer_engine.py::prepare_seeded_stage` measures masked computation
with `time.perf_counter_ns()` and places elapsed time in `CorrectionPush.server_ns`
(baseline lines 1681–1717). `preparation_protocol.py::CorrectionPush.pack`
serializes it as field `e`. No proof here makes that measured duration
independent of masks, or simulates its joint distribution with c and u. Simply
discarding network timestamps while retaining `server_ns` and its variable-width
serialization would not produce the abstract view of Section 6.

For an exact bounded counterexample, take uniform `a∈Z/4Z`, `l=q−a mod 4` and
observable `T=a mod 2`. For every q, l is uniform and T is a uniform bit. Yet
`q mod 2=(l+T) mod 2`; the joint supports for even and odd q are disjoint. Thus
even input-independent timing *marginals* do not justify conditioning pad
privacy on a timing trace. `check_coin_dependent_leakage_breaks_payload_composition`
exhausts this example. It does not claim that measured PLLM durations implement T.

Boundary workers share the paired session, query IDs and common DPF correction
words by design. P and I share inventory/attempt bindings. Even when these
identifier namespaces differ, a mixed coalition can align runs by IP/client
identity, request order and timing. No unlinkability or cross-request anonymity
is claimed. Repeated token values receive fresh queries and do not have a
direct equality tag, but observable control flow can still reveal repetition or
content statistically; no side-channel bound was measured here.

The theorem must not be restated as "the full real system leaks only public
weights and shapes" or as a joint real-transcript/timing guarantee. A local
observation efficiently computable solely from an already simulated abstract
view can be simulated by postprocessing; this does not establish that cross-role
durations, client gaps or host observations satisfy that condition. A deployment
claim needs an independent joint leakage-security argument or enforced side-
channel controls and assessment. HTTPS protects channel contents, not visible
metadata or duration fields disclosed to a receiving endpoint.

## 8. Identity, immutable ownership, and publisher trust

The Rust compiler requires tied weight identity and bias-free complete head;
the Python source loader re-resolves the locked checkpoint, re-lowers the plan,
checks tensor headers and imports that one head. Each worker hashes its actual
immutable signed-byte matrix, scales and interleaved record identity, recomputes
the maximum row L1 bound and compares the descriptor before admitting sessions.
Native `Table::Vocabulary` retains an `Arc<Matrix>` and exact scale bytes;
lookup and head use the same matrix owner rather than separately quantized or
serialized copies. The same public matrix on two workers is deliberate, not a
secret sharing of model weights.

C's thin bundle contains public commitments/scales, not W, and independently
validates the source/schedule/bundle contract. It cannot rehash an absent matrix
or prove a provider followed the quantizer. In the semi-honest model the matching
worker admission supplies consistency with actual source data. A digest from
an untrusted provider, by itself, does not establish the intended source.

Row certificates contain source/weight/scale/lookup commitments, geometry,
numeric profile, width bytes and digests of widths/bounds, plus publisher-code
hashes. Authenticity is an independently caller-approved hash/size/path pin.
There is no signature, zero-knowledge proof or remote attestation. A path or a
hash obtained from the worker serving the artifact is not independent approval.
The client checks the pin and bounded metadata, but cannot recompute the row
bounds without W. Workers do recompute them; malformed authenticated artifacts
still reject. Publisher-code hashes record provenance assertions, not verified
execution of those binaries.

Source-based widths leak only additional public-weight information. A mistaken
pin/layout primarily threatens correctness/availability, and in a malicious
protocol can affect downstream behavior; it does not turn uniformly random
head input shares into clear activations. The alias witness in Section 1 shows
why a post-decode bound check cannot replace publisher/worker validation.

## 9. Lifetimes, cancellation, and erasure

### 9.1 Implemented logical state machines

The client boundary owner progresses `new → opening → active → closed`; it
cannot reopen. Open chooses a 128-bit paired session ID, derives a descriptor/
admission/session/epoch context, authenticates both role-specific admissions,
and attempts cancellation at every contacted role if admission fails. Context
hashes bind source/plan/workload, not private tokens.

Each active operation claims the next monotonic sequence before exchange and
consumes its lookup/head budget. The worker reserves a context before native
binding, marks operations busy, and consumes budget/sequence before native
arithmetic. A lookup context owns its own query-ID consumption ledger. Invalid
frames, replay, expiry, explicit cancellation and completion make the associated
owner terminal. Ingress authentication failures do not authorize cancellation of
another client's owner. Idle expiry is checked on activity/admission; this is
not a wall-clock secure-erasure service.

Head parallelism is limited to the two calls for the current causal step. The
client's operation lock and executor join drain outstanding calls before cleanup;
the active-state check prevents delivery after cancellation. A worker's shielded
native task retains capacity until completion and burns the slot after an
interrupted await. A failed network cancellation can leave remote material until
expiry/process exit; the closed client owner does not reuse it. Completion may
leave unused admitted capacity after early EOS.

Limits include 64 lookup rows per batch, 4,096 admitted lookup rows/head calls,
eight live worker sessions, 4,096 contexts per worker incarnation, 16 MiB data
and 16 KiB control frames, and default 90-second idle expiry. These bound software
owners and denial-of-service exposure; they are not a cryptographic erasure
guarantee or proof of a complete peak-memory budget.

Prepared masks have separate row-reservation and consumption ledgers. Distinct
row ranges use disjoint XOF offsets. Invalid/repeated claims cancel the native
cursor; failure does not refund consumed material. Public identifiers and
freshness rules do not make replay ledgers persistent across process restart.

### 9.2 What is and is not erased

Some Rust DPF states and native prepared-stage buffers use `Zeroize`/
`Zeroizing`. The Python head transport overwrites its mutable left/right NumPy
share arrays in `finally`. These are useful local measures with limited scope:

- `left.tobytes()`, `right.tobytes()`, MessagePack/HTTP bodies and serialized DPF
  `PyBytes` produce immutable copies not covered by those array overwrites.
- `PrivatePageDecoder.cancel()` removes decoder ownership; it does not wipe
  already returned query bytes or a worker's saved transcripts.
- `PreparedStageRows.cancel()` cancels its cursor/verifier but keeps
  `request.seed`. The evidence regenerates a consumed mask from this live field.
- `MaskRows.cancel()` drops SHAKE reader owners; the audited `Stream` type has
  no explicit zeroizing `Drop`, and contains both base and advancing readers.
- P's `prepare_seeded_stage` uses ordinary Python/NumPy seed/mask owners. Its
  HTTP handler also holds serialized seed-bearing `raw`/envelope/request
  objects through acknowledgement processing. There is no audited end-to-end
  wipe of Python copies, allocator pages, crash dumps, swap, logs or backups.

**Static privacy does not need erasure to rehabilitate P+I:** that coalition is
forbidden for the complete history. If adversaries corrupt P during preparation
and I later, their saved views still recover the transmitted residues of x
(full W8A8 values when `t≥8`) even if honest code later frees every object.
The analogous historical-share attack applies to B0 then B1 across time. A mobile
adversary with only one live corruption at once can therefore break a model based
solely on simultaneous non-collusion.

An erasure-based theorem would need a different corruption model: which states
are available at each epoch, what is securely erased, no previously corrupted
party keeping copies, channel forward secrecy, and assumptions about snapshots
and persistent replay state. It would also need an implementation audit that
the current Python/native boundary does not supply. At most, an adversary whose
**cumulative** corruptions stay inside A can be considered without crossing the
obvious reconstruction threshold; a formal adaptive proof still does not follow
automatically from the static simulator above. Claim static privacy here.

## 10. Implementation binding and remaining review

Paths/symbols below refer to baseline source, rather than floating documentation
claims. Line numbers are deliberately secondary to symbols.

| Obligation | Audited source |
| --- | --- |
| Admitted W8A8/full-KV/tied composition | `crates/pllm-compiler/src/decoder_boundary.rs::{validate_composition,phase}` |
| Actual locked head and row/workload bounds | `python/pllm/runtime/private_boundary_loader.py::{load_private_boundary,build_private_boundary_contract}` |
| Paired descriptor/context and strict frames | `python/pllm/runtime/private_boundary_protocol.py::{BoundaryDescriptor,context_bytes,pack_frame,unpack_frame}` |
| Head independent OS masks and local float scales | `python/pllm/runtime/private_boundary_transport.py::linear` |
| Fresh lookup issue, exact scale check, decode | `python/pllm/runtime/private_boundary_transport.py::token_lookup` |
| DPF key generation/evaluation/shared matrix | `crates/pllm-garble/src/private_pages.rs::{issue_layout,stretch,leaf,fill_bits,Server,Table}` |
| Python key-copy/decoder lifetime | `crates/pllm-python/src/private_pages.rs::{private_page_issue,Decoder}` |
| Worker source bounds and burn/drain behavior | `python/pllm/runtime/private_boundary_worker.py::create_private_boundary_worker_app` |
| Uniform/row bound and authenticated pin | `python/pllm/runtime/private_boundary_head_codec.py::HeadOutputCodec`; `private_boundary_certificate.py::{matrix_layout,import_certificate}` |
| Low-bit reconstruction/body packing | `crates/pllm-core/src/offset_transport.rs::{reconstruct_rows,pack_difference,add_packed,unmask_packed,quantize_mask_pack}` |
| Seed draws and P authorization/correction delivery | `python/pllm/runtime/client.py::_prepare_inventory`; `preparation_server.py::prepare_inventory_stage` |
| SHAKE domains and body correction | `python/pllm/runtime/preparation_protocol.py::{_mask_domain,expand_preparation_mask}`; `transformer_engine.py::prepare_seeded_stage` |
| Client one-use body ownership and reconstruction | `python/pllm/runtime/transformer_client.py::{PreparedStageRows,PreparedInventoryLease,PreparedRemoteLinear}`; `crates/pllm-core/src/{mask_stream,prepared_stage}.rs` |
| Full-head dispatch, local state and sampler | `python/pllm/runtime/semantic_executor.py`; `transformer_client.py::MaskedTransformerClientRuntime.sample`; `client.py` generation loop |
| Child credential custody | `python/pllm/runtime/servers.py::_environment` |

For publication, the algebra and conditional static coalition statement can be
used with their premises intact. P2-B should **not** be described as an
independently reviewed end-to-end privacy proof. Remaining work is concrete:

1. Independent cryptographic review/reduction for the actual AES packed128 DPF,
   domain/key generation, multi-query composition and claimed concrete security.
2. Analyze joint implementation leakage, including serialized `server_ns`,
   timing/backpressure and errors, separately from the coin-independent untimed
   theorem. Freeze operator/channel custody for the deployment being measured;
   co-located process tests cannot close these items.
3. Either retain static cumulative non-collusion wording or design and verify a
   separate secure-erasure/adaptive-corruption contract. Current cancellation
   tests establish logical consumption, not memory sanitization.
4. Freeze source/publisher trust, build provenance and complete numerical/state
   parity evidence for the final implementation. This audit runs no model and
   inherits no fresh end-to-end correctness/performance result.

Checksums, authenticated transport, bounded decoders and source digests reject
many malformed/substituted messages, but a malicious authorized worker can
return an in-range wrong product and recompute a public checksum. Lookup has no
verifiable-PIR response proof. The compiled composition rejects verification;
neither arbitrary malicious correctness nor availability is promised. These
limitations are logically separate from the static confidentiality theorem.
