# Maverick: scoped numerical-completion decision

**Mathematical map/quality-screen choices approved — D31, 23 October 2026.**
Implementation and heavy execution are not authorized. M-N's source, proof,
quality and execution gates remain blocked; M-D remains separate. Selected arithmetic, source
pins, tensor inventory, graph and acceptance rules are in
[the numerical proposal](../research/maverick-numeric-source-recovery.md), §§2–4.
This packet changes no stable contract, paper claim, capability or allocation cap.

## Recommendation and exact decision

**Do not call this recovered author numerics or stock FP8 parity.** Approve the
following only if a disclosed source-derived numerical adaptation is acceptable
for the study; otherwise keep the original-artifact evidence gate, not an
unspecified "pick FP8/BF16" task. A source-faithful independent implementation of
unambiguous paper facts needs **no new fidelity permission**. Author-code
unavailability alone is not an adaptation or permission gate. The BF16-head
exception and the proposed numerical realization below need the narrow review
requested here; source facts cannot be created by approval.

Exact numeric-only decision selected by the user; recorded in D31:

> Approve only the mathematical map and quality-screen choices in numerical
> proposal §§2–4, named `M-N/Qwen3-4B-8prompt-candidate-1`, as a separately
> disclosed source-derived adaptation. Use official Qwen3-4B-FP8 revision
> `96b30dc13593a244a5e59e84687309f53c375cfa`, multiplied 128×128 E4M3FN block
> scales, nearest-even fixed-Q8 weights and inputs, direct centered-rational
> `/65,536` to BF16, the specified F32/BF16 operation/cast/reduction order, and
> correctly rounded functions—not the official dynamic-FP8 activation path.
> Preserve client-local BF16 embedding; require the selected artifact's equal
> BF16 head copy and derive its delegated fixed-Q8 head directly from BF16.
> Scope is one eight-ID request, positions 0–7, to all 151,936 final logits;
> no token selection, feedback or reusable KV. Accept only under the stated
> honest no-wrap and all-canonical-reply totality proofs. Fix quality screening
> now to 1,024 pinned eight-token cases, at least 973 top-1 matches against the
> defined original-BF16-checkpoint oracle, mean NLL increase at most ln(1.05),
> zero failed cases, and exact delegated-versus-clear-candidate parity.
> This does not assert equivalence to the author's unknown artifact or official
> FP8 runtime, certify security, authorize checkpoint acquisition/heavy execution,
> select model batches/security parameters, change parties/roles/state/storage,
> raise caps, or waive any evidence requirement. No implementation is requested
> by this documentation-only review.

This is **one scoped numeric decision**, not permission to choose alternative
scales, roundings, checkpoints, heads or tolerances later. A mismatch/failing
certificate or quality result blocks the candidate; it does not authorize repairs.
Rejection leaves the independently implementable source algebra intact and the
whole-model numerical gate unresolved.

## Facts, choices, and evidence: three different things

| Category | Exact boundary |
| --- | --- |
| Governing source facts | arXiv **2609.10264v1**, PDF SHA-256 `c9570fdeb51a26f9d707bcc7bf485f2001e7d3e15ebdf34815123935a1e8e2a0`; BabyBear; nearest-integer `256*W`; centered `/2^16` then BF16; local nonlinearities; optimistic request-end verification; original eight-token Qwen3-4B next-logits workload |
| Known official-Qwen facts | Pinned config/index/implementation; 128×128 scales, dynamic E4M3FN activation code, Q/K norm before RoPE, no biases, tying declaration; source pins and hashes in proposal §2 |
| Missing author evidence | No immutable author repo/evaluated checkpoint/numerical manifest recovered from inspected sources. Author-linked ePrint returned 403. This is **not** proof no artifact exists |
| Proposed numerical choices | Original-family source anchor with unknown author identity; fixed-Q8 activation rule; exact ties/cast path/local arithmetic; explicit BF16 table/head exception; defined quality oracle/thresholds |
| Missing execution evidence | Actual tensor dtypes/values/ties/authentication; honest-domain no-wrap and all-reply finite/i64 proofs; correctness/quality observations; complete deployment admission and security composition |

No new training is proposed. Nevertheless replacing dynamic-FP8 activation
quantization and scaled FP8 matmul by fixed-Q8 field evaluation changes the
inference function relative to **stock FP8**. However, the paper itself specifies
FP8-to-fixed-point field execution: that difference alone is not evidence of a
paper deviation. Unknown author provenance is likewise not proof of substitution.
The explicit BF16-head exception and the completed arithmetic are the actual
reviewed numerical choices; claiming identity to either the author's realization
or the official FP8 runtime remains unsupported. Even a successful quality screen
would not establish either identity. Here Q8 means eight fractional bits, **not**
int8 weights or W8A8.

### Why the cast path must be decided

`r=16,842,753` is a valid centered integer. Exact `r/65,536=257+2^-16` rounds
to BF16 **258** (`0x4381`). F32 rounds r to 16,842,752; dividing gives exactly
257, whose BF16 tie rounds to **256** (`0x4380`). Negatives give -258/-256.
Both readings fit the paper's prose. This packet chooses **direct** conversion,
without claiming the paper establishes it. The E4M3FN subnormal `1/512` also
becomes exactly half an integer at scale 256: even/away ties differ.

### What is now numerically specified

The proposal fixes all of the following, rather than deferring them to an importer:

- Exact 651-tensor FP8-artifact name/shape set and 398-tensor BF16-comparator set;
  E4M3FN bit decoding, scale direction/axes, BF16 source tables/norms, positive
  finite scales, no biases, duplicate/extent checks, original tied head handling.
- Signed i64 Wq/xq; exact nearest-even quantization; canonical BabyBear residues;
  direct BF16 rescaling; subnormals retained; no saturation, integer-i8 clamp,
  zero-point, W8A8 substitution, F32 intermediate decode or fallback import.
- RMSNorm's cast-before-gamma; Q/K shared 128-coordinate norm weights; GQA
  `kv_head=floor(query_head/4)`; split-half RoPE; causal softmax; intermediate
  BF16 casts in attention; SiLU×up; residuals; final norm; full head; F32 wire logits.
- Increasing-index non-FMA F32 reductions, epsilon/frequency/function rounding,
  signed-zero behavior, sum initialization and reshape axes. Correct rounding is
  an oracle requirement, not permission to assume system libm or fused GPU parity.
- Exact input/workload/domain and quality tokenization: immutable Tokenizers
  v0.21.1 commit, no padding/truncation/automatic special tokens, deterministic
  tokenizer configuration, pinned Wikitext file/order/concatenation, fixed blocks,
  ordered F64 NLL arithmetic and tie-breaking.

The absent serialized head in the **original BF16** checkpoint was a real
specification hole in the earlier draft. It is fixed: use that source's tied
embedding, not the candidate's table or an invented head tensor. Source payload
validation remains undone; no missing numerical rule is delegated to user guesswork.

## Hard acceptance and security dependencies

1. **Honest numeric equivalence:** prove each intended integer row dot lies in
   `[-1,006,632,960,+1,006,632,960]` across **all** admitted eight-ID inputs.
   Public row-L1/component certificates or a tighter sound graph proof suffice;
   sampled cancellation and "no overflow observed" do not. Include all rounding.
2. **Active-server totality:** before serving, certify that arbitrary canonical
   responses cannot cause private-dependent overflow, invalid denominator, out-of-i64
   quantization, fallback or early numeric abort. Proposal §4 gives explicit
   residual/norm/attention/MLP bounds, not just an instruction to "check bounds".
   Every decoded linear reply is at most 15,360; context is bounded by 122,880;
   SiLU×up by 235,929,600. The latter's xq can reach **60,397,977,600**: finite
   does not imply safe centered integer dots. Actual source values/certificates
   have not been checked. A failure blocks admission, not secretly narrows inputs.
3. **One-use verification dependency:** fresh mask/correction for every one of
   2,017 claims; immutable masked input/output pairs; freeze every request original
   before primary reveal; freeze every auxiliary reply before secondary sampling;
   verify raw field relations, not rounded outputs; require every batch before
   the sole accepted full-logits release. Cancellation/failure retires the entire
   dependent graph and all private leases; retries cannot reuse them. Rebuilding
   an object cannot reset the security horizon.
4. **Proof is independent of approval:** actual code distance/selection failure,
   dual-LPN applicability, exact-noise distribution, total samples/attempts and
   primary/secondary batch soundness need evidence. Public-state validation is not
   code certification. Secret-index remote P-column fetching leaks mask support.
   Timing/abort/transport composition is not certified by numeric totality.
5. **No hidden role/state adaptation:** keep source client nonlinear work and one
   untrusted linear evaluator. Source-compatible authenticated framing/consuming
   owners require implementation evidence, not new fidelity permission merely
   because code is new. Moving attention/embedding ownership, introducing another
   trusted party, changing optimistic windows or adding persistent KV/token feedback
   is a separate actual adaptation, not covered here. M-D remains unselected.

These dependencies make full method execution **not yet admitted**. M-B model
partition/secondary parameters, M-C actual-code/security evidence and M-S aggregate
public-state storage remain visible gates; numerical approval cannot close them.

### Original quality versus paired protocol control

| Control | Meaning |
| --- | --- |
| Independent clear **candidate** | Exactly the same new numeric map, without delegation: required paired protocol/performance control |
| Original **BF16 checkpoint** oracle | Original pinned model tensors in the precisely defined local graph, with its own tied embedding/head: proposed model-port quality reference |
| Author's quantized clear model / official native FP8 engine | Not recovered/not run. Neither control above establishes their exact numerical or quality identity |

Proposed quality limits are **engineering choices**, not paper-recovered settings:
973/1,024 agreement; mean NLL increase `<=ln(21/20)`; zero rejected/overflowed
cases; complete bit parity against clear candidate. No after-the-fact filtering,
scale tuning or tolerance adjustment. This fixed-eight-token conditional-loss
screen is not standard long-context Wikitext perplexity or a generation-quality
guarantee. If those narrow claims are unacceptable, do not approve this screen as
a substitute; request the desired original-runtime evidence separately.

## Costs and safe order of evidence

**This decision/review uses zero model-weight/tokenizer/dataset payload bytes,
zero native launches and no build/install.** Small metadata/source inspection and exact scalar checks suffice
to review the mathematical choices. No download or heavy execution is requested.

Future work cannot be costed as another 64×64 operator. Exact schedule estimates:

| Item | Cost, not an observed or admitted peak |
| --- | ---: |
| Advertised FP8 source shards alone | 5,190,052,872 bytes |
| Full-vocabulary output | 607,744 bytes |
| One u32 copy of all delegated matrices | 16,089,088,000 bytes |
| All original masked input/output vectors | 66,678,272 bytes before framing/auxiliary/copies |
| Server linear terms per request | 29,455,482,880 field multiply-adds |
| Ordinary resident rate-1/8 P+Q for all 253 matrices | 257,425,408,000 bytes (239.746 GiB), excluding Qaux and all other owners |

First obtain/validate only already-authorized public source evidence and bounded
numeric/admission checks under the parent's serialized resource policy. Do not
launch P/Q/Qaux preprocessing to learn it does not fit. Original BF16 quality
comparison requires its separately authenticated model payload; it is not included
in the FP8 shard cost. Correct-rounded reference evaluation is not a performance
estimate. Caps remain unchanged; a storage adaptation must be separately explicit.

**Next action after this packet:** user accepts or rejects the quoted *numeric-only
adaptation and narrow quality criterion*. Acceptance records a choice, not successful
M-N execution or a certified method. If author artifacts later resolve the ambiguity,
source-faithful independent implementation can use those facts without asking for
permission to reinvent them; discrepancies with this candidate must stay explicit.
