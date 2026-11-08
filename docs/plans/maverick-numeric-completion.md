# Maverick: scoped numerical-completion decision

**Mathematical map/quality-screen choices approved — D31, 23 October 2026.**
**Pinned BF16 scale-storage amendment approved — D32, same date.**
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

### D32 storage-only amendment to D31

The original D31 decision quoted above is unchanged. A subsequent
[header-only audit](../research/maverick-fp8-pinned-header-audit.md) found that all
252 block-scale tensors at the same official pin are stored BF16, contradicting
the originally required F32 storage. The user has explicitly approved only this
reconciliation: accept those pinned BF16 scales and **promote exactly to F32**
before the unchanged F32 multiplication. A finite BF16 word `b` has F32 bits
`b<<16`; this preserves its stored value, not any hypothetical prior F32 precision.

The name remains `M-N/Qwen3-4B-8prompt-candidate-1`, identified by D31 **plus D32**.
Pins, shapes, rounding/reductions, Q8, head, workload and quality criteria do not
change. The original mismatch remains historical evidence, not an undisclosed
repair. Index `total_size` drift is neither corrected nor waived. Payload
integrity, positive finite scales, head/embedding byte equality, totality/no-wrap
proofs and quality remain unproved. No implementation, payload acquisition or
heavy execution is authorized by this amendment; all other gates remain.

## Facts, choices, and evidence: three different things

| Category | Exact boundary |
| --- | --- |
| Governing source facts | arXiv **2609.10264v1**, PDF SHA-256 `c9570fdeb51a26f9d707bcc7bf485f2001e7d3e15ebdf34815123935a1e8e2a0`; BabyBear; nearest-integer `256*W`; centered `/2^16` then BF16; local nonlinearities; optimistic request-end verification; original eight-token Qwen3-4B next-logits workload |
| Known official-Qwen facts | Pinned config/index/implementation and bounded headers; 252 stored BF16 128×128 block-scale tensors, E4M3 projections, BF16 table/head/norm declarations, dynamic E4M3FN activation code, Q/K norm before RoPE, no biases, tying declaration; source pins/hashes in proposal §2, not payload verification |
| Missing author evidence | No immutable author repo/evaluated checkpoint/numerical manifest recovered from inspected sources. Author-linked ePrint returned 403. This is **not** proof no artifact exists |
| Proposed numerical choices | Original-family source anchor with unknown author identity; fixed-Q8 activation rule; exact ties/cast path/local arithmetic; explicit BF16 table/head exception; defined quality oracle/thresholds |
| Missing execution evidence | Payload integrity/values/ties and scale positive-finiteness; index-size discrepancy explanation; actual-source honest no-wrap and all-reply finite/i64 proofs; correctness/quality observations; complete deployment admission and security composition |

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
  E4M3FN bit decoding, D32 stored-BF16/exact-F32 scales with unchanged direction/axes,
  BF16 source tables/norms, positive
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

## Reviewed future component demand: Sparse project-local prefixes

This appendix records reviewed source-derived demand equations for a future
reusable component, not a fee schedule or a whole-method completeness claim.
The typed definitions adopted on 4 November 2026 (4903) are definitions, not
numeric rates, payment proofs, funding grants, or execution authorization. The
user's 7 November 2026 administrative release removed source/test authoring,
funding-first sequencing and Native holds; actual first-advance funding and
issuer evidence remain missing. The 32-fresh-u32 cap is now authored in
`crates/pllm-core/src/maverick_batch/arithmetic.rs::below`: 32 rejections return
`Error::Randomness`; an underlying error propagates immediately. No allocation,
retry or reset is added, and `word_from_fill` zeroization is preserved.
Independent static review passed. Four unit and two integration regressions are
authored; the reference test binary includes the four unit tests, giving six
registered tests under `bounded_draw`. Tests have not been executed; the current
Rust wheel is not qualified. Bounded calls do not bound entropy-provider time;
fresh uniform error-free RNG premises remain unchanged. Standing Native
fresh-worker, exclusive-slot, immutable-input and exact-command protocols and
numeric caps remain in force, without reinstating administrative holds.

The project-local source anchors are
`crates/pllm-core/src/maverick_batch/code.rs:179–223`, with shared owners in
`memory.rs:259–302` and RNG rejection work in `arithmetic.rs::below`. The
equations deliberately reuse allocation, view, indexing, iteration, swap, RNG,
and destruction helpers rather than introducing Sparse-specific parallel
machinery. They describe reached source operations, not CPU instructions,
physical stores, deployed qualification, or RSS.

### Coordinates and ownership

Let `n` be the requested length, `w` the requested weight, and `FIELD` the
existing field modulus. `C` denotes one scalar comparison and `B` one reached
conditional decision. These are typed demand coordinates, not prices. A caller
`D` consists of a `Result` tag observation, its decision, and the reached unwrap
or propagation transfer; it is not a generic call fee.

Named helpers `Z`, `Copy`, `Views`, `Index`, `Swap`, `Iter`, and `Drop` own their
respective bodies exactly once. `Z(s)` denotes the shared zeroed `Words` helper
on extent `s`; `Copy(s)` denotes the shared copy helper, including its owned
zero-allocation work. Constructor field slots own their initialization writes;
do not add duplicate SSA-assignment writes or synthetic moves. Caller tables
below are additive to helper bodies, except where an operation is expressly
assigned to a helper. A helper's internal bounds checks are not duplicated at
the caller.

For a short-circuit OR header with `m` predicates, rejection at predicate
`r` (one-based) has demand

```text
HeaderReject(m, r) = r C + [min(r, m - 1) + 1] B.
HeaderAccept(m)    = m C + m B.
```

Error variant and enclosing `Err` construction are separate reached typed
constructors, not part of `C` or `B`.

### `Sparse::sample`: allocation and reached loop prefixes

The three-predicate header at `code.rs:185–187` rejects with `C + 2B`,
`2C + 3B`, or `3C + 3B`; a valid header has `3C + 3B`. Each rejection constructs
`Shape` and `Err` and performs no subsequent allocation.

After a valid header, the sequential setup is:

1. `Z(n) + D` for the pool `P`.
2. One mutable pool view, `iter_mut` and `enumerate` constructors, `n` iterator
   advances and one terminal advance, `n` `usize`-to-`u32` casts, and `n` `u32`
   pool-stamp writes.
3. One multiplication for `2*w`, then `Z(2w) + D` for the pairs buffer `T`.
4. The range constructor for `0..w`.

Each allocation refusal stops this sequence at that point. Cleanup local to a
failed helper precedes caller propagation and any cleanup of earlier owners.
`Z` owns the initial `n` or `2w` zero writes; pool stamps are distinct later
writes. There are exactly two `Words` buffers, `P(n)` and `T(2w)`: this path
does not call `from_pairs`, perform an extra validation pass, or make a final
copy.

Let `f` count attempted first `below` calls, `a` successful first calls, and `b`
successful coefficient calls. For zero-based failing iteration `i`, the reached
prefixes are:

| Outcome | `(f, a, b)` |
| --- | --- |
| First draw fails | `(i + 1, i, i)` |
| Coefficient draw fails | `(i + 1, i + 1, i)` |
| All iterations succeed | `(w, w, w)` |

For these prefixes, the caller-local loop demand is:

| Coordinate | Reached demand |
| --- | --- |
| Range advances | `f`; one additional terminal advance only on success |
| Subtractions | `f` evaluations of `n-i`, plus `a` evaluations of `FIELD-1` |
| Casts | `f` `usize`-to-`u32`, plus `a` `u32`-to-`usize` |
| Additions | `a` for `j`, `b` for coefficient increment, `b` for coefficient index `+1` |
| Multiplications | `a+b` for pair offsets |
| Caller result handling | `(f+a) D` |
| Views | `a` mutable `P`, `a` shared `P`, `(a+b)` mutable `T` |
| Helpers | `a Swap`, `a Index(P, read)`, `(a+b) Index(T, write)` |
| Additional payload accesses | `a` `u32` reads, `(a+b)` `u32` writes |

`Swap` owns two payload reads, two payload writes, and two bounds
comparison/decision guards, including when its indices are equal. Do not add a
second exchange at the caller. If the coefficient RHS fails, its LHS offset,
index, mutable view, and write have not been reached.

The RNG demands `G_s` and `G_e`, for the first and coefficient draws
respectively, separately own all reached `below` work, random-word work,
rejection attempts, scratch handling, and internal dispatch. The `(f+a)D`
caller term does not replace or duplicate those bodies. **HISTORICAL SOURCE
PREMISES:** the pre-cap derivation assumed no finite rejection bound. Current
source caps calls at 32; that fact alone supplies no complete priced RNG vector.

Success constructs `Sparse` and `Ok`, moves `T` into the returned owner, and
drops `P`. A loop failure drops `T` then `P`, after the failed callee's local
cleanup. Retirement of a successfully returned `T` belongs to a separate future
lifecycle extent, not to early-return cleanup or to the success prefix here.

### `Sparse::from_pairs`: separate borrowed-input prefixes

This is a separate constructor, not a phase of sampling. Its four-predicate
header at `code.rs:208–211` follows `HeaderReject(4,r)` above; the length
observation and multiplication `2*w` occur only if predicate four is reached.
A valid header has `4C + 4B`. A rejected header constructs `Shape` and `Err`.

After header acceptance, there are 1,024 Boolean initialization writes for
`seen` (not a heap allocation), followed by the `chunks_exact(2)` constructor.
Let `v` be the number of visited pairs, `r` the number whose position is in
range, `z` the number also unseen, `t` the number also nonzero, and `k` the
number fully valid. The validation-prefix demand is:

| Coordinate | Reached demand |
| --- | --- |
| Iterator advances | `v`; one terminal advance only on complete validation |
| Position casts | `v` `u32`-to-`usize` |
| Scalar comparisons | `(v+z+t) C` |
| Conditional decisions | `(2v+r+z) B` |
| Pair indexing | `v+z+t` helper invocations |
| Pair payload reads | `v+z+t` `u32` reads |
| Seen indexing | `r+k` helper invocations |
| Seen payload reads | `r` Boolean reads |
| Seen payload writes | `k` Boolean writes |

For `k` completed valid pairs before the failing pair, the exact tuples are:

| Failure | `(v, r, z, t, k)` |
| --- | --- |
| Position out of range | `(k+1, k, k, k, k)` |
| Duplicate position | `(k+1, k+1, k, k, k)` |
| Zero coefficient | `(k+1, k+1, k+1, k, k)` |
| Coefficient outside the field | `(k+1, k+1, k+1, k+1, k)` |

The repeated `pair[1]` accesses are real repeated indexing and reads. In
particular, neither `(v+r+z+t) B` nor an early-rejection `r(C+B)` term is the
correct decision count. Validation failures construct `Field` and `Err`.
Complete validation has `v=r=z=t=k=w`, followed by `Copy(2w) + D`. `Sparse`
and `Ok` are constructed only if that copy succeeds. A copy error propagates
after helper-local cleanup; the borrowed input remains externally owned on
every branch.

### Conditional wipe and release helper corrections

These helper equations are conditional source analysis for Rust **1.85.1** and
zeroize **1.9.0**: release configuration, no test hooks, no UB-debug checks,
non-TSan, and a supported non-Miri native-assembly barrier branch. They are not
qualification of a deployed binary or an RSS bound. Other configurations
require their own reached-body accounting.

Use `O` for typed metadata observations, `U` for typed updates, `A` for
arithmetic, and `K` for typed constructor field initialization. Subscripts name
the observed field, update, or constructor; `K` charges its actual field slots,
not an extra assignment to every SSA name. Let `T(k)` contain traversal
construction, `k` advances, and the terminal advance exactly once. Here `T(k)`
is a traversal helper, distinct from the pairs-buffer name `T` above. `Q`
denotes typed variant dispatch: tag observation, decision, and reached payload
extraction, without synthetic moves. `W_word` and `W_byte` denote typed payload
writes.

For live length `l` and retained capacity `c`, the conditional wipe demand is:

```text
W(l,c) = View(l)
       + T(l) + l (Default_u32 + W_word + Barrier)
       + Clear + Spare + ByteWipe(c).

View(l) = O_len + K_slice(pointer, length).
Clear   = O_len + K_slice(pointer, length) + U_len0.
Spare   = 2 O_len + Capacity + A_sub + K_spareSlice(pointer, length).
Capacity = C(elem_size == 0) + B + O_cap                 [u32 branch].

ByteWipe(c) = O_spareLen
            + CheckedMul(sizeof(u32)) + Q_expect
            + CheckedIsizeConversion + ResultIsOk + B_assert
            + MaybeUninitZero_u8
            + T(4c) + 4c W_byte + Barrier.
```

There is no extra `T::IS_ZST` comparison in `Capacity`. `Default_u32`, checked
multiplication and conversion, and result inspection are named elementary
typed coordinates; these equations do not claim an unproved fully elementary
expansion of them or introduce flat compound fees. The word pass has `l`
barriers, while the byte pass has one final barrier even when `c=0`. The native
assembly barrier is an optimization observation, not an atomic or compiler
fence. The fallback `black_box`, size check, and conditional volatile read are
separate work in an excluded configuration, not a fence hidden in this branch.

The corresponding release equation is:

```text
Release(c) = O_len + K_emptyDropView(pointer, length)
           + CurrentMemory(c) + Q_Option
           + [c > 0] AllocatorDeallocate.

CurrentMemory(0) = O_elemSize + O_cap + 2C + 2B
                 + K_None(no payload).

CurrentMemory(c > 0) = 2 O_elemSize + 2 O_cap + O_elemAlign
                     + 2C + 2B + A_uncheckedMul
                     + K_Layout(size, align)
                     + K_tuple(pointer, layout) + K_Some(tuple).
```

`Q_Option` occurs on both branches. Capacity zero skips deallocation, not all
release-helper work. Release does not publish a capacity update. Private
pointer, ZST, and temporary mechanics are included in their related typed
operations; the allocator's private body remains behind the adopted boundary.
These equations describe the vector wipe/release portion of `Drop`; they do
not silently replace reservation release or other owner work in the shared
`Words` destruction composition.

The pinned helper source anchors are Rust 1.85.1 `alloc/src/vec/mod.rs`,
`as_mut_slice` lines 1592–1607, `clear` 2654–2665, `spare_capacity_mut`
2878–2887, and `Drop` 3775–3781; `alloc/src/raw_vec.rs` lines 520–533; and
zeroize 1.9.0 `src/lib.rs` lines 274–277, 358–362, 435–447, 528–536,
748–765, together with `src/barrier.rs`.

The shared `Words::zeros` capacity check gives a source-only capacity lemma:
on success `c=n`, while allocator refusal leaves `c=0`. Thus a successfully
initialized extent uses `W(n,n) + Release(n)`; a generic unexpected-capacity
mismatch before initialization uses `W(0,c) + Release(c)`; and allocation-error
cleanup uses `W(0,0) + Release(0)`. Failures before vector construction must not
be charged as though that owner already existed. The payload counts are `l`
word writes plus `4c` byte writes in distinct passes, not an assertion about
physical stores, CPU work, or RSS. A returned pairs buffer's retained capacity
and eventual retirement remain a future lifecycle obligation outside
early-return cleanup.

### Remaining boundaries and review

Full helper expansion and component composition, original-source coverage, numeric rates, full admission,
and funding remain separate unfinished obligations. These reusable planning
equations require no receipt collections, copied source trees, evidence
archives, or custom archive/replay verifiers. These equations do not themselves
authorize source changes or launches, raise caps, establish security or claim
whole-method completeness. Source/test, funding-first sequencing and Native
administrative holds were separately released on 7 November 2026; standing
Native protocols remain. The existing 1566 setup/hash erratum elsewhere is
unchanged. Independent Astra review remains scoped to the stated derivations.

### Sampling-component composition (success only)

This composition uses the independently reviewed appendix above and the audited
call graph under `crates/pllm-core/src/maverick_batch/`. The constants in
`mod.rs:28–37` are batch size 2, length 512, secondary length 1,024, privacy
weight 165, check weight 31, two check repetitions, and maximum 16 ordinals.
An ordinary batch completes six `Sparse::sample` calls: two privacy samples
(`protocol.rs:400–423`), two primary checks (`476–558`, `682–685`), and two
secondary checks (`580–596`, `682–685`). The check factor two is repetitions,
not equation pairs; a call site and its callee body are counted only once.

One request issues two ordinals/batches (`request.rs:74–88`, `126–131`,
`166–169`), giving four each of `S(512,165)`, `S(512,31)`, and `S(1024,31)`:
12 completed samples, 24 successful sampler-buffer allocations, `Σn=8,192`,
and `Σw=908`. Each completed sample makes `2w` successful `below` calls: 1,816
per request, or 7,264 for four complete requests using eight issued ordinals
if no intervening burns occur. The authored cap of 32 fresh
`u32` draws per `below` call gives at most 58,112 draws per request and 232,448
for four. These bound source-level draw calls, including rejecting coefficient
bound `FIELD-1`, not entropy-provider time or complete funded work.
Errors abort the remaining suffix; reached calls are not completed calls, and
there is no implicit retry, refund, or new ordinal.

The separate setup count remains 1,566: IMPORT 23 + VALIDATE 1,539 + SPLIT 4.
It excludes requests and their issued batches and is not a whole-allocation
total. Split (`protocol.rs:250–252`) makes one 32-byte entropy fill into a plain
`SESSION` array, not a `Zeroizing` owner or eight `below` calls. `OsRandom::word`
uses four bytes (`arithmetic.rs:41–52`). Neither ordinal nor byte counts pay work.

Actual stage order constrains composition. Ordinary issue samples both privacy
vectors before either claim. A request reserves both ordinals before validating
all four inputs, then reaches the quote plan/control 4,096 reservation and
interleaves privacy sampling with claims. Earlier claim syndromes, buffers,
corrections, and copies precede the next sample (`protocol.rs:342–369`). Primary
request processing validates, copies, and encodes all four replies before the
first pair; that pair and its disclosure copies precede the second pair
(`request.rs:118–131`). Secondary processing validates and copies both replies
before the first pair, completes that pair before the second, and completes both
before deterministic equation checks (`158–175`). These deterministic steps are
excluded from sampling demand, not priced at zero or proved funded.

The following payload coordinates use the selected source-only `capacity=length`
lemma. They retain the conservative two-read/two-write convention for swaps
even when indices coincide; they do not infer physical stores.

| Payload coordinate | Per completed `S(n,w)` | Per complete request |
| --- | ---: | ---: |
| Initial pool/pairs `u32` writes | `n+2w` | 10,008 |
| Pool-stamp `u32` writes | `n` | 8,192 |
| Swap-data `u32` reads | `2w` | 1,816 |
| Swap-data `u32` writes | `2w` | 1,816 |
| Pool-index `u32` reads | `w` | 908 |
| Pair `u32` writes | `2w` | 1,816 |
| Pool-drop word writes | `n` | 8,192 |
| Pool-drop capacity-byte writes | `4n` | 32,768 |
| Future pairs-drop word writes | `2w` | 1,816 |
| Future pairs-drop capacity-byte writes | `8w` | 7,264 |

Summing helper-return extents gives 2,724 data `u32` reads, 30,024 `u32`
writes, and 32,768 capacity-byte writes. This is not one chronological instant
or a peak: it excludes all returned-pairs retirement even if a caller actually
drops some pairs before later samples. Including each eventual pairs retirement
once gives 2,724 reads, 31,840 word writes, and 40,032 capacity-byte writes;
per sample, the full owned lifecycle is `3w` reads, `3n+8w` word writes, and
`4(n+2w)` capacity-byte writes. Optional positive-resize zero-value clone scalar
reads are `n+2w-2` per sample, or 9,984 per request, separate from secret-data
reads. These helper coordinates do not infer private optimizer physical reads.
Metadata, control, typed constructors, elementary coordinates, RNG scratch,
provider traffic, and release-operation vectors are not included in these totals.

```text
S_return_success(n,w) = L_success + Z(n) + Z(2w)
                      + Σ G_success + Drop(P).
S_lifecycle_success(n,w) = S_return_success(n,w) + eventual Drop(T).
Request_sampling_only = 4 S(512,165) + 4 S(512,31) + 4 S(1024,31).
```

`L_success` owns caller result handling `D`; `G` owns `below`, random-word
internal dispatch, and scratch work. Do not duplicate these bodies or count a
drop twice. A once-only teardown contract does not prove authorization for
future returned-pairs work. A literal success vector does not bound error-only
constructors or dispatch. An all-outcome component upper bound must join
reachable prefixes (success, allocation/provider errors, and now-authored
32-attempt exhaustion), or take `sup_prefix λ·V(prefix)` for nonnegative,
unadopted `λ`. Label that result **BOUND**, not an exact trace. **HISTORICAL
SOURCE PREMISES:** before the cap, reachable coefficient `G_max` was unbounded.
Current source bounds draw calls at 32, but complete bookkeeping, provider-call,
cleanup and all-outcome funding composition remain unproved; no finite complete
funded `W` follows from the cap alone. No padding, generic call, or SSA-assignment
fees are invented.

This component excludes `from_pairs`, retained-pairs copies and disclosures,
dense syndromes, outputs, FFI/native entry, whole-caller/control work, and
equations. A borrow does not cover new backing. Original Root debit linkage,
coverage, rates, issuer first-advance authorization, authorized retirement
custody, and full WVT remain unproved. The 7 November 2026 administrative
release removed authoring, funding-first sequencing and Native holds, not the
missing first-advance evidence, security obligations or numeric caps. This source
profile proves neither deployed CPU/RSS nor unique backing, peak memory, or
physical stores. The cap is authored and independently statically reviewed;
tests remain unexecuted and the current Rust wheel unqualified. No copied sources, evidence archives, or
custom verifier are required by this composition.

### Partial request-entry demand inventory

**PARTIAL; independently reviewed source-local inventory, not complete accounting.**
Independent Astra review confirmed this partial inventory against Rust 1.85.1.
Scope: successful ordinal entry (`request.rs:74–76`,
`protocol.rs:308–329`) and complete validation of four valid 64-word inputs
(`request.rs:77–81`, `arithmetic.rs:5–12`, `mod.rs:28–29`), before the first
sample. Other quote/reserve/RNG bodies remain **UNCOMPOSED**; this is neither
full coefficient composition nor a fee schedule.

The adopted 4903 normalization counts helper bodies once and traversal
progress/termination once. Private pointer, ZST, and temporary representation
mechanics belong to named semantic operations: no invented SSA, implicit-copy,
generic-call, or return-store fees. `DResult` owns result-tag observation,
decision, and unwrap/propagation. Private `Try`/`ControlFlow`/`from_residual`
representations add no constructors for that same transfer; a new semantic
error conversion would remain separate.

Both current `Memory.used.fetch_update` and `Client.issued.fetch_update`, with
their current checked-add/filter bodies across **all callback retries**, are
already single adopted atomic invocations. Neither assumes one callback nor
requires callback-body expansion. Surrounding work remains separate.

Four reached semantic `Ok` constructor sites in ordinal entry have payloads:
issued `map_err`: `u32`; live-CAS `map_err`: `bool`; reserve return:
`(Header, Lease)`; request `checked_add(1).ok_or`: `u32`. These are not four
equal writes or a complete constructor inventory. Atomic-helper result
constructors remain owned inside their atomic operations. `Header`'s five
fields, `Lease`'s one field, and the tuple's two fields are disjoint actual
initialized fields, not interchangeable constructor costs. Request line 76
eagerly constructs `Error::Attempts` even on `Some`; the two `map_err` error
closures are unreached on success. `second = first` is an implicit `Copy`,
not a new fee. Other reached helpers and checked arithmetic remain uncomposed.

For the complete valid-input prefix, project-owned subtotal is 256 payload
`u32` reads, four length observations, 260 scalar comparisons `C`, eight
source branches `B` (four shape `if`s plus four `any` `if`s), four `Ok(())`
constructors, and four caller `DResult`s. The 256 predicate-result decisions
belong to `Any_s`, not those eight branches; do not charge predicates twice.

```text
Traversal = 3 Ia + 6 Sa + 3 Ta + 3 Da(0) + 4 Is + 4 Any_s(64,false).
Any_s(64,false) = 64 Ss + Ts + 64 B_predicate + false Boolean return.
```

`Ia` constructs an array `IntoIterator` (one outer, two inner); `Sa`/`Ta`
own successful/terminal array advancement and consumer `Option` dispatch once.
`Da(0)` retires an exhausted array iterator: no backing allocation, wipe, or
destruction of referenced data; it is not `DResult`. `Is` constructs a slice
iterator; **`Any_s` excludes `Is`** and owns `next`, its consumer `Option`
dispatch, predicate-result decisions, and Boolean return. Pinned
[Rust 1.85.1 slice iterator macros](https://github.com/rust-lang/rust/blob/1.85.1/library/core/src/slice/iter/macros.rs)
implement `any` with `while let Some(x) = self.next()`, not default `try_fold`.
Thus four calls expand to 256 `Ss`, four `Ts`, 256 `B_predicate`, and four
false Boolean returns; `Ss`/`Ts` already include consumer `Option` dispatch.
Boolean returns invent no return fee. Project predicate's 256 reads plus
256 comparisons remain separate. Nonsuccess prefixes are not this vector;
even field failure on the last element has no terminal advance.

Remaining obligations include checked arithmetic, other reached helpers,
unwind/all-outcome composition, authentic Original Root first-advance and
debit/retirement coverage, and numeric rates. Current `below` has the authored
32-draw cap; this inventory does not close its all-outcome work composition.
The 7 November 2026 administrative release removed source/test, funding-first
sequencing and Native holds. This inventory proves no test execution, qualified
runtime, complete funding or full-method bound.

### Shared Words ownership composition (ordinary outcomes)

**SOURCE-LOCAL; conditional ordinary-`Result` composition; review pending.**
This design-only composition uses the supplied source-current `memory.rs:230–303`
preimage and the helper equations above. Rust **1.85.1**, zeroize **1.9.0**,
release, no `cfg(test)` hooks, no UB-debug checks, non-TSan, and the qualified
non-Miri native-assembly barrier branch remain conditions, not deployment proof.
It closes multiplicities and ownership for `Z`, `Copy`, and destruction only
subject to independent Astra readback; it does not expand all standard bodies.

#### Typed ownership and reservation prefixes
`DResult` retains its adopted tag/decision/unwrap-or-propagation meaning.
`K_X` means the actual typed constructor and its field slots, not a flat fee;
payload types and reached variants matter. Named helpers own internal dispatch
and constructors exactly once. `MapErr` owns its dispatch and new `Result`
construction; its failure closure alone constructs `Error::Allocation`.
Atomic result constructors remain inside the adopted atomic operation.
Propagation adds no private `Try`, `ControlFlow`, residual, or synthetic move.

For `b(n)=checked(4*n+128)`, define byte-calculation prefixes:

```text
Calc_mul = CheckedMul(4) + OptionAndThen(no add)
         + K_ErrorAllocation + OptionOkOr(Err) + DResult_calc.
Calc_add = CheckedMul(4) + OptionAndThen(CheckedAdd(128))
         + K_ErrorAllocation + OptionOkOr(Err) + DResult_calc.
Calc_ok  = CheckedMul(4) + OptionAndThen(CheckedAdd(128))
         + K_ErrorAllocation + OptionOkOr(Ok_usize) + DResult_calc.
```

`OptionAndThen` owns input `Option` dispatch: on `None` it constructs `None`;
on `Some` it invokes `CheckedAdd` once and forwards its `Option` without a new
constructor. `OptionOkOr` consumes `Option` dispatch and owns its new typed
`Result` constructor. The separately owned `Error::Allocation` argument is
eager on every path, including success. Each row has one `and_then`, one
`ok_or`, and one caller `DResult_calc`; checked arithmetic has no additional
explicit project-level `C`/`B`. Standard bodies retain their named coordinates.

The reservation helper's complete project-local ordinary prefixes are:

```text
R_denied(b) = UsedFetchUpdate(b) + MapErr_reserve(Err_Allocation)
            + DResult_reserve_internal.
R_ok(b) = UsedFetchUpdate(b) + MapErr_reserve(Ok_usize)
        + DResult_reserve_internal + A_add(previous,b)
        + PeakFetchMax(previous+b)
        + MemoryClone + K_Reservation(memory,bytes) + K_OkReservation.
MemoryClone = ArcStrongClone(State) + K_Memory(inner).
```

`UsedFetchUpdate` includes checked-add/filter and **all callback retries** in
one adopted atomic invocation. Denial includes atomic-used overflow or the cap
condition, not only a provider refusal. `R` includes its own internal
`DResult`; `zeros` additionally owns `DResult_zeros_reserve` below. No new Arc
backing or weak increment occurs: exactly one strong clone on reserve success.
`Reservation` initializes two fields; wrapper construction is helper-owned.
Successful `UsedFetchUpdate` establishes the meter charge. Reservation ownership
begins at its constructor after peak update and Memory clone, before `Ok` return;
successful return transfers it into the `zeros` local. Refusal returns no Reservation.

#### Six zero-allocation outcomes
Let `N = VecNew + ZeroizingNew`, each owning its internal constructors. Define
the reached allocation prefixes, not additional compound charge coordinates:

```text
P_R = Calc_ok + R_denied(b) + DResult_zeros_reserve.
P_A = Calc_ok + R_ok(b) + DResult_zeros_reserve + N.
T_err = TryReserveExact(n,Err) + MapErr_alloc(Err_Allocation)
      + DResult_zeros_alloc.
T_ok  = TryReserveExact(n,Ok_unit) + MapErr_alloc(Ok_unit)
      + DResult_zeros_alloc.
H(c,n) = Capacity + C_neq(c,n) + B_if.
```

`MapErr_alloc` owns the mapped result constructor; its `Allocation` error
constructor occurs only on failure. `TryReserveExact` retains its own internal
result constructors. `Capacity` is the existing `C(elem_size==0)+B+O_cap`
for `u32`, so `H` has **two `C` and two `B`** outside other opaque bodies and
dispatch. `H` occurs only after allocation success, including mismatch.

| Ordinary outcome | Complete reached composition | Owned state at exit |
| --- | --- | --- |
| Checked multiply overflow | `Calc_mul` | No vector or reservation |
| Checked add overflow | `Calc_add` | No vector or reservation |
| Reservation denied | `P_R` | No vector or reservation |
| `try_reserve_exact` error | `P_A + T_err + DZ(0,0) + DR(b)` | All local owners retired |
| Capacity mismatch | `P_A + T_ok + H(c,n) + K_ErrorAllocation + K_ErrWords + DZ(0,c) + DR(b)` | `c != n`, length zero; all local owners retired |
| Success | `P_A + T_ok + H(n,n) + ZerosL(n) + AllocationsFetchAdd(1) + K_Words(data,reservation) + K_OkWords` | Returned length/capacity `n,n`, reservation `b` |

`Z_b(n)` names exactly its selected row; `Z_ok(n)` names the success row.
Under the conditional helper lemma, allocation error leaves length/capacity
`0,0`; mismatch leaves length zero and actual `c != n`, not necessarily `c>n`.
Success alone establishes `c=n` and, after resize, `l=n`. `ZerosL(n)` owns
`resize(n,0)`, its internal work, and the single initialization pass of `n`
word writes: no second resize-payload/zero-pass charge. Reservation denial
reaches three `DResult` sites total; allocation error, mismatch, and success
reach four. Those sites are typed dispatch/propagation, not equal write fees.
At `n=0`, the success path still reserves 128, clones Memory, checks capacity,
resizes, and increments the counter; that increment does not establish that a
provider call occurred.

#### Destruction and borrowed-source copy
```text
DZ(l,c)   = W(l,c) + Release(c).
DR(b)     = UsedFetchSub(b,AcqRel) + ArcDrop_State.
DW(l,c,b) = DZ(l,c) + DR(b).
```

`W` and `Release` are the already printed conditional equations, not new fees.
`Words` has no manual `Drop`: declared `data: Zeroizing<Vec<u32>>` precedes
`_reservation: Reservation`. Automatic field destruction therefore runs `DZ`
then `DR`; error locals also retire data then reservation, reverse construction.
`DR` includes Reservation's explicit used decrement followed by automatic
Memory/Arc destruction exactly once. There is no extra Words Arc drop or
second Zeroizing wipe. Error-path Arc drops are non-last while borrowed Memory
lives; eventual returned-owner Arc drop may be last, retaining that branch in
the standard `ArcDrop` coordinate. Source State fields are plain.
Success performs no immediate `DW`; eventual retirement is `DW(n,n,b)` once.
One `W(l,c)` has `l` word writes and `4c` byte writes in overlapping but distinct
passes, with `l+1` barriers, including the byte barrier at zero capacity.
Zero capacity skips actual deallocation, not helper work. Subtracting `b` is
neither work refund nor allocation-counter decrement: MemoryByteMeter and
WorkFunding remain distinct.

For an already borrowed source of length `n`, `L_source` is its one length
observation before `Z`; `Copy` calls `zeros` once:
```text
Copy_err(n) = L_source + Z_b(n) + DResult_copy_from_Zerr.
Copy_ok(n)  = L_source + Z_ok(n) + DResult_copy_from_Zsuccess
            + V_mut + S_std(n,n) + K_OkWords.
```

In `Copy_err`, `Z_b` ranges only over the five failure rows and owns all
callee-local failure cleanup. `V_mut` owns the actual typed Zeroizing/Vec
mutable-view body once. `S_std(n,n)` denotes exactly one reached standard
`copy_from_slice` body, including entry/length control and its `n` payload reads
and `n` payload writes once, excluding separately owned mutable-view construction.
Those payload accesses are inside the body, not additive charges; initial zero
writes remain distinct. This notation records source multiplicity and ownership
under Coordinates and ownership above, not a new flat compound tariff,
invocation fee, or accounting boundary. Its elementary vector remains unexpanded.
Equal lengths and valid shared source plus fresh owned target exclude ordinary
mismatch error/aliasing; no unwind claim follows. No automatic input `as_slice`
charge exists unless the actual caller reaches it. Final `Ok` is its own typed
constructor, not assignment or copying a result; caller handling and returned
owner retirement remain outside `Copy_ok`.

Residual standard coordinates include CheckedMul/Add, Option dispatch bodies,
map_err bodies, Vec/Zeroizing construction, try-reserve/resize, views/coercions,
slice-copy, and ArcDrop: multiplicities closed, elementary vectors unexpanded.
No rates or new compound flat fees are adopted. Fully elementary expansion,
unwind/abort, capped `below` all-outcome composition, finite Maverick `W`, Original Root
first-advance, coverage, full admission, retirement prepay, funding, and security
remain open. The 7 November 2026 administrative release removed source/test,
funding-first sequencing and Native holds. Standing Native fresh-worker,
exclusive-slot, immutable-input and exact-command protocols and numeric caps
remain; no whole-method claim, receipt archive or custom verifier follows.
Conditional ordinary-outcome completion requires actual parent Astra readback.
Independently reviewed against the current source and adopted helper normalization. Conditional source-local ordinary-`Result` composition closes shared-owner multiplicities and cleanup ownership; elementary standard-body expansion and broader method, funding, and deployment obligations remain open.

### U1–U8: partial conditional standard-body refinements

**DESIGN ONLY; independently reviewed conditional source-body refinements.**
Two scoped Astra readbacks covered U1/U2/U3/U8 and U4/U5/U6/U7,
including shared normalization and ownership compatibility. This review
does not establish fully elementary or whole-method closure.
Conditions remain exactly lines 623–629: Rust 1.85.1, zeroize 1.9.0, release,
no `cfg(test)` hooks, no UB-debug checks, non-TSan, qualified non-Miri native
barrier. This is partial helper refinement, not fully elementary closure.
Existing `Words`, `Z`, `Copy`, `DZ`, `DR`, and `DW` equations stay unchanged.
Below are replacements inside their owned helpers, never additions beside them.
Use existing typed `O/C/B/K/A` coordinates and dispatch ownership. Source entries
and forwarding edges are structural annotations, not charged generic calls.
Unexpanded bodies remain outcome-indexed residual subvectors, not flat fees;
no SSA writes, synthetic moves, or new accounting boundaries are introduced.
Under adopted 4903 normalization, reached semantic metadata observations/updates,
comparisons, decisions, typed constructions, ownership effects, atomics, fences,
and deallocation remain distinct. Private pointer/ZST/temporary representation
mechanics stay owned inside their named semantic operation. Structural arrows
below expose unexpanded source bodies, not new invocation/fee coordinates;
describing a helper adopts no tariff boundary or duplicate payload traffic.

#### U1 — checked arithmetic and byte-extent conversion

For `usize`, `overflowing_mul` casts both operands to `ActualT`, reaches opaque
`I_mul_with_overflow`, structurally destructures its intrinsic-owned result,
casts the product back to `usize`, and constructs one explicit `(usize,bool)`
tuple. That wrapper tuple is distinct from the intrinsic's own result; tuple
destructuring adds no SSA-copy fee. `ActualT=u_w` follows target pointer width
in `core/src/num/mod.rs`; these source casts imply no physical instructions.
`CheckedMul` then reaches `Unlikely(flag)` and branches on its returned Boolean,
constructing only selected `None` or `Some(product)`. Keep `Unlikely` as a
residual body owning its own branch, Boolean return, and overflow `cold_path`
intrinsic; replacing it by the caller's branch alone would omit reached work.
`CheckedAdd` instead reaches opaque `I_add_with_overflow`, observes result `.1`,
reaches `Unlikely(flag)`, and branches: overflow constructs `None`; success
reaches opaque `I_unchecked_add` and constructs `Some(sum)`. No `overflowing_add`
wrapper is reached. Arithmetic intrinsics remain unexpanded subvectors.

`isize::try_from(z)` casts `isize::MAX` to `usize`, compares `z > bound`, and
branches. Success casts the value and constructs `Ok(isize)`; failure constructs
`TryFromIntError(())` and `Err`, with typed unit payload, not a flat unit fee.
Overflow in `zeros`' multiply stops before `and_then`'s `Some`-branch addition.
Under the conditional valid-Vec bound `4c <= isize::MAX`, ByteWipe's multiply
and conversion succeed, including `c=0`; this is not panic/unwind coverage.

#### U2 — substitution of Option/Result bodies

These replace existing named dispatch bodies, never add dispatch beside them:
`and_then(None) = Q_None + K_None`; `and_then(Some)` owns `Q_Some`, reaches
`CheckedAdd`, and forwards its existing `Option` without another constructor.
`ok_or` owns selected `Q` and a fresh `Ok(usize)` or `Err(Error)` wrapping its
already eager `Error::Allocation` argument. Do not construct that error again.
The reservation `MapErr` success payload is `usize`; allocation `MapErr` success
payload is unit. Each owns `Q_Result` plus a fresh mapped `Result` on **both**
branches; only its error closure constructs new `Error::Allocation`.

`expect(Some)` owns `Q` and forwards existing payload, no new constructor;
`expect(None)` owns `Q` and residual `expect_failed`/panic body. This substitutes
for `Q_expect`, not another `Q` beside it. `is_ok` borrows
`Result<isize,TryFromIntError>`: tag observation and variant decision produce
a Boolean without payload extraction or new `Result`. This replaces
`ResultIsOk`. Release's current-memory `Q_Option` is a separate reached site,
as is each caller `DResult`; none creates private Try/residual/move fees.

#### U3 — empty Vec and Zeroizing construction

Inside `N = VecNew + ZeroizingNew`, annotate this structural reached-body path:

```text
N body: AlignQuery_u32 → Transmute<usize,Unique<u8>> [private, unexpanded]
      → K_RawVecInner(ptr,Cap::ZERO,alloc) → K_RawVec(inner,marker)
      → K_Vec(buf,len=0) → K_Zeroizing(vec).
```

`RawVecInner.cap` receives associated constant `Cap::ZERO`; its constant
initializer does not add a runtime `K_Cap(0)`. Fieldless `Global`/`PhantomData`
values introduce no positive ZST fees. Actual enclosing constructor field
slots are owned once, not flattened and then counted again. Private transmute
and its `Unique` result stay inside `N`: no extra NonNull/marker construction is
inferred. No allocator call or new backing occurs; length/capacity are `0,0`.

#### U4 — `TryReserveExact` from length/capacity `0,0`

`Vec::try_reserve_exact` observes the length field and forwards to `RawVec`;
forwarding itself constructs no new `Result`. The initial `needs_to_grow`
test occurs even for `n=0`. Every success reaches a second `needs_to_grow`,
Boolean negation, and the `assert_unchecked` site before outer `Ok(())`.
Define reached-body notation, not independently chargeable compound coordinates:

```text
Prefix(n) = O_len_field + GrowTest(0,n,0).
Post(n,c) = GrowTest(0,n,c) + BooleanNegation + AssertUncheckedSite.
Reserve_ok(0) = Prefix(0) + Post(0,0) + K_OkUnit_outer.
Reserve_ok(n>0) = Prefix(n) + GrowExactPrefix(n) + Layout_success(n)
                + CurrentMemory(0) + Finish_success(n)
                + Install(ptr,cap=n) + K_OkUnit_growExact
                + Post(n,n) + K_OkUnit_outer.
```

`GrowTest` refines the reached capacity observation and `needs_to_grow`
arithmetic/comparison; its branch decision belongs to its actual consuming
site, once. `GrowExactPrefix` starts with the element-size comparison/branch,
then `checked_add(0,n)`, eager `CapacityOverflow` construction, and `ok_or`.
Checked-add/dispatch follow U1/U2 with their intrinsic residuals and typed
error payload here. A failing add stops before layout; its error argument is eager.
For positive `u32` growth, layout follows `layout_array` → `repeat` →
`pad_to_align` → `repeat_packed`/checked multiplication →
`from_size_alignment` → tuple construction → `map` → `map_err`.
`Layout_success` replaces that whole reached chain, not a fee beside it.
Layout arithmetic, alignment, pointer, and internal error-conversion bodies
remain branch-sensitive residual subvectors, including their owned dispatch.

After layout success, `current_memory` evaluates to `None` **before**
`finish_grow` begins. `CurrentMemory(0)` is the existing reached equation,
not a second allocation check. `Finish` reaches `alloc_guard`, then `Option`
dispatch, then allocation on the `None` branch. Its allocation-error mapping
constructs exactly `TryReserveErrorKind::AllocError { layout, non_exhaustive: () }`,
`TryReserveError { kind }`, and mapped `Err`; typed fields, not equal flat fees.
The existing Global allocation boundary is reached once. Its private body is
inside that boundary, never additive to it. Allocation is not zeroed allocation.
On success, pointer **and** capacity are installed, then `grow_exact` returns
`Ok(())`; forwarding adds no `Result`, and outer post-assert/`Ok(())` follow.
Errors install neither pointer nor capacity and never reach the post-assert.
Each error vector is only its reached prefix: initial test, reached growth and
layout prefix, and, only after layout success, current-memory/finish prefix.
It includes reached typed error constructors/propagation, not success-only
suffixes or a stale whole-layout/whole-finish compound on every error branch.
For `n=0`, no allocator or backing is reached; success retains `0,0`.
Positive success leaves length zero, capacity `n`; ordinary errors retain `0,0`.
Outer `MapErr_alloc`, caller `DResult`, receiver Zeroizing projection, and
cleanup stay with their existing owners, outside this reserve refinement.

#### U5 — `ZerosL(n)`: resize from length zero, capacity `n`

Distinguish `Len_method` from bare existing `O_len`: `Vec::len()` observes
the field, compares `len <= T::MAX_SLICE_LEN`, and reaches `intrinsics::assume`.
Replace a reached method body with this refinement; do not add its comparison
to every old `O_len`, or count its field read twice. Positive resize reaches
`Len_method` twice: initial resize and pointer-offset preparation; zero resize
reaches it once. Intrinsic implementation remains residual, not a CPU claim.

At `n=0`, resize's `new_len > len` comparison is false and reaches `truncate(0)`.
Its `len > self.len` test is false at zero. Truncation observes the length field
twice, for comparison and subtraction, then reaches pointer/add-zero mechanics,
an empty raw mutable slice, length-zero publication, and empty `[u32]` drop.
Those are reached operations despite no payload initialization or allocation.

For `n>0`, resize reaches `extend_with` and always calls `reserve(n)`.
Adequate capacity gives exactly one false `needs_to_grow` test: no allocation,
pointer installation, or capacity update. The pointer-offset preparation uses
the second `Len_method`. `SetLenOnDrop` initializes two fields (length reference
and local length); no extra assignment fees accompany those constructor slots.
`T(n-1)` owns traversal construction, advances, and termination exactly once.
Each of its `n-1` iterations clones scalar `u32` via `*self`, rereading the
zero argument once, then initializes one payload word. The final `n>0` element
is written without cloning. There are `n` local guard-length increments;
guard destruction publishes vector length once, not once per payload write.
Initialization, local increments, and final publication stay distinct even
when traversal is empty (`n=1`); zero resize's truncate publication remains above.
Thus `ZerosL(n>0)` owns `n` initialization writes and `n-1` scalar-zero reads,
not source-secret reads. No zeroed-allocation specialization or second payload
zero pass is inferred. Pointer, intrinsic, and private helper implementations
remain residual subvectors inside resize, with traversal and writes owned once.

#### U6 — typed projections and slice views

Zeroizing `Deref`/`DerefMut` project/borrow their field; no adopted extra event
is attached to that projection. Reserve, capacity, and resize stop at `Vec`:
none implies a `Vec`-to-slice coercion or slice-view charge.
The actual `Words::as_mut` path is Zeroizing `DerefMut` → Vec `DerefMut` →
`as_mut_slice`; it supplies exactly one existing mutable view:

```text
V_mut(l) = existing View(l) = O_len + K_mutSlice(pointer,length).
V_shared(l) = O_len + K_sharedSlice(pointer,length).
```

Shared and mutable slice constructors are typed separately, not interchangeable
field fees. This refines the existing view owner; no extra `AsMut` wrapper is
reached. `Copy`'s source is already borrowed: no additional input `as_slice`
or shared-view constructor is introduced merely to explain that borrow.

#### U7 — equal-length `S_std(n,n)`

Reached `copy_from_slice` has three length observations, one `!=` comparison
and its false mismatch branch, shared/mutable pointer extractions, and one
`copy_nonoverlapping` wrapper with its inner intrinsic reached. These entry
annotations are structural, not generic invocation fees. Even `n=0` reaches
them, with zero payload accesses. Under the stated disabled-UB-check condition,
the UB-check gate is false: predicate arguments and predicate body are unreached.
The inherited logical projection is `n` reads and `n` writes **inside** existing
`S_std`, never an added term outside it or a newly adopted backend leaf.
Private backend/intrinsic implementation remains unresolved; no CPU, physical
read, or physical-store count follows. Mutable view, initial zero writes,
final `OkWords`, caller handling, and owner retirement remain outside `S_std`.

#### U8 — partial Arc clone and retirement bodies

Memory wrapper construction and meter atomics remain with their existing owners.
For `Arc<State,Global>`, annotate structural bodies inside existing Arc owners;
private labels below remain unexpanded there, not independent fee coordinates:

```text
I body: O_Arc.ptr → NonNullAsRef [private].
Clone_ok body: I → StrongFetchAdd(1,Relaxed) → C(old>MAX_REFCOUNT) → B_false
            → O_Arc.ptr → GlobalClone [private] → K_Arc(ptr,phantom,alloc).
Drop_prefix body: I → StrongFetchSub(1,Release) → C(old!=1) → B.
Slow body: O_Arc.ptr → K_Weak(ptr,&alloc) → O_Arc.ptr → NonNullAsPtr [private]
         → PlainStateRetirement → ImplicitWeakDrop.
```

Pinned `Arc::clone` reaches actual constructor helper `from_inner_in`;
fieldless PhantomData adds no separate positive fee. Clone's true overflow
branch retains residual `Abort`, not `Clone_ok`'s success suffix. Clone neither
increments weak count, copies State, clears pointers, nor allocates backing.
Non-last strong drop returns after `Drop_prefix`; last strong drop adds
`FenceAcquire + Slow`. Non-TSan `acquire!` does **not** evaluate its `inner()`
argument: no additional inner observation. Release debug assertions are absent.
`Slow`'s temporary Weak borrows allocator and transfers the existing implicit
weak obligation; no weak-count increment or new obligation is created.
State's three `AtomicUsize` fields and one `usize` are plain retirement: no wipe
or new flat destructor fee. Its implicit weak retirement reaches:

```text
WeakInner_live body: O_Weak.ptr → AsPtr/IsDangling [private] → B_false
                   → K_WeakInner(strong_ref,weak_ref) → K_Some.
WeakDrop_prefix body: WeakInner_live → Q_Option → O_WeakInner.weak
                    → WeakFetchSub(1,Release) → C(old==1) → B.
BackingRelease body: O_Weak.alloc → O_Weak.ptr → NonNullCast [private]
                   → O_Weak.ptr → NonNullAsPtr [private]
                   → LayoutForValueRaw<ArcInner<State>> [private]
                   → AllocatorRefForward [private] → existing GlobalDeallocate.
```

`Q_Option` is required before decrement even on known `Some`; its dispatch is
not erased by that source fact. Weak-nonlast stops at `WeakDrop_prefix`;
weak-last adds `FenceAcquire + BackingRelease`. Retain both branches: source
privacy alone does not prove absence of external Weak owners. Last-weak layout
includes ArcInner counters and padding, not merely State. Global deallocation
boundary occurs once; private allocator implementation is inside, not additive.
Pointer, allocator-clone/forwarding, and layout private bodies stay unexpanded
residual subvectors, never new flat fees. These Arc refinements replace the
existing `ArcStrongClone(State)`/`ArcDrop_State` bodies without duplicating
Memory construction, Reservation decrement, or `Words` cleanup composition.

#### Pinned anchors and still-open scope

Rust [1.85.1 RawVec](https://github.com/rust-lang/rust/blob/1.85.1/library/alloc/src/raw_vec.rs),
[Vec](https://github.com/rust-lang/rust/blob/1.85.1/library/alloc/src/vec/mod.rs)
(reserve 1381–1383, truncate 1523–1542, len 2682–2691, resize 3004–3012,
extend_with 3129–3155), and [SetLenOnDrop](https://github.com/rust-lang/rust/blob/1.85.1/library/alloc/src/vec/set_len_on_drop.rs)
anchor allocation/resize. Core [Layout](https://github.com/rust-lang/rust/blob/1.85.1/library/core/src/alloc/layout.rs),
[slice](https://github.com/rust-lang/rust/blob/1.85.1/library/core/src/slice/mod.rs)
(copy 3707–3735), [intrinsics](https://github.com/rust-lang/rust/blob/1.85.1/library/core/src/intrinsics/mod.rs),
[UB checks](https://github.com/rust-lang/rust/blob/1.85.1/library/core/src/ub_checks.rs),
and [Clone](https://github.com/rust-lang/rust/blob/1.85.1/library/core/src/clone.rs),
plus zeroize 1.9.0 `src/lib.rs:641–660`, anchor remaining bodies.
U1 anchors: Rust 1.85.1 `core/src/num/uint_macros.rs:486–499,870–873,2528–2531`,
`num/mod.rs`, and `core/src/convert/num.rs::impl_try_from_upper_bounded`.
U2: `core/src/option.rs:930–935,1252–1257,1447–1455` and
`core/src/result.rs:564–566,853–858`. U3: RawVec `Cap::ZERO`, `new`, `new_in`,
`RawVecInner::new_in`, Vec `new`, zeroize 1.9.0 `Zeroizing::new:613–614`.
U8: [Rust 1.85.1 sync.rs](https://github.com/rust-lang/rust/blob/1.85.1/library/alloc/src/sync.rs),
`from_inner_in`, clone 2170–2192, drop 2549–2593, drop_slow 1894–1906,
Weak inner 3073–3083, Weak drop 3222–3236; constructor symbol checked at pin.
Residual subvectors above and whole-path/error/unwind/abort composition remain
open, as do Original Root first-advance, debit linkage, scope, retirement
coverage/prepay, numeric rates, funding, security, and deployment. Current
`below` has the authored 32-draw cap, not complete all-outcome funding coverage.
Independent static readback is complete
for these conditional refinements only.
This source-only prospective funding design adopts no rates, tariffs, semantic
fees, Root route or funding/security acceptance. The user's 7 November 2026
administrative release separately permits source/test authoring and releases
funding-first sequencing and Native holds. Standing Native protocols and numeric
caps remain; tests are unexecuted and the current Rust wheel is not qualified.

### Successful fixed2batch Words acquisitions and conditional setup composition

**DESIGN ONLY; independently reviewed success-schedule count, not work closure.**
Astra derivation independently reviewed: callgraph PASS; ownership PASS
conditional on the setup shape and counting boundary below.
Scope is the current successful fixed2batch request with positive-sized buffers;
citations below are under `crates/pllm-core/src/maverick_batch/`.
This records acquisition multiplicity, not new helper fees or traffic totals.

#### Per-request successful acquisition schedule

| Reached owner / operation | Successful Words acquisitions | Source anchors |
| --- | ---: | --- |
| Samples: 12 calls, two buffers each | 24 | `request.rs:62–64,85–88,104,130–131,147–149,168–169`; `protocol.rs:682–686`; `code.rs:188–199` |
| Four claims: each syndrome 3 plus masked/correction/disclosure 3 | 24 | `protocol.rs:351–360`; `code.rs:152–162` |
| Server evaluate outputs | 4 | `protocol.rs:732–758` |
| Fixed original replies | 4 | `request.rs:126–127`; `protocol.rs:494–510` |
| Original encodes | 8 | `protocol.rs:538–543`; `code.rs:120–132` |
| Primary disclosure copies | 4 | `protocol.rs:558–563`; `code.rs:226–230` |
| Two server projection calls, each one output plus two `from_pairs` copies | 6 | `protocol.rs:773–783`; `code.rs:202–223` |
| Fixed auxiliary replies | 2 | `request.rs:166–167`; `protocol.rs:604–612` |
| Secondary syndromes | 12 | `protocol.rs:630–649`; `code.rs:152–162` |
| Accepted demasks | 4 | `request.rs:177–180`; `protocol.rs:666–672,689–702` |
| **One complete successful request** | **92** | Sum of the ten disjoint rows |
| **Four complete successful requests** | **368** | `4 × 92` |

Projection `from_pairs` runs once per repetition, **outside the matrix loop**.
There are two repetitions; each sparse copy is reused across both matrices.
The borrowed free `project` helper (`protocol.rs:705–715`) allocates zero Words; do not count it as
another output owner. Secondary syndrome computation is outside the primary
equation loop, not repeated for each primary equation.
`Words::copy` calls `zeros` once, not twice (`memory.rs:290–293`; shared ownership equations above).
Sampler has no `from_pairs` or final copy: its 24 acquisitions are already
inside 92 and must never be added again as an independent request subtotal.

#### Setup shape, ownership, and safe arithmetic

Let `s` count separately owned successful import → validate → split setups of
**exactly the previously reviewed shape**: two 64×64 matrices; two privacy codes
and the primary code each dimension 64 / length 512; secondary code dimension
128 / length 1,024; four descriptor arrays per code; and matching P/Q/Qaux
shapes and bindings. Let `r` count complete successful requests using such
material. The prior setup subtotal remains `1,566 = 23 + 1,539 + 4`
(import + validate + split). The latest callgraph review did **not** re-audit
that setup subtotal; it supplies the request coefficient 92.

```text
A(s,r) = 1566s + 92r.
A(s,4) = 1566s + 368.
A(1,4) = 1934.  # One reused matching setup; four complete requests.
A(2,4) = 3500.  # Two separately owned matching setups; four requests total.
```

`ValidatedPublication::split(self)` consumes one publication and returns **one**
client/server pair (`protocol.rs:250–252`), not several clients from one split.
This arithmetic creates no common-W multi-client route and does not deduplicate
equal source inputs. Requests borrow existing material; Arc clones share
metadata, not separately owned setups. Reuse requires actual admission and
remaining ordinals: two ordinals are burnt per request, including their actual
failure/burn history. Across separate meters, `A` is an aggregate of event
counts, not a claim that one meter reports the aggregate reading.

#### Counting boundary and unchanged obligations

`Words::zeros` counts a successful acquisition after reserve, capacity check,
and resize (`memory.rs:230–303`; shared Words equations above). Selected buffer
extents are positive. This counter is **not guaranteed physical allocator-call
count**, nor evidence of deployed CPU/RSS or unique backing.
Arc<State>, Arc<Control>, and split metadata allocations, and the 4,096 logical
reservation, are excluded from this Words count but are **not free**.
Exact length, initialization, copy, wipe, retirement, and peak fields remain
separate obligations; no new traffic totals are inferred here. Independent
traffic-extent derivation remains separate from this count.

Failed, replayed, malformed, allocation-error, RNG-error, and refusal paths
truncate the successful schedule; 92 is not their all-outcome work bound.
Current `below` caps fresh-u32 calls at 32, not entropy-provider time.
Neither the request count nor setup composition supplies scalar
fees, full elementary expansion, finite whole-method W, or all-outcome coverage.
Original Root paying issuer, first-advance authorization/prepayment, debit
linkage, and authorized retirement coverage remain unproved.
No full Qwen, projection, security, admission or qualification follows.
No rates, new policy, caps or admission are adopted by this count. The user's
7 November 2026 administrative release separately removed source/test,
funding-first sequencing and Native holds; standing Native protocols and numeric
caps remain. Tests have not been executed; the current Rust wheel is not qualified.

### Successful fixed2batch buffer payload extents and retirement

**DESIGN ONLY; independent static source/math/capacity/retention review PASS
CONDITIONAL on the shapes, success schedule, and helper qualifications below.**
This refines payload extents for the same 92 positive-sized Words owners per
successful request, using the reviewed native shapes. Existing `Z`, `Copy`,
`DW`, and U1–U8 contracts remain unchanged; this is neither helper/model adoption
nor a full-Qwen result, rate schedule, or work bound.
Source citations below are under `crates/pllm-core/src/maverick_batch/`.

#### Initialized extents and borrowed-source copies

Each formula sums initialized `u32` lengths, not simultaneous live capacity.
The final column counts only payload extent reached through `Words::copy`.

| Category | Owner count | Initialized `u32` lengths / formula | Sum length | `Words::copy` extent |
| --- | ---: | --- | ---: | ---: |
| Samples | 24 | `4*(pool512+pairs330)+4*(pool512+pairs62)+4*(pool1024+pairs62)` | 10,008 | 0 |
| Four claims | 24 | `4*(dense512+first512+result64+masked64+correction64+request.values64)` | 5,120 | 256 |
| Evaluate | 4 | `4*64` | 256 | 0 |
| Fixed originals | 4 | `4*64` | 256 | 256 |
| Original encode | 8 | `4*(first512+result512)` | 4,096 | 0 |
| Primary disclosures | 4 | `4*pairs62` | 248 | 248 |
| Server project | 6 | `2*(values256+2*pairs62)` | 760 | 248 |
| Fixed auxiliary | 2 | `2*256` | 512 | 512 |
| Secondary syndrome | 12 | `4*(dense1024+first1024+result128)` | 8,704 | 0 |
| Accepted demask | 4 | `4*64` | 256 | 0 |
| **Total per successful request** | **92** | | **30,216** | **1,520** |

Shapes follow `mod.rs:28–36` and `publication.rs:19–27`; encode/syndrome bodies
are `code.rs:116–169`, and sampling/sparse bodies are `code.rs:173–230`.
Claims use `protocol.rs:342–369`; fixed originals, encoding, and disclosures
use `protocol.rs:494–575`; auxiliary/secondary work uses `protocol.rs:604–648`.
Demask uses `protocol.rs:666–715`; evaluate/project uses `protocol.rs:732–788`;
request sequencing is `request.rs:126–181`.

Sampler owns a pool plus **one interleaved pairs buffer** of length `2*weight`,
not separate positions and values buffers. Each `from_pairs` similarly owns
one interleaved pairs backing. Each server project constructs two such copies,
one per repetition, and reuses each across both matrices.
The claim's `Words::copy` source is the masked disclosure, **not raw input**.
Encode computes `G^T*w`; syndrome computes `G*e`. Their numerical-loop traffic
is not silently included in the copy column or initialization count.

#### Conditional logical capacity and restricted payload coordinates

Let `I_req = sum_i n_i = 30,216` be initialized length across the 92 owners,
and `C_req = sum_i c_i` be their logical capacity sum.
Successful `zeros` checks `capacity == length` before resize
(`memory.rs:275–287`). Adopted U5 adequate-capacity resize does not grow;
later exposed slices cannot resize the backing (`memory.rs:296–302`).
Thus each selected successful owner has `c_i = n_i`, and `C_req = 30,216`.
This is conditional on the selected pinned Vec model and existing dependency
qualifications, not a physical footprint, peak, or simultaneous capacity claim.

Restricted existing payload coordinates, each owned once, are:

| Coordinate | Per successful request |
| --- | ---: |
| Initial payload zero writes | 30,216 `u32` writes |
| `Words::copy` source reads | 1,520 `u32` reads (6,080 bytes) |
| `Words::copy` destination writes | 1,520 `u32` writes (6,080 bytes) |
| Eventual initialized-word wipes, all 92 owners | 30,216 `u32` writes |
| Eventual distinct capacity-byte wipes, `4*C_req` | 120,864 byte writes |
| **Restricted full-lifecycle reads** | **1,520 `u32` reads** |
| **Restricted full-lifecycle word writes** | **61,952 `u32` writes** |
| **Restricted full-lifecycle capacity-byte writes** | **120,864 byte writes** |

Word wipes and capacity-byte wipes are distinct adopted passes over overlapping
storage; neither replaces the other. These are not complete helper reads:
U5 scalar-zero clone/resize reads remain separate, as do explicit sampler
stamping, swaps, pair writes, and encode/syndrome/evaluate/project/demask
numerical loops. Controls, metadata, and residual helper bodies remain separate.
These restricted coordinates are not new helper invocation fees, physical
traffic measurements, or proof of complete work or funding.

#### Retirement partition and caller custody

By normal successful `finish_request` return, 74 internally owned buffers
(28,688 words) have retired under the adopted destructor contracts:
28,688 initialized-word wipe writes plus 114,752 capacity-byte wipe writes.
`finish_request_with` consumes the request; only accepted output owners escape
that consuming finish path (`request.rs:142–181`; `protocol.rs:193–200`).

Separately, 18 owners are transferred to callers across the full schedule:

| Caller-transferred category | Owners | Initialized words |
| --- | ---: | ---: |
| Requests | 4 | `4*64 = 256` |
| Server replies | 4 | `4*64 = 256` |
| Primary disclosures | 4 | `4*62 = 248` |
| Server auxiliary replies | 2 | `2*256 = 512` |
| Accepted outputs | 4 | `4*64 = 256` |
| **Aggregate caller-transferred lifecycle** | **18** | **1,528** |

These are **not necessarily 18 still-live owners at return**. Earlier borrowed
caller owners may already have been dropped; views do not consume owners
(`protocol.rs:71–119`; `request.rs:99–103,142–146`). Accepted outputs are
retained at success. Aggregate caller-transferred lifecycle obligation is
1,528 initialized-word wipes plus 6,112 capacity-byte wipes; actual outstanding
obligation depends on caller drops. The partition reconciles as `74+18=92`,
`28,688+1,528=30,216`, and `114,752+6,112=120,864`.
Reservation/Arc retirement is not necessarily final shared-State or setup
release (`memory.rs:246–262`); existing last/non-last Arc contracts still apply.

#### Unchanged authority and open coverage

Administrative source/test, funding-first sequencing and Native holds were
released on 7 November 2026. Actual first-advance/source evidence, Original
Root paying issuer and root-debit linkage, retirement custody/prepayment,
all-outcome RNG/cleanup, and residual helper coverage remain open. Current
`below` has the authored 32-draw cap; it does not close Dstage, all-outcome
funding, authentic issuer/Root linkage, escaping-owner backing custody or security.
No security, whole-projection, whole-Qwen, admission or Native qualification is
established. No fees, rates, new policy or caps are adopted by this section.
Standing Native protocols and numeric caps remain, not administrative holds.
Tests have not been executed; the current Rust wheel is not qualified.
