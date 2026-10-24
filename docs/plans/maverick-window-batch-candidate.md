# Maverick M-X: one original-request window/batch candidate

## 1. Authority, scope and disposition

**Proposal only, commissioned by D37.** This document specifies one candidate,
`M-X/256x128-local-recompute-1`; it adopts no protocol/storage policy and authorizes
no implementation, preprocessing, source acquisition or model execution. D31/D32's
`M-N/Qwen3-4B-8prompt-candidate-1` arithmetic is unchanged. D34's clear research
oracle does not admit a protected executor. M-B, M-C and M-S remain separate
partition/batch, code/security and storage/ownership obligations; M-D generation
is deferred.

The request consists of exactly eight original token IDs at positions 0–7,
36 decoder layers and all **151,936 final-position logits**. There is no token
selection, feedback, KV persistence, extra prompt or reduced vocabulary. The
source is the selected `Qwen/Qwen3-4B-FP8` revision
`96b30dc13593a244a5e59e84687309f53c375cfa`, not established Maverick-author bytes.
The client retains local attention/nonlinear operations; the same untrusted
evaluator receives masked public-matrix products. There is no additional party.

Resident, unsplit P+Q would cost **257,425,408,000 bytes**, before Qaux or any
other owner. It is rejected, not made to fit by preprocessing. The candidate
below instead recomputes complete, finite public publications in bounded local
windows. Its payload arithmetic leaves room under the existing 2 GiB client and
6 GiB compiled-cache ceilings, but **does not prove admission**. The present
64×64 capsule rejects its geometry, and actual-code distance, dual-LPN, complete
source validation, numerical certificates, framing/capacity and finite-work
quotes are missing. Expected current result: **reject before preparation**.

Normative references: [D31/D32 map, §3](../research/maverick-numeric-source-recovery.md#3-complete-numerical-candidate--proposal-only-not-a-recovered-fp8-model),
[numeric plan](maverick-numeric-completion.md),
[source/reproduction contract](../../paper/reproduction-contracts/maverick/README.md),
[Protocol-2 equations](../../paper/methods/delegation-contracts.md#34-batch-verification-modes-and-acceptance)
and [decision register](../../paper/methods/DECISIONS.md).

## 2. Existing contracts to extend, not replace

The consuming dependency implementation at `20aad09c` is the starting point:
[`Client`, `DependentRequest`, `DependentBatch`, `ChallengedBatch`](../../crates/pllm-core/src/maverick_batch/protocol.rs)
and [the dependent transition](../../crates/pllm-core/src/maverick_batch/dependent.rs).
They preserve tentative dependent values, original/auxiliary fixation and
terminal consumption. The task's `Protocol2RequestOwner` denotes that ownership
requirement; there is no literal type by that name in this baseline and no
implemented model-wide window owner is inferred from it.

Reuse the [original private-matrix traversal](../../crates/pllm-compiler/src/replicated_decoder/private_matrix_driver.rs)
from `7fa07d74`, original semantic/source references, and the shared publication
inventory. Do not parse operation names, fuse q/k/v or gate/up, or create one
source-artifact record per tile. Keep the 253 original matrix references and
derive finite tile/claim ordinals from them. The existing inventory's 16,384
artifact and 32 MiB metadata limits remain limits, not permission for 898,920
heap-allocated claim descriptions.

The current [allocation contract](../../crates/pllm-core/src/maverick_batch/memory.rs)
permits only `(m,n,B)=(64,64,2)` under its **16 MiB** coupled-capsule envelope,
including external copies, server buffers and allowances. This proposal neither
calls `for_shape(256,128,2)` successfully nor raises that ceiling. A separately
reviewed window owner must keep active coupled workspace within 16 MiB while
charging the complete request transcript to the existing 2 GiB aggregate client
owner. Simply retaining one current `Material`/`Arc` per tile fails this contract.
The current code validator also caps message dimension at 128; the proposed
256-dimensional primary/secondary codes are not already supported by it.

## 3. Exact partition and batch inventory

Write each original matrix as `M[R_i,C_j]`, with consecutive input intervals of
128 columns and output intervals of 256 rows. All body shapes divide exactly.
The head has 593 full 256-row intervals and one **128-row** tail; its 20 column
intervals are unchanged. No padding or synthetic zero claims are introduced.

The partition identity is a direct field-algebra derivation, not a recovered
author partition manifest:

`(M*x)[R_i] = sum_j M[R_i,C_j]*x[C_j] mod p`.

Quantize the original full input once according to D31, then slice its canonical
field words. Demask each tile reply tentatively, accumulate column partials
modulo BabyBear, and perform D31's centered `/65536` **direct RNE BF16** conversion
only after the complete original row has been assembled. Do not center, divide,
cast, clamp or RMS-normalize partial tiles. Partial integer sums may cross the
field center: only the original complete dot's honest no-wrap certificate gives
the claimed real/integer interpretation. Modular partition equality itself does
not prove no-wrap. D32 scales retain their original global 128×128 indices;
the BF16-source head still uses D31's Q8 `Lin_head`, not a BF16 dot substitute.

For each body tile pair token positions `(0,1)`, `(2,3)`, `(4,5)`, `(6,7)`.
Each Protocol-2 batch has B=2, identical matrix tiles in its ordered slots and
fresh independent slot noises. For the head's single final-position input,
pair adjacent **column** tiles `(0,1),…,(18,19)` within each output interval.
These are two possibly different matrices and input slices of the same `(m,n)`;
their partial outputs are both needed. The tail uses its own primary code.

| Original role | Matrices | Original `(m,n)` | Tiles per matrix | Tile claims/request | B=2 batches/request |
| --- | ---: | --- | ---: | ---: | ---: |
| q | 36 | (4096,2560) | 320 | 92,160 | 46,080 |
| k | 36 | (1024,2560) | 80 | 23,040 | 11,520 |
| v | 36 | (1024,2560) | 80 | 23,040 | 11,520 |
| o | 36 | (2560,4096) | 320 | 92,160 | 46,080 |
| gate | 36 | (9728,2560) | 760 | 218,880 | 109,440 |
| up | 36 | (9728,2560) | 760 | 218,880 | 109,440 |
| down | 36 | (2560,9728) | 760 | 218,880 | 109,440 |
| head, full rows | 1 part | (151808,2560) | 11,860 | 11,860 | 5,930 |
| head, tail | 1 part | (128,2560) | 20 | 20 | 10 |
| **Total** | **253 originals** | | **122,760 distinct tiles** | **898,920** | **449,460** |

The original request has 2,017 unpartitioned claims: 36×7×8+1. Partitioning raises
protocol/query counts, not token count. There are **116,820 distinct ordered
publication groups**: 110,880 body tiles, each reused by four token-pair batches,
plus 5,940 head tile pairs. Aliasing equal public numeric arrays must not merge
claim IDs, noises, ordered source bindings or acceptance obligations.

## 4. Proposed codes, security horizon and conditional budget

These are **proposed M-B/M-C settings**, not recovered evaluated-author settings
or certified code choices. Keep BabyBear `p=2013265921`, four-byte canonical field
elements, source RAA repeat factor eight and independent full source-ensemble
descriptions. Do not substitute seeded descriptions, systematic codes or a PRG.

| Role | Message dimension | Encoded length | Proposed target/parameters |
| --- | ---: | ---: | --- |
| privacy Gx, slot 0 and slot 1 | 128 each | 1024 each | independent codes; target distance 1/2; `tx=164` |
| primary Gy, full rows | 256 | 2048 | target distance 1/2; `t=31`, `ell=2` |
| primary Gy, tail | 128 | 1024 | independent code; same t/ell |
| secondary Gprime | B*n=256 | 2048 | independent of every other code; target distance 1/2; `t'=31`, `ell'=2` |

Five code instances are reused throughout this one request. Their explicit
permutation/scale arrays total `16*(1024+1024+2048+1024+2048)=114,688` bytes per
complete copy. Code labels of equal dimension do not imply equal code instances.
`tx=ceil(236*ln(2))=164` comes from M Eq.(1) at lambda=128, Nx=1024 and delta=1/2;
this is the source's **heuristic**, not a dual-LPN security certificate.

The horizon is **one top-level attempt total**, including failed/cancelled
attempts. A root ledger covers all publications, windows and helper lifetimes;
reloads or object reconstruction do not reset it. Each body publication consumes
four batch attempts and each head publication one, below the current per-client
limit 16, but that limit is not a whole-model security budget. A second request,
the 1,024-case quality campaign, code resampling/grinding or a retry requires a
new aggregate proof/decision; no reset is inferred from rebuilding public state.

There are 898,920 independent privacy-noise samples, 898,920 primary sparse
challenges and 898,920 secondary sparse challenges. Primary challenges become
public only after original fixation; secondary challenges remain client-private.
There are 1,797,840 original scalar checks (449,460 batches × 2 primary repetitions
× 2 slots) and 1,797,840 secondary scalar checks.
Each Gx instance serves 449,460 samples. Five code-distance failure events must
be budgeted, plus any sampling/validation selection failures; no finite
model-dimension distance certificate currently supplies these probabilities.

M Theorem 5.4 gives a bound for **one batch**, not all windows for free. Conditional
on all relevant distances, set

`epsilon = 2^-31 + 1/(p-1) = 31/32212254720`.

The union bound for W=449460 batches is

`W*(epsilon^2+epsilon^2) = 2399617/2882303761517117440 < 2^-40`.

The exact remaining allowance is only
`221823/2882303761517117440` for code failures and other applicable soundness
events. Assigning one fifth to each code would require each failure probability
at most `221823/14411518807585587200`; that allocation is a proof obligation,
not evidence it holds. Shared codes/challenges inside a batch do not add a B
factor, but the distinct batch union does add W. Multi-query privacy must cover
all 898,920 fresh masks and the complete repeated-segment transcript under M
Remark 4.3/Theorem 6.1. Neither Eq.(1), the source's asymptotics nor this conditional
arithmetic certifies the selected BabyBear instances or repeated-segment privacy.
The five-way allowance above is a conservative verification/code-event budget,
not a lambda=128 privacy failure allowance. Privacy needs its own concrete
distribution/horizon argument. In particular, the cited RAA distance theorem's
`q>2^31` hypothesis excludes unchanged BabyBear; no finite certificate follows
from that theorem here.

## 5. Admission and complete public-state lifecycle

1. **Metadata admission first.** Freeze original semantic/source references,
   every partition and token pair, code/parameter identities, horizon, resource
   bounds, local-store capacity, owner multiplicities, finite work and transport
   quotes. Refuse unknown arithmetic, overflow, capacity or work before tensor
   acquisition, hashing, code construction, preprocessing or private issuance.
   Checking whether generated P/Q fit after construction is forbidden.
2. **Source and code validation, after that admission only.** Acquire/validate the
   complete pinned source, tensor extents, FP8 NaN exclusions, positive-finite
   scales, BF16 embedding/head equality and the required source/range proofs.
   Previously observed scale/Gamma ranges do not validate all tensors. Preserve
   the recorded index `total_size` excess of **1,310,720 bytes**; do not repair it
   or count metadata/ranges as whole-shard integrity. Validate all code entries,
   permutations, nonzero scales, role separation and actual M-C certificates.
   Embedding lookup must not turn the private token IDs into externally visible
   row reads: the proposed request schedule scans the entire local BF16 embedding
   in public row order and gathers the eight rows into client-private RAM. It
   changes neither their bits nor D31 lookup semantics. This adds at least
   777,912,320 bytes of local source reads per request beyond setup validation;
   there is no token-indexed remote fetch or hidden resident table allowance.
3. **Issue and compute tentatively.** For each public group in original dependency
   order, acquire its entire matrices/codes, construct and completely validate
   both P, both Q and Qaux against the intended source. Use bounded full local P
   buffers to compute fresh masks/corrections. Read all relevant public data on
   a deterministic public schedule, not by remotely requesting secret supports.
   Server M/Q buffers are separately owned. Freeze each raw reply before using
   its demasked value tentatively. Retain raw masked inputs/replies for later
   checks; erase noise and corrections once demasking has finished. Retire the
   public window before advancing, without retiring the claim or its replay state.
4. **Global original barrier.** Finish every original reply, including the whole
   head, before sampling **any** primary challenge. Tentative logits have no
   accepted-output authority. Recompute/acquire server Q in public group order,
   disclose the two fresh primary challenges per batch, and fix all auxiliary
   replies into a client-controlled local public spool.
5. **Global auxiliary barrier.** Fix **every** auxiliary reply before sampling
   any secondary challenge. Then revisit public groups: reconstruct and completely
   revalidate their public publications, and check all secondary and original
   equations. Qaux is scanned completely through bounded local windows; never
   fetch only its secret `t×t'` coordinates from a server or remote object store.
6. **One terminal result.** Release the complete final logit vector only after
   every required check and owner transition succeeds. Otherwise discard all
   tentative values. Root cancellation, malformed input, source drift, work
   exhaustion/typed `Unresolved`, bad entropy, I/O failure or verification failure
   burns the attempt and all issued material; none produces partial logits or a
   fresh retry. Erase private state before its final reservation refund. Close
   public spool/read handles and charge remaining backing until actual release.

This deliberately retains both global barriers, rather than early-verifying a
tile to save transcript space. Recomputations must reproduce the same immutable
source/code publication, not select fresh codes. The window API must decouple
public backing lifetime from consuming claim/control lifetime: dropping a P/Q
buffer must not recreate a reusable request owner.

The source formulas are M Protocol 3, p.10:
`P_b=M_b*Gx_b`, `Q_b=Gy^T*M_b`, `xhat=x+Gx_b*e_x`, and `y=yhat-P_b*e_x`.
For B=2, M Protocol 2, pp.8–9, defines
`Qtilde=[Q_0|Q_1]^T` of shape `(2n,Ny)` and
`Qaux=Qtilde^T*Gprime` of shape `(Ny,Nprime)`. Check
`e^T*Qaux*eprime=(Gprime*eprime)^T*utilde` and **both** original
`u_b^T*xhat_b=e^T*Gy^T*yhat_b` equations in every primary repetition.
No systematic-code storage shortcut is assumed. The author notes that Qaux
can be reused for the same ordered matrices; the four body token pairs use that
public numeric reuse, not noise/challenge reuse.

## 6. Owners, private state and complete retention

| State | Owner, lifetime and release rule |
| --- | --- |
| Immutable source files, complete code descriptions and original-source catalog | Client-controlled public local backing; evaluator may read only explicitly public read-only backing. A shared physical public file is counted once only with proven read-only aliasing; otherwise charge both copies. No new trusted preparer. |
| P0/P1 | Client public correction windows only, completely acquired/validated. No evaluator service receives private support indices. Retire after that group's issuance; rebuild identically if later required. |
| Q0/Q1 | Public, separately owned active evaluator buffers; client validates intended publication independently. Reload/recompute full buffers on the fixed public schedule. |
| Qaux | Public client-local complete sequential construction/validation and later full sequential scan; **never** a resident whole-model collection or remote private-coordinate lookup. |
| Code randomness / keys | Exact source-ensemble randomness; no seeded-code or PRG replacement. These public codes are not secret keys; Protocols 2/3 introduce no HE key or secret trapdoor. Root/session binding, replay state and any existing transport identity/key owners must be quoted explicitly; no unchosen AEAD/HE key or new channel is introduced here. |
| Private noise, correction and original plaintext intermediates | Consuming client-only owners. Fresh per tile claim; erase noise/correction after demasking, and obsolete activations after last tentative dependent use. No private tensor spills to public backing. |
| Masked input/raw reply records | Retained client-controlled packed arrays through all original checks. They are evaluator-visible ciphertext/response values, not a retained plaintext copy. Integrity, immutable order, capacity and cancellation still require ownership. |
| Primary challenges / auxiliary replies | Evaluator-visible, fixed-length public transcript payloads in bounded trusted-local temporary spools. Validate existing headers on receipt; derive record offsets from the frozen original-source/partition traversal. No report archives, receipt collections or private-support address traces. |
| Secondary challenges, accumulators and check failures | Client-private; allocate after the global auxiliary barrier, process one public group at a time and erase before refund. Public errors contain no values, supports, residuals or tentative logits. Source/protocol errors never become numeric zero. |
| Accepted output | One client-owned 151,936-entry vector, with its canonical representation and any serialization copies charged; no per-batch accepted output escapes the root. |

The existing publication/session/attempt/mode/slot binding must cover the ordered
source references, tile coordinates, request and token-pair use. Equal numeric
tiles are not permission to alias sessions. A future bridge must retain each
group's spent control across evictions, enforce the root horizon as well as the
per-group attempt limit, and resolve actual frame/identity ownership. This is an
explicit blocking bridge obligation, not a claim that the present header alone
already binds the full original request. Role separation also needs a reviewed
transport/isolation owner; co-located public read-only backing does not certify
process isolation, authentication, privacy or timing side channels.

## 7. Byte inventory, windows and aggregate admission

All numbers are design arithmetic, not measured memory, disk occupancy or speed.
One 256×128 B=2 publication contains these logical public field payloads:

| Item | Full-row group bytes | 128-row tail group bytes |
| --- | ---: | ---: |
| Two M slots | 262,144 | 131,072 |
| Two P slots | 2,097,152 | 1,048,576 |
| Two Q slots | 2,097,152 | 1,048,576 |
| Qaux | 16,777,216 | 8,388,608 |

Even **one** full Qaux consumes the entire current 16 MiB ceiling before other
owners. It must not be imported using the existing fully resident constructor.
The proposed active workspace envelope reserves four 1 MiB old/new read windows
across both roles, 2 MiB client P, 2 MiB server Q, two separately owned M-slot sets
(524,288 bytes), bounded source decode/input scratch, code copies, challenges,
encoding/syndrome buffers, comparison/hash state, allocator/control and stack.
Their future complete native quote must total **at most 16 MiB**, including
borrowed external source buffers. The unused difference is not free credit:
until all hidden capacities/copies are bounded, this workspace is unresolved.
Qaux rows are only 8,192 bytes; stream all rows and validate all entries rather
than holding its whole payload. Never retain all 116,820 public groups.

Whole-request retention and wire payloads, excluding headers/copies:

| Item | Bytes |
| --- | ---: |
| Masked inputs, 898,920×128 field words | 460,247,040 |
| Raw original replies, respecting full/tail rows | 920,483,840 |
| Retained original check arrays, sum of preceding two | **1,380,730,880** |
| Primary sparse challenge pairs, 449,460×2×31×8 | 222,932,160 |
| All auxiliary replies, 449,460×2×2×128×4 | 920,494,080 |
| Local public spool payload, preceding two | **1,143,426,240** |
| Protocol wire payload, all four message classes | **2,524,157,120** |

There are 2,696,760 logical frames: two per original claim and two per batch.
Add actual framing, send/receive copies, backpressure buffers and terminal-output
delivery; they are not included in the wire payload. Local spools need not copy
wire headers per record: checked immutable ordinals may reuse the root catalog,
but this requires the actual binding/capacity proof, not an assumed compression.

An explicit **proposed reservation envelope**, not a demonstrated allocator
quote, is: retained original arrays plus 64 MiB for complete local numeric
old/new state and final output, 32 MiB metadata, 16 MiB coupled active workspace,
32 MiB root/frame/control capacity, 128 MiB runtime/allocator/page allowance, and
two complete code-description copies. This is **1,666,172,928 bytes**, leaving
481,310,720 bytes under 2 GiB. The guard is inside the old cap; it cannot hide an
unbounded exact-scalar backend, process RSS, mapped pages, transport helper or
native constructor. Prove all actual owners fit or reject; a configured guard
does not bound observed OS high-water.

For local disk/cache, conservatively charge the recorded **5,190,052,872-byte**
advertised pinned shard footprint without subtracting the unresolved index
discrepancy. Add 1,143,426,240 bytes of public spools, 32 MiB of source/catalog/code
metadata and 1 MiB of temporary-public-file/control allowance: **6,368,082,120
bytes**, leaving **74,368,824 bytes** under 6 GiB. This is tight. Actual file
lengths, allocation units, temporary rename copies, reserved extents, metadata,
source/tokenizer/config files, open-unlinked backing and storage bookkeeping
must fit that charge. Two full shard copies, all P/Q on disk, a full canonical
u32 model, or copied per-frame headers can invalidate it; none is silently placed
outside the budget. Qaux is recomputed, not spooled. Separate deployment storage
or a remote object cache is not an escape hatch. A cold acquisition replacing a
live complete source copy must also reject if old/new overlap exceeds the cap.

## 8. Whole work and I/O: recomputation is not free

Across the distinct ordered publication groups, a complete pass produces:

| Derived public values | Logical bytes per complete pass |
| --- | ---: |
| P, both slots | 244,978,810,880 |
| Q, both slots | 244,978,810,880 |
| Qaux | 1,959,830,487,040 |
| **Total** | **2,449,788,108,800** |

The candidate does a construction and a complete source-referenced validation
in the issuance visit, then repeats both in the final-check visit. Thus
**9,799,152,435,200 logical derived bytes** are processed, not retained, before
accounting for the evaluator's separate Q reconstruction for primary responses.
For the existing explicit RAA encoder each encoded output word takes two field
multiplies and two adds; these four passes consequently entail
**9,799,152,435,200 scalar field operations** in that straightforward schedule,
plus input validation, code checks, hashing, comparisons and copies. This is an
algorithmic count, not a CPU-time estimate or permission to run it. Evaluator Q
construction adds 244,978,810,880 analogous scalar operations in one pass;
validation/copies, if required there, are additional.

The honest evaluator's original matrix products still contain **29,455,482,880**
multiply-add terms. Other exact counts for the straightforward current RAA
schedule are below. A product term is not a measured instruction and is not
interchangeable with the explicitly counted scalar field operations.

| Online work | Count, excluding validation/copies |
| --- | ---: |
| Evaluator auxiliary projection terms, `W*B*ell*n*t` | 7,133,829,120 |
| Client privacy correction terms, `tx*sum(claim output widths)` | 37,739,837,440 |
| Client `Gy^T*yhat`, two multiplies/two adds per encoded word | 7,363,870,720 scalar field operations |
| Client privacy syndromes, five field operations per encoded coordinate | 4,602,470,400 scalar field operations |
| Client secondary syndromes, same explicit algorithm | 9,204,940,800 scalar field operations |
| Secondary Qaux bilinear terms, `W*ell*ell'*t*t'` | 1,727,724,240 |
| Secondary right-hand inner-product terms | 460,247,040 |
| Original left-hand inner-product terms | 230,123,520 |
| Original right-hand sparse terms | 55,733,040 |

Account for each product's reductions, every local D31 operation, conversions,
zeroization and scalar-backend workspace as well. There is no existing public
finite-work bound for all source checking, exact local
transcendentals, distance certification, complete entropy rejection, OS/I/O or
transport; these are blocking unknowns, not zero entries.

Sparse sampling requires 203,155,920 accepted nonzero coefficients and the same
number of accepted support choices: 147,422,880 privacy entries plus 55,733,040
verification entries. Uniform rejection can consume additional random words;
an unbounded sampler cannot supply a finite worst-case quote. No new truncation,
biased fallback or RNG is adopted. Code construction randomness and unsuccessful
attempts also need charges.

The source must be completely acquired/authenticated and structurally/numerically
validated before publication use; repeated conversion reads are additional.
Each client publication pass fully scans P/Q/Qaux locally, even though only sparse
coordinates affect some checks. Those logical scans are not remote/network
traffic and not all disk I/O: generated rows may flow directly through charged
RAM. Price actual source rereads/strided row assembly, full local spool writes
and reads (at least **2,286,852,480 bytes** for one write/read), validation passes,
page faults, readahead, hashes, error cleanup, accepted serialization and both
role buffers. No measured throughput, WAN, price, energy or speedup follows.

Source locators are the pinned M PDF, pp.8–10 (Protocol 2, Theorem 5.4 and
Protocol 3/Eq.(1)), p.6 (multi-query Remark 4.3), pp.12–14 (optimistic acceptance
and original request), and Appendix B (code construction/distance limitation).
Their [existing source lock](../../paper/reproduction-contracts/maverick/source-lock.json)
identifies the retained PDF; this plan creates no runtime source-paper dependency.

## 9. Separate next decisions and stop conditions

**M-X-I — bounded planner implementation question:**

> Authorize only a shared method-neutral, metadata-only partition/window and
> consuming-lifetime admission planner for exactly §3's candidate. Reuse original
> matrix/source traversal and existing budget/resource contracts. Use synthetic
> source/frame/capacity fixtures; require symbolic full-inventory counts, strict
> unknown/overflow refusal, global barriers, spent-state preservation, exact-fit
> and one-short aggregate tests. Do not construct codes, P/Q/Qaux, acquire source
> tensors, issue masks, run local model operators or alter the existing capsule.
> Qualification budget: one process/thread, 30 seconds, 64 MiB actual test memory,
> at most 4,096 metadata records and 1 MiB ordinary output; no network/models.
> Large totals are modeled reservations, never allocation requests. Any Cargo
> execution still requires a separately granted build lane.

**M-X-A — separately selectable storage/partition adaptation question:**

> Accept only this proposed 256×128/tail-128, B=2, five-code, one-attempt horizon
> and complete trusted-local recomputation/window policy as the intended M-B/M-S
> contract, retaining D31/D32 and all global verification barriers. Do not raise
> the 2 GiB/6 GiB/16 MiB ceilings or change parties. Require actual M-C, source,
> numeric, binding, finite-work and aggregate-owner certificates before any
> protocol-owner bridge/preprocessing or execution is separately authorized.

Neither question is answered by D37. M-X-I does not implicitly select M-X-A.
Even accepting M-X-A would not certify its codes or grant an executable full-model
protocol, SDK/quality/benchmark entry, trained response or M-D generation.
Current unknown geometry/binding, storage overlap, sampling/owner capacity or
work must return a typed rejection **before** expensive preparation. The payload
fit and conditional union bound are useful design checks, not evidence that any
actual original-source request is safe, correct, admitted or completed.
