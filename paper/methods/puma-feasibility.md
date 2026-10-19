# PUMA P0 feasibility and native cost inventory

## Integrated capacity audit: current decision

**No-go for all three workloads under current resource defaults, including 4+2.**
After compact18 lookup lands, **64+8 still has an independent hard lifetime
blocker in guarded SiLU**, not merely unknown performance: its legacy sign
traffic alone exceeds 1,200 seconds even at 100 Mbps. See the body bound below.
The sections below headed “Archived baseline” preserve the original full64
lookup/expanded-selection identity. They are not updated with compact-codec
timings. Current dimension-only audit is separately identified as
`pllm.puma.capacity-audit.guard256.v1`, exposed by
`cost::audit_common_capacity(rows, selections)`. It returns diagnostics only.

### Current complete-body scaling, not lookup-only scaling

The audit counts the current expanded body with tiled selection and eight-row
bulk lookup. It prices the actual fixed native RMSNorm, guarded SiLU, softmax,
rotary, attention scale/products, body/head rescale and selection schedules.
Synthetic public coefficients are used **only to query instruction/work shapes**:
unit gamma, bounded rotary coefficients, and cap256 input intervals. This does
not assert the real source gamma, weights, incoming intervals or acceptance rate.
All-RMS-input guards provide a conservative **guard-count** envelope; actual
public proofs may elide some. No decoder Program is constructed or admitted.

| Dimension-derived inventory | 4+2 | 16+2 | 64+8 |
| --- | ---: | ---: | ---: |
| Body/lookup/head nodes priced, excluding guards/source/plumbing | **72,872** | **267,566** | **5,680,622** |
| Maximum RMS input guard tiles | 3,087 | 11,319 | 185,220 |
| Nodes with every such guard | 91,393 | 335,479 | 6,791,941 |
| Current cumulative `16*lanes` charge, no guards | 296,392,368 B | 1,192,077,360 B | 29,406,136,896 B |
| Same charge with every guard | 328,102,016 B | 1,308,346,112 B | 31,308,716,720 B |
| Fixed-shape work, excluding guards | 889,249 ordinals | 3,348,889 | 57,860,716 |
| Fixed-shape work, every RMS input guard | **1,194,861 ordinals** | **4,469,469** | **76,197,495** |
| Per-party AES upper allowance, every guard | 116,396,240 blocks | 422,838,200 | **7,316,939,414** |
| Public/pair/private PRSS upper cursors, every guard | 219,813 / 53,177,190 / 15,470,783 | 807,189 / 192,865,710 / 56,578,919 | 14,140,620 / 3,348,713,859 / 968,403,908 |
| Actual eight-row lookup scans | 2 | 5 | 71 |
| Logical matrix reads per worker with that tiling | 10,081,533,952 B | 13,348,765,696 B | **108,938,002,432 B** |

Work entries are **conditional schedule estimates**, not complete-model numeric
admission or measured performance. Source-specific scalar range proofs and final
integrated graph reconciliation remain required. Current guards appear at RMSNorm
inputs, not every graph edge: cap256 is **not** a claim that all activations,
logits or Q/K products are bounded by 256. Guarded SiLU has fixed instruction
shape but still needs safe source-specific domains for its remaining arithmetic.
The conservative AES allowance is an admission upper bound, not a measured count.

**4+2:** its 72,872 priced unguarded nodes already exceed 65,536; 296.39 MB of
cumulative value charges already exceed 64 MiB. Gate/up multiply plus residual
alone charge 69,009,408 bytes, before headers, any other operator, or guards.
The no-guard schedule has 159,327 ordinals of legacy headroom, but the all-guard
case exceeds the ordinal ceiling by 146,285. Therefore no honest blanket
“4+2 now fits” claim; metadata/body-loop work is required even if numeric guards
mostly elide. The subsequent frozen-wheel source publication reaches layer-eight
attention before rejecting its product range; no complete source program admits.
The original receipt is in [local-v12](../artifacts/puma-sdk/local-v12/README.md).

**16+2:** 267,566 priced unguarded nodes and 1.192 GB cumulative value charges
fail resource admission. Even the no-guard shape schedule is 3.194 times the
ordinal ceiling; guard elision cannot rescue that schedule. AES allowance is
below the existing complete-program cap in this conditional inventory, but that
does not admit nodes/storage/work or establish elapsed time.

**64+8:** legacy ordinals fail by tens of millions regardless of compact wire
encoding. All-guard AES allowance is 2.269 times the existing 3,224,371,200-block
cap; even the no-guard conditional allowance is 5,863,147,636. Source values,
individual tensor width and final result frame do not require blanket increases.

### Which nodes dominate after selection compaction?

Counts below exclude guard elisions and are current lowering formulas, not
secret-dependent execution counts. Qwen2 lowering has Q/K/V biases.

| Current outer node category | 4+2 | 16+2 | 64+8 |
| --- | ---: | ---: | ---: |
| Public matrix bodies | 1,008 | 1,008 | 4,032 |
| Q/K/V bias binary tiles | 5,832 | 21,384 | 349,920 |
| RMSNorm row gather/tensor pairs | 882 | 3,234 | 52,920 |
| Rotary row gather/tensor pairs | 6,912 | 25,344 | 414,720 |
| SiLU gather/loop pairs | 96 | 96 | 384 |
| Gate/up binary gather/product triples | 24,624 | 90,288 | 1,477,440 |
| Residual binary gather/affine triples | 9,072 | 33,264 | 544,320 |
| Attention score query/column tiles | 9,072 | 33,936 | 1,135,680 |
| Attention value query/column tiles | 9,072 | 33,936 | 1,146,432 |
| Attention-scale gather/tensor pairs | 240 | 2,880 | 191,712 |
| Softmax row gather/tensor pairs | 6,048 | 22,176 | 362,880 |
| Bulk lookup gather/owner pairs | 4 | 10 | 142 |
| Head / tiled selection | 6 / 4 | 6 / 4 | 24 / 16 |
| Additional guard nodes, worst case | 18,521 | 67,913 | 1,111,319 |

Each guard tile emits input gather, two-output checked owner, safe gather,
validity gather, all-valid reduction and sticky product: six nodes, except the
first guard needs no sticky product. Its cumulative value charge is
`16*(5*tile_lanes + 2)` bytes, minus the first scalar product. These are not
simultaneously live owners. The existing builder nevertheless adds all such
`16*lanes` charges to `allocated`, so increasing only MAX_NODES cannot help.

### Compact18 codec: current evidence and independent gates

Core's bounded18 equality tile reports **23 ordinals / 1,169 AES blocks**, versus
full64 **28 / 2,817** at 128 lanes. The measured intermediate legacy-word codec
was **60,000 encoded bytes per tile across all six directions**, not per link.
This fixes lookup work at 245,709 / 900,933 / 14,742,540 ordinals for the three
geometries. The landed compact nominal-bit codec now measures **19,920 encoded
bytes / 40 native frames per 128-lane tile**. The 60,000-byte number below is
retained only as an intermediate historical result.

Core fixture directed bytes: 0→2 6,268; 1→0 7,384; 2→1 5,152; 2→0 1,116;
other edges zero for that public-pivot realization. Arithmetic OT pivot changes
role assignment: do **not** multiply one fixture's directed split across an
entire request and call it exact. A2B alone has a fixed **4,100 encoded bytes
on each of the three rotation edges per tile**, independent of that split.
These are native PUT2/PUC1 bytes, including native headers, not transport bodies.
Selection's landed plan uses **7,158 ordinals / 1,293,966 AES-upper blocks per
complete 151,936-vocabulary head**, already included in the schedule table above.

| Compact lookup-only extrapolation | 4+2 | 16+2 | 64+8 |
| --- | ---: | ---: | ---: |
| Predicate tiles | 10,683 | 39,171 | 640,980 |
| All-direction native encoded bytes | 212,805,360 | 780,286,320 | 12,768,321,600 |
| All-direction native frames | 427,320 | 1,566,840 | 25,639,200 |
| A2B bytes on **each** rotation edge | 43,800,300 | 160,601,100 | 2,628,018,000 |
| Fixed-edge A2B lower time, 100 Mbps | 3.504 s | 12.848 s | 210.241 s |
| Fixed-edge A2B lower time, 40 Mbps | 8.760 s | 32.120 s | **525.604 s** |
| Lookup coordinator advances, minimum | 138,880 | 509,224 | **8,332,741** |
| Lookup remote control calls, minimum | 854,642 | 3,133,682 | **51,278,402** |

Lookup-only compact A2B floors no longer prove 1,200-second infeasibility. They do
not prove feasibility either: other equality phases, all body/selection/guard
traffic, serialization, HTTP/WebSocket envelopes, CPU, import and cleanup remain
additional. At 64+8, fixed-edge A2B alone consumes 43.80% of a 1,200-second
40-Mbps budget. The default 60-second client
deadline still cannot serve 64+8 from that bound. Even 16+2 leaves under 28
seconds at that default deadline before the entire rest of execution.

The coordinator failure is **independent of bytes and ordinal admission**.
The bounded predicate has 13 native instructions; `instructions()` excludes
Input. `Worker::advance_inner` starts one at a time, including local maps.
Lookup therefore requires `13*tiles + 1` advances including final completion,
before any other stage. At 64+8 that alone exceeds the worker's **2^20 outer
barriers** by 7,284,165. A larger PRSS/ordinal allowance or compact codec cannot
fix this. Four remote controls are unavoidable per noncomplete advance
(two all-role advance RPCs plus two terminal prepare RPCs); the seven A2B waves
also need at least four controls each (prepare/dispatch). Thus
`4*13*tiles + 4*7*tiles + 2` is a conservative lookup control-count floor,
not a network-round count or a wall-time measurement. Other predicate waves,
seal/setup/finish and every model stage remain additional. Reusable streams
still execute this run loop; a phase-ready native scheduler is separate work.

In the current serial dispatcher each active primitive step is separated by
global control. The 1,024 inner-wave ceiling is a different gate, not solved
by extending outer iterations. Price the actual longest primitive/compound
dependency chain before changing either budget. The new admitted report derives
`scheduled_coordinator_advances` from the recognized operators; an admitted tiny
three-party execution now calibrates that count against actual native advances.

Frozen intermediate all-direction lookup traffic is 640,980,000 / 2,350,260,000 /
38,458,800,000 bytes. A six-link pigeonhole screen gives at least one directed
link 21.37 / 78.34 / **1,281.96 seconds** at 40 Mbps (8.55 / 31.34 / 512.78 at
100 Mbps). It does not identify that link or permit assigning the average to
every edge. Thus the **intermediate** 64+8 codec failed the deadline; the current
compact codec has the separate lower bounds above. Reconcile all six real
directed counters after codec freeze, not the old full64 cycle bound.

### Shortest credible scaling path

**Body traffic is a hard gate, not an unpriced zero.**
`tensor.rs::activation_predicates` emits three Sign conversions per guarded
SiLU lane. `Executor::compact()` selects the compact wire contract only for
BOUNDED_EQUALITY_PROFILE; SiLU still uses full-word PUM1/PUT1 frames.
`conversion::Plan::geometry` sends one n-word split, one n-word initial AND and
six 2n-word carry gates on each directed rotation edge. The backing types shrink
AES draws, but legacy `Message::encode_bytes` writes every word as eight bytes.
Thus Sign alone costs at least `112*n` payload bytes per edge, and guarded SiLU
alone costs `3*112*24*4864*sum(prefix)` payload bytes on **each** cycle edge.
This does not include native headers, B2A, polynomial products, lookup or guards.

| Hard directed-edge floor, existing SiLU schedule | 4+2 | 16+2 | 64+8 |
| --- | ---: | ---: | ---: |
| SiLU Sign payload bytes per edge | 353,009,664 | 1,294,368,768 | **21,180,579,840** |
| SiLU Sign seconds at 100 Mbps | 28.241 | 103.550 | **1,694.446** |
| SiLU Sign seconds at 40 Mbps | **70.602** | 258.874 | **4,236.116** |
| SiLU Sign + compact lookup A2B bytes per edge | 396,809,964 | 1,454,969,868 | **23,808,597,840** |
| Combined floor seconds at 100 Mbps | 31.745 | 116.398 | **1,904.688** |
| Combined floor seconds at 40 Mbps | **79.362** | 290.994 | **4,761.720** |

Consequences: 4+2 cannot fit the default 60-second client deadline at 40 Mbps;
16+2 cannot fit it at either rate. Both need an explicit larger client deadline
even before CPU and remaining traffic are measured. They are not rejected by
this subset against a 1,200-second deadline, but capacity/numeric/remaining-time
gates still fail or remain unknown. **64+8 cannot fit the 1,200-second attempt at
either directed rate with the existing body protocol**, regardless of lookup
file reuse, node limits or coordinator scheduling. All selections execute this
same public schedule even if private range validity eventually fails.

A general compact Boolean codec alone is not enough to claim the study deadline.
Even ideally packing the existing full-word Sign carry circuit's nominal widths
would cost `8n + 8n + 2n*(32+16+8+4+2+1)/8 = 31.75n` payload bytes per Sign per
edge, before any headers. Three such signs over 63,037,440 SiLU lanes already
give **6,004,316,160 bytes / 1,200.863 seconds at 40 Mbps**, excluding everything
else. This is a bound for retaining that circuit, not a universal cryptographic
communication lower bound. A viable 64+8 route needs **less actual nonlinear
comparison work** (for example, separately proved narrower signed predicates or
another reviewed protocol) or a consciously different lifetime/link contract.
Public gate/activation domain proofs for such narrowing are not established by
the lookup token-domain proof, and full-ring failure-safe guards cannot silently
inherit a narrow range assumption.

1. **Keep defaults and primitive envelopes unchanged.** Finish codec/native
   integration and freeze identities before any full-model source admission.
2. **Add reusable outer tensor/row loops**, not millions of metadata nodes:
   two-input elementwise loops for multiply/residual/bias; public-position rotary
   rows; RMSNorm row loops; causal softmax rows; query-row attention contractions;
   score-scale loops; checked-range loops with internal all-valid/sticky reduction.
   Preserve the existing primitive lanes and fresh work/PRSS for every iteration.
   Do not unroll them into the core tensor's fixed 192-instruction/u8 namespace.
3. A plausible looped descriptor target, **not yet a capacity authorization**:
   at 64+8, gate/up 576 outer nodes, residual 1,152, bias 1,152, rotary 768,
   RMSNorm 784, softmax 384, both attention products 768, score scale 384, and
   worst-case norm-guard loops 1,176, plus existing linear 4,032, lookup 142,
   SiLU 384 and head/selection 40: **11,742 priced outer nodes**, before source
   owners, prefix plumbing and template tables. This may preserve 65,536 rather
   than raise it. Each count assumes one/two full-input gathers and a bounded
   operator loop; implement and count the actual graph before fixing a limit.
4. **Separate immutable program bytes from proven peak live owners.** Even
   looped public matrices retain 7.379 GB of the current cumulative charge across
   D9 selections; changing node granularity cannot make that charge a real peak.
   Keep a bounded compiler-construction ledger, immutable metadata ledger and
   retirement-derived runtime frontier plus primitive/compound/copy/frame/import
   overlap. Enforce their combined request-memory envelope at every worker.
   Do not relabel the cumulative number as RSS or simply raise 64 MiB to 32 GB.
5. After a complete graph exists, an **explicit larger decoder capacity profile**
   must carry its exact admitted work/AES/cursor, metadata, live-memory, loop,
   queue and time envelopes, bound into Experiment, manifest and program digest.
   Current fixed-shape all-guard requirement is 76,197,495 ordinals and
   7,316,939,414 AES blocks; these are audit inputs, **not proposed final caps**.
   Ranges, final guard placement and scheduling can change them. Do not round to
   an arbitrary power of two or infer feasibility from a larger numeric ceiling.
6. Resolve the hard nonlinear-body link bound above before pricing a purported
   1,200-second full 64+8 serving profile. Increasing only lifetime is a different
   execution contract, not a communication optimization. Then calibrate admitted
   generated primitives/loops using ordinary SDK and benchmark
   hooks, including all-direction bytes, controls, dependent phases, CPU/AES and
   simultaneous RSS. Only a complete reconciled time envelope can justify a
   lifetime change or retain 1,200 seconds. CPU/public-MAC requirements from the
   baseline below remain; 108.94-GB matrix reads alone need 90.78 MB/s within
   1,200 seconds. Unknown dominant throughput/latency is not zero.

### Exact larger-profile change map and non-negotiable constraints

No changes listed here have been authorized or implemented by this audit.

| File/symbol | Required audit/change, if a distinct larger profile is approved |
| --- | --- |
| `puma/mod.rs::{Limits::program, MAX_PROGRAM_OPERATIONS}`; `prss.rs::{PrssLimits::program, MAX_PROGRAM_AES_BLOCKS}`; `work.rs::WorkBudget::limits` | Add explicit profile-bound construction using complete graph counts; preserve original defaults. Reconcile per-party upper AES and all three cursors. No `Limits::new`/primitive default increase. |
| `puma/prss.rs::{fill_typed,SourceParty}`; `puma/mod.rs::Operation::digest` | Keep u64 ordinal and cursor additions checked, no wrapping/reset/reuse. Session/program/phase/ordinal/source/pivot/shape/consumer bindings stay intact. Call-reset LE-u64-to-AES-128 counter layout and discarded fill tails stay source-pinned. |
| `tensor.rs::{Program,Executor,MAX_NODES,MAX_TILE_LANES,MAX_ROW_LANES}` | Keep 192 instructions, u8 Value/register slots, 128 element tiles, 1,024 row lanes and primitive ledger bounds. Add outer reusable loops rather than widening inner namespaces. |
| `replicated_decoder.rs::{Code,Node,MAX_NODES,MAX_VALUES,MAX_PROGRAM_BYTES}`; `program.rs::{Builder::push,construction_views,compile_with_policy}` | Add explicit loop descriptors, checked iteration/retirement proofs and separate immutable/live/construction budgets. Measure unique interned payload, gather/Words capacity, certificates, retire tables and source descriptors. Do not count only Node headers. |
| `lookup.rs::Plan`; `selection.rs::Plan`; new row-loop plans | Bind geometry/program profiles into digests; count fresh repetitions; bound input copies, selector/index arrays and active primitive overlap. Preserve eight-row lookup / 64-comparison tiles unless separately repriced. |
| `program.rs::lower_with_weights`; `sdk.rs::{validate,source_contract}` | New explicit profile may support 64 input/8 output and rows+outputs 72 (largest executed prefix 71). Existing 32-input/40-total defaults remain. Model layer/head/vocab expansion caps already cover common dimensions. |
| `source.rs::{Manifest,Publisher,finish}`; `public_source.rs::PublicSourceImport`; worker `finish_frame` | Bind the capacity identity before replay/import. Common has 494,032,768 source values, below 536,870,912; table rows 151,936, below 262,144. No source cap increase shown necessary. Keep source digest and row-certificate recomputation. |
| `runtime.rs::{Worker,advance_inner,finish_frame}` and PyO3 `ReplicatedWorker` | Prove max owners/copies/queues and all loop counters, enforce shared request ledger, preserve cancellation drain before refund. Checked final result is 2+1 words (97 B) or 8+1 words (145 B), including PUR1 header; u32 word count is ample. Keep exact session/program/party/count and client validity check, not a new disclosure path. |
| Python `replicated_protocol.py`; `benchmark_memory._replicated_memory`; worker admission | Retain 128-KiB app frames, 1-MiB admission and 16 pending frames unless exact encoded envelopes prove otherwise. Request reservation currently 192 MiB + 8 MiB for streams. Derive new live bound before changing reservation; standard memory admission must see the same profile. |
| Python worker `run`, `_Attempt.expires`; `replicated_peer.PeerStream` | Current 2^20 outer/1,024 inner barriers and 1,200 seconds remain independent gates. Derive outer advances and maximum dependency wave for new loops; an ordinal ceiling does not bound these. Stream sequence <2^63, one in-flight request, queue 1, original frame limit, ordered acceptance and shutdown drain stay checked. |
| `delegated_client.DelegatedClient`, `delegated_transport._Peers`, worker HTTP clients, `PeerStream` | Ordinary SDK timeout defaults to **300 seconds**; the internal direct client defaults to **60 seconds**. Both request clocks start before publisher rescan. Worker lifetime is a separate 1,200-second bound, starting after native compile and before worker source import. Peer HTTP/open timeouts are 60 seconds; stream close timeout is five seconds. Reconcile CLI/gateway/benchmark timeout, worker expiry, import and cleanup; do not change only one. |

**Hard mathematical/namespace constraints:** ring64/Q18/quarter-gap product
domains; valid truncation shift; nominal Boolean width <=64; no secret-dependent
schedule; fresh sessions and one-use correlated material; checked non-repeating
u64 ordinals/PRSS namespaces; fixed format/index widths. u64 cursor exhaustion
is a hard API bound. AES-128's security budget requires separate cryptographic
review: neither 3.224 billion nor the candidate 7.317 billion is a proved
cryptographic threshold. In contrast, 1M ordinals, 65,536 outer nodes, 64 MiB,
192 MiB, barrier counts and 1,200 seconds are conservative resource policies,
not mathematical evidence of security or performance. Changing policy must not
weaken numeric/domain checks, widen unchecked fields or replay consumed material.

Source import additionally owns 1,976,131,072 logical FP32 bytes per worker pass.
Q18 files excluding inline norm coefficients occupy 3,951,910,912 bytes per
worker; conservative SDK reservation including those coefficients is
3,952,262,144 bytes. Matrix/bias/norm import has 120,717 bounded 4,096-value
chunks per worker. Source gamma vectors are 43,904 inline Q18 values; retain
their serialized manifest and interning/public-array memory costs. Exact
manifest bytes depend on actual coefficients and must be checked against 1 MiB
after wheel freeze. Source-present imports are local authenticated file work,
not inferred peer traffic. Each public matrix product has fewer than 2^32 terms
at common geometry; 1M values per secret owner and 1,024 primitive lanes remain
sufficient individually. None establishes the combined live-memory envelope.

### Acceptance gates before any capacity increase or checkpoint execution

* Frozen source-independent loop tests at 4+2, 16+2, 64+8: native schedule totals
  equal sum of per-operator/iteration WorkBudgets; odd/tail tiles and separate
  PRSS fill rounding reconcile; no counter narrowing or wrap near u64 limits.
* Loop-vs-expanded admitted tiny integer oracle traces agree at every hidden
  boundary with coupled/randomness-aware treatment; full vocabulary stable ties,
  feedback, guard-before-unsafe-work and final validity acceptance remain intact.
* Legacy profile serialization/digests/default limits unchanged; explicit new
  profile rejects missing/mismatched worker/SDK/manifest identity. Reject one
  ordinal/AES byte/node/value/frame/queue over each bound before allocation/draw.
* Exact immutable/construction/live owner peaks checked at 71-row prefix with
  template interning both hit/miss cases; retire and cancel every iteration,
  source import, active primitive, slow-peer queue and final-result owner.
* New compact codec measured separately from the frozen 60,000-byte sample;
  six directed counters, native encoded frames, dependent waves, controls and
  whole-attempt time reconcile in the standard benchmark report. Validate
  cross-session substitution, replay, truncated/padded frames and backpressure.
* Shared SDK, gateway, CLI, worker and benchmark timeout/memory/profile checks
  agree; deadline expires mid-import, mid-row, mid-stream and final acceptance
  without refunding live native/Python/file owners. No wheel or checkpoint run
  until the integrated profile audit and wheel freeze gate.

### Serialized SDK/benchmark hook: exact interface

```rust
cost::common_shape_diagnostic_json(rows: usize, selections: usize)
    -> Result<Vec<u8>, String>
cost::admitted_diagnostic_json(program: &Program)
    -> Result<Vec<u8>, String>
```

Both emit `schema: "pllm.replicated.cost-diagnostic.v1"`. Shape scope is
`common_qwen_shape_only`, contains separately identified `baseline` and
`candidate_guard256`, and sets `executable=false, admission_evaluated=false`.
It can explain a rejected hypothetical geometry, but cannot mint a successful
SDK source contract or worker manifest. Admitted scope is
`admitted_native_program`, contains program digest/source-descriptor binding
and real native schedule counters. It does not silently associate an arbitrary
admitted program with common Qwen. Main can expose these byte APIs at the
existing PyO3 inspection boundary and attach parsed dictionaries to ordinary
SDK/benchmark diagnostics. No paper-only driver or alternate execution endpoint.

`admitted` prices CheckedRange, PublicLookup and Selection exhaustively. Bulk
scans are one per **eight-row plan**, not one per whole prefix; interned lookup
program pointees are counted once per allocation using the plan's metadata API.
Selection includes full index ownership, its real plan/predicate workspace,
all negative-tensor invocations and combined value/ID product batches. Tensor
instruction counts include its fixed Sign/FromBit factory, not a guessed zero.
Runtime logical source reads include scalar public gathers/bias reads as well
as matrix/bulk scans; import and physical file traffic stay separate. Tensor
scratch, compound workspace and retained-owner frontier remain separate fields.
Whole peer frames/rounds/bytes remain null without complete measured scheduling.

Validation after the parent hook landed:
`CARGO_BUILD_JOBS=1 cargo test -p pllm-compiler replicated_decoder::cost::tests --lib`
— **7 passed** in working tree, **0.89 seconds** test time; final compile emitted
no warnings. Tiny admitted accounting
now exercises CheckedRange/PublicLookup/Selection and reconciles work by code,
logical reads, selected products, plan/workspace owners and explicit unknown
peer counts. Direct instruction counts use lookup's immutable programs API;
selection uses its invariant two-instruction negative factory and plan tile
count. Native start-step counts are calibrated by the admitted three-party test.
The existing baseline lookup/node/link identity is regression-tested.
Only `cost.rs` and this document changed in this continuation; no caps, other
Rust/Python code, wheels, checkpoints or commits changed.

## Public source lifecycle

Public-mode requests send source manifests, gamma coefficients and fresh private
token shares. They do not upload every weight. The client rescans the locked
checkpoint; each worker independently verifies and rebuilds request-owned Q18
files from its local source. No verified cross-request Q18 cache is installed.

The common geometry needs 3,951,910,912 private-file bytes per worker. The client
and three worker imports together require at least 23,712,519,168 logical local
read/write bytes, before online execution and additional initial source-lock
passes. Page-cache hits can avoid physical reads; these are byte floors, not
measured disk latency or RSS. Missing-worker checkpoint distribution remains a
separate cost. Ordinary SDK timeout is 300 seconds, versus 60 seconds for the
internal direct client; both request clocks begin before publisher rescan. Worker
lifetime starts after native compilation and before source import, so serial
admission consumes the earliest worker's remaining time.

Exact owner paths, framed-publication formulas, independently executable byte
calculations and remaining cancellation/cache gates are retained in the
[source lifecycle audit](../artifacts/puma-sdk/local-v12/source-lifecycle/README.md).
This accounting supplies no new source cache, deadline policy or execution result.

## Archived baseline: original decision

**No-go for complete-checkpoint execution under the current execution contract.**
This is a shape/schedule result, not a performance measurement. No checkpoint was
loaded and no numeric/capacity limit was bypassed. Continue only bounded native
engineering and generated-fixture calibration, then repeat P0 before P3.

* **4+2:** numeric admission remains blocked by the complete-model range proof.
  Included-work ordinal floor is 350,230, but nonlinear/checked-range work,
  full metadata ownership and complete peer traffic are still unknown. The
  30.63-second lookup-A2B link floor is not a feasibility pass.
* **16+2:** at least 1,252,918 ordinals, exceeding the 1,048,576 ceiling even
  before omitted nonlinear/guard work. Current lookup alone already exceeds it.
* **64+8:** at least 20,939,992 ordinals. Existing protected argmax trees alone
  require at least 115,200 outer nodes, exceeding 65,536. Lookup A2B alone needs
  at least **1,837.82 seconds on each 40-Mbps rotation edge**, beyond the
  **1,200-second attempt lifetime**. Bulk public file scans and long-lived peer
  delivery cannot remove that payload floor if the same predicate is retained.

These bounds identify real blockers; unknown dominant costs are not passes.
The approved checked-range implementation adds fresh products, comparisons,
validity ownership and final acceptance; none is priced as free here.

## Scope and reproducible diagnostic interface

`crates/pllm-compiler/src/replicated_decoder/cost.rs` provides:

```rust
cost::estimate(cost::Dimensions::COMMON_QWEN, input_rows, selections)
cost::admitted(&already_admitted_program)
```

Both return serializable, owned public diagnostics, not an executable plan.
`estimate` does not call `lower_with_weights`, source publication or admission.
It accepts the study's hypothetical 64+8 geometry for diagnosis without admitting
that geometry. It uses bounded arithmetic and bounded public primitive queries.
`admitted` uses the actual program's WorkBudget, tensor loop multiplicities,
instruction count, program payload and retained-owner retirement schedule. Its
peak retained-secret bytes exclude tensor scratch, primitive operand copies,
exported frames and queues; `largest_tensor_admitted_bytes` is separate. New
native operators not yet source/MAC-priced increment `unpriced_native_nodes`.
Complete frame/round/byte fields remain null, not inferred from ordinals.

**Parent integration hook (main-owned):** add `pub mod cost;` in
`replicated_decoder.rs`. Add SDK/PyO3 diagnostic serialization using the existing
native boundary. Attach shape diagnostics to SDK inspection/admission failures,
separately from `source_contract` success and executable identity. Attach
`admitted` diagnostics only after normal complete-program admission. Do not use
an estimate as a successful source contract or pre-authorized resource increase.
No parent/program/SDK file is changed by this workstream.

Only core API change: `WorkBudget::rescale(lanes)` becomes public, read-only and
documented. It returns the same checked estimate; it neither issues PRSS material
nor changes `limits()` or any numerical/capacity admission.

Profile `pllm.puma.shape-cost.pre-bulk.v1` deliberately names the audited original
128-lane equality schedule. Bulk lookup, predicate narrowing or schedule templates
must get separately priced diagnostics, not silently reuse baseline numbers.

## Source and schedule assumptions

`examples/papers/common.py` pins
`Qwen/Qwen2.5-0.5B-Instruct` revision
`7ae557604adf67be50417f59c2c2f167def9a775`, with 64 input/eight output bounds.
Metadata-only regression checks dimensions against the existing
`paper/artifacts/puma-sdk/local-v4/qwen-metadata-replay.json` normalized config:
24 layers, hidden 896, intermediate 4,864, 14 query heads, two KV heads,
64-dimensional heads, vocabulary 151,936, tied embedding/head source.

D9 recomputes prefixes 4/5, 16/17, or 64 through 71. Recomputed rows are 9, 33,
and 540. Final RMSNorm covers every row before LastToken; the head covers one
last row per selection; protected stable argmax compares all vocabulary entries.
All 24 layers are charged on each prefix. Public products use both replicated
components. A ring64 multiply-add is counted per coefficient per component,
not a FLOP and not a secret MPC multiply.

### Included native schedule costs

| Included category, ordinals | 4+2 | 16+2 | 64+8 |
| --- | ---: | ---: | ---: |
| Lookup equality, original 128-lane tiles | 299,124 | 1,096,788 | 17,947,440 |
| All body public-linear output rescales | 21,384 | 78,408 | 1,283,040 |
| RMSNorm square, normalize, mean/gamma rescales only | 1,764 | 6,468 | 105,840 |
| Secret causal attention scores and values | 6,048 | 22,848 | 959,616 |
| Rotary: one rescale per coordinate only | 1,728 | 6,336 | 103,680 |
| Gate/up elementwise products | 8,208 | 30,096 | 492,480 |
| Complete head output rescale | 2,374 | 2,374 | 9,496 |
| Complete stable argmax comparisons and two selects | 9,600 | 9,600 | 38,400 |
| **Included-work ordinal lower bound** | **350,230** | **1,252,918** | **20,939,992** |
| Included-subset per-party AES upper allowance | 51,871,718 | 181,128,614 | 2,940,304,568 |

Not included: RMSNorm rsqrt/polynomial/comparison schedule; softmax and score
scaling; guarded SiLU; additional rotary operations; new checked-range guards
and sticky/final validity. Report lists their invocations and lanes explicitly:

| Unfinished category inventory | 4+2 | 16+2 | 64+8 |
| --- | ---: | ---: | ---: |
| RMSNorm row invocations | 441 | 1,617 | 26,460 |
| Softmax causal row invocations | 3,024 | 11,088 | 181,440 |
| Softmax valid score lanes | 8,400 | 97,104 | 6,221,376 |
| Guarded SiLU element lanes | 1,050,624 | 3,852,288 | 63,037,440 |

**AES column is not a whole-program lower bound or a successful AES admission.**
WorkBudget conservatively bounds each party over public pivots. Included ordinals
are exact for the named subset; adding omitted work only increases the ordinal
count. Local operations may allocate nodes/registers but consume no WorkBudget
ordinal, or consume local ordinals inside tensor programs. Neither case permits
equating ordinals, primitive invocations, gate phases, peer frames or HTTP calls.

### Public CPU and file owners

One layer has `2H² + 2H(KV*D) + 3HI` public matrix coefficients.
Let `W` be this count across 24 layers, `T=V*H`, `R=sum(prefix)`, `S=selections`.
Per-worker public MACs are `2*(R*T + R*W + S*T)`. Secret attention additionally
requires `2*24*14*64*sum(n*(n+1)/2)` scalar dot-product terms, before its native
mask/rescale/copy work. No secret attention throughput is inferred from public MACs.

| Per worker, decimal units | 4+2 | 16+2 | 64+8 |
| --- | ---: | ---: | ---: |
| Lookup public ring64 MACs | 2,450,423,808 | 8,984,887,296 | 147,025,428,480 |
| All body public ring64 MACs | 6,440,878,080 | 23,616,552,960 | 386,452,684,800 |
| Complete heads public ring64 MACs | 544,538,624 | 544,538,624 | 2,178,154,496 |
| **All public matrix ring64 MACs** | **9,435,840,512** | **33,145,978,880** | **535,656,267,776** |
| Additional secret attention dot-product terms | 1,075,200 | 12,429,312 | 796,336,128 |
| Original logical matrix reads, bytes | 17,705,074,688 | 43,842,928,640 | 619,715,231,744 |
| Optimistic bulk matrix reads, bytes | 10,081,533,952 | 10,081,533,952 | 40,326,135,808 |

Original reads are `8*(R*T + S*(W+T))`: original lookup scans once per private
row; body matrices and head once per selection. The same read coefficient feeds
both replicated components, so MACs have factor two but file reads do not.
Optimistic bulk reads are `8*S*(W+2T)`: one embedding scan per selection. Tied
head still needs another logical scan. Row tiling may require more scans; this is
an optimistic read target, not a guarantee of a particular bulk implementation.
Bulk scan reuse does not remove the public MACs or fresh private predicates.

Tied public matrix snapshot is **3,951,689,728 bytes per worker**, excluding
norm/bias vectors. Source import FP32 bytes, Q18 output snapshot bytes, hashing
reads, transient input chunks and three separate worker copies remain separate
owners. Snapshot bytes are not simultaneous RSS. Logical reads are not physical
disk reads, source delivery bytes, network frames or physical wire bytes.

To finish within 1,200 seconds, even with all other costs zero, per-worker public
MAC throughput must exceed **7.86 million/s**, **27.62 million/s**, or **446.38
million/s** respectively. At 64+8, original logical-read throughput must exceed
516.43 MB/s; optimistic bulk target requires 33.61 MB/s. These are required
rates, not measured achievable rates, and cannot establish CPU/lifetime admission.

### Tensor and metadata owners

Largest shape-visible outer secret owner is at least 151,936 values (2,430,976
bytes, replicated pair) for 4+2/16+2; at 64+8 the 71-by-4,864 activation is
345,344 values (5,525,504 bytes). These are individual-owner lower bounds,
not request-memory totals. Original lookup additionally owns the vocabulary
selector; a tiled bulk path must price selector tile, accumulated embeddings,
source buffer and primitive overlap. Largest tensor scratch and all simultaneous
owner copies need the admitted report plus runtime calibration.

Protected argmax emits 1,200 nonempty comparison batches per selection. Each
batch emits 12 outer nodes: value gathers/difference/negative, two arithmetic
selects and ID gathers. This alone accounts for 28,800 nodes at 4+2/16+2 and
115,200 at 64+8, excluding every body/lookup node and public indices owner.
At the audited native Node layout (176 bytes), node headers alone are 5,068,800
and 20,275,200 bytes. This layout is target/build-dependent; report derives it
with `size_of`, not a serialization-size assertion. Gathers/Words payload,
tensor metadata, interned programs, retirement vectors and compilation views
remain additional. A bounded iteration/template route could remove duplicated
metadata, but must still charge fresh work for every selection.

## Directed link bound and lifetime

`puma/conversion.rs` Bits conversion uses eight rotations. On **each** directed
cycle edge (`party -> party.next().next()`), its payload is:

```text
split:                 8*n bytes
first 64-bit AND:       8*n bytes
six 2*n 64-bit ANDs:   96*n bytes
total:                112*n bytes
```

These are payload bytes only; every tiled invocation has eight frames per cycle
edge. The original equality invokes full Bits conversion for every token/vocab
pair. Boolean ORs, B2A and frame/authentication/transport overhead are excluded.
The opposite three directed edges are not assigned this same bound. No division
by six, multiplication by three into one link, or conversion from ordinals is used.

| Lookup A2B only | 4+2 | 16+2 | 64+8 |
| --- | ---: | ---: | ---: |
| Payload bytes on each cycle edge | 153,151,488 | 561,555,456 | 9,189,089,280 |
| Frames on each cycle edge | 85,464 | 313,368 | 5,127,840 |
| Original sequential A2B waves | 85,464 | 313,368 | 5,127,840 |
| `8*bytes/100,000,000`, seconds | 12.25 | 44.92 | 735.13 |
| `8*bytes/40,000,000`, seconds | 30.63 | 112.31 | **1,837.82** |

At 1,200 seconds, one directed 100-Mbps link carries at most 15 GB and one
40-Mbps link at most 6 GB. A 100-Mbps-only screen does not reject 64+8 on this
subset; a 40-Mbps constrained directed cycle link does. Under 100-down/40-up
access, each outgoing cycle edge already encounters the 40-Mbps bottleneck.
Concurrent traffic sharing an access link can only worsen these optimistic floors.

The wave count describes dependencies of the **original serial executor**,
not an invariant for a new batched schedule and not an HTTP-call count. Existing
HTTP run loops perform all-role advance/prepare/dispatch barriers with a 2^20
outer ceiling and 1,024 inner-wave ceiling. Local tensor instructions can consume
ordinals without peer waves; one conversion consumes one ordinal but eight
rotations. Long-lived streams reduce control latency, not these payload bytes.
Complete round/HTTP bounds remain unknown until final executor schedule is priced.

## Capacity inventory and ordinary measurement hooks

Audited baseline caps (not authorizations to raise them):

| Owner/check | Baseline cap |
| --- | --- |
| SDK requested input / output / request slots | 32 / 8 / 64 |
| Compiler `rows + outputs` | 40 |
| Common paper request envelope | 64+8; 32 requests |
| Complete-program ordinals | 1,048,576 |
| Complete-program per-party AES allowance | 3,224,371,200 blocks |
| Outer native nodes / program storage | 65,536 / 64 MiB |
| Individual secret owner | 1,048,576 values |
| Public source values | 536,870,912 |
| Individual public product shape | at most 2^32 row*inner*columns terms |
| Public source read buffer | 65,536 bytes |
| Peer app envelope / pending frames | 128 KiB / 16 |
| Worker request memory reservation | 192 MiB, plus any separately priced stream reservation |
| HTTP admission envelope | 1 MiB |
| Attempt lifetime | 1,200 seconds |
| Original barrier loop / inner dispatch waves | 2^20 / 1,024 |

Use existing `benchmark_delegated._inspect_replicated`,
`benchmark_memory._replicated_memory`, `benchmark_targets` and response reports.
Add native shape/admitted cost dictionaries to their ordinary SDK contract
extension, with profile, source pins, numeric/range identity and explicit
estimate-vs-measurement scope. Keep `peer_link.<source>.<destination>.<phase>_bytes`
directed accounting and worker `peer_counts` frames/controls separate. Counters
currently track serialized application bodies; TCP-stream/physical-wire fields
stay unknown. Do not infer a network round from one control/frame/ordinal.

The audited `examples/papers/puma.py` still calls `PumaThreePartyCpu(model)`
without an explicit public-weight composition. Main must select the qualified
public/range/lookup/transport identity before any common paper serving campaign;
changing the recipe does not establish numeric or resource admission.

Calibrate admitted generated cases through the normal SDK/benchmark path: match
native ordinals/AES/tensor invocations with actual frames/waves and six directed
byte counters; include startup/import/cleanup and publication/client/worker costs.
Report simultaneous topology RSS, per-role CPU, snapshot files and failure-owner
retirement. Repeat fresh-process response samples using standard replicate hooks.
No separate PUMA benchmark driver, wheel build, real-checkpoint load or large
parallel benchmark was used for this P0.

## Conditional post-optimization route

1. Finish complete checked-range/numeric schedule; count every guard and final
   validity operation under distinct identity. No execution without admission.
2. Bounded bulk lookup reduces reads toward the 40.33-GB 64+8 matrix-read target,
   but leaves 535.66 billion public component MACs and current A2B payload floor.
3. To preserve 1,200 seconds under 40 Mbps, **change actual predicate/communication
   work**, not only granularity. Public token domain has 18 useful bits
   (`151936 < 2^18`); narrower modular predicate protocols require proof over the
   entire admitted input domain, source PRSS-tail accounting and independent tests.
   Retaining the existing full-64-bit A2B cannot qualify 64+8. Even ignoring all
   other work requires reducing A2B bytes by at least 34.71%; real required
   reduction is larger. No narrower protocol or achievable throughput is claimed.
4. Batch independent primitive phases and use bounded peer streams to reduce
   latency/control overhead; preserve ordering, backpressure and retirement.
5. Bound repeated argmax/body metadata with native iteration/templates, then
   reprice all node/value/storage/AES/queue/timeout limits together. Existing
   1,048,576 ordinal capacity is not adequate even for known 16+2 work.
6. Calibrate dominant CPU/AES/network costs using admitted tiny/scaled generated
   programs through ordinary benchmark hooks. Require a complete upper envelope
   within the selected lifetime; otherwise stop or explicitly renegotiate the
   execution contract. Unknown dominant costs cannot authorize P3.

## Validation evidence

Five focused native tests cover metadata pins, all three D9 geometries, complete
body/head/argmax charging, 40-Mbps lifetime rejection without ordinal/round
confusion, invalid dimensions and admitted tiny-program owner/work reconciliation.
Command:

```sh
CARGO_BUILD_JOBS=1 cargo test -p pllm-compiler replicated_decoder::cost::tests --lib
```

Result: **5 passed**, in 0.07 seconds test time (debug build). Because parent
module integration belongs to main, this was run in a temporary source mirror
with only the `pub mod cost;` hook added there; original parent was not edited.
This is native component evidence, not installed-wheel or whole-Qwen evidence.
Re-run the same command in the main tree after the parent hook/final operators
land. Final full-program AES, exact whole-request frames/rounds, CPU calibration,
source-import peak, and lifetime feasibility remain unestablished.

Working-tree regression:
`CARGO_BUILD_JOBS=1 cargo test -p pllm-core puma::work::tests --lib`:
**2 passed**. Original independent ordinal/AES ceilings and checked composition
remain enforced after exposing the read-only rescale estimate.
