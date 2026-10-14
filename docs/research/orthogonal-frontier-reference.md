# Five orthogonal client-resource investigations

This screen investigates five mechanisms against the existing exact prepared
stack. They target scheduling, persistent state representation, attention scratch,
Preparation kernels, and local output-head search. These are project research
hypotheses, not claims of priority in the literature or measured serving gains.

## Completed screen and next decision

**Attention lifetime tiling is the strongest implementation candidate. No new
Low/Medium serving SOTA is established.** Five independent Sol 6.1 investigations
produced bounded references; parent validation adds serial fresh-process attention
measurements and a source-locked four-tensor Preparation scan.

| Direction | Evidence obtained | Decision |
| --- | --- | --- |
| [Attention lifetime tiling](exact-attention-streaming-reference.md) | 144 small bitwise comparisons; 36 fresh Qwen-shaped attention workers preserve output hashes. At 16 query rows under `prefix_f32`, isolated process peak RSS falls 6.02% / 16.30% at 150 / 288 inputs. | First implementation gate: compiler-bound exclusive region retaining original reductions, strides, capacity and full output. Whole-client gains remain unknown. |
| [Cross-stage MLP conveyor](mlp-conveyor-reference.md) | Exact synthetic tile outputs and one-use toy retirement; shared-link simulation keeps bytes unchanged. A second tile64 slot adds 26.36 MiB to the conservative region bound. | Second priority, with tile16 as a separately priced candidate. Real channel multiplexing, public scheduling contract and whole-client admission precede timing. Simulated gains are not measurements. |
| [Exception-split Preparation](exception-corrections-reference.md) | 180 synthetic correction outputs match. Four real Qwen down tensors have 68.61–75.65% b4 exceptions; all tested packed-base/CSR layouts exceed original i8 storage. | Deprioritize the sparse-tail premise. b7 retains a narrower 20-to-12-call hypothesis, but adds exception work and workspace; no speed result. |
| [Quantization-origin KV](projection-kv-reference.md) | Tiny reconstructed KV and installed attention match original bits. Pages plus captured RoPE coefficients cost 1,208–1,210 bytes versus 896 dense bytes before profile overhead. | Defer pending sanctioned origin capture and checkpoint-wide storage accounting. FP32 snapshots cannot supply missing integer provenance. |
| [Certificate-indexed local head](head-certificate-index-reference.md) | Separated fixtures skip 56/64 rows with every skip independently checked; random boxes skip none. | Blocked by current complete-logit and sampling/state contracts. A safe argmax certificate alone cannot replace the existing head. |

The attention screen uses synthetic values at Qwen geometry, **not checkpoint
attention**, and times only the operator. Sixteen-row `prefix_f32` CPU medians
change by -1.97% / -1.63% over roughly 50–130 ms windows; these small samples do not
establish a CPU speedup. Four-query-row/one-KV-head tiles increase CPU 8.69% / 5.83%. Dense-mode
results have a separate numeric identity. All 36 workers report zero observed
swap growth. Raw cases, build identity and replay are retained in
[`exact-attention-streaming-qwen-shapes.json`](../evidence/exact-attention-streaming-qwen-shapes.json)
and the linked reference.

The Preparation scan visits only public source weights, quantizing with the
installed W8 per-row implementation. It does not execute candidate corrections,
time a native kernel or measure a client. Its source lock comes from independent
retained evidence; the four tensors, hashes, histograms and complete modeled
owners are in
[`exception-corrections-qwen-down-screen.json`](../evidence/exception-corrections-qwen-down-screen.json).
No screen establishes a fresh-request all-link traffic reduction. No candidate
executes a complete provider-backed decoder or measures its remote-MAC fraction.

Parent validation passes **169 focused tests**, Ruff, and the 435-page docs-content
check. The synthetic head evidence replays exactly. Source/build identity and
evidence checks preserve historical measurements; no SDK component is promoted.

The subsequent [compiler-bound attention region](attention-tiling-reference.md)
now passes generated complete-logit/KV, restored-continuation, prepared-child and
gateway checks, with explicit failure retirement. Its fixed complete-response
benchmark selector charges state auditing in both arms. Eighteen fresh pinned
Qwen2.5 288+8 clients preserve 524,973,312 checked logit/KV values with zero
observed swap growth. Attention tiling lowers median client RSS 2.41% atop MLP
tiling and 1.06% atop the combined-placement stack, but digit8 alone regresses
4.93% and exceeds the observed Low threshold. Timing and traffic are nearly
unchanged; 150-token digit8/placement comparisons remain host-admission blocked.
The separate six-client 150+8 integer-Preparation/MLP-tiling pair preserves
98,682,624 checked values with zero observed swap growth. Median client RSS falls
0.75%, but two paired samples regress and the largest peak rises; cold latency
is effectively unchanged. The two workload cohorts are not pooled.
The original isolated screen retains its scope, and independent confirmation
remains required. No Low/Medium SOTA improvement is established.

## Common comparison contract

The whole-client maxima are **268,435,456 bytes (Low)** and
**2,147,483,648 bytes (Medium)**, from
`python/pllm/search/client_memory_tiers.json`. A candidate must preserve the
selected source, numeric and privacy contracts and at least 90% remote body
linear MACs, checked independently in prefill and decode. Local head work,
attention, reconstruction, metadata, transport and runtime overhead still count
toward client CPU and memory.

An isolated working set below a tier ceiling does not qualify the client. The
largest process-lifetime peak across all required fresh-client replicates must
fit, and conservative runtime/whole-topology admission must also pass. Historical
peaks are not spare-allocation reservations for a new implementation.

### Historical comparator

The completed [five-way combination screen](sota-combinations-reference.md)
contains three fresh-client samples per choice and workload, with cold and exact
repeat requests in each process. It uses pinned Qwen2.5-0.5B W8A8 `prefix_f32`,
150/288 input tokens plus eight outputs, native shared 100/40 Mbps links, and
separate co-located Preparation/Inference children. Its independent finalist
confirmation remains open. The incomplete six-recipe
[low-memory screen](low-memory-combinations-reference.md) cannot supply missing
replicates or a new incumbent.

The existing scorecard resolver yields these **largest**, not median, client
peaks from `docs/evidence/research-sota-combinations-qwen25.json`:

| Workload | Choice | Largest client peak, bytes | Observed Low eligibility |
| --- | --- | ---: | --- |
| 150+8 | Control | 223,182,848 | Yes |
| 150+8 | Tiling | 202,276,864 | Yes |
| 150+8 | Digit8 | 225,492,992 | Yes |
| 150+8 | Tiling + digit8 | 202,702,848 | Yes |
| 150+8 | Tiling + digit8 + placement | 202,293,248 | Yes |
| 288+8 | Control | 284,639,232 | No |
| 288+8 | Tiling | 244,318,208 | Yes |
| 288+8 | Digit8 | 281,313,280 | No |
| 288+8 | Tiling + digit8 | 242,483,200 | Yes |
| 288+8 | Tiling + digit8 + placement | 243,499,008 | Yes |

All ten are observed Medium-eligible in that historical cohort. Low's longer
placement stack has only 24,936,448 bytes between its largest recorded peak and
the tier ceiling; this arithmetic is not admission headroom. The new probes use
their own build and evidence identities and cannot inherit these resource results.

Source report SHA-256:
`5fb73b6883a0e42ac42b65b33a84cdfe0c6646d1e459f0df72639c38756ad54d`.
Resolve the table through the existing read-only scorecard implementation:

```sh
node --input-type=module -e 'import {buildScorecard} from "./docs/scripts/research-scorecard.mjs"; for (const c of buildScorecard().cohorts.filter(c => c.id.startsWith("qwen25-sota-combinations-screen") && c.identity.cold)) console.log(JSON.stringify({id: c.id, peaks: c.rows.map(r => ({label: r.label, ...r.clientMemory}))}, null, 2));'
```

## Investigation contracts

| Mechanism | Intended objective | First falsification gate |
| --- | --- | --- |
| Cross-stage MLP conveyor | Recover tiled prefill latency while bounding live tiles | A legal public wavefront must overlap useful work under shared links and serialized kernels, with exact outputs and fresh disjoint tickets. |
| Quantization-origin KV | Reduce exact persistent client state | Integer projections, scales and original operator replay must reproduce every stored float32 bit; all reconstruction and conversion owners count. |
| Reduction-preserving attention streaming | Reduce prefill temporary memory and allocation work | Query/head slicing must preserve the actual reduction layout, causal domain and every result bit for admitted operators. |
| Exception-split Preparation kernels | Reduce correction-generation CPU | Narrow-base work plus exception handling, copies and setup must beat the existing native digit8 implementation. |
| Certificate-indexed local head | Reduce greedy head CPU and paged reads | Conservative bounds must safely skip enough complete blocks after index work and required full-logit fallbacks. |

Each direction keeps its own probe, correctness tests and evidence. Checkpoint
scans and fresh-process measurement cohorts run serially through the existing
frontier lock, memory admission and swap guard. A blocked numeric or cost gate
is a result, not permission to weaken a contract.

## Serving promotion gates

1. Independent integer or bitwise float32 correctness, with boundary and failure
   cases, must precede performance interpretation. Greedy head selection alone
   does not implement complete logits, arbitrary sampling or snapshot contracts.
2. Native/compiler ownership and any changed schedule, state representation or
   selection contract must be bound and admitted before ordinary SDK execution.
   A local synthetic state machine does not establish authenticated transport or
   production one-use retirement.
3. A source-locked generated decoder must preserve complete applicable logits,
   active KV and continuation behavior, including cancellation and restoration.
4. Fresh-client matched controls must charge setup, all live owners, fallback
   work, public artifacts and every directed traffic link. Isolated CPU, payload
   and scratch results remain separate from whole-response metrics.
5. Independent confirmation must cover both tier-eligible incumbents and the
   selected workloads. Client RSS/CPU, Preparation CPU/RSS, TTFT, N−1 decode
   throughput and all-link traffic remain separate objectives. Base,
   output-dependent and unattributed application bodies retain their original
   conservation rules; unknown measurements stay unknown.

The five mechanisms can interact through memory, cache locality and scheduling.
Their isolated gains cannot be added or multiplied into a combined SOTA claim.
