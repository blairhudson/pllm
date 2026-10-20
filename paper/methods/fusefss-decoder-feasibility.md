# FuseFSS complete-decoder feasibility and reusable execution seams

Status: **conditional static inventory, not model admission or a no-go theorem**.
Reviewed 20 October 2026. No tensors, native build, wheel, checkpoint, benchmark,
cap change, or author-code execution. Prior evidence remains unchanged.

## 1. Decision and evidence boundary

Committed F2 supplies real bounded source gates, not a complete decoder. Its
streamed material can be large cumulatively while live ownership stays bounded.
The costs below expose expensive current scalar execution, **not proof that an
exact admitted decoder is impossible**. No complete GPT-2 or Qwen numerical,
compiler-memory, transport, or elapsed-time admission exists in this audit.

Baseline: commit `2cb68f75449589217bc0c73e0190fa73378564b1`, particularly
`crates/pllm-garble/src/fusefss_backend/{composition.rs,lookup.rs,conversions.rs}`
and `composition/{gates.rs,norm.rs}`. Git retains those source versions.
Concurrent uncommitted wire/kernel/frontend work is not silently counted as
qualified execution or savings. In particular, “missing codec” below refers to
the committed F2 baseline, not a claim that no concurrent wire work exists.

Read together:

- [D8/D9](DECISIONS.md#d8-approved-fusefss-source-derived-descriptor-profile):
  ring50/Q12, nearest-even external encoding, source internal floors; public
  weights, private activations, two online holders, separate trusted dealer;
  fixed T, full-prefix feedback, lowest-index argmax, final client-only token IDs.
- [F2 composition evidence](../artifacts/fusefss-composition/local-v1/README.md):
  275 cases, conditional public intervals, bounded source-gate execution only.
- [Ordinary integration](sdk-integration.md), [shared contract](shared-computation-contracts.md),
  [source-profile resolution](../reproduction-contracts/shared-profile-resolution.md).

The calculator derives private bytes, public copies, directed peer frames,
peer bytes, keys, evaluations, triples, and primitive issuances from the source
formulas. Its historical 209-case accounting comparison is recorded in the
original JSON. Reruns need no receipt archive; numerical and ownership checks
live in the ordinary test suites. These are estimates, not measured model costs.

```sh
python3 -B paper/methods/fusefss-decoder-feasibility-costs.py
# Optional: regenerate this audit's own JSON only.
python3 -B paper/methods/fusefss-decoder-feasibility-costs.py --write
```

## 2. Public metadata and unresolved model circuit

| Dimension | GPT-2-base control | Pinned common Qwen |
| --- | ---: | ---: |
| Layers / hidden / intermediate | 12 / 768 / 3,072 | 24 / 896 / 4,864 |
| Query heads / KV heads / head width | 12 / 12 / 64 | 14 / 2 / 64 |
| Vocabulary | 50,257 | 151,936 |
| Norm / MLP / positions | LayerNorm / GELU / learned absolute | RMSNorm / gated SiLU / RoPE |
| Tied embedding/head parameter elements, including body/bias/norm | 124,439,808 | 494,032,768 |

Public configuration, fetched without tensors:

- GPT-2 dimensions: `openai-community/gpt2`, revision
  `607a30d783dfa663caf39e06633721c8d4cfcd7e`,
  [config.json](https://huggingface.co/openai-community/gpt2/raw/607a30d783dfa663caf39e06633721c8d4cfcd7e/config.json).
  This is a pinned dimension control, **not identification of the paper's evaluated
  checkpoint**. Public config says `gelu_new` and LayerNorm epsilon `1e-5`.
- Qwen: `Qwen/Qwen2.5-0.5B-Instruct`, revision
  `7ae557604adf67be50417f59c2c2f167def9a775`,
  [config.json](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/raw/7ae557604adf67be50417f59c2c2f167def9a775/config.json).
  Existing metadata lock: `docs/evidence/orthogonal-frontier-qwen25-source-lock.json`,
  config SHA-256 `18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45`.
  Q/K/V biases, no output/MLP biases, tied head; epsilon `1e-6`, RoPE theta
  `1,000,000`, bfloat16 source declaration. No tensor bytes were inspected.

**Selected complete-model circuit remains UNKNOWN.** Public dimensions do not
decide: exact source tensor decoding to Q12; bias placement; projection/dot/scale
floor order; GPT-2 source GELU versus config `gelu_new`; LayerNorm epsilon;
Qwen RMSNorm adaptation; Qwen RoPE table generation/encoding/product floors;
range certificates; equality encoding width/backend; comparison overflow policy;
scalar versus matrix correlations; final dead-row elimination. D10's PUMA Q18
epsilon decision does **not** select a FuseFSS epsilon. The current source norm
adds 1 to the unaveraged Q24 square sum; calling it checkpoint RMSNorm without a
separate decision would be false. Missing evaluated descriptor manifest still
blocks original FuseFSS numerical-identity claims.

## 3. Reproducible conditional lowering

JSON labels every workload `admitted: false`, every complete circuit unselected,
and exposes costs per phase. The following is a **named candidate using committed
scalar gates**, not a guessed implementation identity:

1. Every selection recomputes all prefix rows; causal cells only. For initial P,
   `R=sum(P+j)` and `C=sum((P+j)*(P+j+1)/2)`, `j=0..T-1`. No persistent KV.
2. Protected equality candidate: `onehot[c]=[token<c+1]-[token<c]` in A50;
   each comparison uses current typed predicate plus B2A. Endpoint c=0 retains
   its actual lower-key descriptor cost. Scan every public embedding row locally.
   No secret index, selected public row, or intermediate token is opened.
3. Public linear dots accumulate all Q24 products, then one exact signed ARS12
   per output. Public biases/positions add locally, with placement still unbound.
4. QK uses fresh scalar ring products for every causal head-dimension pair;
   sum, ARS12, public Q12 scale 512 (head width 64), ARS12. Current source softmax
   retains private max, clipping, two nExp tables/product/floor, denominator,
   width-specific reciprocal, output products/floors. PV sums scalar products
   before one ARS12 per output coordinate. These choices need source binding.
5. GPT-2 uses frozen affine GELU; Qwen uses current original-input SiLU/Phi,
   then gate/up product and ARS12. Qwen rotary candidate floors both public
   products separately. No PUMA numerics or float host nonlinear is substituted.
6. Two source norms per layer plus final norm on every prefix row. Head runs
   on last row only, complete vocabulary, one ARS12 per logit. Ascending-index
   argmax uses strict-greater comparison and two selects (score, index).
   Ties retain earlier index. Comparison by signed subtraction requires a
   nonwrapping difference proof; absent that proof this candidate rejects.
7. Private selected IDs append to role-local prefixes. Only final T IDs reconstruct
   at client. Client boundary share words are priced separately; no logits reveal.

For hidden H, intermediate I, physical KV width K, layers L:

| Count | GPT-2 | Qwen |
| --- | --- | --- |
| Public body matrix MACs, per worker | `LR(4H²+2HI)` | `LR(2H²+2HK+3HI)` |
| Body linear output floors | `LR(5H+I)` | `LR(3H+2K+2I)` |
| QK / PV secret scalar products, each | `LHC` | `LHC` |
| Nonlinear scalar calls | `LRI` | `LRI` |
| Public embedding / head MACs, per worker | `RVH` / `TVH` | `RVH` / `TVH` |
| Stable argmax comparisons | `T(V-1)` | `T(V-1)` |

GQA reduces K projection/storage, **not** query-head QK/PV product count.
All public MACs remain local worker work, not free or protected Beaver products.
The JSON additionally charges public scale, rotary, gamma and lookup-coefficient
multiplications on both workers; it does not count CPU instructions or additions.

### Current gate units

Both workers combined; peer bytes include both directions and actual native
80-byte headers, exclude authentication/outer transport. Scalars use one lane
because current composition issues one primitive per instruction.

| Gate | Private material B | Directed frames | Peer B | Local public-table copies B |
| --- | ---: | ---: | ---: | ---: |
| Product, B2A, or arithmetic Phi select | 48 | 2 | 192 | 0 |
| Interior A50 comparison including B2A | 5,456 | 6 | 560 | 0 |
| Signed A50 ARS12 | 4,376 | 2 | 176 | 0 |
| Positive TR12 then zero extension | 3,592 | 4 | 352 | 0 |
| Frozen affine GELU | 2,744 | 4 | 368 | 24,592 |
| Current composed SiLU | 24,616 | 24 | 2,272 | 49,552 |
| Current nExp | 13,264 | 16 | 1,424 | 24,704 |
| Source LayerNorm, width 768 | 7,308,824 | 6,806 | 627,136 | 396,752 |
| Source RMSNorm, width 896, guard1 candidate | 8,436,992 | 7,828 | 721,168 | 396,752 |

Derivation: ring scalar DCF `24+26b`; vector DCF `32+34b` (two coordinates,
including unused coordinate). Interior comparison retains two queries, three
private Bit immediates, one normalizer AND/triple, input-mask share and B2A.
For table N, `c=ceil(N/256)`, total intervals `N+2c-1`. Per worker: `c` keys,
`1348c` private B, `24(N+2c-1)+8c` copied public B, `80+8c` opening B.
Every chunk executes, even when selected index lies elsewhere. Reciprocal rows
use `N=2^(6+bit_length(k))`, including unreachable sentinel entries. Norm index
uses all 49 comparisons, 98 selects and 42 TR/zero-extension pairs, then rsqrt
lookup and **two distinct** affine floors. No exponent-bin shortcut is assumed.

### Workload dimensions

| Model, P+T | R | C | Equality calls | Nonlinear calls | QK products = PV products |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-2, 128+1 | 128 | 8,256 | 6,432,896 | 4,718,592 | 76,087,296 |
| GPT-2, 128+8 | 1,052 | 69,716 | 52,870,364 | 38,780,928 | 642,502,656 |
| GPT-2, 4+2 | 9 | 25 | 452,313 | 331,776 | 230,400 |
| Qwen, 4+2 | 9 | 25 | 1,367,424 | 1,050,624 | 537,600 |
| Qwen, 64+8 | 540 | 18,516 | 82,045,440 | 63,037,440 | 398,168,064 |

### Cumulative conditional totals — not live memory

GiB = `2^30` bytes. Exact integers and every phase are in
[the JSON](fusefss-decoder-feasibility-costs.json).

| Model, P+T | Dealer private payload to both workers, GiB | Native peer, GiB | Local public copies, GiB | Directed native frames |
| --- | ---: | ---: | ---: | ---: |
| GPT-2, 128+1 | 196.13 | 43.63 | 141.25 | 496,288,962 |
| GPT-2, 128+8 | 1,622.97 | 365.66 | 1,169.05 | 4,158,645,128 |
| GPT-2, 4+2 | 11.43 | 1.05 | 7.78 | 12,221,446 |
| Qwen, 4+2 | 62.67 | 5.40 | 48.88 | 62,168,416 |
| Qwen, 64+8 | 3,835.93 | 457.33 | 3,087.67 | 5,229,682,144 |

Examples of dominant phases, GiB private / peer / public copies:

- GPT-2 128+8: equality **537.30 / 55.15 / 0**; body linear floors
  **355.61 / 14.30 / 0**; softmax **269.43 / 32.48 / 271.12**; QK and PV
  **each 28.72 / 114.89 / 0**; GELU **99.11 / 13.29 / 888.20**.
- Qwen 4+2: SiLU **24.09 / 2.22 / 48.49**; equality **13.90 / 1.43 / 0**;
  body linear floors **11.16 / 0.45 / 0**; full head floors
  **1.24 / 0.05 / 0**, stable selection **1.57 / 0.27 / 0**.
- Qwen 64+8: SiLU **1,445.16 / 133.39 / 2,909.11**; equality
  **833.79 / 85.58 / 0**; body linear floors **669.31 / 26.92 / 0**;
  norms **203.67 / 17.41 / 9.58**. These are work/traffic priorities, not RSS.

DCF path-level counts per worker: respectively 160,786,250,142;
1,330,673,612,240; 8,882,094,148; 55,508,139,850; 3,503,220,481,048.
They are unshared traversal counts from code, **not AES timings or CPU seconds**.

## 4. Delivery, ownership, and timeouts

### Count each directed leg once

1. Dealer→A and dealer→B: each receives **half** the reported cumulative private
   material. That payload includes mask/base/Bit/triple shares, not only DCF keys.
   F2 has no qualified complete dealer codec. Serialized frame counts, binding
   envelopes, encryption/MACs, retries and publication leases remain unknown.
   Do not label logical payload an observed wire count. A relay adds directed
   legs and surviving copies; it cannot hide them inside peer traffic.
2. A→B and B→A: native frames/bytes above, symmetric. Current headers bind
   context/nonces but are not authentication or malicious-result checks. HTTP,
   stream, TLS, relay and response envelopes are additional, unknown costs.
3. Public weights delivered once to both workers, assuming u64 Q12 storage and
   shared tied head: GPT-2 **1,991,036,928 B**; Qwen **7,904,524,288 B**.
   Local availability may avoid delivery, not local residency/import work. Source
   dtype bytes, tokenizer, metadata, rotary tables and import overlap stay separate.
4. Public lookup-copy totals are current **local duplicated ownership work**,
   not automatic dealer/network retransmission. Original retained table/plan
   owners are additional live objects. Public table interning can remove most
   copy churn without changing fresh keys.
5. Logical client boundary: input token shares `16P` B, final ID shares `16T` B
   across both worker legs in this u64 candidate. Codecs/authentication/frame
   counts are unknown. No intermediate selected-ID response is permitted.

### Live admission is a separate ledger

- F2 limits: 32,768 instructions, 65,536 registers, 256 tables, 16 MiB plan,
  `2^26` work units per bounded plan, 64 MiB combined native ownership. All remain
  unchanged. These are **compiler/protocol limits**, not measured process RSS.
  Even GPT-2 4+2 has 4,756,880 candidate primitive calls. A single unrolled scalar
  plan cannot fit 32,768 instructions; this is a real blocker for that lowering,
  not for a separately admitted bounded loop/tile runtime.
- F2 scalar material streams by primitive, but one lookup retains every chunk.
  Two role register files, source plan/tables, dealer construction scratch,
  primitive owners, previous frames, exported outputs and cancellation survivors
  must overlap in admission. Current builder has substantial fixed reserved
  vectors: archived SiLU plan is 2,630,200 B for only 15 instructions. Building
  one fresh retained plan per model element is not viable reuse.
- Largest archived fixture planned envelope 11,882,256 B is a finite-gate result,
  not full-model peak. Sampled reservations are not true allocator/RSS high-water.
  No full-model compiler ownership bound, peak RSS, swap, or source-import overlap
  is measured here. JSON uses null, not zero, for unknown mandatory owners.
- Entire encoded public model, if resident, is 995,518,464 B per GPT worker or
  3,952,262,144 B per Qwen worker, excluding every other owner. It cannot fit
  inside the current 64 MiB primitive budget as one allocation. That is a blocker
  for that **fully resident ownership choice**, not for bounded public-weight
  streaming or separately admitted public model ownership. Neither alternative
  currently supplies complete model admission. Do not silently bypass the budget.
- Cumulative keys/copies/MACs remain necessary preprocessing, IO and work budgets;
  retiring a key fixes its live ownership, not its issuance/delivery cost. A loop
  compiler must count total work even when metadata and live values are bounded.

### Per-request time, not a blanket feasibility verdict

For **one shared aggregate bottleneck** carrying uncompressed dealer payload plus
native peer bytes, conditional ideal 1 Gbit/s minima are 2,059.52 s, 17,082.18 s,
107.24 s, 584.71 s, and 36,878.77 s in table order. This excludes public source,
transport overhead, CPU, dealer generation, round-trip latency and retirement.
It is not a universal network lower bound: independent full-duplex links may
overlap. On separate links use each directed leg's own bytes/rate; report the
critical path, not the sum of parallel legs. Compression/another circuit also
changes the premise. The JSON preserves exact aggregate bandwidth arithmetic at
100 Mbit/s and 1 Gbit/s and rates required for 300/1,200-second windows.

Current shared SDK surfaces differ: `settings.py` defaults HTTPX timeout to 300 s
(not a proof of absolute whole-request expiry); `_benchmark_contract.py` defaults
sampling timeout to 300 s and runner enforces subprocess timeout;
`runtime/replicated_worker.py` gives PUMA attempts a hard 1,200 s expiry and uses
60 s HTTP peer defaults. CLI research operator routes have their own 900 s default
and validation. None silently becomes an approved FSS request contract. Future
FSS admission must bind explicit request/dealer/transport/benchmark deadlines,
include final client reconstruction and cancellation/drain, and reject unsupported
work before import/issuance. No timeout or cap is increased by this audit.

## 5. Cheapest reusable exact improvements, in dependency order

1. **Bounded loop/tile compiler and live-range register allocation.** Reuse
   existing semantic operation IDs, schedules, range bindings and receipt schemas.
   Keep bounded row/gate templates; bind layer/row/tile/selection in counters and
   contexts. Retire scratch/frames/values as dependencies finish; no eager global
   material queue or unrolled per-scalar plan forest. Derive actual plan and
   runtime owner bounds before tensors. Preserve cumulative work admission.
2. **Intern immutable public tables and stream bounded public linear tiles.**
   Share source-identity-bound boundaries/coefficients, never keys/masks. Remove
   redundant zero slope storage/work for public constant LUTs only under a new
   explicit representation/accounting identity. Keep every descriptor-determined
   chunk; never branch on private chunk/index. This directly attacks multi-TiB
   local copy churn without pretending it was network or live RAM.
3. **Batched exact product/rescale/compare owners and dealer delivery.** Current
   composition is scalar despite primitives supporting four lanes. General
   bounded batching and larger scheduling tiles require owner proofs under current
   caps, not a cap raise. For n independent products, ideal single batched native
   opening is `160+32n` B across both directions versus `192n`; scalar triple
   material remains `48n`. New framing/latency savings are conditional, not included
   above. Retain fresh lane/request correlations, one-use burn and exported leases.
4. **Use scalar payload DCF for scalar correction.** Current floor conversions
   carry unused vector coordinate. Existing ring scalar DCF can express the same
   one-coordinate arithmetic correction after review. ARS12 payload would fall
   from 4,376 to 3,352 B across both roles (1,024 B saved, 23.4%); no floors move,
   key reuse or rounding substitution. Update codecs/admission and independent
   negative-floor/wrap tests together. Removing public-normalizer circuit gates
   and unused public-bit work is another small exact compiler simplification.
5. **One masked public-query family for full equality/lookup.** Reuse universal
   public-boundary algebra, evaluate all public vocabulary boundaries under one
   fresh token mask, or compile a reviewed arithmetic-output DPF/trie selection.
   Preserve one-hot embedding, role privacy and full public matrix work. Current
   PUMA trie lowering is a useful graph/layout idea, **not transferable replicated
   Q18 material/security**. Fresh material must remain request/lane bound. Shared
   public-query traversal can reduce repeated DCF expansion work; benchmark only
   after exact shape/privacy/accounting proof. No numerical speedup assumed here.
6. **Tiled private matrix correlations for QK/PV.** Reuse consuming Beaver
   lifecycle, add matrix-shaped dealer correlations only after security/resource
   review. A tile needs arithmetic shares of `(A,B,AB)`, not one scalar triple per
   multiply; opens `X-A,Y-B` and preserves exact dot floor after accumulation.
   This changes preprocessing/round shape, not source arithmetic. Bound causal/GQA
   tiling, live operands, dealer scratch and transport; do not price nonexistent
   matrix products as free public MACs. Dot-specific/batched scalar controls remain
   useful independently.
7. **Source-safe dead final-row elimination and stable reduction.** Only T final
   rows need final norm/head for this response contract, versus R in baseline.
   JSON lists exact removable norm costs separately. Body still recomputes full
   prefix; no KV adaptation. Tree argmax may cut dependency depth while retaining
   `V-1` comparisons, score/index privacy and explicit lowest-index tie semantics.
   No request caching, early EOS, stochastic generation or intermediate reveal.

These are independent engineering targets, not permission to alter D8/D9. A
full-model launch needs **all** mandatory numeric, owner, total-work, transport
and request-lifetime gates; improving one scalar key size does not discharge them.

## 6. Ordinary SDK reuse and proposed contracts — no implementation

Current GKM seams: typed `OpenWeightVerifiableCpu` /
`OpenWeightVerifiableOutsourcing`, `GkmDecoderContract`, source lowering and import,
native composition resolution, real worker adapters, model binding, RuntimeClient,
CLI and ordinary benchmark receipts. GKM's client-visible float/semantic execution,
client-local KV and final linear verification are **not** FSS role/numeric contracts.

Current PUMA seams: `PumaThreePartyCpu` / `PumaThreePartyDecoder`,
`ReplicatedPrivatePrefix`, `ThreePartyReplicatedRoles`,
`profiles.resolve_runtime_composition`, and
`runtime/replicated_{source,client,worker,peer,transport,protocol}.py`.
They demonstrate public-weight choice, source-aware checked admission, tiled/trie
lookup, tiled selection, HTTP/stream choice, lifecycle and fixed-count final-only
response integration. Their three-party replicated sharing, Q18 source arithmetic,
client setup owner, expiry and parked full-Qwen range gate cannot select FuseFSS.

Proposed reusable names (design only; **not registered capabilities**):

| Contract/component | Existing family/seam | Required meaning |
| --- | --- | --- |
| `SharedTensorExecutionContract` | native compiler + runtime schedules | Arithmetic/XOR/bit domains, scales, exact floors, contractions, reductions, typed range certificates and fixed public shapes; reject plaintext fallback. |
| `FssSourceDecoderContract` | native admission + `modeling.lower_model` | Source/tokenizer/frontend identity, complete public operation/range manifest, bounded templates, live and cumulative budgets, deadlines before graph/tensor/role allocation. |
| `FuseFssSourceDecoder` | `protocols`, canonical component discovery | Ring50/Q12 source profile and actual supported bounds; no secrets in parameters. Operator-only capability stays distinct from decoder capability. |
| `TrustedDealerTwoShareRoles` | `roles.topology`, deployment/servers/party | Client, two non-colluding online holders, independent trusted preprocessing dealer, authenticated permitted directed channels and ownership. Never infer topology from worker names. |
| `FixedCountPrivatePrefix` | state components | Protected full-prefix append/recompute, no persistent KV, explicit T, final-only IDs. Reuse semantic lifecycle, not replicated representation. |
| `PrivateTokenBoundaryContract` | compiler boundary + model binding | Equality→one-hot lookup, final norm/full head, stable argmax, final client reconstruction; bind source ranges and token vocabulary. |
| `FssMaterialStream` / `SharedGateBatch` | native runtime owners + transport | Dealer-only issuance, role-specific consuming import, exact bounded codecs, context/PC/shape/ordinal checks, retries burned, cancellation/drain/retirement and directed receipts. Runtime owners, not configuration fields. |
| `FuseFssSourceCpu` | typed profile + `resolve_runtime_composition` | One admitted combination through Pipeline/Experiment; reject incompatible state/kernels/roles/numerics, never method fallback. |

Use existing `ComponentRef`, `ComponentDescriptor`, `create_component`, Pipeline,
Experiment, SDK/gateway/CLI/benchmark chain. Extend shared role/client/transport
dispatch and coverage inventories; no method-specific model loader, configuration
DSL, runner, benchmark metrics or unauthenticated special path. Bind public tables,
source publication/import, dealer preprocessing/delivery, online phases, client
reconstruction and retirement in ordinary receipts. Verify installed-wheel
round trips, separate roles, complete intermediate integer parity and token quality
only after admission. FSS protocol completion is not GKM malicious verification;
transport authenticity is not independent cryptographic qualification.

## 7. Real blockers and next admissible scope

1. **Frontend/circuit identity and checkpoint ranges unqualified.** Freeze complete
   source lowering, Qwen adaptations, binary source encoding and all no-wrap/index
   proofs. Dimension-only calculations cannot admit actual weights or inputs.
2. **No complete bounded tensor/loop runtime admission.** Scalar F2 plan success
   does not establish full-model metadata, public-weight owners, live state, total
   work or deadline fit. Implement/reconcile reusable admission before import.
3. **No qualified complete separate-dealer transport/SDK response path at baseline.**
   Require bounded role-specific material codecs, authentication, binding/replay,
   source/table publication, cancellation and real ordinary request receipts.
4. **No model quality, current full-stack timing/RSS or cryptographic review.**
   Missing evaluated descriptor remains an independent original-model limitation.

Next safe scope: metadata-only canonical manifests and bounded reusable contract
design; independent exact gate/tile tests under unchanged caps when separately
authorized. Then ordinary source-aware preflight. Only admitted source, ownership,
work, topology and request lifetime permit checkpoint/role execution. No cumulative
byte total in this report is treated as simultaneously resident RAM.
