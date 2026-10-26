# Ten-method coverage audit

Audit date: 17 October 2026. Literature cutoff: 16 October 2026.
The selection is the ten methods in [FRAMEWORK-REWRITE-PLAN.md](FRAMEWORK-REWRITE-PLAN.md), in its original order.
This audit uses primary papers, installed source bytes, registry declarations, and existing evidence.
It adds no execution measurements.

**Later implementation follow-up.** The table below retains its inspected build
and historical scope. GKM subsequently gained a bounded native full/seeded decoder,
checkpoint bounds, installed-wheel conformance and a bounded ordinary two-worker
SDK/gateway/CLI/benchmark composition; see
[the SDK reference](../docs/research/gkm-sdk-reference.md) and
[current implementation status](IMPLEMENTATION-STATUS.md). A subsequent
[source-present full-checkpoint response](artifacts/gkm-sdk/qwen-response.md)
passes fresh admission and completes four input tokens plus two verified outputs.
A later [ordinary CLI quality smoke](artifacts/gkm-sdk/quality-smoke.json) matches
3/3 Qwen prefill and 9/9 same-token decode reference selections through both GKM
workers. These close several primitive, decoder, serving-integration and bounded
numerical-check gaps. Broader quality, workload/state qualification and
matched complete-system comparison remain open. The qualified ten-method count
remains zero. The documentary status below must not erase this later progress
or promote it to a serving rank.

The later explicit GKM seeded profile also runs ordinary two-worker SDK/gateway
requests and accepted-only response-state auditing. Generated source/oracle checks
cover both sharing choices. Its first six-cell pinned-Qwen comparison is
[host-memory blocked](../docs/evidence/gkm-sharing-qwen-blocked.json), so this
integration does not qualify checkpoint parity or a serving rank.

A subsequent streamed single-window ownership retry passes independent complete
pinned-Qwen state checks in both sharing profiles at four input tokens plus two
outputs, with zero observed swap growth. Complete logits, active KV, history and
selections agree; see the [SDK reference](../docs/research/gkm-sdk-reference.md).
The separate [six-client sharing screen](../docs/research/gkm-sharing-reference.md)
preserves accepted state and observes zero swap growth. Seeded TCP streams fall
8.39%, with essentially unchanged client RSS. This initial 4+2 ablation supplies
bounded profile costs, not broader quality, source-style prefill batching or a
qualified ten-method comparison row.

PUMA subsequently gained the bounded ordinary `PumaThreePartyCpu` composition:
native semantic compilation, three authenticated worker processes, private
full-prefix feedback, SDK/gateway execution and standard response benchmarks.
Generated decoder controls and bounded primitive oracles are covered in the
[PUMA reference](../docs/research/puma-truncation-reference.md#bounded-sdk-composition).
The shared full-Qwen workload remains outside its admitted envelope. Full-model
numeric/quality qualification and matched performance remain open; the historical
table below and zero-qualified-method count retain their original scope.

FuseFSS subsequently gained the bounded `FuseFssSourceCpu` composition with a
separate offline dealer, two online share holders and ordinary SDK/gateway/CLI
execution. Five generated decoders match 918 complete semantic observations;
standard generated-source benchmarks cover both approved numeric profiles. See
[the SDK reference](../docs/research/fusefss-sdk-reference.md). Request-long
replay/tape limits and complete-checkpoint admission still block trained-model
qualification; these results do not fill a complete-system ranking cell.

**The requested ten-system ranking is not currently supported.**
Six selected methods have executable component adaptations.
None has a qualified, complete PLLM reimplementation under the proposed common comparison contract.
Some SIGMA-derived components occur in complete PLLM responses, but those responses do not reproduce SIGMA.
Gupta–Katz–Miers has related executable primitives and a source-qualified integration specification, but no admitted complete comparator.
PUMA, CacheMir, and ROSETTA have planned interfaces without executable method implementations.

An absent implementation is an unresolved coverage result, not a measured failure.
A failed adaptation does not establish failure of its source system.
All ten methods therefore retain an unresolved serving rank and unresolved client-tier eligibility.
Slalom, SmoothQuant, and easier operator studies do not replace the selected methods.

## Current trained-execution round — 26 October 2026

All ten Astra reviews are complete. This section updates execution coverage, not
the 17 October audit, table, or measured claims below. Latest scoped commits are
`2ff454b` (shared-owner THOR/NEXUS/ROSETTA Rust references) and `4ac1cfb` (Python
experiment). Passing component tests and strict lib/tests Clippy were run on the
stated **dirty working baseline**, not isolated committed HEAD. They do not
qualify unfinished drafts, trained execution, or whole-method security.
**Zero of ten methods qualifies for a public complete-system ranking; serving
ranks and client-tier eligibility remain unresolved.** No whole-method security
claim follows from source recovery, approval, primitives, or generated fixtures.

The ordinary SDK completion target is checkpoint → actual source-bound prepared
tokens → import/lowering → protected body/head → required state/feedback (or
BERT classification) → accepted output → ordinary `Experiment` report.
Source descriptors, bookkeeping and generated correctness controls alone do not
complete this chain. The table names the first executable gap, not merely the
next approval. Factory paths are relative to the repository root.

| Method | Actual entry point and current status | First executable gap / distinct approval gates |
| --- | --- | --- |
| GKM | `examples/papers/gkm.py:experiment` / `seeded`, `compiled_gkm_v1`; existing complete ordinary two-worker SDK/benchmark chain. `examples/papers/gkm_quality.py:experiment` / `seeded` configures paired quality. | Obtain a coherent matching wheel and prepared-prompt integration: completed Sol61 preflight found the installed 18 October native lacks `lookup_batch`/`linear_batch`/`accept_batch` required by `4ac1cfb`. No launch/resource fit is established. Then resource-admit the **unrun** twelve-prompt FP32 diagnostic. Historical 4+2 full/seeded Q11 state checks and three-prompt FP32 smoke remain bounded; this is not a garbling/security hold or broader publication qualification. |
| Maverick | `MaverickDelegatedLinear` and bounded batch primitives; no complete method SDK factory. | Implement source-bound fixed-eight protected continuation, including client-local nonlinear operations and original-dot window lifecycle with Protocol 2 trust. M-X/M-D adoption, actual-code security and no-wrap gates remain separate. |
| SIGMA | Bounded native arithmetic/activation/normalization components; no trained method executor. | Implement the frozen four-lane LayerNorm → private-matrix handoff with consuming leases; current clarifications are source/design only. This is neither joint-view security qualification nor complete trained execution. |
| FuseFSS | `examples/papers/fusefss.py:experiment` and checked smoke factories; complete generated dealer/two-holder SDK path. Three selections already have generic decoder support (limit four). | Integrate D51's corrected cohort and ordinary oversize refusal under unchanged caps. Cached source-template `hello` has 30 tokens: default `fss_decoder` sequence ≤16 and `fss_source._contract` rows+outputs−1 ≤16 reject before tape pricing; no new trained attempt fits this cohort. Historical legacy `user: hi` had three tokens; corrected `hi` is unknown. The old one-selection pressure stop never accepted and supplies no corrected-cohort evidence. |
| BumbleBee | `PrivateEmbeddingLookup(repacking="interleave")` is a bounded component; no complete method SDK factory. | Supply full encrypted OLT, distributed OT and nonlinear execution; B-N1 source/numerical mapping and backend/security selection remain held. Lookup controls do not complete the transformer. |
| PUMA | `examples/papers/puma.py:experiment`; complete compact three-worker generated SDK/gateway path, not trained-qualified pinned full-prefix execution. | Execute an admitted actual trained prefix through accepted private feedback/output. Expanded private-fixture second-RMS refusal is valid for its propagated interval, not a proved propagation bug; compact gateway success does not remove trained-source/range gates. |
| NEXUS | Bounded `PackedEncryptedSoftmax(method="repeated_squaring")` and clear Algorithm 5/shared-owner controls; no trained method SDK factory. | Build the confidential-server-weight dual-key whole executor with its numeric and selection profile. Clear Algorithm 5 is not encrypted trained execution; profile, key/security and lifecycle gates remain held. |
| THOR | Bounded `PackedEncryptedSoftmax(method="square_normalize")` and FFN custody work; no complete encoder SDK factory. | Implement source-bound full BERT encoder/classification with numerical contracts and CKKS backend. FFN custody migration is not the whole method. Use a separate compatible BERT workload, not forced Qwen2 autoregressive generation. |
| CacheMir | Source/calibration descriptors and bounded state/numeric studies; no encrypted-KV method SDK factory. | Supply encrypted KV executor/backend with GQA mapping and numeric certificates. Source/calibration authority, encrypted kernels and complete feedback/selection remain independent held gates. |
| ROSETTA | Source/placement/shared-owner references and clear algebra only; conversion specifications are not implemented encrypted conversion. No full encrypted method SDK factory. | Implement CKKS/LWE conversion and HomComp inside the complete encrypted executor. Conversion key/error graph, numeric/backend and author-parameter gates remain unresolved. |

D51's prepared-prompt candidate is **not landed**; GKM/FSS/PUMA inherit the legacy
`DelegatedClient._encode` path (`FssClient` inherits it). Cached Qwen `hello`
produces **30 tokens under the source template**, versus legacy three; this is
not a count for every text or corrected `hi`. D51’s unlanded eight-file candidate
passed 115 pure checks; matching SDK checks passed 5/6, including both compact
cases. Expanded private-fixture second-RMS refusal is sound for its declared
interval; trained execution remains unqualified.
Actual GKM/FSS descriptor-parity correction remains pending: zero BOS/EOS false
fallback and FSS missing-BOS/template fallback repairs/tests are unapplied source work.
D52's readout/summary
metadata now matches the **already-existing LastToken graph**; it changes no
arithmetic or execution optimizer and proves no checkpoint execution/whole fit.
Do not substitute legacy three-token or four-token metadata for actual 30-token
admission. Prompt changes invalidate old cohort comparisons; source quotations
alone cannot authorize another checkpoint.

ROSETTA's recovered citation is [HE3DB (CCS 2023)](https://github.com/zhouzhangwalker/HE3DB),
commit `1d8d7827baf32b4094b7547aeeaa9faf59253b26`: **source oracle only**, not a
dependency. HE3DB's integer encoding was recovered, but ROSETTA's actual 53-bit
port, scales, noise and key graph remain unknown; the D37 grid is a proposal,
not recovered author parameters.

Reuse existing `DecoderPlan`, coverage/inventory and SSA bookkeeping; the rejected
duplicate SIGMA/BumbleBee obligation inventory has no immediate backend consumer.
Do not add another inventory or planning layer. Native, resource, security and
author-parameter gates remain distinct; no cap increase is authorized. FuseFSS's
64 MiB native ceiling, 32 GiB tape ceiling and default sequence limit 16 are
distinct and unchanged. Execution recipes and scoped evidence:
[paper experiments](../examples/papers/README.md),
[current implementation status](IMPLEMENTATION-STATUS.md),
[execution decisions](../docs/plans/ten-method-execution-decisions.md),
[GKM SDK](../docs/research/gkm-sdk-reference.md),
[FuseFSS SDK](../docs/research/fusefss-sdk-reference.md), and
[PUMA SDK](../docs/research/puma-truncation-reference.md#bounded-sdk-composition).

## Manuscript-ready coverage table

A complete decoder must cover token lookup, the model body, output selection, persistent state, and failure behavior.
A serving rank additionally needs matched complete-request measurements and an accepted security and numerical contract.
The source identifiers below link to the detailed qualifications.

| Selected method | Executable PLLM scope | Available local evidence | Principal missing contract |
| --- | --- | --- | --- |
| [Gupta–Katz–Miers](#gkm) | Shared-linear and boundary primitives exist separately. | Only static comparator qualification exists. | The two-worker lookup/body/head composition, verification, and decoder contracts are missing. |
| [Maverick](#maverick) | A bounded coded matrix–vector operator exists. | Synthetic operators pass. Admission rejects two larger shapes. | Code certification, preprocessing ownership, transport, and decoder integration are missing. |
| [SIGMA](#sigma) | CPU gates occur in a changed PLLM decoder. | Bounded conformance and complete Qwen response pairs exist. | Original protected operations, dealer, numerical behavior, and state need qualification. |
| [FuseFSS](#fusefss) | A fused gate supports generated decoder integration. | Gate costs exist. Completed checkpoint pairs use unfused gates. | General compilation and complete fused decoder measurements are missing. |
| [BumbleBee](#bumblebee) | Interleaving supports bounded encrypted lookup. | Lookup passes. Clear requantization fails exact fidelity. | Distributed setup, nonlinear protocols, and complete decoder integration are missing. |
| [PUMA](#puma) | Decoder and state interfaces remain pending. | No local method measurements exist. | Three-party arithmetic, lookup, nonlinear operations, and state are missing. |
| [NEXUS](#nexus) | A bounded CKKS softmax operator exists. | Synthetic outputs pass. Checkpoint scores exceed the local range. | Complete attention, refresh, output handling, and state are missing. |
| [THOR](#thor) | A different bounded CKKS softmax operator exists. | Synthetic outputs pass. Checkpoint scores exceed the local range. | Source kernels, approximation schedules, refresh, and decoder integration are missing. |
| [CacheMir](#cachemir) | An encrypted-state interface remains pending. | No local method measurements exist. | Cache updates, packing, refresh, and token feedback are missing. |
| [ROSETTA](#rosetta) | Scheme placement and conversion remain pending. | No local method measurements exist. | Conversions, segmented lookup, scheme selection, and complete decoding are missing. |

**Table conclusion:** zero of ten methods qualifies for the requested complete-system ranking.
Component results remain useful within their own contracts.
The table makes no claim that any original system cannot satisfy a memory budget.

## Evidence and identity rules

The [primary review](PRIOR-ART-REVIEW.md) and [source manifest](prior-art-sources.json) supply the dated selection evidence.
The [implementation audit](IMPLEMENTATION-EVIDENCE.md) separates complete PLLM adaptations from source-system reproductions.
The [implementation inventory](../docs/data/research/implementations.json) supplies historical component evidence.
The [packaged method registry](../python/pllm/components/planned_methods.json) supplies implementation and pending-interface status.

Registry names alone do not establish executability.
[`PendingComponent`](../python/pllm/components/_planned.py#L192-L208) rejects construction with `NotYetImplementedError`.
The inspected registry's Gupta–Katz–Miers abstract-only description was stale.
The later [source qualification](COMPETITOR-QUALIFICATION.md) supplies the reviewed full paper and selected author code.
The current bounded SDK component supersedes its pending declaration; the
historical table above retains the original audit scope.

The source hashes below identify exact PDF bytes.
An unversioned URL does not identify a particular revision without those bytes.
The audit recomputed all ten cached source hashes and the separately cached BumbleBee final-paper hash.
All eleven matched the existing manifest.
The BumbleBee final snapshot remains in the manifest's temporary cache, which is not a durable publication archive.

The inspected checkout was `06b5d6fc0b768be8460e056e2c3c775825e90738`.
Static comparisons covered ten selected files in the clean-local-v5 installation and the corresponding checkout files.
They included protocol, nonlinear, state, pass, registry, admission, and relevant runtime declarations.
All ten file pairs matched byte-for-byte.
The installed native binary also matched the accepted conformance identity.

| Accepted execution identity | SHA-256 |
| --- | --- |
| Clean-local-v5 wheel | `1811d2d9d5c6cf1360091122a62fd9678031159acab1ab2a278b13b7872d2e8c` |
| Installed native binary | `826c2f41612912c38a2cd902a1988b50a069b4e278f1f05ffdb6980744e42c64` |
| Runtime `BuildRef` | `8a8803783aa2b61f84ac461c209925f911acbc447dc86bdc3120ee384d4c83d9` |

The [historical results](artifacts/conformance/README.md) retain ordinary reports
for the bounded FSS, lookup and softmax adaptations. Build receipts and archive
manifests have been retired; use standard uv/Maturin builds and normal tests.
The historical 46 passing tests do not represent 46 methods or a performance campaign.
Historical cost records retain their original builds, workloads, ownership scopes, and failures.
This audit neither repeats those executions nor transfers their measurements to clean-local-v5.

Author-reported evaluations describe each source paper's own workload and hardware.
They do not fill the missing PLLM comparison cells.

For the qualifications below, W8A8 means eight-bit weights and activations.
Ring32 and ring64 mean arithmetic modulo `2^32` and `2^64`.
Q24 and Q12 indicate 24 and 12 fractional bits.
Function secret sharing (FSS) gives parties shares of a function for protected evaluation.
Homomorphic encryption (HE) permits calculations on encrypted data.
CKKS supports approximate arithmetic, BFV supports integer arithmetic, and TFHE supports Boolean operations and encrypted lookup.

The KV cache stores attention keys and values from earlier tokens.
Resident set size (RSS) measures resident process memory in the stated observation window.

## Detailed qualifications

<a id="gkm"></a>
### 1. Gupta–Katz–Miers

**Primary source.** Gupta, Katz, and Miers, *Private and Verifiable Outsourcing of Open-Weight LLM Inference*, [ePrint 2026/1849](https://eprint.iacr.org/2026/1849).
The reviewed text is [the cached PDF](../papers/r19-open-weight-verifiable.pdf).
Its SHA-256 is `e1863980621e0bbd2d446b43f2d9415318ebeec400b0e6c818685752f6448e1d`.
The fresh PDF request failed with HTTP 403, so current-PDF byte identity remains unknown.

Locators are §4.1/Fig.1, PDF pp.6–7, §4.3/Figs.3–4, pp.9–10, §4.4/Fig.5, pp.10–11, and §5, pp.12–14.
The [author-code review](COMPETITOR-QUALIFICATION.md) pins `kanav99/dvllm` commit `271637be7196d26a5433ca1e62b1a5357e769a5b`.
It establishes source availability, not identity with the authors' evaluated build.

**Executable scope and differences.** [`OpenWeightVerifiableOutsourcing`](../python/pllm/protocols/__init__.py#L216-L218) remains a pending interface.
The source uses dense XOR private lookup and shared linear computation at the same two public-weight servers.
The client owns nonlinear computation, complete vocabulary logits, and full KV state.

Its paper specifies scale 2048, int16 weights, ring64 vectors, and nearest-integer encoding.
Inspected author code uses truncating casts, which leaves a source-to-paper numerical discrepancy.
PLLM's W8A8/ring32 primitives and packed lookup therefore require an explicit adaptation.

The [compiler](../crates/pllm-compiler/src/decoder_boundary.rs#L160-L195) currently admits the coupled boundary only with prepared-body roles.
That four-role candidate is not the missing two-server comparator.

Privacy assumes an honest model owner and non-colluding servers.
The verified construction also uses published commitments and online or batched client checks, with rejection before accepted output.
Its integrity claim permits malicious, colluding servers under the paper's assumptions.
A semi-honest restriction must explicitly omit that integrity guarantee.
The source qualification records unresolved inspected-path randomness and abort behavior without an artifact-wide security verdict.

**Measurements and missing contracts.** The paper supplies author-reported evaluation in its own contract.
Its 179 MB storage and 4 GB client VM figures are not whole-client peak RSS.
PLLM has no complete matched comparator measurement.

The next gates are the admitted two-worker topology, source-specific lookup and verification, numerical bounds, fresh randomness, and complete state and sampler conformance.
Whole-client memory, simultaneous provider memory, complete traffic, and phase times remain missing.
**Complete decoder: absent as a qualified PLLM comparator. Serving rank: unresolved.**

<a id="maverick"></a>
### 2. Maverick

**Primary source.** *Maverick: Private and Verifiable LLM Inference Made Practical via Matrix–Vector Multiplication Delegation*, [arXiv:2609.10264v1](https://arxiv.org/abs/2609.10264v1), 9 September 2026.
The reviewed [PDF](../papers/r24-maverick.pdf) has SHA-256 `c9570fdeb51a26f9d707bcc7bf485f2001e7d3e15ebdf34815123935a1e8e2a0`.
The relevant locators are §§5.2, 6, and 7.1, Protocol 3, Equation (1), and Appendices A/B.
The original delegates public-weight linear computation to one server with privacy and verification under its coding and dual-LPN assumptions.

**Executable scope and differences.** [`MaverickDelegatedLinear`](../python/pllm/protocols/__init__.py#L178-L208) exposes a bounded matrix–vector operator.
[`coded_delegation_reference.rs`](../crates/pllm-core/src/coded_delegation_reference.rs) owns its preprocessing, fresh sparse masks, and conditional checks.
The operator uses signed-i8 inputs and the BabyBear field, `q = 2,013,265,921`.
It supplies exact centered integer products without decoder scales, biases, logits, or KV state.

Public seeds generate its repeat-accumulate-accumulate codes.
Two weight-41 verification challenges replace the paper evaluation's weight-31 choice.
The cited Appendix B distance bound assumes `q > 2^31`, which does not cover BabyBear directly.
The actual code distance, seeded ensemble, and dual-LPN privacy parameters remain uncertified.
There is no authenticated provider transport or decoder admission.

**Measurements and missing contracts.** The [installed operator cohort](../docs/evidence/research-coded-delegation-cost.json) checks 50,688 synthetic outputs across nine fresh processes.
The [research card](../docs/research/maverick-reference.md) records 51.72/66.45 MB persistent client payload for the admitted full-width shapes.
Both larger MLP shapes reject before allocation.
The [22.90 GB preprocessing lower bound](../docs/evidence/maverick-coded-storage-lower-bound-2026-09-27.json) applies to the specified whole-body layout, not measured RSS.
These are real bounded admission results, not measured failure of the original system.

Certification, an admitted storage design, distributed lifecycle, and complete numerical and request measurements remain missing.
**Complete decoder: absent. Serving rank: unresolved.**

<a id="sigma"></a>
### 3. SIGMA

**Primary source.** *Sigma: Secure GPT Inference with Function Secret Sharing*, [PoPETs 2024(4):61–79](https://petsymposium.org/popets/2024/popets-2024-0107.php).
The final [PDF](../papers/r13-sigma.pdf) has SHA-256 `f911b715b0e0523fa4d77f21c28c6a0440c5be781e67b5b30419f20e505a6ebf`.
Locators are §2.2, PDF p.3, §§2.5–2.6 and 4–5, and §§6–7, pp.10–13.

The source protects model and input in two-party computation with a trusted preprocessing dealer.
Its security model permits static semi-honest corruption of one online party.

**Executable scope and differences.** [`MaskedWireGatedMultiplyQ12`](../python/pllm/nonlinear/__init__.py#L137-L196) supplies a bounded residual-SiLU product.
[`masked_wire_reference.rs`](../crates/pllm-garble/src/masked_wire_reference.rs) and [`masked_wire.py`](../python/pllm/runtime/masked_wire.py) also support compiled CPU regions.
PLLM uses trusted-client float-to-Q24 conversion, floor shifts, Q12 output, and local issuance.
Its CPU distributed point function uses 128-bit seeds rather than SIGMA's packed 127-bit seed and flag representation.

The whole-decoder adaptation keeps attention and other operations client-local under its separate public-weight contract.
It does not reproduce the original protected model, GPU implementation, or full dealer and nonlinear schedules.
[`SigmaFssDecoder`](../python/pllm/protocols/__init__.py#L357-L359) remains pending.

**Measurements and missing contracts.** Bounded execution passes in the accepted family manifest.
Separate [complete-response evidence](../docs/research/nonlinear-decoders.md) exists for a pinned Qwen2.5 adaptation with 30 input and two output tokens.
The [dispatch pair](../docs/evidence/research-nonlinear-dispatch-qwen25.json) records 2,705.809/1,634.109 seconds and 5,716.561 MB of all-link TCP-stream traffic.
A later [material-encoding pair](../docs/evidence/research-nonlinear-material-qwen25.json) is a separate build and cohort.
Neither pair is a SIGMA system measurement.

The [clear numerical diagnostic](../docs/evidence/research-nonlinear-decoder-quality-qwen25.json) has zero exact logit/KV positions out of 48 against unchanged W8A8.
That diagnostic is not protected model execution or representative task quality.
Original role and numerical mapping, protected decoder coverage, quality acceptance, and matched lifecycle measurements remain missing.
**Complete PLLM adaptation: present in a changed scope. Complete SIGMA reproduction and serving rank: unresolved.**

<a id="fusefss"></a>
### 4. FuseFSS

**Primary source.** *FuseFSS: Efficient Secure LLM Inference with Function Secret Sharing*, [arXiv:2606.09551v1](https://arxiv.org/abs/2606.09551v1), 8 June 2026.
The reviewed [PDF](../papers/r14-fusefss.pdf) has SHA-256 `6109399761ab0cc1ca6eb617fec1a811cbe59d474455ce3549467e5a5b77cbf0`.
Locators are §3.1, p.3, §§4–5, Theorems 4.6–4.7, §6, and Appendix B.5, p.15.

The source assumes two online parties, preprocessing, and semi-honest corruption of at most one party.
Model parameters and architecture are public unless otherwise stated.
Its gate-level security statement explicitly permits public shape leakage.

**Executable scope and differences.** `MaskedWireGatedMultiplyQ12(layout="fused")` supplies one CPU multi-view layout under the SIGMA card's changed numerical contract.
The fused and unfused layouts have the same integer function.
Client conversion, local issuance, and relay ownership differ from the source system.
[`FuseFssPredicateFusion`](../python/pllm/passes/__init__.py#L150-L152) remains pending.
The eight-phase gate does not constitute the general compiler or its complete security argument.

**Measurements and missing contracts.** The [operator cohort](../docs/evidence/research-masked-wire-fusion-sdk.json) checks 5,778 outputs across eighteen fresh processes.
At 64 lanes, fusion changes eleven phases to eight and peer frames from 6,020 to 5,552 bytes.
Both-party keys grow from 82,592 to 145,984 bytes in that cohort.

Generated decoder checks and bounded worker transport exist.
The completed real-checkpoint response pairs use the unfused layout, so they cannot establish a fused system cost.
The [older timeout](../docs/evidence/research-nonlinear-decoders-qwen25-timeout.json) lacks the timed-out candidate identity and cannot establish a FuseFSS-specific failure.

General compilation, full operator coverage, distributed security mapping, and complete matched fused/unfused measurements remain missing.
The source's Appendix B.5 also labels its cross-system baseline attempts non-comparable.
**Complete FuseFSS reproduction: absent. Serving rank: unresolved.**

<a id="bumblebee"></a>
### 5. BumbleBee

**Primary source.** *BumbleBee: Secure Two-party Inference Framework for Large Transformers*, [NDSS 2025 final PDF](https://www.ndss-symposium.org/wp-content/uploads/2025-57-paper.pdf).
The final-paper SHA-256 is `184bbe5104f183d4df66db2413f89c96c6c814e0b0ec6297f19745a1fa6e203b`.
Relevant locators are §II.D, p.4, §III/Fig.6, pp.5–7, §V, and §VII.

The local component instead binds the [cached ePrint 2023/1678 PDF](../papers/r22-bumblebee.pdf), §IV/Fig.6.
That PDF has SHA-256 `15583a86593998079d42b8b134dc4f4f5be9ca72a7ced8f31bba05f68893516a`.
These source versions and section numbers remain distinct.

**Executable scope and differences.** [`PrivateEmbeddingLookup`](../python/pllm/protocols/__init__.py#L271-L314) uses `repacking="interleave"` inside a FastQuery-derived lookup.
[`coefficient_repacking.rs`](../crates/pllm-core/src/coefficient_repacking.rs) supplies the ciphertext interleaving mechanism.
The bounded BFV profile uses degree 4,096, plaintext modulus 8,192, and signed packed fields.
Local random-OT issuance and half-gates conversion replace distributed setup and SiRNN.

The original protects model and input against static semi-honest corruption of one party.
Honest one-hot queries and the local dealer remain explicit assumptions.
The reference does not establish the original two-party, private-model transformer security boundary.
The source's HE-to-share boundary includes circuit privacy, which the local adaptation does not establish.
[`BumblebeeTwoPartyTransformer`](../python/pllm/protocols/__init__.py#L226-L228) remains pending.

**Measurements and missing contracts.** The [lookup cohort](../docs/evidence/research-repacked-embedding-cost.json) checks 3,456 integer outputs across eighteen fresh processes.
The accepted family audit separately exercises active interleaving with 43 automorphisms and positive noise budgets.
Neither result supplies full-vocabulary execution or complete decoder costs.

The [clear requantization diagnostic](../docs/evidence/research-packed-embedding-quality-qwen25.json) fails exact logits/KV without execution of encrypted inference.
The [ownership analysis](../docs/research/packed-embedding-reference.md) finds zero released tied-weight bytes while the client retains the output head.

Full-width noise, distributed conversion, circuit privacy, nonlinear protocols, head placement, and complete quality and lifecycle measurements remain missing.
**Complete decoder: absent. Serving rank: unresolved.**

<a id="puma"></a>
### 6. PUMA

**Primary source.** *PUMA: Secure Inference of LLaMA-7B in Five Minutes*, [arXiv:2307.12533v3](https://arxiv.org/abs/2307.12533v3).
The [cached PDF](../papers/puma.pdf) has SHA-256 `84c82810a16be95b41200ba8e0071c39a8cadb15a45008ff8e75e295e0960973`.
Its arXiv stamp identifies v3, 26 September 2023, while the title page dates the manuscript 27 September.
This audit resolves the version directly from those bytes without changing the older unversioned manifest entry.
Relevant locators are the threat model on p.4, §§4.3–4.5, pp.5–6, and §5.4/Table 6, p.9.

**Executable scope and differences.** [`PumaThreePartyDecoder`](../python/pllm/protocols/__init__.py#L377-L379) and registry entry `PumaSharedKvState` are pending interfaces.
No PUMA-specific executable implementation appears in the inspected inventory.

The source uses three-party replicated sharing and permits semi-honest corruption of no more than one computing party.
It protects both the model and client input under that honest-majority assumption.
Its evaluation uses ring64 fixed-point arithmetic with 18 fractional bits and protected truncation.
Those operations differ from PLLM's W8A8/ring32 decoder and client-local nonlinear computation.

**Measurements and missing contracts.** The primary paper supplies generation examples and author-reported LLaMA-7B costs in its own workload.
The title's five-minute result does not establish sustained decoding throughput under the proposed contract.
No local PUMA execution, numerical comparison, resource measurement, or measured failure exists in the inspected evidence.

Three-party transport, replicated arithmetic, secure embedding, nonlinear approximations, state transitions, and output disclosure all need executable contracts.
Exact parity or a predeclared quality criterion and complete lifecycle measurements remain missing.
**Complete decoder: absent. Serving rank: unresolved.**

<a id="nexus"></a>
### 7. NEXUS

**Primary source.** *Secure Transformer Inference Made Non-interactive*, [NDSS 2025](https://www.ndss-symposium.org/wp-content/uploads/2025-868-paper.pdf).
The [reviewed PDF](../papers/r21-nexus.pdf) has SHA-256 `c809182345f27f76e21d983d97dffa0d086dbcd17dabc265f0df9fb6ca9b70f2`.
Locators are §II.A, p.2, §III, §IV/Algorithm 2, p.7, softmax on p.8, and Appendix D, p.17.
On p.8, Equation (4) defines normalized exponentiation.
The following unnumbered formula supplies repeated squaring, so the two locators must not be conflated.

**Executable scope and differences.** [`PackedEncryptedSoftmax(method="repeated_squaring")`](../python/pllm/nonlinear/__init__.py#L25-L49) supplies only a bounded CKKS operator.
[`ckks_softmax.py`](../python/pllm/runtime/ckks_softmax.py) uses a public shift of `1/8` and three unrelaxed reciprocal iterations.
Inputs must lie in `[-1/8,1/8]`, and widths are at most 32.

The original uses client/server HE with model and input privacy against a static semi-honest adversary.
Its secure Argmax output differs from complete logits for arbitrary local sampling.

The local operator has no bootstrap, protected attention, private persistent state, or decoder token loop.
Its framing does not establish circuit privacy, authenticated transport, or malicious-result verification.
[`NexusNoninteractiveTransformer`](../python/pllm/protocols/__init__.py#L246-L248) remains pending.

**Measurements and missing contracts.** The [two-profile synthetic cohort](../docs/evidence/research-encrypted-softmax-cost.json) checks 738 approximate outputs across eighteen fresh processes.
At width 32, each profile uses 260.19 MB setup frames and 25.19 MB online frames for three rows.
These are operator bodies, not whole-request wire traffic or RSS.

The [clear range audit](../docs/evidence/research-encrypted-attention-range-qwen25.json) rejects all 48 inspected positions before encrypted candidate execution.
That result rejects this local range, not NEXUS.
Complete ciphertext coverage, source-compatible approximation bounds, refresh, output functionality, state, and matched quality and resource measurements remain missing.
**Complete decoder: absent and locally unadmitted. Serving rank: unresolved.**

<a id="thor"></a>
### 8. THOR

**Primary source.** *THOR: Secure Transformer Inference with Homomorphic Encryption*, [ePrint 2024/1881](https://eprint.iacr.org/2024/1881).
The [cached PDF](../papers/thor.pdf) has SHA-256 `ddb57004b56ee8e00beffc4f563fd0921242753fda6c704065d4900b792439cd`.
The checked primary record reports CCS 2025 and revision 12 September 2025.
The cached PDF's exact revision correspondence remains unspecified.

Locators are §§3–4, §5.1/Algorithm 3, pp.9–10, §6.1, p.10, §7, pp.11–13, and Appendix C.
In this text, §6 describes the system and §7 supplies evaluation.

**Executable scope and differences.** `PackedEncryptedSoftmax(method="square_normalize")` uses THOR's degree-15 exponential and square/normalize structure.
It fixes `delta1=1`, `delta2=2`, narrow input bounds, and unrelaxed reciprocal iterations.
It omits the source's general relaxation schedule, matrix layouts, and bootstrapping.
The current backend cannot bootstrap CKKS.

Source §6.1 states input privacy against an honest-but-curious server and keeps decrypted results with the secret-key owner.
That statement alone does not establish model confidentiality or malicious-result verification.
Source §6.2 puts embedding calculation at the client.
The audited evaluation concerns BERT, so an autoregressive decoder needs an explicit functionality adaptation.
[`ThorHomomorphicTransformer`](../python/pllm/protocols/__init__.py#L261-L263) remains pending.

**Measurements and missing contracts.** The same [two-profile cohort](../docs/evidence/research-encrypted-softmax-cost.json) supplies bounded THOR-derived measurements.
Its numerical identity differs from the NEXUS-derived profile, so operator timings do not establish a general winner.
The accepted family manifest supplies separate bounded replay and decoder rejection.
The [range audit](../docs/evidence/research-encrypted-attention-range-qwen25.json) is a clear-input rejection, not a THOR system failure.

Source-compatible packing, approximation and refresh schedules, embedding ownership, complete decoder state, output semantics, and lifecycle measurements remain missing.
**Complete decoder: absent and locally unadmitted. Serving rank: unresolved.**

<a id="cachemir"></a>
### 9. CacheMir

**Primary source.** *Cachemir: Fully Homomorphic Encrypted Inference of Generative Large Language Model with KV Cache*, [arXiv:2602.11470v1](https://arxiv.org/abs/2602.11470v1), 12 February 2026.
The [reviewed PDF](../papers/cachemir.pdf) has SHA-256 `67db77545b62a4ec0f6906706b947492e6d1bcd9a37b84e5afc32445b41c98e7`.
Locators are §§2.2–2.3, pp.3–4, §4 packing, §5 cache maintenance, §6 bootstrapping placement, and §7, pp.10–12.

The reviewed contribution concerns encrypted KV maintenance and cache-aware bootstrapping.
It does not substantiate the rewrite plan's “multi-token prediction” inclusion rationale.
The selected method remains CacheMir.

**Executable scope and differences.** [`EncryptedKvState`](../python/pllm/state/__init__.py#L193-L195) is a pending interface.
No CacheMir executable method or complete encrypted-state implementation appears in the inspected inventory.

The paper assumes a semi-honest client and server, private client input, and a server-held proprietary model.
Its approximate RNS-CKKS representation, encrypted cache layout, and ciphertext refresh differ from exact client-held W8A8 state.
PLLM's clear KV retention mechanisms do not implement CacheMir.

**Measurements and missing contracts.** Source §7.1 measures modules to estimate end-to-end runtime.
It estimates baseline costs from operation counts and measured primitive costs.
Those author-reported estimates are not matched complete-request observations in PLLM.
No local CacheMir execution, quality comparison, resource measurement, or measured failure exists in the inspected evidence.

Encrypted append/read/capacity transitions, attention packing, key ownership, refresh scheduling, output disclosure, and complete prefill/decode behavior remain missing.
Approximation quality and complete client/provider lifecycle measurements are also necessary.
**Complete decoder: absent. Serving rank: unresolved.**

<a id="rosetta"></a>
### 10. ROSETTA

**Primary source.** *ROSETTA: Efficient and Accurate Privacy-Preserving LLM Decoding via Hybrid CKKS/TFHE Evaluation*, [arXiv:2609.16915v1](https://arxiv.org/abs/2609.16915v1), 15 September 2026.
The [reviewed PDF](../papers/rosetta.pdf) has SHA-256 `b4c7195d32808881b2da8b3a6876b571d8215bbc02aaafaa78ef327aa27fb4e8`.
Locators are §2.4, p.5, §§3.1–3.3, §4, and §§5.1–5.4, pp.10–13.
This audit cites the reviewed preprint without substituting a later proceedings version.

**Executable scope and differences.** [`RosettaCkksTfheAssignment`](../python/pllm/passes/__init__.py#L155-L157) and registry entry `RosettaSchemeConversion` are pending interfaces.
No executable ROSETTA method appears in the inspected inventory.

The source uses approximate CKKS arithmetic, TFHE segmented lookup, scheme conversion, and scheme-aware operator selection.
Its client/server model assumes honest-but-curious parties, confidential model weights, private input, and public architecture.
PLLM's bounded CKKS softmax and garbled lookup operators do not supply these conversions or this system contract.

**Measurements and missing contracts.** Source §5.1 uses measured module costs to estimate end-to-end runtime.
Section 5.3 reports one decode iteration without prefill.
Section 5.4 evaluates model quality through ciphertext-aligned plaintext functions.
These scopes do not establish matched encrypted complete-request costs or PLLM quality.
No local ROSETTA execution, resource measurement, or measured failure exists in the inspected evidence.

Bidirectional scheme conversion, key custody, segmented lookup precision, scheme selection, encrypted KV state, and complete token feedback remain missing.
Full-request timing, traffic, quality, and synchronized role-memory measurements are also necessary.
**Complete decoder: absent. Serving rank: unresolved.**

## Common gates before a ten-system ranking

Every row needs the following contracts under a new frozen version of [COMPARISON-CONTRACT.md](COMPARISON-CONTRACT.md).
Existing artifacts retain their original contracts.

1. **Source and implementation.** Bind the reviewed source, independent implementation, build, component composition, and every departure from the source.
2. **Functionality.** Cover lookup, all model operations, output disclosure, prefill, repeated decode, and persistent state.
   Compare logits where the output contract permits them, active KV, positions, history, pending tokens, and sampling randomness.
   Specify EOS, continuation, cancellation, replay, failure retirement, and acceptance after verification.
3. **Numerical behavior and quality.** Bind representations, scales, rounding, overflow, approximation ranges, and ciphertext precision.
   Use exact comparison where justified, or freeze a task-quality criterion before cost comparison.
   Token agreement alone does not establish complete state equality.
4. **Security.** Name protected assets, actual roles, allowed collusion, preprocessing trust, leakage, channels, randomness, and integrity guarantees.
   Keep public-weight, private-model, verified, and semi-honest comparison classes visible.
5. **Complete measurement.** Fix checkpoint, prompts, output allowance, repetitions, hardware, provider capacity, and each party's shared 100/40 Mbps access limits.
   Freeze round-trip delay and the initial placement of checkpoints and artifacts.
   Measure public setup, private preparation, prefill, decode, and cleanup for cold-client and warm-public cases.
   Report admission-to-first-token time, prefill time, later-token throughput, directional client traffic, all-link traffic, CPU, and provider count.
6. **Memory eligibility.** Measure the complete client and every helper through finalization.
   Apply the maximum observed client peak to the 256 MiB, 2 GiB, and 16 GiB host-memory tiers.
   Keep device memory and the High-GPU profile separate.
   Measure phase and lifecycle peaks for every role and host, including idle state and setup owners.
   Calculate simultaneous aggregates from synchronized observations rather than sums of unrelated peaks.

No row currently passes all six gates.
Missing measurements remain unknown rather than zero, and admission estimates remain distinct from observed memory.
The defensible present result is a source-qualified coverage audit with bounded adaptations and explicit implementation gaps.
A ten-system performance ranking requires further independent implementation, qualification, and matched measurement.

## Read-only validation

The audit resolved all 67 local link occurrences, including section and source-line anchors.
All sixteen linked JSON documents parsed successfully.
HTTP HEAD requests returned status 200 for all ten external primary-source links.
Those responses establish link availability, not fresh identity of the linked PDF bytes.
The PDF and installed-file comparisons above supply the separate byte-identity checks.
