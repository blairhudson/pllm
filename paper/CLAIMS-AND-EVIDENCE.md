# Publication claims, evidence, and remaining gates

Working index, 23 October 2026. This document tracks the two canonical drafts and
`PUBLICATION-PLAN.md`; it is not a completed publication audit. A checked existing
cohort retains its own build, workload, topology, numeric and measurement identity.
No historical record supplies missing replicates in a new campaign.

## Paper 1 gates

The approved [framework rewrite](FRAMEWORK-REWRITE-PLAN.md) expands the empirical
scope to ten genuine independent PLLM reimplementations faithful to the selected
sources. **Trained-checkpoint response execution: 1/10 (GKM). Publication-qualified
methods: 0/10.** The historical P1-B/P1-C conformance slices do not close this
study. The matched ten-method campaign remains planned. Substantial protocol,
numeric, state, security, or execution adaptations require author approval before
implementation. Components and generated decoders cannot replace incomplete
methods in the serving table.

| Gate | Claim or deliverable | Existing starting points | Work required to close |
| --- | --- | --- | --- |
| P1-A | Fair comparison contract and nested Low/Medium/High budgets | **Normative definitions complete:** `COMPARISON-CONTRACT.md`, `comparison-contract.v1.json`; archived Low sensitivity uses observed maxima | Fresh campaign and sensitivity near Medium/High ceilings remain empirical work; hard aggregate-provider CPU/RSS enforcement is unavailable in the current native driver |
| P1-B (legacy E1) | Reusable executable components | **Historical conformance slice executed:** clean-build replay passes 29 tests, 29,592 exact state values, independent FP32 references, ordinary benchmark and SDK/gateway; all 31 child identities verified. Listing 1 replay in `artifacts/listing1/qwen-fresh-v2/` binds actual provider imports and complete before/after source locks | The defined local slice is complete, not current-build or ten-method acceptance. Independent external reproduction belongs to P1-E. Historical v1 child imports remain unknown |
| P1-C | Ten source-faithful independent method implementations | Source-locked contracts in `methods/`, bounded GKM trained execution, generated PUMA/FuseFSS integration, and native/compiler milestones. The earlier six-family slice passes 46 tests and ten operator/replay pairs with all 26 child identities verified | Complete each selected method's trained decoder, ordinary SDK/gateway/benchmark path, source/numeric/security mapping, state and failure controls. Preserve six-family results as component evidence, not full-method acceptance |
| P1-D (legacy E2/E3) | Findings on fidelity, complete costs and resource feasibility | Four separately audited historical cohorts, 51 clients, original raw samples in `HISTORICAL-RESULTS.md` and `artifacts/historical-results-v1.json`; generated conformance/reference results | Fresh matched publication campaign, workload breadth and uncertainty; full lifecycle and provider accounting; representative task-quality evaluation; preserve blocked cells |
| P1-E (legacy A1) | Reproducible publication | Historical standalone paper builds, versioned experiment definitions, ordinary tests and standard benchmark outputs; standard uv/Maturin wheel builds | Independent reproduction, release, claim/author review and current PDF inspection. The paper command is `npm --prefix docs run papers` ([build instructions](../CONTRIBUTING.md#run-the-documentation-site)); successful rendering does not establish scientific completion |

The initial scaffold's blanket “execution pending” status is superseded by these
specific conformance records. The newer historical table audit checks 51 original
clients across four separate cohorts; it does not execute their tensors anew or
turn them into the new publication campaign. The older two-prompt quality findings
remain historical only. Exact-state parity and task quality retain separate gates.

## Framework claim-to-evidence matrix

Paths below identify existing code, experiments, tests, or ordinary reports. Code
documents a mechanism, tests document a bounded correctness claim, and reports
document their own execution identities. None supplies another category by itself.

| Question or proposed claim | Manuscript location | Existing evidence and exact scope | Remaining boundary |
| --- | --- | --- | --- |
| CacheMir immutable cache append? | [§3.2](manuscript.md#32-model-representation-and-components); [Appendix D](manuscript.md#appendix-d-implementation-contracts-and-controls) | Landed `22bbf91f`: [shared append constructor](../crates/pllm-compiler/src/interleaved_cache_append.rs), [tests](../crates/pllm-compiler/src/interleaved_cache_append_tests.rs); immutable retained parents, staged paired K/V, checked graph/solver/trace admission and independent clear geometry controls; included in the `98bedd49` combined regression | No trained calibration, encrypted kernel/backend, private decoder or complete-method qualification |
| Maverick request-global verification? | [§3.2](manuscript.md#32-model-representation-and-components); [Appendix D](manuscript.md#appendix-d-implementation-contracts-and-controls) | D45: [request owner](../crates/pllm-core/src/maverick_batch/request.rs), [tests](../crates/pllm-core/src/maverick_batch/request_tests.rs); two ordered batches, four masked 64×64 claims, both global fixation barriers and atomic outputs. 15 request tests, 20 existing integration cases and three compile-fail controls | Raw-field ownership/correctness only; no BF16/Q8 decoder, generation, SDK or security qualification |
| SIGMA width-four LayerNorm? | [§3.2](manuscript.md#32-model-representation-and-components); [Appendix D](manuscript.md#appendix-d-implementation-contracts-and-controls) | D43 LocalClearRoot4: [native parent](../crates/pllm-garble/src/masked_wire_reference/typed/layernorm4/parent/runtime.rs), [controls](../crates/pllm-garble/src/masked_wire_reference/typed/layernorm4/norm13_tests.rs), [literal records](../crates/pllm-garble/src/masked_wire_reference/typed/layernorm4/native_oracle_cases.rs); private party-share outputs, 22 literal records (21 distinct triples), within a broader 976-check package gate, not 976 cases | No wider/inherited-input or SDK activation; joint-transcript privacy remains unqualified |
| NEXUS complete clear normalization? | [§3.2](manuscript.md#32-model-representation-and-components); [Appendix D](manuscript.md#appendix-d-implementation-contracts-and-controls) | D46: [row normalization](../crates/pllm-core/src/row_normalization_reference.rs), [core tests](../crates/pllm-core/src/row_normalization_reference_tests.rs), [compiler controls](../crates/pllm-compiler/src/row_normalization_reference_tests.rs); explicit Eq.5 width-768, 32×1024 row-head binary64 contract, no epsilon, zero-energy refusal; 11 new and seven legacy checks | Clear reference, not CKKS profile adoption or an encrypted decoder |
| THOR clear FFN composition? | [§3.2](manuscript.md#32-model-representation-and-components); [Appendix D](manuscript.md#appendix-d-implementation-contracts-and-controls) | D46: [polynomial schedule](../crates/pllm-core/src/polynomial_schedule.rs), [clear FFN](../crates/pllm-compiler/src/clear_ffn_region.rs), [tests](../crates/pllm-compiler/src/clear_ffn_region_tests.rs); 2×4 artifact-call composition before residual/bootstrap; nine unit, one doc and 20 legacy layout checks | No evaluated-author identity, HE execution or complete encoder qualification |
| ROSETTA coefficient reference? | [§3.2](manuscript.md#32-model-representation-and-components); [Appendix D](manuscript.md#appendix-d-implementation-contracts-and-controls) | [Segmented affine reference](../crates/pllm-core/src/segmented_affine_reference.rs), [tests](../crates/pllm-core/src/segmented_affine_reference_tests.rs); explicit 8,192-coordinate coefficient grid, six clear-reference tests | Not HomComp, conversion, a protected kernel or a complete round trip |
| BumbleBee clear coefficient phase? | [§3.2](manuscript.md#32-model-representation-and-components); [Appendix D](manuscript.md#appendix-d-implementation-contracts-and-controls) | Separately landed `8afefb0c`: [clear phase](../crates/pllm-core/src/coefficient_phase_reference.rs), [core tests](../crates/pllm-core/src/coefficient_phase_reference/tests.rs), [adapter tests](../crates/pllm-garble/src/coefficient_phase_reference/tests.rs). Nine ordinary tests and two E0616 compile-fail controls; 780,444,118 declared campaign arithmetic, 32 MiB logical owners and per-invocation `2^28` work. Old Delta oracle/counterexample retained | Clear algebra only; no ciphertext/backend, private-share, security, approximate RNS Down equivalence, source authentication or full-method qualification. Excluded from the earlier `3f9cca4a` combined gate |
| Does one configuration reach the public execution surfaces? | [§3.1](manuscript.md#31-experiment-interface); [Appendix D.1, Listing 1](manuscript.md#d1-experiment-interface) | `../python/pllm/configuration.py`: `Pipeline`, `Experiment`, canonical serialization and digests. `methods/sdk-integration.md`; `artifacts/listing1/qwen-fresh-v2/` records the pinned prepared SDK/gateway example | `resolve()` is composition resolution, not checkpoint execution, memory admission or security qualification. Not every method supports every entry point |
| Does the compiler preserve semantic and execution identity? | [§3.2](manuscript.md#32-model-representation-and-components); [Appendix D.2](manuscript.md#d2-model-representation-and-components) | `../crates/pllm-compiler/src/decoder_runtime_schedule.rs`, `../python/pllm/runtime/model_binding.py`: phase steps, source-plan/composition digests, state handoff and bound runtime stages | Schedule completeness does not prove a protected decoder, numerical fidelity or protocol privacy |
| Can a reusable extension preserve decoder state? | [§3.3](manuscript.md#33-an-extension-and-its-tests); [Appendix D.3](manuscript.md#d3-an-extension-and-its-tests) | `../crates/pllm-compiler/src/mlp_tiling.rs`; `IMPLEMENTATION-EVIDENCE.md`; historical outputs in `artifacts/conformance/`. Source/schedule/composition/tile binding, exclusive consumers, generated exact-state and independent FP32 controls | Generated fixtures and the separate Listing 1 functionality replay do not establish task quality or matched runtime performance |
| Which papers define the ten-method study? | [§5](manuscript.md#5-source-selection-and-reimplementation-scope); [Appendix F, source-selection table](manuscript.md#appendix-f-governing-sources-and-method-contracts) | `methods/source-lock.v1.json` pins ten governing PDF hashes and versions. `methods/DECISIONS.md` records approved profiles. `PRIOR-ART-REVIEW.md` and `COMPETITOR-QUALIFICATION.md` map primary sources and upstream code | BumbleBee targets the final NDSS paper, not the differing released Q16 profile. Source-model validation and common-Qwen extensions remain distinct. Further substantial adaptations need approval |
| Does any selected method accept trained-checkpoint output? | [§5](manuscript.md#5-source-selection-and-reimplementation-scope); [Appendix F](manuscript.md#appendix-f-governing-sources-and-method-contracts) | `artifacts/gkm-sdk/qwen-response-v1/benchmark.json`, cohort `ce21d5fc51cfde677946baeeae3f3d9d75d156c61ec598c242d13900a8035eb7`; `artifacts/gkm-sdk/qwen-response.md` binds the pinned Qwen source, wheel and two local workers. Later state controls and sharing screens remain separately identified in `../docs/research/gkm-sdk-reference.md` | GKM's broader source-model quality, workloads, accepted comparison profile and matched serving costs remain open. No ten-method serving rank follows |
| Are native/compiler advances whole-method results? | [§3.2](manuscript.md#32-model-representation-and-components), [§5](manuscript.md#5-source-selection-and-reimplementation-scope); [Appendix D](manuscript.md#appendix-d-implementation-contracts-and-controls), [F](manuscript.md#appendix-f-governing-sources-and-method-contracts) | SIGMA's owning activation, compiler and private binding slices, plus PUMA prepared-tensor owner work, advance reusable execution contracts. Generated PUMA/FuseFSS SDK controls have their own references and ordinary tests in `IMPLEMENTATION-STATUS.md` | Native test counts and unregistered/private bindings are implementation milestones. They do not establish trained output, complete-method acceptance, reviewed cryptography or publication measurements. A timed-out trained FuseFSS attempt supplies no accepted output or assigned failure cause |
| Do metrics charge all owners and phases? | [§6.1](manuscript.md#61-planned-matched-campaign); [Appendix G.1](manuscript.md#g1-planned-matched-campaign) | `../examples/papers/`, ordinary benchmark reports, and `../docs/research/publication-lifecycle-reference.md`. The bounded local observer uses same-vector role/host sums and directed-stream conservation | Sampled RSS can miss peaks and double-count shared pages. Provider lifetime high-water, device/remote-host telemetry and hard aggregate-provider CPU/RSS controls remain separate gaps. An absent metric is unknown |
| Does first text measure accepted generation? | [§6.2](manuscript.md#62-acceptance-aware-metrics-and-planned-figures); [Appendix G.2](manuscript.md#g2-acceptance-aware-metrics-and-planned-figures) | [Acceptance and request semantics](methods/sdk-integration.md#acceptance-and-request-semantics); `artifacts/gkm-sdk/qwen-response.md`; `../examples/papers/README.md`. GKM accepts after final verification. PUMA's bounded private-prefix path discloses final IDs | Buffered text release is not N−1 model decode throughput. ID-only output cannot supply undisclosed logit errors. The ordinary quality command is not a representative task-quality campaign |
| What do retained CPU-loopback observations show? | [§6.3](manuscript.md#63-audited-historical-reanalysis); [Appendix A](manuscript.md#appendix-a-retained-historical-case-studies), [B](manuscript.md#appendix-b-historical-artifact-index), [G.3](manuscript.md#g3-historical-source-reconstruction) | `HISTORICAL-RESULTS.md`, `artifacts/historical-results-v1.json` and original reports. One 24-client tier cohort plus three distinct 150+8/32/64 boundary cohorts; the boundary cohorts share one recorded build but have distinct private salts and workloads. Raw samples and windows remain cohort-specific | No pooling, new trials, missing-replicate replacement, ten-method ranking, equal provider capacity, WAN/GPU, price, energy, independent operators or cryptographic review |
| Can another researcher reproduce the claim? | [§7](manuscript.md#7-reproducibility-and-limitations); [Appendix H](manuscript.md#appendix-h-reproduction-and-remaining-evidence) | `../CONTRIBUTING.md`, `../examples/papers/README.md`, reusable `.py` factories, normal tests and ordinary reports. Git history preserves earlier source | Release freeze, independent reproduction, author/claim review and layout inspection remain open. No new receipt archive, copied source tree or custom completion driver is required |

The manuscript's title does not establish high performance or autonomous research
productivity. A matched runtime comparison and any claimed automation evaluation
need their own evidence. This revision adds no measurements or method ranks.
The five earlier component gates coexist at `3f9cca4a`; they are not a combined
SDK/native benchmark or performance, quality or security comparison. Trained
response coverage remains 1/10 and publication qualification 0/10; the matched
ten-method campaign is unknown/unmeasured and tenfold performance is unmeasured.
At exact revision `3f9cca4ad6d1455728bce58eee4c3023b75bfa39`, a separate
four-package core/garble/compiler/python regression passed **3,423 unit/integration
executions and 162 doctests**, with 17 normal ignores and **one explicit legacy
N-N six-map exclusion**. Strict four-package `--lib --tests` Clippy passed;
no source fixes or tracked changes were made. This is regression qualification,
not a performance, quality, SDK, security or release-CI result. It excludes later
BumbleBee `8afefb0c`, documentation `0fb00b2f`, unlanded CacheMir SOURCE04 and
the unfinished PUMA sealed path; it is not latest-Main qualification. Pending
PUMA owning-reader/history work and CacheMir source gates supply no additional
completed-method evidence. Those exclusions describe that earlier gate only.
On 25 October 2026, the later gate at exact revision
`98bedd49bfda3d2bd00f3b3e7a480f298b72791a` included landed BumbleBee and CacheMir:
**3,451 unit/integration executions plus 164 doctests = 3,615 checks**, zero failures,
17 existing ignores and the same single explicit N-N six-map exclusion; strict
four-package core/garble/compiler/python `--lib --tests` Clippy passed.
BumbleBee's nine ordinary tests ran once each; its two compile-fail controls are
already included in 164. Its D47 `2^30` repeat allowance covers only that subset
(780,444,118 declared work); `2^28` per invocation and 32 MiB owner caps are unchanged.
This supersedes the earlier source coverage, not its historical results; counts are not additive. It supplies no SDK, trained-model, physical-memory, performance, security or release-CI qualification, and does not complete PUMA sealed execution. Local raw evidence is retained under the review worktree's `target/multi-paper-post-cache-bb-slot-a-{tests,docs,clippy}.log`; these are development logs, not published campaign measurements.
**Later scoped PUMA integration at `0d9dcaaa7628c2f364fbeb39e6fe696560a62c92`:**
raw module-filtered logs record 780 compiler and three Python boundary passes
(783 total), zero failures/ignores, with 540 compiler and 60 Python tests excluded
by those filters. Four-package core/garble/compiler/python strict `--lib --tests`
Clippy passed; no BumbleBee campaign replay occurred. The Python entry exercised
expanded execution, not the D49 compact SDK route. Separately, the prior 1,615-pass
candidate gate includes [sealed runtime controls](../crates/pllm-compiler/src/replicated_decoder/compact_request_storage_tests.rs):
native compact four-to-five-token execution through the original Worker produces
`[1,0]` and validity 1; early/later sticky rejection masks both outputs. Those
counts are not added to 783. The [owning sealed path](../crates/pllm-compiler/src/replicated_decoder/compact_request_sealed.rs)
is a bounded private native vertical, not trained/full-fit/SDK qualification.
D49 registration, 76 pure/scanner tests and gateway test definitions do not
establish execution of the ordinary compact SDK. This evidence qualifies the
committed revision, not the dirty ROOT compiler drafts or latest working tree.

**26 October 2026 — bounded compact SDK at `c6341b3ed16934769ebfa1aa85f188b0788a8882`.**
Explicit public-weight, checked-range, per-token lookup, expanded-Argmax HTTP
execution now passes the [generated-source gateway controls](../tests/test_replicated_compact_gateway.py):
the admitted client publisher and all three workers witness retained compact
storage, with two accepted outputs/private feedback and execution cancellation.
This is a trusted local client gateway receiving plaintext; external workers
retain the existing share boundary. Default expanded identities and fail-closed
compact selection remain covered by [storage controls](../tests/test_replicated_compact_storage.py).
Installed gates: two compact route tests, seven expanded SDK controls and 58 pure
disk/custody checks; earlier 815 native/211 installed-pure checks and strict Clippy
are separate overlapping evidence, not additive totals or native failure-teardown proof.
Qualification binds the `c6341b3e` production postimages, not the later `6a4d4a9`
select refactor. No trained-source fit, wider-model, malicious-source, security or matched-cost claim follows.

**26 October 2026 — D52 readout admission at `9e9ade9b12fc7194116796cafb4629ba20c9039c`.**
The typed registered LastToken mapping corrects the metadata readout envelope to
`[B,V]`; it is not a new arithmetic optimization. Five
[actual symbolic emitted-shape tests](../crates/pllm-models/src/source_readout_shape_tests.rs)
and 12 [source-policy tests](../crates/pllm-compiler/src/replicated_decoder/source_policy_tests.rs)
passed, including the exact 30/31-prefix case within those 12; strict
models/compiler lib+tests Clippy passed. With the actual cached 659-byte config,
each prefix's metadata quote accepted: factory 33,443,680 bytes, working charge
33,451,425 bytes, owner peak 33,500,519 bytes and prepaid work 8,160,144,615;
arithmetic factory entries were zero. Full-prefix normalization, attention, KV,
other shape products, checked overflow and existing 64-MiB/work/descriptor/lane
guards remain. These are metadata quotes, not SourceTensor collection,
whole-request peak/fit, installed SDK or checkpoint execution. The old 30-row
Bounds baseline was **unrun**, not an executed before/after result.

**26 October 2026 — D51 candidate and bounded stochastic oracle, separate scopes.**
The eight-production-Python-file prompt correction remains **unlanded**, with
115 pure passes and source review; the older c634 prompt cohort is not its proof.
Main `2308ca1ffb4faed12e5b0d4dbf6c60acb3bea39b` adds one pure integer-oracle test
file: five tests and Ruff pass; six authorings and six replays are internal work,
not 12 additional test passes. Explicit residue tapes give complete accepted
`world` results `(0,0)` and `(1,1)`; `hello` gives `(4,4)` in three observed
schedules, not universally. SemanticTape checks name uniqueness, shapes,
residues and exhaustion, not native chronological-carry conformance. Universal
SDK proof work remains **in progress**; no new wheel/gateway or trained-result
qualification follows. Campaign, publication and tenfold-memory estimates are
unchanged.

**Isolated HE consumer storage evidence (D48, O850 reviewed).** Upstream
`ca353257` plus Layout patch `c61b57b7`, with four coherent Poulpy v0.9 path
substitutions in isolation, passed five key-free storage tests, 31 filtered;
Main pins remain unchanged. Compilation took 63 s; tests took 0.460 s, output
1353 bytes, observed peak RSS 8,437,760 bytes and cumulative `Swapouts`
104232280 unchanged. Time (60 s) and output (1 MiB) were enforced; memory/swap
were observational, not hard isolation. Typed raw/prepared consumers compile,
but these tests do **not** execute rotation-key adoption. No keys, transforms,
trusted export, full-resource, runtime, security or backend-adoption claim follows.
Ordinary evidence remains under the approved temporary root at
`he-layout-d38-consumer/benchmarks/native/poulpy_refresh/target/d48-61243-*`;
this is not a new complete-method or publication result.

The six-page main-text target and 20-page total guard are distinct layout
requirements. Passing the total-page guard does not establish main-text compliance.

## Paper 2 gates

| Gate | Claim or deliverable | Existing starting points | Work required to close |
| --- | --- | --- | --- |
| P2-A | Original contribution and adequate competitive coverage | **Prior-art overlap identified:** `PRIOR-ART-REVIEW.md`; Gupta–Katz–Miers ePrint 2026/1849 already composes private lookup, public linear delegation and client nonlinear work | Qualify and measure closest complete-system comparators; establish a narrower exact-execution/resource/lifecycle contribution and supported frontier claim |
| P2-B | Exact and private complete composition | [§3.4](low-memory.md#34-correctness-and-conditional-privacy), [Appendix B.4](low-memory.md#b4-correctness-and-conditional-privacy); `SECURITY-AND-CORRECTNESS.md`; four maximal static coalitions, exact DPF/ring/float/state reasoning, conditional untimed simulation; 12 bounded checks and review corrections in `artifacts/security/` | Independent review of the actual packed DPF and implementation side channels; no timing/error, malicious-provider, adaptive-corruption or forensic-erasure theorem is supplied |
| P2-C | Low-tier complete-client eligibility with all providers priced | [§4.2](low-memory.md#42-whole-client-and-provider-accounting), [Appendix C](low-memory.md#appendix-c-implementation-bounds-and-resource-accounts); `docs/research/private-boundary-tiers-reference.md`; `docs/research/private-boundary-decode-followup.md` | Freeze current allocation/build/workload; repeat complete-client maxima; distinguish admission; price setup, source-cold, files and helpers; provider-capacity normalization. The separate tenfold Qwen3-4B 64+8 whole-client target still lacks a matched isolated peak pair; historical Qwen2.5 Low eligibility cannot supply it |
| P2-D | Primary communication and secondary latency advance | 24-client incumbent comparison and separate 150+8/32/64 cohorts | Fresh strong-baseline campaign, varied prompts, ablations, cold/warm crossover, resource/link sensitivity, model-quality checks; freeze finalist |
| P2-E | Independent confirmation | Existing three-finalist 150+8 confirmation is historical supporting evidence | New cohort after the P2-D freeze, complete parity/accounting, predeclared exclusions, independent evidence audit |
| P2-F | Publication identity and replay | Shared publication builder and draft framework citation | Paper 1's assigned identifier, immutable release, generated tables, independent replay, final claim/author review |

## Historical boundary evidence map

| Evidence | What may be said now | What it cannot establish |
| --- | --- | --- |
| `docs/research/private-boundary-tiers-reference.md` | One 24-client 150+8 build/cohort compares the coupled topology with seven prepared recipes; complete-client peaks qualify observed Low | Current-build universal lead, independent operators, same provider capacity, warm/source-cold win |
| `docs/evidence/private-boundary-decode-confirmation-150x8.json` and its `.summary.json` | Nine clients; 78,842,880 values checked; candidate maximum 224.13 MiB; 23.70% lower TCP streams and 24.18% lower request time against that maintained control | All earlier incumbents rerun; globally lowest memory; Paper 2's final comparison; physical wire or Internet latency |
| `docs/evidence/private-boundary-decode-confirmation-150x32.json` and `...150x64.json` | Separate longer-output screens; original controls and identities retained | Short-workload confirmation inherited by longer cells or pooled uncertainty |
| `docs/research/private-boundary-row-codec-benchmark.md` | Modest source-certified head-payload savings, separate publication and delivery costs | Main architectural novelty, overall SOTA or causal memory improvement |
| `docs/research/private-boundary-sdk-reference.md` initial 48+8/150+8 cohorts | Original functionality and traffic tradeoffs | Low-tier qualification from parent/relay-excluding child RSS |

The confirmation's slower decode (32.68%) and later online first text (8.00%)
must accompany its cold traffic/latency results when cited. Zero observed swap
growth is not a claim of zero swap use.

## Initial comparator qualification queue

The dated primary-source review now supplies a targeted qualification map, not an
exhaustive literature survey or a matched-system benchmark. No external entry is
marked defeated or automatically ineligible. Compare
actual adversary/collusion sets, leakage, numerical functions, hardware, complete
cost windows and source availability before setting a disposition.

| Candidate family | Current PLLM coverage | Required next decision |
| --- | --- | --- |
| Maintained prepared, placement/tiling/digit8 choices | End-to-end installed recipes; separate archived cohorts | Freeze best eligible current recipes per metric; include controls in one new build/cohort |
| Two-online-offset and clear client | Installed complete execution with client-owned token/head boundaries | Admit within selected memory/provider budget; clear stays a computational reference; existing offset execution does not cover a two-server remote lookup/full-head comparator |
| Serial/u32 and parallel/residue private boundaries | Installed complete execution | Own-topology ablations; public certificate/kernel options separate |
| Gupta–Katz–Miers open-weight outsourcing | Independent bounded two-worker SDK implementation now accepts trained Qwen output, with full head, client KV and final verification; no publication-qualified matched comparison | Mandatory closest comparator; finish source-model quality, workload and matched-cost qualification. Its source paper's 179 MB storage and 4 GB client VM do not establish whole-client Low eligibility or ineligibility |
| Slalom | Trusted-client Freivalds SDK adaptation, no TEE | Separate verified class and original hardware trust; never label adaptation a complete Slalom reproduction |
| CrypTen/Cheetah/THE-X | Related-work sources; no qualifying complete reproduction in this inventory | Inspect primary source and exact candidate workloads; decide whether a faithful complete comparator is feasible |
| SIGMA/FuseFSS | SIGMA native/compiler ownership slices and bounded FuseFSS generated SDK composition; no accepted trained result or reproduced GPU performance | Complete trained execution and source/numerical/security qualification. Component, generated and timeout records cannot supply serving ranks |
| NEXUS/THOR | Bounded encrypted softmax, no complete decoder/bootstrap | Review full-system applicability and assumptions; bounded-range rejection is not a proof their complete systems lose |
| BumbleBee/FastQuery | Bounded packing/lookup adaptations | Check closest full-system and boundary-placement prior art; charge tied head, conversions, setup and model-quality changes |
| Compact/Curl/LogRow | Bounded nonlinear operators with explicit conversions | Use scoped mechanism comparisons; determine relevant full-system prior art separately |
| PUMA | Bounded generated three-worker SDK composition, including explicit compact checked-public execution at `c6341b3e`; no accepted trained-Qwen response | Complete trained-source admission/execution, wider coverage, quality and matched costs. Bounded generated-route integration does not close method acceptance |
| Maverick, CacheMir and ROSETTA | Selected study rows with source-locked contracts; Maverick has bounded native delegation evidence | Complete source-faithful decoders and ordinary public execution paths; primitive references do not substitute |
| Additional contemporary systems | Dated review includes CipherGPT and recent SoKs | Refresh comparison coverage before submission; do not replace the approved ten methods without author decision |

For each external system, create a reviewed card: bibliographic version and source
pin; implemented equations; supported functionality; privacy and malicious/security
parameters; setup/dealer; client/provider work; numeric equivalence or quality
tolerance; lifecycle accounting; code availability; faithful reproduction plan;
and final evidence-based comparator disposition. Missing implementation is not a
performance result.

The historical closest-comparator gap included placement, not just precision.
Installed `PrivateLookupAndFullHead` requires the prepared body and two additional
boundary roles (`crates/pllm-compiler/src/decoder_boundary.rs`). The current
two-offset baseline therefore does not substitute for a Gupta–Katz–Miers-style
two-server composition that also executes the private lookup and full head.
The separate GKM SDK now supplies bounded two-worker lookup/body/full-head
execution and trained output. That closes an execution gap, not the matched
comparison. Qualify source-model quality, shared access budgets and complete costs
before claiming that four-provider execution leads the eligible field.
The author implementation is public at
[`kanav99/dvllm`, commit `271637be7196d26a5433ca1e62b1a5357e769a5b`](https://github.com/kanav99/dvllm/tree/271637be7196d26a5433ca1e62b1a5357e769a5b).
The subsequent source review supersedes the initial unresolved artifact search.
`COMPETITOR-QUALIFICATION.md` separates original fixed-point execution, a proposed
exact-W8A8 construction adaptation, and a packed-DPF architecture control. Source
availability closes the discovery gap, not numerical, privacy or performance
qualification; no upstream implementation is used as a PLLM runtime dependency.

## Claim ladder

1. **Implemented:** a particular component or composition executes.
2. **Validated:** specified oracles, source bindings, and lifecycle checks pass.
3. **Measured:** a frozen cohort records the stated metric/window.
4. **Confirmed:** a separate predeclared cohort reproduces the finalist's result.
5. **Competitive:** the strongest eligible baselines have adequate coverage.
6. **SOTA within scope:** contemporary coverage and matched evidence support the
   precise minimum/frontier statement, with tradeoffs and exclusions disclosed.

None of these steps automatically proves protocol privacy, representative quality,
physical independence, or an unmeasured deployment. No numeric headline should be
filled from a different step's evidence.
