# Implementation and evidence status

## Current framework-paper checkpoint — ten-method study

The approved scope is the six-page framework rewrite in
[FRAMEWORK-REWRITE-PLAN.md](FRAMEWORK-REWRITE-PLAN.md), with genuine PLLM
implementations of the ten selected methods. **Zero of ten methods currently
qualifies for the complete-system comparison.** The earlier completed conformance
milestones below cover specified framework and component slices, not this larger
study. The manuscript remains a draft, not a submission-ready paper.

Completed foundations:

- Source-locked method specifications and CPU-first decisions are recorded in
  [methods/README.md](methods/README.md) and
  [methods/DECISIONS.md](methods/DECISIONS.md).
- The bounded GKM decoder checks full logits, active KV, history and pending
  selections against independent oracles. Its repaired installed-wheel evidence
  passes 206 Python and 44 native checks, with two optional checkpoint skips;
  [the decoder reference](../docs/research/gkm-decoder-reference.md) describes these checks.
  Its original in-process evidence retains that scope.
- The subsequent bounded GKM composition runs through ordinary Pipeline/Experiment,
  two party-local worker processes, SDK, gateway, CLI and response benchmarks.
  Request-end verification precedes every model-derived output. The final cached
  wheel passes 547 Python and 117 native checks plus Clippy; all 362 installed
  package files match before and after testing. See
  [the SDK reference](../docs/research/gkm-sdk-reference.md). This generated-checkpoint slice
  does not admit the official Qwen checkpoint or qualify publication performance.
- The subsequent [explicit GKM Decoder-capacity slice](../docs/research/gkm-decoder-capacity.md)
  passes 640 installed Python and 134 native checks plus Clippy, with all 362
  installed files matching. Six generated two-worker cases preserve 20,376 complete
  logit/KV values beyond the Capsule vocabulary ceiling. The original full-checkpoint
  preflight rejects host admission at 10.83 GiB required versus 8.11 GiB safe
  headroom. A later [pinned Qwen response](artifacts/gkm-sdk/qwen-response.md) passes
  fresh admission and executes four input tokens plus two verified outputs through
  both workers. Client peak RSS is 1.78 GiB, sampled simultaneous client/worker RSS
  is 3.79 GiB, and observed swap growth is zero. This single functionality sample
  does not supply independent full-logit/KV, source-model quality or matched-cost
  evidence.
- The subsequent [ordinary CLI quality smoke](artifacts/gkm-sdk/quality-smoke.json)
  uses the reusable `examples/papers` factories and three shared Qwen prompts.
  GKM matches 3/3 prefill and 9/9 same-token decode reference selections; prepared
  W8A8 matches 2/3 and 9/9. Worst absolute decode logit errors are 1.865 and 6.121.
  GKM executes both workers with final verification; the control uses its local
  clear-kernel quality path. This small diagnostic supplies bounded source-model
  agreement, with complete-state equality, broader quality and matched serving
  costs still open. Replay uses the standard command in
  [the factory README](../examples/papers/README.md#completed-quality-smoke).
- The separate [Maverick batch capsule](../docs/research/maverick-batch-reference.md)
  implements bounded two-claim algebra with complete P/Q/Qaux validation,
  response-fixation barriers and allocation/retirement checks. Code-distance,
  dual-LPN, model numerics and complete SDK qualification remain open.
- The [PUMA truncation/PRSS reference](../docs/research/puma-truncation-reference.md)
  checks the source AES schedule and replicated arithmetic. Its live payload
  ledger now retains exported-buffer charges through erasure and deallocation.
  Public unwrapped range proofs and complete protected graph/SDK execution remain
  open. D6 approves the named source-derived Q18 interpretation, not an unavailable
  evaluated-build identity.
  The separate [multiply/truncate specification](reproduction-contracts/puma/README.md)
  proves the four-corner product interval and a closed source-ordered PRSS schedule;
  14 bounded oracle tests pass. The native core now composes that multiplication
  and truncation under one party-local owner and the existing payload ledger.
  Frozen transcript/cursor parity, full-shape exported APIs, allocation failures
  and cancellation are checked. The same owner now executes public tensor
  programs for approved Q18 SiLU/GELU with shared branch selection, eager-polynomial
  interval admission and last-use register retirement. The same native owner
  additionally executes protected reciprocal/rsqrt, shifted softmax and centered
  LayerNorm with public affine parameters. Secret exponents remain shared; public
  conditional range proofs cover every rounding. Independent stochastic oracles,
  128-lane tiles and failure/erasure checks pass. D10 additionally approves the
  separate `puma.rmsnorm-epsilon1-q18.v1` profile, implemented as uncentered RMSNorm
  with integer-one epsilon and distinct rescales. Zero-row and source-profile
  identity controls pass. Half-split rotary additionally preserves two separately
  truncated public coefficient products per output. Register-slot reuse and
  shape-aware allocation additionally admit checked 896/1,024-value RMSNorm,
  896-value LayerNorm and 512-value softmax rows under the same 256 KiB party cap.
  The later bounded `PumaThreePartyCpu` composition binds checkpoint/position
  metadata and complete semantic operations through three authenticated workers,
  ordinary SDK/gateway and response benchmarks. D9's complete private prefix is
  re-executed for every selection; only final fixed-count IDs are disclosed.
  Generated Qwen2/Qwen3/Llama controls include nonzero attention/MLP prefix
  dependence, private feedback and failure retirement. Client publication and
  directed peer traffic remain charged. The shared full-Qwen source exceeds this
  bounded envelope; full-checkpoint ranges, complete hidden-state oracles,
  source-model quality and matched publication costs remain open. See the
  [composition contract](../docs/research/puma-truncation-reference.md#bounded-sdk-composition).
- PUMA's ordinary quality path now scores final-token agreement against FP32
  through fresh full-prefix three-worker requests; hidden-logit metrics stay
  unknown. Complete generated decoders exercise 4,864-wide MLP contractions
  with one final truncation. That slice's pinned common-Qwen metadata rejects
  admission, so it supplies no full-checkpoint quality result.
- The explicit `PumaThreePartyDecoder(weights="public")` adaptation streams
  source-certified Q18 files and computes complete public-weight products on
  replicated activations. Batched lookup predicates and stable argmax preserve
  private feedback; default private-weight identities remain. The common Qwen
  metadata now clears the compiler-view gate through compact activation spans and
  bounded tensor loops. Instruction-derived sequence/AES pricing admits public
  4+2 metadata under the original ceilings; 16+2 rejects its lookup-only sequence
  floor. Native-derived source-header limits also admit ordinary 4+2 SDK memory
  preflight. Streamed source row certificates and RMSNorm row proofs now admit
  the first attention product. Guarded shared softmax retains the source's strict
  cutoff and retained-domain rounding while charging its extra product. Source
  publication then rejects layer-zero post-attention RMSNorm. The
  [attention-range slice](../docs/research/puma-truncation-reference.md#complete-attention-row-and-residual-energy-proofs) certifies complete
  softmax mass and residual energy under the original arithmetic. Pinned publication
  clears that RMSNorm and next rejects layer-zero SiLU before worker launch.
  The [guarded-activation slice](../docs/research/puma-truncation-reference.md#guarded-piecewise-activations) then
  preserves the approved SiLU/GeLU interior arithmetic and zero/identity tails
  through private preselection. Pinned publication clears the first MLP and rejects
  layer-one RMSNorm at the propagated residual range. Full-checkpoint ranges, complete
  work admission, quality and response costs remain open. See the
  [public-weight contract](../docs/research/puma-truncation-reference.md#explicit-public-weight-adaptation).
  The [completion plan](methods/puma-completion-plan.md) orders complete cost
  qualification, whole-model range handling, execution scaling, pinned-checkpoint
  conformance and shared benchmarks, with a first-day feasibility decision.
  The approved checked-range composition now combines private membership checks
  and request-wide final acceptance with tiled full-vocabulary lookup/selection
  and authenticated peer streams. Generated independent integer/AES controls cover
   complete hidden tensors and later-feedback rejection. The
   [checked-execution reference](../docs/research/puma-truncation-reference.md#explicit-checked-range-and-compact-execution-policies) records 462
   passing Python checks, 218 native checks and 64 generated oracle cases. A separate 4+2 pinned
  publication reaches layer-eight attention, then rejects the product bound before
  worker launch with zero observed swap growth. The
  [complete feasibility account](methods/puma-feasibility.md) identifies additional
  metadata, live-owner, protected-work and directed-traffic gates. Full-Qwen
  execution, shared 64+8 capacity, quality and serving measurements remain open.
  The [trie/attention follow-up](../docs/research/puma-truncation-reference.md#trie-lookup-and-complete-attention-vector-certificates) adds
  provenance-bound trie lookup, worker-recomputed Q/K energy certificates and
  shared typed quality-outcome reports. Its broad installed wheel passes 656
  Python checks with eight skips; the bound-intersection follow-up passes 59
  targeted SDK checks. Native validation includes 70 complete generated oracle
  cases and 962 negative controls. Both pinned trie-policy 4+2 publications reject
  layer-six RMSNorm before worker launch, with zero observed swap growth.
  Compact nonlinear constructors and the bounded body-layer executor retain
  separate native tests and remain unselected by the outer decoder.
  Further PUMA implementation is parked at that RMSNorm admission gate.
- The [FuseFSS scalar specification](reproduction-contracts/fusefss/README.md)
  freezes typed descriptors, a Protocol-3 compiler and independently checked GELU
  tables. The [native scalar follow-up](artifacts/fusefss-scalar/local-v1/README.md)
  now composes the frozen ring50/Q12 GELU descriptor with shared vector DCF,
  universal public-boundary lookup and fresh consuming Beaver products.
  Independent replay checks 58,400 compiler gates and 43,544 protected outputs,
  including all 38 exact-rational encoding cases. The public specialization uses
  1,372 logical private bytes plus a 12,296-byte public table copy per party/lane;
  fixed public copies and 512 DCF evaluations per party remain charged. Separate
  engineering reviews cover provenance, allocation and failure retirement.
  This bounded native slice has no authenticated role transport, complete gate
  set, signed38 frontend proof, model execution or selectable SDK component.
  The separate [gate-primitives follow-up](artifacts/fusefss-gates/local-v1/README.md)
  adds protected Boolean execution, exact floor conversions, B2A/Phi/products and
  fixed-chunk complete public-table lookup. It passes 239 native and 66 Python
  tests, with independent replay of 66 selected cases and 18,560 table outputs;
  six of the selected cases are local external encodings. Exported owners retain
  their reservations, final budgets are zero and 31 archive mutations reject.
  Table replay charges 1.428 GB private material and 6.565 GB public table copies
  despite its 2.14 MB logical peak. The separate
  [composed-gate slice](artifacts/fusefss-composition/local-v1/README.md) executes
  SiLU, nExp, protected softmax, private normalization indexing and LayerNorm/RMSNorm
  through party-local schedules and external fresh-material feeds. All 275 frozen
  cases pass independent checking; six remain local external encodings. Public
  range promises support conditional no-wrap proofs, not checkpoint admission.
  Original table owners, cumulative public copies, private material, frames and
  logical live bounds remain separate. Width-768 normalization executes; full
  width-4,096 schedules reject the unchanged instruction ceiling. Complete-model
  ranges/work, authenticated roles and SDK/model qualification remain open.
   The later [bounded decoder and SDK](../docs/research/fusefss-sdk-reference.md)
   adds semantic learned-position and dense-RoPE profiles, checked source import,
   a separate offline dealer, two authenticated online holders and encrypted
   one-use preprocessing tapes. Five generated decoders match 918 complete
   semantic observations. Generated SDK/gateway execution, cancellation and
   ordinary CLI benchmarks pass. Explicit Decoder capacity adds shared streamed
   weights, file-backed replay and externally authenticated tape metadata. The
   opt-in D14 profile passes pinned Qwen 4+1 numeric/work admission; actual trained
   request acceptance remains unmeasured. The integrated follow-up passes 1,025
   native and 816 installed Python checks plus two three-child Decoder tests.
   Standard outputs and reusable factories are linked from
   [the SDK benchmarks](artifacts/fusefss-sdk/README.md). Full trained checkpoints,
   source-model quality and matched performance remain unqualified. The
   [implementation plan](methods/fusefss-completion-plan.md) retains complete
   trained execution, independent parity, quality and performance gates.
  Existing scalar, shared-rescale
  and masked-wire references retain their own numerical and execution scopes.
- The local cold lifecycle observer has independently checked, portable tiny
  evidence: three clients, 289 samples and eleven process instances, with exact
  state agreement and zero observed swap growth. This is measurement-infrastructure
  qualification, not the publication performance campaign. See
  [the lifecycle record](../docs/research/publication-lifecycle-reference.md).
- Framework SDK/gateway conformance, historical evidence audits and standalone
  PDF/arXiv builds retain their separately recorded identities below.

Remaining critical path:

1. Extend GKM's first admitted full-checkpoint response with independent complete
   numeric/state oracles, broader activation coverage and source-model quality.
   Qualify original-model and common-Qwen workloads under their frozen contracts.
   Preserve source arithmetic, final-window acceptance and failure retirement.
   Seeded transport and broader generation policies retain separate contracts.
   The shared gate is [methods/sdk-integration.md](methods/sdk-integration.md).
2. Complete the other nine source-faithful methods. Resolve source ambiguities,
   protected-state and token-feedback contracts, and HE refresh/backend admission.
   Existing adaptations do not substitute for complete implementations.
3. Freeze the publication workloads, quality criteria, cache lifecycles, provider
   resource conditions and network profile. Measure client and all-link traffic
   by phase, role instance counts/peaks, simultaneous provider/host memory, prefill,
   accepted first output, decode and complete lifecycle costs under the declared
   100/40 Mbps conditions. Keep unsupported and resource-blocked cells explicit.
4. Execute matched fresh-process campaigns and independent confirmation. Audit
   numeric/state fidelity, ownership, resource limits, exclusions and replay.
5. Replace the main manuscript with the focused rewrite and accepted results;
   complete scientific/author review, current-source review, release/CI evidence,
   PDF inspection and standalone arXiv replay.

The separate Low-tier method paper additionally retains its composed-security,
closest-competitor and new-confirmation gates. Writing and build work cannot
close missing method or measurement evidence.

## Initial two-paper scaffold — 16 October 2026

The complete strategy is saved in `PUBLICATION-PLAN.md`; per-claim evidence and
the initial comparator qualification queue are in `CLAIMS-AND-EVIDENCE.md`.

- Drafted the framework introduction, private multi-party vocabulary, three
  research questions, tiers, numerical/security tracks and lifecycle rules in
  `manuscript.md`. Tier rationale/sensitivity remains P1-A work.
- Preserved the old measurements in historical appendices. E1/E2/E3/A1 map to
  P1-B/P1-D/P1-E; none is silently marked completed by rewriting prose.
- Created `low-memory.md` with the coupled-boundary design, linear reconstruction
  argument, resource ownership and matched-evaluation scaffold. P2-A through
  P2-F remain open; pairwise non-collusion does not close the four-provider analysis.
- Added separate PDF/MDX/rebuild and standalone arXiv-source outputs for the two
  technical manuscripts. Both are visibly working drafts.

Validation for this slice:

- `npm --prefix docs run papers`: framework PDF seven pages, method PDF four
  pages, whitepaper two pages; all US Letter. First-page layouts inspected.
- Full documentation workflow: 263 Node tests and six Python paper-build tests
  pass; all 439 publication records pass the content/link check.
- Both extracted arXiv archives compile with cached Tectonic resources, without
  repository inputs. Tests also verify distinct canonical rebuild sources,
  resolved citations, PDF copies, and untracked generated outputs.
- Both extracted Markdown source archives independently rebuild their own
  technical PDF with the documented `--pdf-only` command.
- `npm --prefix docs run typecheck` and Ruff on the builder/tests pass.
- Two optional checkpoint-dependent paper network tests skip because
  `PLLM_RUN_REAL_QWEN25` was unset. This does not establish that the cached source
  is unavailable.

Next substantive work at that checkpoint: finalize P1-A's comparison specification and P2-A's
primary-source novelty/competitor review. Then select P1-C's small main-text
mechanism set and freeze the publication campaigns. This slice collects no new
checkpoint, serving, security or model-quality measurements.

## Comparison, source review and conformance follow-up — 16 October 2026

- `COMPARISON-CONTRACT.md` and its versioned JSON freeze the comparison rules,
  explicit 192/224/256/288/320 MiB Low sensitivity, source/cache lifecycles,
  normalized-resource targets and unsupported measurement cells.
- `PRIOR-ART-REVIEW.md` identifies Gupta–Katz–Miers ePrint 2026/1849 as the closest
  mandatory comparator. Private lookup plus shared linear delegation is prior art;
  the method draft now isolates exact execution and resource/lifecycle ownership.
- `SECURITY-AND-CORRECTNESS.md` gives the conditional untimed four-provider
  argument and exact reconstruction. Separate automated document/code review closed three
  claim corrections; 12 bounded witnesses pass. Actual DPF security and runtime
  timing/error leakage retain separate review gates.
- `HISTORICAL-RESULTS.md` and its generated JSON audit 51 original clients in four
  separate cohorts. Metric contracts, role CPU sums, integer-byte peaks, explicit
  recipe labels and maximum-client qualification have 23 focused regressions.
- `IMPLEMENTATION-EVIDENCE.md` summarizes generated conformance and links ordinary
  test entry points. The main extension checks 29,592 exact logit/KV values and
  independent FP32 errors below the declared 0.1 fixture tolerance. The historical
  two-byte equalization framing failure remains distinct from its later repair.
- Listing 1's Qwen SDK/gateway functionality passes at 40+8 tokens with matching
  text and zero observed swap growth. Its first archive does not establish actual
  provider-child imports or retain the full model-file lock. A later replay
  supplied those checks, as summarized below.

The drafts now contain scoped historical results rather than empty numerical
headline slots. New publication campaign, complete-system competitor qualification,
independent confirmation and final author/release review remain open. Neither
draft claims a new SOTA result.

## Historical conformance follow-ups — 16 October 2026

- A separate fresh installation passed 180 trajectory array pairs (88,776 values),
  six binding rejections and twelve boundary checks. This was local testing with
  cached dependencies, not independent reproduction.
- The later generated SDK/gateway run passed 29 tests and checked 29,592 exact
  logit/KV values against its untiled control, with separate upstream FP32 checks.
  A family run passed 46 tests and bounded operator checks. Their original
  benchmark outputs remain under `artifacts/conformance/`.
- Equalization checks now account for encoded `server_ns` separately from
  arithmetic and non-timing framing. The repaired case and 19 accounting controls
  passed. The original failure's uncaptured frames prevent retrospective attribution.
- `artifacts/listing1/qwen-fresh-v2/` freezes a separate successful 40+8 Qwen
  SDK/gateway replay from the earlier fresh-target release chain. Text, actual
  provider imports and complete model-file locks agree; observed swap growth is
  zero. It is functionality evidence, not a new performance comparison.
- `COMPETITOR-QUALIFICATION.md` pins the public GKM author implementation. Dense
  XOR-PIR, remote full-head execution and full KV are prior art. Original fixed
  point, an exact-W8A8 adaptation and a packed-DPF control have separate labels.
  Complete-decoder qualification and matched costs remain open.

Build receipt bundles and archive-verifier tooling have been retired. New work
uses standard uv/Maturin builds, normal tests, reusable experiment files and
ordinary benchmark outputs. Historical observations do not qualify current code.
Fresh publication-cost cohorts, external reproduction, concrete cryptographic
review and author/submission approval remain open.

## Historical implementation and evidence notes

The earlier paper described the prepared public-weight baseline:

- client-owned plaintext, token boundaries, nonlinear state, and sampling;
- offline Preparation computation and acknowledged correction upload;
- sealed one-use inventory at Inference;
- packed prefill and persistent decode transport;
- native exact-ring matrix execution; and
- immutable text-free dashboard records.

The October 5 papers used distinct case studies:

- `docs/evidence/wan-tps-qwen25-2026-10-04.json`: one matched four-candidate
  Qwen2.5 cohort under enforced 100/40 Mbps access and 40 ms added round-trip
  delay. Seed-first dispatch improves offset decode throughput 1.94×; prepared
  overlap improves request throughput 17.3%. Online bodies do not decrease.
- `docs/evidence/slalom-prepared-topologies-cold-cpu-2026-09-26.json`: one
  cold response per topology. Prepared aggregate CPU is 34.66 s versus the
  two-worker control's 28.65 s; verified prepared uses 142.83 s.
- `docs/evidence/qwen3-4b-client-paged-2026-10-05.json`: one 16+8-token
  prepared response, 421.31 MB client/dashboard lifetime peak RSS, no new swap.
  It has no matched resident-client control or independent quality comparison.
- `docs/evidence/preparation-memory-qwen3-4b-2026-10-05.json`: two fresh
  processes per mode, 144 stages and 71 rows per stage. Paged Preparation reduces
  median peak RSS 6.24×, with matching correction content, 12.41% more load-plus-
  issuance CPU and 3.63 GB private snapshot disk. This is an isolated probe.

The papers introduce the agent-facing research workflow; they do not evaluate
autonomous discovery rates. Local link emulation is not an Internet deployment.
Full wire, independent providers, broad quality and tenfold whole-system gains
remain separate gates. Historical runtime and BFV records stay in the evidence
archive and are not combined with these cohorts.

The model-neutral compiler binds untransformed Qwen2, dense Qwen3 and the
text-only Qwen3.5 decoder, among other checked adapters. The public prepared,
client-owned, two-worker offset, and Freivalds-verified prepared choices bind
the same compiled stages; verified stages require one-use verifier material
before session execution. The provider checks plan and body commitments at
session start. Proprietary protocols remain separate. Pinned Qwen2.5-0.5B and
Qwen3-0.6B checkpoints pass separate
**clear native-kernel** prefill-to-decode functionality tests. Qwen3-0.6B's
W4A4 path differs from the FP32 reference on two checked prompts, and
its W8A8 path differs on a short prompt. These tests do not establish real-model
generation quality, a prepared-protocol benchmark or protected full-decoder
execution; see `docs/evidence/qwen3-0.6b-reference-probe-2026-09-24.json`.
The independent, opt-in two-prompt prefill quality benchmark records 0/2 W4A4
and 1/2 W8A8 top-token agreement on each of the pinned Qwen2.5-0.5B and
Qwen3-0.6B checkpoints. Its local compiled clear-kernel scope and separate
checkpoint/cohort digests are retained in
`docs/evidence/qwen2.5-0.5b-reference-quality-2026-09-24.json` and
`docs/evidence/qwen3-0.6b-reference-quality-2026-09-24.json`; it is not a
provider, full-generation or model-quality claim.

`docs/evidence/slalom-freivalds-tiny-baseline.json` and
`docs/evidence/slalom-freivalds-tiny-verified.json` retain one completed run
each on the same generated tiny workload. They demonstrate optional verified
transport functionality only: no warmup, real-model check, PlanLock digest,
performance conclusion, or malicious-Preparation guarantee follows. Neither
cohort can be merged with the historical Qwen2.5 performance study.

`docs/evidence/slalom-prepared-topologies-2026-09-26.json` separately records
one pinned real Qwen2.5 W8A8 response per four selectable role graphs, with
the same fingerprint, input/output counts and covered online bodies. The
verified path sent no extra online stage bodies but added 92.81 MB of covered
initial material; offline CPU, full wire and full-response compute-cap
admission remain unmeasured. The generated Qwen3.5 path now also has a
separate pinned official-checkpoint prefill/decode diagnostic and two-child
request at `docs/evidence/qwen35-4b-text-reference-2026-09-26.json`;
W8A8's 2/4 same-token selection agreement does not establish generation quality.
