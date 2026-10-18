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
  [the archive](artifacts/gkm-decoder/README.md) retains exact execution identities.
  Its original in-process evidence retains that scope.
- The subsequent bounded GKM composition runs through ordinary Pipeline/Experiment,
  two party-local worker processes, SDK, gateway, CLI and response benchmarks.
  Request-end verification precedes every model-derived output. The final cached
  wheel passes 547 Python and 117 native checks plus Clippy; all 362 installed
  package files match before and after testing. See
  [the SDK archive](artifacts/gkm-sdk/README.md). This generated-checkpoint slice
  does not admit the official Qwen checkpoint or qualify publication performance.
- The subsequent [explicit GKM Decoder-capacity slice](artifacts/gkm-sdk/decoder-capacity.md)
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
- `IMPLEMENTATION-EVIDENCE.md` records generated conformance and six mechanism
  families. The main extension checks 29,592 exact logit/KV values and independent
  FP32 errors below the declared 0.1 fixture tolerance. Original failed attempts,
  including a two-byte equalization framing assertion, remain archived.
- Listing 1's Qwen SDK/gateway functionality passes at 40+8 tokens with matching
  text and zero observed swap growth. Its first archive does not establish actual
  provider-child imports or retain the full model-file lock. Stronger isolated
  replay and clean-cache/fresh-environment release checks are in progress.

The drafts now contain scoped historical results rather than empty numerical
headline slots. New publication campaign, complete-system competitor qualification,
independent confirmation and final author/release review remain open. Neither
draft claims a new SOTA result.

## Fresh-target release follow-up — 16 October 2026

- `artifacts/release/README.md` binds an empty-target, locked offline build to a
  separate fresh installation and generated replay. All 355 package files match
  the new wheel; 180 trajectory array pairs (88,776 values), six binding
  rejections and twelve boundary checks pass. The read-only verifier passes.
- This native binary differs from the historical cached build. The release uses
  existing toolchains, Cargo sources and RECORD-checked Python dependency payloads;
  it is neither an independent attestation nor a clean operating-system image.
  Historical execution and failed/interrupted attempts retain their identities.
- Equalization's exact byte check now separates the actual encoded `server_ns`
  field from arithmetic and non-timing framing. The repaired case and 19 negative
  and accounting controls pass. The original failed frames were not retained,
  so that historical failure remains unresolved. Fresh full conformance still
  requires complete provider shutdown receipts.

## Complete fresh conformance follow-up — 16 October 2026

- A second empty-target build in `artifacts/clean-build/2026-10-16-local-v5/`
  verifies 127 receipt artifacts and 26,231 installed distribution files. It keeps
  its own wheel/native identity and reuses recorded dependency/toolchain inputs.
- `artifacts/conformance/2026-10-16-clean-local-v1/` passes 29 tests, 29,592 exact
  logit/KV values, independent FP32 reference checks, the SDK/gateway and ordinary
  benchmark. All 31 Python children retain complete installed-package identities.
- `artifacts/conformance/2026-10-16-families-clean-v1/` passes 46 tests and the ten
  bounded operator/replay pairs. All 26 child identities pass, including both
  provider roles. Actual equalization frames are retained and reconcile; the old
  missing-frame failure remains unresolved rather than being rewritten.
- `artifacts/listing1/qwen-fresh-v2/` freezes a separate successful 40+8 Qwen
  SDK/gateway replay from the earlier fresh-target release chain. Text, actual
  provider imports and complete model-file locks agree; observed swap growth is
  zero. It is functionality evidence, not a new performance comparison.
- `COMPETITOR-QUALIFICATION.md` pins the public GKM author implementation. Dense
  XOR-PIR, remote full-head execution and full KV are prior art. Original fixed
  point, an exact-W8A8 adaptation and a packed-DPF control have separate labels.
  Complete-decoder qualification and matched costs remain open.

These results close the specified local conformance gaps. Fresh publication-cost
cohorts, external replay, concrete cryptographic review and author/submission
approval retain their separate gates.

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
