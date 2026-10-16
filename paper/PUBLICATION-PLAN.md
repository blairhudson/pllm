# PLLM two-paper publication plan

Decision date: 16 October 2026. Status: execution started; neither paper is ready
for submission. This is the consolidated publication strategy, manuscript
scaffolds, critique, and delivery plan agreed with the author.

## Execution update: evidence and revised decision

The original scaffolds below remain the design record. The current comparison
rules are normative in `COMPARISON-CONTRACT.md` and `comparison-contract.v1.json`;
current gate status is in `CLAIMS-AND-EVIDENCE.md`.

- **Paper 1 now has executed framework evidence.** Installed-wheel conformance
  covers the tiled-prefill extension, incompatible configurations, exact generated
  state, independent FP32 references, the ordinary benchmark and SDK/gateway.
  Its exact pinned-Qwen Listing 1 also completes through both entry points.
- **Six mechanism families have explicit implementation cards.** The supplement
  retains the original 26 passing tests and failed two-byte equalization assertion.
  A frame-aware repair passes 20 focused checks while preserving that failed record.
  LogRow's original algorithm locator and the larger retained setup are documented.
- **Fresh Listing 1 replay closes its child-process provenance gap.** Actual
  provider package/native identities and complete before/after model-file locks
  bind the Qwen SDK/gateway run to a fresh-target build. The first archive retains
  its narrower digest-only/parent scope. Separate fresh build identities remain
  distinct; the full generated and mechanism replays still need their own records.
- **Historical tables are reproducible.** A semantic audit retains all 51 clients
  in four original cohorts, their build identities, metric contracts, per-role
  observations and maximum-client eligibility. It checks historical evidence;
  it does not substitute for the new publication campaign.
- **Paper 2 must narrow its novelty.** Gupta–Katz–Miers, ePrint 2026/1849, already
  combines private embedding lookup, shared public linear inference and local
  nonlinear work. The viable hypothesis is exact decoder/state preservation,
  bounded coupled ownership and a useful measured resource frontier. Generic
  composition or “lightweight client” is not a new contribution.
- **The closest implementation is public.** The pinned `kanav99/dvllm` source
  supports private embeddings, full-head delegation and full client KV. Its dense
  XOR-PIR, fixed-point arithmetic and verification lifecycle differ from PLLM's
  packed DPF/W8A8 choices. `COMPETITOR-QUALIFICATION.md` records the source mapping
  and unresolved gates; existing offset execution with a client-owned head is not
  that complete comparator.
- **The four-provider argument is written and audited within its model.** Its
  maximal allowed static coalitions are P+B0, P+B1, I+B0 and I+B1. Correctness
  is separated from a conditional untimed privacy argument. Actual packed-DPF
  security, timing/error leakage and external deployment remain review gates;
  lifecycle retirement does not establish secure erasure.
- **No new SOTA is established.** Closest-system qualification and fresh matched
  screening/confirmation remain necessary. Hard aggregate-provider CPU/RSS
  enforcement and complete provider lifetime measurements are unavailable in the
  current native benchmark. Link-normalized runs cannot stand in for them.

Final execution order is therefore: finish full fresh-environment conformance;
freeze a reproducible publication release; close the stated measurement and
external-system qualification gaps; run the frozen campaign and confirmation;
then select the strongest supported paper claims. Paper 1 may proceed on its
own evidence. If Paper 2's narrowed contribution or comparison fails, revise the
method rather than manufacture a SOTA headline.

## 1. Purpose and division of contributions

The argument connecting the papers is:

> Sensitive applications need remote computation without surrendering their
> inputs. Private multi-party inference offers that capability under explicit
> assumptions. Its practical costs span clients, providers, preprocessing, and
> communication. PLLM makes those costs and assumptions comparable. That
> comparison exposes a Low-tier frontier. The second paper must demonstrate an
> advance against qualified competitors.

| | Paper 1: framework and benchmark | Paper 2: Low-tier method |
| --- | --- | --- |
| Question | How should private LLM inference be implemented and compared? | How can complete communication improve under a 256 MiB client budget? |
| Contribution | Executable contracts, resource tiers, reusable implementations, reproducible evaluation | A specific design, its correctness/privacy argument, and a confirmed frontier advance |
| Prior work | Representative independently implemented mechanisms and broader related work | Closest complete-system competitors and strongest eligible incumbents |
| Evidence | Framework fidelity, complete component costs, representative end-to-end baselines | Matched serving measurements, ablations, crossover, resource sensitivity, confirmation |
| Headline | Comparable evidence and supported empirical findings | Precise communication/latency improvement under stated constraints |
| Canonical source | `paper/manuscript.md` | `paper/low-memory.md` |

Paper 1 stands alone and need not wait for a new SOTA result. Paper 2 cites the
framework, but also cites the original methods it uses. It restates enough of
the threat model and measurement contract to be readable independently.

Do not withhold relevant public results to manufacture a follow-up contribution.
Paper 1 can acknowledge experimental methods while Paper 2 owns the detailed
design and its new evaluation. Freeze a benchmark version, not a leaderboard
forever: Paper 2 reruns strong current baselines on a matched build.

The existing two-page whitepaper remains an accessible introduction, not a third
arXiv submission. Target 10–12 main-text pages for Paper 1 and 8–10 for Paper 2,
plus references and appendices. These are writing targets, not arXiv limits.

## 2. Shared comparison contract

### Resource tiers

| Tier | Whole-client maximum |
| --- | ---: |
| Low | 256 MiB |
| Medium | 2 GiB |
| High | 16 GiB |

These are PLLM conventions motivated by constrained consumer clients, not an
industry standard or a claim about total available device RAM. Tiers are nested.
Report sensitivity around the ceilings instead of treating nearly identical
peaks on either side of a cutoff as categorically different designs.

Qualification uses the largest complete-client peak across accepted fresh-process
replicates. Include SDK, supervisor, dashboard, relay, audit, and client helpers.
For multiple client processes, define simultaneous ownership rather than adding
independent lifetime peaks. Providers, private files, filesystem cache, GPU/device
memory, and total device memory retain separate scope. Observed RSS eligibility
does not certify conservative runtime admission or a universal memory bound.
Keep admission failures and pressure/swap-rejected attempts as explicit cells.

### Metrics and identity

Retain the seven priority metrics: client RSS/CPU, Preparation CPU/RSS, online
TTFT, N−1 decode throughput, and all-link traffic. Also report full request time,
startup-through-cleanup time, aggregate CPU, public delivery, and disk ownership.
Report base/output-dependent traffic only where attribution is complete; leave
unattributed bodies visible. Do not collapse everything into a universal score.

Fix checkpoint/source, numerical contract, workload, sampling, lifecycle, and
security constraints before ranking. All-link application bodies, TCP streams,
and physical wire are different measurements. Missing Preparation is not measured
zero work. Exact W8A8 equivalence is separate from BF16/FP32 fidelity or task quality.
Support exact-execution and quality-matched tracks with separate eligibility rules.

For predominantly remote execution, the version-one comparison contract fixes
at least 90% of body-linear MACs separately for prefill and decode.
Client-only execution is a computational reference, not an outsourcing winner.
Provider hardware, thread counts, worker count, memory and access capacity remain
explicit. Different trust models and verification strengths do not silently rank
together. Local co-location does not establish independent operators.

### Lifecycle cells

1. Source-present workers, cold client/public-artifact cache.
2. Warm public artifacts, fresh private request/material.
3. Repeated inputs or continuation, explicitly labeled and separately ranked.
4. Source-cold deployment: measure distribution or leave that cost unknown.

Cold delivery, public index/certificate publication, worker provisioning,
checkpoint delivery, renewal, and fallback cannot disappear through amortization.
State the reuse horizon and separately price pre-positioned artifacts.

## 3. Paper 1 scaffold

### Title

**PLLM: a high-performance private multi-party inference runtime and extensible autonomous research harness**

### Abstract scaffold

Private language-model inference allows clients to use remote computation without
revealing their inputs under a stated security model. Its practical cost,
however, depends on more than the protected computation: preprocessing,
communication, model delivery, and client memory can determine whether a system
is usable. We present PLLM, an open-source framework for implementing and
evaluating these tradeoffs. PLLM separates model semantics from numerical
representations, protocols, and execution placement, and measures costs across
participating roles and request lifecycles. We introduce client-memory tiers and
independently implement selected mechanisms from prior work. Evaluation across
**[methods, models, and workloads]** identifies **[two principal findings]**.
PLLM provides executable configurations, correctness checks, and reproducible
evidence for developing private inference systems under explicit resource
constraints.

### Introduction scaffold

Language-model applications increasingly process private documents, conversation
histories, and application state. Remote inference provides computational
capacity, but conventional services receive the information they process.
Encryption in transit protects the connection, not the information from the
service itself. Local inference preserves custody of that information, but
requires the client to accommodate model computation and storage.

Private multi-party inference offers another division of responsibility. A
client distributes protected representations of its inputs among computational
parties, which evaluate a model under an explicit security and collusion model.
In secret-sharing approaches, individual shares conceal the underlying values.
Public linear operations can often be evaluated directly on those shares;
nonlinear operations require additional protocols or trusted-client computation.
Multiple servers alone do not provide privacy: the protocol and permitted
collusions determine the guarantee. **[Cite foundational MPC/private inference.]**

Language models make this division difficult. Generating each token involves
dependent computation, communication, and persistent state. A method may reduce
online latency by moving work into preprocessing, reduce client traffic by adding
inter-provider traffic, or reduce weight storage while increasing temporary
memory. Numerical approximations introduce another tradeoff: a cheaper
computation may no longer evaluate the same model.

Consequently, a fast protected operator does not necessarily produce a fast
request, and an outsourced model does not necessarily produce a lightweight
client. Evaluating practical progress requires a common account of numerical
behavior, security assumptions, resource ownership, and request lifecycle.

We present PLLM to support that evaluation. PLLM compiles model and execution
choices into checked configurations and runs supported compositions through a
shared SDK and benchmark path. It combines independently implemented research
mechanisms with explicit resource accounting and reproducible experiment records.

Research questions:

1. Can different designs share an execution framework without obscuring assumptions?
2. How do complete costs change interpretation of component-level improvements?
3. Which designs remain feasible under constrained client memory?

Expand to roughly 800–1,000 words only where needed. Explain semi-honest versus
malicious behavior, input/output versus model privacy, offline versus online
roles, and software versus hardware trust in plain language. HE, garbling, and
FSS are related/composable techniques, not identical security contracts. Avoid
unsupported claims that existing systems ignore memory or no framework exists.

### Main sections and outstanding work

1. **Introduction:** application problem, multi-party inference, LLM difficulty,
   comparison gap, PLLM, research questions and contributions.
2. **Scope and evaluation contract:** public-weight decoder scope, trust/numeric
   classes, tiers, complete ownership, lifecycle and metrics.
   **[P1-A: freeze rationale, definitions and threshold sensitivity.]**
3. **Framework:** semantic IR, component contracts, source/schedule binding,
   runtime ownership; one architecture diagram and one executable SDK example.
   **[P1-B: extension, conformance, rejected composition, SDK/gateway replay.]**
4. **Reimplementing prior work:** precise mechanism coverage and deviations.
   **[P1-C: primary-source mapping, representative shortlist, measured records.]**
5. **Evaluation:** answer the three questions with matched evidence.
   **[P1-D: complete campaign, variability, blocked cells and current baselines.]**
6. **Discussion/conclusion:** two or three supported findings, limitations,
   and the unresolved Low-tier communication/latency frontier.
7. **Artifact appendix:** exact code/build/configuration/evidence links.
   **[P1-E: immutable publication release and independent clean replay.]**

Contributions: reusable model-neutral execution contracts; resource-tiered
evaluation; independently implemented mechanisms; empirical findings that depend
on complete accounting. Python/Rust is implementation detail, not the novelty.

### Prior-work selection

| Family | Initial candidates | Required characterization |
| --- | --- | --- |
| Verified delegation | Slalom | SDK adaptation; no enclave/attestation reproduction |
| Calibration | SmoothQuant | Separate numeric identity; quality measured separately |
| Nonlinear protocols | Compact, Curl, LogRow; SIGMA/FuseFSS | Bounded complete operators or selected regions; deviations and conversions charged |
| Lookup/packing | FastQuery, BumbleBee | Bounded mechanism adaptations, not full systems |
| Encrypted computation | NEXUS, THOR | Bounded softmax primitives; missing bootstrap/decoder explicit |
| Delegation/state | Maverick; MPCache/CipherPrune | Optional appendix references; security, quality and resource gates explicit |

Select a small main-text set by relevance and coverage, not bibliography size.
CrypTen, Cheetah, THE-X and other systems remain related work unless actually
reimplemented at the claimed scope. Refresh the literature before submission.

Use three evidence classes: end-to-end executable adaptation; executable
component reimplementation; analytical/blocked reference. Every reimplementation
card binds paper version, equations, preserved assumptions, backend/rounding/dealer
changes, omissions, oracle, geometry, setup/offline/online/conversion/delivery costs,
raw evidence, and replay. Say "PLLM implementation of mechanism X," not "system X
performance." Upstream implementations remain specification/oracle inputs, not
runtime dependencies.

### Evaluation and figures

- Framework suite: generated decoder controls, independent references, one
  component extension, source/schedule checks, replay/cancellation/retirement,
  ordinary SDK and gateway example.
- Mechanism suite: small oracle cases and admitted model-shaped cases; full
  gated blocks and conversions, setup-inclusive HE, material/storage rejection,
  and changed-numeric quality. Incompatible contracts keep separate tables.
- Whole-response suite: client-only, two-offset, prepared, and verified prepared
  where admitted. Pin checkpoint, W8A8, workload, cache state, build, threads,
  hardware, and network. Show verified/unverified contracts separately.
- Main visuals: architecture/ownership; implementation coverage; complete-cost
  breakdown; client-memory/traffic frontier with latency; full baseline table.

The results should demonstrate (A) how complete accounting changes conclusions,
(B) how Low changes the feasible frontier, and (C) a measured remaining opportunity,
such as client vocabulary delivery versus recurring remote-boundary traffic.
Do not fabricate these findings if the final campaign does not establish them.

## 4. Paper 2 scaffold

### Title and central claim

**Communication-Efficient Private LLM Inference under a 256 MiB Client Budget**

Candidate method: coupled private embedding lookup and full-head sharing over
immutable boundary-worker weights, composed with the prepared decoder body.
Novelty relative to prior work is not yet established. Avoid "SOTA" in the title
until comparator coverage supports it.

Primary objective: all-link communication. Secondary: complete-request latency.
Low memory is an eligibility constraint, not a claim of the smallest footprint.
Keep decode, TTFT, client CPU and all provider costs visible.

### Abstract scaffold

Private inference can outsource transformer computation while leaving substantial
model-boundary costs at the client. Token lookup and output prediction may require
a large vocabulary matrix, increasing cold delivery and local ownership. We
present **[method name]**, a design that jointly moves these operations to two
non-colluding boundary workers while retaining client-side state and token
selection. The method reconstructs complete logits and preserves the execution
contract of a pinned quantized decoder. Using the PLLM benchmark, we evaluate
**[models and workloads]** against **[qualified baselines]**. Under **[security
and resource conditions]**, the method reduces all-link communication by **[X]**
and complete-request latency by **[Y]**, with a maximum measured client footprint
of **[M] MiB**. We characterize the accompanying decode, provider-resource, and
cold/warm tradeoffs.

### Introduction scaffold

A client can outsource most transformer computation and still retain a
substantial inference footprint. Vocabulary lookup, output prediction,
tokenization, decoder state, and temporary buffers remain necessary somewhere
in the system. For clients with limited memory and access bandwidth, those
remaining responsibilities can determine feasibility and cold-request cost.

The PLLM framework establishes a common basis for measuring these costs under
explicit numerical, security, and resource contracts **[cite Paper 1]**. Its Low
tier limits the complete client footprint to 256 MiB. Meeting this limit does
not identify a single best design: existing configurations trade communication,
latency, local computation, and provider resources.

This paper studies the model boundary. With tied embeddings, the same vocabulary
matrix supports both token lookup and output prediction. Moving lookup alone
leaves the output-head matrix at the client. Moving both operations avoids that
ownership, but a straightforward implementation adds communication and dependent
work during generation.

We present **[method name]**, which combines private lookup and exact shared
output-head evaluation over a shared immutable worker-side matrix. The client
receives the required embeddings and reconstructs complete logits, retaining its
original sampler and decoder state. The design targets lower complete-request
communication within the Low-tier budget.

Our evaluation measures where this exchange is beneficial, where it becomes
costly, and whether it improves on the strongest eligible alternatives.

### Main sections and outstanding work

1. **Introduction:** boundary costs, PLLM evaluation contract, tied ownership,
   coupled design, proposed contribution.
2. **Problem/related work:** freeze eligible methods and objective; identify new
   composition versus reused primitives. **[P2-A: novelty and competitor review.]**
3. **Method:** admission; private lookup; prepared body; shared head; complete
   reconstruction/sampling; retirement. One lifecycle diagram.
   **[P2-B: formal correctness and privacy/composition analysis.]**
4. **Implementation/resources:** immutable weights, frames/scratch/session bounds,
   all five roles, delivery/publication, failure ownership.
   **[P2-C: implementation-to-analysis binding and complete resource checks.]**
5. **Evaluation:** exactness, strongest-baseline comparison, ablations, crossover,
   and normalized provider-resource sensitivity.
   **[P2-D: fresh matched campaign and frozen finalist.]**
   **[P2-E: independent confirmation and evidence audit.]**
6. **Discussion/conclusion:** precise winning scope and measured regressions.
7. **Artifact appendix:** immutable release and Paper 1 citation.
   **[P2-F: final source/evidence archive and submission metadata.]**

The client still owns tokenization, nonlinear work, attention, full KV, complete
logits and sampling. Smaller vocabulary ownership does not imply smaller peak RSS.
Parallel HTTP or a smaller reply width alone is unlikely to support the central
novelty claim. Establish a systems composition contribution if justified; do not
invent cryptographic novelty.

### Comparison and experiment campaign

- Review closest contemporary complete systems and their adversaries, numerical
  contracts and resource assumptions. Independently implement qualifying complete
  paths where feasible. Missing code is not evidence a competitor loses.
  Analytical exclusion needs a valid bound, not an unsupported projection.
- Baselines: maintained prepared; strongest Low-memory, traffic/placement and
  latency/decode recipes; two-offset where admitted; serial/u32 boundary;
  selected method. Clear client-only stays a reference. Preserve topology leaders.
- Initial workload grid: 150/288 input tokens and 8/32/64 outputs; varied held-out
  prompts, fixed-length performance and separate natural-EOS cases; cold/warm
  public artifacts and explicit reuse cells. A second checkpoint is required for
  a model-general result; otherwise narrow to the measured checkpoint.
- Freeze build and configurations; use balanced serial execution on a controlled
  host. Five fresh-process replicates per primary cell is a starting design,
  not proof of adequate uncertainty. Distinct prompts establish workload breadth;
  repeated execution of one prompt does not. Predeclare exclusions and reruns.
- Exact checks: full logits, active KV/positions, history, pending token, selections,
  sampler/RNG, greedy/stochastic trajectories, cancellation and failure. Compare
  against the pinned unprotected numerical contract and separately measure
  independent-reference fidelity and held-out task quality.
- Ablations: coupled ownership; serial/parallel; u32/uniform residues;
  uniform/source-certified row residues; memory lifetimes; publication/delivery.
  Historical ablations are not fresh matched arms.
- Crossover: outputs, cold/warm cache, public-artifact reuse, shorter/longer inputs.
- Resource sensitivity: the candidate adds two workers. Equal per-party caps add
  aggregate link capacity, so state allowed collusions, charge all providers and
  include an aggregate-resource-normalized comparison for latency. Native stream
  pacing remains local emulation. Add another access profile and controlled RTT
  if the claim extends beyond the measured 100/40 setting.
- Independent confirmation begins only after finalist and primary claims freeze.
  Preserve cohorts, output identities, rejected attempts and uncertainty separately.

Existing 150+8 confirmation (23.70% less traffic, 24.18% shorter requests,
224.13 MiB largest client peak) motivates this direction. It also reports 32.68%
slower decode and 8.00% later online first text, uses source-present workers, and
reruns selected finalists rather than every earlier incumbent. These historical
numbers do not fill the new paper's result placeholders.

### SOTA claim gate

Paper 1 defines a fair comparison; Paper 2 must win it. Narrow the claim by
functionality, numerical contract, security/collusion, client budget, remote-work
requirement, provider resources, lifecycle and workload. Definitions must be
application-motivated, not chosen to eliminate stronger methods retrospectively.

Before external coverage is adequate: "best among evaluated compatible
configurations." After it is adequate, a possible sentence is:

> Among **[defined contemporary class]**, our method achieves the lowest measured
> **[primary metric]** under **[resource/security/workload conditions]**, preserving
> **[reference computation]**, with **[regressions]**.

A Pareto improvement does not establish minimum communication or universal
superiority. Correct execution tests are not a privacy proof. If the method fails
the novelty or performance gate, revise the method/claim; do not force a SOTA paper.

## 5. Scaffold-driven critique and decisions

1. **Infrastructure needs findings.** Select Paper 1's research questions and
   figures before expanding the method inventory. A small well-supported set wins
   over a long list of loosely benchmarked papers.
2. **Memory is a constraint for Paper 2.** Communication is primary; complete
   latency is secondary. Vocabulary removal alone is not a peak-RSS improvement.
3. **Novelty, privacy, and SOTA are separate gates.** Maintain independent reviews
   of original contribution, composition analysis, and competitive evidence.
4. **Component coverage cannot close system comparisons.** Add end-to-end
   competitor qualification. Unsupported or unavailable methods remain unresolved.
5. **Cold lifecycle and additional workers are central.** Put crossover and
   resource-normalized sensitivity in the main evaluation, not footnotes.
6. **Freeze methodology, refresh baselines.** Paper 2 has its own matched build
   and measurements. Separate historical cohorts supply context, not missing arms.
7. **The papers must stand alone.** No duplicated central result, hidden public
   baseline, unassigned arXiv identifier, or publication-dependent circular proof.

## 6. Final execution plan

### Paper 1

| Step | Deliverable | Exit gate |
| --- | --- | --- |
| 1 | Introduction, questions, contributions | Private multi-party inference and need for PLLM are understandable |
| 2 | Comparison contract and tier rationale | A new method can be evaluated without rewriting the rules |
| 3 | Curated mechanisms and primary-source cards | Every benchmark maps to actual implementation/scope |
| 4 | Conformance, complete costs and end-to-end evidence | Matched replayable records answer the questions |
| 5 | Manuscript organized around figures | Every main figure supports a finding |
| 6 | Reviewed paper and standalone artifact | No unresolved placeholder in a published claim |

Publication bar: useful independently reproducible framework/benchmark paper,
with no requirement to establish a new SOTA result.

### Paper 2

| Step | Deliverable | Exit gate |
| --- | --- | --- |
| 1 | Prior-art map and novelty statement | Contribution beyond renamed primitives/configuration tuning |
| 2 | Comparator class and primary objective | SOTA has a meaningful eligible population |
| 3 | Correctness/privacy/resource analysis | Implemented composition matches claimed guarantees |
| 4 | Frozen workloads, controls, exclusions and ablations | Confirmation cannot select its own winner |
| 5 | Matched full-response measurements | Wins survive complete accounting |
| 6 | New confirmation cohort and independent audit | Frozen method retains the headline result |
| 7 | Method paper and separate artifact | Novelty/SOTA wording matches demonstrated scope |

Publication bar: specific new design, justified composition contract, and confirmed
improvement over strongest qualified alternatives.

## 7. Artifacts and submission

- Paper 1 is the canonical framework citation. Paper 2 initially cites its working
  manuscript explicitly, then replaces that reference with the assigned identifier.
- Each paper gets an immutable source tag, build/environment lock, exact configs,
  raw records, failed/blocked-cell manifest, claim/evidence index and replay scripts.
  Archive a release with a persistent DOI when feasible. Website leaderboards stay
  mutable; publication evidence is frozen separately.
- Public fixtures may be released; private prompts/salts/credentials and restricted
  checkpoint weights are excluded. Document legitimate source acquisition.
- Canonical Markdown emits PDF, website MDX, a rebuild-source ZIP and standalone
  citeproc-resolved arXiv TeX. Keep generated outputs ignored. Test each extracted
  archive in a clean directory without repository inputs or bibliography tools.
- Submit Paper 1 first; insert its real public identifier before submitting Paper 2.
  Initial category proposal: cs.CR primary, cs.DC cross-list where warranted.
  Check current endorsement/category requirements; author approves metadata/license
  and submission. arXiv is dissemination, not peer review.
- Official packaging/submission references:
  <https://info.arxiv.org/help/submit/index.html>,
  <https://info.arxiv.org/help/submit_tex.html>,
  <https://info.arxiv.org/help/cross.html>.

## 8. First implementation slice

Save this strategy; turn both scaffolds into canonical working manuscripts;
establish the claim/evidence and competitor checklists; keep retained historical
evidence under its original scope; wire distinct reproducible outputs and checks.
Then prioritize Paper 1's contract and Paper 2's novelty/competitor review before
launching more measurements. Track completion in `paper/IMPLEMENTATION-STATUS.md`.
