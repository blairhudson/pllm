# A shorter framework paper

Status: approved study direction; implementation and measurement in progress.
Initial plan date: 16 October 2026.

The author confirmed the full ten-method study after the source audit.
The study must use genuine independent PLLM reimplementations of the selected
papers, with new reusable components where necessary. Substantial changes to a
protocol, numerical contract, state, security model, or execution strategy need
the author's approval before implementation. Component substitutes do not satisfy
this requirement. Rankings and the final manuscript follow complete validation
and measurements.

The author selected CPU-first execution on Apple M-series hardware and
Qwen2.5-0.5B-Instruct as the common model. Preserve original-paper validation and
identify common-Qwen extensions separately. Prefer an existing native Rust HE
backend with the required primitives; another-language backend needs a decision.
See the [recorded implementation decisions](methods/DECISIONS.md).

[`TEN-METHOD-COVERAGE.md`](TEN-METHOD-COVERAGE.md) records the starting coverage.
Six methods have component adaptations, but none currently qualifies as a complete
reimplementation for the proposed comparison. This is an implementation gap, not
a result against the original papers.

This plan applies to `manuscript.md`.
The title stays unchanged.
The target is six main-text pages, with references and a separate supplement.
Use about 2,500 words of main prose.

## 1. Main message

The paper answers one question:

> Which private-inference methods give the best performance for each client memory budget under the same network conditions?

Start with this explanation:

> Private LLM inference divides computation between a client and one or more providers.
> Cryptographic protocols protect the client's input, intermediate values, and output.
> Under each protocol's security assumptions, providers learn no more than the declared leakage.
> Leakage means information that the protocol permits, such as input length.
>
> This creates two opportunities.
> Sensitive workloads can use remote compute without disclosure of their content.
> Independent providers can supply compute without access to that content.
> These providers can include individuals and operators of data centers around the world.

Define collusion as cooperation between parties that combine information the protocol keeps separate.
Explain each method's assumptions about provider behavior and collusion.
Privacy and correct computation are separate properties.

Then introduce PLLM:

> PLLM combines a Rust execution core with Python interfaces and an automated research harness.
> It supplies reusable components for model execution, protected arithmetic, transport, state, and measurement.
> Researchers can use these components to reproduce published methods and compare their costs under common conditions.

Use one diagram for the client, providers, Rust runtime, and experiment harness.
Explain automated experiment generation, resource checks, measurement, and replay in one paragraph.
Give a matched runtime comparison to support the high-performance claim.

## 2. Paper structure

| Section | Purpose | Prose budget |
| --- | --- | ---: |
| Abstract | State the problem, framework, study, and confirmed result. | 120 words |
| 1. Private inference and its opportunity | Define privacy and explain the two opportunities. | 350 words |
| 2. PLLM | Explain the Rust core, component interfaces, and research harness. | 350 words |
| 3. Comparison design | Introduce ten methods, budgets, workloads, assumptions, and metrics. | 650 words |
| 4. Results and tradeoffs | Compare prefill, decode, and traffic. Explain the measured relationships. | 850 words |
| 5. Conclusion | Name confirmed leaders for each budget and objective. State the remaining opportunity. | 150 words |

Use four main tables: method coverage, comparison conditions, phase results, and memory by role.
Use one tradeoff figure with separate panels for the main metrics.
Keep related work beside the relevant method or result.

Put source manifests, build receipts, proof details, full metric definitions, and extended sensitivity results in the supplement.
Keep the assumptions necessary to interpret each main result beside that result.

## 3. Ten-method study

Select ten representative methods through a dated source review.
Cover distinct approaches and the closest recent alternatives.
Freeze the selection rule before measurement.

The selected list comes from the [source review](PRIOR-ART-REVIEW.md).
It is not a performance ranking.

| Selected method | Main reason for inclusion |
| --- | --- |
| Gupta–Katz–Miers | Open-weight outsourcing with private lookup and shared linear computation. |
| Maverick | Private, verifiable linear delegation with public preprocessing. |
| SIGMA | Function secret sharing for private GPT inference. |
| FuseFSS | Fused operations for private inference with function secret sharing. |
| BumbleBee | Two-party transformer inference with combined cryptographic methods. |
| PUMA | Secure multi-party transformer inference. |
| NEXUS | Transformer inference with homomorphic encryption. |
| THOR | An alternative transformer design with homomorphic encryption. |
| CacheMir | Encrypted autoregressive inference with reusable KV state. |
| ROSETTA | Recent encrypted transformer kernels and nonlinear operations. |

Use Slalom, Cheetah, and CrypTen as foundational citations.
Keep Compact, Curl, LogRow, FastQuery, and other component studies in the supporting evidence where relevant.
Complete source qualification must precede each implementation contract.

Current evidence covers six reviewed mechanism families and several measured component adaptations.
It does not establish ten complete, comparable decoder implementations.
The ten-system study therefore includes new engineering and measurement work.
The smaller study of existing PLLM configurations is not a substitute.

For each method, record its source version, security assumptions, numerical rules, and changes from the source.
Produce an independent PLLM implementation of the declared scope.
Compare complete outputs, decoder state, and failure behavior with suitable references.
Measure complete requests before assignment of a serving rank.

Keep component and complete-system results in their stated scopes.
Incomplete methods have an explicit coverage result and no serving rank.
A missing implementation does not establish that the source method cannot meet a budget.
Claim ten complete reimplementations only after all ten pass these requirements.

## 4. Client budgets and memory by role

Keep the existing limits for whole-client host memory.
The tiers are nested maximum budgets available to PLLM.
They are not specifications of a device's total installed memory.

| Tier | Host memory limit | Intended setting |
| --- | ---: | --- |
| Low | 256 MiB | A thin client with a small application memory budget. |
| Medium | 2 GiB | A consumer application with space for more state and selected local computation. |
| High | 16 GiB | A workstation with substantial resources and a model that exceeds local capacity. |

Use a CPU-only client as the common hardware profile.
Add a separate High-GPU profile with one GPU and a proposed 24 GiB device-memory limit.
Give host memory and device memory separate limits and measurements.
Results for the two High profiles have separate hardware labels.

For scale, 200 billion parameters at four bits need about 100 GB for weights alone.
State and workspace add memory.
This example explains the outsourcing opportunity, rather than an evaluated model size.

Use the largest complete-client peak across repetitions for tier eligibility.
Count all client processes and helpers.
State the separate scope of disk, filesystem cache, and device memory.
Apply tier filters to the same observations when the configuration stays unchanged.

### Non-client memory

Client tiers constrain only the client.
Each implementation must also report memory for every required non-client role.
These can include Preparation, Inference, boundary workers, dealers, and transport helpers.
Use the implementation's actual role graph.

For each role, record:

- The number of role instances and their host placement.
- The largest individual instance peak.
- The simultaneous aggregate peak across all instances and their helper processes.
- Separate host-memory and device-memory measurements.
- The phase in which each peak occurs.
- Memory kept while idle between requests.

Measure the same phases as traffic: public setup, private preparation, prefill, decode, and cleanup.
Include model loading, key generation, expanded material, retained state, and temporary buffers.
Give phase peaks and the complete-lifecycle peak.
An offline role remains part of the implementation's resource requirements.

Use this reporting shape for each method:

| Non-client role | Instances | Setup peak | Preparation peak | Prefill peak | Decode peak | Cleanup peak | Lifecycle peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Each required role | Count | Host/device | Host/device | Host/device | Host/device | Host/device | Host/device |
| All non-client roles | Total count | Host/device | Host/device | Host/device | Host/device | Host/device | Host/device |

Each aggregate peak uses memory observed at the same time.
For role r, calculate the maximum over time of its instances' combined memory.
For the non-client total, calculate the maximum over time of all non-client memory.
Do not add independent lifetime peaks and label that sum a simultaneous peak.
Keep the largest per-instance peak visible because each host must accommodate its assigned work.
Also report each host's simultaneous aggregate when several roles share that host.
Report total provisioned host and device capacity separately from simultaneous use.
Unused memory on one host does not increase another host's capacity.

Count each process once in the non-client total.
If roles share one process, report a joint owner unless separate attribution is supported.
State the treatment of shared pages and unified CPU/GPU memory to prevent duplicate accounting.
Keep private-file and artifact storage separate from RAM.

Use measured peaks as observed capacity requirements, with sample intervals and measurement limits stated.
Keep admission estimates and enforced limits separate from measured peaks.
For each repetition, calculate the aggregates before selecting the largest observed peak across repetitions.
Missing simultaneous or phase measurements remain unknown.

The [bounded lifecycle observer](../docs/research/publication-lifecycle-reference.md)
now supplies phase/role/host sampled RSS and conserved stream observations for
the local cold prepared graphs. Three generated-checkpoint trials pass independent
archive replay. Complete-process high-water, device memory, matched provider
budgets, and the real ten-method campaign remain separate gates.

## 5. Measurement objectives

Use shared limits of 100 Mbps download and 40 Mbps upload for each party.
Peers share that party's access capacity.
Specify round-trip time separately and keep the existing delay sensitivity cases.
Choose the main delay case before measurement.
Label local network emulation explicitly.

Prefill processes the prompt.
Decode processes subsequent tokens.
Time to first token includes work before the first output.
It is a separate metric from prefill time.

| Objective | Main measurements |
| --- | --- |
| Prompt processing | Prefill seconds and processed input tokens per second. |
| First output | Time to first token from request admission, including request setup and preparation. |
| Continued output | Decode tokens per second after the first output. |
| Client communication | Separate upload and download MB for each phase. |
| Total communication | All-link MB for each phase, with every directed transfer counted once. |
| Resource eligibility | Complete-client peak host memory and separate device-memory peak. |
| Non-client capacity | Phase and lifecycle memory peaks for each role, each host, and all non-client roles together. |

Divide traffic into public setup, private preparation, prefill, decode, and cleanup.
Keep unclassified bytes visible and reconcile phase totals with the full request.
Specify checkpoint and artifact placement before each case.
Report cold-client and warm-public cases separately.
Warm-public cases keep declared public artifacts and use fresh private material.

Use decimal MB for traffic and binary MiB or GiB for memory.
Identify whether each count covers application bodies, TCP streams, or physical wire.
For N output tokens, decode throughput uses N−1 intervals between the first and last outputs.
Undefined throughput remains unknown.

Keep request time, client CPU, provider CPU, and provider count beside the main results.
The role-memory table is a primary result beside the client-tier comparison.
Use a fixed provider budget for the main comparison.
Resource sensitivity covers extra workers or different hardware.

Keep model, prompt workload, and output allowance fixed within a comparison.
Changed numerical behavior needs a fixed quality requirement before cost comparison.
Separate privacy and verification classes remain visible.
Include a small correctness model and larger admitted checkpoints.
Evidence near the Medium and High limits is necessary for conclusions specific to those tiers.

The [comparison contract](COMPARISON-CONTRACT.md) supplies the detailed campaign rules.
Freeze changes in a new version before measurements start.
Historical results keep their original contracts.

## 6. Results and conclusions

Answer these questions in order:

1. Which methods fit each budget?
2. Which eligible methods give the fastest prefill and decode?
3. Which methods use the least client and all-link traffic in each phase?
4. How much memory must each non-client role and the complete provider deployment supply?
5. What additional resources or assumptions accompany each gain?

Name the best confirmed method for each objective and budget.
Give its primary citation, exact configuration, and comparison group.
For combined methods, cite each contribution and measure its effect separately.
Close with a short recommendation table and one paragraph of limits.

Correlation describes associations within matched groups and does not prove causation.
Keep workload, numerical rules, lifecycle, hardware, and privacy class fixed.
Identify topology changes separately.
Show raw values and sensitivity to removal of each configuration.
Repeated trials do not count as additional method designs.

The [existing relationship analysis](../docs/research/metric-relationships-reference.md) gives a useful control.
Its traffic/decode correlation changes sign when a different topology enters the group.
Preserve this distinction in the new paper.

## 7. Work order and completion

1. Freeze the outline, selection rule, and definitions.
2. Build the ten-row coverage table and identify missing complete-system capabilities.
3. Close correctness and security-mapping gaps for the selected methods.
4. Freeze workload, hardware, quality, and measurement rules, including synchronized role-memory observations.
5. Do the matched campaign and independent confirmation of leading choices.
6. Write the short manuscript from accepted results.
7. Do scientific, language, evidence, and PDF-layout reviews.

The opening and framework sections can proceed while implementation work continues.
The abstract's result sentence and final recommendations depend on confirmed measurements.

Use the shared [writing guide](WRITING-GUIDE.md).
Keep active voice, one topic per paragraph, and consistent technical terms.
Define each necessary domain term at first use.
Keep descriptive sentences within 25 words and instructions within 20 words.
Maintain full-width abstracts in the body font and size.

Completion means a short, readable paper with reproducible results for its stated study.
Every claim about ten complete systems must have complete-system evidence.
