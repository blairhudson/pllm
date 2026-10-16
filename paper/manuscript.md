---
title: "PLLM: A Framework and Benchmark for Resource-Constrained Private LLM Inference"
description: "Executable contracts, client-memory tiers, and complete-cost evaluation for private language-model inference. Working framework-paper draft."
author:
  - "Blair Hudson"
affiliation: "deployscience labs"
email: "blair@deployscience.com"
date: "16 October 2026"
documentclass: article
classoption: [twocolumn, letterpaper]
fontsize: 10pt
geometry: [letterpaper, margin=0.72in]
colorlinks: true
linkcolor: black
urlcolor: blue
citecolor: black
bibliography: paper/references.bib
pdf: "paper.pdf"
source: "paper-source.zip"
arxiv: "paper-arxiv-source.zip"
abstract: |
  Private language-model inference lets clients use remote computation without revealing their inputs under a stated security model. Practical cost depends on more than protected computation: preprocessing, communication, model delivery, and client memory can determine whether a design is usable. We present PLLM, an open-source framework that separates model semantics from numerical representations, protocols, and execution placement, and records costs across participating roles and request lifecycles. We define nested client-memory tiers and distinguish executable adaptations of prior work from bounded component references. A fresh-target build and fresh-environment replay preserve 88,776 generated logit and cache values exactly; a separate pinned-checkpoint listing passes through the SDK and gateway. A reanalysis of 51 archived client executions across four distinct cohorts finds different leaders for traffic, memory, and decode speed: the lowest-traffic configuration need not have the smallest footprint or highest decode throughput. These scoped results support executable, auditable comparison rather than a universal ranking. [P1-D: complete the broader matched publication campaign before extending these findings to additional workloads or quality claims.]
---

*Working manuscript; evidence placeholders remain explicit.*

# 1 Introduction

Language-model applications increasingly process private documents, conversation
histories, and application state. Remote inference provides computational capacity,
but conventional services receive the information they process. Encryption in
transit protects the connection, not the information from the service itself.
Local inference preserves custody of that information, but requires the client to
accommodate model computation and storage. The practical question is whether a
resource-constrained client can benefit from remote computation without revealing
its private inputs and intermediate state to an individual service operator.

Private multi-party inference offers another division of responsibility. A client
distributes protected representations of its inputs among computational parties,
which evaluate a model under an explicit security and collusion model. With fresh
uniform additive sharing, an individual share is independent of the secret value.
Public linear operations can be evaluated on shares; nonlinear operations require
additional protocols or a return to trusted-client computation. Secure computation
systems such as CrypTen and Cheetah provide relevant foundations
[@knott2021crypten; @huang2022cheetah].

Multiple servers alone do not provide privacy. The protocol specifies which
parties may collude, what information is disclosed, and whether parties must follow
the computation. A semi-honest party follows the protocol but inspects everything
it receives; a malicious party may deviate. Input and output privacy is also
different from hiding the model itself. Homomorphic encryption, garbled circuits,
and function secret sharing provide related building blocks, while trusted
execution introduces hardware assumptions. These choices do not have identical
security or deployment contracts [@chen2022thex; @zahur2015halfgates;
@gupta2024sigma; @tramer2019slalom].

Autoregressive language models make this division difficult. Prefill processes
the prompt; each subsequent decoding step depends on earlier output. Attention
retains keys and values from executed tokens, usually called the KV cache.
Nonlinear functions, normalization, and numerical conversions add work that a
public matrix product alone does not capture. Vocabulary lookup and output
prediction, tokenization, persistent state, and temporary buffers can remain
substantial client responsibilities even when transformer-body weights are remote.

A method may reduce online latency by moving work into preprocessing, reduce
client traffic by adding inter-provider traffic, or reduce resident weight storage
while increasing temporary memory and file I/O. Approximation and quantization
introduce another tradeoff: cheaper computation may no longer evaluate the same
numerical model. A fast protected operator therefore does not necessarily produce
a fast request, and an outsourced model does not necessarily produce a lightweight
client.

Existing frameworks and protocol implementations address different models,
numerical representations, adversary assumptions, and execution environments.
Recent systematizations already examine reproducibility, standardized benchmarks,
and comparison contracts [@chen2026sok; @das2026comparability;
@albadawi2026sok; @kei2026sok]. PLLM complements that work with executable
compositions, independently implemented mechanisms, and measured allocation and
request lifecycles under a fixed client budget. We present PLLM to make
these choices executable and their costs inspectable. Model adapters produce a
shared decoder representation; checked components specify numerical, protocol,
state, and placement choices. Supported compositions run through a common SDK,
gateway, and benchmark path [@pllm2026].

The paper contributes (1) source-bound execution contracts for reusable decoder
and method implementations, (2) a resource-tiered evaluation methodology,
(3) scoped independent implementations of selected published mechanisms, and
(4) a checked reanalysis of cost and resource tradeoffs in four archived cohorts.
The fresh conformance checks and historical reanalysis have separate evidence
identities. **[P1-D: add the broader matched publication campaign.]**

We ask three research questions: **RQ1**, can different designs share an execution
framework without obscuring their assumptions? **RQ2**, how does complete
accounting change the interpretation of component-level improvements? **RQ3**,
which designs remain feasible under constrained client memory? The baseline
end-to-end paths keep nonlinear computation and decoder state at the client;
additional research operators have their own execution scope. Framework membership
alone is not evidence of a complete private decoder or a new security proof.

# 2 Scope and comparison contract

## 2.1 Functionality, numerics, and security

The core serving comparison concerns public-weight decoder inference: client
inputs, intermediate values, and outputs are protected under each stated role
contract, while model weights are public. Version 1 of the comparison contract
fixes checkpoint identity, numerical representation, workload, sampling, permitted
leakage, adversary assumptions, lifecycle and resources before measurement.
Verified and unverified results retain different contracts. The machine-readable
constants and full rules accompany the artifact as
`paper/comparison-contract.v1.json` and `paper/COMPARISON-CONTRACT.md`.

The common remote profile permits static semi-honest corruption of at most one
provider role, with trusted client/setup and confidential authenticated channels.
More permissive coalition families require separate composition arguments; the
role count alone does not supply one. Public geometry, executed lengths, message
sizes, stage-level access patterns, timing and termination are disclosed. Secret
token indices cannot be relabeled as public access patterns. Client-only execution
is a computational reference, and verified-linear and hardware-trusted methods
retain their additional assumptions. An unavailable competitor remains unresolved.

The exact-execution track compares complete logits, active KV, positions, executed
history, pending selections, and sampler behavior against the same unprotected
quantized decoder. This is separate from fidelity to an independent BF16/FP32
reference and from task quality. A quality-matched track must declare its task
suite and acceptable loss before evaluating changed numerical methods. It cannot
inherit the exact track's identity from matching a few selected tokens.

## 2.2 Client-memory tiers

| Tier | Whole-client ceiling |
| :--- | ---: |
| Low | 256 MiB |
| Medium | 2 GiB |
| High | 16 GiB |

These nested ceilings are PLLM evaluation conventions, not industry standards or
total device-RAM requirements. Their fixed eightfold spacing is a coarse
logarithmic budget grid: Low asks for a footprint well below a GiB; each subsequent
tier relaxes the allowance by the same factor. This convention is not a device
survey or an application-suitability result. Report exact peaks and sensitivity
at 0.75, 0.875, 1, 1.125 and 1.25 times each ceiling. The Low grid is therefore
192/224/256/288/320 MiB. Section 6.3 reclassifies historical observations; behavior
near the Medium and High limits remains unmeasured.

Observed eligibility uses the largest complete-client peak across accepted
fresh-process replicates. The ordinary response worker includes SDK, supervisor,
dashboard, relay, and audit. Any additional client helper belongs in that scope.
For a multi-process client, simultaneous ownership must be measured; summing
independent lifetime peaks does not establish it. Provider peaks, private files,
filesystem cache, GPU allocations, and device memory remain separate. Observed
RSS eligibility and conservative runtime admission are distinct results.

## 2.3 Costs and lifecycle

Primary reporting retains client RSS/CPU, Preparation CPU/RSS, online time to first
token, N−1 decode throughput, and all-link traffic. Also report complete-request
latency, startup-through-cleanup time, aggregate CPU, and artifact/storage costs.
Missing measurements stay unknown; an absent Preparation role is not a measured
zero-cost implementation of preparation.

Separate source-present cold clients, warm public artifacts with fresh private
material, repeated/continued workloads, and source-cold deployment. Public indexes,
certificates, checkpoints, and preprocessing do not become free because they were
positioned before a response. All-link application bodies, TCP-stream bytes, and
physical wire have different scopes. Base and output-dependent traffic are
reported only where attribution is complete.

The eligible frontier fixes the security and outsourcing requirement before
optimizing resource objectives. Predominantly remote execution means at least 90%
of logical body-linear MACs off-client, separately in prefill and decode. This
declares a maximum one-tenth client share of that work, not of total CPU. Count a
logical product once; protocol duplication and preprocessing remain additional
costs. Lookup, output head, attention products and nonlinear work are outside this
denominator. Report 80/85/90/95/100% sensitivity; a per-row counter alone cannot
qualify phase-dependent placement. No universal weighted score replaces the
individual metrics.

More workers can add both trust requirements and aggregate bandwidth. The
per-party profile allocates 100/40 Mbps down/up to every role. The normalized
profile retains client access but divides a fixed 200/80 Mbps provider pool,
four CPU-core equivalents and 8 GiB provider allowance equally among providers.
These are synthetic comparison budgets. Current native execution expresses link
partitions but does not enforce that aggregate CPU/RSS pool; such a run must be
labeled link-normalized only. Full resource-normalized evidence remains a gate.

# 3 Design and implementation

## 3.1 Experiment interface

An `Experiment` combines an immutable `Pipeline`, a `Deployment`, and an `ExecutionBudget`. The pipeline contains the model source and execution components; the deployment specifies role locations; and the budget bounds requests and token counts. A researcher can change a kernel or transport while retaining the workload. Listing 1 shows the Python interface [@pllm2026].

Resolution checks a configuration. Before execution, the runtime also binds it to the model source, weights, numerical choices, stage schedule, and role placement. The compiler rejects unsupported combinations, including those that require an unavailable protected operator. The Python interface, command-line benchmark, and application gateway use the same supported configuration (Figure 1).

    Specification
      Model + Pipeline + Deployment + Budget
          |
    Compilation and binding
      Decoder IR -> validated stages and role placement
          |
    Execution (prepared placement)
      Trusted Client <-> Inference
      Preparation supplies one-use corrections before online work
          |
    Benchmark record
      Configuration + workload + environment + measurements

**Figure 1.** From experiment configuration to benchmark report. Prepared execution uses an offline Preparation service and an online Inference service; the client retains nonlinear computation and decoder state.

``` python
from pllm import (
    Deployment, ExecutionBudget, Experiment, Model,
)
from pllm.kernels import Cpu
from pllm.profiles import MaskedLinearCpu

model = Model.hf(
    "Qwen/Qwen2.5-0.5B-Instruct",
    revision="7ae557604adf67be50417f59c2c2f167def9a775",
)
experiment = Experiment(
    "prepared-qwen",
    MaskedLinearCpu(model, kernels=Cpu(threads=4)),
    Deployment.local(root="local://paper"),
    ExecutionBudget(
        requests=4, max_input_tokens=64, max_new_tokens=8,
    ),
)
resolved = experiment.resolve()
assert resolved.requires_preparation
```

**Listing 1. Experiment configuration.** A pinned model, CPU kernel, local deployment, and request budget. Resolution checks the configuration without running inference. Co-located roles are used here for development.

## 3.2 Model representation and components

Model adapters translate decoder operators, layer identities, and persistent state into a shared intermediate representation (IR). Compiler passes operate on these semantic fields instead of model-specific node names. Qwen2 and dense Qwen3 are among the implemented adapters. Adapter coverage describes the model representation; checkpoint execution and numerical fidelity are evaluated separately.

Python handles experiment orchestration, source resolution, and role lifecycles. Rust implements integer kernels, scheduling, bounded codecs, and one-use material through a PyO3 interface. This division keeps experiment construction in Python while checking numerical and resource bounds at the native boundary. Components select quantization, linear protocols, kernels, caching, verification, and placement; the compiler determines which combinations are executable.

Client-side weight paging binds authenticated private snapshots to local token
lookup and output-head execution without changing decoder semantics. Preparation
paging is also separately selectable. The historical Preparation measurements in
Appendix A predate that installed choice and retain their isolated probe scope.

## 3.3 An executed extension and conformance check

**[E1: generated implementation checks recorded]**

`MlpPrefillTiling` is a concrete extension through the existing interfaces. It
recognizes row-independent gate/up, SiLU, multiply and down-projection regions,
checks exclusive consumers, and executes bounded public row tiles. Full outputs,
attention and KV retain their original semantics. The native contract binds the
source plan, schedule, composition and tile size; incompatible Metal and verified
compositions reject before execution.

A frozen installed-wheel replay runs 29 focused tests for this extension, source
binding and complete nonlinear operators. Generated Qwen2, Qwen3 and Llama fixtures
preserve 29,592 byte-exact tiled/untiled logit and active-KV values over batched
prefill and three same-token decode steps. Independent Transformers FP32 execution
has maximum logit errors 0.042739, 0.051512 and 0.047203 respectively, below the
declared 0.1 fixture tolerance. This second comparison tests quantization fidelity;
it is not an exact FP32 claim or representative task-quality evaluation.

A separate empty-target build and fresh-environment replay check 180 array pairs,
containing 88,776 logit and active-KV values, across full and tiled executions of
the same fixture families. Its different native binary retains a separate identity;
the earlier execution records are not relabeled.

The installed-wheel replay also completes an ordinary response benchmark, the unchanged published
prepared SDK example, and its HTTP gateway path through separate local provider
children. A separate cached-checkpoint replay executes Listing 1's exact pinned
Qwen configuration using the fresh build: SDK and gateway both process 40 input and eight output tokens,
return matching text, and observe zero swap growth. The SDK audit records masked
stages and zero plaintext prompt/token-ID bytes sent. These are functionality
checks, with an in-process ASGI gateway and TCP provider children, not performance
or independent-operator evidence.
The fresh archive binds both provider processes to the installed SDK/native build
and retains complete checkpoint-file locks before and after execution.

`paper/IMPLEMENTATION-EVIDENCE.md` records source/build identities, independent
oracles, commands and retained failed attempts. `scripts/replay_paper_listing.py`
extracts the published listing rather than maintaining another configuration.
**[P1-E: immutable publication release and independent external replay.]**

# 4 Baseline runtime and trust model

The trusted client retains plaintext inputs, activation scales, nonlinear operations, decoder state, and output selection. Token lookup and ordinarily the output head also run locally. An application gateway runs inside this boundary; providers hold public transformer-body weights.

The distributed protocols assume honest-but-curious services that follow the protocol and obey the declared collusion restriction. Preparation may instead be hosted by the client. Co-located processes exercise the implementation but do not provide operator separation. The algebraic privacy model is untimed: public weights, geometry, admitted lengths and semantic stopping are disclosed, while implementation timing and side channels require separate analysis. Transport authentication checks role identities; it does not establish non-collusion. Logical retirement of a one-use owner is also distinct from secure erasure: retained seed copies can regenerate masks. No adaptive-compromise or forensic-erasure guarantee follows from the present runtime.

## 4.1 Two-worker execution

Let $W\in R^{m\times n}$ be a public integer matrix and $x\in R^n$ an encoded activation, where $R$ is a finite ring supported by the numerical configuration. The client forms fresh additive shares $a$ and $b=x-a$. Two workers each receive one share and return its product with $W$. The client reconstructs

$$
Wa+Wb=Wx.
$$

With uniform masks, either share is independent of $x$ in this algebraic model. Seed-based encodings additionally depend on their expansion assumptions. The workers perform two matrix products in total. This is twice the body-linear arithmetic of a clear product, rather than a prediction of total request CPU time.

## 4.2 Prepared execution

Prepared execution separates mask preparation from online evaluation. Fresh client seed material defines an input mask $r$, an output mask $s$, and a one-use ticket for each stage row. Preparation computes $c=Wr-s$ and transfers the correction to Inference before online work begins. The client then sends the ticket and $u=x-r$; Inference returns

$$
Wu+c=W(x-r)+Wr-s=Wx-s.
$$

The client adds $s$, reconstructs the bounded integer result, and performs scaling and nonlinear computation locally. Preparation is idle during the response. Reserved material becomes unusable after use, cancellation, replay, or failure, and subsequent requests require fresh material. Prefill batches rows; decode reuses an authenticated connection.

This arrangement removes one product from the online critical path while retaining two products across preparation and evaluation. Optional client-side Freivalds checks use fresh projections supplied by Preparation, inspired by verified delegation in Slalom [@tramer2019slalom]. PLLM's placement does not use trusted execution hardware, and these checks do not protect against malicious Preparation.

# 5 Related work and reimplementation scope

CrypTen provides secure tensor computation and a machine-learning interface
[@knott2021crypten]. Cheetah develops two-party neural-network protocols
[@huang2022cheetah], THE-X studies homomorphic Transformer evaluation
[@chen2022thex], and Slalom combines trusted hardware with private verified
delegation [@tramer2019slalom]. These works motivate different parts of the
design space; citing them does not establish a PLLM reproduction.

Our representative set covers six families: Slalom-derived verified delegation;
SmoothQuant-derived public equalization [@xiao2023smoothquant]; complete gated
adaptations of Compact, Curl and garbled lookup [@islam2024compact;
@santos2024curl; @heath2024logrow]; SIGMA/FuseFSS nonlinear mechanisms
[@gupta2024sigma; @ma2026fusefss]; FastQuery/BumbleBee embedding and packing
[@lin2024fastquery; @lu2025bumblebee]; and NEXUS/THOR encrypted softmax
[@zhang2025nexus; @moon2024thor]. Their numerical, conversion, setup and transport
contracts differ. The implementation inventory records those differences rather
than assigning whole-system paper performance to a bounded operator.

| Evidence class | What the result establishes |
| :--- | :--- |
| End-to-end adaptation | A stated checkpoint/workload executes through the SDK under the adaptation's contract |
| Component reference | A bounded implemented mechanism passes its oracle and has scoped costs |
| Analytical or blocked reference | A construction, cost bound, or rejection; no completed serving measurement |

Each reimplementation card binds the source version and algorithm, preserved
assumptions, changes in rounding/backend/dealer, omissions, oracle, geometry,
cost windows, and raw replay evidence. Original software is a specification or
oracle input, not a runtime dependency. Full-system external comparisons require
their own qualification. An unavailable implementation is not a defeated baseline.

The complete gated-operator cards illustrate the required precision. Curl's
public Haar table uses exact replacement rescaling; the LogRow-labeled reference
uses the paper's lookup and row-reduction construction while retaining a larger
one-hot setup. Compact adapts density-guided fitting to dyadic polynomial
intervals and Boolean evaluation. All charge private selection, conversion,
multiplication and final rounding. A fresh bounded replay checks 32 independent
integer-oracle outputs and another 32 using fresh material. Their changed profiles
do not preserve the checkpoint's complete logits/KV, and their operator costs
cannot rank the original full systems.

The fresh six-family replay passes 46 tests, including captured-frame traffic
accounting and its rejection controls. Ten bounded operator/replay pairs check
116 exact integer outputs and eight approximate CKKS outputs in total, within
their declared contracts. All 26 child processes retain matching installed-package
start/end identities. The earlier two-byte equalization assertion remains a
separate failed record: its original frames were not captured. The repaired test
reconciles actual timing-field widths while rejecting unrelated framing changes.
These checks establish mechanism functionality, not external paper performance
or complete encrypted-decoder execution.

The 16 October primary-source review, source hashes and exact implementation cards
are `paper/PRIOR-ART-REVIEW.md`, `paper/prior-art-sources.json` and
`paper/IMPLEMENTATION-EVIDENCE.md`. The review distinguishes publication versions
and abstract-only observations from full-text review. LogRow's locator is
Section 5, Construction 1 and Figures 3--5; the retained setup differs from its
Section 6 accounting. The defined P1-C mechanism/conformance slice is complete;
fresh matched cost sweeps remain P1-D work. Broader mechanisms remain appendix
references rather than automatic additions to the measured main-text set.

# 6 Evaluation design

## 6.1 Method

The benchmark runner compares candidates under a common workload and supports grid or seeded-random configuration search. Comparisons fix the checkpoint, encoded model body, prompt cohort, generated-token counts, cache state, hardware, and network conditions. Reports store configuration and source identifiers while omitting private prompts, token IDs, masks, and credentials. Salted prompt-cohort identifiers support matching within a benchmark invocation, not across independent invocations.

Measurements separate online execution, setup-inclusive request work, and cold-first process lifecycles. Aggregate CPU includes the client and all measured roles, including Preparation, over the stated interval. Communication counts cover application bodies rather than complete physical-wire traffic. Decode throughput divides the $N-1$ tokens after the first by their execution time. Peak resident set size (RSS) is measured per process and excludes filesystem cache; role peaks need not occur simultaneously.

Appendix A retains archived repository case studies [@pllm2026], with their
original workload and measurement scope. They motivate the evaluation but do not
supply missing replicates or controls for a new publication cohort.

**[E2: evidence to add]**

**[P1-D: matched evaluation.]** Add repeated, balanced runs with pinned
hardware/software, thread limits, preprocessing/cache policy, and workload lengths.
Report raw paired changes, variability, all-role costs, and blocked cells. Distinct
prompts establish workload breadth; more executions of one prompt do not.
Predeclare exclusions, memory-pressure handling, and finalist confirmation. E2
is the previous draft's repeated-performance gate.

## 6.2 Research questions and planned figures

- **RQ1: framework fidelity.** Generated decoder controls, independent references,
  extension/conformance evidence, and failure retirement. Figure 1 and Listing 1
  connect the public API to the executable contract.
- **RQ2: complete costs.** Compare full operators with their primitives, charge
  conversions and preprocessing, and separate cold delivery from online work.
  **[P1-D Figure 2: complete-cost breakdowns for admitted mechanisms.]**
- **RQ3: resource feasibility.** Compare client-only, two-offset, prepared, and
  verified prepared paths where admitted, with distinct trust classes. Qualify
  Low/Medium/High by complete-client maxima and show the resulting tradeoffs.
  **[P1-D Figure 3/current-baseline table: tier frontier and matched whole-response costs.]**

**[E3: evidence to add]**

For P1-D, add current protected/clear and independent-reference comparisons. Pin
tokenization and sampling; report full logits/state checks and separate numerical
errors, prefill/decode agreement, and held-out task quality. The historical
two-prompt checks in Appendix A do not characterize representative generation.

## 6.3 Audited historical reanalysis

To make the accounting distinction concrete, we also retain a matched 24-client
case study: eight recipes, three fresh clients each, one pinned Qwen2.5-0.5B
W8A8 source, and 150 input plus eight output tokens. All recipes use local shared
100/40 Mbps access pacing and source-present cold clients. A read-only publication
audit checks the original report, frozen source/plan bindings, complete-state
commitments, and individual measurements; it does not rerun their computation.
Table 1 is generated from those original observations.

| Recipe | All-link TCP MB | Request s | Decode tok/s | Maximum client MiB |
| :--- | ---: | ---: | ---: | ---: |
| Maintained prepared | 478.65 | 106.64 | 3.79 | 213.44 |
| MLP tiling | 478.65 | 108.49 | 3.79 | 189.59 |
| Digit8 Preparation | 478.65 | 104.03 | 3.78 | 224.30 |
| Tiling + digit8 | 478.65 | 106.44 | 3.76 | 185.56 |
| Tiling + digit8 + placement | 458.78 | 107.10 | 4.39 | 189.56 |
| Digit8 + placement | 458.78 | 105.97 | 4.39 | 226.50 |
| Serial digit8 issuance | 478.65 | 108.75 | 3.79 | 226.89 |
| Coupled boundary | 367.35 | 82.95 | 1.92 | 210.22 |

**Table 1.** Original 150+8 tier cohort. Traffic, request time, and decode rate are
medians; memory is the largest complete-client peak across all three replicates.
The last row adds two boundary workers, and therefore has a different trust and
aggregate-resource contract. Its values are an explicit cross-topology comparison,
not a same-resource speedup. MB is decimal; MiB is binary. Indexed-tokenizer and
checkpoint delivery, physical wire, and Internet delay are unmeasured.

The cohort records 210,247,680 checked logit/KV values and zero observed swap
growth. It illustrates three different objectives. Tiling plus digit8 has the
smallest client peak; selective placement lowers traffic with a declared 90.45%
per-row remote body-linear share; coupled boundaries save cold delivery but add per-token
work. The latter's aggregate cold-first CPU is 61.87 seconds, versus
53.81--58.55 seconds across the prepared recipes. These windows end at the first
response and exclude cleanup. A shorter request does not imply less total work.

Threshold sensitivity also changes eligibility. At 128 MiB no recipe qualifies;
at 192 MiB only tiling, tiling plus digit8, and tiling plus digit8 plus placement
qualify; all eight fit the observed 256 MiB ceiling. These are reclassifications of
the same measured maxima, not fresh admission or memory measurements. They justify
reporting budgets and sensitivity explicitly rather than declaring one recipe
universally best. They do not make 256 MiB an externally standardized threshold.

The publication audit also preserves three separate boundary cohorts at 150+8,
150+32, and 150+64. Their different builds/cohort commitments remain distinct;
no samples are pooled. Reproduction is
`node scripts/audit_publication_evidence.mjs --check`; exact source hashes,
replicate vectors, all seven primary metrics, and paired changes are in
`paper/artifacts/historical-results-v1.json`. The broader fresh P1-D campaign and
independent model-quality evaluation remain separate gates.

# 7 Availability, limitations, and open questions

PLLM is alpha software distributed as `pllm.run` under Apache-2.0; its Python import
and command are `pllm` [@pllm2026]. This working paper's publication source/build
lock is not yet frozen. The artifact index identifies an earlier reviewed archive,
not the execution revision of every historical measurement.

**[A1: evidence to add]**

**[P1-E: publication artifact.]** Freeze source, build, configurations, raw reports,
failed cells, and reproduction commands. Audit each claim against its record and
replay the release independently. A1 retains the previous provenance gate.

Compiler checks and one-use material handling enforce implementation rules, not
a composable privacy proof. Local child processes establish neither non-collusion
nor independent operation. Local access-link emulation is not Internet latency;
application bodies and TCP streams are not complete physical wire. Resource tiers
do not establish model quality or protection against arbitrary malicious parties.

The Low tier poses a concrete follow-up question: can an exact execution design
reduce complete communication and request time without exceeding client capacity
or relocating uncharged costs? Client vocabulary delivery versus recurring remote
boundary work is one candidate tradeoff. A new method must beat strong current
eligible baselines under this contract; the framework does not predetermine its
winner. **[P1-D: support the final opportunity statement with measured ownership
and traffic breakdowns.]**

# 8 Conclusion

PLLM connects model semantics, numerical and protocol choices, resource ownership,
and measurement through executable contracts. Generated extension checks preserve
exact decoder state, while independent references assess a different question:
numerical fidelity. The audited historical cohorts show why communication,
memory, and decoding speed need separate leaders and explicit resource budgets.
These results support scope-aware comparison; broader workload and task-quality
claims retain the P1-D campaign gate. A useful comparison explains which costs
fall, which rise, and under what conditions the result remains feasible.

# Appendix A: retained historical case studies

These records retain the earlier manuscript's reported values and their original
limits. They are not the new matched publication evaluation.

## A.1 Dispatch under emulated links

The dispatch study (W1) compares four Qwen2.5-0.5B-Instruct candidates using eight-bit weights and activations (W8A8), 39 input tokens, and eight generated tokens. Local link emulation gives each party a shared 100 Mbps downlink and 40 Mbps uplink. An added 20 ms egress delay at each endpoint produces 40 ms additional request–reply delay. All four candidates produce matching output digests.

Sending the independent worker request first raises offset decode throughput from 0.1027 to 0.1994 tokens/s ($1.94\times$), while online application-body bytes remain unchanged (Table 2). Overlapping prepared delivery and issuance raises request throughput by 17.3%, with little change in decode throughput. Scheduling reduces observed latency rather than bytes. Each configuration has one sample; latency distributions are unmeasured.

| Candidate | Request (s) | Decode (tok/s) | Body MB |
| :--- | ---: | ---: | ---: |
| Prepared control | 122.87 | 0.1976 | 71.193 |
| Prepared overlap | 104.73 | 0.1984 | 71.193 |
| Offset control | 155.01 | 0.1027 | 112.005 |
| Offset seed-first | 105.33 | 0.1994 | 112.005 |

**Table 2.** Local link-emulation results (W1); one sample per candidate. Request timing excludes provider startup and checkpoint distribution. Body MB denotes decimal megabytes of online application bodies.

## A.2 Preprocessing and memory

A separate Qwen2.5 W8A8 study (C1) measures CPU from benchmark and role startup through generation of one token from 30 input tokens. Each placement has one cold-first response; source resolution before benchmark startup is excluded. Aggregate CPU is 11.71 s for clear client execution, 28.65 s for two workers, 34.66 s for prepared execution, and 142.83 s for verified prepared execution. Preparation alone accounts for 109.87 CPU seconds in the verified variant. Prepared execution therefore uses more aggregate CPU than the two-worker control in this workload, despite moving a product offline. Clear execution is a computational reference with no remote outsourcing.

The paging study (M1) isolates Qwen3-4B Preparation on an arm64 macOS host with 32 GiB RAM. It uses 144 stages, 71 rows per stage, a four-stage window, and two fresh processes per mode. Paging lowers median peak RSS from 4,613.77 to 739.09 MB ($6.24\times$). Correction contents match after excluding timing. Loading-plus-issuance CPU rises 12.41%, and each paged process adds 3.63 GB of private snapshots. The reduction applies to Preparation-process RSS in this probe, not client memory or full-response memory; filesystem cache is excluded.

## A.3 Numerical fidelity

Numerical evaluation requires two comparisons: protected execution against the same unprotected integer computation, and integer computation against an independent model reference. Agreement among PLLM variants is an internal consistency check; a shared numerical error can remain in all variants.

Archived checks Q1 and Q2 compare local clear-kernel prefill with an independent FP32 implementation on two prompts per model. For each of Qwen2.5-0.5B and Qwen3-0.6B, top-token agreement is 1/2 with W8A8 and 0/2 with four-bit weights and activations (W4A4). These historical checks are separate from the performance studies and do not characterize generation quality.

# Appendix B: historical artifact index

The records below are stored in repository snapshot `54605c1bcc699d801bc3b62d9b02c17d69a56f0e` [@pllm2026]. This identifies the reviewed archive, not the execution revision of every historical experiment. Missing execution metadata is tracked by placeholder A1/P1-E.

| ID | Archived record | Scope of the reported evidence |
| :--- | :--- | :--- |
| W1 | [WAN dispatch comparison (4 October 2026)](https://github.com/blairhudson/pllm/blob/54605c1bcc699d801bc3b62d9b02c17d69a56f0e/docs/evidence/wan-tps-qwen25-2026-10-04.json) | Four single-sample candidates; local link emulation; matching output digests. |
| C1 | [Cold-first CPU comparison (26 September 2026)](https://github.com/blairhudson/pllm/blob/54605c1bcc699d801bc3b62d9b02c17d69a56f0e/docs/evidence/slalom-prepared-topologies-cold-cpu-2026-09-26.json) | One response per placement; benchmark and role startup included; earlier source resolution excluded. |
| M1 | [Qwen3-4B Preparation paging (5 October 2026)](https://github.com/blairhudson/pllm/blob/54605c1bcc699d801bc3b62d9b02c17d69a56f0e/docs/evidence/preparation-memory-qwen3-4b-2026-10-05.json) | Two fresh processes per mode; isolated probe, not full-response execution. |
| Q1 | [Qwen2.5-0.5B reference check (24 September 2026)](https://github.com/blairhudson/pllm/blob/54605c1bcc699d801bc3b62d9b02c17d69a56f0e/docs/evidence/qwen2.5-0.5b-reference-quality-2026-09-24.json) | Two prompts; local clear-kernel prefill against an FP32 reference. |
| Q2 | [Qwen3-0.6B reference check (24 September 2026)](https://github.com/blairhudson/pllm/blob/54605c1bcc699d801bc3b62d9b02c17d69a56f0e/docs/evidence/qwen3-0.6b-reference-quality-2026-09-24.json) | Separate model/source identity; same limited prefill-check scope as Q1. |

# References
