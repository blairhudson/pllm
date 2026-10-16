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
  Private language-model inference lets clients use remote computation without revealing their inputs under a stated security model. Practical cost depends on more than protected computation: preprocessing, communication, model delivery, and client memory can determine whether a design is usable. We present PLLM, an open-source framework for implementing and evaluating these tradeoffs. PLLM separates model semantics from numerical representations, protocols, and execution placement, and records costs across participating roles and request lifecycles. We define nested client-memory tiers and distinguish executable adaptations of prior work from bounded component references. The evaluation studies correctness, complete costs, and resource feasibility. [P1-D: complete the matched evaluation and insert its methods, workloads, and principal findings.] Executable configurations, correctness checks, and replayable evidence support comparisons without treating different numerical or security contracts as interchangeable.
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
Their results do not directly identify which compositions remain effective under
a fixed client budget and a complete request lifecycle. We present PLLM to make
these choices executable and their costs inspectable. Model adapters produce a
shared decoder representation; checked components specify numerical, protocol,
state, and placement choices. Supported compositions run through a common SDK,
gateway, and benchmark path [@pllm2026].

The paper contributes (1) source-bound execution contracts for reusable decoder
and method implementations, (2) a resource-tiered evaluation methodology,
(3) scoped independent implementations of selected published mechanisms, and
(4) an empirical account of complete costs and feasible configurations.
**[P1-D: finalize the fourth contribution from the matched results.]**

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
inputs, intermediate values, and outputs are private under each stated role
contract, while model weights are public. A comparison fixes checkpoint identity,
numerical representation, workload, sampling, permitted leakage, and adversary
assumptions. Verified and unverified results retain different contracts.

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
total device-RAM requirements. They make an explicit client budget part of the
comparison. Low represents a small inference footprint, Medium an application
worker, and High a compute-rich worker; application suitability still needs
measurement. **[P1-A: finalize deployment rationale and sensitivity around each
threshold.]**

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
optimizing resource objectives. An initial predominantly remote comparison proposes
at least 90% of body linear MACs off-client in both prefill and decode; this
threshold and its sensitivity require justification in P1-A. Client-only execution
remains a useful reference. More workers can add both trust requirements and
aggregate bandwidth. Cross-topology latency results therefore report all providers
and include a resource-normalized sensitivity comparison. No universal weighted
score replaces the individual metrics.

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

**[E1: evidence to add]**

**[P1-B: extension and conformance evidence.]** Add one implemented extension,
its unchanged interfaces, conformance checks, an ordinary benchmark invocation,
and a rejected incompatible configuration. Rerun the published SDK example through
the gateway and archive exact commands and outcomes. Resolution alone does not
demonstrate a completed inference request. E1 is the earlier draft's alias for
this gate.

# 4 Baseline runtime and trust model

The trusted client retains plaintext inputs, activation scales, nonlinear operations, decoder state, and output selection. Token lookup and ordinarily the output head also run locally. An application gateway runs inside this boundary; providers hold public transformer-body weights.

The distributed protocols assume honest-but-curious services that follow the protocol and do not collude. Prepared execution additionally requires Preparation to erase mask material; Preparation may instead be hosted by the client. Co-located processes exercise the implementation but do not provide operator separation. Model weights, tensor dimensions, lengths, timing, and access patterns are outside the protection offered by these protocols. Transport authentication checks role identities; it does not establish non-collusion.

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

The initial mechanism selection includes complete gated-operator adaptations of
Compact, Curl, and logarithmic-ciphertext lookup [@islam2024compact;
@santos2024curl; @heath2024logrow], selected SIGMA/FuseFSS nonlinear mechanisms
[@gupta2024sigma; @ma2026fusefss], and NEXUS encrypted softmax primitives
[@zhang2025nexus]. Their numeric, conversion, setup, and transport contracts differ.
The implementation inventory records these differences rather than assigning
whole-system paper performance to a bounded operator.

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

**[P1-C: primary-source and coverage freeze.]** Select the representative main-text
mechanisms, verify citations and deviations against primary sources, and link
every table row to a frozen configuration and report. Broader inventory belongs
in an appendix. No ranking across incompatible numerical/security contracts follows.

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
  **[P1-D Figure 3/Table 2: tier frontier and matched whole-response costs.]**

**[E3: evidence to add]**

For P1-D, add current protected/clear and independent-reference comparisons. Pin
tokenization and sampling; report full logits/state checks and separate numerical
errors, prefill/decode agreement, and held-out task quality. The historical
two-prompt checks in Appendix A do not characterize representative generation.

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
and measurement through executable contracts. The framework supports independent
mechanism implementations and scope-aware comparison across client-memory tiers.
**[P1-D: replace this placeholder with the principal matched findings.]** A useful
comparison explains which costs fall, which rise, and under what conditions the
result remains feasible.

# Appendix A: retained historical case studies

These records retain the earlier manuscript's reported values and their original
limits. They are not the new matched publication evaluation.

## A.1 Dispatch under emulated links

The dispatch study (W1) compares four Qwen2.5-0.5B-Instruct candidates using eight-bit weights and activations (W8A8), 39 input tokens, and eight generated tokens. Local link emulation gives each party a shared 100 Mbps downlink and 40 Mbps uplink. An added 20 ms egress delay at each endpoint produces 40 ms additional request–reply delay. All four candidates produce matching output digests.

Sending the independent worker request first raises offset decode throughput from 0.1027 to 0.1994 tokens/s ($1.94\times$), while online application-body bytes remain unchanged (Table 1). Overlapping prepared delivery and issuance raises request throughput by 17.3%, with little change in decode throughput. Scheduling reduces observed latency rather than bytes. Each configuration has one sample; latency distributions are unmeasured.

| Candidate | Request (s) | Decode (tok/s) | Body MB |
| :--- | ---: | ---: | ---: |
| Prepared control | 122.87 | 0.1976 | 71.193 |
| Prepared overlap | 104.73 | 0.1984 | 71.193 |
| Offset control | 155.01 | 0.1027 | 112.005 |
| Offset seed-first | 105.33 | 0.1994 | 112.005 |

**Table 1.** Local link-emulation results (W1); one sample per candidate. Request timing excludes provider startup and checkpoint distribution. Body MB denotes decimal megabytes of online application bodies.

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
