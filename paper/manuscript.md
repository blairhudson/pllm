---
title: "PLLM: a high-performance private multi-party inference runtime and extensible autonomous research harness"
description: "PLLM compares private language-model inference with tests, client-memory limits, and costs for the full request. This paper is a draft."
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
  Private language-model inference lets clients use remote computers while their inputs stay private if the specified security assumptions apply. Its costs include preprocessing, communication, model delivery, and client memory. We measure the largest total resident memory of all client processes at the same time. PLLM is an open-source framework with different components for model operations, numerical formats, protocols, and computation locations. It records costs for each role through the request lifecycle. Its comparisons use nested memory limits and identify which adaptations complete full requests and which supply only individual components.

  Tests with a new build and a new environment give the same bytes for 88,776 generated logit and cache values. Logits are scores for possible next tokens. The key-value (KV) cache keeps attention keys and values from processed tokens. A different test does inference through the software development kit (SDK) and gateway with the listing's specified checkpoint.

  We also examine 51 archived client runs from four cohorts. A cohort is a recorded group of runs. Different configurations have the lowest traffic, the lowest memory use, and the highest speed after the first output token. Thus, a configuration with lower traffic can use more memory or give lower speed. These results give comparisons with specified limits, but no evidence for one best method for all workloads. [P1-D: Complete the wider set of matched publication experiments before claims about more workloads or model quality.]
---

*This manuscript is a draft. Placeholders identify evidence that is necessary for publication.*

# 1 Introduction

Language-model applications process private documents, conversation histories, and
application state. Remote inference gives clients more computation capacity, but
usual services receive the information that they process. Encryption in transit
keeps data private on the connection, but the service receives the information.
Local inference keeps the information at the client, which must do the computation
and keep the model. Can a client with small resource budgets use
remote computation while its private inputs and intermediate state stay private
from each service operator?

Private multi-party inference divides computation between a client and other
parties. The client sends protected input representations to these parties, which
do model computation with specified security assumptions. One such representation uses
shares: values that together give a secret value. With new uniform additive
sharing, each individual share is independent of the secret value. A party can
calculate public linear operations on its share.

Nonlinear operations make other protocols or trusted-client computation
necessary. Secure-computation systems
such as CrypTen and Cheetah supply methods for this work
[@knott2021crypten; @huang2022cheetah].

More servers alone do not give privacy. Under the privacy assumptions, specified
information must stay separate. Collusion occurs when parties use that information together.

The collusion model identifies the groups for
which the privacy guarantee continues to apply. The protocol also identifies
information that parties receive and the instructions that they must obey. A
semi-honest party obeys the protocol but examines all information that it receives.
A malicious party can do operations outside the protocol.

Input and output privacy does not, by itself, give model privacy. Homomorphic
encryption, garbled circuits, and function secret sharing are other methods for
private computation. Trusted execution uses hardware assumptions. These methods
have different security and deployment requirements [@chen2022thex;
@zahur2015halfgates; @gupta2024sigma; @tramer2019slalom].

Autoregressive language models give output tokens in sequence. Prefill processes
the input prompt, and decode calculates each subsequent output from earlier tokens.
Attention keeps keys and values from processed tokens in the KV cache. Logits give
the scores from which the client selects the next token. Nonlinear functions,
normalization, and numerical conversions add work beyond a public matrix product.

The client can do much work when providers hold the transformer-body
weights. This work includes vocabulary lookup, output prediction, tokenization,
and management of persistent state and temporary buffers. Thus, client memory
use can be high when providers hold the model weights.

A method can decrease online latency if it moves work into preprocessing. It can
decrease client traffic but increase traffic between providers. It can decrease
resident weight storage but increase temporary memory and file I/O. Approximation
and quantization can decrease computation costs but change the numerical model.
Thus, a fast protected operator can be part of a slow full request.

Quantization uses a smaller set of numbers to approximate selected model values.

Current frameworks use different models, numerical formats, adversary
assumptions, and environments. Recent reviews examine reproducibility,
standardized benchmarks, and comparison contracts [@chen2026sok;
@das2026comparability; @albadawi2026sok; @kei2026sok]. Because of these differences,
comparisons with a constant client budget are not easy. Software implementations, specified
assumptions, and measurements through the request lifecycle are necessary for a
comparison.

PLLM supplies one framework for these experiments. Model adapters supply the
same decoder representation. Components give the numerical rules, protocols,
state, and computation locations, and the compiler examines their compatibility.
Permitted combinations use the same SDK, gateway, and benchmark path
[@pllm2026]. Measurements of independently developed mechanisms give the costs
of these choices.

This paper gives four results:

1. Execution contracts that connect decoders and methods to their source for use
   in different experiments
2. An evaluation method with client-memory tiers
3. Independently developed versions of selected published mechanisms, each with a
   specified scope
4. An audited reanalysis of costs and resources in four archived cohorts.

The new tests and the historical reanalysis keep different evidence identities.
**[P1-D: Add the wider set of matched publication experiments.]**

The three research questions are:

- **RQ1:** Can different designs use one execution framework while their
  assumptions stay clear?
- **RQ2:** How do costs for the full request change the interpretation of
  improvements to individual components?
- **RQ3:** Which designs can operate with the specified client-memory limits?

The baseline paths for full requests keep nonlinear computation and decoder
state at the client. Other research operators have their own execution scope. An
operator in the framework is not, by itself, evidence of a full private decoder
or a new security proof.

# 2 Scope and comparison contract

## 2.1 Functionality, numerics, and security

The primary comparison is for decoder inference with public weights. Each role
contract gives the privacy requirements for client inputs, intermediate values,
and outputs. Model weights stay public. A numerical contract gives the number
formats, arithmetic rules, and error limits that a comparison must use.

Version 1 of the comparison contract sets the conditions before measurements
start. These conditions include the checkpoint identity, numerical representation,
workload, sampling, permitted leakage, adversary assumptions, lifecycle, and
resources. Verified and unverified results keep different contracts. The artifact
contains the machine-readable constants and full rules in
`paper/comparison-contract.v1.json` and `paper/COMPARISON-CONTRACT.md`.

The shared remote profile includes static semi-honest corruption of at most one
provider role. With static corruption, the adversary selects the compromised role
before computation starts. The assumptions include a trusted client and setup, with
confidential authenticated channels. A more permissive collusion model makes a new
composition argument necessary. The number of roles alone does not give this
argument.

The profile gives providers public geometry, processed lengths, message sizes,
access patterns at each stage, timing, and termination. Secret token indices do
not become public access patterns through a change of label. Client-only
execution is a computation reference. Verified-linear methods and methods with
trusted hardware keep their other assumptions. A comparison with an
unavailable competitor stays unresolved.

The exact-execution track compares the protected decoder with the same unprotected
quantized decoder. It compares all logits, active KV values, positions, processed
history, pending selections, and sampler behavior. These comparisons are
different from numerical fidelity to an independent BF16/FP32 reference and from
task quality.
Before tests of changed numerical methods, a quality-matched track must have a
specified task suite and permitted quality loss. Agreement on some selected
tokens does not give the identity of an exact-execution result.

## 2.2 Client-memory tiers

| Tier | Whole-client ceiling |
| :--- | ---: |
| Low | 256 MiB |
| Medium | 2 GiB |
| High | 16 GiB |

These nested ceilings are PLLM evaluation conventions, not industry standards or
total device-RAM requirements. Each ceiling is eight times the previous ceiling,
which gives a coarse logarithmic budget grid. The Low tier has a memory limit much less than
a GiB. These limits do not come from a device survey and give no evidence of
suitability for an application.

Each report must give measured peaks. It must give sensitivity at 0.75, 0.875, 1,
1.125 and 1.25 times each ceiling. The Low grid is thus 192/224/256/288/320 MiB. Section 6.3
classifies historical observations again with these limits. Measurements near the
Medium and High limits are unavailable.

Observed eligibility uses the largest peak for the full client across accepted
replicates, each with a new process. The usual response worker includes the SDK,
supervisor, dashboard, relay, and audit. The client-memory measure also includes
each other client helper. For a client with multiple processes, the
measurement must include their memory at the same time. The sum of peaks from
different times does not give this measurement.

Other memory measures stay outside this client-memory measure. They include
provider peaks, private files, filesystem cache, GPU allocations, and device
memory. Observed eligibility from resident set size (RSS) is different from
conservative admission decisions before runtime work. A measured peak below a
ceiling does not set an upper bound for subsequent runs.

## 2.3 Costs and lifecycle

Measurements use specified cost windows. A cost window is an interval with
specified start and end events and a specified set of roles. The primary measures
are client RSS/CPU, Preparation CPU/RSS, online time to first token, N−1 decode
throughput, and all-link traffic. Each report must also include full-request
latency, time from startup through cleanup, aggregate CPU, and artifact/storage
costs. Missing measurements stay unknown. If Preparation has no role in a run,
this absence is not a measurement of zero preparation cost.

The comparison uses different cost windows for different initial conditions. A
source-present cold client starts with model source files available. Other
conditions include warm public artifacts with new private material, repeated
or continued workloads, and source-cold deployment. Public indexes, certificates,
checkpoints, and preprocessing have costs before the response.
These costs belong in the applicable window.

All-link application bodies, TCP-stream bytes, and physical-wire traffic are
different measures. A report divides base traffic from output-dependent traffic
only if it can identify the correct part for all bytes.

An eligible configuration obeys the comparison's security, memory, and outsourcing
rules. Researchers set these rules before the comparison. An eligible configuration is on
the frontier if no eligible competitor is better in one metric and no worse in the others.

Predominantly remote execution puts at least 90% of logical
body-linear multiply-accumulate operations (MACs) outside the client in each
phase. This limit applies to prefill and decode independently. Thus, the client
does at most one-tenth of this work, not one-tenth of total CPU work.

The denominator counts each logical product one time. Protocol duplication and
preprocessing add costs beyond this count. The denominator does not include
lookup, the output head, attention products, or nonlinear work. Reports must give
80/85/90/95/100% sensitivity. A per-row counter alone does not give sufficient
evidence of remote-share eligibility in each phase. The comparison keeps individual metrics instead of one
weighted score for all methods.

More workers can add trust requirements and aggregate bandwidth. The per-party
profile gives each role 100/40 Mbps down/up. The normalized profile keeps the
client access allowance. It divides a constant 200/80 Mbps provider pool, four
CPU-core equivalents, and 8 GiB equally between providers. These budgets are
synthetic comparison conditions.

Current native execution applies the link partitions without controls for the
aggregate CPU/RSS pool. Such a run has the label link-normalized only. Evidence
with full resource normalization is necessary.

# 3 Design and implementation

## 3.1 Experiment interface

An `Experiment` contains an immutable `Pipeline`, a `Deployment`, and an
`ExecutionBudget`. The pipeline contains the model source and execution
components. The deployment gives role locations, and the budget sets limits on
requests and token counts. A researcher can change a kernel or transport while
the workload stays the same. Listing 1 shows the Python interface [@pllm2026].

Resolution compares a configuration with its rules. Before execution, the runtime binds the
configuration to the model source, weights, numerical choices, stage schedule,
and role placement. The compiler rejects a combination if its components cannot
operate together. An unavailable protected operator causes rejection. The Python interface,
command-line benchmark, and application gateway use the same permitted
configuration (Figure 1).

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

**Figure 1.** From experiment configuration to benchmark report. Prepared execution
uses an offline Preparation service and an online Inference service. The client
keeps nonlinear computation and decoder state.

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

**Listing 1. Experiment configuration.** The configuration uses a pinned model, a
CPU kernel, a local deployment, and a request budget. Resolution does
configuration tests without inference. The example puts roles on one host for
development.

## 3.2 Model representation and components

Model adapters put decoder operators, layer identities, and persistent state into
the same intermediate representation (IR). Compiler passes use these fields
instead of the different node names in each model. Available adapters include
Qwen2 and dense Qwen3. An adapter supplies the model representation. Tests of
checkpoint execution and numerical fidelity give different evidence.

Python controls experiments, source resolution, and role lifecycles. Rust supplies
integer kernels, scheduling, bounded codecs, and one-use material through a PyO3
interface. Thus, researchers make experiments in Python, and native code
does tests of numerical and resource limits. Components select quantization,
linear protocols, kernels, caching, verification, and placement. The compiler
identifies compatible combinations.

Client-side weight paging uses authenticated private snapshots for local token
lookup and output-head execution. It keeps the same decoder semantics. A
researcher can also select Preparation paging independently. The historical
Preparation measurements in Appendix A occurred before this installed option.
They apply only to the isolated probe.

## 3.3 An extension and its tests

**[E1: generated implementation checks recorded]**

`MlpPrefillTiling` extends the framework through its current interfaces. It finds
row-independent regions with gate/up, SiLU, multiply, and down-projection
operations. The compiler rejects a region if outside operations use its intermediate
values. The scheduler does computation on public row tiles of bounded size.

Full outputs, attention, and KV keep the same semantics. The native
contract binds the source plan, schedule, composition, and tile size. Before
execution, the runtime rejects Metal and verified compositions that are not
compatible with this extension.

A replay from an installed wheel with a specified identity does 29 tests of the
extension, source binding, and full nonlinear operators. Generated Qwen2, Qwen3,
and Llama fixtures give the same bytes for 29,592 logit and active-KV values with
and without tiling. These tests include batched prefill and three decode steps
with the same tokens.

Comparisons with independent Transformers FP32 execution give maximum logit
errors of 0.042739 for Qwen2, 0.051512 for Qwen3, and 0.047203 for Llama. All three
errors are below the specified fixture tolerance of 0.1. This comparison measures
quantization fidelity. It gives no evidence of exact FP32 results or task quality
on representative workloads.

A different replay uses a build from an empty target directory and a new
environment. It compares 180 array pairs with 88,776 logit and active-KV values
across full and tiled runs of the same fixture families. This replay uses a different native
binary and keeps its own identity. The earlier execution records keep their
initial labels.

The installed-wheel replay also completes the usual response benchmark and the
published prepared SDK example without changes. It also does inference through
the example's HTTP gateway and different local provider child processes.

A different replay uses a cached checkpoint and the new build for inference with
the pinned Qwen configuration in Listing 1 without changes. The SDK and gateway each process
40 input and eight output tokens. They give the same text and record zero swap
growth. The SDK audit records masked stages and zero plaintext prompt/token-ID
bytes sent.

These are function tests with an in-process ASGI gateway and provider child
processes that use TCP. They do not give performance evidence or evidence of
independent operators. The new archive binds the two provider processes to the
installed SDK/native build. It also keeps full checkpoint-file locks before and
after execution.

`paper/IMPLEMENTATION-EVIDENCE.md` records source/build identities, independent
oracles, commands, and failed attempts. An oracle supplies the results that a
test must give. `scripts/replay_paper_listing.py` reads
the published listing directly instead of a second configuration.
**[P1-E: Supply an immutable publication release and an independent external
replay.]**

# 4 Baseline runtime and trust model

The trusted client keeps plaintext inputs, activation scales, nonlinear
operations, decoder state, and output selection. The client does token lookup and
usually the output-head computation locally. An application gateway operates inside this trust
boundary. Providers hold public transformer-body weights.

The distributed protocols use honest-but-curious services, which obey the
protocol and the specified collusion restriction. Honest-but-curious has the same
meaning as semi-honest. The client can also do the Preparation work. Processes on one
host use the implementation but give no evidence of separation between operators.

The algebraic privacy model does not include time. It gives providers the public
weights, geometry, admitted lengths, and semantic stopping information.
Semantic stopping obeys the decoder's token rules and request limits.
The model assumes that those choices are independent of cryptographic randomness.
Implementation timing and side channels make further analysis necessary.
Transport authentication does tests of role identities, but it gives no evidence
of non-collusion.

A one-use owner is a program object that controls access to consumable protocol
material. Logical retirement of this owner is different from secure erasure. The
runtime can prevent use of an owner again. But a party with copies of the seeds
can calculate the masks again.
Thus, the present runtime gives no guarantee for adaptive compromise or forensic
erasure.

## 4.1 Two-worker execution

The public integer matrix is $W\in R^{m\times n}$, and the encoded activation is
$x\in R^n$. The numerical configuration supplies a finite ring $R$. The client
makes new additive shares $a$ and $b=x-a$. Each of two workers receives one
share and sends its product with $W$. The client adds the products:

$$
Wa+Wb=Wx.
$$

With uniform masks, either share is independent of $x$ in this algebraic model.
Seed-based encodings also have expansion assumptions. The workers
calculate two matrix products in total. This is two times the body-linear
arithmetic of a clear product. This arithmetic count does not give total CPU time
for a request.

## 4.2 Prepared execution

Prepared execution does mask preparation before online computation. New client
seed material gives an input mask $r$, an output mask $s$, and a one-use ticket
for each stage row. Preparation calculates $c=Wr-s$ and sends this correction
to Inference before online work starts. The client then sends the ticket and
$u=x-r$. Inference sends:

$$
Wu+c=W(x-r)+Wr-s=Wx-s.
$$

The client adds $s$ to reconstruct the bounded integer result. It does scaling and
nonlinear computation locally. Preparation stays idle during online execution. Reserved
material becomes unusable after use, cancellation, replay, or failure, so
new material is necessary for subsequent requests. Prefill uses batches of rows,
and decode uses the same authenticated connection again.

This method removes one product from the online critical path. The total across
preparation and online computation stays at two products. Optional Freivalds
tests at the client use new projections from Preparation, with Slalom's verified
delegation as the reference [@tramer2019slalom]. PLLM's placement does not use
trusted execution hardware. These tests do not give protection against malicious
Preparation.

# 5 Related work and reimplementation scope

CrypTen supplies secure tensor computation and a machine-learning interface
[@knott2021crypten]. Cheetah gives two-party neural-network protocols
[@huang2022cheetah]. THE-X examines homomorphic Transformer evaluation
[@chen2022thex]. Slalom uses trusted hardware with private verified delegation
[@tramer2019slalom]. These papers give different references for the design.
A citation alone is not evidence of a PLLM reproduction.

The selected set includes six families:

- Verified delegation from Slalom
- Public equalization from SmoothQuant [@xiao2023smoothquant]
- Adaptations of full gated operators from Compact, Curl, and garbled lookup
  [@islam2024compact; @santos2024curl; @heath2024logrow]
- Nonlinear mechanisms from SIGMA/FuseFSS [@gupta2024sigma; @ma2026fusefss]
- Embedding and packing from FastQuery/BumbleBee [@lin2024fastquery;
  @lu2025bumblebee]
- Encrypted softmax from NEXUS/THOR [@zhang2025nexus; @moon2024thor].

These families have different numerical, conversion, setup, and transport
contracts. The implementation inventory records these differences. Measurements
of a bounded operator give no performance evidence for a published full system.

| Evidence class | Result scope |
| :--- | :--- |
| End-to-end adaptation | The SDK completes inference for a specified checkpoint and workload with the adaptation's contract. |
| Component reference | A bounded mechanism gives results that agree with its oracle and costs for its specified scope. |
| Analytical or blocked reference | The record gives a construction, a cost limit, or a rejection. It gives no completed serving measurement. |

Each reimplementation card records the source version, algorithm, and assumptions
that stay the same. It records changes to rounding, backend, and dealer. It also
records omissions, the oracle, geometry, cost windows, and raw replay evidence.
The earlier software supplies a specification or oracle input, but the PLLM
runtime does not use it. Comparisons of external full systems must have their own
qualification. The absence of an implementation is not evidence of a weaker baseline.

The full gated operators have important numerical and cost differences. Each operator
includes a SiLU approximation, multiplication by an up-projection value, and
numerical conversions. Curl's public Haar table uses exact replacement
rescaling. The LogRow-labeled reference uses the paper's lookup and row-reduction
construction, but keeps a larger one-hot setup. Compact adapts density-guided
fitting to dyadic polynomial intervals and Boolean evaluation. All three include
the costs of private selection, conversion, multiplication, and the last rounding
step.

A new bounded replay compares 32 outputs with an independent integer oracle.
It also compares 32 outputs from new material. These changed profiles do not
keep the checkpoint's full logits and KV values the same. Their operator costs
give no evidence of the relative performance of the earlier full systems.

The new six-family replay completes 46 tests with results that agree with their
test contracts. They include traffic counts from captured frames and tests of
rejection controls. Ten bounded operator/replay pairs compare 116 exact integer
outputs and eight approximate CKKS outputs in total. All outputs agree with their
specified contracts. All 26 child processes record the same installed-package identities at
their start and end.

The earlier two-byte equalization assertion stays in its own failed record.
The earlier test did not record its frames. The corrected test uses the
actual timing-field widths and rejects unrelated changes to the framing. In these
tests, the mechanisms obey their contracts. These tests give no performance
results for the external papers or evidence of a full encrypted decoder.

The review of primary sources has the date 16 October. The review, source hashes,
and full implementation cards are in `paper/PRIOR-ART-REVIEW.md`,
`paper/prior-art-sources.json`, and `paper/IMPLEMENTATION-EVIDENCE.md`. The review
identifies publication versions and records whether observations use abstracts
alone or full texts. LogRow's locator is Section 5, Construction 1 and
Figures 3--5. The PLLM setup has different costs from those in LogRow's Section 6
accounting.

We completed the specified P1-C work on mechanisms and contract tests. New
measurements across matched configurations are P1-D work. Other mechanisms stay
as references in the appendices. The measured set in the primary text does not
automatically include them.

# 6 Evaluation design

## 6.1 Method

The benchmark runner compares candidates with the same workload. It can use grid
search or seeded-random search through configurations. Comparisons keep the same
checkpoint, encoded model body, prompt cohort, generated-token counts, cache
state, hardware, and network conditions. Reports record configuration and source
identifiers, but not private prompts, token IDs, masks, or credentials. Salted
prompt-cohort identifiers identify the same cohort within one benchmark
invocation. They do not identify the same cohort across independent invocations.

Measurements use different windows for online execution, request work with setup,
and cold-first process lifecycles. Aggregate CPU includes the client and all
measured roles, with Preparation, through the specified interval. Communication
counts include application bodies, not full physical-wire traffic. Decode
throughput divides the $N-1$ tokens after the first by their execution time.

The measurements record peak RSS for each process. RSS measures resident process
memory and does not include filesystem cache. The peaks of different roles can
occur at different times.

Appendix A keeps archived repository case studies [@pllm2026] with their archived
workloads and measurement scope. These studies help select the evaluation
questions. They do not supply missing replicates or controls for a new
publication cohort.

**[E2: evidence to add]**

**[P1-D: matched evaluation.]** Add balanced runs and replicates with specified
hardware/software, thread limits, preprocessing/cache policy, and workload
lengths. Record raw paired changes, variability, all-role costs, and blocked
cells. Set exclusions, memory-pressure procedures, and finalist confirmation
before the runs.

Different prompts give evidence about more workloads. More runs of one prompt do
not give this evidence. E2 keeps the previous draft's requirement for repeated
performance measurements.

## 6.2 Research questions and planned figures

- **RQ1: framework fidelity.** The evidence includes generated decoder controls,
  independent references, extension tests, and retirement after failures.
  Figure 1 and Listing 1 connect the public API to the execution contract.
- **RQ2: complete costs.** Compare full operators with their primitives.
  Include conversion and preprocessing costs.
  Measure cold delivery and online work in different windows.
  **[P1-D Figure 2: complete-cost breakdowns for admitted mechanisms.]**
- **RQ3: resource feasibility.** Compare permitted client-only, two-offset,
  prepared, and verified prepared paths with different trust classes.
  Use maxima for the full client to select Low/Medium/High eligibility.
  Show the changes in each resource measure.
  **[P1-D Figure 3/current-baseline table: tier frontier and matched whole-response costs.]**

**[E3: evidence to add]**

For P1-D, add current protected/clear comparisons and comparisons with independent
references. Keep tokenization and sampling constant. Record tests of full logits
and state. Record numerical errors, prefill/decode agreement, and held-out task
quality as different results.

The historical two-prompt tests in Appendix A do not measure generation quality on
representative workloads.

## 6.3 Audited historical reanalysis

An archived matched study of 24 clients examines costs in different measurement
windows. The study has eight recipes, with three new clients for each recipe.
All recipes use one pinned Qwen2.5-0.5B W8A8 source and 150 input plus eight
output tokens. W8A8 uses eight-bit weights and activations. All recipes use local
shared 100/40 Mbps access pacing and source-present cold clients.

Digit8 is a Preparation kernel that uses eight-bit digits with bounded float32
products and sums. Integer reconstruction gives the exact ring products.
Selective placement moves specified
body-linear operations to the client. Coupled boundaries move private token lookup
and the full output head to an additional worker pair.

A read-only publication audit examines the archived report, locked source/plan
bindings, digests of complete decoder state, and individual measurements. It does not
do the computation again. Table 1 gives the archived observations.

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
medians, while memory is the largest peak for the full client across all three
replicates.
The last row adds two boundary workers, so it has a different trust contract and
aggregate-resource contract. These values compare different topologies, not
speed with the same resources. MB is decimal, and MiB is binary.

The study does not measure indexed-tokenizer delivery, checkpoint delivery,
physical-wire traffic, or Internet delay.

The cohort records tests of 210,247,680 logit/KV values and zero observed swap
growth. The recipe with tiling plus digit8 has the smallest client peak. Selective
placement decreases traffic with a specified 90.45% per-row remote body-linear
share. Coupled boundaries decrease cold delivery but add work for each token.
These results give different choices for different resource objectives.

The coupled-boundary recipe uses 61.87 seconds of aggregate cold-first CPU.
The prepared recipes use 53.81--58.55 seconds. These cost windows end at the first
response and do not include cleanup. Thus, a shorter request can use more total
work.

Threshold sensitivity also changes eligibility. At 128 MiB, each recipe's peak is
above the limit. At 192 MiB, only tiling, tiling plus digit8, and tiling plus digit8
plus placement use no more than the limit. All eight use no more than the observed 256 MiB
ceiling. These eligibility results use the same measured maxima. They are not new
admission results or new memory measurements.

Thus, specified budgets and sensitivity results are necessary for a comparison.
One recipe is not best for all resource objectives. These results do not make
256 MiB an external standard.

The publication audit also keeps three different boundary cohorts at 150+8,
150+32, and 150+64. Each cohort keeps its own build identity and cohort
commitment. The audit keeps their samples apart. The reproduction command is
`node scripts/audit_publication_evidence.mjs --check`.

`paper/artifacts/historical-results-v1.json` contains the full source hashes,
replicate vectors, all seven primary metrics, and paired changes. New
measurements are necessary for the wider P1-D experiments. Model-quality
evaluation with independent references is also necessary.

# 7 Availability, limitations, and open questions

PLLM is alpha software with the distribution name `pllm.run` and the Apache-2.0
license. Its Python import and command are `pllm` [@pllm2026]. The source and build
identities for publication can change. The artifact index identifies an earlier
reviewed archive. It does not identify the execution
revision of each historical measurement.

**[A1: evidence to add]**

**[P1-E: publication artifact.]** Keep source, build, configurations, raw reports,
failed cells, and reproduction commands in an immutable publication archive.
Compare each claim with its record. Replay the release independently. A1 keeps the
previous requirement for provenance evidence.

Compiler tests and one-use material management apply implementation rules.
They do not give a composable privacy proof. Local child processes give no evidence of
non-collusion or independent operation. Local access-link emulation does not
measure Internet latency. Application bodies and TCP streams do not give full
physical-wire traffic. Resource tiers give no evidence of model quality or protection
against arbitrary malicious parties.

The Low tier gives a question for further research. Can an exact-execution design
decrease full communication and request time without memory above the client
limit or costs outside the comparison? One possible choice compares client vocabulary
delivery with repeated work at remote boundaries. A new method must give better
results than strong current baselines with this contract. The framework does
not select the best method before measurements. **[P1-D: Add measured resource
ownership and traffic breakdowns as evidence for this research question.]**

# 8 Conclusion

PLLM connects model semantics, numerical rules, protocols, resource ownership,
and measurements through execution contracts. The generated extension tests give
the same decoder state. Comparisons with independent references measure
numerical fidelity, which is a different question.

The audited historical cohorts have different best configurations for
communication, memory, and decode speed. For these comparisons, specified
resource budgets and scope are necessary. The P1-D experiments
are necessary for claims about more workloads and task quality. A useful
comparison records which costs decrease, which costs increase, and whether resource
use is at or below its limits.

# Appendix A: retained historical case studies

These records keep the values from the earlier manuscript and their initial
limits. They are not the new matched publication evaluation.

## A.1 Dispatch under emulated links

The dispatch study (W1) compares four Qwen2.5-0.5B-Instruct candidates with
eight-bit weights and activations (W8A8). The workload has 39 input tokens and
eight generated tokens. Local link emulation gives each party a shared 100 Mbps
downlink and 40 Mbps uplink. An added 20 ms egress delay at each endpoint
gives 40 ms more request–reply delay. All four candidates give the same output
digests.

The seed-first schedule sends the independent worker request first. Offset
decode throughput then increases from 0.1027 to 0.1994 tokens/s
($1.94\times$), with the same online application-body bytes (Table 2).
Prepared delivery and issuance at the same time increase request throughput by
17.3%, with a small change in decode throughput. Thus, scheduling decreases
observed latency, but does not decrease bytes. Each configuration has one
sample. The study does not measure latency distributions.

| Candidate | Request (s) | Decode (tok/s) | Body MB |
| :--- | ---: | ---: | ---: |
| Prepared control | 122.87 | 0.1976 | 71.193 |
| Prepared overlap | 104.73 | 0.1984 | 71.193 |
| Offset control | 155.01 | 0.1027 | 112.005 |
| Offset seed-first | 105.33 | 0.1994 | 112.005 |

**Table 2.** Local link-emulation results (W1). The study has one sample for each
candidate. Request timing does not include provider startup or checkpoint
distribution. Body MB gives decimal megabytes of online application bodies.

## A.2 Preprocessing and memory

A different Qwen2.5 W8A8 study (C1) measures CPU from benchmark and role startup
through generation of one token from 30 input tokens. Each placement has one
cold-first response. The cost window does not include source resolution before
benchmark startup.

Aggregate CPU is 11.71 s for clear client execution and 28.65 s for two workers.
It is 34.66 s for prepared execution and 142.83 s for verified prepared
execution. Preparation alone uses 109.87 CPU seconds in the verified variant.
Prepared execution uses more aggregate CPU than the two-worker control in this
workload, although it moves a product offline. Clear execution is a computation
reference with no remote outsourcing.

The paging study (M1) isolates Qwen3-4B Preparation on an arm64 macOS host with
32 GiB RAM. It uses 144 stages, 71 rows for each stage, a four-stage window, and two
new processes for each mode. Paging decreases median peak RSS from 4,613.77 to
739.09 MB ($6.24\times$). The corrections have the same contents when the
comparison does not include timing. CPU for loading and issuance increases
12.41%. Each paged process adds 3.63 GB of private snapshots.

This decrease applies to Preparation-process RSS in the isolated probe. It does
not apply to client memory or full-response memory. The RSS measure does not
include filesystem cache.

## A.3 Numerical fidelity

Numerical evaluation has two necessary comparisons. One compares protected
execution with the same unprotected integer computation. The other compares the
integer computation with an independent model reference. Agreement between PLLM variants
is evidence of internal consistency. The same numerical error can occur in all
variants.

Archived tests Q1 and Q2 compare local clear-kernel prefill with an independent
FP32 implementation on two prompts for each model. For each of Qwen2.5-0.5B and
Qwen3-0.6B, top-token agreement is 1/2 with W8A8. It is 0/2 with four-bit weights
and activations (W4A4). These historical tests are different from the performance
studies and do not measure generation quality.

# Appendix B: historical artifact index

Repository snapshot `54605c1bcc699d801bc3b62d9b02c17d69a56f0e` contains these
records [@pllm2026]. This snapshot identifies the reviewed archive, not the
execution revision of each historical experiment. Placeholder A1/P1-E identifies
the missing execution metadata.

| ID | Archived record | Evidence scope |
| :--- | :--- | :--- |
| W1 | [WAN dispatch comparison (4 October 2026)](https://github.com/blairhudson/pllm/blob/54605c1bcc699d801bc3b62d9b02c17d69a56f0e/docs/evidence/wan-tps-qwen25-2026-10-04.json) | Four candidates each have one sample with local link emulation. All output digests are the same. |
| C1 | [Cold-first CPU comparison (26 September 2026)](https://github.com/blairhudson/pllm/blob/54605c1bcc699d801bc3b62d9b02c17d69a56f0e/docs/evidence/slalom-prepared-topologies-cold-cpu-2026-09-26.json) | Each placement has one response. The window includes benchmark and role startup, but not earlier source resolution. |
| M1 | [Qwen3-4B Preparation paging (5 October 2026)](https://github.com/blairhudson/pllm/blob/54605c1bcc699d801bc3b62d9b02c17d69a56f0e/docs/evidence/preparation-memory-qwen3-4b-2026-10-05.json) | Each mode uses two new processes. This is an isolated probe, not a full response. |
| Q1 | [Qwen2.5-0.5B reference check (24 September 2026)](https://github.com/blairhudson/pllm/blob/54605c1bcc699d801bc3b62d9b02c17d69a56f0e/docs/evidence/qwen2.5-0.5b-reference-quality-2026-09-24.json) | Two prompts compare local clear-kernel prefill with an FP32 reference. |
| Q2 | [Qwen3-0.6B reference check (24 September 2026)](https://github.com/blairhudson/pllm/blob/54605c1bcc699d801bc3b62d9b02c17d69a56f0e/docs/evidence/qwen3-0.6b-reference-quality-2026-09-24.json) | The model/source identity is different. Its prefill tests have the same two-prompt scope as Q1. |

# References
