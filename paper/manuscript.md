---
title: "PLLM: a high-performance private multi-party inference runtime and extensible autonomous research harness"
description: "PLLM compares private language-model inference with tests, client-memory limits, and costs for the full request. This paper is a draft."
author:
  - "Blair Hudson"
affiliation: "deployscience labs"
email: "blair@deployscience.com"
date: "23 October 2026"
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
  Private language-model inference exchanges local work for protected computation, communication, and preprocessing under explicit security assumptions. PLLM connects Python experiment configurations to a Rust compiler and runtime with numerical, state, role, and resource contracts. We define comparisons with client-memory limits and separate costs by role and request phase. The planned study uses independent PLLM reimplementations of ten selected methods. One method currently completes a trained-checkpoint response, but none qualifies for publication comparison. Existing evidence covers generated-state conformance, bounded mechanisms, software development kit integration, and 51 archived clients in four separate cohorts. The historical observations identify different choices for traffic, memory, and decode speed. They do not rank the ten selected methods. Matched experiments, broader quality evaluation, and independent reproduction remain necessary.
---

*This manuscript is a draft. The matched ten-method campaign is planned, not completed. Placeholders identify outstanding publication evidence.*

# 1 Introduction

Private language-model inference must trade client work against protected
computation, communication, and preprocessing without changing its declared
security or numerical contract. A faster matrix operation need not give a faster
request: nonlinear work, vocabulary boundaries, decoder state, setup, and cleanup
remain owned somewhere. More providers can add both capacity and non-collusion
assumptions. Recent reviews identify comparability and reproducibility as open
systems problems [@chen2026sok; @das2026comparability; @albadawi2026sok;
@kei2026sok].

PLLM connects Python experiment definitions to a Rust compiler and runtime
[@pllm2026]. Model adapters describe decoder semantics; components declare
numerics, protocols, placement, and state; admission binds a supported composition
to its source and workload. The same experiment reaches the SDK, gateway, and
ordinary benchmark. This separates three questions: **RQ1**, whether an extension
preserves its declared computation; **RQ2**, how full-request costs change an
operator-level result; and **RQ3**, which configurations fit a client budget.

The contributions are execution contracts, a tiered evaluation design, independent
mechanism controls, and reanalysis of four historical cohorts. The source-locked
ten-method study is the planned complete-system evaluation, not a completed
result. **Trained-checkpoint response execution is 1/10; publication-qualified
methods are 0/10.** Generated decoders, components, and historical measurements
cannot fill the missing serving cells. Appendices retain the detailed contracts,
source versions, controls, and reproduction procedures.

# 2 Scope and comparison contract

## 2.1 Functionality, numerics, and security

The common workload is autoregressive decoding with public weights. Prefill
processes the prompt; decode advances token by token, retaining attention keys and
values in the KV cache. Some selected methods additionally protect model weights;
that property is not removed from their source contracts. Comparisons fix source,
arithmetic, workload, sampling, leakage, adversary, lifecycle, and resource class
before measurement. Verified and unverified execution remain distinct.

The baseline assumes a trusted client/setup, authenticated confidential channels,
and static semi-honest corruption of at most one provider. Public geometry,
processed lengths, message sizes, stage access patterns, timing, and termination
are disclosed. Other collusion sets require their own composition arguments;
process separation does not establish operator independence.

Exact execution means agreement with the same clear quantized decoder on logits,
active KV, positions, history, pending selections, and sampling. It is not
agreement with BF16/FP32 or representative task quality. Changed arithmetic
requires a predeclared quality-matched track. Appendix C gives the full contract.

## 2.2 Client-memory tiers

Low, Medium, and High cap whole-client RSS at **256 MiB, 2 GiB, and 16 GiB**.
These nested evaluation conventions are neither device-RAM requirements nor
industry standards. Eligibility uses the largest accepted fresh-process peak,
including the SDK, supervisor, relay, dashboard, audit, and other serving helpers.
Separate helpers require simultaneous sums, not sums of unrelated lifetime peaks.
Provider memory, files, filesystem cache, and device allocations remain separately
charged. An observed eligible peak is not conservative admission or a future bound.
Sensitivity uses 0.75/0.875/1/1.125/1.25 times each ceiling; historical evidence
does not cover the Medium/High boundaries.

## 2.3 Costs and lifecycle

Reports distinguish online, request-with-setup, and startup-through-cleanup
windows. They price client and provider CPU/RSS, first accepted output, $N-1$
decode throughput, request/target latency, and artifacts. Application bodies,
directed TCP streams, and physical-wire bytes are different measures. Missing
metrics remain unknown. Source-present cold clients have empty public bundle
caches while workers already hold checkpoint/tokenizer sources; warm-public and
source-cold deployment are separate cells.

Predominantly remote execution requires at least 90% of logical body-linear MACs
outside the client in both prefill and decode; it does not bound total client CPU.
Protocol duplication and preprocessing are additional work. Per-party links give
each role 100/40 Mbps; the normalized contract shares 200/80 Mbps, four CPU-core
equivalents, and 8 GiB among providers. Current native enforcement partitions links,
not the aggregate CPU/RSS pool, so such runs are only *link-normalized*.

# 3 Design and implementation

## 3.1 Experiment interface

An immutable `Experiment` combines a model/component `Pipeline`, placement intent
in `Deployment`, and an `ExecutionBudget`. Canonical serialization identifies the
configuration. Resolution selects a supported composition and role graph; it
neither loads a checkpoint nor proves memory admission. Compilation and
source-aware admission bind files, numerics, workload, stages, and roles.
Unsupported combinations reject rather than silently choosing another method.

    Specification
      Model + Pipeline + Deployment + Budget
          |
    Compilation and binding
      Decoder representation -> validated stages and role placement
          |
    Execution (prepared placement)
      Trusted Client <-> Inference
      Preparation supplies one-use corrections before online work
      Acceptance and retirement obey the selected method
          |
    Benchmark record
      Configuration + workload + environment + measurements

**Figure 1.** From experiment configuration to benchmark report. Prepared execution
uses offline Preparation and online Inference; nonlinear work and decoder state
remain at the trusted client. Listing 1 in Appendix D gives the executable SDK
configuration.

## 3.2 Model representation and components

Adapters lower Qwen2 and dense Qwen3 into a shared decoder representation.
Schedules identify prefill/decode operations, values, weights, executors, and state
handoff, binding model and composition digests. Python owns experiment and role
lifecycles; Rust supplies kernels, schedules, bounded codecs, and consumable
material through PyO3. Admission checks source geometry, tensor/frame storage, and
operation budgets before the corresponding allocation. Private seeds and pending
work remain in runtime owners. Logical retirement prevents reuse, not forensic
recovery. Allocation estimates are not measured RSS or OS-enforced limits.

## 3.3 An extension and its tests

`MlpPrefillTiling` tiles row-independent gate/up, SiLU, multiply, and down-projection
regions. Exclusive-consumer checks and source/schedule/tile binding preserve the
declared semantics; incompatible Metal and verified compositions reject.
A historical installed-wheel replay passes 29 tests with 29,592 byte-equal
logit/active-KV values. Independent FP32 comparisons measure fixture quantization
fidelity, not exact FP32 execution or trained task quality. Separate clean-build
and pinned-checkpoint SDK/gateway replays retain their own identities and do not
measure comparative performance. Appendix D preserves all values and procedures.

# 4 Baseline runtime and trust model

The trusted client keeps plaintext, scales, nonlinear work, KV state, selection,
and usually token lookup/head weights; a gateway lies inside that boundary.
Two-worker execution sends fresh shares $a,b=x-a$ of an encoded activation to
public-weight workers and reconstructs $Wa+Wb=Wx$ in the admitted ring. Uniform
sharing gives algebraic single-share privacy; seed encodings add expansion
assumptions. Two products are performed, not one product's total request CPU.

Prepared execution instead sends $c=Wr-s$ from Preparation to Inference before
online work. The client sends $u=x-r$ and recovers $Wx$ by adding $s$ to $Wu+c$.
Preparation is idle online, but preparation plus inference still costs two matrix
products. One-use tickets retire after use, replay, failure, or cancellation.
Optional client Freivalds checks are Slalom-derived, without Slalom's trusted
hardware; they do not protect against malicious Preparation [@tramer2019slalom].
The untimed model, seed custody, and protocol equations are detailed in Appendix E.
Neither one-use ownership nor local child processes establish secure erasure,
adaptive security, side-channel protection, or non-collusion.

# 5 Source selection and reimplementation scope

The planned study selects Gupta–Katz–Miers, Maverick, SIGMA, FuseFSS, BumbleBee,
PUMA, NEXUS, THOR, CacheMir, and ROSETTA. Appendix F gives governing versions,
source hashes, and comparison roles. Each method needs source-faithful lookup,
body, output function, token feedback, persistent state, and failure behavior.
Upstream code supplies specifications/oracles, not PLLM runtime dependencies.
Substantial protocol, numerical, security, or execution adaptations require author
approval; CPU ports and common-Qwen extensions remain distinct from original-model
validation.

GKM currently completes a trained-checkpoint response with signed 16-bit weights,
11 fractional bits, and arithmetic modulo $2^{64}$; request-end verification
precedes acceptance. This is not prepared W8A8. Generated PUMA/FuseFSS SDK controls
and six-family operator controls remain bounded integration/mechanism evidence.
They do not change the **1/10 trained, 0/10 publication-qualified** coverage or
support a ten-method ranking. Source-model quality and matched serving costs
remain unqualified.

# 6 Evaluation design

## 6.1 Planned matched campaign

The matched campaign is unperformed. Shared factories in `examples/papers/`
feed ordinary response and quality benchmarks using the pinned Qwen source in
Listing 1. Cells fix source, numerical/privacy track, workload, sampling, stopping,
lifecycle, hardware, provider resources, links, and instrumentation. Fresh-process
replicates estimate within-cell variability; prompt diversity requires different
prompts. Freeze quality margins, exclusions, and confirmation rules before runs.
Preserve blocked, timed-out, and rejected cells rather than replacing them with
zero costs or component measurements. **[P1-D/E2: campaign gate remains open.]**

Role/host RSS uses the maximum simultaneous observation vector; summing independent
lifetime peaks is invalid. Directed transfers count once and reconcile with
phase/request totals. Sampled RSS can miss peaks and double-count shared pages;
provider lifetime high-water, remote/device telemetry, and hard aggregate resource
controls remain gaps. Appendix G specifies windows, observers, and planned figures.

## 6.2 Acceptance-aware metrics and planned figures

First output means accepted output. GKM buffers until request-end verification;
PUMA's bounded private-prefix composition releases final IDs after protected
feedback. Buffer-release intervals are not model decode throughput, and final IDs
cannot supply undisclosed logit errors. Exact state, independent-reference error,
and held-out task quality are separate gates. **[P1-D/E3: correctness and quality
qualification remains open.]** Planned RQ2/RQ3 figures require qualified complete
methods; existing component controls cannot populate them.

## 6.3 Audited historical reanalysis

An archived matched study of 24 clients examines costs in different measurement
windows. The study has eight recipes, with three new clients for each recipe.
All recipes use one pinned Qwen2.5-0.5B W8A8 source and 150 input plus eight
output tokens. W8A8 uses eight-bit weights and activations. All recipes use local
shared 100/40 Mbps access pacing and source-present cold clients.

Digit8 is a Preparation kernel that uses eight-bit digits with bounded float32
products and sums. Integer reconstruction gives the exact ring products.
Selective placement moves specified body-linear operations to the client.
Coupled boundaries move private token lookup and the full output head to an
additional worker pair. A read-only audit checks archived state digests and
measurements; it does not execute the computation again.

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
medians; memory is the largest full-client peak across three replicates. The last
row adds two boundary workers and changes trust and aggregate resources: this is
not speed at equal resources. MB is decimal; MiB is binary.

The cohort checks 210,247,680 logit/KV values and records zero observed swap
growth, not zero swap use. Tiling plus digit8 gives the smallest client peak;
selective placement reduces traffic with a 90.45% per-row remote body-linear
share. This counter alone does not establish phase-wide remote eligibility.
Coupled boundaries reduce cold delivery but add per-token work. Their aggregate
cold-first CPU is 61.87 seconds versus 53.81--58.55 for prepared recipes, ending
at the first response and excluding cleanup. A shorter request can cost more work.

At 128 MiB no recipe qualifies; at 192 MiB only tiling, tiling plus digit8, and
tiling plus digit8 plus placement qualify; all eight satisfy the observed 256 MiB
ceiling. Reclassifying these same maxima is not new admission or measurement.
Indexed-tokenizer/checkpoint delivery, physical wire, and Internet delay are
unmeasured. Three later boundary cohorts at 150+8/32/64 share a recorded build
but have distinct private salts, workloads, and samples; none is pooled with this
24-client cohort. Appendices G/H retain report paths, metric definitions, and
reproduction instructions.

# 7 Reproducibility and limitations

The reproducible unit is a versioned experiment and ordinary benchmark output,
with correctness in normal tests. Standard uv/Maturin builds and installed-wheel
tests are detailed in Appendix H. Release freeze, independent reproduction,
claim/author review, and PDF inspection remain open **[P1-E/A1]**.

Compiler conformance is not a composable privacy proof. Local processes and access
pacing establish neither independent custody nor Internet performance. No result
establishes representative quality, malicious-complete security, GPU capacity,
price, energy, secure erasure, or independent cryptographic review. Historical Low
eligibility cannot measure the separate unmeasured Qwen3-4B 64+8 tenfold
whole-client memory target. The title's high-performance and autonomous-research
scope still needs matched runtime and productivity evidence. A new Low-tier method
must improve complete costs against strong eligible controls, not merely move
costs outside its window.

# 8 Conclusion

PLLM connects decoder semantics, numerical rules, protocols, ownership, and
measurement through execution contracts. Historical controls establish bounded
state conformance and fixture fidelity; archived cohorts show different choices
for communication, memory, and decode speed. Neither supplies the planned
ten-method comparison. The evaluation design makes those missing measurements,
quality gates, and trust/resource differences explicit instead of assigning an
unsupported method rank.

```{=latex}
\clearpage
\appendix
```

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

# Appendix C: comparison-contract details

## C.1 Background terminology

Language-model applications process private documents, conversation histories, and
application state. Remote inference gives clients more computation capacity, but
usual services receive the information that they process. Encryption in transit
keeps data private on the connection, but the service receives the information.
Local inference keeps the information at the client, which must do the computation
and keep the model. Can a client with small resource budgets use
remote computation while its private inputs and intermediate state stay private
from each service operator?

Private multi-party inference divides computation between a client and providers
under specified security assumptions. Additive shares reconstruct a secret value,
but each fresh uniform share is independent of that value. Providers can calculate
public linear operations on shares. Nonlinear operations need additional
protocols or trusted-client computation. CrypTen and Cheetah supply relevant
secure-computation methods
[@knott2021crypten; @huang2022cheetah].

More servers alone do not give privacy. Collusion combines information that the
protocol keeps separate. The collusion model identifies which groups preserve
the guarantee. A semi-honest party obeys the protocol but examines its received
information. A malicious party can depart from the protocol. Privacy and correct
computation remain different properties.

Input and output privacy does not, by itself, give model privacy. Homomorphic
encryption, garbled circuits, and function secret sharing are other methods for
private computation. Trusted execution uses hardware assumptions. These methods
have different security and deployment requirements [@chen2022thex;
@zahur2015halfgates; @gupta2024sigma; @tramer2019slalom].

Autoregressive language models give output tokens in sequence. Prefill processes
the input prompt, and decode calculates each subsequent output from earlier tokens.
Attention keeps keys and values from processed tokens in the key-value (KV) cache. Logits give
the scores from which the client selects the next token. Nonlinear functions,
normalization, and numerical conversions add work beyond a public matrix product.

Provider-owned body weights do not remove client costs for vocabulary lookup,
output prediction, tokenization, persistent state, and temporary buffers.

Preprocessing can decrease online latency without less total work. Less client
traffic can mean more provider traffic. Less resident storage can mean more
temporary memory and file I/O. Quantization approximates values with a smaller
numerical set and can change model behavior. Thus, a fast protected operator can
be part of a slow full request.

Current frameworks use different models, numerical formats, adversary
assumptions, and environments. Recent reviews examine reproducibility,
standardized benchmarks, and comparison contracts [@chen2026sok;
@das2026comparability; @albadawi2026sok; @kei2026sok]. Because of these differences,
comparisons with a constant client budget are not easy. Software implementations, specified
assumptions, and measurements through the request lifecycle are necessary for a
comparison.

The baseline paths for full requests keep nonlinear computation and decoder
state at the client. Other research operators have their own execution scope. An
operator in the framework is not, by itself, evidence of a full private decoder
or a new security proof.

## C.2 Functionality, numerics, and security

The common workload is decoder inference with public model weights. Some selected
source methods additionally protect the model. Their source contracts keep that
property distinct from the common workload. Each role contract specifies input,
intermediate-value, and output privacy. A numerical contract specifies formats,
arithmetic rules, and permitted errors.

Version 1 of the comparison contract sets the conditions before measurements
start. These conditions include the checkpoint identity, numerical representation,
workload, sampling, permitted leakage, adversary assumptions, lifecycle, and
resources. Verified and unverified results keep different contracts. The artifact
contains the machine-readable constants and full rules in
`paper/comparison-contract.v1.json` and `paper/COMPARISON-CONTRACT.md`.

The baseline remote profile includes static semi-honest corruption of at most one
provider role. With static corruption, the adversary selects the compromised role
before computation starts. The assumptions include a trusted client and setup, with
confidential authenticated channels. A more permissive collusion model makes a new
composition argument necessary. The number of roles alone does not give this
argument.

The profile gives providers public geometry, processed lengths, message sizes,
access patterns at each stage, timing, and termination. Secret token indices do
not become public access patterns through a change of label. Client-only
execution is a computation reference. Verified-linear methods, protected-model
methods, and methods with trusted hardware keep their other assumptions. A comparison with an
unavailable competitor stays unresolved.

The exact-execution track compares the protected decoder with the same unprotected
quantized decoder. It compares all logits, active KV values, positions, processed
history, pending selections, and sampler behavior. These comparisons are
different from numerical fidelity to an independent BF16/FP32 reference and from
task quality.
Before tests of changed numerical methods, a quality-matched track must have a
specified task suite and permitted quality loss. Agreement on some selected
tokens does not give the identity of an exact-execution result.

## C.3 Client-memory tiers

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

## C.4 Costs and lifecycle

Measurements use specified cost windows. A cost window is an interval with
specified start and end events and a specified set of roles. The primary measures
are client RSS/CPU, Preparation CPU/RSS, online time to first token, N−1 decode
throughput, and all-link traffic. Each report must also include full-request
latency, time from startup through cleanup, aggregate CPU, and artifact/storage
costs. Missing measurements stay unknown. If Preparation has no role in a run,
this absence is not a measurement of zero preparation cost.

The comparison uses different cost windows for different initial conditions. In
the retained source-present cold requests, a fresh client has an empty public
bundle cache while workers already hold checkpoint and tokenizer sources. Other
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

# Appendix D: implementation contracts and controls

## D.1 Experiment interface

An `Experiment` contains an immutable `Pipeline`, a `Deployment`, and an
`ExecutionBudget`. The pipeline contains the model source and typed components.
The deployment gives placement intent, and the budget bounds requests and token
counts. Canonical serialization gives configuration and pipeline digests. A
profile name does not replace the component specification. Listing 1 shows the
Python software development kit (SDK) [@pllm2026].

Resolution identifies a supported composition and its role graph. It does not
load a checkpoint or prove that a request fits available memory. Source-aware
admission and compilation then bind model files, numerical choices, workload,
stage schedule, and roles. Unsupported combinations reject instead of selecting
another method. The SDK, command-line benchmark, and gateway consume the same
experiment contract (Figure 1).

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

## D.2 Model representation and components

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

The runtime schedule separates prefill and decode. Each step identifies its
operations, input values, weights, outputs, and executor. The schedule binds the
model plan and component composition. State inputs and outputs describe the
handoff between phases. This representation lets protocols share model structure
without a common arithmetic representation or privacy proof.

Method-specific admission applies bounds before the corresponding resource
allocation. Relevant bounds include source geometry, graph size, tensor storage,
frames, and consumable material. Runtime owners keep private seeds and pending
operations outside public configurations. Failure and cancellation retire these
owners under the selected lifecycle contract. A resource estimate remains
different from measured RSS or an enforced operating-system limit.

Client-side weight paging uses authenticated private snapshots for local token
lookup and output-head execution. It keeps the same decoder semantics. A
researcher can also select Preparation paging independently. The historical
Preparation measurements in Appendix A occurred before this installed option.
They apply only to the isolated probe.

## D.3 An extension and its tests

`MlpPrefillTiling` extends the framework through its current interfaces. It finds
row-independent regions with gate/up, SiLU, multiply, and down-projection
operations. The compiler rejects a region if outside operations use its intermediate
values. The scheduler does computation on public row tiles of bounded size.

Full outputs, attention, and KV keep the same semantics. The native
contract binds the source plan, schedule, composition, and tile size. Before
execution, the runtime rejects Metal and verified compositions that are not
compatible with this extension.

A historical replay from an installed wheel with a specified identity does 29 tests of the
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

The installed-wheel replay also completes the ordinary response benchmark and
prepared SDK example through local provider child processes and an HTTP gateway.

A separate historical replay uses a cached checkpoint and its own build for inference with
the pinned Qwen configuration in Listing 1 without changes. The SDK and gateway each process
40 input and eight output tokens. They give the same text and record zero swap
growth. The SDK audit records masked stages and zero plaintext prompt/token-ID
bytes sent.

These function tests use an in-process ASGI gateway and TCP provider processes.
Their records bind provider imports and before/after checkpoint-file locks.
They do not establish performance or independent operators.

`paper/IMPLEMENTATION-EVIDENCE.md` identifies these historical tests and their
source/build records. An oracle supplies the expected results independently of
the protected execution. `scripts/replay_paper_listing.py` reads the published
listing directly. Section 7 specifies the remaining reproduction requirements.

# Appendix E: baseline protocols and custody

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

## E.1 Two-worker execution {#two-worker-execution}

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

## E.2 Prepared execution {#prepared-execution}

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

# Appendix F: governing sources and method contracts

CrypTen supplies secure tensor computation and a machine-learning interface
[@knott2021crypten]. Cheetah gives two-party neural-network protocols
[@huang2022cheetah]. THE-X examines homomorphic Transformer evaluation
[@chen2022thex]. Slalom uses trusted hardware with private verified delegation
[@tramer2019slalom]. These papers give different references for the design.
A citation alone is not evidence of a PLLM reproduction.

The planned study selects ten methods from the dated primary-source review.
The following table identifies the governing versions. The source lock records the exact
PDF hashes in `paper/methods/source-lock.v1.json`. Mutable publication links do
not replace those hashes.

| Selected method | Governing source | Comparison focus |
| :--- | :--- | :--- |
| Gupta–Katz–Miers | ePrint 2026/1849, hash-pinned revision [@gupta2026outsourcing] | Private lookup and verified public linear delegation |
| Maverick | [arXiv:2609.10264v1](https://arxiv.org/abs/2609.10264v1) | Coded private and verifiable delegation |
| SIGMA | PoPETs 2024(4), pp. 61–79 [@gupta2024sigma] | Function secret sharing and protected decoder operations |
| FuseFSS | arXiv:2606.09551v1 [@ma2026fusefss] | Fused function secret sharing |
| BumbleBee | NDSS 2025 final paper [@lu2025bumblebee] | Two-party transformer protocols and conversions |
| PUMA | [arXiv:2307.12533v3](https://arxiv.org/abs/2307.12533v3) | Three-party replicated arithmetic |
| NEXUS | NDSS 2025, DOI 10.14722/ndss.2025.230868 [@zhang2025nexus] | Non-interactive encrypted transformer inference |
| THOR | ePrint 2024/1881, hash-pinned revision [@moon2024thor] | Homomorphic transformer operations |
| CacheMir | [arXiv:2602.11470v1](https://arxiv.org/abs/2602.11470v1) | Encrypted autoregressive state |
| ROSETTA | [arXiv:2609.16915v1](https://arxiv.org/abs/2609.16915v1) | Encrypted kernels and scheme conversion |

**Selected sources.** This table defines the planned study, not measured results.
Each row needs an independent PLLM implementation faithful to its selected source.
Substantial protocol, numerical, state, security, or execution changes need author
approval. CPU ports and common-Qwen extensions keep their declared differences
from original-paper validation.

| Evidence class | Result scope |
| :--- | :--- |
| End-to-end adaptation | The SDK completes inference for a specified checkpoint and workload with the adaptation's contract. |
| Component reference | A bounded mechanism gives results that agree with its oracle and costs for its specified scope. |
| Analytical or blocked reference | The record gives a construction, a cost limit, or a rejection. It gives no completed serving measurement. |

Each method contract maps source equations and code to numerical rules, roles,
state, setup, output disclosure, and acceptance. Upstream software supplies
specifications or independent oracle inputs, not runtime dependencies. Complete
coverage includes lookup, body, head or declared output function, token feedback,
persistent state, and failure behavior. A component cannot replace an incomplete
method in the serving comparison.

The current evidence includes a trained-checkpoint GKM response through the
ordinary SDK and benchmark. Its declared source-derived arithmetic uses signed
16-bit weights, 11 fractional bits, and arithmetic modulo $2^{64}$.
This numerical contract differs from prepared W8A8. Request-end verification
precedes accepted output.

Generated PUMA and FuseFSS compositions supply additional integration controls,
not accepted trained-checkpoint results. Thus, one of ten methods reaches trained
response execution, and zero qualifies for publication comparison. GKM's broader
source-model quality and matched-cost qualification remain open.

Earlier controls cover six mechanism families. They include Slalom-derived
verification, SmoothQuant equalization, gated operators, function secret sharing,
embedding packing, and encrypted softmax [@xiao2023smoothquant;
@islam2024compact; @santos2024curl; @heath2024logrow; @lin2024fastquery].
Their historical replay passes 46 tests, with 116 exact integer outputs and eight
approximate encrypted outputs across ten bounded operator/replay pairs.
The 26 child-process identities agree before and after those tests.

These controls keep their declared deviations. Compact uses dyadic polynomial
intervals, Curl uses exact replacement rescaling, and the LogRow reference keeps
a larger one-hot setup. LogRow's source locator is Section 5, Construction 1,
and Figures 3--5, not its Section 6 cost accounting. The failed two-byte
equalization assertion remains separate from the later frame-aware repair.
`paper/IMPLEMENTATION-EVIDENCE.md` records the controls and original evidence.
They do not satisfy the ten-method acceptance gate.

# Appendix G: complete campaign and measurement protocol

## G.1 Planned matched campaign

The ten-method campaign remains unperformed. Reusable experiment factories in
`examples/papers/` use a shared model, workload bounds, and public prompt file.
The shared source is the Qwen revision in Listing 1. Ordinary
`pllm benchmark run` and `pllm benchmark quality` consume these factories.
The quality command and response benchmark answer different questions.

A comparison cell fixes source, numerical track, privacy class, workload,
sampling, output allowance, lifecycle, hardware, provider budget, network, and
instrumentation. Replicates use fresh processes within that cell. Changed
arithmetic needs a predeclared quality criterion, not an assumed W8A8 identity.
Original-model validation and the common-Qwen extension remain separate.

Reports record configuration, source, and installed-build identifiers without
private prompts, token IDs, masks, or credentials. Salted prompt-cohort identifiers
match workloads within one benchmark invocation, not across independent
invocations. Source-present cold, warm-public with fresh private material, and
source-cold deployment keep different cost windows.

Measurements use different windows for online execution, request work with setup,
and cold-first process lifecycles. Aggregate CPU includes the client and all
measured roles, with Preparation, through the specified interval. Communication
counts distinguish covered application bodies from directed TCP streams. Decode
throughput divides the $N-1$ tokens after the first by their execution time.

Each directed transfer counts once in all-link traffic. Phase totals must
reconcile with the declared request window, with unclassified bytes visible.
The phases are public setup, private preparation, prefill, decode, and cleanup.
Request setup and inter-step gaps remain explicit where the observer separates
them. Physical-wire traffic remains outside these counters.

The campaign will report instance counts, per-instance peaks, phase peaks, and
simultaneous aggregates by role, host, and complete provider deployment. For
process set $G$, sampled aggregate RSS is $\max_t\sum_{i\in G}\mathrm{RSS}_i(t)$.
The sum uses one observation vector before selection of the maximum across
replicates. Independent lifetime peaks cannot supply this statistic.

The existing local lifecycle observer supplies bounded phase and ownership
controls, not ten-method performance evidence. Its sampled sums can miss short
peaks and count shared pages twice. Provider lifetime high-water, device memory,
and remote-host measurements remain unavailable. Filesystem cache and private
files keep separate scopes.

Appendix A keeps archived repository case studies [@pllm2026] with their archived
workloads and measurement scope. These studies help select the evaluation
questions. They do not supply missing replicates or controls for a new
publication cohort.

**[P1-D/E2: campaign gate.]** Freeze quality margins, workloads, thread limits,
provider capacity, delay, cache policy, exclusions, and confirmation rules before
measurements. Use the declared shared 100/40 Mbps access limits for each party.
Report raw paired changes and uncertainty within matched cells. Preserve
unsupported, admission-blocked, timed-out, and rejected attempts separately.

Different prompts give evidence about more workloads. More runs of one prompt do
not give this evidence. E2 keeps the previous draft's requirement for repeated
performance measurements.

## G.2 Acceptance-aware metrics and planned figures

Accepted output determines first-output latency. GKM buffers tentative tokens
until request-end verification. PUMA's fixed-count private-prefix composition
discloses final IDs after protected feedback. Buffer-release intervals cannot
measure model decode throughput. These profiles therefore keep streaming decode
throughput unavailable and report supported semantic timings separately.

Output contracts also limit quality measurements. A final-ID-only method can
supply token agreement but not undisclosed logit errors. Exact state parity,
error against an independent numerical reference, and held-out task quality
remain separate results. Generated fixtures establish bounded correctness, not
trained-model quality.

- **RQ1: framework fidelity.** The evidence includes generated decoder controls,
  independent references, extension tests, and retirement after failures.
  Figure 1 and Listing 1 connect the public API to the execution contract.
- **RQ2: complete costs.** Compare admitted complete methods, with operator
  controls for conversion and preprocessing costs.
  Measure cold delivery and online work in different windows.
  **[P1-D Figure 2: phase costs and traffic for qualified methods.]**
- **RQ3: resource feasibility.** Compare qualified methods within their declared
  classes. Keep client-only, two-offset, prepared, and verified prepared controls separate.
  Use maxima for the full client to select Low/Medium/High eligibility.
  Show the changes in each resource measure.
  **[P1-D Figure 3: qualified ten-method coverage, tier frontier, and role memory.]**

**[P1-D/E3: correctness and quality gate.]** Compare protected execution with its
declared clear computation and an independent source-model reference. Keep
tokenization and the supported sampling policy constant. Unsupported output or
state observations remain unknown, not equal by assumption.

The historical two-prompt tests in Appendix A do not measure generation quality on
representative workloads.

## G.3 Historical-source reconstruction

The publication audit checks the archived report, locked source/plan bindings,
complete decoder-state digests, and individual measurements behind Table 1. It
does not execute the tensors anew. It also keeps three boundary cohorts at 150+8,
150+32, and 150+64. They share one recorded build but use distinct private salts
and workloads. Each keeps its own cohort commitment and samples. The
recorded-table check is
`node scripts/audit_publication_evidence.mjs --check`.

`paper/artifacts/historical-results-v1.json` contains the full source hashes,
replicate vectors, all seven primary metrics, and paired changes. New
measurements are necessary for the wider P1-D experiments. Model-quality
evaluation with independent references is also necessary.

# Appendix H: reproduction and remaining evidence

PLLM is alpha software with the distribution name `pllm.run` and the Apache-2.0
license. Its Python import and command are `pllm` [@pllm2026]. The source and build
identities for publication can change. The artifact index identifies an earlier
reviewed archive. It does not identify the execution
revision of each historical measurement.

The reproducible unit is a versioned experiment definition and its ordinary
benchmark output, with correctness in normal tests. `examples/papers/common.py`
pins the common checkpoint and workload bounds. Method factories select
components, and the ordinary SDK owns execution, admission, and measurements.
`examples/papers/README.md` gives response and quality commands. A reproduction
must preserve source, numerical, acceptance, workload, build, and metric identities.

Standard uv/Maturin commands build the wheel. Tests of an installed wheel disable
checkout import precedence with `pytest -o pythonpath=`. `CONTRIBUTING.md` gives
the build procedure. Historical reports keep their original builds and failed
attempts. Git history supplies old source versions, without copied source trees
or new receipt collections.

**[P1-E/A1: publication release.]** Freeze the release, experiment definitions,
ordinary reports, and source-acquisition instructions. Independent reproduction,
claim review, author approval, and PDF inspection remain open. Build success is
not evidence of scientific completion.

Compiler tests and one-use material management apply implementation rules.
They do not give a composable privacy proof. Local child processes give no evidence of
non-collusion or independent operation. Local access-link emulation does not
measure Internet latency. Application bodies and TCP streams do not give full
physical-wire traffic. Resource tiers give no evidence of model quality or protection
against arbitrary malicious parties.

The selected CPU-first implementation work does not reproduce source GPU
performance. No current result supports GPU capacity, price, energy, Internet
deployment, or cryptographic-review claims. The title's high-performance scope
still needs a matched runtime comparison. Local conformance and historical
tradeoffs do not supply that comparison or evaluate autonomous research productivity.

The Low tier gives a question for further research. Can an exact-execution design
decrease full communication and request time without memory above the client
limit or costs outside the comparison? One possible choice compares client vocabulary
delivery with repeated work at remote boundaries. A new method must give better
results than strong current baselines with this contract. The framework does
not select the best method before measurements. **[P1-D: Add measured resource
ownership and traffic breakdowns as evidence for this research question.]**

# References
