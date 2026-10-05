---
title: "PLLM: A Research Platform for Private LLM Inference"
description: "A technical introduction to PLLM, its private-inference controls, and an evidence-bound workflow for autonomous optimization research."
author:
  - "Blair Hudson"
affiliation: "deployscience labs, Sydney NSW Australia"
email: "blairhudson@me.com"
date: "5 October 2026"
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
  Private inference aims to use remote model computation without disclosing a client's inputs and intermediate activations to individual providers. Useful implementations must also control communication, duplicated computation, memory, and numerical error. We introduce PLLM, an open-source runtime and research platform that makes these constraints explicit through model-neutral compilation, composable experiments, and workload-bound evidence. We explain its prepared masked-linear path and two-worker additive-sharing control, then describe how external research agents can propose, implement, and evaluate improvements using the same execution machinery. Local studies illustrate the method: scheduling improves two-worker decode throughput 1.94× under emulated consumer links; a Qwen3-4B Preparation probe reduces peak resident memory 6.24× with a CPU and disk tradeoff; and a cold compute comparison rejects an apparent advantage from moving work offline. These are scoped engineering results, not a new cryptographic proof or an evaluation of autonomous discovery.
---

# Introduction

Hosted language models normally receive the user's plaintext context. Local
inference avoids that disclosure but requires the user to supply the model's
memory and compute. Private inference seeks a third option: outsource useful
computation while protecting inputs and intermediate values from the operators
performing it. Encryption, secret sharing, and trusted execution offer different
trust and cost tradeoffs; none makes deployment cost disappear.

PLLM is a Python/Rust platform for investigating these tradeoffs on executable
decoder workloads. Its contribution is the research infrastructure: a common
model representation, explicit protocol and placement contracts, and matched
measurements across candidate implementations. The current system outsources
public-weight linear operations and keeps nonlinear computation and state at the
trusted client. More ambitious protected compositions remain research candidates.

The same interfaces support human-led and agent-driven work. An external agent
can change a component, construct an experiment, execute controls, and use the
reports to choose its next hypothesis. This paper introduces that workflow and
three motivating problems: WAN communication, the duplicated computation of a
simple two-worker design, and memory at real-checkpoint scale. It does not claim
a novel masking construction, fully automated cryptographic review, or an
established tenfold system improvement.

# System and trust model

The trusted client holds plaintext input, activation scales, nonlinear operators,
decoder state, and output selection. An optional application-facing gateway runs
inside that boundary. Public token lookup and, ordinarily, the output head also
run locally. Providers hold public transformer-body weights. Moving application
requests to a different ordinary provider URL does not create this boundary.

Python owns application orchestration, source resolution, and role lifecycles.
Rust owns validated numeric kernels, semantic planning, bounded codecs, and
one-use material. Model adapters lower into a shared decoder representation;
compiler passes operate on semantic roles rather than family-specific node names.
A configuration-only plan and a checkpoint-bound execution are distinct objects.
Runtime binding checks the source, weights, numeric choices, stage schedule, and
placement before reserving material.

The baseline assumes honest-but-curious services that follow the protocol and do
not collude. In the prepared path, Preparation must erase masks; self-hosting it
keeps this trust client-local. Authenticated transport prevents unrelated callers
from impersonating roles but does not prove operator independence. Model weights,
tensor dimensions, timing, lengths, and access patterns are not hidden by the
baseline. Local child-process experiments test execution, not non-collusion.

# Private-linear execution

Let $W$ be a public integer matrix and $x$ a private activation vector. Operations
below use an admitted exact ring; bounded integer results are reconstructed before
client-side scaling and nonlinear work. One-use masks must never be recycled.

## Two online offset workers

The client samples a uniform share $a$ and constructs $b=x-a$. Distinct workers
receive one share each and return $Wa$ and $Wb$. The client reconstructs

$$Wa+Wb=Wx.$$

An individual share hides $x$ under the stated non-collusion assumption. Both
workers perform the matrix product, so body-linear arithmetic is doubled relative
to a single clear execution. This does not imply exactly twice the full request's
CPU time: client work, communication, loading, and scheduling also contribute.
PLLM implements this control through the same compiled decoder used by other
placements.

## Offline Preparation, online Inference

For each stage row the client supplies fresh seed material to Preparation.
Domain-separated expansion produces input mask $r$, output mask $s$, and a
one-use ticket. Preparation computes

$$c=Wr-s$$

and sends the correction to Inference before the inventory becomes ready.
Online, the client sends the ticket and $u=x-r$. Inference consumes the row and
returns

$$Wu+c=W(x-r)+Wr-s=Wx-s.$$

The client adds $s$. Preparation is idle during the response. Reservations burn
on use, cancellation, replay, or failure; idle refill creates fresh material.
Prefill batches rows and decode reuses an authenticated connection. Numerical
range bounds determine exact packed ring widths.

This moves one matrix product offline but still performs **two matrix products
in total**. It therefore changes the online critical path without automatically
beating the two-worker compute control. An optional client-side Freivalds check
adds authenticated one-use projections supplied by Preparation. It is inspired
by verified delegation such as Slalom [@tramer2019slalom], but is neither a TEE
implementation nor a malicious-Preparation guarantee.

# An evidence-bound research loop

PLLM exposes immutable model, pipeline, workload, and deployment specifications.
Components select protocols, quantization, kernels, state reuse, verification,
and role placement. Compilation rejects unsupported combinations; it cannot
silently replace a missing protected operator with a weaker execution path.
The live SDK, gateway, and benchmark consume the admitted composition.

An external research agent can use this interface as follows:

1. **State a falsifiable hypothesis.** Name the control, objective, workload,
   trust model, quality tolerance, and resource bounds. For example: overlap
   independent worker requests to reduce WAN latency without changing bytes.
2. **Implement the smallest discriminating probe.** Pin literature and upstream
   revisions as specifications or independent oracles. Implement the candidate
   within PLLM rather than importing a paper's runtime as the result.
3. **Check the contract.** Compare against independent numeric references; test
   malformed, cancelled, and replayed attempts; price retained weights, transient
   buffers, state, and material before launching larger work.
4. **Run matched controls.** Lock checkpoint, quantized body, prompt cohort,
   output counts, cache state, hardware, and network settings. Record generated
   output identity or a declared quality comparison.
5. **Retain the evidence.** Keep successful and rejected hypotheses. Promote a
   method into ordinary component selection only after its execution and resource
   contracts are implemented.

Reports bind configurations and measured bodies to exact cohorts. Private prompts,
token IDs, masks, and credentials are excluded from the archive; salted cohort
digests support comparisons within a benchmark invocation. Different invocations
cannot be retrospectively treated as a matched prompt cohort. Unknown costs stay
unknown rather than becoming zero.

The shipped planner searches a bounded set of supported placements using explicit
cost evidence. An agent's broader loop can propose new implementations and run
experiments, but automatic protocol invention and the productivity of autonomous
research have not been evaluated. Compiler legality also does not certify a
candidate's cryptographic security or model quality.

# Optimization objectives

## WAN: bytes and round trips

Large activations and repeated stage round trips can dominate local-kernel speed.
For a party $p$, a simple bandwidth floor is

$$T_p\geq\max(B_p^\uparrow/R_p^\uparrow,\;B_p^\downarrow/R_p^\downarrow),$$

where bytes and rates use consistent units and the party's peers share its access
link. Dependent exchanges add latency. Moving bytes from the client to a peer link
does not remove them from all-link cost; compressing public artifacts does not
necessarily reduce fresh online activation traffic.

PLLM separates online, measured-run, setup-inclusive, and cold-first application
bodies. These are not complete physical-wire counts. End-to-end throughput uses
all $N$ generated outputs; decode throughput uses the $N-1$ outputs after the
first and its corresponding execution window. Link emulation checks actual queue
settings rather than labeling an unconstrained local run as WAN performance.

## Compute: the two-worker control

The relevant budget is total work over a declared lifecycle:

$$C_{\mathrm{full}}=C_{\mathrm{client}}+\sum_j C_{\mathrm{role}\,j}.$$

A cold-first comparison includes loading, initial material, and the first
response. Warm or amortized comparisons must identify reuse and its horizon.
Preparation CPU cannot be omitted because it precedes the first token. The
research objective is $C_{\mathrm{candidate}}<C_{\mathrm{offset}}$ on matched
hardware, quality, and outsourcing constraints, not merely fewer online products.
Moving the full model to the client changes the service being compared.

## Memory and fidelity are separate gates

Weight owners, copies, caches, masks, GPU snapshots, and decoder state have
different lifetimes. Paging can reduce process resident memory while increasing
disk and CPU. Fresh-process high-water marks are needed for peak comparisons;
filesystem cache is outside process RSS. Likewise, matching a tiny checkpoint or
one greedy output is weaker than representative generation quality. A 10× claim
must name its baseline, denominator, workload, and cost boundary. Ratios from
different cohorts cannot be multiplied.

# Measured case studies

The following studies are separate cohorts, not a joint system score. Their JSON
reports, configurations, source identities, and reproduction commands accompany
the repository [@pllm2026]. MB and GB below are decimal.

## WAN scheduling

One four-candidate Qwen2.5-0.5B-Instruct cohort used the same W8A8 body, 39 input
tokens, eight outputs, and matching output digests. Each party had shared 100 Mbps
download and 40 Mbps upload, with 20 ms added egress delay per party. The local
emulation therefore added 40 ms request/reply delay; it was not an Internet test.

| Candidate | Request s | Decode tokens/s |
|:--|--:|--:|
| Prepared control | 122.87 | 0.1976 |
| Batched delivery, windowed issuance | 104.73 | 0.1984 |
| Seeded offset control | 155.01 | 0.1027 |
| Seed-first offset | 105.33 | 0.1994 |

Sending the independent worker request first improved offset decode throughput
**1.94×** without reducing its 112.005 MB online bodies. Prepared overlap
improved request throughput **17.3%**, with 71.193 MB online bodies unchanged.
These are single samples; request timing excludes provider startup and checkpoint
distribution. A public-artifact rANS candidate was rejected: its 127.748 MB
transfer exceeded zlib's 124.900 MB and increased client decode CPU. The retained
artifact is `wan-tps-qwen25-2026-10-04.json`.

## Whole-lifecycle compute

A separate Qwen2.5 W8A8 30-input/one-output cold-first cohort measured aggregate
CPU from benchmark and role startup through the response:

| Placement | Aggregate CPU s |
|:--|--:|
| Client-only clear | 11.71 |
| Prepared | 34.66 |
| Two online offset workers | 28.65 |
| Freivalds-verified prepared | 142.83 |

Prepared execution did not beat the two-worker control. Verification used 109.87
CPU seconds in Preparation alone and exceeded the comparator 4.98×. This is one
co-located cohort, not a universal ranking; it illustrates why online-only counts
cannot establish the compute objective. The artifact is
`slalom-prepared-topologies-cold-cpu-2026-09-26.json`.

## Qwen3-4B memory

One cached-source Qwen3-4B W8A8 prepared response completed 16 input tokens and
eight capped outputs with paged client artifacts. Client/dashboard peak RSS was
**421.31 MB**; Inference and Preparation peaks were 4.50 and 4.03 GB. No new host
swap was observed. These lifetime peaks need not occur simultaneously. The run
establishes functionality and resource observations, not reference quality or a
10× client-memory reduction: no matched resident-client control was run. Its
artifact is `qwen3-4b-client-paged-2026-10-05.json`.

A separate Preparation-only probe used 144 stages, 71 rows per stage, a four-stage
window, and two fresh processes per mode. Raw authenticated paging reduced median
peak RSS from **4,613.77 MB to 739.09 MB (6.24×)**. Correction-frame contents matched
after excluding the timing field. Issuance CPU changed by +0.14%, while loading
plus issuance rose from 117.387 to 131.953 CPU seconds (+12.41%). Each paged process
owned 3.63 GB of private snapshots. This probe is not a selectable live topology;
filesystem cache and full-response cost remain outside it. The artifact is
`preparation-memory-qwen3-4b-2026-10-05.json`.

Together, the 4B observations direct research toward weight ownership and loading
peaks rather than assuming arithmetic kernels alone determine feasibility. They
also show why isolated reductions must be rechecked in complete responses.

# Related work and limitations

Secure-inference systems such as Cheetah [@huang2022cheetah] optimize two-party
neural-network evaluation; THE-X [@chen2022thex] investigates encrypted
Transformer execution. Slalom [@tramer2019slalom] combines trusted execution with
private, verified delegation. These motivate different parts of PLLM's research
space; their threat models and benchmarks are not interchangeable with its current
client-local nonlinear path. PLLM does not claim to outperform these systems.

Current evidence is primarily local, co-located, and workload-specific. Independent
operators, complete wire accounting, broad model quality, malicious participants,
and specialist cryptographic review remain separate validation tasks. One-use
ledgers and compiler checks enforce engineering contracts, not a composable
security theorem. Future protected nonlinear or resident-share execution must
pass its own complete numeric, privacy, lifecycle, and cost gates.

# Conclusion

PLLM makes private-inference research executable as a loop: specify, implement,
check, compare, and retain evidence. Its controls expose the distinction between
online speed and total work, and its 4B measurements reveal memory costs hidden
by small-model tests. The opportunity for autonomous research is a reusable,
auditable experiment surface. The next goal is measured system-level improvement
under explicit trust, quality, network, and compute constraints.

# References
