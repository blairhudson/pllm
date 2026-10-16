---
title: "Communication-Efficient Private LLM Inference under a 256 MiB Client Budget"
description: "A working method-paper draft on coupled private lookup and full-head execution, with explicit novelty and comparison gates."
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
pdf: "low-memory.pdf"
source: "low-memory-source.zip"
arxiv: "low-memory-arxiv-source.zip"
abstract: |
  Private inference can outsource transformer computation while leaving substantial model-boundary costs at the client. Token lookup and output prediction may require a large vocabulary matrix, increasing cold delivery and local ownership. We study a coupled design that moves these operations to two non-colluding boundary workers while retaining client-side state and token selection. The method reconstructs complete logits and preserves the execution contract of a pinned quantized decoder. Using the PLLM benchmark, we evaluate [P2-D: models and workloads] against [P2-A: qualified baselines]. Under [P2-B/P2-C: security and resource conditions], the method reduces all-link communication by [P2-E: confirmed effect] and complete-request latency by [P2-E: confirmed effect], with a largest measured client footprint of [P2-E: value] MiB. The evaluation characterizes accompanying decode, provider-resource, and cold/warm tradeoffs. These result slots remain unfilled pending the paper-specific comparison and confirmation.
---

*Working manuscript; novelty, security analysis, and the publication comparison
remain open. Historical results below are not a new SOTA claim.*

# 1 Introduction

A client can outsource most transformer computation and still retain a substantial
inference footprint. Vocabulary lookup, output prediction, tokenization, decoder
state, and temporary buffers remain necessary somewhere in the system. For clients
with limited memory and access bandwidth, these remaining responsibilities can
determine feasibility and cold-request cost.

The PLLM framework establishes a common basis for measuring these costs under
explicit numerical, security, and resource contracts [@hudson2026pllmframework].
Its Low tier uses a 256 MiB complete-client ceiling. Meeting this ceiling does not
identify a single best design: eligible configurations trade communication,
latency, client computation, and provider resources. The framework manuscript is
currently a working draft; its publication identifier and benchmark version remain
to be frozen.

This paper studies the model boundary. Token lookup maps an input token to an
embedding; the output head maps the decoder's hidden vector to a score, or logit,
for every vocabulary token. In models with tied embeddings, these operations use
the same matrix. Moving lookup alone therefore leaves the head matrix at the
client. Moving both avoids that ownership, but a straightforward remote head adds
communication and dependent work during every generation step.

We study coupled private boundary execution: two workers share responsibility for
private lookup and exact full-head evaluation over their existing immutable weight
owners. The client receives the required embeddings and reconstructs every logit,
retaining its original sampler and decoder state. The prepared transformer body
continues to execute through its original protocol. The design targets lower
complete-request communication within the Low-tier budget.

The proposed contribution is the complete composition and its implementation,
rather than a claim to have invented private lookup, additive sharing, or parallel
transport. **[P2-A: determine the original contribution against the closest
published systems before finalizing this statement.]** We evaluate where the
exchange is beneficial, where it becomes costly, and whether it improves on the
strongest eligible alternatives. Smaller vocabulary ownership alone is not a
claim of smaller process peak memory.

# 2 Problem and comparison class

## 2.1 Objective

The primary objective is complete-request all-link communication, subject to a
Low-tier client budget, the stated privacy contract, and preservation of a fixed
quantized decoder. Complete-request latency is secondary. Online first-text time,
N−1 decode throughput, client CPU/RSS, and all provider resources remain reported
tradeoffs. Source-present cold clients and warm public artifacts have different
comparison cells; source-cold distribution needs its own measurement.

Exactness means the same complete logits, active KV and positions, executed
history, pending token, and sampling behavior as the reference quantized execution.
It does not mean equivalence to the checkpoint's BF16/FP32 computation or establish
task quality. Independent-reference fidelity and held-out task quality have
separate evidence. A single checkpoint supports a checkpoint-specific claim.

## 2.2 Parties and trust

The trusted client communicates with offline Preparation, online Inference, and
two additional boundary workers. The intended contract is semi-honest execution;
the boundary workers do not collude, and Preparation retains its existing erasure
and non-collusion requirements. Public weights, geometry, request lengths, and
declared timing/access leakage are outside input-value confidentiality.

**[P2-B: specify every permitted and forbidden coalition across all four remote
roles, setup/publisher assumptions, and leakage. Give a composition argument for
the actual lookup and linear protocols.]** The two pairwise non-collusion
statements alone are not a completed multi-role security analysis. Authenticated
transport, digests, and correctness tests do not establish transcript privacy or
protection against arbitrary malicious computation.

## 2.3 Prior work and novelty gate

Private tensor computation, two-party inference, and trusted verified delegation
provide related design points [@knott2021crypten; @huang2022cheetah;
@tramer2019slalom]. Their numerical and security contracts differ from this
software-only public-weight composition. They are background, not automatically
qualifying end-to-end baselines.

**[P2-A: primary-source review of distributed point functions/private lookup,
secret-shared full heads, tied-weight ownership, and contemporary complete private
LLM systems.]** Record what each competitor protects, assumes, implements, and
measures. Missing code remains missing evidence, not a loss. Qualify current
complete-system baselines before using field-wide SOTA wording. Novelty and a
benchmark win are separate obligations.

# 3 Coupled boundary execution

    Offline Preparation ---- corrections ----> Inference
             ^                                  ^
             |                                  |
             +------------- Client -------------+
                           /      \
                  Boundary A      Boundary B
                 private lookup and shared full head

**Figure 1.** Five-role execution. The boundary pair is additional to the prepared
body. Each remote role has explicit custody and resource costs.

## 3.1 Admission and private lookup

Before a session, the client and workers bind the source, model configuration,
schedule, quantized weights/scales, public workload, and codec choices. Each
boundary worker owns one immutable tied matrix shared by lookup and head
evaluation. The client bundle contains no vocabulary matrix.

The installed lookup uses paired packed distributed-point-function queries. Each
worker evaluates its query against the complete table, and the client combines
the replies to recover the original quantized row and scale. Repeated token IDs
receive fresh queries. **[P2-B: cite and describe the precise construction, its
security parameters and assumptions, and the packed-layout adaptation.]**

## 3.2 Decoder body and full head

The client keeps tokenization, activation quantization and scales, nonlinear work,
attention, full KV, complete logits, and sampling. Preparation issues fresh body
corrections before online execution; Inference handles the original masked stages.

For a quantized hidden vector $h$ and public head $W$, the client draws fresh
uniform $a$ in the admitted ring and sets $b=h-a$. The workers return their
respective products. In ring32,

$$
Wa+Wb=Wh \pmod{2^{32}}.
$$

An admitted signed-output bound makes centered reconstruction unambiguous. The
client applies the original float32 scale expression and its existing sampler.
This algebra establishes the linear reconstruction step, not the complete
protocol's privacy. Identical embedding records, integer results, and local
arithmetic provide the basis for decoder equivalence; execution tests check the
actual bindings and state transitions.

## 3.3 Delivery and lifetime

Independent head requests can execute concurrently. Exact residue replies discard
only bits that vanish in the public output ring; input shares remain ring32.
Uniform geometry bounds and optional source-certified per-row bounds have distinct
admission and publisher contracts. They are separate ablations rather than assumed
improvements. No vocabulary rows are excluded from logits.

One-use contexts bind roles, stages and operation counts. Replay, malformed replies,
partial admission, cancellation and expiry retire owned material; concurrent calls
drain before retirement. The final selected token remains pending until executed
by a later step. **[P2-B: complete the formal state machine and equivalence argument.]**

# 4 Implementation and resource ownership

The SDK selects this composition with `PrivateLookupAndFullHead` and
`PreparedBoundaryRoles`. Native contracts bind sources and bounded operations;
Python owns orchestration and application state. Boundary workers share the same
immutable native weight owner between lookup and head, avoiding a second complete
lookup-table snapshot per worker.

Measure the whole client, including SDK, supervisor, dashboard, rate relay, audit,
and any additional helper. All four provider roles have separate CPU/memory
owners. The complete startup-through-cleanup window charges setup and retirement.
Public artifacts, private files, checkpoint distribution, and provider provisioning
retain explicit delivery and storage costs.

More workers also add access capacity under equal per-party caps. Report both
deployment-realistic per-party limits and a resource-normalized sensitivity
comparison before attributing latency gains to the algorithm.
**[P2-C: freeze exact live allocation bounds, workload admission, and resource
contracts; distinguish observed peak RSS from conservative admission.]**

# 5 Evaluation protocol

**[P2-D: freeze this campaign before measurements and finalist selection.]**

- **Baselines:** maintained prepared; strongest eligible memory, traffic/placement,
  and latency/decode recipes; two-offset where admitted; serial/u32 boundary;
  final coupled design. Preserve verified and unverified classes. Include clear
  client-only as a reference, not an outsourcing winner.
- **Workloads:** varied held-out prompts, initially 150/288 inputs and 8/32/64
  outputs; separate fixed-length and natural-EOS cells. Vary cold/warm public
  artifacts and source distribution explicitly. Claim model generality only if
  multiple checkpoints qualify.
- **Correctness:** every complete logit/KV state, positions/history, pending
  selections, output and sampler witnesses; greedy and stochastic execution;
  failure/cancellation controls. Independent model quality stays separate.
- **Measurements:** one frozen build per matched comparison, balanced serial fresh
  clients, controlled host load, complete directed traffic and CPU/memory windows.
  Report variability, individual peaks, exclusions and pressure-rejected attempts.
- **Ablations:** coupled ownership, serial/parallel, u32/uniform residues,
  uniform/certified residues, client lifetimes, publication and delivery.
- **Sensitivity:** output length, cache lifecycle, link/RTT, provider capacity and
  client budget. Do not extrapolate an unmeasured cold/warm crossover.

**[P2-D Figure 2/Table 1: complete-cost frontier against current eligible baselines.]**
**[P2-D Figure 3: ablations, crossover and provider-resource sensitivity.]**
**[P2-E: independently confirm the frozen finalist and audit raw evidence.]**

## 5.1 Existing evidence, separately scoped

An archived source-present Qwen2.5 150+8 confirmation compares nine fresh clients
across maintained prepared, serial/u32 boundary, and parallel/residue boundary
execution. All 78,842,880 checked logit/KV values and state/output commitments agree,
with zero observed swap growth [@pllm2026boundaryevidence]. Against its maintained
prepared control, parallel/residue traffic falls 23.70% and full request latency
24.18%; the largest complete-client peak is 224.13 MiB. Decode is 32.68% slower
and online first text is 8.00% later.

This confirms those selected finalists under that build and workload. It does not
rerun all earlier incumbents, measure source/tokenizer distribution, establish
Internet latency, or fill the new campaign's result slots. Separate longer-output
screens retain their own evidence tier. The complete historical source and raw
cohort paths are linked in the cited archive.

# 6 Discussion and conclusion

Coupling lookup and head removes client vocabulary delivery while adding recurring
boundary traffic and computation. The useful region depends on request length,
cache lifecycle, provider resources, and the explicit trust contract. Fitting Low
is distinct from minimizing memory, and exact quantized execution is distinct from
representative model quality.

**[P2-E: state the confirmed primary result and measured regressions.]** A claim of
lowest measured communication requires comparison with the strongest eligible
communication baseline. A Pareto improvement alone establishes neither that
minimum nor universal superiority. Use "best among evaluated compatible
configurations" until the external-competitor review supports a broader statement.

**[P2-F: freeze the publication artifact, replace the working Paper 1 citation
with its assigned identifier, and independently replay the complete release.]**

# References
