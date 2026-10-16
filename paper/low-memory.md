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
  Private inference can outsource transformer computation while leaving substantial model-boundary costs at the client. Token lookup and output prediction may require a large vocabulary matrix, increasing cold delivery and local ownership. We study a coupled design that moves both operations to two boundary workers under explicit non-collusion assumptions, retaining the prepared decoder body, client-side state, complete logits and original sampler. Private lookup and shared linear delegation are prior art; the candidate contribution is exact execution with bounded resource and lifecycle ownership. A historical nine-client Qwen2.5 confirmation, with 150 input and eight output tokens under local 100/40 Mbps access emulation, records 23.70% fewer all-link TCP-stream bytes and 24.18% shorter requests than its maintained prepared control. The largest complete-client peak is 224.13 MiB, but decode is 32.68% slower and online first text arrives 8.00% later. Source-present cold-client savings therefore establish a scoped tradeoff, not a general speed or memory improvement. [P2-A/P2-D/P2-E: qualify current complete-system competitors and complete the paper-specific campaign and independent confirmation before making a frontier or SOTA claim.]
---

*Working manuscript; novelty, concrete primitive review, and the publication
comparison remain open. Historical results below are not a new SOTA claim.*

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
client. Moving both avoids that ownership, but remote head calls add per-token
traffic and dependent work.

We study coupled private boundary execution: two workers share responsibility for
private lookup and exact full-head evaluation over their existing immutable weight
owners. The client receives the required embeddings and reconstructs every logit,
retaining its original sampler and decoder state. The prepared transformer body
continues to execute through its original protocol. The design targets lower
complete-request communication within the Low-tier budget.

Private lookup and shared linear delegation are established techniques.
Gupta, Katz and Miers already combine private embedding lookup, public-weight
delegation and client-local nonlinear work [@gupta2026outsourcing]. Our candidate
contribution is narrower: preserving a compiled decoder's exact numerical and
state contract while coupling tied-weight ownership, bounded lifecycles and
complete-client resource measurement. We evaluate where that composition saves
communication and where recurring boundary work makes it costly. Smaller
vocabulary ownership alone is not a claim of smaller process peak memory.
**[P2-A: qualify the closest complete systems and determine whether the remaining
implementation contribution and measured frontier advance support a method paper.]**

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

The trusted client communicates with offline Preparation P, online Inference I,
and two boundary workers B0 and B1. Providers are static semi-honest: they follow
the computation and may retain their views. An allowed coalition contains neither
both P and I nor both B0 and B1. The four maximal allowed coalitions are P+B0,
P+B1, I+B0 and I+B1; their subsets are allowed. Every larger coalition contains a
forbidden pair. B0+B1 reconstructs boundary secrets; P+I removes the body input
mask to the transmitted packed width, recovering the signed input when that
encoding is injective on its admitted range. Separate operator labels do not
establish this separation in deployment.

Section 3.4 gives a conditional argument for the untimed abstract protocol.
It assumes trusted client/setup, confidential authenticated channels, fresh
independent randomness and the stated primitive security. Public model geometry,
logical schedule, message structure and semantic input/output counts are
disclosed. Token values, private scales, complete logits and KV are not public
inputs. Actual timing, clock-derived fields, backpressure and errors are outside
that theorem; their joint leakage requires separate analysis.

Logical one-use retirement is not secure erasure. Retained stage seeds can
regenerate masks, and saved lookup keys can survive destruction of one runtime
owner. The coalition restriction applies to historical views as well as live
processes. No adaptive-corruption, forensic-erasure or malicious-provider
guarantee is claimed. Source/certificate publishers are trusted for admitted
identities; a digest is not remote attestation.

## 2.3 Prior work and novelty gate

Private tensor computation, two-party inference, and trusted verified delegation
provide related design points [@knott2021crypten; @huang2022cheetah;
@tramer2019slalom]. Their numerical and security contracts differ from this
software-only public-weight composition. They are background, not automatically
qualifying end-to-end baselines.

Distributed point functions and function secret sharing supply the private-lookup
lineage [@boyle2015fss; @boyle2016fss]. FastQuery studies communication-efficient
encrypted embedding queries, and BumbleBee combines HE and secret-sharing
Transformer inference [@lin2024fastquery; @lu2025bumblebee]. Their primitives,
numeric adaptations and model-privacy contracts differ from this design.

The closest architectural antecedent is *Private and Verifiable Outsourcing of
Open-Weight LLM Inference* [@gupta2026outsourcing]. It combines two-server XOR-PIR
for embeddings, shared public linear inference including the full head, and
client-local nonlinear computation. Model-owner preprocessing supplies reusable
verification commitments; the client still performs individual or batched checks.
Its public implementation has been located, although its original measured build
and complete-client peak remain unqualified. Hence neither client-local
nonlinearity nor combining private lookup with delegation is new here. The present candidate instead keeps
the existing prepared body, adds a separate DPF/full-head worker pair, and binds
exact float/state reconstruction and resource ownership. Whether those differences
earn a new frontier claim requires a matched complete-system comparison.

`paper/PRIOR-ART-REVIEW.md` records the 16 October source review;
`paper/COMPETITOR-QUALIFICATION.md` pins the closest implementation and its
numerical, security and placement gaps. Source availability, compatible
functionality and measured complete-client memory are separate gates. Missing
code or an unmeasured memory footprint is not evidence that a competitor loses.
No field-wide SOTA claim follows from the current local comparisons.

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

The lookup is a clean-room binary-tree XOR DPF with 128-bit root seeds and
128 selection bits per terminal group, following the DPF/FSS lineage
[@boyle2015fss; @boyle2016fss]. The two selection vectors XOR to the one-hot vector
for the private token. Each worker scans the full table and XORs its selected
records; combining both replies therefore returns exactly the requested row,
including its scale bytes. Padding of the final public group is ignored.
Fresh roots and query identifiers protect repeated queries under the assumed
single-key privacy of this actual packed construction. Variable-key AES-128
supplies domain-separated tree expansion. Independent review of the concrete
instantiation and its security reduction remains required.

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
by a later step. The current boundary composition rejects previous-response state
reuse, so pending state does not imply an installed continuation API. On failure,
already delivered output is only a prefix; successful complete-response equivalence
does not apply to an unfinished attempt.

## 3.4 Correctness and conditional privacy

For signed i8 head weights and quantized inputs in $[-127,127]$, row $j$ has public
bound $B_j=127\sum_i|W_{ji}|$. Workers may return residues modulo $2^{b_j}$ if
$B_j<2^{b_j-1}$. Reduction is a ring homomorphism, so adding the two residues and
center-decoding yields the original integer even when either individual share
wraps. The client then applies the original separate float32 scale multiplies,
without reassociation. Geometry-only widths and source-certified widths use
different public commitments.

Induction over the unchanged compiled operations gives decoder equivalence.
Lookup returns the same records; the prepared body reconstructs the same integer
products; local attention, nonlinear operations and KV updates receive identical
inputs; and the head returns all original logits. With identical sampler RNG
state, selections and its subsequent state agree. Cryptographic randomness stays
fresh and independent of sampler seeding. The argument covers the fixed quantized
decoder, not equivalence to BF16/FP32 or a new task-quality result.

For privacy, consider the maximal coalition I+Bi. Replace secret seeded body
masks with independent uniform vectors under the stream-expansion assumption.
For a public body matrix A, the map

$$
(r,s)\longmapsto(u=x-r,\ c=Ar-s)
$$

is bijective for every fixed x. A causal simulator samples c offline, later
samples u uniformly, and derives the reply $Au+c$. Conditional on c, r remains
uniform; s need not. Independently, it replaces Bi's lookup key by a fresh key
for a dummy index under the actual DPF single-key assumption, and samples its
head input share uniformly. For P+Bi it instead honestly samples P's seeds and
corrections, which never depend on x, and applies the same boundary simulation.
Projecting these four joint views covers every allowed smaller coalition.

This argument applies to an honest abstract execution with semantic lengths and
stopping independent of cryptographic coins. Fresh rows, queries and head shares
permit causal hybrids through adaptive token feedback. It is not valid to
condition that argument on arbitrary real timing/error traces: even uniform
marginal payloads can correlate with coin-dependent leakage. Actual `server_ns`
fields, timeouts and machine side channels need a separate joint analysis.
`paper/SECURITY-AND-CORRECTNESS.md` supplies assumptions, transitions and proofs;
12 bounded tests exercise 57,600 algebra cases, every toy coalition and explicit
leakage/retirement counterexamples. They do not prove the cryptographic assumptions.
**[P2-B: independent concrete-DPF and implementation-side-channel review.]**

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

**[P2-D Figure 2/current-baseline table: complete-cost frontier against current eligible baselines.]**
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

The following table preserves those three separate original cohorts. Each row is
three fresh clients; request time, traffic, and decode are medians, while client
memory is the largest complete-client lifetime peak. All use source-present cold
requests and local shared 100/40 Mbps native access pacing.

| Inputs + outputs | Recipe | TCP MB | Request s | Decode tok/s | Max client MiB |
| :--- | :--- | ---: | ---: | ---: | ---: |
| 150+8 | Maintained prepared | 478.65 | 98.96 | 4.92 | 234.20 |
| 150+8 | Coupled parallel/residue | 365.22 | 75.03 | 3.31 | 224.13 |
| 150+32 | Maintained prepared | 536.30 | 106.50 | 4.94 | 218.81 |
| 150+32 | Coupled parallel/residue | 445.96 | 84.32 | 3.32 | 230.69 |
| 150+64 | Maintained prepared | 613.15 | 124.07 | 4.80 | 235.11 |
| 150+64 | Coupled parallel/residue | 553.61 | 105.28 | 3.26 | 230.56 |

**Table 1.** Retained source-present cohorts; 150+8 is a confirmation and the longer
workloads are separate initial screens. Each coupled configuration adds two
providers. The table is neither equal-aggregate-resource timing nor a comparison
with all contemporary methods. Cold tokenizer/checkpoint distribution and physical
wire remain unmeasured.

The complete audit additionally retains serial/u32 controls. Parallel/residue
decode improves 52.87%, 53.79%, and 52.89% against those respective controls, yet
remains slower than maintained prepared execution. At eight outputs, online first
text is 8.00% later than prepared; at 64 outputs, aggregate cold-first CPU is
48.69 versus 47.76 seconds, measured through the first response and excluding
cleanup. These tradeoffs must accompany the traffic result.

Original reports document 78,842,880, 336,605,184, and 729,833,472 checked logit/KV
values respectively, with zero observed swap growth. The publication audit
rechecks recorded commitments and costs, not the original tensor execution.
`node scripts/audit_publication_evidence.mjs --check` reproduces the table sources;
`paper/artifacts/historical-results-v1.json` records source/build/configuration
digests and every replicate. No cohort supplies another's missing observations.

Memory eligibility is sensitive near a threshold: the short confirmation's
candidate exceeds 224 MiB, despite fitting 256 MiB. The observed largest peak,
not its lower median or vocabulary-storage reduction, determines this result.

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
