---
title: "Communication-Efficient Private LLM Inference under a 256 MiB Client Budget"
description: "Exact coupled vocabulary boundaries, explicit resource ownership, and scoped CPU-loopback evidence for private quantized decoding."
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
  Outsourcing transformer computation does not remove client costs at the vocabulary boundary. We study a coupled design that places private embedding lookup and complete output-head evaluation on two additional workers sharing immutable public weights within each worker. The prepared decoder body is unchanged; the client retains attention, nonlinear work, full key-value state, complete logits, and sampling. Admission binds the source, compiled schedule, numerical contract, operation budgets, and output codec.

  Private lookup and shared linear delegation are prior art. The contribution examined here is their exact quantized-decoder composition with explicit lifetime and resource ownership. Algebraic correctness requires admitted signed bounds and unchanged float32 operation order. The privacy argument is conditional on the concrete distributed point function and independent fresh masks, and covers only static semi-honest coalitions in an untimed abstraction, not implementation timing or errors.

  A historical nine-client Qwen2.5 confirmation, with 150 input and eight output tokens, records 23.70% fewer all-link TCP-stream bytes and 24.18% shorter requests than its maintained prepared control. The largest complete-client peak is 224.13 MiB, below the 256 MiB budget, but decode is 32.68% slower and online first text 8.00% later. These CPU-loopback measurements use source-present workers, cold client caches, and local 100/40 Mbps pacing. They establish neither a tenfold memory reduction nor Internet performance, representative model quality, or superiority to qualified contemporary complete systems. A fresh matched campaign and independent security, quality, and comparison review remain necessary.
---

*This manuscript is a draft. Novelty, review of the concrete primitives, and the
publication comparison are open items. The previous results below are not a new
SOTA claim.*

# 1 Introduction

Moving transformer-body weights to providers leaves vocabulary lookup, output
prediction, tokenization, state, and buffers at the client. In a tied model,
private lookup alone cannot remove the vocabulary matrix while the full head
remains local. Moving both operations can remove its client ownership, but adds
dependent traffic and provider work for each token.

We examine two additional workers sharing immutable lookup/head weights *within
each worker*, alongside the unchanged prepared body. The client keeps attention,
nonlinear work, full KV state, every logit, and sampling. The objective is lower
complete-request communication within PLLM's 256 MiB whole-client Low tier
[@hudson2026pllmframework], not minimum memory regardless of cost.

Private lookup and public linear delegation are prior art, including the
two-server design of Gupta, Katz and Miers (GKM) [@gupta2026outsourcing]. The
composition studied here binds exact quantized state, tied-weight ownership, and
bounded lifecycles to complete-client measurement. Historical evidence identifies
a cold-delivery/recurring-work tradeoff, not a competitive frontier or tenfold
memory improvement. Detailed contracts and proofs are retained in the appendices.

# 2 Problem and comparison class

## 2.1 Objective

The primary metric is complete-request all-link TCP-stream bytes, counting each
directed transfer once, including Preparation and both boundary workers. Latency
is secondary; first text, $N-1$ decode throughput, CPU/RSS, and provider resources
remain separate objectives. Application bodies and physical-wire bytes are not
TCP streams. Source-present cold clients start with empty bundle caches and fresh
private material while workers already hold checkpoint/tokenizer sources;
warm-public and source-cold provisioning require separate cells.

Exactness is to a fixed per-row W8A8 decoder, including scales, rounding, signed
bounds, float32 operation order, logits, active KV/positions, history, pending
selection, and sampling. It is not BF16/FP32 equivalence or representative quality.
Independent-reference fidelity and held-out quality remain separate gates.

## 2.2 Parties and trust

The trusted client uses Preparation P, Inference I, and boundary workers B0/B1.
Static semi-honest coalitions may contain neither P+I nor B0+B1; the maximal
allowed sets are P+B0, P+B1, I+B0, and I+B1. Security assumes trusted client/setup
and source publishers, authenticated confidential channels, fresh independent
randomness, body-stream pseudorandomness, and the actual packed DPF's single-key
privacy. Public geometry, schedule, message structure, and semantic lengths leak.

The conditional theorem is untimed and excludes clock fields, completion races,
errors, and backpressure. Logical retirement is not erasure; no adaptive-corruption,
malicious-provider integrity, or availability guarantee is supplied. The installed
composition is local-child-only and rejects network deployment and verification.
Measurements do not demonstrate independent custody. Appendices A/B retain the
coalition reasoning, precise premises, and proof.

## 2.3 Prior work and contribution scope

CrypTen, Cheetah, and Slalom have different computation/trust contracts
[@knott2021crypten; @huang2022cheetah; @tramer2019slalom]. DPF/FSS provides the
lookup basis [@boyle2015fss; @boyle2016fss]; FastQuery and BumbleBee address
embedding queries and protected Transformer computation [@lin2024fastquery;
@lu2025bumblebee]. Related-work citation alone does not qualify a baseline.

GKM places lookup, body, and full head on the same two servers with client-local
nonlinearity and owner-preprocessed verification. This candidate instead retains
the prepared body and adds a separate boundary pair. GKM source is public; its
unmeasured client footprint is not evidence of inferiority. A current-build,
qualified complete-system comparison remains missing. Appendix A preserves the
source review and numerical/security/placement distinctions.

# 3 Coupled boundary execution

    Offline Preparation ---- corrections ----> Inference
             ^                                  ^
             |                                  |
             +------------- Client -------------+
                           /      \
                  Boundary A      Boundary B
                 private lookup and shared full head

**Figure 1.** Five-role execution. Boundary A/B are B0/B1, additional to the
prepared body. Every remote role has separate custody and resource costs.

## 3.1 Admission and private lookup

Admission binds source, schedule, weights/scales, workload, and codecs. The
checked specialization requires batch one, a tied bias-free full head, CPU W8A8,
and the ordinary prepared body; it rejects state reuse, placement/region variants,
and decode epochs. Actual source imports, not request-supplied hashes alone,
authorize the session. A binary-tree XOR DPF supplies one selection share to each
worker. Both scan their whole table; XORing replies recovers the original embedding
record and scale bits. Single-key privacy of this concrete packed construction
still needs independent review (Appendix B.1).

## 3.2 Decoder body and full head

For each body row, Preparation supplies $c=Ar-s$; the client sends $u=x-r$,
receives $Au+c$, and adds $s$ to recover $Ax$ in the admitted ring. Attention,
nonlinearity, dequantization, and KV updates remain unchanged at the client.
For the head, fresh uniform ring32 shares $a,b=h-a$ give
$Wa+Wb=Wh\pmod{2^{32}}$. Full ring32 inputs remain unchanged when output residues
are narrower. The client reconstructs all logits using the reference float32
expression and sampler, not an approximate candidate shortlist.

## 3.3 Delivery and lifetime

Parallel dispatch overlaps the two *current* head calls, not future autoregressive
steps. The retained candidate uses public uniform residue bounds, distinct from
optional source-certified row widths. Replay, malformed replies, partial
admission, cancellation, and expiry retire one-use owners after concurrent work
drains. The last selection remains pending until a completed feedback step;
previous-response state reuse is unsupported. Per-token scans and full-vocabulary
head replies add recurring work. Appendix B.3 gives exact operation/payload counts,
EOS accounting, and failure semantics; these do not locate a cold/warm crossover.

## 3.4 Correctness and conditional privacy

A residue width must satisfy the public signed-output bound before centered
reconstruction; an in-range result cannot detect modular aliasing from an
undersized layout. Identical embedding records, integer products, float32
multiplication order, initial state, and sampler RNG give decoder equivalence by
induction. This is exact quantized execution, not a BF16/FP32 or task-quality result.

Conditional on the premises in §2.2, causal hybrids simulate each allowed static
coalition for semantic executions with equal coin-independent leakage. Body masks,
lookup keys, and head shares use independent cryptographic randomness, separate
from sampler seeding. Neither SHAKE collision resistance nor the DPF name supplies
these premises. Real duration/error transcripts require separate joint leakage
analysis. Appendix B.4 retains the integer/float argument, simulator, 12 bounded
tests and 57,600 algebra cases; tests do not prove cryptographic assumptions.

# 4 Implementation and resource ownership

## 4.1 Shared owners and bounded interfaces

`PrivateLookupAndFullHead` and `PreparedBoundaryRoles` select the composition.
Each worker holds one immutable matrix/scale owner for lookup and head; the two
workers deliberately replicate public weights, rather than sharing one process's
storage. The client keeps public scales but no tied matrix. Native admission
bounds geometry, frames, operations, and contexts; these are not whole-process RSS
bounds. Appendix C.1 retains payloads, interface limits, source owners, and
conservative scratch/frame estimates.

## 4.2 Whole-client and provider accounting

Eligibility uses the largest unrounded accepted fresh-client peak, not median
RSS. In these reports one process owns SDK, supervisor, dashboard, relay, and
audit. `process.peak_rss` includes target finalization; its final observation is
before envelope serialization and exit, not an after-exit OS high-water mark.
Separate serving helpers would require simultaneous sums. Mapped resident pages
count; provider memory, external cache, private files, and devices remain separate.
Observed eligibility is neither whole-topology admission nor a future bound.
Zero observed swap growth does not mean zero swap use.

Historical aggregate CPU ends at first response; provider RSS is request-window
sampled, not a lifetime or simultaneous aggregate peak. Request timing excludes
provider startup; target timing includes ordinary startup/cleanup, not source
provisioning. Four providers each receive 100/40 Mbps, twice the summed provider
allowance of the two-provider control. Link normalization alone cannot equalize
CPU/RSS or a shared uplink. Full provider-lifetime accounting and hard aggregate
CPU/RSS enforcement remain missing. Appendix C.2 gives the complete accounting
contract; lower local latency is not an equal-resource algorithmic gain.

## 4.3 Memory-reduction claim boundary

The urgent client-memory research target is tenfold lower matched whole-client
peak RSS without changing the body, tokens, numerical contract, private workload,
or outsourcing requirement. The separate Qwen3-4B 64+8 starting estimate is
8.25 GiB, giving an approximately 0.83 GiB working estimate budget
(`docs/research/low-client-network.md`). Neither number is measured RSS; that
budget is also distinct from the 256 MiB Low tier. Repricing existing last-use
release, reducing a mask inventory, or removing a vocabulary payload cannot close
this target. No matched isolated Qwen3-4B 64+8 complete-client peak pair is supplied
here. A tenfold claim must use unrounded matched peak observations, not the rounded
0.83 GiB estimate ceiling. The Qwen2.5 results below cannot serve as its baseline
or candidate.

# 5 Evaluation protocol

Ordinary experiment/response/suite interfaces select maintained prepared,
serial/u32, and parallel/uniform-residue arms. Each cohort fixes build, source,
tokenizer, fixture, output cap, numerical body, and links; three fresh processes
run one cold request each, in balanced serial order with one thread and no warmup.
The client-charged state audit compares complete logits, active KV, positions,
selections, RNG, and output commitments. Temperature zero/seed 42 trajectories
are not exhaustive intermediate-state or stochastic conformance, nor task quality.
Different private salts prevent cross-cohort commitment comparison; repetitions of
one prompt do not supply prompt breadth. Appendix D preserves full event windows,
unused-material accounting, rejected attempts, and reproducible report paths.

## 5.1 Previous evidence, with separate scope

An archived confirmation compares nine fresh clients across three Qwen2.5
configurations with 150 input tokens and eight output tokens. These configurations
are maintained prepared, serial/u32 boundary, and parallel/residue boundary
execution. This confirmation uses source-present cold clients. All 78,842,880
compared logit/KV values agree. All state/output commitments agree. The
recorded swap growth is zero [@pllm2026boundaryevidence].

Compared with maintained prepared, parallel/residue traffic decreases by 23.70%,
and full-request latency decreases by 24.18%. The largest complete-client peak
for parallel/residue is 224.13 MiB. Decode is 32.68% slower, and online first-text
time is 8.00% longer. The confirmation neither reruns every earlier incumbent nor
measures source distribution, Internet latency, or the new campaign.

![**Figure 2.** Historical source-present CPU-loopback observations. Each marker pairs one replicate's client-process RSS peak through target finalization (MiB; §4.2) with its all-link TCP-stream bytes through scenario end (decimal MB); overlaps are not jittered. Provider RSS is not plotted. Dashed lines mark 256 MiB. Panels are distinct 150+8 confirmation and 150+32/64 screens: the same recorded build, different private salts, not new trials or pooled observations. Coupled arms add two workers. Physical-wire and source-distribution costs are unmeasured; no crossover or publication ranking is inferred.](figures/low-memory-history.png)

Each table row has three fresh clients: traffic, request time, and decode are
medians; memory is the largest full-client peak. All cohorts use source-present
cold requests and shared native 100/40 Mbps access pacing. The longer-output
cohorts are separate initial screens, not extensions of the short confirmation.

| Inputs + outputs | Recipe | TCP MB | Request s | Decode tok/s | Max client MiB |
| :--- | :--- | ---: | ---: | ---: | ---: |
| 150+8 | Maintained prepared | 478.65 | 98.96 | 4.92 | 234.20 |
| 150+8 | Coupled parallel/residue | 365.22 | 75.03 | 3.31 | 224.13 |
| 150+32 | Maintained prepared | 536.30 | 106.50 | 4.94 | 218.81 |
| 150+32 | Coupled parallel/residue | 445.96 | 84.32 | 3.32 | 230.69 |
| 150+64 | Maintained prepared | 613.15 | 124.07 | 4.80 | 235.11 |
| 150+64 | Coupled parallel/residue | 553.61 | 105.28 | 3.26 | 230.56 |

**Table 1.** Retained source-present cohorts. The 150+8 cohort is a confirmation,
and the longer workloads are separate initial screens. Each coupled configuration
adds two providers. The table gives neither timing with equal aggregate resources
nor a comparison with all available methods. The cohorts do not measure cold
tokenizer/checkpoint distribution or physical-wire traffic.

Against serial/u32, parallel/residue decode increases 52.87%, 53.79%, and 52.89%
in the respective cohorts, but remains below maintained prepared. At 64 outputs,
aggregate cold-first CPU is 48.69 seconds for parallel/residue versus 47.76 for
prepared, through first response and excluding cleanup; fewer bytes do not imply
less aggregate CPU. The respective cohorts compare 78,842,880, 336,605,184, and
729,833,472 logit/KV values and record zero swap growth. Historical table checking
does not rerun these tensors. The short candidate's maximum exceeds 224 MiB but
is below 256 MiB: this is observed eligibility, not lowest memory or new admission.

# 6 Discussion and conclusion

Removing client vocabulary delivery adds recurring boundary work. The supported
result is a source-present cold-request communication/latency tradeoff against the
retained control, with slower decode and later first text. Longer-output screens
show smaller traffic savings, not a measured crossover. Low eligibility is not
minimum memory; exact quantized execution is not task quality or a tenfold gain.

Missing experiments include matched isolated Qwen3-4B 64+8 peaks; a qualified
current-build complete-system comparison including GKM-style lookup/body/head;
varied held-out greedy/stochastic and independent-quality tests; cold/warm and
source-distribution costs; and provider-lifetime/resource-normalized measurements.
A frozen finalist needs fresh confirmation with retained controls and predeclared
exclusions. The broader ten-method implementation effort cannot fill these cells.

Independent packed-DPF, joint timing/error, and deployment-custody review remains
open. No GPU, physical-WAN, energy, secure-erasure, malicious-complete, universal
memory, or field-wide SOTA result is supplied. Release freeze, independent
reproduction, benchmark/quality review, and author review remain necessary. The
framework citation is still a working manuscript without an assigned public
identifier. Appendices preserve technical detail, not additional completed claims.

```{=latex}
\clearpage
\appendix
```

# Appendix A: comparison and prior-art details

## A.1 Boundary motivation

Outsourcing most transformer computation does not necessarily make a client
lightweight. Vocabulary lookup, output prediction, tokenization, decoder state,
and temporary buffers remain owned somewhere. Their placement determines both
live memory and the public artifacts a cold client must acquire. A source-present
cold request starts fresh client processes and private material with an empty
client bundle cache, while workers already possess checkpoint/tokenizer sources.
This differs from provisioning a source-cold deployment.

We use the PLLM framework's numerical, security, and resource contracts
[@hudson2026pllmframework]. Its Low tier constrains complete-client peak RSS to
256 MiB, including serving helpers over their simultaneous lifetime. Arithmetic,
scales, rounding, and operation order define the reference execution. The ceiling
is an eligibility condition, not an objective that makes every eligible design
equivalent: communication, latency, local computation, and provider resources
retain separate costs. The framework citation remains a working manuscript.

This paper examines the model boundary. Token lookup selects an embedding, which
is a numerical vector, for an input token. The output head calculates a score,
or logit, for each vocabulary token from the decoder's hidden vector. Models with
tied embeddings use the same matrix for the two operations. If only lookup moves
to a worker, the head matrix stays at the client. If lookup and head evaluation
move, the client can remove this matrix, but remote head calls add per-token
traffic and dependent work.

The coupled design uses two workers for private lookup and exact full-head
evaluation. With the specified assumptions, private lookup supplies the requested
embedding without disclosure of the token to either worker. Each worker keeps
its immutable weight owner and uses it for the two operations. A weight owner is
a storage object that controls access to a matrix and its lifetime.

The client receives embeddings, reconstructs every logit, and applies the unchanged
sampler. It retains decoder state, including the key-value (KV) cache of previous
attention keys and values. The prepared transformer body keeps the same protocol.
The objective is lower complete-request communication within the Low-tier ceiling.

Private lookup and shared linear delegation are prior art. In shared linear
delegation, workers calculate products from shares that together encode a private
input. With the specified privacy assumptions, either worker's share alone does not
disclose the input. Gupta, Katz and Miers use private embedding lookup,
public-weight delegation, and client-local nonlinear work together
[@gupta2026outsourcing].

The contribution under examination is exact numerical/state preservation coupled
to tied-weight ownership, bounded lifecycles, and complete-client measurements.
We examine where this composition decreases communication and where repeated
boundary work increases costs. Less vocabulary storage alone does not establish
a lower process-memory peak.

The evidence therefore addresses a narrower question: can removing client
vocabulary ownership preserve the admitted decoder while exchanging cold delivery
for recurring private boundary work? It does not yet establish a competitive
complete-system frontier.

## A.2 Objective and numerical identity

The primary objective is lower all-link TCP-stream communication for a complete
request, counting each directed link once, including provider-to-provider
preparation and both boundary workers. Application bodies and physical-wire bytes
are separate measurements; neither is interchangeable with TCP-stream bytes. The
design must keep the fixed quantized decoder and obey the specified privacy
contract. It must not use more client memory than the Low-tier budget. The
secondary objective is complete-request latency.

The admitted path uses symmetric per-row W8A8 quantization. That label alone is
not a numerical identity: scales, rounding, signed bounds, nonlinear expressions,
and float32 operation order belong to the fixed reference computation.

The reports must also give online first-text time, N−1 decode throughput, client
CPU/RSS, and all provider resources. For N output tokens, N−1 decode throughput
divides the number of tokens after the first by their execution time. RSS is
resident set size. These measurements give the other costs of each design.

In a source-present cold-client case, workers have the checkpoint and tokenizer
sources before the request. The case uses new processes and new private material.
The client's public bundle cache is empty. A warm-public case keeps public
artifacts but uses new private material. It does not reuse private responses,
prefixes, or KV. These cases have separate comparison cells.

Public artifacts at the client before the request have separate costs. In the
previous cohorts, indexed-tokenizer delivery stays unmeasured.

A source-cold case also includes checkpoint/tokenizer distribution, provider
setup, public indexing/certification, and startup before the first response.
Separate measurements are necessary for that case.

Exactness refers to equality with the reference quantized execution. It includes
complete logits, active KV and positions, executed history, the pending token,
and sampling behavior. A pending token is a selected token that the decoder did
not use as input in a completed step.

Exactness here does not include equivalence to the checkpoint's BF16/FP32
computation. It gives no result for task quality. Independent-reference fidelity
and held-out task quality have separate evidence. One checkpoint gives evidence
for a checkpoint-specific claim only.

## A.3 Parties and trust

The trusted client communicates with offline Preparation P, online Inference I,
and two boundary workers B0 and B1. Providers are static semi-honest. A static
adversary selects its providers before execution. Those providers obey the
protocol but can keep their views. A view includes the messages, secrets, and
state that a party can access.

A permitted coalition contains no P+I pair and no B0+B1 pair. The four maximal
permitted coalitions are P+B0, P+B1, I+B0 and I+B1. Their subsets are also
permitted. Each larger coalition contains P+I or B0+B1.

B0+B1 reconstructs boundary secrets. P+I removes the body input mask to the
transmitted packed width. It reconstructs the signed input if that encoding is
injective on its admitted range. Different operator labels are not evidence for
this separation in a deployment.

Section 3.4 summarizes the conditional theorem; Appendix B.4 gives its argument
for the untimed abstract protocol with static providers. Its assumptions include
a trusted client/setup, confidential authenticated channels, fresh independent
randomness, and the specified primitive
security. Public leakage includes model geometry, the logical schedule, message
structure, and semantic input/output counts. Token values, private scales,
complete logits, and the KV cache are not public inputs. The theorem excludes
runtime timing, clock-derived fields, backpressure, and errors. A separate analysis
must examine their joint leakage.

A retirement rule prevents reuse of a one-use context. Logical one-use retirement
is not secure erasure. A party with stored stage seeds can calculate the masks
again. A saved lookup key can stay available after destruction of one runtime
owner. The coalition restriction applies to previous views and live processes.
We give no guarantee for adaptive corruption, forensic erasure, or malicious
providers.

The assumptions also include trusted source/certificate publishers for admitted
identities. A digest is not remote attestation.

The implemented SDK composition launches local child processes and rejects a
network deployment. The confidentiality argument describes roles under the channel
and coalition assumptions; the measurements do not demonstrate those assumptions
in operation. A host administrator, shared relay, or tracing service that can read
both members of either forbidden pair is outside the model. Public weights need
no model-confidentiality guarantee. The composition rejects verification and
provides no malicious-worker output-integrity or availability guarantee.

## A.4 Prior work and contribution scope

Private tensor computation, two-party inference, and trusted verified delegation
give related designs [@knott2021crypten; @huang2022cheetah; @tramer2019slalom].
Their numerical and security contracts differ from this software-only composition
with public weights. They give background information. That role alone does not
give qualification as end-to-end baselines.

Distributed point functions (DPFs) and function secret sharing (FSS) give previous
methods for private lookup [@boyle2015fss; @boyle2016fss]. FastQuery examines
communication-efficient encrypted embedding queries. BumbleBee uses homomorphic
encryption (HE) and secret-sharing Transformer inference together [@lin2024fastquery;
@lu2025bumblebee]. Their primitives, numerical adaptations, and model-privacy
contracts differ from this design.

The closest previous architecture is *Private and Verifiable Outsourcing of
Open-Weight LLM Inference* by Gupta, Katz and Miers (GKM)
[@gupta2026outsourcing]. XOR-PIR is private information retrieval with exclusive-OR
operations. GKM uses two-server XOR-PIR for embeddings and shared
public linear inference, which includes the full head. It also keeps nonlinear
computation at the client. The same two servers do lookup, linear body work, and
the full head.

Model-owner preprocessing supplies reusable verification commitments. The client
does individual or batched checks online.

The public GKM implementation is available. The source review does not identify
it as the build for the paper's measurements or give a measured complete-client
peak. The source at the recorded revision supplies documentary evidence only.
The review used the upstream code only as data, without a build, execution,
import, or link.

Thus, client-local nonlinearity and private lookup with delegation are not new
contributions here. The candidate keeps the prepared body and adds a
separate DPF/full-head worker pair. It binds exact float/state reconstruction and
resource ownership. A matched complete-system comparison is necessary before
these differences can give evidence for a new frontier claim. The matched
comparison with a complete two-worker system for lookup, body, and full head is
missing.

`paper/PRIOR-ART-REVIEW.md` records the 16 October source review.
`paper/COMPETITOR-QUALIFICATION.md` identifies the fixed implementation and its
numerical, security, and placement gaps. Source availability, compatible
functionality, and measured complete-client memory are separate requirements.
Missing code or an unmeasured memory footprint gives no evidence that a competitor
has worse results. These local comparisons do not give evidence for a field-wide
SOTA claim.

# Appendix B: protocol, exactness, and conditional privacy

## B.1 Admission and private lookup

Before a session, the client and workers bind the source, model configuration,
schedule, quantized weights/scales, public workload, and codec choices. One weight
owner in each boundary worker holds the immutable tied matrix. Lookup and head
evaluation use that same owner. The client bundle contains no vocabulary matrix.

This is a checked specialization of the common decoder plan, not a second decoder
implementation. `crates/pllm-compiler/src/decoder_boundary.rs` requires batch one,
one main lookup, a tied bias-free complete head, CPU W8A8, and the ordinary
prepared body. It rejects verification, state/cache reuse, placement/region
variants, and decode epochs. The source bridge in
`python/pllm/runtime/private_boundary_loader.py` re-lowers the locked checkpoint,
checks tensor headers and both phase contracts, imports only the declared head,
and compares its actual matrix and scale identities. Request-supplied digests
alone cannot authorize a session. Low-level A4 descriptors do not imply A4 support
in this compiled path.

The lookup uses a binary-tree XOR DPF, a distributed point function. This
clean-room implementation uses DPF/FSS construction principles
[@boyle2015fss; @boyle2016fss]. Each root seed has 128 bits. Each terminal group
has 128 selection bits. The XOR of the two selection vectors is the one-hot
vector for the private token.

Each worker reads the full table and calculates the XOR of its selected records.
The XOR of the two replies gives the requested row and its scale bytes without
change. The workers ignore padding in the last public group.

With fresh roots and query identifiers, the assumed single-key privacy of this
packed construction also applies to repeated queries. Variable-key AES-128
supplies domain-separated tree expansion. An independent review of this concrete
instantiation and its security reduction is necessary.

## B.2 Decoder body and full head

The client keeps tokenization, activation quantization and scales, nonlinear work,
attention, the full KV cache, complete logits, and sampling. For each one-use body
row, Preparation expands domain-separated input/output mask streams $r,s$ and supplies
$c=Ar-s$ to Inference before online use. The client sends $u=x-r$; Inference
returns $Au+c$. Adding $s$ recovers $Ax$ in the admitted ring. The same signed
bound and local dequantization as the reference recover the body result. Moving
the vocabulary boundary does not change this schedule or move attention and KV
to a provider.

For a quantized hidden vector $h$ and public head $W$, the client samples fresh
uniform $a$ in ring32. It sets $b=h-a$. These vectors are additive
shares of $h$. Head inputs use full ring32 shares, independently of any narrower
output codec. Each worker sends the product for its share. Their sum in ring32
is:

$$
Wa+Wb=Wh \pmod{2^{32}}.
$$

Ring32 arithmetic keeps only the low 32 bits of each integer result. With the
public signed-output bound, the modular result has only one possible signed
integer. Centered reconstruction selects that integer. The client uses the same
float32 scale expression and sampler as the reference. This algebra proves the
linear reconstruction step only. It does not prove privacy for the full protocol.

The decoder-equivalence argument uses the same embedding records, integer
results, and local arithmetic. The execution tests include checks of the
implementation's bindings and state transitions.

## B.3 Delivery and lifetime

Parallel dispatch overlaps the two worker calls for the current head operation;
it does not evaluate future autoregressive steps before their inputs exist.

A residue is the remainder after integer division by the modulus. Exact residue
replies discard only bits that have no effect in the public output
ring. The input shares stay ring32. Uniform geometry bounds and optional
source-certified per-row bounds have different admission and publisher contracts.
The retained Table 1 candidate uses uniform residues, not source-certified widths.
The head includes each vocabulary row in the logits.

One-use contexts bind roles, stages, and operation counts. Replay, malformed
replies, partial admission, cancellation, and expiry cause retirement of owned
material. The client waits for all concurrent calls to complete before retirement.

The last selected token stays pending until the decoder completes a subsequent
step with that token as input. This boundary composition rejects previous-response
state reuse. Thus, pending state is not evidence for an installed continuation API.

For $n$ prompt tokens and $N$ emitted non-EOS tokens reaching the cap, execution
uses $n+N-1$ lookup rows and $N$ head calls. Only the first $N-1$ selections are
fed back into KV/history. Natural EOS consumes a selection/head operation but
emits no EOS token and performs no feedback lookup for it. Unused admitted
material remains charged rather than disappearing from the request account.

The native payload account exposes the recurring cost. Let
$d=\lceil\log_2\lceil V/128\rceil\rceil$. Each worker's packed DPF key is
$97+17d$ bytes and its lookup reply is $65+H+4$ bytes. Each head sends two
$4H$-byte input shares and receives two $4V$-byte u32 outputs, or two packed
outputs of $\lceil\sum_j b_j/8\rceil$ bytes plus residue headers. These counts
exclude outer protocol/HTTP framing, controls, and the unchanged body; they are
not whole-request stream measurements. Removing one $VH$-byte client matrix
therefore adds lookup traffic per executed token and full-vocabulary replies per
selection. Each lookup still scans both worker tables. Small query size is not
small provider work, and this payload account alone cannot locate a cold/warm
crossover.

On failure, the output that the client received is only a prefix. The equivalence
claim for a successful complete response does not apply to an unfinished attempt.

## B.4 Correctness and conditional privacy

For signed i8 head weights and quantized inputs in $[-127,127]$, row $j$ has the
public bound $B_j=127\sum_i|W_{ji}|$. Workers can send residues modulo $2^{b_j}$
if $B_j<2^{b_j-1}$. Reduction is a ring homomorphism. Thus, the sum of the two
residues and centered reconstruction give the reference integer. Either individual
share can wrap without change to this result.

The client then evaluates
$\operatorname{fl}_{32}(\operatorname{fl}_{32}(\operatorname{float32}(z_j)
\,s_h)\,s_{W,j})$, with separate multiplications, the original scale bits, and
no reassociation. Integer exactness alone would not justify replacing this
expression by an algebraically equivalent real-number product. Uniform widths
use the conservative signed-byte bound $128\cdot127H$; source-certified row
widths require an independently approved publisher pin and worker rescanning of
the immutable matrix. An in-range decoded value cannot validate an undersized
layout: modular aliasing can produce an incorrect value that still passes the
range check.

Conditional on honest admitted source data, valid integer bounds, identical local
kernels and initial state, induction on the unchanged compiled operations gives
decoder equivalence. The lookup gives the same records. The prepared body
reconstructs the same integer
products. Local attention, nonlinear operations, and KV updates receive the same
inputs. The head supplies all reference logits. With the same sampler RNG state,
the selections and the subsequent sampler state agree.

Cryptographic randomness stays fresh and independent of sampler seeding.

The argument applies only to the fixed quantized decoder. It does not prove
equivalence to BF16/FP32 or give a new result for task quality.

The conditional privacy argument uses hybrid experiments and a causal simulator,
which supplies messages in protocol order without knowing future private inputs.
Its premises are trusted client/setup and source admission, confidential
authenticated channels, joint pseudorandomness of the domain-separated body mask
streams, single-key privacy of the actual packed128 DPF under fresh adaptive
queries, and independent OS randomness for body seeds, lookup keys, and head
shares. SHAKE collision resistance alone does not imply the mask-stream premise.
The DPF premise is an open cryptographic-review condition, not a theorem inherited
from the name of the primitive.

Under these premises, any two honest semantic executions with equal declared
coin-independent leakage have computationally indistinguishable joint abstract
views for every allowed static coalition. This statement excludes clocks,
race-dependent completion order, errors, and backpressure; it does not condition
a real run on observed success or timing.

For the maximal coalition I+Bi, the stream-expansion assumption lets a hybrid
replace secret seeded body masks with independent uniform vectors. Here, Bi is
B0 or B1. For a public body matrix A, this map is bijective for each fixed x:

$$
(r,s)\longmapsto(u=x-r,\ c=Ar-s)
$$

A causal simulator samples c offline. It subsequently samples u uniformly and
calculates the reply $Au+c$. Conditional on c, r stays uniform, but s can be
nonuniform.

Independently, the simulator replaces Bi's lookup key with a fresh key for a
dummy index. This replacement uses the single-key assumption for the DPF used here.
The simulator also samples Bi's head input share uniformly. For P+Bi, it instead
samples P's seeds and corrections honestly. These values are independent of x.
The simulator uses the same boundary simulation.

Each permitted smaller coalition receives a projection of one of these four joint
views. The conditional privacy theorem applies only to an honest, untimed abstract
execution with static corruption. Semantic lengths and termination must be
independent of cryptographic randomness. With fresh rows, queries, and head shares, the
proof uses causal hybrids through adaptive token feedback. Adaptive token feedback
does not give an adaptive-corruption guarantee.

The argument cannot condition on arbitrary traces of real timing or errors. Even
uniform marginal payloads can correlate with leakage that depends on cryptographic
randomness. A separate joint analysis is necessary for runtime `server_ns` fields,
timeouts, and machine side channels.

`paper/SECURITY-AND-CORRECTNESS.md` gives assumptions, transitions, and proofs.
The recorded 12 bounded tests include 57,600 algebra cases, each coalition in the small
test model, and leakage/retirement counterexamples. They do not prove the
cryptographic assumptions.

The conditional statement therefore separates algebraic exactness from
cryptographic assumptions and from implementation leakage. In particular,
serialized duration fields are part of the real transcript even if external
network timestamps are discarded. No theorem here bounds their joint leakage.

# Appendix C: implementation bounds and resource accounts

## C.1 Shared owners and bounded interfaces

The SDK selects this composition with `PrivateLookupAndFullHead` and
`PreparedBoundaryRoles`. Native contracts bind sources and bounded operations.
Python controls the remote calls and application state. Lookup and head
operations in each boundary worker use the same native weight owner. This owner
is immutable. Thus, a second full snapshot of the lookup table is not necessary
in either worker.

Concretely, `crates/pllm-garble/src/private_pages.rs::Table::Vocabulary` retains an
`Arc<Matrix>` plus the original scale bytes. Sharing is within one worker, not
between workers: the pair deliberately replicates public weights. If $V$ is the
vocabulary size and $H$ the hidden width, each worker's matrix payload is $VH$
bytes and its scale payload $4V$ bytes. The client retains public scales but no
tied weight matrix. These payload counts exclude source-import overlap, scratch,
frames, runtime overhead, and the decoder's other owners.

The compiler admits $V\leq262{,}144$, $H\leq65{,}532$, and
$V(H+4)\leq512$ MiB, subject also to ring and frame checks. The protocol bounds
lookup batches at 64 rows, admitted lookup rows and head calls at 4,096 each,
data frames at 16 MiB, and control frames at 16 KiB. Worker defaults permit eight
live sessions, 4,096 distinct contexts per incarnation, and 90-second idle expiry
checked on activity. These are independent limits, not a promise that every
combination of their maxima is admissible. They bound interfaces and owners;
they do not prove a whole-process RSS ceiling.

`python/pllm/runtime/benchmark_memory.py::private_boundary_memory` prices the same
owners before execution. It allows $64V$ scratch bytes per active lookup, not per
issued query, and separately charges session ledgers and repeated frame copies.
It retains full-width frame allowances when residue replies shrink. Parallel head
dispatch adds thread-stack and simultaneous-reply allowances; row codecs add
layout, certificate, and import work. Thus fewer transmitted bits do not imply
an equal reduction in live memory. These are conservative allocation estimates,
not observations or a proof against out-of-memory failure.

## C.2 Whole-client and provider accounting

For accepted isolated replicate $r$, let $M_r$ be the complete-client lifetime
peak. Low eligibility requires $\max_r M_r\leq268{,}435{,}456$ bytes, using
unrounded observations, not median RSS. In the retained benchmark, one fresh
process owns the SDK, supervisor, dashboard, native rate relay, and state audit;
the idle suite launcher performs no serving work. `process.peak_rss` includes
target finalization, while `client.peak_rss` retains the shorter scenario window.
The final observation precedes result-envelope serialization and process exit;
it is not an after-exit OS high-water reading.
If serving helpers become separate processes, qualification requires their
simultaneous total, not a sum of unrelated lifetime peaks. Mapped pages resident
in RSS remain charged. Provider RSS, filesystem cache outside RSS, disk snapshots,
GPU/device allocations, and total device memory are separate categories.

Whole-topology admission (`Sampling.memory_bytes`), host-pressure checks, and
observed client-tier eligibility answer different questions. A small observed
client does not establish admission under an equally small host budget. The
historical acceptance rule rejects observed swap growth; zero growth is not zero
swap use. Pressure-rejected or failed attempts supply no eligible RSS sample.

All four provider roles retain their own cost owner. Available historical CPU
accounting ends at the first response; provider RSS is sampled in the request
window. Neither supplies startup-through-cleanup provider CPU, lifetime peaks, or
a simultaneous aggregate provider peak. Request latency excludes provider startup;
target latency includes ordinary startup and cleanup, not source provisioning.
Checkpoint/tokenizer distribution, public indexing/certification, private files,
and source-cold provisioning cannot be inferred from request measurements.

Each additional boundary worker receives its own 100/40 Mbps access allowance in
the historical per-party profile. Consequently the four-provider topology has
twice the summed provider access allowance of the two-provider control. Normalizing
link rates alone does not equalize CPU, RSS, or one shared uplink. Hard aggregate
provider CPU/RSS enforcement and complete provider lifetime accounting remain
missing for this composition; the local latency result is not evidence of an
algorithmic gain at equal aggregate resources.

## C.3 Memory-reduction claim boundary

The urgent client-memory research target is tenfold lower matched whole-client
peak RSS without changing the body, tokens, numerical contract, private workload,
or outsourcing requirement. The separate Qwen3-4B 64+8 starting estimate is
8.25 GiB, giving an approximately 0.83 GiB working estimate budget
(`docs/research/low-client-network.md`). Neither number is measured RSS; that
budget is also distinct from the 256 MiB Low tier. Repricing existing last-use
release, reducing a mask inventory, or removing a vocabulary payload cannot close
this target. No matched isolated Qwen3-4B 64+8 complete-client peak pair is supplied
here. A tenfold claim must use unrounded matched peak observations, not the rounded
0.83 GiB estimate ceiling. The Qwen2.5 results below cannot serve as its baseline
or candidate.

# Appendix D: experiment and reproduction details

## D.1 Protocol and rejected attempts

The retained experiments use the ordinary `Experiment`, `BenchmarkTarget.response`,
and `BenchmarkSuite` interfaces, not a separate paper execution engine.
`benchmarks/research/private_boundary_decode.py::recipes` changes only explicit
head dispatch/codec choices for the boundary ablations.
`benchmarks/research/private_boundary_decode_confirmation.py::matched_suite`
selects maintained prepared, serial/u32, and parallel/uniform-residue arms with
three fresh-process replicates, balanced serial order, a thread setting of one, no warmups,
and one cold request per process. Each cohort fixes its installed build, source,
tokenizer, input fixture, output cap, numerical body fingerprint, and network
conditions. The 8 GiB suite allowance is whole-topology admission, not a measured
client peak or enforced provider pool.

The state audit charges its work to the client and compares complete logits and
active KV at selection boundaries, positions, selections, RNG, and output
commitments. The retained trajectories use temperature zero and sampler seed 42;
cryptographic randomness remains fresh. These records do not by themselves check
every intermediate tensor, executed-history/pending-token field, or stochastic
trajectory. The semantic argument and conformance tests have their own scope.
Exactness to this W8A8 reference is separate from independent BF16/FP32 fidelity
and held-out task quality; neither quality qualification nor adoption of another
method's numeric profile follows from component conformance.

Traffic reconciliation checks directed stream counters and the enclosed
application-body ledger, including preparation and boundary calls. TCP streams do
not include physical-wire overhead; shutdown traffic after the final counter
snapshot is not automatically included. Request time, online first text, N−1
decode time, and target time retain their distinct event boundaries. An output cap
is not forced-length execution: EOS may shorten a request, and equal-work ranking
requires the same stopping trajectory. The positive-output auditor charges unused
prepared material; zero-output telemetry remains unsupported.

Within a cohort, a common private salt permits equality checks without publishing
private prompts. Different cohort salts, builds, or workloads prohibit direct
commitment comparison and pooled rankings. Three repeated executions of one prompt
measure variability, not prompt diversity. The initial five-choice confirmation
was rejected after one client recorded 80,084,992 bytes of host swap growth; its
completed samples are not reused in the accepted three-choice cohort
(`docs/research/private-boundary-decode-followup.md`). Missing, timed-out, and
rejected cells remain distinct from measured zero cost.

## D.2 Retained report identities

The three raw reports are
`docs/evidence/private-boundary-decode-confirmation-150x8.json`,
`...150x32.json`, and `...150x64.json`, with their corresponding `.plan.json`
definitions. `paper/artifacts/historical-results-v1.json` retains each cohort's
source/build/configuration digests, metric contracts, and replicate observations;
`paper/HISTORICAL-RESULTS.md` identifies the original report hashes. Ordinary
experiment definitions and benchmark outputs are the reusable evidence. Historical
table reconstruction checks recorded values, not newly executed tensors;
independent reproduction uses the experiments and ordinary correctness tests,
not a custom archive-verification gate. No cohort supplies missing observations
for another cohort.

# References
