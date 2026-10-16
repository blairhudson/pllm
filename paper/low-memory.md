---
title: "Communication-Efficient Private LLM Inference under a 256 MiB Client Budget"
description: "This draft examines private lookup and full-head execution together. More evidence is necessary for the proposed contribution and comparisons."
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
  Private inference can move transformer computation to providers, but the client can have large costs at the model boundary. Token lookup selects an embedding, the numerical vector for an input token. The output head calculates one score, or logit, for each vocabulary token. A large vocabulary matrix can be necessary for these operations at the client. A cold client's public bundle cache is empty at startup. The matrix can increase local storage and, in that case, data delivery.

  We examine a coupled design in which two boundary workers do the two operations with specified non-collusion assumptions. With these assumptions, private lookup supplies the requested embedding without disclosure of the token to either worker. The privacy argument is conditional, untimed, and static. The prepared decoder body keeps the same protocol. The client keeps its state, complete logits, and sampler. The client budget is 256 MiB for the complete client.

  Private lookup and linear delegation with shares are prior art. With the specified assumptions, the shares together encode an input without disclosure of that input to either worker. The possible new contribution is exact execution with bounded resource use and specified ownership of material during its full lifetime. Exact execution keeps the reference decoder's numerical and state rules.

  A previous nine-client Qwen2.5 confirmation uses 150 input tokens and eight output tokens with local 100/40 Mbps access emulation. In this source-present case, the workers have checkpoint and tokenizer sources before the request. The cold-client case uses new processes and new private material. The client's public bundle cache is empty. Public artifacts at the client before the request have separate costs. Compared with its maintained prepared control, this confirmation records 23.70% fewer all-link TCP-stream bytes and 24.18% shorter requests.

  The candidate's largest complete-client peak is 224.13 MiB. But decode is 32.68% slower, and online first-text time is 8.00% longer. These savings apply to source-present cold clients. They are not sufficient evidence for a general speed or memory improvement.

  [P2-A/P2-D/P2-E: Do the qualification checks for complete-system competitors available at publication. Complete this paper's campaign and an independent confirmation before a frontier or SOTA claim.]
---

*This manuscript is a draft. Novelty, review of the concrete primitives, and the
publication comparison are open items. The previous results below are not a new
SOTA claim.*

# 1 Introduction

A client can use providers for most transformer computation and have large
inference costs. Vocabulary lookup, output prediction, tokenization, decoder
state, and temporary buffers are necessary somewhere in the system. The client
has limits on memory and access bandwidth. These costs can prevent operation at
or below those limits.

A cold client uses new processes and new private material. Its public bundle
cache is empty at startup. The costs above can also change the cost of a cold
request.

The PLLM framework gives the same measurement basis for specified numerical,
security, and resource contracts [@hudson2026pllmframework]. The numerical
contract gives the rules for arithmetic, scales, rounding, and operation order.
The client budget is the maximum permitted simultaneous total memory for the client and all
its helpers. This limit is for their full lifetime. The Low tier sets this budget at 256 MiB.

The Low-tier limit does not identify one best design. Eligible configurations have
different costs for communication, latency, client computation, and provider
resources.

The framework manuscript is a draft. Its publication identifier and fixed
benchmark version are open items.

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

The client receives the embeddings and reconstructs each logit. It keeps the same
sampler to select the next token from these logits.

The client keeps its decoder state, which includes the key-value cache (KV cache).
This cache holds the attention keys and values for tokens from previous decoder
steps. The prepared transformer body keeps the same protocol. The objective is
lower communication for a complete request at or below the Low-tier memory limit.

Private lookup and shared linear delegation are prior art. In shared linear
delegation, workers calculate products from shares that together encode a private
input. With the specified privacy assumptions, either worker's share alone does not
disclose the input. Gupta, Katz and Miers use private embedding lookup,
public-weight delegation, and client-local nonlinear work together
[@gupta2026outsourcing].

Our possible contribution keeps the compiled decoder's exact numerical and state
contract. It connects tied-weight ownership, bounded lifecycles, and resource
measurements for the complete client. We examine where this composition decreases
communication and where repeated boundary work increases costs. Less vocabulary
storage alone is not sufficient evidence for a lower peak in process memory.

**[P2-A: Do the qualification checks for the closest complete systems. Examine
whether the implementation contribution and measured frontier improvement give
sufficient evidence for a method paper.]**

# 2 Problem and comparison class

## 2.1 Objective

The primary objective is lower all-link communication for a complete request.
All-link communication counts traffic in the two directions on each link. The
design must keep the fixed quantized decoder and obey the specified privacy
contract. It must not use more client memory than the Low-tier budget. The
secondary objective is complete-request latency.

Quantization uses a smaller set of numbers to approximate selected model values.

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

## 2.2 Parties and trust

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

Section 3.4 gives a conditional theorem for the untimed abstract protocol with
static providers. Its assumptions include a trusted client/setup, confidential
authenticated channels, fresh independent randomness, and the specified primitive
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

## 2.3 Prior work and novelty gate

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

# 3 Coupled boundary execution

    Offline Preparation ---- corrections ----> Inference
             ^                                  ^
             |                                  |
             +------------- Client -------------+
                           /      \
                  Boundary A      Boundary B
                 private lookup and shared full head

**Figure 1.** Five-role execution. The boundary pair is additional to the prepared
body. Boundary A and Boundary B are B0 and B1. Each remote role has specified costs
for custody and resources.

## 3.1 Admission and private lookup

Before a session, the client and workers bind the source, model configuration,
schedule, quantized weights/scales, public workload, and codec choices. One weight
owner in each boundary worker holds the immutable tied matrix. Lookup and head
evaluation use that same owner. The client bundle contains no vocabulary matrix.

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

## 3.2 Decoder body and full head

The client keeps tokenization, activation quantization and scales, nonlinear work,
attention, the full KV cache, complete logits, and sampling. Preparation supplies
fresh body corrections before online execution. Inference does the same
masked stages.

For a quantized hidden vector $h$ and public head $W$, the client samples fresh
uniform $a$ in the admitted ring. It sets $b=h-a$. These vectors are additive
shares of $h$. Each worker sends the product for its share. Their sum in ring32
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

## 3.3 Delivery and lifetime

Workers can do the work for independent head requests at the same time.

A residue is the remainder after integer division by the modulus. Exact residue
replies discard only bits that have no effect in the public output
ring. The input shares stay ring32. Uniform geometry bounds and optional
source-certified per-row bounds have different admission and publisher contracts.
Separate ablations must measure their effects before a claim of improvement.
The head includes each vocabulary row in the logits.

One-use contexts bind roles, stages, and operation counts. Replay, malformed
replies, partial admission, cancellation, and expiry cause retirement of owned
material. The client waits for all concurrent calls to complete before retirement.

The last selected token stays pending until the decoder completes a subsequent
step with that token as input. This boundary composition rejects previous-response
state reuse. Thus, pending state is not evidence for an installed continuation API.

On failure, the output that the client received is only a prefix. The equivalence
claim for a successful complete response does not apply to an unfinished attempt.

## 3.4 Correctness and conditional privacy

For signed i8 head weights and quantized inputs in $[-127,127]$, row $j$ has the
public bound $B_j=127\sum_i|W_{ji}|$. Workers can send residues modulo $2^{b_j}$
if $B_j<2^{b_j-1}$. Reduction is a ring homomorphism. Thus, the sum of the two
residues and centered reconstruction give the reference integer. Either individual
share can wrap without change to this result.

The client then does the same separate float32 scale multiplications without
reassociation. Geometry-only widths and source-certified widths use different
public commitments.

Induction on the unchanged compiled operations gives decoder equivalence. The
lookup gives the same records. The prepared body reconstructs the same integer
products. Local attention, nonlinear operations, and KV updates receive the same
inputs. The head supplies all reference logits. With the same sampler RNG state,
the selections and the subsequent sampler state agree.

Cryptographic randomness stays fresh and independent of sampler seeding.

The argument applies only to the fixed quantized decoder. It does not prove
equivalence to BF16/FP32 or give a new result for task quality.

The conditional privacy proof uses hybrid experiments and a causal simulator.
A hybrid is an intermediate experiment in the proof. A causal simulator supplies
messages in protocol order, before it knows subsequent private inputs.

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
The 12 bounded tests include 57,600 algebra cases, each coalition in the small
test model, and leakage/retirement counterexamples. They do not prove the
cryptographic assumptions.

**[P2-B: Complete the independent review of the concrete DPF and implementation
side channels.]**

# 4 Implementation and resource ownership

The SDK selects this composition with `PrivateLookupAndFullHead` and
`PreparedBoundaryRoles`. Native contracts bind sources and bounded operations.
Python controls the remote calls and application state. Lookup and head
operations in each boundary worker use the same native weight owner. This owner
is immutable. Thus, a second full snapshot of the lookup table is not necessary
in either worker.

Measure the complete client. Include the SDK, supervisor, dashboard, rate relay,
audit, and all additional helpers.

Each of the four provider roles has a separate CPU/memory owner. Measure the full
period from startup through cleanup. Include setup and retirement costs in this
period.

Record delivery and storage costs for public artifacts, private files, checkpoint
distribution, and provider provisioning.

Equal per-party caps give more total access capacity to configurations with more
workers. Give the per-party limits for realistic deployments. Also give a
sensitivity comparison with normalized resources. Give this information before
a statement that the algorithm caused a latency improvement.

**[P2-C: Set exact bounds on live allocations. Set the workload admission and
resource contracts. Keep these bounds and contracts fixed. Keep measured peak RSS
separate from conservative admission limits.]**

# 5 Evaluation protocol

**[P2-D: Set this campaign before measurements and selection of the final
configuration. Keep the campaign fixed.]**

- **Baselines:** Include maintained prepared and the best eligible configurations
  for memory, traffic/placement, and latency/decode. Include two-offset where
  admitted, serial/u32 boundary, and the final coupled design. Keep verified and
  unverified classes separate. Include clear client-only execution as a reference.
  It uses no remote inference.
- **Workloads:** Start with varied held-out prompts with 150/288 input tokens and
  8/32/64 output tokens. Keep fixed-length and natural-EOS cells separate. Change the
  public-artifact state between cold and warm cases. Change source distribution as
  a separate factor. Make claims across models only after qualification of
  multiple checkpoints.
- **Correctness:** Compare each complete logit/KV state and the positions/history.
  Compare pending selections, output, and sampler witnesses. Include greedy and
  stochastic execution, with failure and cancellation controls. Keep independent
  model quality separate.
- **Measurements:** Use one fixed build for each matched comparison. Use fresh
  clients, serial execution, balanced order, and controlled host load. Measure
  traffic in each direction and use full CPU/memory measurement periods.
  Record variability, individual peaks, exclusions, and attempts that resource
  pressure causes the system to reject.
- **Ablations:** Compare coupled and separate weight ownership. Compare
  serial/parallel calls, u32/uniform residues, and uniform/certified residues.
  Compare client lifetimes, publication, and delivery choices.
- **Sensitivity:** Change output length, cache lifecycle, link/RTT, provider capacity,
  and client budget. Do not extrapolate a cold/warm crossover without measurements.

**[P2-D Figure 2/current-baseline table: complete-cost frontier against current eligible baselines.]**

**[P2-D Figure 3: ablations, crossover and provider-resource sensitivity.]**

**[P2-E: Complete an independent confirmation of the selected fixed configuration.
Do an audit of the raw evidence.]**

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
time is 8.00% longer.

This confirmation applies to those selected configurations for that build and
workload. It does not compare all previous control configurations again. It does
not measure source/tokenizer distribution or give Internet latency results. It does
not supply results for the new campaign.

The separate longer-output screens have their own evidence tier. The cited
archive links the full source for these measurements and the raw cohort paths.

The table below gives the same three separate cohorts as the source reports.
Each row contains results from three fresh clients. Request time, traffic, and
decode are medians. Client memory gives the largest peak during the complete
client's lifetime.

All cohorts use source-present cold requests. The local measurements use shared
native access pacing at 100/40 Mbps.

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

The full audit also keeps the serial/u32 controls. Compared with those controls,
parallel/residue decode throughput increases by 52.87%, 53.79%, and 52.89% for the
respective cohorts. But decode stays slower than maintained prepared
execution.

Online first-text time is 8.00% longer than prepared at eight outputs. At 64
outputs, aggregate cold-first CPU is 48.69 seconds for parallel/residue and
47.76 seconds for maintained prepared. This measurement period continues through
the first response and does not include cleanup. Give these tradeoffs with the
traffic result.

The source reports record 78,842,880, 336,605,184, and 729,833,472 compared logit/KV
values for the respective cohorts. Their recorded swap growth is zero. The
publication audit does new checks of recorded commitments and costs. It does not
do the cohort's tensor computation again.

The command `node scripts/audit_publication_evidence.mjs --check` gives the
table sources. `paper/artifacts/historical-results-v1.json` records
source/build/configuration digests and each replicate. No cohort supplies
missing observations for another cohort.

Memory eligibility changes near a threshold. The short confirmation's candidate
has a largest measured peak above 224 MiB but below 256 MiB. The largest measured
peak gives this eligibility result. Neither a lower median nor less vocabulary
storage changes this result.

# 6 Discussion and conclusion

The coupled design removes vocabulary-matrix delivery to the client. But it
adds repeated boundary traffic and computation. The lowest-cost design can change
with request length, cache lifecycle, provider resources, and the specified trust
contract.

Low-tier eligibility is not minimum memory use. Exact quantized execution is not
evidence for representative model quality.

**[P2-E: Give the primary result from the confirmation and the measured
regressions.]**

A comparison with the strongest eligible communication baseline is necessary
before a claim of lowest measured communication. A Pareto improvement alone
is not sufficient evidence for that minimum or a better result in each case.
Use "best among evaluated compatible configurations" until the external-competitor
review gives evidence for a more general statement.

**[P2-F: Complete the publication artifact and keep its version fixed. Replace the
draft Paper 1 citation with its assigned identifier. Complete an independent
replay of the full release.]**

# References
