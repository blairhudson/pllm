---
title: "PLLM: Private Multi-Party LLM Inference and Evidence-Bound Research Composition"
author: |
  Blair Hudson  
  deployscience labs  
  [blair@deployscience.com](mailto:blair@deployscience.com)
date: 23 September 2026
web-date: September 2026
edition: "08"
description: A prepared private-inference runtime, independent paper-derived components, and matched evidence for capability selection.
pdf: paper.pdf
subject: private language-model inference, multi-party systems, autonomous research harness, reproducible benchmarking, additive masking
web-note: Paper-derived methods are partial and independently implemented; historical baseline, tiny verification and current pinned four-topology diagnostics have separate scopes.
abstract: |
  PLLM is a private LLM multi-party inference runtime and extensible autonomous
  research harness. Its public-weight path uses trusted Client, non-colluding
  Preparation, and Inference: offline one-use $Wr-s$ corrections permit online
  masked linear work while Client retains plaintext, state, and decoding.
  Model-neutral compilation binds typed component composition to a plan; bounded
  search and matched evidence enable comparison of eligible candidates. We map
  six cited research directions to independently implemented *portions* of
  PLLM, not complete reproductions. Historical CPU-loopback measurements cover
  nine Qwen2.5-0.5B baseline responses. Optional verified linear work has a
  separate tiny control and a pinned, one-run four-topology diagnostic with
  incomplete cold compute/wire accounting. Garbling and cache components
  lack whole-model comparative measurements. Thus these data neither establish
  economic value nor rank PLLM against external state of the art. We define
  comparability and metric gates required to make such a claim.
bibliography: paper/references.bib
link-citations: true
reference-section-title: References
documentclass: article
classoption: [9pt, twocolumn]
papersize: letter
geometry: [margin=0.68in, columnsep=0.24in]
colorlinks: false
indent: true
---

## Introduction

Ordinary hosted inference gives one operator the user's plaintext prompt, token
identities, intermediate state, and generated output. Transport encryption
protects these values in transit but not from the endpoint performing inference.
Private neural-inference systems distribute computation using homomorphic
encryption, secret sharing, or secure two-party computation
[@huang2022cheetah; @chen2022thex; @hao2022iron].

PLLM combines a prepared masked-linear runtime with model-neutral compilation
and an extensible autonomous research harness. Its automation generates bounded
valid candidates and repeats experiments; researchers still implement methods
independently. We contribute (1) one-use corrections and explicit trust
boundaries, (2) canonical component-bound schedules with fail-closed coverage,
and (3) evidence rules and metrics for comparing eligible alternatives. This
systems report is **not** a security proof or a matched external SOTA benchmark.

## Prepared inference protocol

### Roles and application boundary

The **Client** is trusted. It owns plaintext input and output, tokenization,
one-time root seeds and masks, private activation scales, attention and other
nonlinear state, sampling, and decoding. For public runtime bundles it also
evaluates token lookup and the output head locally. The optional `pllm gateway`
runs on loopback inside this boundary. Applications submit ordinary Responses API
or Chat Completions API requests to that gateway, which invokes the private PLLM
client path. Preparation and Inference are protocol services, not
application-facing OpenAI-compatible endpoints.

**Preparation** holds the public body and offline seeds; it must erase masks
and not collude with **Inference**, which holds the same body and performs
online matrix work. Both roles must follow the protocol.

### Offline preparation and online use

For public quantized matrix $W$ and private integer activation $x$, the Client
sends Preparation a fresh root seed and row count for each remote stage.
Domain-separated expansion binds model, stage, quantization, arithmetic, and
inventory identity to fresh input mask $r$, output mask $s$, and ticket per
row. Preparation computes

$$c = Wr-s$$

and pushes $c$ to Inference. The latter acknowledges stage batches and seals
the inventory after every stage loads. It sees no root seed; Client sees no $c$.

Online, the Client sends Inference a one-use ticket and masked activation $x-r$.
Inference atomically consumes the matching correction and returns

$$W(x-r)+c=W(x-r)+(Wr-s)=Wx-s.$$

The Client adds $s$ and center-decodes $Wx$. Preparation receives no online
message. Fresh pseudorandom $r$ masks $x$ from Inference, while fresh $s$ masks
the correction. Reusing a row would reveal relations between activations, so
reuse is forbidden.

### Inventory lifecycle

Inventory commits to model/body, stage set, quantization, ring, and attempt
budget. Responses atomically reserve disjoint rows; cancellation, failure,
replay, and early completion burn unused reservations. Restart, replacement,
and idle expiry discard in-memory inventory.

Prefill packs stage tickets and masked rows; decode reuses a persistent
authenticated connection with fresh tickets. Each stage selects an exact
$2^{16}$, $2^{24}$, or $2^{32}$ ring from its signed output bound.

## Threat model and limits

Inference does not receive literal prompt text, token IDs, decoded output, root
seeds, masks, private activation scales, local attention state, or sampling
choices. Preparation receives seeds and can derive masks, but does not receive
online masked activations. Under fresh pseudorandom masks, authenticated channels,
protocol-conforming execution, Preparation erasure, and non-collusion, neither
remote role alone has both views needed to reconstruct client activations.

The split fails if Preparation and Inference collude: Preparation's $r$ and
Inference's $x-r$ reconstruct $x$. Two services controlled by one operator do not
establish meaningful non-collusion. Self-hosting Preparation keeps its trust in
the client environment, but co-locating all roles is only a functional topology,
not evidence of independent operators.

Both services see public model identity, tensor shapes, timing, traffic volume,
and approximate sequence length. Authentication does not prove correct
execution, independence, secure deletion, or side-channel resistance. Arbitrary
malicious participants are out of scope; erasure of Preparation's Python memory
is not established. The local gateway and host remain trusted.

## Implementation status

### Prepared runtime

The `pllm.run` Python distribution includes native Rust kernels. Python owns
checkpoint import, protocol, gateway, and state; Rust executes immutable,
validated integer matrices, bounded codecs, and CPU dispatch. Public online
execution uses prepared masks, **not** homomorphic encryption. Attention,
normalization, token boundaries, nonlinear state, and sampling remain with
Client. Historical prepared generation covers Qwen2.5-0.5B; it does not make
the client-local regions protected components.

### Semantic IR and compiler

Adapters lower Qwen2, dense Qwen3, and bounded Qwen3.5, Phi, and Gemma text
configurations into a model-neutral IR of operators, layers, and persistent
state. Compiler passes avoid adapter-specific node names. Semantic lowering
alone proves neither checkpoint import nor executable or private coverage.

For untransformed plans, baseline components schedule independent weighted
operators sharing an input and order local operators by dependency. Binding
checks tokenizer, runtime, local tensors, quantized stages, scales, and
preparation commitments. A plan-bound session enforces feedback and bounds.
The pinned Qwen2.5-0.5B checkpoint passes a **clear native-kernel**
prefill-to-decode check; historical masked-protocol measurements are separate.
Pinned Qwen3-0.6B passes the same clear compiled prefill-to-decode functionality
check; its W4A4 output fails a small FP32 next-token parity probe, so model
quality remains unestablished. Verified prepared compilation now runs with
one-use verifier inventory; transformed cache and protected whole-decoder
composition remain incomplete.

### Cited work versus runnable baseline

Table 1 maps **independent PLLM adaptations**, not original authors' code, to
the strongest evidence available. Historical baseline and current pinned
verified timings are distinct cohorts. A paper citation is neither
evidence of a whole-paper reimplementation nor a cross-method speedup.

| Source | PLLM part | Current measurement | Pinned Qwen? |
|:---|:---|:---|:---:|
| PLLM baseline | Prepared masked linear | Nine warm responses | Yes, older revision |
| DASH [@sander2024dash] | Q7 garbled SiLU reference | Error at most 0.02285 | No |
| ReDASH [@maurer2025redash] | Bounded Q7 rescale/arithmetic | Numeric tests only | No |
| CRT garbling [@ball2017garbling] | Q7 multiply gadget | 245,397 B scalar payload | No |
| Half-gates [@zahur2015halfgates] | Boolean circuit reference | Correctness tests only | No |
| Slalom [@tramer2019slalom] | Freivalds stage check | One-run four-topology diagnostic | Yes, current revision |
| MPCache [@zeng2025mpcache] | KV-eviction plan pass | Plan lineage only | No |

: Scope of independent research adaptations. The masked baseline is PLLM's
protocol, not a Slalom reproduction. Payload bytes are not response traffic
or latency; the verified diagnostic lacks cold compute and full-wire parity.

### Component extensions, search, and evidence

Research methods enter capability families instead of paper-specific runtime
branches. Canonical `Pipeline` identity uses serialized model and component
values; the compiler binds operator, numeric, state, method, and schedule
contracts to the plan and fails closed on gaps. Provider manifests are
discoverable without importing code, but native provider loading is not
exposed. Papers are attribution and specification, not executable dependencies.

`Experiment` binds composition, workload, and deployment. Grid and seeded
random search generate validated candidates. Evidence records actual results;
their identity includes plan and environment digests. Historical baseline
measurements predate those plan locks. The
[experiment workflow](https://pllm.run/research/recipes/experiments/)
documents the runnable SDK and CLI paths.

### Optional verified linear work

`FreivaldsVerify` adapts Slalom's delegated-linear verification
[@tramer2019slalom] to a **different**, TEE-free trusted-client/preparation
boundary. For secret uniform $a\in\mathbb F_{2^{32}-5}$, Client checks
$\langle a,y\rangle=\langle W^\mathsf{T}a,x\rangle$ before dequantization.
Signed bounds prevent field wrap; independent checks and process-wide attempt
accounting bound false acceptance. Preparation sends authenticated one-use
projections bound to inventory and stage; Inference sees no challenge.
Malicious Preparation and collusion remain out of scope. Verification does
**not** reduce online traffic. The compiled verified executor admits a session
only with matching one-use verifier inventory; bare callbacks fail closed.

## Evaluation and evidence boundaries

### Method

**Historical prepared runtime.** `docs/evidence/current-runtime-2026-09-11.json`
records source revision `277d19f`, CPython 3.13.15, and Qwen2.5-0.5B-Instruct
checkpoint revision `7ae5576` (full lock in the artifact) [@qwen25] on one
Apple M5 CPU host (32 GiB). Client, Preparation, and
Inference ran as separate loopback processes. One excluded warmup preceded
three warm runs at each exact input length. Output was capped at 16 tokens;
the 30-token cohort produced only nine. Only completed responses with settled
telemetry were included. These records **predate** the semantic compiler.

| In/out | TTFT, s | Full, s | Client MB |
|---:|---:|---:|---:|
| 30 / 9 | 0.978 | 4.674 | 63.33 |
| 63 / 16 | 1.374 | 5.347 | 126.29 |
| 255 / 16 | 5.158 | 13.059 | 432.71 |

: Historical cohort medians (three runs each); decimal MB is total Client
traffic. TTFT starts after inventory readiness; full time includes offline
preparation. Online medians were 2.618, 3.194, and 7.442 s; offline correction
pushes were 59.80, 72.88, and 252.18 MB, respectively.

All nine records report zero Preparation requests and protocol operations during
the online interval, and zero literal plaintext prompt or token-ID bytes in the
instrumented counters. These observations validate the measured implementation
path, not a general privacy theorem. Preparation remains real computation and
traffic despite occurring outside online timing.

**Verified tiny control.** Separate `docs/evidence/` records named
`slalom-freivalds-tiny-baseline.json` and
`slalom-freivalds-tiny-verified.json` share generated model fingerprint,
workload, environment, and one completed local response per
composition. The baseline took 0.666 s and verified path 0.627 s full
latency. With **one sample, no warmup, synthetic weights, and no real-Qwen
checkpoint**, this comparison shows the optional verification path works in a
matched tiny topology; it cannot estimate performance or prove security.
Neither record contains a PlanLock digest. The functional verifier tests cover
forged stage output, authenticated material import, one-use consumption, and
session-wide attempt accounting; they do not establish malicious Preparation
resistance.

**Compiled Qwen2.5 diagnostic.** A separate [pinned, four-topology W8A8
one-response cohort](https://github.com/blairhudson/pllm/blob/main/docs/evidence/slalom-prepared-topologies-2026-09-26.json)
matched body fingerprint and 30+1 tokens: verified Preparation exchanged the
same 47.89 MB covered online bodies as ordinary Preparation, against 95.81 MB
for two online offset workers. Verification added 92.81 MB covered initial
material. Full offline CPU and wire bytes remain unmeasured, so this does not
establish the aggregate-compute cap, generation parity, or independent-party
privacy.

**Limits.** No measured concurrent load, physical network, accelerator,
model-quality study, secure-erasure study, independent operator deployment, or
cross-system matched cohort exists. Offline preparation incurs substantial
traffic and full latency. The historical results cannot be extrapolated to
operating cost or provider-level security.

## Defining an eligible state-of-the-art comparison

PLLM's metric definitions cover latency, throughput, communication, memory,
energy, accuracy, perplexity, and cost with explicit units and optimization
directions. A defined metric is **not a measured value**. The retained Qwen
study reports latency and protocol traffic, but no price, energy, quality, or
external-system result. Search validates candidate configurations; benchmark
records preserve exact cohorts, and Pareto comparison uses declared metric
directions without implicitly ranking incomparable evidence.

We define an eligible comparison as the *same* checkpoint/body, numeric policy,
input and output counts, output cap, warm state, hardware/network environment,
quality threshold, and compatible privacy and assurance contract. Candidates
must first pass correctness and trust gates. Within that cohort, a candidate
dominates another only on **measured** metrics with explicit directions and at
least one strict gain. Absent measurements cannot be imputed; a cheaper
response must count client work, offline rows used or burned, online compute,
network, and deployment resources. This defines how a future external SOTA
claim could be tested, not a claim that one currently exists.

## Bounded research components

Inspired by arithmetic garbling [@sander2024dash; @maurer2025redash], a
one-use Q7 SiLU region computes $q(x)=x/2+x^2/4$ on $[-1,1]$ with
ties-to-even rounding. Exhaustive encoded-domain testing bounds absolute
SiLU error by 0.02285. Evaluation is limited to 128 elements and burns
digest-bound material once.

A separate region jointly garbles SiLU and Q7 multiplication without decoding
intermediate labels. Selectable methods retain a dense binary-table baseline
and add compact mixed-modulus multiplication based on CRT projection gadgets
[@ball2017garbling]. Half-gates supply a separate experimental Boolean
reference [@zahur2015halfgates]. One-lane evaluator payloads measure
2,387,674 and 245,397 bytes, respectively; four independent compact lanes
measure 980,573 bytes. These primitives have no whole-decoder tensor-scale
execution, cryptographic review, or production-security claim and do not
replace the prepared additive-masking protocol.

## Conclusion

PLLM combines prepared private inference with independently implemented,
composition-bound research components. Its historical Qwen baseline, tiny
verified control, pinned four-topology diagnostic and isolated numeric/garbling
tests are **different evidence levels**.
Protected whole-decoder execution, matched method measurements on the pinned
model, and external SOTA and economic comparisons remain open.
