---
title: "PLLM: Prepared Private LLM Inference and Composition-Bound Experimentation"
author: |
  Blair Hudson  
  deployscience labs  
  [blair@deployscience.com](mailto:blair@deployscience.com)
date: 23 September 2026
web-date: September 2026
edition: "07"
description: Prepared public-weight inference, component-bound compilation, and scoped reproducible evidence.
pdf: paper.pdf
subject: private language-model inference, multi-party computation, reproducible experimentation, semantic compilation, additive masking
web-note: Implemented prepared inference, compiler coverage, and scoped historical and tiny-model evidence; not a cross-system performance comparison.
abstract: |
  PLLM implements public-weight private inference with a trusted Client,
  non-colluding Preparation, and Inference. Preparation uploads one-use
  $Wr-s$ corrections offline; online stage data comprise tickets, masked
  activations $x-r$, and masked outputs $Wx-s$. The Client retains plaintext,
  state, and decoding. A model-neutral semantic IR and canonical component
  composition bind the untransformed Qwen2 and synthetic dense-Qwen3 baseline
  schedules; protected whole-decoder execution remains incomplete. An optional
  trusted-client Freivalds check verifies prepared linear results before
  dequantization. Independent Q7 garbling methods remain bounded experiments.
  We report nine retained warm Qwen2.5-0.5B CPU-loopback runs from an earlier
  revision and a separate matched tiny-model verified/unverified functional
  cohort. These do not establish current compiler performance, WAN behavior,
  operator independence, or superiority to other systems.
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
and a typed experimentation system. We contribute (1) one-use offline
corrections with explicit trust and inventory boundaries, (2) canonical
component-bound model schedules that reject missing capabilities, and (3)
cohort-scoped evidence separating functional tests from historical performance
measurements. This is a systems report, **not** a new security proof or a
matched state-of-the-art benchmark. A semantic plan does not imply executable
private inference.

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

Adapters lower Qwen2, dense Qwen3, exact Qwen3.5-4B text, Phi-4-mini, and
Gemma 4 E2B/E4B text configurations into a model-neutral decoder IR with
explicit operators, layers, and persistent state. Compiler passes use this
vocabulary instead of parsing adapter-specific names. Semantic lowering alone
establishes neither checkpoint import nor executable, quality, or privacy
coverage.

For untransformed plans, baseline components schedule independent weighted
operators sharing one input and order remaining local operators by dependency.
Binding checks plan, tokenizer, runtime configuration, local tensors, quantized
stage bytes, scales, and preparation commitments. A plan-bound session enforces
token feedback, bounds, stage shapes, and failure poisoning. The pinned
Qwen2.5-0.5B checkpoint passes a **clear native-kernel** prefill-to-decode
check; historical masked-protocol evidence is separate. Tiny Qwen2 and dense
Qwen3 use the same compiled binding. Gemma 4 fails closed on unsupported local
operators. Verified compiled execution, transformed cache execution, Qwen3.5,
Phi, and protected whole-decoder composition remain incomplete.

### Component extensions, search, and evidence

Research methods enter capability families rather than paper-specific runtime
branches. Canonical `Pipeline` identity depends on serialized model and
component values, not a preset class or display label. The compiler binds
operator, numeric, state, method, and scheduling contracts to the resulting
plan; incomplete combinations fail closed. Provider manifests are discovered
without importing code; native provider loading is not exposed. Research
papers remain specification inputs, not executable dependencies. The
`KvCacheEviction` pass records lineage but its transformed plan does not run
as a complete decoder [@zeng2025mpcache].

`Experiment` binds composition, workload, and deployment. Grid and seeded
random search generate validated candidates. Latency ranks only inside exact
model/body, token-count, output-cap, and warm-state cohorts. Metric semantics
and privacy assumptions must also agree for cross-system comparison. Historical
measurements predate plan-lock evidence; no matched external comparison exists.
The [experiment workflow](https://pllm.run/research/recipes/experiments/)
documents the SDK and CLI routes for repeating supported comparisons.

### Optional verified linear work

`FreivaldsVerify` adapts Slalom's delegated-linear verification
[@tramer2019slalom] to a **different**, TEE-free trusted-client/preparation
boundary. For secret uniform $a\in\mathbb F_{2^{32}-5}$, Client checks
$\langle a,y\rangle=\langle W^\mathsf{T}a,x\rangle$ before dequantization.
Signed bounds prevent field wrap; independent checks and process-wide attempt
accounting bound false acceptance. Preparation sends authenticated one-use
projections bound to inventory and stage; Inference sees no challenge.
Malicious Preparation and collusion remain out of scope. Verification does
**not** reduce online traffic. Compiled verified execution fails closed until
bound to a verifier-enforcing executor.

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

: Historical cohort medians (three runs each); MB is decimal. TTFT starts
after inventory readiness; full time includes offline preparation and
transition. Online medians were 2.618, 3.194, and 7.442 s; offline correction
pushes to Inference were 59.80, 72.88, and 252.18 MB, respectively.

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

**Limits.** No measured concurrent load, physical network, accelerator,
model-quality study, secure-erasure study, independent operator deployment, or
cross-system matched cohort exists. Offline preparation incurs substantial
traffic and full latency. These results cannot be extrapolated to other models,
current compiled execution, operating cost, or provider-level security.

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

PLLM supplies prepared public-weight private inference, composition-bound
semantic compilation, and a framework for reproducible comparison. Its
three-role runtime and optional trusted-client linear check have different
evidence cohorts. Protected whole-decoder execution, verified compiled
execution, and matched external performance comparisons remain open. Any
stronger performance or security claim requires new revision-bound evidence.
