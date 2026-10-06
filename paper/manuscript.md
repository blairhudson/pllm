---
title: "PLLM: A Research Platform for Private LLM Inference"
description: "A common framework for implementing and comparing private language-model inference, with local studies of dispatch, preprocessing and paging."
author:
  - "Blair Hudson"
affiliation: "deployscience labs"
email: "blair@deployscience.com"
date: "6 October 2026"
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
  PLLM is an open-source framework for implementing and comparing private language-model inference methods. It provides a Python interface for composing model adapters, numerical representations, protocols, and execution placements, backed by Rust kernels. Local execution, two-worker additive sharing, and an offline-prepared protocol use the same decoder representation. The current runtime outsources linear operations on public weights to honest-but-curious, non-colluding services; nonlinear operations and decoder state remain at the trusted client. Experiments connect these choices to compilation, execution, and measurement through a common specification. We describe the interface and runtime, and present local case studies of dispatch scheduling, preprocessing cost, and weight paging. The studies illustrate how execution choices affect latency, aggregate CPU time, communication, and memory. PLLM is released under Apache-2.0 with source code, documentation, and experiment records.
---

*Working manuscript; evidence placeholders remain explicit.*

# 1 Introduction

Private inference allows a client to use remote computation while keeping its inputs private under a stated threat model. Secure multi-party computation, homomorphic encryption, and trusted execution provide different ways to construct such systems [1, 2, 3, 4]. For language-model decoders, protocol choices interact with numerical precision, the placement of computation, and preprocessing. Moving a matrix product offline, for example, can reduce online latency while leaving the total arithmetic unchanged.

We present PLLM, an open-source framework for developing and evaluating private inference on decoder workloads [5]. PLLM separates the model definition from how and where it executes. Model adapters produce a common decoder representation; components specify quantization, linear protocols, kernels, and placement. Local execution, two-worker additive sharing, and prepared execution use this representation, so researchers can compare execution choices without changing the decoder implementation.

The design has three requirements: reusable decoder implementations, replaceable execution components, and measurements that include preprocessing and all participating roles. The current distributed paths outsource public-weight linear operations, while the client performs nonlinear operations and retains decoder state. We describe the resulting interfaces and use archived local experiments to examine dispatch order, preprocessing cost, and weight paging.

# 2 Design and implementation

## 2.1 Experiment interface

An `Experiment` combines an immutable `Pipeline`, a `Deployment`, and an `ExecutionBudget`. The pipeline contains the model source and execution components; the deployment specifies role locations; and the budget bounds requests and token counts. A researcher can change a kernel or transport while retaining the workload. Listing 1 shows the Python interface [5].

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

## 2.2 Model representation and components

Model adapters translate decoder operators, layer identities, and persistent state into a shared intermediate representation (IR). Compiler passes operate on these semantic fields instead of model-specific node names. Qwen2 and dense Qwen3 are among the implemented adapters. Adapter coverage describes the model representation; checkpoint execution and numerical fidelity are evaluated separately.

Python handles experiment orchestration, source resolution, and role lifecycles. Rust implements integer kernels, scheduling, bounded codecs, and one-use material through a PyO3 interface. This division keeps experiment construction in Python while checking numerical and resource bounds at the native boundary. Components select quantization, linear protocols, kernels, caching, verification, and placement; the compiler determines which combinations are executable.

Client-side weight paging binds authenticated private snapshots to local token lookup and output-head execution without changing decoder semantics. The Preparation paging study in Section 4 is a separate probe rather than a selectable runtime placement.

**[E1: evidence to add]**

Add one implemented extension, its unchanged interfaces, conformance tests, ordinary benchmark invocation, and a rejected incompatible configuration.

# 3 Runtime and trust model

The trusted client retains plaintext inputs, activation scales, nonlinear operations, decoder state, and output selection. Token lookup and ordinarily the output head also run locally. An application gateway runs inside this boundary; providers hold public transformer-body weights.

The distributed protocols assume honest-but-curious services that follow the protocol and do not collude. Prepared execution additionally requires Preparation to erase mask material; Preparation may instead be hosted by the client. Co-located processes exercise the implementation but do not provide operator separation. Model weights, tensor dimensions, lengths, timing, and access patterns are outside the protection offered by these protocols. Transport authentication checks role identities; it does not establish non-collusion.

## 3.1 Two-worker execution

Let $W\in R^{m\times n}$ be a public integer matrix and $x\in R^n$ an encoded activation, where $R$ is a finite ring supported by the numerical configuration. The client forms fresh additive shares $a$ and $b=x-a$. Two workers each receive one share and return its product with $W$. The client reconstructs

$$
Wa+Wb=Wx.
$$

With uniform masks, either share is independent of $x$ in this algebraic model. Seed-based encodings additionally depend on their expansion assumptions. The workers perform two matrix products in total. This is twice the body-linear arithmetic of a clear product, rather than a prediction of total request CPU time.

## 3.2 Prepared execution

Prepared execution separates mask preparation from online evaluation. Fresh client seed material defines an input mask $r$, an output mask $s$, and a one-use ticket for each stage row. Preparation computes $c=Wr-s$ and transfers the correction to Inference before online work begins. The client then sends the ticket and $u=x-r$; Inference returns

$$
Wu+c=W(x-r)+Wr-s=Wx-s.
$$

The client adds $s$, reconstructs the bounded integer result, and performs scaling and nonlinear computation locally. Preparation is idle during the response. Reserved material becomes unusable after use, cancellation, replay, or failure, and subsequent requests require fresh material. Prefill batches rows; decode reuses an authenticated connection.

This arrangement removes one product from the online critical path while retaining two products across preparation and evaluation. Optional client-side Freivalds checks use fresh projections supplied by Preparation, inspired by verified delegation in Slalom [4]. PLLM’s placement does not use trusted execution hardware, and these checks do not protect against malicious Preparation.

# 4 Evaluation

## 4.1 Method

The benchmark runner compares candidates under a common workload and supports grid or seeded-random configuration search. Comparisons fix the checkpoint, encoded model body, prompt cohort, generated-token counts, cache state, hardware, and network conditions. Reports store configuration and source identifiers while omitting private prompts, token IDs, masks, and credentials. Salted prompt-cohort identifiers support matching within a benchmark invocation, not across independent invocations.

Measurements separate online execution, setup-inclusive request work, and cold-first process lifecycles. Aggregate CPU includes the client and all measured roles, including Preparation, over the stated interval. Communication counts cover application bodies rather than complete physical-wire traffic. Decode throughput divides the $N-1$ tokens after the first by their execution time. Peak resident set size (RSS) is measured per process and excludes filesystem cache; role peaks need not occur simultaneously.

The studies below use archived repository reports [5], identified in the artifact index. They have separate workloads and measurement scopes.

**[E2: evidence to add]**

Add repeated, matched runs with pinned hardware/software, thread limits, preprocessing/cache policy, and workload lengths. Report uncertainty and per-role latency, CPU, bytes, and peak memory.

## 4.2 Dispatch under emulated links

The dispatch study (W1) compares four Qwen2.5-0.5B-Instruct candidates using eight-bit weights and activations (W8A8), 39 input tokens, and eight generated tokens. Local link emulation gives each party a shared 100 Mbps downlink and 40 Mbps uplink. An added 20 ms egress delay at each endpoint produces 40 ms additional request–reply delay. All four candidates produce matching output digests.

Sending the independent worker request first raises offset decode throughput from 0.1027 to 0.1994 tokens/s ($1.94\times$), while online application-body bytes remain unchanged (Table 1). Overlapping prepared delivery and issuance raises request throughput by 17.3%, with little change in decode throughput. Scheduling reduces observed latency rather than bytes. Each configuration has one sample; latency distributions are unmeasured.

| Candidate | Request (s) | Decode (tok/s) | Body MB |
| :--- | ---: | ---: | ---: |
| Prepared control | 122.87 | 0.1976 | 71.193 |
| Prepared overlap | 104.73 | 0.1984 | 71.193 |
| Offset control | 155.01 | 0.1027 | 112.005 |
| Offset seed-first | 105.33 | 0.1994 | 112.005 |

**Table 1.** Local link-emulation results (W1); one sample per candidate. Request timing excludes provider startup and checkpoint distribution. Body MB denotes decimal megabytes of online application bodies.

## 4.3 Preprocessing and memory

A separate Qwen2.5 W8A8 study (C1) measures CPU from benchmark and role startup through generation of one token from 30 input tokens. Each placement has one cold-first response; source resolution before benchmark startup is excluded. Aggregate CPU is 11.71 s for clear client execution, 28.65 s for two workers, 34.66 s for prepared execution, and 142.83 s for verified prepared execution. Preparation alone accounts for 109.87 CPU seconds in the verified variant. Prepared execution therefore uses more aggregate CPU than the two-worker control in this workload, despite moving a product offline. Clear execution is a computational reference with no remote outsourcing.

The paging study (M1) isolates Qwen3-4B Preparation on an arm64 macOS host with 32 GiB RAM. It uses 144 stages, 71 rows per stage, a four-stage window, and two fresh processes per mode. Paging lowers median peak RSS from 4,613.77 to 739.09 MB ($6.24\times$). Correction contents match after excluding timing. Loading-plus-issuance CPU rises 12.41%, and each paged process adds 3.63 GB of private snapshots. The reduction applies to Preparation-process RSS in this probe, not client memory or full-response memory; filesystem cache is excluded.

## 4.4 Numerical fidelity

Numerical evaluation requires two comparisons: protected execution against the same unprotected integer computation, and integer computation against an independent model reference. Agreement among PLLM variants is an internal consistency check; a shared numerical error can remain in all variants.

Archived checks Q1 and Q2 compare local clear-kernel prefill with an independent FP32 implementation on two prompts per model. For each of Qwen2.5-0.5B and Qwen3-0.6B, top-token agreement is 1/2 with W8A8 and 0/2 with four-bit weights and activations (W4A4). These historical checks are separate from the performance studies and do not characterize generation quality.

**[E3: evidence to add]**

Add current protected/clear and independent-reference comparisons. Pin tokenization and sampling; report logit errors, prefill/decode agreement, and representative generation or task quality against declared tolerances.

# 5 Related work

CrypTen [1] exposes secure tensor computation, automatic differentiation, and modular neural networks through a machine-learning interface. PLLM instead focuses on decoder implementations, replaceable execution components, and per-role measurements, with nonlinear computation at the client. Cheetah [2] develops two-party neural-network protocols; THE-X [3] approximates Transformer operations for homomorphic evaluation; and Slalom [4] combines trusted hardware with private, verified delegation. Their threat models and workloads differ from the evaluated PLLM placements. This paper compares PLLM configurations rather than ranking these systems.

# 6 Availability and limitations

PLLM is alpha software distributed as `pllm.run` under Apache-2.0; its Python import and command are `pllm` [5]. This paper describes repository snapshot `54605c1bcc69`. Historical experiments may use earlier revisions; A1 tracks missing provenance.

The evaluated public-weight runtime keeps nonlinear computation client-local. The local studies do not evaluate independent-operator deployment, malicious-participant protection, complete wire traffic, or representative generation quality. Compiler checks and one-use material handling enforce implementation rules rather than a composable security proof.

**[A1: evidence to add]**

Freeze a release linking results to execution revisions, configurations, raw reports, and reproduction commands. Rerun the example and conformance tests; identify unresolved historical provenance.

# 7 Conclusion

PLLM provides a common implementation basis for comparing clear, two-worker, and prepared decoder execution. Local studies illustrate how dispatch ordering changes latency, preprocessing affects aggregate CPU, and paging exchanges resident memory for CPU and storage. Its shared model representation and experiment interface support investigation of these tradeoffs alongside numerical fidelity.

# Artifact index

The records below are stored in repository snapshot `54605c1bcc699d801bc3b62d9b02c17d69a56f0e` [5]. This identifies the reviewed archive, not the execution revision of every historical experiment. Missing execution metadata is tracked by placeholder A1.

| ID | Archived record | Scope of the reported evidence |
| :--- | :--- | :--- |
| W1 | [WAN dispatch comparison (4 October 2026)](https://github.com/blairhudson/pllm/blob/54605c1bcc699d801bc3b62d9b02c17d69a56f0e/docs/evidence/wan-tps-qwen25-2026-10-04.json) | Four single-sample candidates; local link emulation; matching output digests. |
| C1 | [Cold-first CPU comparison (26 September 2026)](https://github.com/blairhudson/pllm/blob/54605c1bcc699d801bc3b62d9b02c17d69a56f0e/docs/evidence/slalom-prepared-topologies-cold-cpu-2026-09-26.json) | One response per placement; benchmark and role startup included; earlier source resolution excluded. |
| M1 | [Qwen3-4B Preparation paging (5 October 2026)](https://github.com/blairhudson/pllm/blob/54605c1bcc699d801bc3b62d9b02c17d69a56f0e/docs/evidence/preparation-memory-qwen3-4b-2026-10-05.json) | Two fresh processes per mode; isolated probe, not full-response execution. |
| Q1 | [Qwen2.5-0.5B reference check (24 September 2026)](https://github.com/blairhudson/pllm/blob/54605c1bcc699d801bc3b62d9b02c17d69a56f0e/docs/evidence/qwen2.5-0.5b-reference-quality-2026-09-24.json) | Two prompts; local clear-kernel prefill against an FP32 reference. |
| Q2 | [Qwen3-0.6B reference check (24 September 2026)](https://github.com/blairhudson/pllm/blob/54605c1bcc699d801bc3b62d9b02c17d69a56f0e/docs/evidence/qwen3-0.6b-reference-quality-2026-09-24.json) | Separate model/source identity; same limited prefill-check scope as Q1. |

# References

[1] Brian Knott, Shobha Venkataraman, Awni Hannun, Shubho Sengupta, Mark Ibrahim, and Laurens van der Maaten. **CrypTen: Secure Multi-Party Computation Meets Machine Learning.** Advances in Neural Information Processing Systems 34, 2021. <https://arxiv.org/abs/2109.00984>.

[2] Zhicong Huang, Wen-jie Lu, Cheng Hong, and Jiansheng Ding. **Cheetah: Lean and Fast Secure Two-Party Deep Neural Network Inference.** 31st USENIX Security Symposium, pp. 809–826, 2022. <https://www.usenix.org/conference/usenixsecurity22/presentation/huang-zhicong>.

[3] Tianyu Chen, Hangbo Bao, Shaohan Huang, Li Dong, Binxing Jiao, Daxin Jiang, Haoyi Zhou, Jianxin Li, and Furu Wei. **THE-X: Privacy-Preserving Transformer Inference with Homomorphic Encryption.** Findings of the Association for Computational Linguistics: ACL 2022, pp. 3510–3520. <https://doi.org/10.18653/v1/2022.findings-acl.277>.

[4] Florian Tramèr and Dan Boneh. **Slalom: Fast, Verifiable and Private Execution of Neural Networks in Trusted Hardware.** International Conference on Learning Representations, 2019. <https://arxiv.org/abs/1806.03287>.

[5] Blair Hudson. **PLLM: Source code and research evidence.** 2026. Repository snapshot `54605c1bcc699d801bc3b62d9b02c17d69a56f0e`. <https://github.com/blairhudson/pllm>.
