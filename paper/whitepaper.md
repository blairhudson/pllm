---
title: "PLLM: Private Inference and Reproducible LLM Experimentation"
author: |
  Blair Hudson  
  deployscience labs  
  [blair@deployscience.com](mailto:blair@deployscience.com)
date: 23 September 2026
web-date: September 2026
edition: "07"
description: A short introduction to PLLM's prepared private-inference runtime and composable research system.
pdf: whitepaper.pdf
subject: private LLM inference, multi-party systems, reproducible experimentation
documentclass: article
classoption: [10pt, twocolumn]
papersize: letter
geometry: [margin=0.72in, columnsep=0.28in]
colorlinks: false
indent: false
---

## Why PLLM

Ordinary hosted inference protects requests in transit but gives the endpoint
access to plaintext prompts and model state. Private-inference research offers
alternatives; its results often use different models, hardware, trust assumptions,
and metrics. Those results cannot be ranked by latency alone.

PLLM combines a working, **public-weight private-inference path** with an
experimentation system for testing compatible research methods on comparable
workloads. An ordinary provider endpoint does not become private by changing
its URL: the computation and the trust boundary must change.

## A private-inference path

The current public-weight path separates three roles:

- **Client** keeps plaintext prompts, tokenization, nonlinear and attention
  state, sampling, and output decoding. A local Responses and Chat Completions
  gateway can serve existing applications inside this trusted boundary.
- **Preparation** receives one-use seeds before generation, computes masked
  corrections for the public model's matrix stages, and uploads them to Inference.
- **Inference** holds the public transformer body and evaluates masked integer
  activations online. It never receives the corresponding plaintext activation.

Client reserves each prepared row once; unused rows burn on cancellation or
failure. Preparation is idle during online generation. This is an explicit
trust arrangement: Preparation must follow the protocol, erase masks, and **not
collude** with Inference. A single operator controlling both services could
reconstruct activations. Running all roles on one machine tests functionality,
not independent deployment.

## A way to test research

PLLM treats papers as provenance, not executable dependencies. Researchers can
reimplement a method behind a typed component contract, combine compatible
choices into an immutable experiment, search a bounded set of valid candidates,
and benchmark them. Static provider manifests allow discovery without loading
provider code; extension is subject to the same capability and execution checks.

Evidence is useful only within a matched cohort: model and checkpoint, workload,
warm state, environment, numeric policy, trust assumptions, and metric meaning
must be comparable. Search proposes candidates; it does not certify their
privacy, output quality, or security. PLLM has **not** demonstrated a
state-of-the-art cross-system comparison.

## Status and next step

The prepared runtime, local gateway, one-use inventory, and loopback benchmark
driver work today. A pinned Qwen2.5-0.5B checkpoint has historical three-role
runtime measurements and a separate clear compiled-kernel functionality test.
An optional trusted-client Freivalds check has matched **tiny-model** functional
evidence, not a real-checkpoint performance study.

Semantic adapters cover additional model configurations, and experimental
garbling and cache transforms exist as components. They are **not** a complete
protected decoder or proof that those checkpoints run privately end to end.
The nine retained Qwen2.5 measurements come from one Apple M5 CPU loopback host;
they establish neither WAN/GPU performance nor production non-collusion.

Next: complete protected whole-model execution, measure matched alternatives
at realistic scale, and review security and quality claims against that evidence.
The [technical paper](https://pllm.run/research/paper/) records the protocol,
exact cohorts, implementation status, and limits.
