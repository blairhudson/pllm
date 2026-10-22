---
title: "PLLM: Private inference, measured"
description: "A short introduction to PLLM, its privacy boundary, and the measured trade-offs of remote inference."
author: "Blair Hudson"
affiliation: "Independent Researcher, Australia"
date: "October 2026"
edition: "01 / OCTOBER 2026"
pdf: "whitepaper.pdf"
source: "whitepaper-source.zip"
abstract: |
  Use remote computation. Keep private context local. PLLM is an open-source runtime and research platform for experiments with private inference and its costs.
---

## Use remote computation. Keep private context local.

Hosted AI usually gives a provider access to our conversations and documents.
Local inference keeps the model at the client but uses local memory and computation.
Private inference lets remote services calculate while the client's inputs stay
private from each service under specified assumptions.

**PLLM supplies a framework for these experiments.** A Python software development
kit (SDK) and local application programming interface (API) gateway connect applications to the native runtime.
Researchers make reproducible experiments with protocols, numerical methods,
storage, and computation locations.

## A division of work

The client keeps plaintext, nonlinear calculations, model state, and output decoding.
Before a response, Preparation calculates fresh mask corrections. During the
response, Inference calculates matrix products on masked inputs. The client removes
the output masks. Preparation stays idle online.

> Supply corrections offline. Calculate on masked inputs. Reconstruct only at the client.

These services must obey the protocol. **Their private protocol information must stay separate.**
Logical one-use retirement does not guarantee secure erasure. Client-side
Preparation keeps that trust within the client boundary. Providers can see public model weights, timing, and traffic
patterns. Processes on one host give no evidence of independent operators.

## The question is practical

Privacy alone does not make a system useful. A useful comparison includes three measures:
data transfer, total computation, and memory for each party.
The specified model behavior must stay the same.

\newpage

## Progress has several measures

Researchers independently write each method's software, do checks of contracts and
model behavior, compare matched configurations, and keep the evidence.
Each score has a Python configuration link to do the benchmark again.

## Initial evidence. Three configurations.

In the initial local Qwen2.5-0.5B W8A8 experiment, each configuration processed the same
150-token input and gave eight output tokens. All three output digests agreed.
W8A8 uses eight-bit weights and activations.

| Configuration | Covered bodies |
|:---|---:|
| Prepared baseline | 603.45 MB |
| Exact prepared stack | 476.95 MB |
| Stack + 96 public prefix tokens | 266.91 MB |

The exact stack uses packed inputs, smaller output residues, pruned rows,
demand-aware preparation, paged weights, compressed delivery, and an indexed
tokenizer together. The last configuration uses trusted state for an explicitly public prefix.

**Less traffic does not always give higher speed.** The public-prefix configuration used
55.8% fewer covered application-body bytes, but request latency increased from 11.04 to
11.72 seconds. Cold aggregate process CPU decreased from 37.11 to 27.06 seconds. Its public
artifacts add 12.25 MB. Their distribution and the publisher's 8.03 CPU seconds are
outside those response measurements.

## Keep the comparison honest

This is one cold response for each configuration, on one host with cached checkpoints.
The byte count includes recorded application bodies, not full physical-wire traffic or model
downloads. The public-prefix result applies only when that prefix is available.
It does not measure private-prefix discovery, Internet latency, or representative
generation quality. Comparisons of peak client memory with new processes are still necessary.

PLLM uses state of the art (SOTA) for the best **measured eligible composition for a specified metric and workload**.
This is not a field-wide claim.

In this initial cohort, the baseline has the lowest latency, but another configuration has the lowest traffic.
The live scorecard adds measured CPU/Metal and client-placement combinations.
Offline preparation still uses computation. Studies with independent providers are still necessary.

[Research scorecard](https://pllm.run/research/papers/) ·
[Technical paper](https://pllm.run/research/paper/) ·
[SDK](https://pllm.run/sdk/)
