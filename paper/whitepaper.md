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
  Use outside compute. Keep sensitive context local. PLLM is an open-source runtime and research platform for testing how far private inference can go—and what it costs.
---

## Use outside compute. Keep sensitive context local.

Hosted AI usually asks us to trust a provider with our conversations and documents.
Running a model locally changes that bargain, but demands local memory and compute.
Private inference offers another possibility: let remote services calculate without
giving any one of them the sensitive context.

**PLLM makes that possibility testable.** A Python SDK and local API gateway connect
applications to a native runtime. Researchers compose protocols, numerical methods,
storage and placement choices into reproducible experiments.

## A deliberate division of work

The client keeps plaintext, nonlinear calculations, model state and output decoding.
Before a response, Preparation computes fresh masking corrections. During the
response, Inference performs matrix products on masked inputs. The client removes
the output masks. Preparation stays idle online.

> Prepare offline. Calculate on masked inputs. Reconstruct only at the client.

These services must follow the protocol and **not collude**. Preparation must erase
its masks. Self-hosting it keeps that trust within the client boundary. Public model
weights, timing and traffic patterns are not hidden. Co-located benchmark processes
do not demonstrate independent operators.

## The question is practical

Privacy alone does not make a system useful. Every improvement must answer three
questions: how much data moves, how much total computation runs, and how much memory
each party needs. Model behavior must survive the change.

\newpage

## Progress is a scorecard, not one number.

PLLM's research loop is simple: implement a method independently, check its contracts
and model behavior, compare matched configurations, and retain the evidence.
Each score links to its Python configuration so the benchmark can be rerun.

## One model. Three configurations.

In a local Qwen2.5-0.5B W8A8 experiment, each configuration processed the same
150-token input and generated eight tokens. All three output digests matched.

| Configuration | Covered bodies |
|:---|---:|
| Prepared baseline | 603.45 MB |
| Exact prepared stack | 476.95 MB |
| Stack + 96 public prefix tokens | 266.91 MB |

The exact stack combines packed inputs, smaller output residues, pruned rows,
demand-aware preparation, paged weights, compressed delivery and an indexed
tokenizer. The final configuration imports trusted state for an explicitly public prefix.

**Less traffic is not automatically faster.** The public-prefix configuration used
55.8% fewer covered application-body bytes, but request latency rose from 11.04 to
11.72 seconds. Cold aggregate process CPU fell from 37.11 to 27.06 seconds. Its public
artifacts add 12.25 MB; their distribution and the publisher's 8.03 CPU seconds are
outside those response measurements.

## Keep the comparison honest

This is one cold response per configuration, on one host with cached checkpoints.
The byte count covers recorded application bodies, not full physical wire or model
downloads. The public-prefix result applies only when that prefix is available.
It does not measure private-prefix discovery, Internet latency or representative
generation quality. Client peak memory needs a fresh-process comparison.

SOTA means the best **measured composition for a particular metric and workload**.
Baseline wins latency; another configuration wins traffic. Offline preparation still
costs compute. Larger-model and independent-provider studies remain open.

[Research scorecard](https://pllm.run/research/papers/) ·
[Technical paper](https://pllm.run/research/paper/) ·
[SDK](https://pllm.run/sdk/)
