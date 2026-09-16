# Compare

Compare runs with matching inputs and report every relevant cost and uncertainty.

[View canonical HTML](https://pllm.run/research/recipes/compare/)

Document ID: `pllm.docs.measure.compare`  
Release: `0.1.0`  
Build: `sha256:dab1bcb88380ca9a4c2b79e3de085391f482bb97d841a9f0741979579c87dbfb`  
Source hash: `sha256:9bfb025e7fb48f4e7dd4f6da1ccce96a43aa6b0d2c6f2ce2c6df76b354b11198`

Matched comparison freezes model/tokenizer, workload, numeric semantics, quality acceptance, privacy contract, roles/topology, hardware, software, preparation/freshness, cache, warmups, repetitions, failure accounting, and metrics.

Report TPS, latency, peak memory, CPU, network, disk, preparation, quality, and privacy as vector. Different cohorts remain separate. A Pareto improvement is no worse on every comparable objective and strictly better on at least one under predeclared uncertainty treatment.

Current generic comparison CLI: unavailable. Blocker: complete evidence-producing benchmark/search/compare facade and immutable cohort contract are not exposed by CLI.
