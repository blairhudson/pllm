# Experiments

Design reproducible experiments with fixed plans, comparable runs, and recorded failures.

[View canonical HTML](https://pllm.run/research/recipes/experiments/)

Document ID: `pllm.docs.research.experiments`  
Release: `0.1.0`  
Build: `sha256:65f4ad316621284cc28b60b1a825a6d9c6d1f108cbb5032aee0983de1e5c4966`  
Source hash: `sha256:fb74cab7b8065fcba5070e81afb8cf2a9202786a299e63581e2a8cf9027ca5fe`

Before execution, an experiment records its hypothesis, baseline, changed
component, model and workload limits, plan digests, environment, metrics, quality
requirements, assurance work, stopping rule, and expected artifacts.

Report throughput, latency, memory, CPU, online and offline network traffic, disk,
preparation work, quality, and assurance separately. Keep failed and negative
results. A publication claim must isolate the contribution from its baseline and
identify the method unambiguously.

See [benchmarks](/sdk/research/benchmarks/), [metrics](/research/records/metrics/), and [publication assessment](/research/publications/).
