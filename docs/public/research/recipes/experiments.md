# Experiments

Design reproducible experiments with fixed plans, comparable runs, and recorded failures.

[View canonical HTML](https://pllm.run/research/recipes/experiments/)

Document ID: `pllm.docs.research.experiments`  
Release: `0.1.0`  
Build: `sha256:96b6d9446e37113d9d2892113cefbcf72b64f5f3fc8eeb331b7caddd36ad60a0`  
Source hash: `sha256:fb74cab7b8065fcba5070e81afb8cf2a9202786a299e63581e2a8cf9027ca5fe`

Before execution, an experiment records its hypothesis, baseline, changed
component, model and workload limits, plan digests, environment, metrics, quality
requirements, assurance work, stopping rule, and expected artifacts.

Report throughput, latency, memory, CPU, online and offline network traffic, disk,
preparation work, quality, and assurance separately. Keep failed and negative
results. A publication claim must isolate the contribution from its baseline and
identify the method unambiguously.

See [benchmarks](/sdk/research/benchmarks/), [metrics](/research/records/metrics/), and [publication assessment](/research/publications/).
