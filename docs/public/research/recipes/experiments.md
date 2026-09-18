# Experiments

Design reproducible experiments with fixed plans, comparable runs, and recorded failures.

[View canonical HTML](https://pllm.run/research/recipes/experiments/)

Document ID: `pllm.docs.research.experiments`  
Release: `0.1.0`  
Build: `sha256:4a02d6d6c164f106cb6698f1e14f6ed79d2e45e68a44719d9f0aa417993b4f8a`  
Source hash: `sha256:fb74cab7b8065fcba5070e81afb8cf2a9202786a299e63581e2a8cf9027ca5fe`

Before execution, an experiment records its hypothesis, baseline, changed
component, model and workload limits, plan digests, environment, metrics, quality
requirements, assurance work, stopping rule, and expected artifacts.

Report throughput, latency, memory, CPU, online and offline network traffic, disk,
preparation work, quality, and assurance separately. Keep failed and negative
results. A publication claim must isolate the contribution from its baseline and
identify the method unambiguously.

See [benchmarks](/sdk/research/benchmarks/), [metrics](/research/records/metrics/), and [publication assessment](/research/publications/).
