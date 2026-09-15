# Benchmarks

Measure PLLM systems with reproducible records and compare only equivalent runs.

[View canonical HTML](https://pllm.run/sdk/research/benchmarks/)

Document ID: `pllm.docs.benchmarks`  
Release: `0.1.0`  
Build: `sha256:1e873bcc571b7b7717ee66b62aff1ec3e361ead12088e6b707b77e88ee008328`  
Source hash: `sha256:369d7f9a2f9f69127d1d4a73a1c6fb0019d8f665ee6d02c3af644c92d2c61913`

`pllm.benchmark(...)` measures supported native compiled regions and returns an
immutable `EvidenceReport`. `pllm.deployment_benchmark(request)` validates
observations supplied by the caller, but cannot prove where they came from. The
CLI does not currently have a `pllm benchmark` command.

Every record identifies whether it covers a compiled region, protocol, deployment,
complete model, privacy test, or quality test. Compare runs only when the model,
body fingerprint, plan, workload, numeric settings, environment, threads, SIMD
settings, warmup, repetitions, and cold or warm mode match.

The local dashboard helps diagnose a loopback deployment, but it does not create a
canonical evidence record. See [metrics](/research/records/metrics/) and
[research experiments](/research/recipes/experiments/).

## Python SDK example

```python
from inspect import signature
from pllm import benchmark

print(signature(benchmark))
```

This shows required benchmark inputs without implying that an uncompiled model can run.
