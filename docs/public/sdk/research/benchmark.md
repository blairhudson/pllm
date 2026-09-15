# Benchmark

Run scoped Python benchmarks and understand what the development dashboard does not measure.

[View canonical HTML](https://pllm.run/sdk/research/benchmark/)

Document ID: `pllm.docs.measure.benchmark`  
Release: `0.1.0`  
Build: `sha256:bf56c232413fe9b57bc2befe009368956690afaf0d708914c7620c3b7d5d9531`  
Source hash: `sha256:d9e866861ec739e8124d211dfa4c32cdef110cab8fd5df439054c92792200cb6`

`pllm.benchmark(...)` measures supported native compiled regions and returns immutable `EvidenceReport`. It requires explicit plan, immutable weights/input bytes, IDs, privacy/numeric cohorts, and environment. Warmups, repetitions, threads, SIMD, failures, and oracle comparison remain report data.

`pllm.deployment_benchmark(request)` validates supplied observations; it does not prove their origin. No parser-visible `pllm benchmark` command exists.

Developer visualization is available only as:

```bash
pllm dev dashboard --model Qwen/Qwen2.5-0.5B-Instruct --no-open
```

The dashboard binds to loopback, starts the current development roles, and stores
sanitized local history. It is a development tool and does not create a benchmark
evidence record. Do not cite dashboard results unless you record them separately
under the required comparison contract.

## Python SDK example

```python
from inspect import signature
from pllm import benchmark

print(signature(benchmark))
```

`benchmark` requires a plan returned by `pllm.compile` plus exact immutable inputs; this inspection does not run a benchmark.
