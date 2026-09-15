# Research APIs

Use the Python SDK to benchmark plans, run scoped assurance checks, and produce immutable evidence records.

[View canonical HTML](https://pllm.run/sdk/research/)

Document ID: `pllm.docs.measure`  
Release: `0.1.0`  
Build: `sha256:bf56c232413fe9b57bc2befe009368956690afaf0d708914c7620c3b7d5d9531`  
Source hash: `sha256:63b44326188d0ac42bf56905b6d40393d344955453f8f8742bcc224306fb2532`

- [Benchmark](/sdk/research/benchmark/) documents `pllm.benchmark(...)` and
`pllm.deployment_benchmark(...)`.
- [Assure](/sdk/research/assure/) documents `pllm.assure()` and the limits of
its result.
- [Benchmark records](/sdk/research/benchmarks/) explain the immutable evidence returned
by the SDK.
- [Assurance records](/sdk/research/assurance/) explain what an assurance result can and
cannot support.

For study design, source reproduction, and comparison rules, use the
[Research workflows](/research/recipes/) section instead.

## Python SDK example

```python
from pllm import assure

report = assure()
print(report.schema_version)
```

This runs only the package's scoped deterministic assurance fixtures.
