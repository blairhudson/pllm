# Research APIs

Use the Python SDK to benchmark plans, run scoped assurance checks, and produce immutable evidence records.

[View canonical HTML](https://pllm.run/sdk/research/)

Document ID: `pllm.docs.measure`  
Release: `0.1.0`  
Build: `sha256:4a02d6d6c164f106cb6698f1e14f6ed79d2e45e68a44719d9f0aa417993b4f8a`  
Source hash: `sha256:dd7e5035522877c46fd0b86b7fc5146db157fdfad877e1e26e12de3b67f143fa`

- [Benchmark](/sdk/research/benchmark/) documents `pllm.benchmark(...)` and
`pllm.deployment_benchmark(...)`.
- [Assure](/sdk/research/assure/) documents `pllm.assure()` and the limits of
its result.
- [Benchmark records](/sdk/research/benchmarks/) explain the immutable evidence returned
by the SDK.
- [Assurance records](/sdk/research/assurance/) explain what an assurance result can and
cannot support.
- [`pllm benchmark run`](/cli/reference/benchmark/run/) documents the separate
real-role loopback diagnostic.

For provenance and implementation boundaries, use
[method implementations](/research/methods/). For measurement scope and claim
limits, use [research evidence](/research/evidence/).

## Python SDK example

```python
import pllm

report = pllm.assure()
findings = {finding["id"]: finding for finding in report.to_dict()["findings"]}
assert report.schema_version == "pllm.assurance_report.v1"
assert findings["mask_reuse"]["outcome"] == "refuted_in_scope"
```

API: [`pllm.assure`](/sdk/reference/python/pllm/#objects-and-signatures)

This runs only the package's scoped deterministic assurance fixtures.
