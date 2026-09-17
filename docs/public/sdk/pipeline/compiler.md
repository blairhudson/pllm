# Compiler

Learn how PLLM turns a model plan and component choices into a checked execution plan.

[View canonical HTML](https://pllm.run/sdk/pipeline/compiler/)

Document ID: `pllm.docs.compiler`  
Release: `0.1.0`  
Build: `sha256:2b873610e88902ce44935954b3c8be5ba82c51e9673ea0929e4c8f08e5a4bf62`  
Source hash: `sha256:c0d3ee32135e041d2e3501df584823d60b898062ef00819484113f15f9084ffb`

Compilation combines a model plan, component choices, workload limits, and policy.
It checks operator support, value formats, conversions, numeric settings, party
placement, preparation, native kernels, and assurance requirements. The result is
an executable plan with a content digest.

[Numeric semantics and model quality](/learn/numeric-semantics-and-model-quality/)
explains why compiling every operator still does not establish generation fidelity.

The coverage report names every missing operator, conversion, party role, numeric
bound, assurance record, or compatible component version. Required gaps stop
compilation instead of triggering a hidden fallback. All model families use the
same semantic model format.

Current research profiles cannot execute a complete listed model plan. A model
adapter can therefore be available while full private inference remains unsupported.

This example requires a PLLM source checkout and must run from its repository root
because it reads the checked-in compiler fixture.

## Python SDK example

```python
from pathlib import Path

from pllm import compile

request = Path("schemas/fixtures/compile-request.valid.json").read_text(encoding="utf-8")
plan = compile(request)
assert plan.input_shape == (2, 3)
assert plan.output_shape == (2, 2)
```

API: [Python objects and signatures](/sdk/reference/python/pllm/#objects-and-signatures)

Compilation rejects unsupported coverage rather than hiding it behind a fallback. Complete model compilation remains unsupported for current research profiles.
