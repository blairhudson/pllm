# Model plans

Lower a model configuration, add components, and inspect compiler coverage.

[View canonical HTML](https://pllm.run/sdk/plans/)

Document ID: `pllm.docs.sdk.plans`  
Release: `0.1.0`  
Build: `sha256:50bdd8d40f8b1f362adbabe08e845d37acfffd39aacc3351d7706456b26c4b1a`  
Source hash: `sha256:77c0bb7e4708ea66eaf0a632bf7df99881f11c9dc4e09d349978b38befd5d39d`

`pllm.lower_model(...)` turns a supported model configuration and workload limits
into a `ModelPlan`. Lowering records model operations and state; it does not load
weights or make the plan executable.

## Python SDK example

```python
from pathlib import Path

from pllm import KvCacheEviction, lower_model

fixture = Path("crates/pllm-models/tests/fixtures/mini-coder-4b-c87892d-config.json")
base = lower_model(fixture.read_bytes(), batch=1, max_input_tokens=16, max_new_tokens=4)
optimized = base.apply(KvCacheEviction())
assert optimized.digest != base.digest
assert optimized.to_dict()["transformations"][-1]["component"] == "pllm/kv-cache-eviction"
```

API: [Python SDK objects and signatures](/sdk/reference/python/pllm/#objects-and-signatures)

Run this example from a PLLM source checkout; package installations do not include
the model-adapter test fixture.

Use `plan.apply(component)` to create a new plan with a research or runtime
component. Use `plan.coverage(profile)` to inspect the operators a compiler profile
can and cannot execute. A complete semantic plan is not evidence of runtime,
privacy, model quality, or deployment support.

See [models](/sdk/build/models/) for supported adapters and [compiler](/sdk/pipeline/compiler/)
for coverage rules.
