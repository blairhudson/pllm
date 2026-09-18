# Model plans

Lower a model configuration, apply a compiler component, and inspect execution coverage.

[View canonical HTML](https://pllm.run/sdk/plans/)

Document ID: `pllm.docs.sdk.plans`  
Release: `0.1.0`  
Build: `sha256:87cbf81718b764da3fb4871c21f12e5943efd717a5d5854e5a0cecf969808585`  
Source hash: `sha256:ca573de30b839639a16e1fa927ef801b9496c39566ef0db5af7143cca802bbe1`

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

`KvCacheEviction` is a deterministic structural compiler pass for dense Qwen2 and
Qwen3 plans. It preserves fixed-capacity Key/Value state across prefill and decode,
adds explicit static-selection index state, and rewires decode attention through
bounded per-query dynamic gathers. Gemma and incompatible cache topologies are
rejected before mutation.

Use `plan.coverage(profile)` to inspect the operators a compiler profile can and
cannot execute. A valid structural transform is not by itself evidence of protected
runtime execution, privacy, model quality, or deployment support.

See [models](/sdk/build/models/) for supported adapters and [compiler](/sdk/pipeline/compiler/)
for coverage rules.
