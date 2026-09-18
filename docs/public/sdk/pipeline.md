# Pipeline

Combine model, privacy protocol, preparation, runtime, and deployment choices in one configuration.

[View canonical HTML](https://pllm.run/sdk/pipeline/)

Document ID: `pllm.docs.pipeline`  
Release: `0.1.0`  
Build: `sha256:96b6d9446e37113d9d2892113cefbcf72b64f5f3fc8eeb331b7caddd36ad60a0`  
Source hash: `sha256:0aa653548a4251764aa395f5171ec48d98d5aea590cde612107eef0b8a99d83a`

A `Pipeline` records what you intend to run. Profiles provide visible defaults,
and component overrides keep their versions. The configuration digest covers
normalized values, not live connections or provider state.

Creating a pipeline does not run anything. Model lowering, research transforms,
compilation, deployment, preparation, and session creation are separate steps.
Each step returns a record or a structured error. The
[parties and offline work](/learn/parties-and-offline-work/) overview explains why
preparation and live inference remain distinct.

Run [`pllm config show`](/cli/reference/config/show/) to inspect
`examples/pllm.yaml` without loading a model or runtime.

## Python SDK example

```python
from pllm import Cpu, MaskedLinear, Model, ModelAwareCorrections, Pipeline

pipeline = Pipeline.from_profile(
    "baseline.masked_linear_cpu",
    model=Model("Qwen/Qwen2.5-0.5B-Instruct"),
    components={"linear": MaskedLinear(), "preparation": ModelAwareCorrections(), "kernels": Cpu()},
)
spec = pipeline.to_spec()
assert spec["components"]["linear"] == {"component": "pllm/masked-linear", "params": {}}
assert spec["components"]["kernels"]["params"]["threads"] == 1
```

API: [Python objects and signatures](/sdk/reference/python/pllm/#objects-and-signatures)

This creates public configuration only; it does not start a runtime.
