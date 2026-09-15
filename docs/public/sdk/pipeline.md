# Pipeline

Combine model, privacy protocol, preparation, runtime, and deployment choices in one configuration.

[View canonical HTML](https://pllm.run/sdk/pipeline/)

Document ID: `pllm.docs.pipeline`  
Release: `0.1.0`  
Build: `sha256:bf56c232413fe9b57bc2befe009368956690afaf0d708914c7620c3b7d5d9531`  
Source hash: `sha256:72b22d4f092c06db38a7b4360e2634832cf9961a257398ceea409fea405f9ee7`

A `Pipeline` records what you intend to run. Profiles provide visible defaults,
and component overrides keep their versions. The configuration digest covers
normalized values, not live connections or provider state.

Creating a pipeline does not run anything. Model lowering, research transforms,
compilation, deployment, preparation, and session creation are separate steps.
Each step returns a record or a structured error.

Run `pllm config show examples/pllm.yaml` to inspect a configuration without
loading a model or runtime.

## Python SDK example

```python
from pllm import Cpu, MaskedLinear, Model, ModelAwareCorrections, Pipeline

pipeline = Pipeline.from_profile(
    "baseline.masked_linear_cpu",
    model=Model("Qwen/Qwen2.5-0.5B-Instruct"),
    components={"linear": MaskedLinear(), "preparation": ModelAwareCorrections(), "kernels": Cpu()},
)
print(pipeline.to_spec())
```

This creates public configuration only; it does not start a runtime.
