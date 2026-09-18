# Build

Define a reproducible PLLM system from configuration, models, components, and research methods.

[View canonical HTML](https://pllm.run/sdk/build/)

Document ID: `pllm.docs.build`  
Release: `0.1.0`  
Build: `sha256:87cbf81718b764da3fb4871c21f12e5943efd717a5d5854e5a0cecf969808585`  
Source hash: `sha256:2ec7b240cfeedc5313d59b889fce38bc1698ca68a0597207203e763335f1c860`

- [SDK configuration](/sdk/configuration/) records what you intend to build.
- [Models](/sdk/build/model-adapters/) translate model-family settings into shared operations.
- [Model plans](/sdk/plans/) separate model structure from compiler and runtime support.
- [Components](/sdk/components/) select versioned capabilities.
- [Research methods](/research/recipes/build-method/) record provenance and plan changes.

## Python SDK example

```python
from pllm import Cpu, Model, Pipeline

pipeline = Pipeline.from_profile(
    "public-weight-local",
    model=Model("Qwen/Qwen2.5-0.5B-Instruct"),
    components={"kernel": Cpu(threads=2)},
)
assert pipeline.to_spec()["components"]["kernel"]["params"]["threads"] == 2
```

API: [Python SDK objects and signatures](/sdk/reference/python/pllm/#objects-and-signatures)
