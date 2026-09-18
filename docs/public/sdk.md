# Python SDK

Plan models, compose research methods, and use supported PLLM runtimes from Python.

[View canonical HTML](https://pllm.run/sdk/)

Document ID: `pllm.docs.sdk`  
Release: `0.1.0`  
Build: `sha256:8db228446618c1e95120afa32d874de6f8cba2e6e356c6b0c1157f0c5b7cc58e`  
Source hash: `sha256:606373a08d7caf7bf31c9e8e64913fe55cb0c817ce7a035b9acf459cd2af28cd`

PLLM ships one Python package backed by Rust. The public API separates model
planning from live runtime state so you can inspect and reproduce a plan without
including credentials, masks, sessions, or model weights.

## Install the package

Add the `pllm` package from PyPI to a uv project:

```bash
uv add pllm
```

## Python SDK example

`pllm.lower_model` accepts model configuration and workload bounds. It returns
an immutable semantic `ModelPlan`; it does not load a checkpoint or start an
inference session.

```python
from pathlib import Path

from pllm import lower_model

fixture = Path("crates/pllm-models/tests/fixtures/mini-coder-4b-c87892d-config.json")
plan = lower_model(fixture.read_bytes(), batch=1, max_input_tokens=16, max_new_tokens=4)
assert plan.to_dict()["adapter"] == "pllm.qwen3.v1"
assert plan.prefill["output"] == "token_feedback"
```

API: [Python SDK objects and signatures](/sdk/reference/python/pllm/#objects-and-signatures)

Run this example from a PLLM source checkout; package installations do not include
the model-adapter test fixture.

## Compose research methods

Apply a compatible component to a model plan to create a new plan with a record
of the transformation. Compatibility with a semantic plan does not by itself
establish compiler, runtime, privacy, quality, or deployment support.

## Run supported workflows

Runtime APIs cover specific public-weight and confidential-weight workflows.
Support depends on the model, protocol, operator set, deployment, and threat
model. Check [current support](/sdk/reference/status/) before choosing a path.

## Continue

- [Python API reference](/sdk/reference/python/pllm/) lists the installed public
objects.
- [Get started](/learn/start/) covers installation and the first inspection
workflow.
- [OpenAI Python SDK](/learn/integrations/openai-python/) connects the official
client to the trusted local gateway.
- [Runtime](/sdk/pipeline/runtime/) explains live state and execution boundaries.
- [Models](/sdk/build/models/) separates semantic planning from executable support.
