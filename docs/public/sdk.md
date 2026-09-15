# Python SDK

Plan models, compose research methods, and use supported PLLM runtimes from Python.

[View canonical HTML](https://pllm.run/sdk/)

Document ID: `pllm.docs.sdk`  
Release: `0.1.0`  
Build: `sha256:1e873bcc571b7b7717ee66b62aff1ec3e361ead12088e6b707b77e88ee008328`  
Source hash: `sha256:0e6d16dcf1a31cb99eebdb984d9542920f6a153c82303db248898e4b2a9a4298`

PLLM ships one Python package backed by Rust. The public API separates model
planning from live runtime state so you can inspect and reproduce a plan without
including credentials, masks, sessions, or model weights.

## Install the package

Add the `pllm` package from PyPI to a uv project:

```bash
uv add pllm
```

## Plan a model

`pllm.lower_model` accepts model configuration and workload bounds. It returns
an immutable semantic `ModelPlan`; it does not load a checkpoint or start an
inference session.

```python
import json
from pathlib import Path

from pllm import lower_model

config = json.loads(Path("model-config.json").read_text(encoding="utf-8"))
plan = lower_model(config, batch=1, max_input_tokens=128, max_new_tokens=32)
print(plan.digest)
```

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
- [Runtime](/sdk/pipeline/runtime/) explains live state and execution boundaries.
- [Models](/sdk/build/models/) separates semantic planning from executable support.
