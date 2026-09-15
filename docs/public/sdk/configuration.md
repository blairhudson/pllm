# Configuration

Load, inspect, and export a reproducible PLLM experiment configuration.

[View canonical HTML](https://pllm.run/sdk/configuration/)

Document ID: `pllm.docs.sdk.configuration`  
Release: `0.1.0`  
Build: `sha256:bf56c232413fe9b57bc2befe009368956690afaf0d708914c7620c3b7d5d9531`  
Source hash: `sha256:fb66e983414478a59dd0064a2384ca08d8c222c259d007e0a4566db76ab12dc3`

PLLM configuration objects describe public intent. Creating one does not download
a model, contact a service, compile a plan, or create private material.

Inspect the included example:

```bash
uv run pllm config show examples/pllm.yaml
```

Export the normalized configuration as JSON:

```bash
uv run pllm config export examples/pllm.yaml --output /tmp/pllm-normalized.json
```

The exported file does not include credentials, live service handles, seeds,
masks, or prepared material.

Python users can construct the same `Experiment`, `Pipeline`, `Model`,
`Deployment`, and `ExecutionBudget` objects directly. See the generated
[Python API inventory](/sdk/reference/python/pllm/) for exact signatures.

## Python SDK example

```python
from pllm import Model

model = Model("Qwen/Qwen2.5-0.5B-Instruct")
print(model.to_spec())
```
