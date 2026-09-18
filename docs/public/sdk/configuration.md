# Configuration

Load, inspect, and export a reproducible PLLM experiment configuration.

[View canonical HTML](https://pllm.run/sdk/configuration/)

Document ID: `pllm.docs.sdk.configuration`  
Release: `0.1.0`  
Build: `sha256:a919906eb7ec6dc3da86474b11b2e5fee671ce20bf3e0b394186df1a41084acd`  
Source hash: `sha256:f2109427fb5a3b204a7fd15b0173c686175aca157c0596687ef5d36b7935a8c8`

PLLM configuration objects describe public intent. Creating one does not download
a model, contact a service, compile a plan, or create private material.

Inspect the included example with
[`pllm config show`](/cli/reference/config/show/):

```bash
uv run pllm config show examples/pllm.yaml
```

Export the normalized configuration as JSON with
[`pllm config export`](/cli/reference/config/export/):

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
assert model.to_spec() == {"source": "Qwen/Qwen2.5-0.5B-Instruct"}
```

API: [Python SDK objects and signatures](/sdk/reference/python/pllm/#objects-and-signatures)
