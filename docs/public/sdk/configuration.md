# Configuration

Load, inspect, and export a reproducible PLLM experiment configuration.

[View canonical HTML](https://pllm.run/sdk/configuration/)

Document ID: `pllm.docs.sdk.configuration`  
Release: `0.1.0`

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
from pllm import Model, TinyModel

model = Model("Qwen/Qwen2.5-0.5B-Instruct")
assert model.to_spec() == {"source": "Qwen/Qwen2.5-0.5B-Instruct"}

pinned = Model.hf(
    "Qwen/Qwen2.5-0.5B-Instruct",
    revision="7ae557604adf67be50417f59c2c2f167def9a775",
)
assert pinned.kind == "huggingface"
assert pinned.revision is not None
assert isinstance(Model.tiny(), TinyModel)
```

API: [Python SDK objects and signatures](/sdk/reference/python/pllm/#objects-and-signatures)
