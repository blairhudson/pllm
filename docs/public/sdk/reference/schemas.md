# Schema reference

Versioned schemas for configurations, plans, evidence, assurance, and research records.

[View canonical HTML](https://pllm.run/sdk/reference/schemas/)

Document ID: `pllm.docs.reference.schemas`  
Release: `0.1.0`  
Build: `sha256:d19e46409d656eb3dd08fdadbb8ef6a9e8ca4c33fb48893359235fe855b4a7c4`  
Source hash: `sha256:d4fa3ab4d1294ce4ab3ed247051f481137bec7ffa7dfe7a11e0209a7e93d7cc5`

Schemas under root `schemas/` define serialized document contracts. Rust and Python validators must agree on identity, required fields, bounds, unknown-field policy, and canonical digest construction.

Configuration, semantic decoder plans, compile requests, executable plans, component descriptors, benchmark evidence, assurance evidence, research sources, methods, recipes, and publication assessment evolve independently. A schema-valid document can still be unsupported by a selected compiler or runtime profile.

`model.schema.json` covers the typed public model source shared by configuration and runtime requests. `model-source-lock.schema.json` covers path-independent checkpoint/config/tokenizer file hashes produced by the public model loader. `dense-qwen-runtime-schedule.schema.json` covers the complete batch-one, untransformed Qwen2 schedule emitted for `baseline.masked_linear_cpu`. `runtime-model-binding.schema.json` covers the model-plan, schedule, tokenizer, runtime-configuration, local-tensor, quantized-stage, scale, and preparation commitments checked before that schedule can execute. The schedule and binding schemas describe the model-aware baseline only; they do not authorize `research.single_evaluator`, MPCache execution, another model family, or protected client-local operators.

## Python SDK example

```python
from pllm import ComponentRef

document = ComponentRef("pllm/cpu", {"threads": 4}).to_spec()
assert document == {"component": "pllm/cpu", "params": {"threads": 4}}
```

Serialization produces public configuration data; schema validity alone does not establish support.

API: [`pllm.ComponentRef`](/sdk/reference/python/pllm/#objects-and-signatures)
