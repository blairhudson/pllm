# Schema reference

Versioned schemas for configurations, plans, evidence, assurance, and research records.

[View canonical HTML](https://pllm.run/sdk/reference/schemas/)

Document ID: `pllm.docs.reference.schemas`  
Release: `0.1.0`  
Build: `sha256:d7e297a96b58353cf8b221cf4bcd8c22a00010928161fd1a366d85134a3e5756`  
Source hash: `sha256:e1cf3d369c6b800eb009362e779c998232e2058bf986b5044f6a25e40cd8d7d7`

Schemas under root `schemas/` define serialized document contracts. Rust and Python validators must agree on identity, required fields, bounds, unknown-field policy, and canonical digest construction.

Configuration, semantic decoder plans, compile requests, executable plans, component descriptors, benchmark evidence, assurance evidence, research sources, methods, recipes, and publication assessment evolve independently. A schema-valid document can still be unsupported by a selected compiler or runtime profile.

## Python SDK example

```python
from pllm import ComponentRef

document = ComponentRef("pllm/cpu", {"threads": 4}).to_spec()
assert document == {"component": "pllm/cpu", "params": {"threads": 4}}
```

Serialization produces public configuration data; schema validity alone does not establish support.

API: [`pllm.ComponentRef`](/sdk/reference/python/pllm/#objects-and-signatures)
