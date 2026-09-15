# Schema reference

Versioned schemas for configurations, plans, evidence, assurance, and research records.

[View canonical HTML](https://pllm.run/sdk/reference/schemas/)

Document ID: `pllm.docs.reference.schemas`  
Release: `0.1.0`  
Build: `sha256:bf56c232413fe9b57bc2befe009368956690afaf0d708914c7620c3b7d5d9531`  
Source hash: `sha256:f0cf48e2c79c86a92975d95626fd2a28425bc818020e69fcfe0707d7f6293c9e`

Schemas under root `schemas/` define serialized document contracts. Rust and Python validators must agree on identity, required fields, bounds, unknown-field policy, and canonical digest construction.

Configuration, semantic decoder plans, compile requests, executable plans, component descriptors, benchmark evidence, assurance evidence, research sources, methods, recipes, and publication assessment evolve independently. A schema-valid document can still be unsupported by a selected compiler or runtime profile.

## Python SDK example

```python
from pllm import ComponentRef

document = ComponentRef("pllm/cpu", {"threads": 4}).to_spec()
print(document)
```

Serialization produces public configuration data; schema validity alone does not establish support.
