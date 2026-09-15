# Preparation

Prepare one-time masked material before a private inference request starts.

[View canonical HTML](https://pllm.run/sdk/pipeline/preparation/)

Document ID: `pllm.docs.preparation`  
Release: `0.1.0`  
Build: `sha256:bf56c232413fe9b57bc2befe009368956690afaf0d708914c7620c3b7d5d9531`  
Source hash: `sha256:64ab1c2348710022c2ae1c8d5d7da7d30d50e3420ada65221646745e750deb89`

Preparation components define how one-time material is created, batched, committed,
transferred, confirmed, erased, expired, and invalidated. They also identify who
owns the material and how much can be stored. Protocol and kernel choices remain
separate.

For public masked-linear inference, the client sends seed batches to preparation.
Preparation computes corrections, and inference stores them in a sealed inventory.
The client refills inventory only while idle. A restart or idle timeout discards
inventory held in memory.

Prepared material is part of the system cost. Benchmarks must state whether
preparation and loading are included in cold and warm measurements.

## Python SDK example

```python
from pllm.preparation import ModelAwareCorrections

descriptor = ModelAwareCorrections.describe()
print(descriptor.lifecycle_phase, descriptor.role_eligibility)
```

This inspects preparation metadata; it does not create one-time material.
