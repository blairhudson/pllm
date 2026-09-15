# Arithmetic garbling

Mixed-modulus labels and projection gates over bounded arithmetic values.

[View canonical HTML](https://pllm.run/sdk/pipeline/protocols/garbling/arithmetic/)

Document ID: `pllm.docs.protocols.garbling.arithmetic`  
Release: `0.1.0`  
Build: `sha256:1e873bcc571b7b7717ee66b62aff1ec3e361ead12088e6b707b77e88ee008328`  
Source hash: `sha256:484c57928397b7fb2aee9bf31cda903b494a4cff1310f3888240c8bf834c1007`

Arithmetic garbling represents values with labels per modulus. Free compatible arithmetic can avoid tables; nonlinear projection reconstructs signed values jointly from a coprime residue bundle and emits output labels. Gate material is shape-bound, strictly serialized, and consumed once.

Reference exhaustive correctness and tamper rejection establish implementation behavior only. Cryptographic review, compiler activation, complete-model coverage, and deployment assurance are not recorded.

## Python SDK example

```python
from pllm.components import list_components

available = [item.component for item in list_components() if "garbl" in item.component]
print(available)
```

No arithmetic-garbling component is currently published through the SDK.
