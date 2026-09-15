# Half-gates

Boolean AND-gate representation and the obligations surrounding free-XOR circuit execution.

[View canonical HTML](https://pllm.run/sdk/pipeline/protocols/garbling/half-gates/)

Document ID: `pllm.docs.protocols.garbling.half-gates`  
Release: `0.1.0`  
Build: `sha256:bf56c232413fe9b57bc2befe009368956690afaf0d708914c7620c3b7d5d9531`  
Source hash: `sha256:4c90b36f508d5031ac8ee3aaf937841a25a2a676da015f8f9e53719e338582da`

Half-gates reduce encrypted table material for Boolean AND gates under compatible label and hash assumptions. A component contract must bind circuit identity, wire labels, correlation assumptions, evaluator material, one-time use, serialization, and output decoding.

PLLM documents this category for composition and research mapping. Presence in the taxonomy does not state that a reviewed native implementation or executable profile exists.

## Python SDK example

```python
from pllm.components import list_components

available = [item.component for item in list_components() if "garbl" in item.component]
print(available)
```

No half-gate component is currently published through the SDK.
