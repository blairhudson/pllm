# Arithmetic garbling

Mixed-modulus labels and projection gates over bounded arithmetic values.

[View canonical HTML](https://pllm.run/sdk/pipeline/protocols/garbling/arithmetic/)

Document ID: `pllm.docs.protocols.garbling.arithmetic`  
Release: `0.1.0`  
Build: `sha256:bf56c232413fe9b57bc2befe009368956690afaf0d708914c7620c3b7d5d9531`  
Source hash: `sha256:d5feabb35d2218d2c5ea845ba12a9d49a53d30da7887d8feabfb1b80c5490f7c`

Arithmetic garbling represents values with labels per modulus. Free compatible arithmetic can avoid tables; nonlinear projection reconstructs signed values jointly from a coprime residue bundle and emits output labels. Gate material is shape-bound, strictly serialized, and consumed once.

Reference exhaustive correctness, tamper rejection, plan commitments, and one-use process-local burn checks establish implementation behavior only. The compiler can select the exact experimental Q7 SiLU projection profile. Cryptographic review, durable replay protection, complete-model coverage, and deployment assurance are not recorded.

## Python SDK example

```python
from pllm.components import list_components

available = [item.component for item in list_components() if "garbl" in item.component]
print(available)
```

The arithmetic-garbling evaluator remains internal to the native compiled-plan boundary; it is not part of the stable Python runtime facade.
