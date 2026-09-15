# Lookup tables

Bounded function tables with explicit domains, indexing, leakage, and material size.

[View canonical HTML](https://pllm.run/sdk/pipeline/protocols/garbling/lookup-tables/)

Document ID: `pllm.docs.protocols.garbling.lookup-tables`  
Release: `0.1.0`  
Build: `sha256:bf56c232413fe9b57bc2befe009368956690afaf0d708914c7620c3b7d5d9531`  
Source hash: `sha256:115c124569938979704946f952dbe4c4f1c4de3aa7d22fe0c9aa3710d96d1dc3`

A lookup-table component maps a finite encoded domain to an encoded range. Its contract records signed interpretation, table cardinality, invalid indices, material ownership, one-time use, communication, and whether selection is data-oblivious under the stated threat model.

Tables are useful for nonlinearities only when domain bounds and total storage remain explicit. Approximation and model-quality evidence belong to the numeric policy, not the table name.

## Python SDK example

```python
from pllm.components import list_components

available = [item.component for item in list_components() if "lookup" in item.component]
print(available)
```

No lookup-table component is currently published through the SDK.
