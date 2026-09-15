# Protocols

See how PLLM defines parties, messages, privacy assumptions, and failure behavior.

[View canonical HTML](https://pllm.run/sdk/pipeline/protocols/)

Document ID: `pllm.docs.protocols`  
Release: `0.1.0`  
Build: `sha256:1e873bcc571b7b7717ee66b62aff1ec3e361ead12088e6b707b77e88ee008328`  
Source hash: `sha256:e16f915fab73ebc12c737397f438d1e89b0debcd60351e2b3bba109ed04f2811`

A protocol defines its parties, offline and online phases, inputs, outputs,
messages, authentication, replay behavior, one-time material, visible information,
trust assumptions, and failure behavior. Model adapters and deployment policy are
separate concerns.

Current protocols include [masked-linear inference](/sdk/pipeline/protocols/masked-linear/)
and experimental [garbling components](/sdk/pipeline/protocols/garbling/). Check execution
support, security review, assurance, and deployment status separately.

## Python SDK example

```python
from pllm.components import list_components

protocols = [item for item in list_components() if item.category == "pllm/protocol-method"]
print([item.component for item in protocols])
```

This lists published protocol descriptors; discovery does not start a protocol.
