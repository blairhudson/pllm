# Weighted path

Decision-path representations for bounded piecewise or tree-structured secure evaluation.

[View canonical HTML](https://pllm.run/sdk/pipeline/protocols/garbling/weighted-path/)

Document ID: `pllm.docs.protocols.garbling.weighted-path`  
Release: `0.1.0`  
Build: `sha256:1e873bcc571b7b7717ee66b62aff1ec3e361ead12088e6b707b77e88ee008328`  
Source hash: `sha256:cf1dc05adb69acfd73fab2a76605345ac6c34c7555c6dd5700f03d7f9dfac5e2`

Weighted paths encode a sequence of predicates and selected contributions rather than a dense table. A component must state path topology, hidden and public structure, comparison semantics, branching leakage, numeric bounds, and worst-case work.

This is an extension category, not a shipped support claim. Candidate research enters through source and method records before any compiler profile can select it.

## Python SDK example

```python
from pllm.components import list_components

available = [item.component for item in list_components() if "weighted" in item.component]
print(available)
```

No weighted-path component is currently published through the SDK.
