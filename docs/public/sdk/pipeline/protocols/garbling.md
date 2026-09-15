# Garbling

Reference arithmetic and Boolean garbling components with explicit maturity and composition boundaries.

[View canonical HTML](https://pllm.run/sdk/pipeline/protocols/garbling/)

Document ID: `pllm.docs.protocols.garbling`  
Release: `0.1.0`  
Build: `sha256:1e873bcc571b7b7717ee66b62aff1ec3e361ead12088e6b707b77e88ee008328`  
Source hash: `sha256:7720307af32555b1bd2f51e2fdc987a5841b768261cf3a4a9d29733d168f4931`

Garbling is a family of representations and protocols, not one interchangeable backend. PLLM records arithmetic projection gates, Boolean half-gates, lookup tables, and weighted paths separately because their domains, costs, proofs, and conversion obligations differ.

Current clean-room arithmetic work is reference-only and unreviewed. It is excluded from executable compiler profiles until protocol, transport, coverage, and assurance gates pass. See [research reproductions](/research/recipes/reproductions/).

## Python SDK example

```python
from pllm.components import list_components

garbling = [item.component for item in list_components() if "garbl" in item.component]
print(garbling)
```

The empty built-in result represents current unsupported SDK state; it is not an execution fallback.
