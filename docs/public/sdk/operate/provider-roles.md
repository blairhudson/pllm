# Provider roles

Understand the responsibilities of deployment operators, preparation services, and inference services.

[View canonical HTML](https://pllm.run/sdk/operate/provider-roles/)

Document ID: `pllm.docs.operate.provider-roles`  
Release: `0.1.0`  
Build: `sha256:1e873bcc571b7b7717ee66b62aff1ec3e361ead12088e6b707b77e88ee008328`  
Source hash: `sha256:f5dd91b975c2ba60f9bcb7897df343c7bfc131a5dfed9aaabd871c6ff445159c`

Provider supplies static component metadata and implementation artifacts. Operator supplies process placement, authenticated identity, authorization, storage, and operational policy. Preparation creates or installs method-specific material. Inference consumes assigned plan work. A deployment may combine processes, but cannot erase semantic role boundaries or establish non-collusion.

Metadata discovery through `pllm components` never loads providers or native libraries. Research discovery never executes recipes. No parser-visible party-service command exists; authenticated role-plan delivery and complete target lifecycle contract remain blockers.

## Python SDK example

```python
from pllm.components import get_component

method = get_component("pllm/masked-linear")
print(method.role_eligibility)
```

Role metadata does not establish operator identity, authentication, or non-collusion.
