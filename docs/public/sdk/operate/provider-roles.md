# Provider roles

Understand the responsibilities of deployment operators, preparation services, and inference services.

[View canonical HTML](https://pllm.run/sdk/operate/provider-roles/)

Document ID: `pllm.docs.operate.provider-roles`  
Release: `0.1.0`  
Build: `sha256:65f4ad316621284cc28b60b1a825a6d9c6d1f108cbb5032aee0983de1e5c4966`  
Source hash: `sha256:eee97122d4ad7884373a0a4f0a8136daaa74552e7eab4ead3e707d2ab5cae805`

Provider supplies static component metadata and implementation artifacts. Operator supplies process placement, authenticated identity, authorization, storage, and operational policy. Preparation creates or installs method-specific material. Inference consumes assigned plan work. A deployment may combine processes, but cannot erase semantic role boundaries or establish non-collusion.

Metadata discovery through `pllm components` never loads providers or native libraries. Research discovery never executes recipes. No parser-visible party-service command exists; authenticated role-plan delivery and complete target lifecycle contract remain blockers.

## Python SDK example

```python
from pllm.components import get_component

method = get_component("pllm/masked-linear")
assert method.lifecycle_phase == "compilation"
assert method.role_eligibility == ("client", "preparation", "inference")
```

API: [`pllm.components.get_component`](/sdk/reference/python/pllm/#objects-and-signatures)

Role metadata does not establish operator identity, authentication, or non-collusion.
