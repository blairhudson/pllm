# Provider roles

Understand the responsibilities of deployment operators, preparation services, and inference services.

[View canonical HTML](https://pllm.run/sdk/operate/provider-roles/)

Document ID: `pllm.docs.operate.provider-roles`  
Release: `0.1.0`  
Build: `sha256:e5e20904c12dfdeeed24c72fdb072c93f4be72953e9293c63a83ddce3d871d76`  
Source hash: `sha256:a50a07c33083d4790c19b3645ea7d0b2f09692426aaf7e2446fccaa0f2803872`

Provider supplies static component metadata and implementation artifacts. Operator supplies process placement, authenticated identity, authorization, storage, and operational policy. Preparation creates or installs method-specific material. Inference consumes assigned plan work. A deployment may combine processes, but cannot erase semantic role boundaries or establish non-collusion.

Metadata discovery through `pllm components` never loads providers or native libraries. Research discovery never executes recipes. `pllm serve inference` and `pllm serve preparation` start the two roles, but authenticated role-plan delivery and a complete target lifecycle contract remain blockers. See the [serving guide](/learn/integrations/local-gateway/).

## Python SDK example

```python
from pllm.components import get_component

method = get_component("pllm/masked-linear")
assert method.lifecycle_phase == "compilation"
assert method.role_eligibility == ("client", "preparation", "inference")
```

API: [`pllm.components.get_component`](/sdk/reference/python/pllm/#objects-and-signatures)

Role metadata does not establish operator identity, authentication, or non-collusion.
