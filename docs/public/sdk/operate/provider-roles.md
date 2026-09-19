# Provider roles

Understand the responsibilities of deployment operators, preparation services, and inference services.

[View canonical HTML](https://pllm.run/sdk/operate/provider-roles/)

Document ID: `pllm.docs.operate.provider-roles`  
Release: `0.1.0`

Provider supplies static component metadata and implementation artifacts. Operator supplies process placement, authenticated identity, authorization, storage, and operational policy. Preparation creates or installs method-specific material. Inference consumes assigned plan work. A deployment may combine processes, but cannot erase semantic role boundaries or establish non-collusion.

Metadata discovery through `pllm components` never loads providers or native libraries. Research discovery never executes recipes. `pllm serve inference` and `pllm serve preparation` start the two roles, but authenticated role-plan delivery and a complete target lifecycle contract remain blockers. See the [serving guide](/learn/integrations/local-gateway/).

## Python SDK example

```python
from pllm.components import get, get_component
from pllm.roles import Inference

role_class = get("pllm/inference")
method = get_component("pllm/masked-linear")
assert role_class is Inference
assert role_class.describe().role_eligibility == ("inference",)
assert method.role_eligibility == ("client", "preparation", "inference")
```

API: [`pllm.components.get` and `get_component`](/sdk/reference/python/pllm/#objects-and-signatures)

Role metadata does not establish operator identity, authentication, or non-collusion.
