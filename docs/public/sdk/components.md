# Components

Inspect versioned capabilities and apply compatible components to a model plan.

[View canonical HTML](https://pllm.run/sdk/components/)

Document ID: `pllm.docs.sdk.components`  
Release: `0.1.0`  
Build: `sha256:55dedf191ed9de69d95eb69a68160a404194419bf2f721a57103e95fdce4a4e0`  
Source hash: `sha256:332d39c20ae8b0e8576995e6d06b0acb7f23f55a341c9ca698c3ecc14b651dce`

A component identifies one versioned capability and its public parameters. The
record also states its implementation status, compatible value formats, roles,
model coverage, evidence, and known limits.

List the built-in records with
[`pllm components list`](/cli/reference/components/list/):

```bash
uv run pllm components list
```

Inspect one record as JSON with
[`pllm components show`](/cli/reference/components/show/):

```bash
uv run pllm --format json components show pllm/kv-cache-eviction
```

Python users select a component with `ComponentRef` and apply it to a compatible
`ModelPlan`. Compatibility alone does not prove compiler coverage, runtime support,
performance, privacy, or generation quality.

## Python SDK example

```python
from pllm.components import get_component

component = get_component("pllm/kv-cache-eviction")
assert component.lifecycle_phase == "model-lowering"
assert "bounded-kv-cache-eviction" in component.capabilities
```

API: [Python SDK objects and signatures](/sdk/reference/python/pllm/#objects-and-signatures)

See the generated [component inventory](/sdk/reference/components/) for current
versions and status. Interpret any linked measurements and limitations under the
[research evidence](/research/evidence/) rules.
