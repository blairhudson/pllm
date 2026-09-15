# Components

Inspect versioned capabilities and apply compatible components to a model plan.

[View canonical HTML](https://pllm.run/sdk/components/)

Document ID: `pllm.docs.sdk.components`  
Release: `0.1.0`  
Build: `sha256:1e873bcc571b7b7717ee66b62aff1ec3e361ead12088e6b707b77e88ee008328`  
Source hash: `sha256:1b822ba66c418538aef39693fdf4a6f2675af9950fd086a5ba973922160b1760`

A component identifies one versioned capability and its public parameters. The
record also states its implementation status, compatible value formats, roles,
model coverage, evidence, and known limits.

List the built-in records:

```bash
uv run pllm components list
```

Inspect one record as JSON:

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
print(component.lifecycle_phase, component.capabilities)
```

See the generated [component inventory](/sdk/reference/components/) for current
versions and status.
