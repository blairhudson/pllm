# Components

Inspect versioned capabilities and apply compatible components to a model plan.

[View canonical HTML](https://pllm.run/sdk/components/)

Document ID: `pllm.docs.sdk.components`  
Release: `0.1.0`

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

Concrete component classes live in capability families such as `pllm.protocols`, `pllm.preparation`, `pllm.correlation`, `pllm.kernels`, `pllm.metrics`, `pllm.nonlinear`, `pllm.schedulers`, `pllm.state`, `pllm.passes`, `pllm.roles`, and `pllm.verification`; model-source specializations live in `pllm.sources` outside component slots. `pllm.components` contains the generic `ComponentRef`/`ComponentDescriptor` machinery and the class-derived registry. `get(identity)` returns the registered class; `get_component(identity)` returns that class's descriptor. Python users construct a concrete class and apply it to a compatible `ModelPlan` or profile slot. Compatibility alone does not prove compiler coverage, runtime support, performance, privacy, or generation quality.

## Python SDK example

```python
from pllm.components import get, get_component
from pllm.passes import KvCacheEviction

component_class = get("pllm/kv-cache-eviction")
descriptor = get_component("pllm/kv-cache-eviction")
assert component_class is KvCacheEviction
assert component_class.describe() is descriptor
assert descriptor.lifecycle_phase == "model-lowering"
```

API: [Python SDK objects and signatures](/sdk/reference/python/pllm/#objects-and-signatures)

See the generated [component inventory](/sdk/reference/components/) for current
versions and status. Interpret any linked measurements and limitations under the
[research evidence](/research/evidence/) rules.
