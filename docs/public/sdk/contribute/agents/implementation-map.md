# Implementation map

Locate model planning, compiler contracts, native execution, Python APIs, CLI code, schemas, and documentation.

[View canonical HTML](https://pllm.run/sdk/contribute/agents/implementation-map/)

Document ID: `pllm.docs.agents.implementation-map`  
Release: `0.1.0`  
Build: `sha256:dab1bcb88380ca9a4c2b79e3de085391f482bb97d841a9f0741979579c87dbfb`  
Source hash: `sha256:ef33a0646ac659357e0b7e180477a382f7d0b3a01fbe4932360fcbe98edd66aa`

- `crates/pllm-models`: model-neutral semantic decoder IR and family adapters.
- `crates/pllm-method-mpcache`: generic research transform over semantic plans.
- `crates/pllm-types`: canonical Rust plan, benchmark, and assurance records.
- `crates/pllm-compiler`: capability resolution and executable-plan validation.
- `crates/pllm-core`: bounded matrix arithmetic and native kernels.
- `crates/pllm-python`: narrow PyO3 binding.
- `python/pllm`: installed public package, orchestration, and runtime.
- `schemas`: canonical document boundaries.
- `docs`: static Fumadocs publication and generated references.

Dependency direction remains Python to PyO3 to Rust core; semantic model code must not depend on Python or web frameworks.

## Python SDK example

```python
from pllm.research import render_agents_guide

guide = render_agents_guide()
assert "`python/pllm/` is the only installed Python namespace" in guide
```

The generated guide locates public and implementation surfaces without importing implementation providers.

API: [`pllm.research.render_agents_guide`](/sdk/reference/python/pllm/#objects-and-signatures)
