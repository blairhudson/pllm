# Implementation map

Locate model planning, compiler contracts, native execution, Python APIs, CLI code, schemas, and documentation.

[View canonical HTML](https://pllm.run/sdk/contribute/agents/implementation-map/)

Document ID: `pllm.docs.agents.implementation-map`  
Release: `0.1.0`  
Build: `sha256:bf56c232413fe9b57bc2befe009368956690afaf0d708914c7620c3b7d5d9531`  
Source hash: `sha256:186d77e7290e98cfd085b9499983345cbda52c3a0387ed15e884f9263c83b4c3`

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

print(render_agents_guide())
```

The generated guide locates public and implementation surfaces without importing implementation providers.
