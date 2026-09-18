# Implementation map

Locate model planning, compiler contracts, native execution, Python APIs, CLI code, schemas, and documentation.

[View canonical HTML](https://pllm.run/sdk/contribute/agents/implementation-map/)

Document ID: `pllm.docs.agents.implementation-map`  
Release: `0.1.0`  
Build: `sha256:d19e46409d656eb3dd08fdadbb8ef6a9e8ca4c33fb48893359235fe855b4a7c4`  
Source hash: `sha256:5da2bf5b4ec123fd8020decc043a08e8b1b0651ca0c9f8f340aaf9f83b70695d`

- `crates/pllm-models`: model-neutral semantic decoder IR and family adapters.
- `crates/pllm-models::cache`: model-state and cache-policy transforms over semantic plans.
- `crates/pllm-types`: canonical Rust plan, benchmark, and assurance records.
- `crates/pllm-compiler`: capability resolution and executable-plan validation.
- `crates/pllm-core`: bounded matrix arithmetic and native kernels.
- `crates/pllm-python`: narrow PyO3 binding.
- `python/pllm`: installed public package, orchestration, and runtime.
- `schemas`: canonical document boundaries.
- `docs`: static Fumadocs publication and generated references.

Dependency direction remains Python to PyO3 to Rust core; semantic model code must not depend on Python or web frameworks.

No public Python API generates this implementation map. Treat it as orientation,
then inspect source and the [research backlog](/research/backlog/) before changing
component boundaries.
