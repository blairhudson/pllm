# Architecture

Understand PLLM's plans, runtime stages, role ownership, and side effects.

[View canonical HTML](https://pllm.run/learn/understand/architecture/)

Document ID: `pllm.docs.understand.architecture`  
Release: `0.1.0`  
Build: `sha256:14dfd6dfb589b89eccacdb134c3034e325dd156876b4d3e1dee4a6f7c4e70863`  
Source hash: `sha256:7fb0e3fb1eb830d498b29238555a6fa4ebffdb042e9c0f1e154f746cb0a73133`

```text
Experiment -> resolved public configuration
model config -> ModelPlan -> component lineage -> coverage
compile request -> immutable logical/execution/lock records
preparation -> role-local one-use material + public readiness receipt
runtime -> bounded execution
benchmark/assurance -> cohort-bound evidence
publication -> scoped claim linked to evidence
```

Python owns immutable declarations and coarse calls. Rust owns lowering, validation, hot paths, clocks, and assurance where implemented. Constructors perform no downloads, device discovery, provider loads, compilation, preparation, process startup, or network work.

Current runtime modules predate complete lifecycle orchestration. Their importability is not support. See [status](/sdk/reference/status/) and normative [architecture](https://github.com/blairhudson/pllm/blob/main/ARCHITECTURE.md).
