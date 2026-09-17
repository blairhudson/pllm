# Compositions and recipes

Learn how PLLM records research recipes and changes to a model plan.

[View canonical HTML](https://pllm.run/research/compositions/)

Document ID: `pllm.docs.research.compositions`  
Release: `0.1.0`  
Build: `sha256:a56b03bbcab50da4d618a10a27d57d6fe15bc392dc06b104c6c4a924b7db4025`  
Source hash: `sha256:aa7404a9b15ad8b26a476f3435638fc049c8b9be619d612c7aeee9a3ccf19171`

Research workflows describe source acquisition, target model operations,
required evidence, review gates, and failure policy. They are documentation, not
executable package records. Start with the [research backlog](/research/backlog/)
and choose a workflow under [research recipes](/research/recipes/).

A [composed plan](/sdk/plans/) records the method, value formats, conversions, numeric policy,
parties, threat model, visible information, state lifetime, workload, component
version from the [component inventory](/sdk/reference/components/), and coverage.
Matching shapes and data types are not enough. A change to the source,
implementation, configuration, plan, workload, or cohort creates new plan
history. Components contribute declared capabilities; they do not establish
runtime coverage, privacy, fidelity, or benchmark parity by their presence alone.

MPCache is one example: `pllm/kv-cache-eviction` structurally adapts a compatible
plan through code in `crates/pllm-models/src/cache.rs`. It is not a protected
MPCache runtime. See [method implementations](/research/methods/) and the public
[component APIs](/sdk/components/).
