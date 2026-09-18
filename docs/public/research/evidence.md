# Research evidence

Understand which evidence supports fidelity, performance, privacy, and deployment claims.

[View canonical HTML](https://pllm.run/research/evidence/)

Document ID: `pllm.docs.research.evidence`  
Release: `0.1.0`  
Build: `sha256:96b6d9446e37113d9d2892113cefbcf72b64f5f3fc8eeb331b7caddd36ad60a0`  
Source hash: `sha256:1143da9f355941d7d227919abb30be9d25e8915beb18f91a788ae52500e5f2dd`

PLLM records source claims, clean-room fidelity, native implementation agreement,
compiler coverage, runtime support, generation quality,
[matched benchmarks](/sdk/research/benchmarks/),
[assurance](/sdk/research/assurance/), and deployment observations separately.
Missing evidence is unknown; it is not a zero result.

Accepted records live under [`docs/evidence/`](https://github.com/blairhudson/pllm/tree/main/docs/evidence).
Each record keeps its environment, workload, software revision, and limitations.
A result does not apply to a changed component, plan, model, parameter, or
deployment unless a new validation explicitly connects them.

Keep negative and unavailable results in the record. `not_refuted` means only that
the named attack failed within the recorded budget.
