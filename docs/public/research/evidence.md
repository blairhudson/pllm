# Research evidence

Understand which evidence supports fidelity, performance, privacy, and deployment claims.

[View canonical HTML](https://pllm.run/research/evidence/)

Document ID: `pllm.docs.research.evidence`  
Release: `0.1.0`  
Build: `sha256:dab1bcb88380ca9a4c2b79e3de085391f482bb97d841a9f0741979579c87dbfb`  
Source hash: `sha256:40bd5a42a6d476151a99b2f55d1a9bcf77b7c9ac462851f9c6eb659d838319c4`

PLLM records source claims, clean-room fidelity, native implementation agreement,
compiler coverage, runtime support, generation quality,
[matched benchmarks](/sdk/research/benchmarks/),
[assurance](/sdk/research/assurance/), and deployment observations separately.
Missing evidence is unknown; it is not a zero result.

Accepted records live under [`research/evidence/`](https://github.com/blairhudson/pllm/tree/main/research/evidence).
Each record keeps its environment, workload, software revision, and limitations.
A result does not apply to a changed component, plan, model, parameter, or
deployment unless a new validation explicitly connects them.

Keep negative and unavailable results in the record. `not_refuted` means only that
the named attack failed within the recorded budget.
