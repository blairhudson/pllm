# Research evidence

Understand which evidence supports fidelity, performance, privacy, and deployment claims.

[View canonical HTML](https://pllm.run/research/evidence/)

Document ID: `pllm.docs.research.evidence`  
Release: `0.1.0`  
Build: `sha256:bf56c232413fe9b57bc2befe009368956690afaf0d708914c7620c3b7d5d9531`  
Source hash: `sha256:0c75025889d82d4e3d3606346cef3b27912faef9fedb03a8370fa91b4c864d0d`

PLLM records source claims, clean-room fidelity, native implementation agreement,
compiler coverage, runtime support, generation quality, matched benchmarks,
assurance, and deployment observations separately. Missing evidence is unknown;
it is not a zero result.

Accepted records live under [`research/evidence/`](https://github.com/blairhudson/pllm/tree/main/research/evidence).
Each record keeps its environment, workload, software revision, and limitations.
A result does not apply to a changed component, plan, model, parameter, or
deployment unless a new validation explicitly connects them.

Keep negative and unavailable results in the record. `not_refuted` means only that
the named attack failed within the recorded budget.
