# Method implementations

Understand the difference between a published method and PLLM's independent implementation of it.

[View canonical HTML](https://pllm.run/research/methods/)

Document ID: `pllm.docs.research.methods`  
Release: `0.1.0`  
Build: `sha256:1e873bcc571b7b7717ee66b62aff1ec3e361ead12088e6b707b77e88ee008328`  
Source hash: `sha256:5be647c45b001772f2398e2744bcacddec2d3b9c57cf13f4102bb3eb94fdd0e7`

The current registered implementation is
`pllm.method.mpcache-structural-adaptation.v1`, inspired by
[`pllm.source.mpcache.arxiv-2501.06807v2`](https://arxiv.org/abs/2501.06807v2).
It changes KV-cache structure in a `DecoderPlan`. It does not reproduce the
paper's protected three-party runtime.

The generated [method catalog](/research/records/method-catalog/#method-implementations)
lists versions, provenance, compatible representations and roles, coverage,
evidence, and limitations. Fidelity, protected execution, assurance, benchmarks,
and promotion have not been evaluated for the current method.

A method that resembles a paper is not automatically a faithful reproduction.
