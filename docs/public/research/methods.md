# Method implementations

Understand the difference between a published method and PLLM's independent implementation of it.

[View canonical HTML](https://pllm.run/research/methods/)

Document ID: `pllm.docs.research.methods`  
Release: `0.1.0`  
Build: `sha256:4a93c61285a110010f1bafefa367e198ed52465071615e2d4d9a91a45f2d82e2`  
Source hash: `sha256:a264b922a593a3e1837d3391f6491c31b83e142d509a5ee5b13b1aba27897782`

The current registered implementation is
`pllm.method.mpcache-structural-adaptation.v1`, inspired by
[`pllm.source.mpcache.arxiv-2501.06807v2`](https://arxiv.org/abs/2501.06807v2).
Inspect its record with
[`pllm research methods show`](/cli/reference/research/methods/show/). It changes
KV-cache structure in a [`DecoderPlan`](/sdk/plans/). It does not reproduce the
paper's protected three-party runtime.

The generated [method catalog](/research/records/method-catalog/#method-implementations)
lists versions, provenance, compatible representations and roles, coverage,
evidence, and limitations. Fidelity, protected execution, assurance, benchmarks,
and promotion have not been evaluated for the current method.

A method that resembles a paper is not automatically a faithful reproduction.
