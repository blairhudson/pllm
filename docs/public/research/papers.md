# Reimplemented papers

Trace published private-inference methods to their independent PLLM components, evidence, and current limitations.

[View canonical HTML](https://pllm.run/research/papers/)

Document ID: `pllm.docs.research.papers`  
Release: `0.1.0`  
Build: `sha256:1e873bcc571b7b7717ee66b62aff1ec3e361ead12088e6b707b77e88ee008328`  
Source hash: `sha256:9f2002c5fb871e81d81f18b237ed9b99d6e7b70c25bef456c1abbbaad0613e83`

This catalog is about published work reimplemented as PLLM components. A source
record alone does not qualify. Each entry must link the original paper, PLLM's
method record, the component that implements it, and the evidence that establishes
its current maturity.

## MPCache

[MPCache: MPC-Friendly KV Cache Eviction for Efficient Private Large Language
Model Inference](https://arxiv.org/abs/2501.06807v2) inspired PLLM's independent
`pllm.method.mpcache-structural-adaptation.v1` method. It is exposed as the
`pllm/kv-cache-eviction` component and the `KvCacheEviction` plan transform.

- [Source and method records](/research/records/method-catalog/)
- [Component catalog](/sdk/reference/components/)
- [SDK component usage](/sdk/components/)
- [Current support status](/sdk/reference/status/)

The current component transforms compatible Qwen2 semantic plans. It does not
reproduce MPCache's complete three-party protected execution, and its fidelity,
benchmark, assurance, and protected-execution statuses remain unchecked.

## Reading copies and PLLM publications

Each source record stores authors, title, primary URL, revision, publication
details, license, artifact digests, local paper digest, and acquisition status.
Local PDFs under `papers/` are private reading copies and are not included in the
package or website. PLLM's [current paper](/research/paper/) and
[whitepaper](/research/whitepaper/) are separate project publications.

A citation gives attribution. It does not prove that PLLM reproduced the method,
matched a benchmark, or assessed its security.
