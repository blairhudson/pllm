# Method implementations

Understand the difference between a published method and PLLM's independent implementation of it.

[View canonical HTML](https://pllm.run/research/methods/)

Document ID: `pllm.docs.research.methods`  
Release: `0.1.0`  
Build: `sha256:426652b6512bb11e794ef7caf6e150d2b19f0c9f7b933d041295b3bacebb441a`  
Source hash: `sha256:ca2c53148211ceefd423aa628568d13bbec20e4084a222ba9988e21e03389c35`

PLLM's MPCache-inspired work lives in
[`crates/pllm-models/src/cache.rs`](https://github.com/blairhudson/pllm/blob/main/crates/pllm-models/src/cache.rs).
It is exposed through the `pllm/kv-cache-eviction` capability and Python
`KvCacheEviction` component. Applying it changes KV-cache structure in a
[`ModelPlan`](/sdk/plans/); it does not reproduce the paper's protected
three-party runtime.

Use the [component guide](/sdk/components/) and generated
[component inventory](/sdk/reference/components/) for public parameters and
capabilities. The [MPCache paper page](/research/papers/r23-mpcache/) records
provenance and limitations. Fidelity, protected execution, assurance, and
matched benchmarks have not been evaluated for this structural adaptation.

A method that resembles a paper is not automatically a faithful reproduction.
