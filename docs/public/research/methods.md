# Method implementations

Understand the difference between a published method and PLLM's independent implementation of it.

[View canonical HTML](https://pllm.run/research/methods/)

Document ID: `pllm.docs.research.methods`  
Release: `0.1.0`  
Build: `sha256:bab6f73b33765ac11862794abf645d4eb324a2fd52abc43cc2a9cd21c09e27c7`  
Source hash: `sha256:984700ca8104eb825f52b2dbbccbef80c6141a73b4e45137252e0f43f9bd42e4`

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

Protected Q7 `SiLU(gate) * up` follows the same rule. Python exposes
`BinaryTableGatedMultiplyQ7` and `R03CrtGatedMultiplyQ7` as interchangeable
method components. `ScalarProtectedTensorSchedule` and
`IndependentLanesProtectedTensorSchedule` select scheduling separately. All
four are distinct component identities in two capability families; R03 is
provenance for one implementation, not a package boundary. Four lanes are the
current maximum, and no selection establishes complete-decoder coverage or
cryptographic assurance.
