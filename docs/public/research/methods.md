# Method implementations

Understand the difference between a published method and PLLM's independent implementation of it.

[View canonical HTML](https://pllm.run/research/methods/)

Document ID: `pllm.docs.research.methods`  
Release: `0.1.0`  
Build: `sha256:2e8eacabaf27e4c40f6b41814c3af934ab5186546825b569c6c9c1cb0a7b4db4`  
Source hash: `sha256:bade14cd2b0d57d95384fd8eb9fbb288e2109dd7a8dae416db987d94d2545bba`

PLLM's MPCache-inspired work lives in
[`crates/pllm-models/src/cache.rs`](https://github.com/blairhudson/pllm/blob/main/crates/pllm-models/src/cache.rs).
It is exposed through the `pllm/kv-cache-eviction` capability and Python
`KvCacheEviction` component. Applying it to a dense Qwen2 or Qwen3
[`ModelPlan`](/sdk/plans/) preserves fixed-capacity Key/Value state, adds explicit
static-selection index state, and rewires decode attention through bounded dynamic
gathers. It does not reproduce the paper's protected three-party runtime.

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
