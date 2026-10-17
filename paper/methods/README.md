# Ten-method implementation contracts

These documents define the source requirements for independent PLLM
reimplementations. They are specifications, not completed implementations or
benchmark results.

| Contract | Methods |
| --- | --- |
| [Delegation](delegation-contracts.md) | Gupta–Katz–Miers; Maverick |
| [Shared computation](shared-computation-contracts.md) | SIGMA; FuseFSS; BumbleBee; PUMA |
| [Encrypted computation](encrypted-computation-contracts.md) | NEXUS; THOR; CacheMir; ROSETTA |

[Source lock v1](source-lock.v1.json) records all ten governing PDFs, their byte
lengths, hashes, and local cache paths. The
[coverage audit](../TEN-METHOD-COVERAGE.md) records existing implementation scope.
The [reproduction workflow](../reproduction-contracts/README.md) defines approval,
validation, and benchmark requirements. Independent contract reviews are retained
there separately.

The [author decision record](DECISIONS.md) selects Apple-M CPU-first execution
and Qwen2.5-0.5B-Instruct as the common model. The
[Rust HE review](rust-he-backend-qualification.md) identifies Poulpy as the
approved bounded qualification candidate. Its torus representation still needs
an explicit mapping to each source method; no Go backend has been approved.
The [first admission gate](../../benchmarks/native/poulpy_refresh/README.md)
rejects the eager key bundle before allocation. No encrypted refresh has run.
The [staged owner review](poulpy-staged-owner-bound.md) rejects one bounded-key
allocation policy and records the remaining ownership gaps for a tighter policy.
The approved 8 GiB ceiling and cryptographic parameters remain unchanged.

Implement the published algorithms in reusable native, compiler, and runtime
components. Use upstream code only as specification and oracle material.
Substantial changes to arithmetic, protocol, trust, model functionality, or
execution require approval before implementation.
The [ordinary SDK gate](DECISIONS.md#d5-ordinary-sdk-integration-is-required)
requires shared component discovery/serialization, `Pipeline`, `Experiment`,
native admission, SDK/gateway, CLI and benchmark integration. Internal probes
are milestones; they do not complete a method or authorize a separate public API.
The [integration contract](sdk-integration.md) identifies the shared extension
points, the bounded GKM milestone and required end-to-end checks.

The first selected implementation path is a complete Gupta–Katz–Miers request.
Its source-resolution decisions precede a verified linear capsule, private
embedding lookup, compiler-bound decoder, and two-worker SDK request. A primitive
or generated fixture alone cannot qualify the complete method.

The bounded [native capsule](../../docs/research/gkm-delegation-capsule.md) and
[seeded-share profile](../../docs/research/gkm-seeded-shares.md) now pass their
arithmetic and ownership checks. The separate
[matrix inventory](../../docs/research/decoder-matrix-inventory.md) extracts
compiler groups without selecting W8A8. Native metadata and rebuilt Python
binding tests pass. The pinned Qwen source scan also passes its source-only
numeric gate after correcting macOS swap telemetry. Complete numeric-graph,
publication, actual-input-bound, state and transport admission remain open.
The separate [bounded GKM decoder](../../docs/research/gkm-decoder-reference.md)
now composes the compiler schedule, original lookup bytes, source fixed-point
linears and request-end acceptance. Generated Qwen2/Qwen3/Llama full/seeded
prefill/decode matches independent logits and final-KV oracles. The small
conformance envelope rejects real-checkpoint geometry; original-model, full-width,
distributed and performance qualification remains open.
The admission-repaired wheel passes 206 Python and 44 native checks. Pre-lowering
geometry budgets and bounded source-map reads close the integration findings;
earlier functional evidence retains its separate identity.
The subsequent [bounded SDK composition](../../docs/research/gkm-sdk-reference.md)
adds actual party-local worker processes and the ordinary component, Experiment,
SDK, gateway, CLI and benchmark paths. Its final cached wheel passes 547 Python
and 117 native checks plus Clippy. Full-checkpoint admission, original-model and
common-Qwen quality, independent operators and publication measurements remain open.

## Current validation host

The inspected local host has an Apple M5, ten logical CPUs, and 32 GiB of RAM.
It is suitable for bounded native correctness work. It does not reproduce the
NVIDIA GPU hardware used by several sources. Docker is installed, but its daemon
was unavailable at inspection. Existing Docker role launchers can enforce CPU
quotas and memory limits; ordinary native worker counts do not enforce them.

Choose and admit the final CPU/GPU campaign environment separately. Each result
must identify its actual backend, host/device owners, and enforced capacity.
Source hardware measurements cannot supply missing PLLM measurements.
