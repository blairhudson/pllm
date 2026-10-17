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

Implement the published algorithms in reusable native, compiler, and runtime
components. Use upstream code only as specification and oracle material.
Substantial changes to arithmetic, protocol, trust, model functionality, or
execution require approval before implementation.

The first selected implementation path is a complete Gupta–Katz–Miers request.
Its source-resolution decisions precede a verified linear capsule, private
embedding lookup, compiler-bound decoder, and two-worker SDK request. A primitive
or generated fixture alone cannot qualify the complete method.

## Current validation host

The inspected local host has an Apple M5, ten logical CPUs, and 32 GiB of RAM.
It is suitable for bounded native correctness work. It does not reproduce the
NVIDIA GPU hardware used by several sources. Docker is installed, but its daemon
was unavailable at inspection. Existing Docker role launchers can enforce CPU
quotas and memory limits; ordinary native worker counts do not enforce them.

Choose and admit the final CPU/GPU campaign environment separately. Each result
must identify its actual backend, host/device owners, and enforced capacity.
Source hardware measurements cannot supply missing PLLM measurements.
