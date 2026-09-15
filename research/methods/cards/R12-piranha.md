# R12 · Piranha: A GPU Platform for Secure Computation

**Priority 12 · 2022 · mpc_comparison · source checked 2026-09-14**

Authors: Jean-Luc Watson, Sameer Wagh, Raluca Ada Popa.  
Primary source: https://www.usenix.org/conference/usenixsecurity22/presentation/watson  
Access in this handoff: `primary_abstract`. Full source/artifact content hashes are not yet locked. A full-text acquisition gate is required.

## What the source contributes

Piranha separates device, protocol and application layers for GPU-accelerated secret-sharing computation.

This short source summary is separate from the proposed PLLM design below. The source has not been reproduced merely by adding this card.

## Native implementation scope

Wrap original GPU kernels through a narrow native interface before selective Rust/CUDA ports. Preserve protocol party counts and numeric domains in comparisons.

Data flow: **Protocol-specific shares in device buffers → corresponding modular GPU operations under the original role count.**

Register the following independently versioned native components:

- `kernel.modular_gpu`
- `runtime.device_buffers`

Dependencies: None; foundational work item. Public compilation/model-data scans and all online evaluation happen in Rust or reviewed native accelerator libraries called from Rust. Python selects immutable parameters and displays public results. It does not implement a timed alternative or silently repair unsupported native execution.

## Required fixture and benchmark specification

Integer-kernel parity and matching original protocol workloads; device execution and transfer time, all workers and GPU allocation counted.

Mandatory paper-specific gates:

- CPU/GPU exactness including overflow
- Asynchronous synchronization and pinned buffers
- All devices and transfers included in timing

Each fixture locks input/output representations, ring/field parameters, numeric graph, party count and material lifetime. Compare an optimized implementation with identical functionality and applicable privacy assumptions; record original-artifact numbers and adaptations separately. A source-advertised speedup is not an expected threshold for different hardware.

## Privacy, correctness and proof obligations

Kernel arithmetic plus original sharing-protocol assumptions; GPU memory/caches and side channels need explicit deployment scope.

Kernel reuse does not turn a multi-online-worker protocol into the preferred single-evaluator topology.

Provide the corrupted party's permitted view, known plaintext/public inputs, randomness model, allowed leakage and attack budget. Formal or empirical results have explicit scope. Passing functional tests cannot grant a production privacy badge. A changed hash, field, layout, actor role or precision creates a documented adaptation obligation.

## What PLLM already has

Previous CPU limb/packing experiments are not a GPU Piranha reproduction.

The original experiments and limitations are under `legacy/`. This handoff adds contracts and research tasks, **not a completed native reproduction of this paper**.

## Reproduction gates

`R12.acquire → R12.specify → R12.reference → R12.native → R12.assure → R12.benchmark → R12.document`.

The machine-readable recipe at `research/recipes/R12.json` is a work specification, not a pretend executable paper command. During acquisition, store the exact original artifact invocation and commit; leave them unclaimed until obtained. During native integration, attach the actual PLLM command, all environment/source locks, raw measurements and assurance report.

## Developer-facing explainer requirement

The eventual method page needs a worked operator example, a role diagram in prose or graphics, source section links, why/when to use it, configuration parameters, unsupported cases, numeric envelope, failure behavior and source-vs-PLLM differences. Generate tables from evidence rather than copying old performance ratios. Show `not implemented`, `adapted`, `reference only` or `reproduced at scope X` honestly.

## Default policy

`eligible_default = false` until implementation, full coverage, source/security review and deployment gates are met. The track restriction remains binding even after successful reproduction. Experimental or comparison methods cannot silently enter the preferred single-evaluator/no-HE profile.
