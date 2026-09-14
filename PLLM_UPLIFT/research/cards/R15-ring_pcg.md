# R15 · Efficient Pseudorandom Correlation Generators over Z/p^k Z

**Priority 15 · 2025 · preparation_comparison · source checked 2026-09-14**

Authors: Zhe Li, Chaoping Xing, Yizhou Yao, Chen Yuan.  
Primary source: https://link.springer.com/chapter/10.1007/978-3-032-01884-7_7  
Access in this handoff: `primary_abstract`. Full source/artifact content hashes are not yet locked. A full-text acquisition gate is required.

## What the source contributes

Galois-ring constructions extend pseudorandom correlation generation to integer rings, including multiplication-related correlations.

This short source summary is separate from the proposed PLLM design below. The source has not been reproduced merely by adding this card.

## Native implementation scope

Implement reviewed parameter sets and the actual requested correlation schema. Expose seed setup, expansion time, memory and transcript requirements; do not substitute ordinary PRG seeds.

Data flow: **Approved correlated-seed setup → per-role pseudorandom correlation schema, not shared unrestricted seeds.**

Register the following independently versioned native components:

- `prepare.ring_pcg`
- `correlation.ole`
- `correlation.authenticated_triple`

Dependencies: None; foundational work item. Public compilation/model-data scans and all online evaluation happen in Rust or reviewed native accelerator libraries called from Rust. Python selects immutable parameters and displays public results. It does not implement a timed alternative or silently repair unsupported native execution.

## Required fixture and benchmark specification

Correct correlation identities, original artifact outputs, communication versus expansion compute, and current attack-aware parameter checks.

Mandatory paper-specific gates:

- Correlation identities
- Modern parameter/attack review
- Expansion memory and CPU compared with material transmission

Each fixture locks input/output representations, ring/field parameters, numeric graph, party count and material lifetime. Compare an optimized implementation with identical functionality and applicable privacy assumptions; record original-artifact numbers and adaptations separately. A source-advertised speedup is not an expected threshold for different hardware.

## Privacy, correctness and proof obligations

Distribution and computational indistinguishability argument for the actual ring/PCG parameters; toy algebra is insufficient.

PCGs do not automatically compress arbitrary garbled tables or spectral coefficients. Small toy parameters are not deployment security parameters.

Provide the corrupted party's permitted view, known plaintext/public inputs, randomness model, allowed leakage and attack budget. Formal or empirical results have explicit scope. Passing functional tests cannot grant a production privacy badge. A changed hash, field, layout, actor role or precision creates a documented adaptation obligation.

## What PLLM already has

Earlier Galois-ring checks were algebra only, not a PCG implementation.

The original experiments and limitations are under `legacy/`. This handoff adds contracts and research tasks, **not a completed native reproduction of this paper**.

## Reproduction gates

`R15.acquire → R15.specify → R15.reference → R15.native → R15.assure → R15.benchmark → R15.document`.

The machine-readable recipe at `research/recipes/R15.json` is a work specification, not a pretend executable paper command. During acquisition, store the exact original artifact invocation and commit; leave them unclaimed until obtained. During native integration, attach the actual PLLM command, all environment/source locks, raw measurements and assurance report.

## Developer-facing explainer requirement

The eventual method page needs a worked operator example, a role diagram in prose or graphics, source section links, why/when to use it, configuration parameters, unsupported cases, numeric envelope, failure behavior and source-vs-PLLM differences. Generate tables from evidence rather than copying old performance ratios. Show `not implemented`, `adapted`, `reference only` or `reproduced at scope X` honestly.

## Default policy

`eligible_default = false` until implementation, full coverage, source/security review and deployment gates are met. The track restriction remains binding even after successful reproduction. Experimental or comparison methods cannot silently enter the preferred single-evaluator/no-HE profile.
