# R03 · Garbling Gadgets for Boolean and Arithmetic Circuits

**Priority 3 · 2016 · single_evaluator · source checked 2026-09-14**

Authors: Marshall Ball, Tal Malkin, Mike Rosulek.  
Primary source: https://eprint.iacr.org/2016/969  
Access in this handoff: `primary_abstract`. Full source/artifact content hashes are not yet locked. A full-text acquisition gate is required.

## What the source contributes

Arithmetic garbling provides free additions and specialized gadgets, with a stated symmetric-primitive security model.

This short source summary is separate from the proposed PLLM design below. The source has not been reproduced merely by adding this card.

## Native implementation scope

Create a reviewed arithmetic-label core with modulus-specific offsets, permitted constant operations, projection and mixed-modulus gadgets. Keep offsets opaque.

Data flow: **Arithmetic/Boolean label gadgets with source-defined input/output moduli and permitted constants.**

Register the following independently versioned native components:

- `garble.arithmetic`
- `gate.mixed_modulus`

Dependencies: None; foundational work item. Public compilation/model-data scans and all online evaluation happen in Rust or reviewed native accelerator libraries called from Rust. Python selects immutable parameters and displays public results. It does not implement a timed alternative or silently repair unsupported native execution.

## Required fixture and benchmark specification

Published gadget truth tables and byte formulas, negative constant/modulus cases, exhaustive finite-domain differential tests.

Mandatory paper-specific gates:

- Forbidden constants and mixed moduli
- Secret offset exposure through public metadata
- Repeated distinct values under one wire instance

Each fixture locks input/output representations, ring/field parameters, numeric graph, party count and material lifetime. Compare an optimized implementation with identical functionality and applicable privacy assumptions; record original-artifact numbers and adaptations separately. A source-advertised speedup is not an expected threshold for different hardware.

## Privacy, correctness and proof obligations

Source garbling game with exact hash assumptions, modulus preconditions and compatible composition.

Functionality alone does not validate affine-label correlations or an arbitrary hash replacement. Reject unsupported non-unit constants or lower them through justified gadgets.

Provide the corrupted party's permitted view, known plaintext/public inputs, randomness model, allowed leakage and attack budget. Formal or empirical results have explicit scope. Passing functional tests cannot grant a production privacy badge. A changed hash, field, layout, actor role or precision creates a documented adaptation obligation.

## What PLLM already has

PLLM now has a clean-room mixed-modulus reference and one bounded single-modulus Q7 SiLU projection slice with exhaustive finite-domain parity, authenticated plan binding, strict gate serialization and bounded process-local one-use enforcement. This adapted component is not a proof-equivalent reproduction of the source and has no cryptographic review.

The original experiments and limitations are under `legacy/`. This handoff adds contracts and research tasks, **not a completed native reproduction of this paper**.

## Reproduction gates

`R03.acquire → R03.specify → R03.reference → R03.native → R03.assure → R03.benchmark → R03.document`.

The public research backlog records the remaining implementation and validation work. Native integration requires the actual PLLM command, source locks, raw measurements, and assurance report.

## Developer-facing explainer requirement

The eventual method page needs a worked operator example, a role diagram in prose or graphics, source section links, why/when to use it, configuration parameters, unsupported cases, numeric envelope, failure behavior and source-vs-PLLM differences. Generate tables from evidence rather than copying old performance ratios. Show `not implemented`, `adapted`, `reference only` or `reproduced at scope X` honestly.

## Default policy

`eligible_default = false` until implementation, full coverage, source/security review and deployment gates are met. The track restriction remains binding even after successful reproduction. Experimental or comparison methods cannot silently enter the preferred single-evaluator/no-HE profile.
