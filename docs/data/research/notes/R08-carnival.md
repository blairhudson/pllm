# R08 · Slalom at the Carnival: Privacy-preserving Inference with Masks from Public Knowledge

**Priority 8 · 2024 · quarantined_comparison · source checked 2026-09-23**

Authors: Ida Bruhns, Sebastian Berndt, Jonas Sander, Thomas Eisenbarth.  
Primary source: https://cic.iacr.org/p/1/3/40  
The primary full text is fingerprinted in the paper library. The complete Carnival method remains unimplemented.

## What the source contributes

Carnival uses subset-sum-based masks to accelerate preprocessing for linear outsourcing.

This short source summary is separate from the proposed PLLM design below. The source has not been reproduced merely by adding this card.

## Native implementation scope

The bounded mod-2 public-subspace leakage witness from Maverick Appendix E is available through `pllm.assurance.PublicSubspaceMaskRegression`. Implementing an unaffected Carnival variant requires separate source-specific review and comparison.

Data flow: **Published subset-sum mask/setup variant; isolated from eligible production methods.**

Register the following independently versioned native components:

- `prepare.subset_sum`
- `attack.carnival_projection`

Dependencies: R07. Public compilation/model-data scans and all online evaluation happen in Rust or reviewed native accelerator libraries called from Rust. Python selects immutable parameters and displays public results. It does not implement a timed alternative or silently repair unsupported native execution.

## Required fixture and benchmark specification

Preparation time, traffic, distribution tests and the specific published attack, with the attacked variant and assumptions identified.

Mandatory paper-specific gates:

- Exact parameter recreation
- Maverick attacked-variant reproduction
- A matched full-uniform mask negative-control baseline

Each fixture locks input/output representations, ring/field parameters, numeric graph, party count and material lifetime. Compare an optimized implementation with identical functionality and applicable privacy assumptions; record original-artifact numbers and adaptations separately. A source-advertised speedup is not an expected threshold for different hardware.

## Privacy, correctness and proof obligations

Variant-specific cryptanalysis determines eligibility; no empirical mask randomness test substitutes for its security assumption.

Maverick reports a privacy flaw for a Carnival instantiation. This is an attack-reproduction target, not an approved default or a blanket rejection of every variant.

Provide the corrupted party's permitted view, known plaintext/public inputs, randomness model, allowed leakage and attack budget. Formal or empirical results have explicit scope. Passing functional tests cannot grant a production privacy badge. A changed hash, field, layout, actor role or precision creates a documented adaptation obligation.

## What PLLM already has

PLLM's assurance control demonstrates leaked input parity for deficient public mask bases over u16/u24/u32 rings. It does not implement Carnival or claim that PLLM's private seeded inventory exposes such a basis.

The original experiments and limitations are under `legacy/`. This control is **not a completed native reproduction of Carnival**.

## Reproduction gates

`R08.acquire` is complete for the paper PDF. The remaining method gates are `R08.specify → R08.reference → R08.native → R08.assure → R08.benchmark → R08.document`.

The public research backlog records the remaining implementation and validation work. Native integration requires the actual PLLM command, source locks, raw measurements, and assurance report.

## Developer-facing explainer requirement

The eventual method page needs a worked operator example, a role diagram in prose or graphics, source section links, why/when to use it, configuration parameters, unsupported cases, numeric envelope, failure behavior and source-vs-PLLM differences. Generate tables from evidence rather than copying old performance ratios. Show `not implemented`, `adapted`, `reference only` or `reproduced at scope X` honestly.

## Default policy

`eligible_default = false` until implementation, full coverage, source/security review and deployment gates are met. The track restriction remains binding even after successful reproduction. Experimental or comparison methods cannot silently enter the preferred single-evaluator/no-HE profile.
