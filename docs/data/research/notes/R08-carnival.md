# R08 · Slalom at the Carnival: Privacy-preserving Inference with Masks from Public Knowledge

**Priority 8 · 2024 · quarantined_comparison · source checked 2026-09-14**

Authors: Ida Bruhns, Sebastian Berndt, Jonas Sander, Thomas Eisenbarth.  
Primary source: https://cic.iacr.org/p/1/3/40  
Access in this handoff: `primary_abstract`. Full source/artifact content hashes are not yet locked. A full-text acquisition gate is required.

## What the source contributes

Carnival uses subset-sum-based masks to accelerate preprocessing for linear outsourcing.

This short source summary is separate from the proposed PLLM design below. The source has not been reproduced merely by adding this card.

## Native implementation scope

Implement the exact published parameterized variant in an isolated comparison backend. Reproduce the reported Maverick attack before eligibility review.

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

Prior discussion identified this attack; no claim that our fresh full-ring masks instantiate Carnival.

The original experiments and limitations are under `legacy/`. This handoff adds contracts and research tasks, **not a completed native reproduction of this paper**.

## Reproduction gates

`R08.acquire → R08.specify → R08.reference → R08.native → R08.assure → R08.benchmark → R08.document`.

The public research backlog records the remaining implementation and validation work. Native integration requires the actual PLLM command, source locks, raw measurements, and assurance report.

## Developer-facing explainer requirement

The eventual method page needs a worked operator example, a role diagram in prose or graphics, source section links, why/when to use it, configuration parameters, unsupported cases, numeric envelope, failure behavior and source-vs-PLLM differences. Generate tables from evidence rather than copying old performance ratios. Show `not implemented`, `adapted`, `reference only` or `reproduced at scope X` honestly.

## Default policy

`eligible_default = false` until implementation, full coverage, source/security review and deployment gates are met. The track restriction remains binding even after successful reproduction. Experimental or comparison methods cannot silently enter the preferred single-evaluator/no-HE profile.
