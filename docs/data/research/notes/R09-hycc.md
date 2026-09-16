# R09 · HyCC: Compilation of Hybrid Protocols for Practical Secure Computation

**Priority 9 · 2018 · compiler · source checked 2026-09-14**

Authors: Niklas Büscher, Daniel Demmler, Stefan Katzenbeisser, David Kretzmer, Thomas Schneider.  
Primary source: https://doi.org/10.1145/3243734.3243786  
Access in this handoff: `primary_author_search_excerpt`. Full source/artifact content hashes are not yet locked. A full-text acquisition gate is required.

## What the source contributes

HyCC compiles high-level programs into hybrid secure-computation protocols.

This short source summary is separate from the proposed PLLM design below. The source has not been reproduced merely by adding this card.

## Native implementation scope

Reproduce a fixed published hybrid-selection benchmark. Separate its original frontend and backend assumptions from PLLM region planning, protected-state effects and generative workloads.

Data flow: **Numeric graph + admitted backend/conversion costs → hybrid plan; no cryptographic material at the optimizer.**

Register the following independently versioned native components:

- `compiler.region_selection`
- `compiler.conversion_costs`

Dependencies: R03, R04. Public compilation/model-data scans and all online evaluation happen in Rust or reviewed native accelerator libraries called from Rust. Python selects immutable parameters and displays public results. It does not implement a timed alternative or silently repair unsupported native execution.

## Required fixture and benchmark specification

Best uniform backend versus legal regional selection; include conversion, peak memory and preparation rather than only kernel time.

Mandatory paper-specific gates:

- Uniform baseline versus hybrid selection
- Forbidden topology/leakage changes
- Measured versus predicted whole-region cost

Each fixture locks input/output representations, ring/field parameters, numeric graph, party count and material lifetime. Compare an optimized implementation with identical functionality and applicable privacy assumptions; record original-artifact numbers and adaptations separately. A source-advertised speedup is not an expected threshold for different hardware.

## Privacy, correctness and proof obligations

Planner preservation of types/effects and composition preconditions; selected primitives still need their own security proofs.

A cost optimizer cannot trade away a privacy contract or numeric semantics. Type-correct composition is not an automatic security theorem.

Provide the corrupted party's permitted view, known plaintext/public inputs, randomness model, allowed leakage and attack budget. Formal or empirical results have explicit scope. Passing functional tests cannot grant a production privacy badge. A changed hash, field, layout, actor role or precision creates a documented adaptation obligation.

## What PLLM already has

Prior architecture cites HyCC; original artifact not run. Canonical DOI verified via author-source metadata; full text must be locked.

The original experiments and limitations are under `legacy/`. This handoff adds contracts and research tasks, **not a completed native reproduction of this paper**.

## Reproduction gates

`R09.acquire → R09.specify → R09.reference → R09.native → R09.assure → R09.benchmark → R09.document`.

The public research backlog records the remaining implementation and validation work. Native integration requires the actual PLLM command, source locks, raw measurements, and assurance report.

## Developer-facing explainer requirement

The eventual method page needs a worked operator example, a role diagram in prose or graphics, source section links, why/when to use it, configuration parameters, unsupported cases, numeric envelope, failure behavior and source-vs-PLLM differences. Generate tables from evidence rather than copying old performance ratios. Show `not implemented`, `adapted`, `reference only` or `reproduced at scope X` honestly.

## Default policy

`eligible_default = false` until implementation, full coverage, source/security review and deployment gates are met. The track restriction remains binding even after successful reproduction. Experimental or comparison methods cannot silently enter the preferred single-evaluator/no-HE profile.
