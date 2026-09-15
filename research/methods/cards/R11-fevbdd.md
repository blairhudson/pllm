# R11 · Factored Edge-Valued Binary Decision Diagrams

**Priority 11 · 1997 · public_compilation · source checked 2026-09-14**

Authors: Paul Tafertshofer, Massoud Pedram.  
Primary source: https://link.springer.com/article/10.1023/A:1008691605584  
Access in this handoff: `primary_abstract`. Full source/artifact content hashes are not yet locked. A full-text acquisition gate is required.

## What the source contributes

Factored edge-valued diagrams share subfunctions through additive and multiplicative edge normalization.

This short source summary is separate from the proposed PLLM design below. The source has not been reproduced merely by adding this card.

## Native implementation scope

Port public EVBDD/FEVBDD compilation, canonical residual normalization and variable-order search to Rust. Keep encrypted traversal as a separate experimental method.

Data flow: **Public integer LUT → exact public normalized diagram and ordering; encrypted traversal is NOT part of this paper.**

Register the following independently versioned native components:

- `compiler.weighted_decision_diagram`
- `compiler.variable_order`

Dependencies: None; foundational work item. Public compilation/model-data scans and all online evaluation happen in Rust or reviewed native accelerator libraries called from Rust. Python selects immutable parameters and displays public results. It does not implement a timed alternative or silently repair unsupported native execution.

## Required fixture and benchmark specification

Exhaustive fixed integer LUT equivalence, graph hashes, node counts and compile-memory measurements; then separately benchmark protected execution.

Mandatory paper-specific gates:

- Exhaustive LUT equivalence
- Canonical additive/multiplicative normalization
- Node/memory/compile-time trade-off across bit orders

Each fixture locks input/output representations, ring/field parameters, numeric graph, party count and material lifetime. Compare an optimized implementation with identical functionality and applicable privacy assumptions; record original-artifact numbers and adaptations separately. A source-advertised speedup is not an expected threshold for different hardware.

## Privacy, correctness and proof obligations

Public deterministic compiler correctness only. No privacy theorem or security level is applicable to diagram compression.

This is function compression, not a privacy protocol. No security claim follows from a smaller public graph.

Provide the corrupted party's permitted view, known plaintext/public inputs, randomness model, allowed leakage and attack budget. Formal or empirical results have explicit scope. Passing functional tests cannot grant a production privacy badge. A changed hash, field, layout, actor role or precision creates a documented adaptation obligation.

## What PLLM already has

Prior tenx archive contains exact public diagram experiments and a custom encrypted extension.

The original experiments and limitations are under `legacy/`. This handoff adds contracts and research tasks, **not a completed native reproduction of this paper**.

## Reproduction gates

`R11.acquire → R11.specify → R11.reference → R11.native → R11.assure → R11.benchmark → R11.document`.

The machine-readable recipe at `research/recipes/R11.json` is a work specification, not a pretend executable paper command. During acquisition, store the exact original artifact invocation and commit; leave them unclaimed until obtained. During native integration, attach the actual PLLM command, all environment/source locks, raw measurements and assurance report.

## Developer-facing explainer requirement

The eventual method page needs a worked operator example, a role diagram in prose or graphics, source section links, why/when to use it, configuration parameters, unsupported cases, numeric envelope, failure behavior and source-vs-PLLM differences. Generate tables from evidence rather than copying old performance ratios. Show `not implemented`, `adapted`, `reference only` or `reproduced at scope X` honestly.

## Default policy

`eligible_default = false` until implementation, full coverage, source/security review and deployment gates are met. The track restriction remains binding even after successful reproduction. Experimental or comparison methods cannot silently enter the preferred single-evaluator/no-HE profile.
