# R16 · Practical Secure Delegated Linear Algebra with Trapdoored Matrices

**Priority 16 · 2025 · linear_comparison · source checked 2026-09-14**

Authors: Mark Braverman, Stephen Newman.  
Primary source: https://arxiv.org/html/2502.13060v3  
Access in this handoff: `primary_full_text_html`. Full source/artifact content hashes are not yet locked. A full-text acquisition gate is not required for reading, but exact paper and original artifact pinning remain required.

## What the source contributes

This work uses trapdoored pseudorandom matrices for efficient delegated linear algebra under an LPN-based approach.

This short source summary is separate from the proposed PLLM design below. The source has not been reproduced merely by adding this card.

## Native implementation scope

Implement the paper-specific matrix/trapdoor generation and delegation protocol before attempting a setup adaptation. State who knows each operand and each trapdoor.

Data flow: **Source operand ownership + trapdoored matrix generation → delegated linear operation.**

Register the following independently versioned native components:

- `prepare.trapdoored_matrix`
- `protocol.delegated_linear`

Dependencies: None; foundational work item. Public compilation/model-data scans and all online evaluation happen in Rust or reviewed native accelerator libraries called from Rust. Python selects immutable parameters and displays public results. It does not implement a timed alternative or silently repair unsupported native execution.

## Required fixture and benchmark specification

Original linear workloads, malicious-result handling where specified, full client/offline work and public-model-adaptation comparison.

Mandatory paper-specific gates:

- Source-specific honest/malicious cases
- Public low-rank mask attack control
- Full client and trapdoor setup cost

Each fixture locks input/output representations, ring/field parameters, numeric graph, party count and material lifetime. Compare an optimized implementation with identical functionality and applicable privacy assumptions; record original-artifact numbers and adaptations separately. A source-advertised speedup is not an expected threshold for different hardware.

## Privacy, correctness and proof obligations

LPN/trapdoor assumptions and operand ownership; a new OLE/garbling-setup use requires its own reduction.

Public low-rank masks leak nullspace projections. This construction must not be approximated by an unproven structured mask.

Provide the corrupted party's permitted view, known plaintext/public inputs, randomness model, allowed leakage and attack budget. Formal or empirical results have explicit scope. Passing functional tests cannot grant a production privacy badge. A changed hash, field, layout, actor role or precision creates a documented adaptation obligation.

## What PLLM already has

Identified as a lead previously; not executed in the supplied archives.

The original experiments and limitations are under `legacy/`. This handoff adds contracts and research tasks, **not a completed native reproduction of this paper**.

## Reproduction gates

`R16.acquire → R16.specify → R16.reference → R16.native → R16.assure → R16.benchmark → R16.document`.

The machine-readable recipe at `research/recipes/R16.json` is a work specification, not a pretend executable paper command. During acquisition, store the exact original artifact invocation and commit; leave them unclaimed until obtained. During native integration, attach the actual PLLM command, all environment/source locks, raw measurements and assurance report.

## Developer-facing explainer requirement

The eventual method page needs a worked operator example, a role diagram in prose or graphics, source section links, why/when to use it, configuration parameters, unsupported cases, numeric envelope, failure behavior and source-vs-PLLM differences. Generate tables from evidence rather than copying old performance ratios. Show `not implemented`, `adapted`, `reference only` or `reproduced at scope X` honestly.

## Default policy

`eligible_default = false` until implementation, full coverage, source/security review and deployment gates are met. The track restriction remains binding even after successful reproduction. Experimental or comparison methods cannot silently enter the preferred single-evaluator/no-HE profile.
