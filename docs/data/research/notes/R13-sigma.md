# R13 · SIGMA: Secure GPT Inference with Function Secret Sharing

**Priority 13 · 2024 · two_online_comparison · source checked 2026-09-14**

Authors: Kanav Gupta, Neha Jawalkar, Ananta Mukherjee, Nishanth Chandran, Divya Gupta, Ashish Panwar, Rahul Sharma.  
Primary source: https://petsymposium.org/popets/2024/popets-2024-0107.php  
Access in this handoff: `primary_abstract`. Full source/artifact content hashes are not yet locked. A full-text acquisition gate is required.

## What the source contributes

SIGMA provides FSS-based secure transformer operations and GPU-backed inference with preprocessing.

This short source summary is separate from the proposed PLLM design below. The source has not been reproduced merely by adding this card.

## Native implementation scope

Reproduce the original FSS helpers and complete reference workload. Register softmax, activation, normalization and truncation independently with exact phase contracts.

Data flow: **Two online share-holding roles + dealer keys → secure transformer helpers and source workload.**

Register the following independently versioned native components:

- `protocol.fss`
- `gate.fss_nonlinear`
- `prepare.fss`

Dependencies: None; foundational work item. Public compilation/model-data scans and all online evaluation happen in Rust or reviewed native accelerator libraries called from Rust. Python selects immutable parameters and displays public results. It does not implement a timed alternative or silently repair unsupported native execution.

## Required fixture and benchmark specification

Original model/token accounting first; sustained autoregressive decode is a separate workload, not inferred from one forward-pass time.

Mandatory paper-specific gates:

- FSS key freshness and widths
- Secure truncation and private attention
- Two-party rounds and fresh key consumption

Each fixture locks input/output representations, ring/field parameters, numeric graph, party count and material lifetime. Compare an optimized implementation with identical functionality and applicable privacy assumptions; record original-artifact numbers and adaptations separately. A source-advertised speedup is not an expected threshold for different hardware.

## Privacy, correctness and proof obligations

Dealer/corruption-set proof and specific helper composition, not a single-evaluator claim.

Two online parties and the dealer model must remain explicit. No implicit substitution into the one-evaluator default.

Provide the corrupted party's permitted view, known plaintext/public inputs, randomness model, allowed leakage and attack budget. Formal or empirical results have explicit scope. Passing functional tests cannot grant a production privacy badge. A changed hash, field, layout, actor role or precision creates a documented adaptation obligation.

## What PLLM already has

Resident-share and spectral archives are different constructions, not SIGMA reproductions.

The original experiments and limitations are under `legacy/`. This handoff adds contracts and research tasks, **not a completed native reproduction of this paper**.

## Reproduction gates

`R13.acquire → R13.specify → R13.reference → R13.native → R13.assure → R13.benchmark → R13.document`.

The public research backlog records the remaining implementation and validation work. Native integration requires the actual PLLM command, source locks, raw measurements, and assurance report.

## Developer-facing explainer requirement

The eventual method page needs a worked operator example, a role diagram in prose or graphics, source section links, why/when to use it, configuration parameters, unsupported cases, numeric envelope, failure behavior and source-vs-PLLM differences. Generate tables from evidence rather than copying old performance ratios. Show `not implemented`, `adapted`, `reference only` or `reproduced at scope X` honestly.

## Default policy

`eligible_default = false` until implementation, full coverage, source/security review and deployment gates are met. The track restriction remains binding even after successful reproduction. Experimental or comparison methods cannot silently enter the preferred single-evaluator/no-HE profile.
