# R19 · Private and Verifiable Outsourcing of Open-Weight LLM Inference

**Priority 19 · 2026 · two_online_comparison · source checked 2026-09-14**

Authors: Kanav Gupta, Jonathan Katz, Ian Miers.  
Primary source: https://eprint.iacr.org/2026/1849  
Access in this handoff: `primary_abstract`. Full source/artifact content hashes are not yet locked. A full-text acquisition gate is required.

## What the source contributes

The abstract describes private and verifiable open-weight inference with two malicious, non-colluding servers and model-owner-published information.

This short source summary is separate from the proposed PLLM design below. The source has not been reproduced merely by adding this card.

## Native implementation scope

Acquire the complete protocol and original artifact first. Create a faithful isolated adapter with a documented client, model-owner, server and proof boundary.

Data flow: **Model-owner setup + client + two malicious non-colluding servers, according to acquired full source.**

Register the following independently versioned native components:

- `protocol.open_weight_verified`
- `verify.model_owner_setup`

Dependencies: None; foundational work item. Public compilation/model-data scans and all online evaluation happen in Rust or reviewed native accelerator libraries called from Rust. Python selects immutable parameters and displays public results. It does not implement a timed alternative or silently repair unsupported native execution.

## Required fixture and benchmark specification

Original workload versus true autoregressive decoding, setup/model-owner cost, online server multiplicity and complete verification costs.

Mandatory paper-specific gates:

- Source acquisition and artifact lock first
- Original reported workload recreated
- Sustained decode distinguished from a forward-pass throughput claim

Each fixture locks input/output representations, ring/field parameters, numeric graph, party count and material lifetime. Compare an optimized implementation with identical functionality and applicable privacy assumptions; record original-artifact numbers and adaptations separately. A source-advertised speedup is not an expected threshold for different hardware.

## Privacy, correctness and proof obligations

Complete malicious-security/model-owner setup theorem and both server roles; abstract alone is insufficient to implement or inherit it.

Abstract claims alone do not specify a reproducible protocol or near-native sustained TPS. This remains source-acquisition-gated.

Provide the corrupted party's permitted view, known plaintext/public inputs, randomness model, allowed leakage and attack budget. Formal or empirical results have explicit scope. Passing functional tests cannot grant a production privacy badge. A changed hash, field, layout, actor role or precision creates a documented adaptation obligation.

## What PLLM already has

Only abstract-level comparison exists in the supplied work.

The original experiments and limitations are under `legacy/`. This handoff adds contracts and research tasks, **not a completed native reproduction of this paper**.

## Reproduction gates

`R19.acquire → R19.specify → R19.reference → R19.native → R19.assure → R19.benchmark → R19.document`.

The machine-readable recipe at `research/recipes/R19.json` is a work specification, not a pretend executable paper command. During acquisition, store the exact original artifact invocation and commit; leave them unclaimed until obtained. During native integration, attach the actual PLLM command, all environment/source locks, raw measurements and assurance report.

## Developer-facing explainer requirement

The eventual method page needs a worked operator example, a role diagram in prose or graphics, source section links, why/when to use it, configuration parameters, unsupported cases, numeric envelope, failure behavior and source-vs-PLLM differences. Generate tables from evidence rather than copying old performance ratios. Show `not implemented`, `adapted`, `reference only` or `reproduced at scope X` honestly.

## Default policy

`eligible_default = false` until implementation, full coverage, source/security review and deployment gates are met. The track restriction remains binding even after successful reproduction. Experimental or comparison methods cannot silently enter the preferred single-evaluator/no-HE profile.
