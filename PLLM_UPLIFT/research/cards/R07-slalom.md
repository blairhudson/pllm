# R07 · Slalom: Fast, Verifiable and Private Execution of Neural Networks in Trusted Hardware

**Priority 7 · 2019 · client_heavy_baseline · source checked 2026-09-14**

Authors: Florian Tramèr, Dan Boneh.  
Primary source: https://arxiv.org/abs/1806.03287  
Access in this handoff: `primary_abstract`. Full source/artifact content hashes are not yet locked. A full-text acquisition gate is required.

## What the source contributes

Slalom outsources masked linear work and uses local verification within a trusted execution partition.

This short source summary is separate from the proposed PLLM design below. The source has not been reproduced merely by adding this card.

## Native implementation scope

Reproduce the original blinding and verification boundary. Register PLLM trusted-client/dealer adaptation separately; port modular kernels and verification state without changing the numerical graph silently.

Data flow: **Client masked activation + installed correction → masked linear output; client verification where specified.**

Register the following independently versioned native components:

- `protocol.masked_linear`
- `verify.preprocessed_linear`

Dependencies: None; foundational work item. Public compilation/model-data scans and all online evaluation happen in Rust or reviewed native accelerator libraries called from Rust. Python selects immutable parameters and displays public results. It does not implement a timed alternative or silently repair unsupported native execution.

## Required fixture and benchmark specification

Original representative layers; archived PLLM partition; same clear integer graph, unmasked remote partition and optimized native model as separate baselines.

Mandatory paper-specific gates:

- Fresh and reused masks
- No-wrap bound and prime-field verification mismatch
- Trusted hardware original versus trusted-client adaptation

Each fixture locks input/output representations, ring/field parameters, numeric graph, party count and material lifetime. Compare an optimized implementation with identical functionality and applicable privacy assumptions; record original-artifact numbers and adaptations separately. A source-advertised speedup is not an expected threshold for different hardware.

## Privacy, correctness and proof obligations

Joint-view simulation, trusted component definition, bound-correct reconstruction and independent verification secrets.

A trusted client is not a reproduced TEE deployment. Static non-collusion, integrity and later compromise require distinct claims.

Provide the corrupted party's permitted view, known plaintext/public inputs, randomness model, allowed leakage and attack budget. Formal or empirical results have explicit scope. Passing functional tests cannot grant a production privacy badge. A changed hash, field, layout, actor role or precision creates a documented adaptation obligation.

## What PLLM already has

Original PLLM manuscript reports nine Qwen runs; this archive is not a fresh reproduction of Slalom.

The original experiments and limitations are under `legacy/`. This handoff adds contracts and research tasks, **not a completed native reproduction of this paper**.

## Reproduction gates

`R07.acquire → R07.specify → R07.reference → R07.native → R07.assure → R07.benchmark → R07.document`.

The machine-readable recipe at `research/recipes/R07.json` is a work specification, not a pretend executable paper command. During acquisition, store the exact original artifact invocation and commit; leave them unclaimed until obtained. During native integration, attach the actual PLLM command, all environment/source locks, raw measurements and assurance report.

## Developer-facing explainer requirement

The eventual method page needs a worked operator example, a role diagram in prose or graphics, source section links, why/when to use it, configuration parameters, unsupported cases, numeric envelope, failure behavior and source-vs-PLLM differences. Generate tables from evidence rather than copying old performance ratios. Show `not implemented`, `adapted`, `reference only` or `reproduced at scope X` honestly.

## Default policy

`eligible_default = false` until implementation, full coverage, source/security review and deployment gates are met. The track restriction remains binding even after successful reproduction. Experimental or comparison methods cannot silently enter the preferred single-evaluator/no-HE profile.
