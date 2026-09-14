# R14 · FuseFSS: Efficient Secure LLM Inference with Function Secret Sharing

**Priority 14 · 2026 · two_online_comparison · source checked 2026-09-14**

Authors: Yuhan Ma, Yong Li, Stefan Schmid.  
Primary source: https://arxiv.org/abs/2606.09551  
Access in this handoff: `primary_abstract`. Full source/artifact content hashes are not yet locked. A full-text acquisition gate is required.

## What the source contributes

FuseFSS targets fusion and efficiency of secure nonlinear/helper operations in FSS-based LLM inference.

This short source summary is separate from the proposed PLLM design below. The source has not been reproduced merely by adding this card.

## Native implementation scope

Acquire and pin the full source and artifact; port complete fused helpers with their offline key generation, remaining secure multiplication and rescaling.

Data flow: **Published fused FSS helpers with explicit remaining polynomial products and rescaling.**

Register the following independently versioned native components:

- `gate.fss_fused`
- `compiler.fss_fusion`

Dependencies: R13. Public compilation/model-data scans and all online evaluation happen in Rust or reviewed native accelerator libraries called from Rust. Python selects immutable parameters and displays public results. It does not implement a timed alternative or silently repair unsupported native execution.

## Required fixture and benchmark specification

Matched SIGMA functions, bit widths, hardware and token workloads; count every online dependency and fresh key byte.

Mandatory paper-specific gates:

- Exact SIGMA-matched operator semantics
- Remaining dependency rounds
- Fused-key footprint and expanded local state

Each fixture locks input/output representations, ring/field parameters, numeric graph, party count and material lifetime. Compare an optimized implementation with identical functionality and applicable privacy assumptions; record original-artifact numbers and adaptations separately. A source-advertised speedup is not an expected threshold for different hardware.

## Privacy, correctness and proof obligations

Proof for fused operators and source adversary model; scalar numerical matches do not establish protected evaluation.

A fused key lookup does not make subsequent private polynomial evaluation free. Public model support does not eliminate the second online role.

Provide the corrupted party's permitted view, known plaintext/public inputs, randomness model, allowed leakage and attack budget. Formal or empirical results have explicit scope. Passing functional tests cannot grant a production privacy badge. A changed hash, field, layout, actor role or precision creates a documented adaptation obligation.

## What PLLM already has

Prior reference fused-polynomial experiments are alternative methods, not a FuseFSS implementation.

The original experiments and limitations are under `legacy/`. This handoff adds contracts and research tasks, **not a completed native reproduction of this paper**.

## Reproduction gates

`R14.acquire → R14.specify → R14.reference → R14.native → R14.assure → R14.benchmark → R14.document`.

The machine-readable recipe at `research/recipes/R14.json` is a work specification, not a pretend executable paper command. During acquisition, store the exact original artifact invocation and commit; leave them unclaimed until obtained. During native integration, attach the actual PLLM command, all environment/source locks, raw measurements and assurance report.

## Developer-facing explainer requirement

The eventual method page needs a worked operator example, a role diagram in prose or graphics, source section links, why/when to use it, configuration parameters, unsupported cases, numeric envelope, failure behavior and source-vs-PLLM differences. Generate tables from evidence rather than copying old performance ratios. Show `not implemented`, `adapted`, `reference only` or `reproduced at scope X` honestly.

## Default policy

`eligible_default = false` until implementation, full coverage, source/security review and deployment gates are met. The track restriction remains binding even after successful reproduction. Experimental or comparison methods cannot silently enter the preferred single-evaluator/no-HE profile.
