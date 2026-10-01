# R14 · FuseFSS: Efficient Secure LLM Inference with Function Secret Sharing

**Priority 14 · 2026 · two_online_comparison · source checked 2026-09-14**

Authors: Yuhan Ma, Yong Li, Stefan Schmid.  
Primary source: https://arxiv.org/abs/2606.09551  
Full 27-page primary paper is cached as `papers/r14-fusefss.pdf` and verified against the SHA-256 in `paper-library.json` (`6109399761ab0cc1ca6eb617fec1a811cbe59d474455ce3549467e5a5b77cbf0`). This is specification material; upstream code is not a PLLM dependency or a locked runtime artifact.

## What the source contributes

FuseFSS compiles typed scalar operators into a packed private comparison and a vector interval lookup on one public one-time-masked wire, followed by ordinary share-based multiplications and conversions. Its mask-independent public query and interval shapes are part of the leakage contract (Sections 3–5, Appendix F). It does not compile vector reductions or eliminate the dealer, keys, second online party, or remaining interactive rounds. The original paper compares against SIGMA's matching FSS topology and functions, not PLLM's clear baseline.

This short source summary is separate from the proposed PLLM design below. The source has not been reproduced merely by adding this card.

## Native implementation scope

Pin the full source (completed for the paper PDF); implement and compare against complete matching SIGMA helpers, including offline key generation, mask-independent shape padding, secure multiplication and rescaling.

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

The bounded `pllm.runtime.scalar_fss_reference` compiles a public two-interval,
8-bit affine scalar description into a fixed-shape private predicate call and
one vector-coefficient table lookup on the same masked input. One share per
worker executes an independently authenticated opening and Beaver product;
signed ReLU values and sign predicates pass full-domain source-side checks.
The dense table is an information-theoretic function sharing reference:
**4,096 table bytes per worker per scalar**, plus 306 point-key bytes for
one predicate, before transport and Beaver material. It cannot demonstrate
FuseFSS's compact lookup, fused production performance, polynomial/
fixed-point helpers, or whole-model coverage. The work remains an isolated
test-local reference, not a selectable `gate.fss_fused` component.

The original experiments and limitations are under `legacy/`. This reference
does **not** reproduce the paper's native two-call implementation or its
reported cost advantage.

## Reproduction gates

`R14.acquire → R14.specify → R14.reference → R14.native → R14.assure → R14.benchmark → R14.document`.

The public research backlog records the remaining implementation and validation work. Native integration requires the actual PLLM command, source locks, raw measurements, and assurance report.

## Developer-facing explainer requirement

The eventual method page needs a worked operator example, a role diagram in prose or graphics, source section links, why/when to use it, configuration parameters, unsupported cases, numeric envelope, failure behavior and source-vs-PLLM differences. Generate tables from evidence rather than copying old performance ratios. Show `not implemented`, `adapted`, `reference only` or `reproduced at scope X` honestly.

## Default policy

`eligible_default = false` until implementation, full coverage, source/security review and deployment gates are met. The track restriction remains binding even after successful reproduction. Experimental or comparison methods cannot silently enter the preferred single-evaluator/no-HE profile.
