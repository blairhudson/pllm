# R15 · Efficient Pseudorandom Correlation Generators over Z/p^k Z

**Priority 15 · 2025 · preparation_comparison · source checked 2026-09-14**

Authors: Zhe Li, Chaoping Xing, Yizhou Yao, Chen Yuan.  
Primary source: https://link.springer.com/chapter/10.1007/978-3-032-01884-7_7  
Full 56-page primary paper is cached as `papers/r15-ring-pcg.pdf` and verified against the SHA-256 in `paper-library.json` (`130178519ec48cbbe64b201e42bb581b513f77503f671e1a56e4b0909b4aa375`). This is specification material; upstream code is not a PLLM dependency or a locked runtime artifact.

## What the source contributes

The paper's Z/2^kZ OLE and authenticated-triple PCGs require a Galois-ring extension, a Hensel-lifted primitive polynomial, its generalized trace and Frobenius maps, QA-SD/Ring-LPN assumptions, and per-party sparse-point FSS seeds (Sections 4–6). A plain ring PRG cannot replace that correlated construction. Section 7 and Appendix B warn that older QA-SD parameters `(c=3, t=27, m=2)` succumb to a newer attack; any selectable parameters must pass a current independent security review.

This short source summary is separate from the proposed PLLM design below. The source has not been reproduced merely by adding this card.

## Native implementation scope

Implement reviewed parameter sets and the actual requested correlation schema. Expose seed setup, expansion time, memory and transcript requirements; do not substitute ordinary PRG seeds.

Data flow: **Approved correlated-seed setup → per-role pseudorandom correlation schema, not shared unrestricted seeds.**

Register the following independently versioned native components:

- `prepare.ring_pcg`
- `correlation.ole`
- `correlation.authenticated_triple`

Dependencies: None; foundational work item. Public compilation/model-data scans and all online evaluation happen in Rust or reviewed native accelerator libraries called from Rust. Python selects immutable parameters and displays public results. It does not implement a timed alternative or silently repair unsupported native execution.

## Required fixture and benchmark specification

Correct correlation identities, original artifact outputs, communication versus expansion compute, and current attack-aware parameter checks.

Mandatory paper-specific gates:

- Correlation identities
- Modern parameter/attack review
- Expansion memory and CPU compared with material transmission

Each fixture locks input/output representations, ring/field parameters, numeric graph, party count and material lifetime. Compare an optimized implementation with identical functionality and applicable privacy assumptions; record original-artifact numbers and adaptations separately. A source-advertised speedup is not an expected threshold for different hardware.

## Privacy, correctness and proof obligations

Distribution and computational indistinguishability argument for the actual ring/PCG parameters; toy algebra is insufficient.

PCGs do not automatically compress arbitrary garbled tables or spectral coefficients. Small toy parameters are not deployment security parameters.

Provide the corrupted party's permitted view, known plaintext/public inputs, randomness model, allowed leakage and attack budget. Formal or empirical results have explicit scope. Passing functional tests cannot grant a production privacy badge. A changed hash, field, layout, actor role or precision creates a documented adaptation obligation.

## What PLLM already has

`pllm.runtime.ring_pcg_reference` now checks the paper's *four distinct sparse
cross-term distributions* using one independently issued additive point key per
term. Each test-only worker holds its own sparse terms and point keys and
expands them once; tests reconstruct all Galois-ring OLE identities in a tiny
`GR(2^k,2)[X_1,...]/(X_i^3−1)` example for 8- and 64-bit rings. A separate
bounded trace/Frobenius path issues the *additional four* cross-term families
and checks both base-ring `Z/2^k` OLE lanes per CRT point against the paper's
extraction identity. The
session-bound issuer rejects excessive key material before generating keys.
This validates the algebra and share distribution, **not a secure or silent
PCG**: the small QA-SD instances have no hardness, the dealer sees both
inputs, expansion uses a bounded quadratic reference instead of the paper's
quasilinear algorithm, and setup/transport/modern attack parameters remain
unimplemented. No runtime component is admitted.

The original experiments and limitations are under `legacy/`. This handoff adds contracts and research tasks, **not a completed native reproduction of this paper**.

## Reproduction gates

`R15.acquire → R15.specify → R15.reference → R15.native → R15.assure → R15.benchmark → R15.document`.

The public research backlog records the remaining implementation and validation work. Native integration requires the actual PLLM command, source locks, raw measurements, and assurance report.

## Developer-facing explainer requirement

The eventual method page needs a worked operator example, a role diagram in prose or graphics, source section links, why/when to use it, configuration parameters, unsupported cases, numeric envelope, failure behavior and source-vs-PLLM differences. Generate tables from evidence rather than copying old performance ratios. Show `not implemented`, `adapted`, `reference only` or `reproduced at scope X` honestly.

## Default policy

`eligible_default = false` until implementation, full coverage, source/security review and deployment gates are met. The track restriction remains binding even after successful reproduction. Experimental or comparison methods cannot silently enter the preferred single-evaluator/no-HE profile.
