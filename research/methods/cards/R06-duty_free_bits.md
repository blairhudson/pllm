# R06 · Duty-Free Bits: Projectivizing Garbling Schemes

**Priority 6 · 2026 · single_evaluator · source checked 2026-09-14**

Authors: Nakul Khambhati, Anwesh Bhattacharya, David Heath.  
Primary source: https://eprint.iacr.org/2026/476  
Access in this handoff: `primary_abstract`. Full source/artifact content hashes are not yet locked. A full-text acquisition gate is required.

## What the source contributes

This work studies projective conversion from Yao-style bit labels into affine encodings and related vector-OLE applications.

This short source summary is separate from the proposed PLLM design below. The source has not been reproduced merely by adding this card.

## Native implementation scope

Implement only conversions justified by the acquired full specification. Represent input/output label families, field, security assumptions and direction as types.

Data flow: **Source-authorized bit-label → affine-label direction, with exact field and projective encoding.**

Register the following independently versioned native components:

- `convert.bit_to_affine`
- `prepare.vector_ole`

Dependencies: R03, R04. Public compilation/model-data scans and all online evaluation happen in Rust or reviewed native accelerator libraries called from Rust. Python selects immutable parameters and displays public results. It does not implement a timed alternative or silently repair unsupported native execution.

## Required fixture and benchmark specification

Published conversion test vectors and complete Boolean-to-arithmetic regions; compare widths and include the reverse path only if separately implemented.

Mandatory paper-specific gates:

- Published conversion examples
- Illegal reverse cast rejected
- Full conversion cost compared with a generic bridge

Each fixture locks input/output representations, ring/field parameters, numeric graph, party count and material lifetime. Compare an optimized implementation with identical functionality and applicable privacy assumptions; record original-artifact numbers and adaptations separately. A source-advertised speedup is not an expected threshold for different hardware.

## Privacy, correctness and proof obligations

Projectivity and correlation assumptions of this particular direction; a custom reverse path has a separate proof.

The title does not grant free bidirectional conversion or a safe interface from freely editable masked scalars.

Provide the corrupted party's permitted view, known plaintext/public inputs, randomness model, allowed leakage and attack budget. Formal or empirical results have explicit scope. Passing functional tests cannot grant a production privacy badge. A changed hash, field, layout, actor role or precision creates a documented adaptation obligation.

## What PLLM already has

Primary abstract consulted; full construction acquisition and review are blocking tasks.

The original experiments and limitations are under `legacy/`. This handoff adds contracts and research tasks, **not a completed native reproduction of this paper**.

## Reproduction gates

`R06.acquire → R06.specify → R06.reference → R06.native → R06.assure → R06.benchmark → R06.document`.

The machine-readable recipe at `research/recipes/R06.json` is a work specification, not a pretend executable paper command. During acquisition, store the exact original artifact invocation and commit; leave them unclaimed until obtained. During native integration, attach the actual PLLM command, all environment/source locks, raw measurements and assurance report.

## Developer-facing explainer requirement

The eventual method page needs a worked operator example, a role diagram in prose or graphics, source section links, why/when to use it, configuration parameters, unsupported cases, numeric envelope, failure behavior and source-vs-PLLM differences. Generate tables from evidence rather than copying old performance ratios. Show `not implemented`, `adapted`, `reference only` or `reproduced at scope X` honestly.

## Default policy

`eligible_default = false` until implementation, full coverage, source/security review and deployment gates are met. The track restriction remains binding even after successful reproduction. Experimental or comparison methods cannot silently enter the preferred single-evaluator/no-HE profile.
