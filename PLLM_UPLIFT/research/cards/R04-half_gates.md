# R04 · Two Halves Make a Whole: Reducing Data Transfer in Garbled Circuits using Half Gates

**Priority 4 · 2015 · single_evaluator · source checked 2026-09-14**

Authors: Samee Zahur, Mike Rosulek, David Evans.  
Primary source: https://eprint.iacr.org/2014/756  
Access in this handoff: `primary_abstract`. Full source/artifact content hashes are not yet locked. A full-text acquisition gate is required.

## What the source contributes

Half-gates reduce AND-gate communication while remaining compatible with free XOR.

This short source summary is separate from the proposed PLLM design below. The source has not been reproduced merely by adding this card.

## Native implementation scope

Implement Boolean label ownership, free XOR and half-gates using a precisely documented reviewed hash instantiation. Batch native cryptographic operations and expose circuit I/O conversions.

Data flow: **Two Boolean input labels → Boolean AND output; free XOR and two encrypted half-gate records.**

Register the following independently versioned native components:

- `garble.boolean`
- `kernel.half_gates`
- `kernel.free_xor`

Dependencies: None; foundational work item. Public compilation/model-data scans and all online evaluation happen in Rust or reviewed native accelerator libraries called from Rust. Python selects immutable parameters and displays public results. It does not implement a timed alternative or silently repair unsupported native execution.

## Required fixture and benchmark specification

All input truth assignments, independent original-artifact differential cases, two-ciphertext accounting and native AES/SIMD benchmarks.

Mandatory paper-specific gates:

- All four truth assignments
- Independent garbler/evaluator reference parity
- Domain-separation collision and stale gate IDs

Each fixture locks input/output representations, ring/field parameters, numeric graph, party count and material lifetime. Compare an optimized implementation with identical functionality and applicable privacy assumptions; record original-artifact numbers and adaptations separately. A source-advertised speedup is not an expected threshold for different hardware.

## Privacy, correctness and proof obligations

Applicable half-gate/free-XOR security game; malformed/corrupted garbler behavior not covered by semi-honest assumptions.

Free-XOR-compatible correlation assumptions, domain separation and circuit freshness are mandatory. Malicious garbler defenses are a separate construction.

Provide the corrupted party's permitted view, known plaintext/public inputs, randomness model, allowed leakage and attack budget. Formal or empirical results have explicit scope. Passing functional tests cannot grant a production privacy badge. A changed hash, field, layout, actor role or precision creates a documented adaptation obligation.

## What PLLM already has

Python SHA-based half-gates in tenx references are adaptations; not a native audited implementation.

The original experiments and limitations are under `legacy/`. This handoff adds contracts and research tasks, **not a completed native reproduction of this paper**.

## Reproduction gates

`R04.acquire → R04.specify → R04.reference → R04.native → R04.assure → R04.benchmark → R04.document`.

The machine-readable recipe at `research/recipes/R04.json` is a work specification, not a pretend executable paper command. During acquisition, store the exact original artifact invocation and commit; leave them unclaimed until obtained. During native integration, attach the actual PLLM command, all environment/source locks, raw measurements and assurance report.

## Developer-facing explainer requirement

The eventual method page needs a worked operator example, a role diagram in prose or graphics, source section links, why/when to use it, configuration parameters, unsupported cases, numeric envelope, failure behavior and source-vs-PLLM differences. Generate tables from evidence rather than copying old performance ratios. Show `not implemented`, `adapted`, `reference only` or `reproduced at scope X` honestly.

## Default policy

`eligible_default = false` until implementation, full coverage, source/security review and deployment gates are met. The track restriction remains binding even after successful reproduction. Experimental or comparison methods cannot silently enter the preferred single-evaluator/no-HE profile.
