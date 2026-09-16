# R05 · Garbled Circuit Lookup Tables with Logarithmic Number of Ciphertexts

**Priority 5 · 2024 · single_evaluator · source checked 2026-09-14**

Authors: David Heath, Vladimir Kolesnikov, Lucien K. L. Ng.  
Primary source: https://eprint.iacr.org/2024/369  
Access in this handoff: `primary_abstract`. Full source/artifact content hashes are not yet locked. A full-text acquisition gate is required.

## What the source contributes

The lookup construction reduces the number of security-parameter-sized ciphertexts; total communication retains a table-data term.

This short source summary is separate from the proposed PLLM design below. The source has not been reproduced merely by adding this card.

## Native implementation scope

Reproduce the original lookup construction, not its headline complexity. Provide Boolean input/output contracts and explicit arithmetic conversion edges.

Data flow: **Boolean index labels + prepared lookup → Boolean output labels; explicit arithmetic conversions around it.**

Register the following independently versioned native components:

- `gate.logrow`
- `convert.lookup_io`

Dependencies: R04. Public compilation/model-data scans and all online evaluation happen in Rust or reviewed native accelerator libraries called from Rust. Python selects immutable parameters and displays public results. It does not implement a timed alternative or silently repair unsupported native execution.

## Required fixture and benchmark specification

Matched 8/12/16-bit tables versus flat, Boolean and weighted-path variants. Include the table-data term and both conversion directions.

Mandatory paper-specific gates:

- Small exhaustive tables
- Serialized table-data term versus ciphertext count
- Secret-index queries, malformed labels and repeated use

Each fixture locks input/output representations, ring/field parameters, numeric graph, party count and material lifetime. Compare an optimized implementation with identical functionality and applicable privacy assumptions; record original-artifact numbers and adaptations separately. A source-advertised speedup is not an expected threshold for different hardware.

## Privacy, correctness and proof obligations

Lookup privacy and input-binding argument, including arithmetic bridge obligations and observable access behavior.

Do not translate logarithmic ciphertext count into logarithmic total bytes or claim our flat-table improvements beat this unimplemented baseline.

Provide the corrupted party's permitted view, known plaintext/public inputs, randomness model, allowed leakage and attack budget. Formal or empirical results have explicit scope. Passing functional tests cannot grant a production privacy badge. A changed hash, field, layout, actor role or precision creates a documented adaptation obligation.

## What PLLM already has

Analytical formulas only in previous work; no original or PLLM logrow execution.

The original experiments and limitations are under `legacy/`. This handoff adds contracts and research tasks, **not a completed native reproduction of this paper**.

## Reproduction gates

`R05.acquire → R05.specify → R05.reference → R05.native → R05.assure → R05.benchmark → R05.document`.

The public research backlog records the remaining implementation and validation work. Native integration requires the actual PLLM command, source locks, raw measurements, and assurance report.

## Developer-facing explainer requirement

The eventual method page needs a worked operator example, a role diagram in prose or graphics, source section links, why/when to use it, configuration parameters, unsupported cases, numeric envelope, failure behavior and source-vs-PLLM differences. Generate tables from evidence rather than copying old performance ratios. Show `not implemented`, `adapted`, `reference only` or `reproduced at scope X` honestly.

## Default policy

`eligible_default = false` until implementation, full coverage, source/security review and deployment gates are met. The track restriction remains binding even after successful reproduction. Experimental or comparison methods cannot silently enter the preferred single-evaluator/no-HE profile.
