# R20 · Oblivious Ciphertext Compression via Linear Codes

**Priority 20 · 2026 · representation_specific · source checked 2026-09-14**

Authors: Pascal Giorgi, Bruno Grenet, Mark Simkin.  
Primary source: https://eprint.iacr.org/2026/329  
Access in this handoff: `primary_abstract`. Full source/artifact content hashes are not yet locked. A full-text acquisition gate is required.

## What the source contributes

This work develops oblivious ciphertext compression using linear codes and syndrome decoding.

This short source summary is separate from the proposed PLLM design below. The source has not been reproduced merely by adding this card.

## Native implementation scope

Reproduce the exact source construction and its decoding-side-information assumptions. Bind codecs to representations and valid domains; compare the PLLM syndrome design as an adaptation.

Data flow: **Source ciphertext/side-information representation → compressed form with proven restricted decoding domain.**

Register the following independently versioned native components:

- `codec.oblivious_linear_code`
- `codec.domain_certificate`

Dependencies: None; foundational work item. Public compilation/model-data scans and all online evaluation happen in Rust or reviewed native accelerator libraries called from Rust. Python selects immutable parameters and displays public results. It does not implement a timed alternative or silently repair unsupported native execution.

## Required fixture and benchmark specification

Original correctness domain, payload and encoder/decoder cost; public-bound packing and uncompressed matched controls; explicit out-of-domain collision cases.

Mandatory paper-specific gates:

- Original sparse-domain correctness
- Uncompressed and public-bound controls
- Out-of-domain collision and adaptive fallback leakage

Each fixture locks input/output representations, ring/field parameters, numeric graph, party count and material lifetime. Compare an optimized implementation with identical functionality and applicable privacy assumptions; record original-artifact numbers and adaptations separately. A source-advertised speedup is not an expected threshold for different hardware.

## Privacy, correctness and proof obligations

Correctness/privacy for the original encrypted setting; PLLM ring-side-information compression is a separate adaptation.

Lossless compression is conditional on its domain and side information. Successful decode is not proof of correct inference, and adaptive retries can leak.

Provide the corrupted party's permitted view, known plaintext/public inputs, randomness model, allowed leakage and attack budget. Formal or empirical results have explicit scope. Passing functional tests cannot grant a production privacy badge. A changed hash, field, layout, actor role or precision creates a documented adaptation obligation.

## What PLLM already has

Network archives contain a custom bounded-overflow codec, not this paper's implementation.

The original experiments and limitations are under `legacy/`. This handoff adds contracts and research tasks, **not a completed native reproduction of this paper**.

## Reproduction gates

`R20.acquire → R20.specify → R20.reference → R20.native → R20.assure → R20.benchmark → R20.document`.

The machine-readable recipe at `research/recipes/R20.json` is a work specification, not a pretend executable paper command. During acquisition, store the exact original artifact invocation and commit; leave them unclaimed until obtained. During native integration, attach the actual PLLM command, all environment/source locks, raw measurements and assurance report.

## Developer-facing explainer requirement

The eventual method page needs a worked operator example, a role diagram in prose or graphics, source section links, why/when to use it, configuration parameters, unsupported cases, numeric envelope, failure behavior and source-vs-PLLM differences. Generate tables from evidence rather than copying old performance ratios. Show `not implemented`, `adapted`, `reference only` or `reproduced at scope X` honestly.

## Default policy

`eligible_default = false` until implementation, full coverage, source/security review and deployment gates are met. The track restriction remains binding even after successful reproduction. Experimental or comparison methods cannot silently enter the preferred single-evaluator/no-HE profile.
