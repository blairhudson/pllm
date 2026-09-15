# R17 · Curl: Private LLMs through Wavelet-Encoded Look-Up Tables

**Priority 17 · 2024 · mpc_comparison · source checked 2026-09-14**

Authors: Manuel B. Santos, Dimitris Mouris, Mehmet Ugurbil, Stanislaw Jarecki, José Reis, Shubho Sengupta, Miguel de Vega.  
Primary source: https://eprint.iacr.org/2024/1127  
Access in this handoff: `primary_abstract`. Full source/artifact content hashes are not yet locked. A full-text acquisition gate is required.

## What the source contributes

Curl uses wavelet-encoded lookup tables for private nonlinear operations and discusses probabilistic truncation security.

This short source summary is separate from the proposed PLLM design below. The source has not been reproduced merely by adding this card.

## Native implementation scope

Reproduce original lookup construction, encoding, scale and truncation behavior. Treat a wavelet approximation imported into garbling as a separate numerical/protocol adaptation.

Data flow: **Wavelet-encoded table + source protected access/truncation → nonlinear helper under declared precision.**

Register the following independently versioned native components:

- `gate.wavelet_lookup`
- `numeric.wavelet`
- `gate.probabilistic_truncation`

Dependencies: None; foundational work item. Public compilation/model-data scans and all online evaluation happen in Rust or reviewed native accelerator libraries called from Rust. Python selects immutable parameters and displays public results. It does not implement a timed alternative or silently repair unsupported native execution.

## Required fixture and benchmark specification

Log/reciprocal/activation functions at matched numeric accuracy, preprocessing and helper rounds, then original language-model fixtures.

Mandatory paper-specific gates:

- Wavelet approximation quality and tails
- Truncation distribution and security definition
- Private lookup work rather than public sparse coefficient count

Each fixture locks input/output representations, ring/field parameters, numeric graph, party count and material lifetime. Compare an optimized implementation with identical functionality and applicable privacy assumptions; record original-artifact numbers and adaptations separately. A source-advertised speedup is not an expected threshold for different hardware.

## Privacy, correctness and proof obligations

Source stand-alone/composition scope, private access and rounding randomness. Universally composable behavior is not inferred.

A stand-alone truncation argument does not imply universal composability. Sparse transform coefficients do not make private access free.

Provide the corrupted party's permitted view, known plaintext/public inputs, randomness model, allowed leakage and attack budget. Formal or empirical results have explicit scope. Passing functional tests cannot grant a production privacy badge. A changed hash, field, layout, actor role or precision creates a documented adaptation obligation.

## What PLLM already has

Prior report cited Curl; no Curl protocol was reproduced.

The original experiments and limitations are under `legacy/`. This handoff adds contracts and research tasks, **not a completed native reproduction of this paper**.

## Reproduction gates

`R17.acquire → R17.specify → R17.reference → R17.native → R17.assure → R17.benchmark → R17.document`.

The machine-readable recipe at `research/recipes/R17.json` is a work specification, not a pretend executable paper command. During acquisition, store the exact original artifact invocation and commit; leave them unclaimed until obtained. During native integration, attach the actual PLLM command, all environment/source locks, raw measurements and assurance report.

## Developer-facing explainer requirement

The eventual method page needs a worked operator example, a role diagram in prose or graphics, source section links, why/when to use it, configuration parameters, unsupported cases, numeric envelope, failure behavior and source-vs-PLLM differences. Generate tables from evidence rather than copying old performance ratios. Show `not implemented`, `adapted`, `reference only` or `reproduced at scope X` honestly.

## Default policy

`eligible_default = false` until implementation, full coverage, source/security review and deployment gates are met. The track restriction remains binding even after successful reproduction. Experimental or comparison methods cannot silently enter the preferred single-evaluator/no-HE profile.
