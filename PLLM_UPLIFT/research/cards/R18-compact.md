# R18 · Compact: Approximating Complex Activation Functions for Secure Computation

**Priority 18 · 2024 · numeric_transform · source checked 2026-09-14**

Authors: Mazharul Islam, Sunpreet S. Arora, Rahul Chatterjee, Peter Rindal, Maliheh Shirvanian.  
Primary source: https://arxiv.org/abs/2309.04664  
Access in this handoff: `primary_abstract`. Full source/artifact content hashes are not yet locked. A full-text acquisition gate is required.

## What the source contributes

Compact builds input-density-aware piecewise polynomial approximations for complex activations.

This short source summary is separate from the proposed PLLM design below. The source has not been reproduced merely by adding this card.

## Native implementation scope

Implement its fitting objective and published approximation profile separately from protected evaluation. Freeze coefficient/interval hashes and add explicit tail policy.

Data flow: **Public fitting/calibration data → immutable piecewise polynomial profile; protected evaluation separate.**

Register the following independently versioned native components:

- `numeric.piecewise_fit`
- `compiler.approximation_profile`

Dependencies: None; foundational work item. Public compilation/model-data scans and all online evaluation happen in Rust or reviewed native accelerator libraries called from Rust. Python selects immutable parameters and displays public results. It does not implement a timed alternative or silently repair unsupported native execution.

## Required fixture and benchmark specification

Approximation envelopes and full-checkpoint quality, not scalar error alone; protected evaluator includes interval selection and all truncation.

Mandatory paper-specific gates:

- Held-out and tail approximation errors
- Coefficient/interval hash reproducibility
- Full-model quality and all private interval-selection costs

Each fixture locks input/output representations, ring/field parameters, numeric graph, party count and material lifetime. Compare an optimized implementation with identical functionality and applicable privacy assumptions; record original-artifact numbers and adaptations separately. A source-advertised speedup is not an expected threshold for different hardware.

## Privacy, correctness and proof obligations

Numerical fidelity and calibration-data boundary; no privacy conclusion follows from a good approximation fit.

Calibration is not a proof of a worst-case range. Model quality and exact agreement with the chosen approximation are distinct.

Provide the corrupted party's permitted view, known plaintext/public inputs, randomness model, allowed leakage and attack budget. Formal or empirical results have explicit scope. Passing functional tests cannot grant a production privacy badge. A changed hash, field, layout, actor role or precision creates a documented adaptation obligation.

## What PLLM already has

Our Fourier/boundary-matched fits are distinct hypotheses, not Compact reproductions.

The original experiments and limitations are under `legacy/`. This handoff adds contracts and research tasks, **not a completed native reproduction of this paper**.

## Reproduction gates

`R18.acquire → R18.specify → R18.reference → R18.native → R18.assure → R18.benchmark → R18.document`.

The machine-readable recipe at `research/recipes/R18.json` is a work specification, not a pretend executable paper command. During acquisition, store the exact original artifact invocation and commit; leave them unclaimed until obtained. During native integration, attach the actual PLLM command, all environment/source locks, raw measurements and assurance report.

## Developer-facing explainer requirement

The eventual method page needs a worked operator example, a role diagram in prose or graphics, source section links, why/when to use it, configuration parameters, unsupported cases, numeric envelope, failure behavior and source-vs-PLLM differences. Generate tables from evidence rather than copying old performance ratios. Show `not implemented`, `adapted`, `reference only` or `reproduced at scope X` honestly.

## Default policy

`eligible_default = false` until implementation, full coverage, source/security review and deployment gates are met. The track restriction remains binding even after successful reproduction. Experimental or comparison methods cannot silently enter the preferred single-evaluator/no-HE profile.
