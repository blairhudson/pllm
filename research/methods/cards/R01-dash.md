# R01 · Dash: Accelerating Distributed Private Convolutional Neural Network Inference with Arithmetic Garbled Circuits

**Priority 1 · 2024 · single_evaluator · source checked 2026-09-14**

Authors: Jonas Sander, Sebastian Berndt, Ida Bruhns, Thomas Eisenbarth.  
Primary source: https://arxiv.org/html/2302.06361v2  
Paper SHA-256: `4f3dfd31cbf853f51b5ae8f93672af294429e3cc5f8cf7fe5181a4546a436e7a`.  
Artifact: `https://github.com/UzL-ITS/dash` at `894e2bb7e104c2996933d53f4aa7ca58f2c802be`. The artifact is GPL-3.0 and may be run as an isolated reproduction input, but its implementation must not be copied or linked into PLLM. Source acquisition is complete; reproduction is not.

## What the source contributes

Arithmetic garbling and LabelTensors provide the foundation for offline preparation and local encoded public-weight evaluation.

This short source summary is separate from the proposed PLLM design below. The source has not been reproduced merely by adding this card.

## Native implementation scope

Implement affine-label arithmetic, constant multiplication with the required algebraic preconditions, projections and tensorized evaluation. Preserve the original circuit functionality before adapting to transformers.

Data flow: **Labelled public matrix → arithmetic-label result; projection/nonlinear material is prepared by the offline garbler.**

Register the following independently versioned native components:

- `garble.affine_label`
- `kernel.label_tensor`
- `gate.projection`

Dependencies: R03, R04. Public compilation/model-data scans and all online evaluation happen in Rust or reviewed native accelerator libraries called from Rust. Python selects immutable parameters and displays public results. It does not implement a timed alternative or silently repair unsupported native execution.

## Required fixture and benchmark specification

Original CNN/model workloads first; then matched public linear operators and a fixed-point decoder region. Report label lanes, storage and all setup.

Mandatory paper-specific gates:

- Public matrix constants that are zero or non-units for the selected modulus
- Cross-circuit label/offset confusion
- Complete setup+evaluation at the source precision

Each fixture locks input/output representations, ring/field parameters, numeric graph, party count and material lifetime. Compare an optimized implementation with identical functionality and applicable privacy assumptions; record original-artifact numbers and adaptations separately. A source-advertised speedup is not an expected threshold for different hardware.

## Privacy, correctness and proof obligations

Affine-label view simulation and applicability of the chosen correlation-robust hash; additional integrity mechanisms are separate.

Do not inherit malicious security by importing kernels: the paper discusses additional integrity mechanisms and trusted-hardware or cryptographic configurations.

Provide the corrupted party's permitted view, known plaintext/public inputs, randomness model, allowed leakage and attack budget. Formal or empirical results have explicit scope. Passing functional tests cannot grant a production privacy badge. A changed hash, field, layout, actor role or precision creates a documented adaptation obligation.

## What PLLM already has

Prior single-server, model-aware and tenx archives contain adaptations, not a reproduced Dash release.

The original experiments and limitations are under `legacy/`. This handoff adds contracts and research tasks, **not a completed native reproduction of this paper**.

## Reproduction gates

`R01.acquire → R01.specify → R01.reference → R01.native → R01.assure → R01.benchmark → R01.document`.

The machine-readable recipe at `research/recipes/R01.json` is a work specification, not a pretend executable paper command. During acquisition, store the exact original artifact invocation and commit; leave them unclaimed until obtained. During native integration, attach the actual PLLM command, all environment/source locks, raw measurements and assurance report.

## Developer-facing explainer requirement

The eventual method page needs a worked operator example, a role diagram in prose or graphics, source section links, why/when to use it, configuration parameters, unsupported cases, numeric envelope, failure behavior and source-vs-PLLM differences. Generate tables from evidence rather than copying old performance ratios. Show `not implemented`, `adapted`, `reference only` or `reproduced at scope X` honestly.

## Default policy

`eligible_default = false` until implementation, full coverage, source/security review and deployment gates are met. The track restriction remains binding even after successful reproduction. Experimental or comparison methods cannot silently enter the preferred single-evaluator/no-HE profile.
