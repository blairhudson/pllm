# R02 · ReDASH: Fast and efficient Scaling in Arithmetic Garbled Circuits for Secure Outsourced Inference

**Priority 2 · 2025 · single_evaluator · source checked 2026-09-14**

Authors: Felix Maurer, Jonas Sander, Thomas Eisenbarth.  
Primary source: https://arxiv.org/html/2506.14489v1  
Access in this handoff: `primary_full_text_html`. Full source/artifact content hashes are not yet locked. A full-text acquisition gate is not required for reading, but exact paper and original artifact pinning remain required.

## What the source contributes

ReDASH generalizes residue-number-system scaling and uses ScaleQuant+ to reduce encoded-inference overhead.

This short source summary is separate from the proposed PLLM design below. The source has not been reproduced merely by adding this card.

## Native implementation scope

Implement signed base extension, the published scaling gadget and its quantization policy as separately versioned components. Expose exact and approximate semantics explicitly.

Data flow: **CRT arithmetic labels → rescaled/base-extended arithmetic labels with a frozen rounding definition.**

Register the following independently versioned native components:

- `convert.rns_base_extension`
- `gate.scale`
- `numeric.scalequant_plus`

Dependencies: R01, R03. Public compilation/model-data scans and all online evaluation happen in Rust or reviewed native accelerator libraries called from Rust. Python selects immutable parameters and displays public results. It does not implement a timed alternative or silently repair unsupported native execution.

## Required fixture and benchmark specification

Exhaustive small CRT domains, signed boundary fixtures, reference CNN comparison and identical decoder-region rescaling.

Mandatory paper-specific gates:

- Global signed boundaries spanning multiple residues
- Negative ties and maximal values
- Source approximation/error profile versus exact alternatives

Each fixture locks input/output representations, ring/field parameters, numeric graph, party count and material lifetime. Compare an optimized implementation with identical functionality and applicable privacy assumptions; record original-artifact numbers and adaptations separately. A source-advertised speedup is not an expected threshold for different hardware.

## Privacy, correctness and proof obligations

Proof of the actual conversion and sign/rounding envelope; no local field-inverse shortcut.

Preserve the specified sign/scaling assumptions. Modular inversion is not ordinary fixed-point division; per-residue ReLU is not global integer ReLU.

Provide the corrupted party's permitted view, known plaintext/public inputs, randomness model, allowed leakage and attack budget. Formal or empirical results have explicit scope. Passing functional tests cannot grant a production privacy badge. A changed hash, field, layout, actor role or precision creates a documented adaptation obligation.

## What PLLM already has

Only researched previously; earlier CRT kernels did not implement full ReDASH scaling.

The original experiments and limitations are under `legacy/`. This handoff adds contracts and research tasks, **not a completed native reproduction of this paper**.

## Reproduction gates

`R02.acquire → R02.specify → R02.reference → R02.native → R02.assure → R02.benchmark → R02.document`.

The machine-readable recipe at `research/recipes/R02.json` is a work specification, not a pretend executable paper command. During acquisition, store the exact original artifact invocation and commit; leave them unclaimed until obtained. During native integration, attach the actual PLLM command, all environment/source locks, raw measurements and assurance report.

## Developer-facing explainer requirement

The eventual method page needs a worked operator example, a role diagram in prose or graphics, source section links, why/when to use it, configuration parameters, unsupported cases, numeric envelope, failure behavior and source-vs-PLLM differences. Generate tables from evidence rather than copying old performance ratios. Show `not implemented`, `adapted`, `reference only` or `reproduced at scope X` honestly.

## Default policy

`eligible_default = false` until implementation, full coverage, source/security review and deployment gates are met. The track restriction remains binding even after successful reproduction. Experimental or comparison methods cannot silently enter the preferred single-evaluator/no-HE profile.
