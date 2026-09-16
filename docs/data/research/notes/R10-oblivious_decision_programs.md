# R10 · Oblivious decision program evaluation

**Priority 10 · 2013 · single_evaluator_adaptation · source checked 2026-09-14**

Authors: Salman Niksefat, Babak Sadeghiyan, Payman Mohassel.  
Primary source: https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-ifs.2012.0032  
Access in this handoff: `primary_full_text_html`. Full source/artifact content hashes are not yet locked. A full-text acquisition gate is not required for reading, but exact paper and original artifact pinning remain required.

## What the source contributes

Oblivious decision-program protocols evaluate protected decision structures using OT-based reductions.

This short source summary is separate from the proposed PLLM design below. The source has not been reproduced merely by adding this card.

## Native implementation scope

Reproduce the original actor roles first. Then separately specify a model-aware garbler/offline-dealer adaptation for encoded nonlinear paths, including all input label delivery.

Data flow: **Encoded decision inputs + encrypted program → source-defined protected result; role adaptation is separately named.**

Register the following independently versioned native components:

- `gate.oblivious_branching_program`
- `convert.branching_input`

Dependencies: R04. Public compilation/model-data scans and all online evaluation happen in Rust or reviewed native accelerator libraries called from Rust. Python selects immutable parameters and displays public results. It does not implement a timed alternative or silently repair unsupported native execution.

## Required fixture and benchmark specification

Fixed public depths, original decision-program fixtures, path access observables and arithmetic I/O conversion costs.

Mandatory paper-specific gates:

- Same public path budget for distinguishability pairs
- Node identity and state-access observables
- Input labels provisioned without plaintext index disclosure

Each fixture locks input/output representations, ring/field parameters, numeric graph, party count and material lifetime. Compare an optimized implementation with identical functionality and applicable privacy assumptions; record original-artifact numbers and adaptations separately. A source-advertised speedup is not an expected threshold for different hardware.

## Privacy, correctness and proof obligations

Oblivious decision-program proof in its original role assignment; masked arithmetic outputs require a new composition argument.

Role reversal and arithmetic output are adaptations. Fixed path length alone does not prove that node identities or graph topology are hidden.

Provide the corrupted party's permitted view, known plaintext/public inputs, randomness model, allowed leakage and attack budget. Formal or empirical results have explicit scope. Passing functional tests cannot grant a production privacy badge. A changed hash, field, layout, actor role or precision creates a documented adaptation obligation.

## What PLLM already has

Weighted transducer archive uses related ideas but is not a reproduction of this protocol.

The original experiments and limitations are under `legacy/`. This handoff adds contracts and research tasks, **not a completed native reproduction of this paper**.

## Reproduction gates

`R10.acquire → R10.specify → R10.reference → R10.native → R10.assure → R10.benchmark → R10.document`.

The public research backlog records the remaining implementation and validation work. Native integration requires the actual PLLM command, source locks, raw measurements, and assurance report.

## Developer-facing explainer requirement

The eventual method page needs a worked operator example, a role diagram in prose or graphics, source section links, why/when to use it, configuration parameters, unsupported cases, numeric envelope, failure behavior and source-vs-PLLM differences. Generate tables from evidence rather than copying old performance ratios. Show `not implemented`, `adapted`, `reference only` or `reproduced at scope X` honestly.

## Default policy

`eligible_default = false` until implementation, full coverage, source/security review and deployment gates are met. The track restriction remains binding even after successful reproduction. Experimental or comparison methods cannot silently enter the preferred single-evaluator/no-HE profile.
