# R18 · Compact: Approximating Complex Activation Functions for Secure Computation

**Priority 18 · 2024 · numeric_transform · source checked 2026-09-23**

Authors: Mazharul Islam, Sunpreet S. Arora, Rahul Chatterjee, Peter Rindal, Maliheh Shirvanian.  
Primary source: https://arxiv.org/abs/2309.04664  
Source: complete 17-page arXiv v2 PDF, SHA-256 `e4497a81055e60cca1c462dac5ae04b4504c5f3fc76e392e170cc32bf8fee86c`. The paper is an implementation specification, not a runtime dependency.

## What the source contributes

Compact builds input-density-aware piecewise polynomial approximations for complex activations.
Its weighted mean objective is defined in Eq. (6); Chebyshev interpolation,
piecewise boundaries, and an accuracy-constrained search are described in
Sections 4.1–4.3. The reported 2–5× gain is for the paper's classification
models and MPC libraries, not for PLLM or Qwen. Its normal input-density prior
depends on batch normalization (Sections 4.2.1, 6); RMSNorm decoder activations
do not satisfy that assumption without separate calibration evidence.

This short source summary is separate from the proposed PLLM design below. The source has not been reproduced merely by adding this card.

## Native implementation scope

Implement its fitting objective separately from protected evaluation. Lock coefficient/interval hashes and specify an explicit tail policy.

The first independent PLLM slice fits the existing **signed Q7 SiLU** domain
`[-128, 128]` using explicit *public, offline* calibration counts and
Chebyshev quadratic pieces. It uses the weighted absolute-error objective from
Eq. (6), plus a unit pseudocount for each encoded input so unobserved tails
remain measured. Splits are deterministic and bounded; coefficients and
intervals are fixed-point, digest-bound and checked on all 257 encoded inputs.
The offline fitter uses floating-point transcendental functions. Its digest
locks the resulting Q20 coefficients, but identical outputs across platforms
are not yet established.
Out-of-domain inputs fail instead of borrowing Compact's `[-5, 5]` tail policy.
This first slice is a scoped numeric adaptation, **not** the paper's approximate
continuous-range optimizer, its 2PC/3PC evaluation, model accuracy, or a
private interval selector. The `CompactPiecewiseActivation` SDK class remains
pending while those composition and evidence gates remain open.

A second independent Rust reference uses a one-use half-gates circuit to select
one of the fitted intervals without opening the Q7 input or the selected index
to its evaluator. It consumes circuit material on evaluation; output labels
can be decoded only by the trusted client. This **isolated selection circuit**
does not evaluate the polynomial, transport material between roles, bind a
compiler plan, or establish a reviewed private method. The SDK stub remains
pending, and the Python API exposes only the public numeric reference.

An additional in-process half-gates lookup oracle evaluates the fitted Q7
profile on a hidden client input and returns an output label decoded only by
the client. It precomputes all 257 encoded results from the immutable profile
and costs 2,304 AND gates (73,728 half-gate ciphertext bytes) per scalar,
excluding constants, instructions, and input/output labels. This oracle tests
protected encoded-domain fidelity; **it is not Compact's piecewise-polynomial
evaluation or a measured speedup**. The separate selector and lookup are not
composed into a compiled runtime method.

The next bounded Rust circuit combines private piece selection with exact
signed ties-to-even division by each public interval width. It returns the
selected piece and its Q20 normalized Chebyshev coordinate as opaque labels,
decoded only by the trusted client. Uniform and skewed profiles match the
independent integer rounding oracle at boundaries, including negative ties.
This still does not multiply protected coordinates by public coefficients,
evaluate the polynomial, or produce a compiler-bound whole-model component.

Data flow: **Public fitting/calibration data → immutable piecewise polynomial profile; protected evaluation separate.**

The target capability slots, **not yet registered runtime components**, are:

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

The public research backlog records the remaining implementation and validation work. Native integration requires the actual PLLM command, source locks, raw measurements, and assurance report.

## Developer-facing explainer requirement

The eventual method page needs a worked operator example, a role diagram in prose or graphics, source section links, why/when to use it, configuration parameters, unsupported cases, numeric envelope, failure behavior and source-vs-PLLM differences. Generate tables from evidence rather than copying old performance ratios. Show `not implemented`, `adapted`, `reference only` or `reproduced at scope X` honestly.

## Default policy

`eligible_default = false` until implementation, full coverage, source/security review and deployment gates are met. The track restriction remains binding even after successful reproduction. Experimental or comparison methods cannot silently enter the preferred single-evaluator/no-HE profile.
