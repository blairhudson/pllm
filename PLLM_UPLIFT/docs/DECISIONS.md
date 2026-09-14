# Consolidated direction and prior research ledger

## Current target supersedes earlier exploration

Keep a minimal-work Client, model-aware offline Preparation and one online Inference evaluator. Optional HE/two-online/TEE profiles are comparisons. Allowing Preparation the model intentionally eliminates model-free OT/HE label-base setup in the preferred branch; continue studying those methods only when their distinct constraint is useful.

## Preserve, but do not overclaim

| Prior branch | What is retained | What is not promoted |
|---|---|---|
| Original masked-linear runtime | Reported Qwen baseline, exact rings, packed prefill, native matrices and telemetry provenance | Client-heavy execution is not thin-client; no-plaintext counters do not prove confidentiality |
| Local verification revision | Signed bounded-integer verification, independent verifier secrets, lifetime arguments | No model-free authenticated verifier or full malicious protocol was implemented |
| Network codecs | Public-bound wire packing; bounded lowbits/syndrome experiment; side-information typing | Out-of-domain decoding can silently collide; packet savings are not TPS |
| Model-free preparation | OT/Paillier reference role separation as a comparator | Huge offline costs; unnecessary after Preparation may have W |
| Resident shares/FSS | Public linear algebra and immutable KV correlation experiments in two-online profile | Violates preferred online-worker budget |
| Spectral/polynomial gates | Numerical fits and two-party translated-basis hypothesis | Exposing shifted coefficients to one evaluator leaks masks/phase |
| Arithmetic garbling | One-evaluator target, affine labels, tensorized public W, native port priorities | Multiple label lanes and nonlinear/rescaling preparation remain expensive |
| Model-aware compiler | Local WB, certified range/geometry, fused exact gates, packed weight layout | Synthetic bounds and matrix tests are not full-model/GPU results |
| Weighted decision programs | Exact public LUT compression; candidate encrypted paths; conversion-aware comparisons | Custom garbling is unaudited; strong logrow baseline not implemented |
| Rare mathematics | Galois-ring PCG lead, translation-rank diagnostic, wavelet alternative, explicit counterexamples | No free privacy from small masks, symmetries, Frobenius, modular inverse or shared seeds |

## Attack regressions to keep

Distinct-input pad reuse; affine wire/base reuse; exposed Delta; public lane reconstruction; sparse point-and-permute support; scalar nonlinear translator finite differences; shifted polynomial coefficients and Fourier phase; rank-deficient masks; norm-preserving rotations; truncation carry omission; field inverse substituted for division; hidden-domain codec collision; secret-dependent fallback/early exit; node masks derived from evaluator-known state keys; unsafe CRT narrowing; timeout recycling; snapshot/cloning; changed model/quantization after preparation.

Each regression names the intentionally weakened construction and its exact violation. A failed variant does not refute the source protocol whose assumptions differ. Conversely, mathematical exactness or random-looking messages do not rescue a variant with a valid distinguishing attack.

## Defaults and evidence

No prior artifact establishes full thin-client, one-online-evaluator, near-native private Qwen execution. Baseline first, complete native regions next, integrated target third. The best eventual default is an empirically validated constrained plan, not the latest technique added to the repository.

Use archived results only with `origin=source_reported` / `archived_not_rerun`. Retain originals under hashes. Never overwrite them with newly normalized values or silently revise the definition of TTFT, bytes or scope. New measurements get their own plan and evidence manifests.
