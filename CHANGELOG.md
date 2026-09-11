# Changelog

## 0.17.0a1

- Consolidate all maintained Python code under `python/pllm`.
- Split the Rust numeric core from the PyO3 binding in a Cargo workspace.
- Use Maturin as the sole Python build backend.
- Load public Python exports on demand and add `python -m pllm`.
- Keep native matrix dimensions and weights private in the Rust API.
- Make offline seeded inventory the public-weight path: preparation pushes
  committed `W*r-s` batches before inference reports `READY`.
- Reserve and burn one-time rows per execution, retain unreserved rows in memory,
  and refill only while the client is idle.
- Select exact `u16`, `u24`, or `u32` stage rings and keep activation scales local.
- Compact prompt prefill into packed multi-row stage requests while decode reuses
  its persistent client-to-inference connection.
- Move public token lookup and output-head matrices to the client; schema 2 shares
  one canonical matrix for tied embeddings.
- Add a loopback three-role OpenTelemetry dashboard with insert-only sanitized
  run history and model/context comparisons; random tiny weights remain an
  explicit transport-only smoke mode.
- Keep BFV, blinded confidential-weight paths, and authenticated arithmetic as
  separately selected legacy or research modes rather than public defaults.
- Align tests, wheel validation, CI, deployment and Fumadocs with the workspace.
- Retain historical measurements without reclassifying them as Rust results.


## 0.16.0a2

- Maturin/PyO3 package with Rust modular arithmetic, coefficient arithmetic,
  codecs, masking, quantization, and OS random sampling.
- Immutable compiled stages and a persistent Rayon executor.
- Widen products before accumulation for large residues.
- Remove runtime C++ compilation; preserve C++ only as a benchmark control.
- Add portable wheel CI, native tests and matched benchmark tooling.
- Source migration; local Rust compilation and speed measurements remain unrun.


## 0.16.0a1 — 2026-09-06

This is a repository and release engineering revision, not a new inference
performance result.

* Consolidate the persisted runtime, Fumadocs app, manuscript and evidence.
* Keep the public `pllm` package and compatibility namespace.
* Add PR checks, HE checks, source/wheel checks and manuscript compilation.
* Add PyPI trusted publication and static GitHub Pages deployment.
* Correct repository-relative installation and deployment paths.
* Make Pages links, downloads and browser search work below a repository path.
* Add dependency lock generation, release validation and repository tests.
* Retain the prior fixes that keep activation scales local and use cryptographic
  randomness for masks and default integrity challenges.

The native runtime is the persisted C++ implementation. No Rust or GPU build is
claimed. The paper's inference measurements remain the preserved study results.
