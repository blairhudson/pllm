# Admission-repaired GKM decoder conformance

This archive retains a third cached build and a fresh installed-wheel test run.
`local-v1/` and `local-v2/` retain their original evidence, including failures.
The [review record](../admission-audit-v3.md) describes the admission repairs.

The guarded build completed in 39.584 seconds. Its largest sampled build-tree
RSS was 1,355,612,160 bytes. Observed swap-use and actual compressor-swapout
growth were zero. Source identity, observer identity and cleanup checks passed.
This is a cached local build, not a clean-build or release attestation.

Wheel SHA-256:
`ea6abd2dca646137fdf2e7b6e81d84da9de70bd22bc257976e69157ebea38eb3`.
All 356 installed package files match the wheel before and after testing.
These parent-package checks are not provider-child execution attestations.
Dependencies were installed offline into a new environment from pinned versions.

The Python suite passes **206 tests**: 51 delegated-decoder/admission cases and
155 shared-executor regressions. Two optional real-Gemma checks remain skipped.
The native suite passes **44 tests**: three source-mapping admission tests,
14 inventory tests, 22 schedule tests and five decoder/session tests. Clippy
passes with the existing `needless_range_loop` and `nonminimal_bool` exclusions.

The same twelve generated family/tied-head/full-or-seeded executions preserve
2,112 complete logit and final-active-KV float32 values, plus all selections,
executed histories and positions. New admission checks prove bounded source
mapping access and configuration encoding. Native sentinel tests prove geometry
rejection before graph construction, including both follow-up shape counterexamples.

Native source snapshots match the build. Python test-file hashes remain equal
before and after execution; retained test snapshots match those executed inputs.
The archive retains selected source/test files and complete build-input hashes.
Both logical GKM workers remain in-process. Full-width checkpoints, original-model
quality, distributed execution, cryptographic certification and serving resources
remain separate gates.

Read-only replay from the repository root:

```sh
.venv/bin/python paper/artifacts/gkm-decoder/verify.py
```
