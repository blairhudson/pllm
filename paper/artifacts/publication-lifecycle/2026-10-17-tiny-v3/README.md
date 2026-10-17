# Bounded publication-lifecycle smoke

`archive.json` binds 144 original/source files (4,401,845 bytes). `v3` contains
three successful local generated-checkpoint cold trials: 289 observations,
11 process incarnations, equal full-state bindings, and zero observed swap growth.
`v1` and `v2` preserve rejected attempts. They are not successful observations.

`sources/` retains all ten v3 observer/recipe inputs under their original hashes.
`v1/executed-sources/` retains ten checked inputs from the first rejected attempt.
The second rejected attempt retains source hashes without its earlier driver
snapshot. Installed package/build identities stay in the
original receipts. This archive does not contain the installed environment.

Read-only replay from the repository root:

```sh
paper/artifacts/clean-build/2026-10-16-local-v5/work/venv/bin/python -B scripts/audit_publication_lifecycle.py
```

The auditor relocates only the external trace-reference path in memory. The
original manifests, receipts, prompts, and reports remain byte-identical.
Source-present generated checkpoints establish no real-model quality or speed
result. Provider resources are unnormalized. Sampled RSS, ordinary target peaks,
and the post-serialization self peak keep separate windows.

Full contract: [publication lifecycle reference](../../../../docs/research/publication-lifecycle-reference.md).
