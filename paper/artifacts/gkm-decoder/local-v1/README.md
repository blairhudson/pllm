# Bounded GKM decoder conformance

This append-only archive retains the guarded cached build, installed-package
postcheck, native and Python tests, failed regression attempt and repair, and
selected runtime/test sources. `index.json` binds every retained payload to its
original path, size and SHA-256. Build-source manifests retain all source hashes;
only the selected implementation sources are copied here.

The cached build completed in 72.005 seconds with 1,158,823,936 bytes maximum
sampled owned RSS and zero observed swap-use or actual compressor-swapout growth.
Its source/observer checks and cleanup pass. This is not an empty-target build.
The wheel's SHA-256 is
`09a70827af66d0dab580a3384ac033a6d18056693f5dc62fbc6faf8da6535223`.
The post-test installed-package check matches all 356 package files to that wheel;
it is not a provider-child execution attestation.

Forty native tests pass: 14 inventory, 22 schedule and four decoder/session tests.
Nineteen installed-wheel decoder tests pass. Six generated family/tied-head
fixtures each execute full and seeded prefill/decode: 12 requests compare 2,112
float32 full-logit/final-active-KV values, plus selections, history and position.
These use independent Python integer/model equations, not upstream execution.

The initial shared-executor run retains 151 passes, four failures and two
optional real-Gemma skips. Its synthetic token-boundary fixture omitted an
existing boundary-owner field, and Transformers was missing. The repaired
affected files pass seven tests with the same two skips. The test-identity union
therefore has 155 passes and two skips, not a new 158-case cohort. Original logs
and the original fixture remain separate from the repair.

All execution here is bounded conformance. Both GKM workers share a process;
the generated hybrid SDK/gateway checks exercise the existing prepared executor,
not a distributed GKM path. Real Qwen geometry, quality, full-width publication,
source batched prefill, confidential transport and serving resources remain open.

Read-only replay:

```sh
.venv/bin/python paper/artifacts/gkm-decoder/verify.py
```

The verifier needs no installed GKM binding or checkpoint. It checks the retained
wheel when available and otherwise reports that check as absent. The wheel stays
under `target/`; automatic target cleanup protects wheel and evidence files.
