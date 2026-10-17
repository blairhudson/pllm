# Final formatted-source GKM decoder conformance

This archive retains a separate cached wheel build and installed-wheel test run.
The earlier `local-v1/` evidence remains unchanged.

The build completed in 78.203 seconds. Its largest sampled build-tree RSS was
1,155,350,528 bytes. Observed swap-use and actual compressor-swapout growth were
zero. Source identity, observer identity and cleanup checks passed.

Wheel SHA-256:
`1ef71860df830cc6bb9a72798d4fe44c90e54b46d76e738b229b3bb72751ebcc`.
The isolated installation's 356 package files match the retained wheel bytes.
This post-test check is not a provider-child execution attestation.

The complete Python run passes 174 tests: 19 delegated-decoder tests and 155
shared-executor regressions. Two opt-in real-Gemma checks remain skipped.
The native run passes 40 tests: 14 inventory, 22 schedule and four
decoder/session tests. Clippy passes with the recorded existing lint exclusions.

The first Python run reached the shell's 120-second timeout. Its partial log
does not establish completed test results. The remaining owned inference child
was explicitly retired. The separately recorded retry completed in 106.43 seconds.
Both attempts remain here. The offline dependency-install rejection and successful
pinned online installation also remain separate.

Six generated family/tied-head fixtures each execute full and seeded requests.
These twelve executions compare 2,112 complete logit/final-active-KV float32
values, plus all selections, executed histories and positions. They use
independent Python integer/model equations. Both GKM workers share one process.
The existing prepared SDK/gateway regressions do not establish distributed GKM.

The archive contains selected source/test snapshots and complete build-input
hash manifests. It is local conformance evidence, not a clean-build attestation,
independent security certification, checkpoint-quality result or serving benchmark.

From the repository root:

```sh
.venv/bin/python paper/artifacts/gkm-decoder/verify.py
```
