# Bounded GKM decoder evidence

`local-v1/` preserves the first cached build and its conformance history.
`local-v2/` preserves the formatted-source build and a complete
installed-wheel regression run. `local-v3/` preserves the admission-repaired
build, 206 installed Python passes and 44 native passes. Each archive has its
own wheel identity.
The [v2 admission audit](admission-audit-v2.md) found two blocking
admission defects despite its passing functional checks. Historical archives
retain their original scope and bytes. The [repair review](admission-audit-v3.md)
also records the two shared shape-pricing findings and their closure.

Read-only replay:

```sh
.venv/bin/python paper/artifacts/gkm-decoder/verify.py
.venv/bin/python paper/artifacts/gkm-decoder/check_controls.py
```

The verifier checks every indexed file, selected build-source snapshots,
observer identities, test results and installed-package receipts. It also
checks retained wheel bytes when they are available. It reports a missing
wheel check explicitly. No model, checkpoint or benchmark executes during replay.

`verification-controls.json` pins the current verifier and final archive index. Six
negative controls reject corrupted payloads, an unlisted file, a rehashed source
substitution, a rehashed failed-test result, a changed native source identity and
a changed executed-test snapshot.
The original verifier and controls remain byte-for-byte under `verification-v2/`.
`check_install.py` records the selected interpreter's complete package identity;
it does not execute a decoder.
`format-reconciliation-v3.json` records a subsequent whitespace/quote-only cleanup
of two files with equal Python ASTs. This static check does not replace the
archived v3 source bytes or claim a new installed-wheel execution.

The retained Clippy run used `-D warnings` with the existing
`clippy::needless_range_loop` and `clippy::nonminimal_bool` exclusions.

The contract and remaining gates are in
[`docs/research/gkm-decoder-reference.md`](../../../docs/research/gkm-decoder-reference.md).
Generated conformance does not qualify full-width Qwen, distributed GKM,
checkpoint quality or serving resources.
