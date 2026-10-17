# Interrupted GKM checkpoint source scans: negative evidence

`2026-10-17-interrupted-v1-v2/` is an append-only historical archive, distinct
from mutable working code and the original `target/research/` output locations.
Its index binds exact copied bytes; files are made read-only when archived.

Both runs failed on `new_swapout`. Neither decoded a tensor or completed the
checkpoint/tokenizer source lock. The second run's admission and header geometry
are partial metadata observations, not source-numeric eligibility. A consistent
archive does not upgrade those results.

The historical scanner has two subsequently identified limitations:

1. Its `code_hashes` included checkout Python files rather than necessarily the
   imported wheel's metadata modules. The native module path was taken from the
   imported module. Historical Python hashes must not be called executed-module
   identity, and this archive does not retrospectively repair provenance.
2. Its resource guard timestamp preceded telemetry reads. Its final work snapshot
   preceded JSON serialization/publication and was not a process-lifetime guard.

Original scanner, oracle, tests and research document are preserved as bytes for
review. **Do not execute or import these historical snapshots.** The current
working scanner has separate provenance/timing fixes and unit-only validation.

Read-only verification:

```sh
.venv/bin/python paper/artifacts/gkm-checkpoint-bounds/verify.py
```

The verifier checks the byte index, snapshot identities recorded in both original
receipts, and retained failure status. It reads only this small archive; it does
not inspect the checkpoint cache or execute model/source scans. The index is an
integrity manifest, not an independent authentication or eligibility certificate.
