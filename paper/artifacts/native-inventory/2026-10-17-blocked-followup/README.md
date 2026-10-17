# Fourth cached-build rejection

A 30-second host observation recorded no new swap use or swapout and at least
12,448,366,592 available bytes. The subsequent build used the same 4 GiB owner
allowance, 4 GiB host reserve, two Cargo jobs, and zero-additional-swap policy.

The guard stopped the build on **65,536 bytes of host swapout growth** after
1.18 seconds. Cleanup reported no error. The before/after source snapshots
matched; no wheel was produced. This does not attribute host-wide swapout to
compilation. Sampled owned RSS reached 144,343,040 bytes before the stop, which
is not a completed-build peak measurement.

`preflight/receipt.json` and `v4/` copy the original files verbatim. `index.json`
records their source paths, lengths, and SHA-256 hashes. The embedded driver is
historical source, not an executable replay entry point. The
[first three attempts](../2026-10-17-blocked-wheels/README.md) remain unchanged.

Installed-binding qualification remains open: two binding tests and three
generated-family inventory tests require an accepted native build. This archive
adds no model scan, runtime execution, or benchmark result.
