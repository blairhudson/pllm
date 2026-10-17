# Interrupted cached native-wheel builds

These files retain the three original local build attempts. No attempt produced
an accepted wheel. They establish no installed binding, clean build, release,
performance, or source-numeric qualification.

| Attempt | Recorded outcome |
| --- | --- |
| v1 | Rejected before build launch: insufficient headroom for its 8 GiB owner allowance plus 4 GiB host reserve. |
| v2 | Guard cleanup raised `PermissionError`; no ordinary final receipt was written. The later `recovery.json` retains its separate diagnostic scope. |
| v3 | Repaired guard stopped on 704,512 bytes of host swapout growth. Cleanup reported no error, source snapshots matched, and no wheel was produced. |

v2/v3 used a 4 GiB owner allowance, 4 GiB host reserve, two Cargo jobs, and the
unchanged zero-additional-swap policy. Host-wide swapout is not attributed to
the build. v3 observed at most 53,346,304 bytes of sampled descendant RSS before
its stop; this is not a compiler peak estimate or successful resource result.

`index.json` lists original paths, byte lengths, and SHA-256 hashes. The archive
copies all available regular files verbatim, including the original failure and
recovery records. Empty wheel directories contain no artifacts. The embedded
Python files are historical driver snapshots; do not execute them for replay.
The v3 guard snapshot is the corrected local driver, not a clean-build attestation.

Next work is an admitted cached build, installation in a separate validation
environment, and the two native binding plus three generated-family inventory
tests. Native compiler and scanner-double tests cannot substitute for those
checks. The interrupted Qwen tensor scan remains a separate gate.
