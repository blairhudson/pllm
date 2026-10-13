# Authenticated lossless artifact-update delta screen

Synthetic-only public-object screen. No credible authenticated actual revision,
equalization, quantization or placement old/new object pair found in directory
entries inspected under `target/prepared-frontier-artifacts` and
`target/qwen3-memory-artifacts`. This is not an exhaustive cache search. No model
download/import, provider, checkpoint scan, timing or RSS cohort ran.
Whole-client <=256 MiB eligibility remains **unknown**.

The requested ranked hypothesis #8 differs from H8 in
`docs/plans/low-tier-all-link-hypotheses.md`: that document describes interrupted
download continuation, not update deltas. Existing artifact identity includes
content and metadata-domain hashes; identical object caching and live native
snapshot reuse are controls, not new savings. No private tensor/state deltas,
approximation or quality comparison are involved.

## Frozen recipe and independent control

Calibration: five deterministic 32 KiB public synthetic changes, seed 8108.
Compare fixed-position copy/literal versus copy/XOR/literal, zlib level 9.
Freeze smallest aggregate encoded patch before constructing held-out seed 9108,
five 256 KiB changes. Sparse, independent dense, dense low-entropy, shifted and
appended bytes are synthetic mechanisms, **not actual weight update evidence**.

Independently encode complete new bytes with zlib levels 1/6/9 and 64 KiB/1 MiB
frames; use smallest as deliberately strong per-case control, also retain current
level-1/1-MiB size. This favorable control oracle is not a deployed selection
policy. Original new-object SHA-256 and metadata-domain key are preserved.
Evidence JSON records exact lengths, hashes and reconstruction results.

| Held-out synthetic change | Strongest full + manifest | Delta hit, all added bodies | Net saving |
| --- | ---: | ---: | ---: |
| Sparse byte flips | 262,547 | 1,871 | 260,676 |
| Independent dense bytes | 262,547 | 263,677 | -1,130 |
| Dense zero-filled target | 594 | 1,763 | -1,169 |
| Shift by 12 bytes | 262,547 | 263,677 | -1,130 |
| Append 480 bytes | 262,602 | 1,463 | 261,139 |

XOR calibration total 69,227 bytes versus replace 134,694. Sparse/append require
successful base-hit rates above 0.596%/0.440% under the pessimistic fully attempted
miss model. Each needs two successful updates to repay an extra cold base retained
only for deltas, before other setup/disk cost. Other three cases cannot break even
even at 100% hits. No actual model update conclusion follows. Regression suite:
31 passing tests; Ruff passes. Evidence includes source/docs/probe/test hashes.

## Format, bounds and authentication

`PLLMDEL1`, big-endian header length, canonical JSON header with base/new manifest
rows, version, codec and 64 KiB block size. Each ordered record carries one-byte
copy/XOR/literal tag plus big-endian raw/encoded uint32 lengths. Copy has no body;
others hold independent zlib streams. XOR uses same-position base bytes, zero-pad
beyond base end. Header <=16 KiB, raw object <=8 MiB, patch <=9 MiB; no unbounded
offset table, random-match index or learned dictionary. Per-record decode has
bounded output and rejects excess length, trailing stream data and incomplete
streams. Producer builds small fixture patches in memory; client reconstruction
is streaming. Explicit live byte-buffer allowance is conservatively 576 KiB;
Python, zlib internals, native imports and full client are excluded, so this is
not a measured RSS bound.

Authentication is a **precondition**: publisher-authenticated old/new rows and
patch digest must arrive through existing trusted manifest/control machinery.
A self-hash is not publisher authentication. Header rows must exactly match
trusted rows; base is rehashed before use, patch and reconstructed target are
hashed before publication. Filesystem base must be caller-owned immutable owner
throughout reconstruction, such as existing native snapshot semantics; arbitrary
writable cache paths are not production-safe. This standalone prototype does
not integrate native ownership or enforce concurrent base immutability.

Unpublished same-directory temporary file, file fsync, atomic replace after all
checks. Failure deletes pending file only and preserves base and prior target.
Base/target path and hardlink alias rejected. This guarantees atomic visibility,
not crash-durable directory publication (directory fsync absent). Base mismatch,
corruption, truncation, oversize, trailing bytes, forged domain, wrong target,
wrong patch commitment, invalid copy and bounded-decompression bomb regressions
are tested. Independent decoder reconstructs all held-out mechanisms.

## Charge every owner and miss

Common new manifest row charged both sides. Delta additionally pays full header
table, record framing, serialized base/new request and authenticated patch offer
(patch size/digest and base/new identity). Existing authenticated channel assumed;
any additional signature/session setup remains an extra parent-measured charge.
Already-needed base first
delivery charged as common cold cost, not invented free cache. If retained only
for deltas, add its strongest compressed first delivery and retention cost.
Evidence records old/new publisher source payloads, patch, base cache, new pending
or new cache, previous destination, old native snapshot and potential additional
new private/native import owner separately. Optional patch spool is another
owner. Atomic replacement does not retire live old snapshots. A shared native
import may avoid copying new cache bytes only after native integration proves
immutable ownership; this prototype promises no such saving. Source checkpoints,
tokenizer/source/index files, metadata, filesystem cache and calibration/import
scratch remain outside these small payload totals and require parent audit.

No learned table distribution. Synthetic calibration CPU and scratch unmeasured;
real patch production and calibration must be charged, not hidden as setup.
One publisher-to-client application-body link measured. Other role/control links
are unchanged by design, unmeasured here; HTTP/TLS/TCP and checkpoint distribution
are excluded. No whole-response all-link percentage follows.

Let F = strongest full new encoding; D = patch plus delta-only request and offer; h = base
hit/success rate. Pessimistic misses pay D+F, not free fallback. Expected savings
per update = hF-D-extra_setup. Break-even h > (D+extra_setup)/F. With U updates and
extra base/setup distribution B: U(hF-D-extra_setup) > B. JSON reports D/F and
minimal integer updates to repay extra compressed base plus base manifest at h=1, excluding other
setup. If D>=F, no hit rate helps. Preflight miss can cost less but requires
charged lookup protocol; not assumed. Dense, shifted, highly compressible targets
and small objects can regress; unchanged objects should skip deltas altogether.

## Replay and next gate

```sh
target/frontier-sdk-env/bin/python -m pytest -o pythonpath=. tests/test_artifact_update_delta_probe.py -q
target/frontier-sdk-env/bin/python -m ruff check scripts/probe_artifact_update_delta.py tests/test_artifact_update_delta_probe.py
target/frontier-sdk-env/bin/python -m scripts.probe_artifact_update_delta --output docs/evidence/research-artifact-update-delta.json
```

No full scan/import mode needed to reach this negative evidence gate. Parent next:
locate authenticated cached real update pair, select <=8 MiB public slices with
exact source/revision/quantizer/placement provenance, freeze recipe and independent
full control before held-out comparison. Only real byte wins warrant native
immutable-owner integration, serial fresh-client import/full-response cohorts,
all-link setup/miss accounting and <=256 MiB whole-client admission. No concurrent
performance run or unimplemented serial-worker command is advertised.
