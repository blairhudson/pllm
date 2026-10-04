# Bounded bundle memory reduction

The public prepared and two-offset services now stream the existing MessagePack
bundle from immutable binary segments. Matrix segments retain their read-only
Rust owner. Metadata is independently snapshotted, and read-only facades over
caller-mutable storage are copied before commitment. Contiguous `client_bundle()`
remains available to direct callers.

Artifact export uses those same segments rather than unpacking and repacking a
full bundle. The client verifies every fetched object and streams the canonical
raw digest over the reconstructed document before the existing bundle validator
accepts it. It no longer materializes a second contiguous raw bundle for artifact
import. Raw downloads reject data exceeding the authenticated declared length
while reading. Raw identities, artifact identities and global zlib frame
boundaries are unchanged.

## Isolated measurement

`bundle-memory-2026-10-05.json` records two isolated processes over one 64 MiB
native i8 matrix, using the same bytes and metadata. The materialized control
copies binary leaves and packs, parses and canonically repacks the document;
the candidate snapshots metadata and streams native-backed segments.

| Metric | Materialized | Segmented |
| --- | ---: | ---: |
| Serialized bytes | 67,108,909 | 67,108,909 |
| Process peak RSS | 441.94 MB | 244.83 MB |
| Retained serialized-copy bytes | 134.22 MB | 0 |
| Serialization CPU | 0.0982 s | 0.0842 s |

Peak RSS fell **44.6%** in this single bounded mechanism pair. Both paths produced
SHA-256 `bb68d2502b119bca15dbd6b3dee480cc598b68b78153584a9eff3727936c0a6b`.
No swap growth was observed. This is provider serialization evidence, not client
import RSS, whole-decoder peak memory, WAN performance or generation quality.

## Verification and reproduction

Canonical MessagePack boundary cases, fixed compressed frame bytes, native-owner
lifetime, mutable-source isolation, direct artifact import, object/raw corruption,
cache behavior and live prepared/offset HTTP checks passed. Native admission now
charges copied metadata and bounded delivery buffers instead of five full bundle
copies. Client and older-Docker copy bounds remain conservative.

The retained `qwen3-4b-segmented-admission-2026-10-05.json` preflight lowers the
native whole-topology estimate from 23.60 to **21.12 GiB**. It still rejects
against the observed **5.27 GiB** safe headroom; no 4B weights were loaded.

```bash
.venv/bin/python scripts/probe_bundle_memory.py --output /tmp/pllm-bundle-memory-new.json
.venv/bin/pytest -q tests/test_bundle_document.py tests/test_bundle_artifacts.py \
  tests/test_bundle_compression.py tests/test_client_bundle_cache.py \
  tests/test_transformer_engine.py tests/test_artifact_delivery.py tests/test_offset_worker.py
```

The probe requires 2 GiB of available physical RAM after the host reserve and
uses the existing pressure/swap watchdog. It performs no model download.
