# Exact public prepared artifact locality — 2026-10-01

## Result and validated admission

The owned artifact export, authenticated HTTP delivery, reconstruction, and
bounded SDK object-cache path passed real two-child SDK execution on Tiny Qwen2
and cached pinned `Qwen/Qwen2.5-0.5B-Instruct`. Prefill and decode logits are
bit-exact against ordinary raw W8 execution, as are generated output hashes.
The existing loopback benchmark reports `checks.passed=true` and records artifact
body bytes through its existing privacy audit/setup ledger.

**Production SDK admission is validated.** The archived ordinary command exited
successfully with `admission.temporary_whitelist_probe=false`, pinned Qwen
included, and `canonical_benchmark.checks.passed=true`. Constructor, dashboard,
benchmark, and history admission use their normal registered options/fields.
The probe contains no temporary whitelist context, code-object rewriting, or
admission-bypass CLI option. Selection is the executable immutable public
prepared `ClientBundleTransport("artifacts")` profile, with normal native
validation and no raw/zlib fallback.

Numbers below come from the production JSON archive. That archive remains the
record of the full Tiny/pinned-Qwen run; removing the unused admission shim was
subsequently checked with a tiny-only ordinary probe, not another Qwen run.

## Measured artifact bodies

Pinned revision: `7ae557604adf67be50417f59c2c2f167def9a775`.
Model downloads were disabled; checkpoint download bytes are **zero**.

| SDK operation | Tiny object requests / bytes | Qwen object requests / bytes | Qwen manifest bytes |
|---|---:|---:|---:|
| Baseline cold | 19 / 92,680 | 173 / 145,280,349 | 195,074 |
| Baseline fresh SDK, common cache warm | 0 / 0 | 0 / 0 | 195,074 |
| Switch to client attention | 4 / 98,304 | 48 / 44,040,192 | 230,556 |
| Attention fresh SDK, cache warm | 0 / 0 | 0 / 0 | 230,556 |
| Return to baseline, new role children | 0 / 0 | 0 / 0 | 195,074 |
| Corrupt shared object, repair | 1 / 512 | 1 / 3,584 | 195,074 |

The new attention objects are matrix values only: scales were already public
stage metadata and are reused. Token/head storage, norm vectors, tokenizer,
and unchanged public scales are shared. Qwen's retained serialized object payload
after both placements is **189,320,541 bytes**, below the explicitly tested
512 MiB cap. The implementation default is 2 GiB, not 10 GiB.

Every artifact import fetches a fresh authenticated manifest. Warm imports do
not claim zero traffic: control bodies above remain charged to
`PrivacyAudit.bundle_network_bytes`. Existing bundle hit/miss counters retain
bundle-level semantics; `artifact_cache_stats` separately counts objects and
references. A repeated scale reference cannot label a cold bundle as warm.

For both models, raw, zlib, artifact cold/warm, placement switching, and repaired
imports captured both prefill and decode logit arrays. Qwen hashes are
`85a22a5395fb8772dc7dd4e121f7963cb201de55b6de1c022c6217d0505ac831`
and `c8448d44187c11536efefa1f8909adbdf60c1dbb1841742e557e5fcd9be1cc7b`.
All matching-placement controls compare arrays with `np.array_equal`.
Body fingerprint and runtime source identity remain common across placements;
stage commitments change with placement and are checked by normal native import.
Raw transport digests can change across role restarts because original metadata
contains creation time; reconstruction still verifies each exact current raw digest.

## Raw/zlib comparison and compute costs

Raw and zlib controls use separate cold monolithic cache roots. These are real
SDK downloads, not computed compression ratios alone.

| Qwen placement | Raw body | Zlib framed body | Extra diagnostic hash + zlib CPU |
|---|---:|---:|---:|
| Baseline | 145,977,006 | 124,900,312 | 4.069 s |
| Client attention | 190,217,984 | 164,303,219 | 5.390 s |

The diagnostic raw re-download is recorded separately as
`extra_probe_control_download_bytes`; it is not hidden inside request totals.

Qwen baseline artifact cold cache hash/I/O CPU: **0.403 s**; manifest validation
CPU: **0.00884 s**; reconstruction CPU including those object operations:
**0.831 s**. Warm baseline reconstruction still costs **0.724 s**, including
**0.489 s** hash/I/O and **0.00843 s** manifest parsing. These timings are nested,
not additive. Attention-switch reconstruction CPU is **1.169 s**, including
**0.798 s** hash/I/O and **0.01037 s** parsing.

All-linear MAC accounting includes the vocabulary head, excludes token lookup.
Tiny baseline is **96.40%** remote, attention **85.69%** remote. An additional
actual untied Tiny `OutputHeadAtInference` SDK profile is **100%** remote,
passes exact raw logits/output, and measures 0.00973 s hash/I/O CPU, 0.00125 s
manifest CPU, and 0.0139 s reconstruction CPU against 0.1738 s client request CPU.

Qwen baseline is **72.44%** remote of **493,961,216** all-linear MACs per row;
attention is **63.52%**, despite 87.69% of body-only MACs remaining remote.
**Qwen does not satisfy the >74% all-linear threshold.** Its tied head is local;
the installed remote-head runtime rejects tied token/head checkpoints. This
scope blocker is recorded rather than presenting body-only percentages as
all-linear evidence.

These counts are MACs **per evaluated row**, including the whole vocabulary
projection for that row. They are not whole-response MAC totals: prompt/decode
rows, stage calls, and material costs remain recorded separately in the audit.
Final-logit execution can evaluate fewer head rows than body rows, so the
per-row percentages do not establish whole-response remote-compute shares.

## Horizons and ownership

Each source/placement has one measured cold request and one fresh-SDK warm
request. Horizons 10 and 100 are explicitly projections, not repeated loops.
Totals include covered preparation uploads/downloads, pushed corrections,
session authorization, prefill/decode stage bodies, artifact manifests/objects,
and zero checkpoint download bytes. They exclude HTTP/TLS framing and other
control bodies outside the existing audit ledger.

| Qwen placement | 1 request, measured | 10 requests, projected | 100 requests, projected |
|---|---:|---:|---:|
| Baseline covered bodies | 236,730,217 | 1,059,779,029 | 9,290,267,149 |
| Attention switch covered bodies | 120,144,974 | 805,088,012 | 7,654,518,392 |

Attention projections start after the measured baseline cache has been populated.
Fresh one-use material is charged at every projected request; no private cache
reuse is assumed. Every actual SDK client obtains a distinct inventory ID;
reserved rows are not reused. The altered-graph probe fails the original raw
commitment with **zero preparation rows, uploads, or authorization bytes**.

Cache ownership is serialized public objects only. Qwen source checkpoint
storage is **988,097,824 bytes**; original HF ModelSource resolution takes
1.397 s CPU and records its own source lock separately from the effective
runtime path lock. Child startup/source CPU and wall times are separately recorded.
Qwen client NumPy payload is 138,856,448 bytes baseline / 183,093,248 attention;
native CPU int8 snapshot payload is 136,134,656 / 180,174,848 bytes. The
shape-derived native stage payload floor per role is 630,095,872 bytes.
These are distinct ownership categories, **not peak RAM measurements**. Server
raw-bundle and artifact memoization can coexist and also consume memory.

## Supported contract and bounds

- Public compiled schema-2 **prepared** bundles only. Proprietary exports reject.
  Offset live delivery is outside this supported scope.
- Ordered MessagePack skeleton preserves original graph descriptors and map
  insertion order. Each raw SHA-256/size commitment is verified before
  `ClientBundle.unpack` and normal compiled native binding.
- Reusable blobs have content SHA-256 plus canonical source/shape/dtype,
  physical orientation, and numeric domain commitments. Equal bytes with
  changed shape or numeric context cannot reuse the same object identity.
  Tied matrices retain one physical storage object across graph orientations.
- Only validated matrix/vector and tokenizer binary schema fields export;
  unknown private records, wrong dtypes, and binary fields outside that schema
  reject. No model-family or checkpoint tensor-path exceptions are used.
- Original raw admission: 4 GiB, accommodating the stated 2.79 GB baseline
  raw bundle size; manifest: 16 MiB; 16,384 objects; bounded graph depth/nodes,
  arrays/maps/extensions. Boolean integers and duplicate MessagePack keys reject.
- Object HTTP and reassembly use 64 KiB slices; artifact transport performs no
  decompression. Explicit unsupported artifact negotiation fails closed.
- Same inference-origin bearer authorization; manifests contain no object URLs;
  URL paths are constructed locally and redirects are rejected.
- Cache reads verify content and domain; corruption evicts and re-fetches.
  Admission verifies before atomic rename/fsync. Cross-client single-flight,
  byte/count LRU, bounded directory scans, and abandoned-write cleanup apply.
  Readers own verified copies, so eviction cannot destroy active readers.
- Directory descriptors and no-follow opens reject user-controlled symlinks.
  Darwin's system `/var` and `/tmp` aliases are normalized lexically to
  `/private/...`, never traversed as cache symlinks.
- Public cache is `ClientBundleCachePath/public-artifacts`; private KV/material
  paths and native snapshots remain separate. It never stores raw manifests,
  original raw bundles, KV, masks, or one-use material.

## Commands and checks

Executed successfully in the existing `.venv`:

```sh
.venv/bin/maturin develop --release
.venv/bin/python scripts/probe_artifact_locality.py --cached-qwen --cache-mib 512 --canonical-benchmark --archive docs/evidence/artifact-locality-2026-10-01.json
.venv/bin/python -m pytest tests/test_bundle_artifacts.py tests/test_artifact_delivery.py tests/test_bundle_compression.py tests/test_dashboard_history.py -q
.venv/bin/ruff check scripts/probe_artifact_locality.py tests/test_bundle_artifacts.py tests/test_artifact_delivery.py
git diff --check
```

**46 focused tests passed**; scoped Ruff and whitespace checks passed. Named adversarial
tests cover exact raw order/digest, changed numeric/shape domains, private/wrong
dtype/bool records, duplicate keys, invalid references, authenticated object
bindings, old-provider/redirect rejection, corruption repair, byte/count LRU,
active-reader ownership, single-flight, cache modes, no-follow paths, tokenizer
reuse, and tied auxiliary embedding storage.

Tiny-only verification after removing the unused admission shim (separate output,
preserving the production pinned-Qwen archive):

```sh
.venv/bin/python scripts/probe_artifact_locality.py --cache-mib 512 --canonical-benchmark
```

Optional local real checkpoint, including a cached 9 GB ModelSource:
`--cached-model-source /absolute/checkpoint/directory`. That optional source was
not executed in this archive; no source downloads are performed. For explicit
512 MiB public cache ownership outside the probe:

```python
from pllm.runtime.bundle_artifacts import configure_artifact_cache

cache = configure_artifact_cache(client, max_bytes=512 << 20)
```

Selection remains `ClientBundleTransport("artifacts")` in the immutable profile.
Omitted transport and existing `none`/`zlib` parameters retain their original
contract identity. No new component ID, commits, or staging were introduced.
