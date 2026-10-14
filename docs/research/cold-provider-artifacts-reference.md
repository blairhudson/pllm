# Cold provider artifacts: optimization #7

## Decision and scope

**Feasible research boundary, not integrated cold-start optimization.** A provider
can import authenticated role-specific W8A8 bytes and compile native immutable
matrices without downloading BF16 or quantizing locally. The fixture demonstrates
this for generated tiny tensors using installed SDK APIs. It does not demonstrate
a complete decoder, portable production provider checkpoint, network throughput,
or a speedup. No actual checkpoint values or timings were loaded/measured.

The security distinction is essential: an independent trusted publisher key
authenticates the publisher's statement about source and compilation. Hashing
every delivered object verifies those objects, **not absent original BF16 files**.
An original source digest written in a manifest is not proof that quantized bytes
derive from that source. The fixture includes a deliberately dishonest signed
publisher case: consistently replacing weights and digests still imports. A
consumer that requires independent derivation must download/requantize original
bytes, obtain an independently trusted reproducible-build attestation, or add a
verified transformation proof. None is supplied by this research fixture.

Consumer hardware/software privacy assumptions remain unchanged: no TEE, no
private prompt/KV/preprocessing data in public artifacts, no collapsed Inference
and Preparation trust roles. Authentic model delivery does not prove honest
execution or strengthen privacy against role collusion.

## Actual repository behavior

| Boundary | Source citation | Consequence |
|---|---|---|
| Source identity | `python/pllm/model_loader.py:_source_lock` (144–234), `_file_digest` | Rehashes actual source files; validates recorded size/hash, Hub blob ETag identity where applicable, missing files, and stat changes during hashing. Config-only identity is not a weight-checkpoint lock. |
| Party bootstrap | `python/pllm/runtime/party.py:PartyController.start` (98–146) | Requires already installed checkpoint directory; actual source lock must match configured lock before role engines and authenticated offers. No portable role-artifact route here. |
| Role factory | `python/pllm/runtime/servers.py:load_role_adapter` (32–93) | Preparation: `weight_residency="provider"`; Inference: `"provider_and_bundle"`. Both ordinarily resolve/load source. |
| Retention | `python/pllm/runtime/transformer_engine.py` (874–888, 1343–1452, 1763–1766) | Even omitted stages are quantized before `release_weight()`. Preparation retains provider-owned linear stages; Inference also retains bundle boundaries/local body placements. Tied main token/head table can avoid duplicate token retention, but auxiliary slices remain independent. Remote head and client placement change ownership. |
| Private compiled cache | Same file, `_stage_cache_key`, `_quantize_sources`, `_quantize_sources_disk` (1082–1341); `python/pllm/runtime/safetensors_store.py:source_signature` (203–221) | Large-stage cache key includes source signatures, orientation, stage/bits/equalization. Source signatures contain filename/size/mtime/shape/dtype, not file-content hash. Cache hit checks file size and metadata then opens mappings; this is not signed public delivery or independent source-to-W8 proof. Ordinary bootstrap source hashing remains separate. Small stages follow quantization path directly. |
| Runtime binding | `python/pllm/runtime/model_binding.py:StageBinding`, `CompiledRuntimeModel._validated_schedule` (236 onward, 362–372) | Bound IDs/weights/scales/bias/tokenizer/config/plan/composition must still match. A research stage-list hash cannot replace actual production model-plan and runtime binding. |
| Public client bundle | `python/pllm/runtime/bundle_artifacts.py` module contract, `export_bundle`; `scripts/probe_client_storage.py` | Existing schema-2 delivery preserves exact raw bundle commitment and ordered MessagePack representations. It delivers client tensors, not missing provider body matrices; cache accounting excludes native/source snapshots. Client probe discards remote weights and is explicitly not provider execution. |
| Load-ready execution capsule | `python/pllm/runtime/execution_capsule.py` (1–67), `scripts/probe_execution_capsule.py` (29–44) | Capsule requires trusted digest/build/compiled binding. Probe constructs full fixture before importing capsule. Instructions/metadata ownership does not transport missing provider weights or establish BF16-free provider bootstrap. |
| Role memory bill | `python/pllm/runtime/benchmark_memory.py` (382–415) | Separate role retention/loading, delivery/cache work, omitted-stage staging and correction/preparation work are charged. Disk-cache eviction does not retire existing immutable native readers. |

Repository references are locked by file SHA-256 in evidence. Installed quantizer,
wrapper, and native extension are locked separately, because execution uses
`target/frontier-sdk-env`, not upstream Python runtime imports.

## Bounded independently replayable fixture

Files: `scripts/probe_cold_provider_artifacts.py`,
`tests/test_cold_provider_artifacts.py`, and
`docs/evidence/cold-provider-artifacts-synthetic.json`.

Four generated BF16 matrices have exact representable values. Source witness
contains full file sizes/SHA-256, ordered tensor IDs/shapes/offsets/orientation,
source revision, config, tokenizer, norm, and bias identities. Publisher hashes
complete generated source bytes before installed per-row W8 quantization.

Role artifact contains canonical JSON manifest, 64-byte Ed25519 signature,
framing, and ordered raw payload. Signature binds source witness, role, synthetic
plan/composition/compiler and actual installed execution build hashes; importer
requires external expected binding and public key. Every payload byte is claimed
by contiguous ordered extents and hashed before native compilation. Weight/scales
digest, exact dtype/shape, W8A8 IDs/order/retention, tokenizer/local-source contents
and finite positive scales are checked. Public deterministic signing key is for
tests only: **it provides no real-world authentication**.

Inference artifact retains all four matrices plus exact config/tokenizer/local
norm/bias bytes. Preparation retains two body matrices, with all stage/source
witness metadata still charged. Synthetic model is untied; no tied-table saving
is claimed. Cold importer never calls source generation or weight quantization.
It compiles raw W8 bytes with installed native `MaskedGEMM(threads=1, build=False)`, compares
every retained weight/scale byte to publication and integer outputs to independent
NumPy int64 products using W8A8 activation values. It does not claim full biased,
dequantized decoder execution or end-to-end masked preparation correctness.

Tests cover payload/manifest/signature tampering, wrong key, truncation/trailing
bytes, external source/model/composition/compiler/runtime/role mismatches, signed
stage/object-order/offset/shape/retention substitutions, scale-digest mismatch,
individual weight/scales/tokenizer/norm/bias changes, original-source mutation,
immutable import surviving delivery-buffer mutation, and execution rejection
after importer retirement. Supported bytes/bytearray/memoryview inputs are checked
using buffer byte count before immutable copying; unsupported types and over-cap
typed views are rejected. A bytearray conversion sentinel proves oversized input
rejection precedes conversion. Unsupported expected roles are rejected even with
a valid test-publisher signature. Construction failures (constructor, absent
native executor, or partial matrix compilation) retire references; successful
close is idempotent and releases matrices, payloads, and executor owner.
Private in-process snapshot retirement means references
are dropped; it does not promise OS secure erasure or revocation of copies held
elsewhere. No private inventory, KV, masks, or corrections are exported.

Replay (serial; no checkpoint downloads):

```sh
target/frontier-sdk-env/bin/python -m pytest -q -o pythonpath=. tests/test_cold_provider_artifacts.py
target/frontier-sdk-env/bin/python scripts/probe_cold_provider_artifacts.py --output docs/evidence/cold-provider-artifacts-synthetic.json
```

Evidence intentionally contains no clocks or RSS timing claims. Installed binary
changes alter signed artifact hashes. Re-running on unchanged files/environment
is deterministic; compare evidence replay locks before treating a replay as exact.
Evidence separately prices complete artifact bytes and out-of-band fixture public
key/expected-binding bytes; real attestation/certificate bootstrap is additional
and unspecified. Native build-file rehash I/O remains charged, not a timing claim.

## Full cold bill versus warm providers

Let `B` be sum of body matrix elements retained by Preparation, `I` all distinct
Inference/bundle matrix elements after legitimate tied-source sharing, `rP/rI`
row counts, `L` local tensors/tokenizer/config payload, `MP/MI` complete witness,
stage/source, compiler/execution metadata and signatures. For this layout:

* Preparation delivery: `B + 4*rP + MP` bytes.
* Inference delivery: `I + 4*rI + L + MI` bytes.
* Both-role cold distribution: **sum**, not one artifact; shared physical caches
  may deduplicate public bytes but must not collapse role credentials/private state.
* Include BF16 source download/storage/full-file hashing plus compilation at
  publisher once, and trust-root/attestation bootstrap at each cold consumer.
* Each cold provider still pays transport, every-object hashing, parsing,
  installed runtime startup, native compilation/snapshot allocation, metadata,
  and preparation/private session work. A raw i8 artifact is execution-ready data,
  not an architecture-independent serialized native matrix.
* Transient peak includes serialized input + object copies + immutable native
  snapshots; production delivery needs bounded streamed verification/atomic
  immutable publication. Fixture is capped at 1 MiB per envelope and has less
  than 1 MiB generated arrays, not a large-model memory demonstration.
* Keep source store, private compiled disk cache, public delivery/object cache,
  immutable execution snapshots, and private inventory/KV as separate ledgers.
  Eviction of one is not deletion of the others.

If both providers are already loaded, avoided provider source setup is **zero**
for the measured request. If only client is cold, this proposal does not reduce
client bundle or token/head import; client-only cold gain may be **zero**. A fresh
process with existing compiled/source cache is disk-warm, not fully cold. Only
fully cold provider(s), with separately priced publisher and authentication
bootstrap, can claim avoiding original download/requantization. No timing benefit
is established here.

## Exact next integration gate and serial sizing proposal

Before any production change: independently authorize publisher/source/build
attestation policy; pin actual source lock/checkpoint, semantic stage source order,
orientation, bits/scales/bias/equalization, tokenizer/local tensors, full model
plan/composition/runtime/config/body/seeded commitments. Export artifacts from
ordinary verified loader, then import into fresh isolated Inference and Preparation
engines with source checkpoint **unavailable**. Recompute production binding and
all public commitments, execute exact prefill/decode integer stages and complete
response/KV parity, preserve separate authenticated offers/credentials/privacy,
and test mismatch/mutation/retirement across disk snapshots and active readers.
Current `PartyController.start` requires source: this gate is unmet and must not
be bypassed by injecting self-asserted source-lock metadata.

Parent may next do a **serial HEADER-only scan** of public safetensors files,
without checkpoint value reads or timing work. Fetch 8-byte header length, then
bounded JSON header via Range; require HTTP 206/exact Content-Range (abort on
server returning whole body), pin repository commit/file sizes and available
upstream content identities, reject oversized headers before allocation. Enumerate
actual semantic source/orientation/tied aliases and compare with config plan;
count BF16 payload separately from full source file sizes and source witnesses.
HEADER identities alone do not verify payload or successful quantization.

Resource estimate for parent: one shard at a time, **16 MiB maximum header**,
about **64 MiB extra RAM headroom** for JSON and descriptors, no added tensor
arrays/native model snapshots; network `sum(8 + header_length)` plus bounded
config/index/tokenizer metadata. Disk stays below 32 MiB if headers are discarded
after recording sizes/hashes. These are conservative caps, not measured peaks.
Derive role byte formulas above from schedule, not parameter-count shortcuts;
claim no actual Qwen/Gemma size until this scan is completed.

## Primary specifications

* [RFC 8032, sections 5.1.6–5.1.7](https://www.rfc-editor.org/rfc/rfc8032#section-5.1.7): Ed25519 sign/verify; key trust and derivation attestation are external.
* [Safetensors official format](https://github.com/huggingface/safetensors#format): eight-byte header length, JSON dtype/shape/data offsets, followed by tensor bytes; headers are not payload verification.
* [Hugging Face Hub download API](https://huggingface.co/docs/huggingface_hub/package_reference/file_download): pinned revision/commit and ETag metadata distinguish repository/blob identity from local verified contents.
