from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import msgpack
import pytest

from pllm.runtime.bundle_artifacts import (
    ArtifactError, ArtifactObjectCache, export_bundle, parse_manifest, reconstruct_bundle,
    reconstruct_document,
)
from pllm.runtime.bundle_document import BundleDocument
from pllm.runtime.transformer_client import ClientBundle
from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_engine import MaskedTransformerEngine
from pllm import Model
from pllm.model_loader import resolve_model


@pytest.fixture
def bundles(tmp_path):
    root = create_tiny_llama_checkpoint(tmp_path / "weights", hidden_size=128,
                                       intermediate_size=512, head_dim=32)
    result = []
    for roles in ((), ("attention_output", "qkv_projection")):
        engine = MaskedTransformerEngine(threads=1, client_linear_roles=roles)
        asyncio.run(engine.load(resolve_model(Model.path(str(root), model_id="artifact-model")).manifest))
        result.append(engine.client_bundle("artifact-model"))
    return result


def parsed(payload):
    exported = export_bundle(payload)
    manifest = parse_manifest(exported.manifest, fingerprint=hashlib.sha256(payload).hexdigest(),
                              size=len(payload))
    return exported, manifest


def test_original_raw_digest_order_and_common_objects(bundles):
    baseline, attention = [parsed(payload) for payload in bundles]
    for payload, (exported, manifest) in zip(bundles, (baseline, attention), strict=True):
        assert reconstruct_bundle(manifest, lambda row: exported.objects[row["sha256"]]) == payload
    base_keys, attention_keys = [set(e.objects) for e, _ in (baseline, attention)]
    assert base_keys <= attention_keys
    missing = attention_keys - base_keys
    assert missing
    # Only newly placed matrix values; their scales were already public stage metadata.
    assert all(row["metadata"]["dtype"] == "i1" for row in attention[1]["objects"]
               if row["sha256"] in missing)


def test_segmented_export_and_direct_import_preserve_raw_commitment(bundles, monkeypatch):
    raw = bundles[0]
    document = BundleDocument(msgpack.unpackb(raw, raw=False))
    expected = export_bundle(raw)
    exported = export_bundle(document)
    assert exported.manifest == expected.manifest
    assert exported.objects == expected.objects
    manifest = parse_manifest(exported.manifest, fingerprint=hashlib.sha256(raw).hexdigest(), size=len(raw))
    ordinary = ClientBundle.unpack(raw)
    reads = []
    def read(row):
        reads.append(row["sha256"])
        return bytes(exported.objects[row["sha256"]])
    def forbidden(*args, **kwargs):
        raise AssertionError("direct import must not build or unpack a contiguous bundle")
    monkeypatch.setattr(msgpack, "packb", forbidden)
    monkeypatch.setattr(msgpack, "unpackb", forbidden)
    value = reconstruct_document(manifest, read)
    imported = ClientBundle._from_document(value)
    assert imported.cfg == ordinary.cfg
    assert imported.privacy == ordinary.privacy
    assert len(reads) == len(set(reads)) == len(exported.objects)
    for stage_id, stage in imported.stages.items():
        import numpy as np
        control = ordinary.stages[stage_id]
        np.testing.assert_array_equal(stage.weight_scales, control.weight_scales)
        if stage.client_weight is not None:
            np.testing.assert_array_equal(stage.client_weight, control.client_weight)
            assert not stage.client_weight.flags.writeable


def test_direct_import_checks_object_and_whole_bundle_before_return(bundles):
    exported, manifest = parsed(bundles[0])
    def read(row):
        return exported.objects[row["sha256"]]
    with pytest.raises(ArtifactError, match="content digest"):
        reconstruct_document(manifest, lambda row: bytes(row["size"]))
    with pytest.raises(ArtifactError, match="immutable"):
        reconstruct_document(manifest, lambda row: bytearray(read(row)))
    manifest["raw"]["sha256"] = "0" * 64
    with pytest.raises(ArtifactError, match="raw digest"):
        reconstruct_document(manifest, read)


def test_artifact_export_preserves_subclass_privacy_boundary(bundles, tmp_path):
    value = msgpack.unpackb(bundles[0], raw=False)
    value["privacy"]["mode"] = "proprietary"
    payload = msgpack.packb(value, use_bin_type=True)
    class RestrictedEngine(MaskedTransformerEngine):
        def client_bundle(self, model_id):
            return payload
    root = create_tiny_llama_checkpoint(tmp_path / "restricted")
    engine = RestrictedEngine(threads=1)
    asyncio.run(engine.load(resolve_model(Model.path(str(root), model_id="restricted")).manifest))
    with pytest.raises(ArtifactError, match="public"):
        engine.client_bundle_artifacts("restricted")


def test_equal_bytes_changed_numeric_or_shape_domain_never_reuse(bundles):
    original, _ = parsed(bundles[0])
    value = msgpack.unpackb(bundles[0], raw=False)
    key = next(iter(value["local_tensors"]))
    row = value["local_tensors"][key]
    row["shape"] = [1, *row["shape"]]
    changed, _ = parsed(msgpack.packb(value, use_bin_type=True))
    assert len(set(changed.objects) - set(original.objects)) == 1
    value = msgpack.unpackb(bundles[0], raw=False)
    for stage in value["stages"].values():
        stage["activation_bits"] = 7
    numeric, _ = parsed(msgpack.packb(value, use_bin_type=True))
    assert set(numeric.objects) != set(original.objects)


@pytest.mark.parametrize("change", ["private", "private_binary", "dtype", "bool_shape", "schema"])
def test_private_records_and_malformed_matrix_rejected(bundles, change):
    value = msgpack.unpackb(bundles[0], raw=False)
    if change == "private":
        value["privacy"]["mode"] = "proprietary"
    elif change == "private_binary":
        value["privacy"]["zeroPrivateRecord"] = b"\0" * 16
    elif change == "dtype":
        next(iter(value["client_weights"].values()))["dtype"] = "u1"
    elif change == "bool_shape":
        next(iter(value["local_tensors"].values()))["shape"] = [True]
    else:
        value["v"] = True
    with pytest.raises(ArtifactError):
        export_bundle(msgpack.packb(value, use_bin_type=True))


@pytest.mark.parametrize("change", ["bool_size", "domain", "unused", "ref", "bad_raw", "duplicate"])
def test_manifest_validation_before_any_object_request(bundles, change):
    exported, manifest = parsed(bundles[0])
    value = msgpack.unpackb(exported.manifest, raw=False)
    if change == "bool_size":
        value["objects"][0]["size"] = True
    elif change == "domain":
        value["objects"][0]["metadata"]["dtype"] = "f8"
    elif change == "unused":
        value["objects"].append(value["objects"][0])
    elif change == "bad_raw":
        value["raw"]["sha256"] = "0" * 64
    elif change == "ref":
        graph = msgpack.unpackb(value["skeleton"], raw=False)
        stage = next(iter(graph["stages"].values()))
        stage["weight_scales"] = msgpack.ExtType(42, (65535).to_bytes(4, "big"))
        value["skeleton"] = msgpack.packb(graph, use_bin_type=True)
    else:
        # Duplicate top-level keys must not silently overwrite commitments.
        document = b"\x86" + msgpack.packb("schema") + msgpack.packb(1)
        document += msgpack.packb(value, use_bin_type=True)[1:]
        with pytest.raises(ArtifactError):
            parse_manifest(document, fingerprint=manifest["raw"]["sha256"], size=manifest["raw"]["size"])
        return
    with pytest.raises(ArtifactError):
        parse_manifest(msgpack.packb(value, use_bin_type=True),
                       fingerprint=manifest["raw"]["sha256"], size=manifest["raw"]["size"])


def test_raw_commitment_rejects_changed_graph_after_verified_objects(bundles):
    exported, manifest = parsed(bundles[0])
    manifest["graph"]["tokenizer"]["chat_template"] = "altered"
    with pytest.raises(ArtifactError, match="raw digest|admission cap"):
        reconstruct_bundle(manifest, lambda row: exported.objects[row["sha256"]])


def test_cache_corruption_atomic_cap_lru_and_active_copy(bundles, tmp_path):
    exported, manifest = parsed(bundles[0])
    rows = sorted(manifest["objects"], key=lambda r: r["size"])
    first, second = rows[:2]
    cap = max(first["size"], second["size"])
    cache = ArtifactObjectCache(tmp_path / "public", max_bytes=cap)
    payload = exported.objects[first["sha256"]]
    cache.put(first, payload)
    lease = cache.get(first)
    cache.put(second, exported.objects[second["sha256"]])
    assert lease == payload  # reader owns bytes; eviction cannot destroy lease
    assert cache.stats.retained_payload_bytes <= cap
    assert cache.stats.evictions == 1
    path = cache.root / (second["sha256"] + ".blob")
    path.write_bytes(b"x" * second["size"])
    assert cache.get(second) is None
    assert cache.stats.corruptions == 1 and not path.exists()
    with pytest.raises(ArtifactError):
        cache.put(first, b"bad")
    assert not list(cache.root.glob(".pending-*"))
    assert not list(cache.root.glob("*.msgpack"))


def test_cache_no_symlink_traversal(bundles, tmp_path):
    exported, manifest = parsed(bundles[0])
    row = manifest["objects"][0]
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    cache = ArtifactObjectCache(link)
    with pytest.raises(OSError):
        cache.put(row, exported.objects[row["sha256"]])
    cache = ArtifactObjectCache(real)
    target = tmp_path / "private"
    target.write_bytes(exported.objects[row["sha256"]])
    (real / (row["sha256"] + ".blob")).symlink_to(target)
    with pytest.raises(OSError):
        cache.get(row)
    assert target.read_bytes() == exported.objects[row["sha256"]]


def test_cross_client_singleflight_download_and_cache_modes(bundles, tmp_path):
    exported, manifest = parsed(bundles[0])
    row = manifest["objects"][0]
    payload = exported.objects[row["sha256"]]
    requests, lock = [], threading.Lock()

    def fetch():
        with lock:
            requests.append(1)
        time.sleep(.01)
        return payload

    def load(_):
        cache = ArtifactObjectCache(tmp_path / "common")
        return cache.fetch(row, fetch)

    with ThreadPoolExecutor(max_workers=4) as executor:
        assert list(executor.map(load, range(4))) == [payload] * 4
    assert len(requests) == 1
    assert ArtifactObjectCache(tmp_path / "common", mode="read-only").fetch(row, fetch) == payload
    assert len(requests) == 1
    assert ArtifactObjectCache(tmp_path / "common", mode="refresh").fetch(row, fetch) == payload
    assert len(requests) == 2
    assert ArtifactObjectCache(tmp_path / "off", mode="off").fetch(row, fetch) == payload
    assert not (tmp_path / "off").exists()


def test_tied_matrix_single_storage_with_auxiliary_lookup(tmp_path):
    from pllm.runtime.tiny_gemma import create_tiny_gemma4_checkpoint

    root = create_tiny_gemma4_checkpoint(tmp_path / "gemma", num_hidden_layers=1)
    engine = MaskedTransformerEngine(threads=1)
    asyncio.run(engine.load(resolve_model(Model.path(str(root), model_id="tied")).manifest))
    payload = engine.client_bundle("tied")
    exported, manifest = parsed(payload)
    assert reconstruct_bundle(manifest, lambda row: exported.objects[row["sha256"]]) == payload
    value = msgpack.unpackb(payload, raw=False)
    assert value["stages"]["token_lookup"]["client_weight"]["ref"] == "tied_embeddings"
    tied_shape = value["client_weights"]["tied_embeddings"]["shape"]
    assert sum(row["metadata"]["dtype"] == "i1" and row["metadata"]["shape"] == tied_shape
               for row in manifest["objects"]) == 1


def test_public_tokenizer_binary_object_is_reusable_and_unknown_binary_rejected(bundles):
    value = msgpack.unpackb(bundles[0], raw=False)
    value["tokenizer"] = {"kind": "tokenizer_json", "model": b'{"version":"1.0"}'}
    payload = msgpack.packb(value, use_bin_type=True)
    exported, manifest = parsed(payload)
    tokenizer = [row for row in manifest["objects"] if row["metadata"]["orientation"] == "serialized-tokenizer"]
    assert len(tokenizer) == 1
    assert tokenizer[0]["metadata"]["dtype"] == "u1"
    assert reconstruct_bundle(manifest, lambda row: exported.objects[row["sha256"]]) == payload
    value["tokenizer"]["private_record"] = b"\0" * 16
    with pytest.raises(ArtifactError, match="outside public"):
        export_bundle(msgpack.packb(value, use_bin_type=True))


def test_cache_object_count_and_abandoned_atomic_write_bound(bundles, tmp_path, monkeypatch):
    from pllm.runtime import bundle_artifacts

    exported, manifest = parsed(bundles[0])
    monkeypatch.setattr(bundle_artifacts, "_MAX_CACHE_OBJECTS", 2)
    cache = ArtifactObjectCache(tmp_path / "bounded", max_bytes=1 << 20)
    rows = manifest["objects"][:3]
    cache.put(rows[0], exported.objects[rows[0]["sha256"]])
    pending = cache.root / (".pending-" + "0" * 32)
    pending.write_bytes(b"abandoned")
    cache.put(rows[1], exported.objects[rows[1]["sha256"]])
    assert not pending.exists()
    lease = cache.get(rows[0])
    cache.put(rows[2], exported.objects[rows[2]["sha256"]])
    assert len(list(cache.root.glob("*.blob"))) == 2
    assert lease == exported.objects[rows[0]["sha256"]]
    assert cache.get(rows[1]) is None  # LRU, not insertion-order eviction.
