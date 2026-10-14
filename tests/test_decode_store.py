"""Storage-only adversarial units plus parent-run generated native bit oracles.

Native tests build synthetic model fixtures, not real checkpoints or providers.
They are intentionally not run by the implementation worker before wheel rebuild.
"""
from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest
from cryptography.exceptions import InvalidTag
from filelock import Timeout

from pllm.runtime.decode_store import DecodeStore, _DOMAIN, _preflight_metadata
from pllm.runtime.tiered_execution import _canonical
from pllm.runtime.transformer_client import LayerCache, RuntimeSnapshot, _snapshot_state_basis
from pllm.state.decode import DecodeCache, DecodeCacheHead


class StorageContract:
    """Storage unit stand-in only; native tests below use real native contracts."""
    digest = "c" * 64

    def __init__(self):
        self.admissions = []

    def to_dict(self):
        return {"token_bound": 32}

    def admit(self, position, count):
        assert 0 < position < 32 and count == 1
        self.admissions.append((position, count))


class StorageRuntime:
    default_kv_heads = 1
    default_head_dim = 2
    layers = 2
    cfg = {"vocab_size": 8}
    _state_binding_digest = "b" * 64

    def _validate_full_snapshot(self, snapshot):
        assert snapshot.semantic_binding_digest == self._state_binding_digest
        assert not snapshot.shared_kv and len(snapshot.caches) == 2
        for row in snapshot.caches:
            assert row.length == snapshot.position
            assert row.key.shape == row.value.shape == (snapshot.position, 1, 2)
            assert row.key.dtype == row.value.dtype == np.float32
            assert np.all(np.isfinite(row.key)) and np.all(np.isfinite(row.value))

    def _validate_verification_basis(self, basis):
        assert basis.valid() and not basis.verification_failure_bits

    def _cache_basis_valid(self, basis):
        return basis.cache_eligible(generated_prefixes=True)


@pytest.fixture
def storage_binding(monkeypatch):
    def binding(self, compiled, runtime):
        return hashlib.sha256(_DOMAIN + compiled.digest.encode() + _canonical(self.limits)).hexdigest(), StorageContract()
    monkeypatch.setattr(DecodeStore, "_binding", binding)
    return SimpleNamespace(digest="d" * 64), StorageRuntime()


def state(ids):
    position = len(ids)
    caches = []
    for layer in range(2):
        key = np.asarray([[t + layer, -t] for t in ids], np.float32).reshape(position, 1, 2)
        caches.append(LayerCache(key, key + np.float32(0.5), position))
    basis = _snapshot_state_basis("canonical_incremental", position, "b" * 64, "c" * 64, "e" * 64,
                                  None, position)
    return RuntimeSnapshot(position, caches, {}, "b" * 64, basis), np.arange(8, dtype=np.float32)


def opened(tmp_path, binding, **kwargs):
    cache = DecodeCache(tmp_path / "decode", b"k" * 32, **kwargs)
    cache.bind(*binding)
    return cache


def same(left, right):
    assert left[0].position == right[0].position
    assert np.array_equal(left[1].view(np.uint32), right[1].view(np.uint32))
    for a, b in zip(left[0].caches, right[0].caches, strict=True):
        for kind in ("key", "value"):
            assert np.array_equal(getattr(a, kind).view(np.uint32), getattr(b, kind).view(np.uint32))
            assert not np.shares_memory(getattr(a, kind), getattr(b, kind))


def test_stage_ack_branches_private_index_and_independent_restart(tmp_path, storage_binding):
    cache = opened(tmp_path, storage_binding)
    root = state([1, 2])
    root_ticket = cache.stage([1, 2], *root)
    assert cache.lookup([1, 2]) is None and cache.head is None
    with pytest.raises(ValueError, match="acknowledgement"):
        cache.commit([root_ticket], acknowledged=False)
    root_head = cache.commit([root_ticket], acknowledged=True)
    a, b = state([1, 2, 3]), state([1, 2, 4])
    tickets = [cache.stage([1, 2, 3], *a), cache.stage([1, 2, 4], *b)]
    assert cache.lookup_child([1, 2], 3) is None
    head = cache.commit(tickets, acknowledged=True)
    assert DecodeCacheHead.from_spec(head.to_spec()) == head
    same(cache.lookup_child([1, 2], 3), a)
    same(cache.lookup_child([1, 2], 4), b)
    assert cache.storage_metrics["page_count"] == 16  # 4 rows * 2 layers * K/V
    assert cache.storage_metrics["node_count"] == 3
    ticket = cache.stage([1, 2, 5], *state([1, 2, 5]))
    cache.discard([ticket])
    assert cache.lookup_child([1, 2], 5) is None
    assert cache.storage_metrics["unpublished_bytes"] > 0
    cache.close()
    with pytest.raises(ValueError, match="closed"):
        cache.lookup([1, 2])
    for path in (tmp_path / "decode").iterdir():
        if path.name != ".writer":
            assert path.stat().st_mode & 0o077 == 0
            assert b"canonical_incremental" not in path.read_bytes()
    reopened = DecodeCache(tmp_path / "decode", b"k" * 32, head=head)
    reopened.bind(*storage_binding)
    restored = reopened.lookup([1, 2, 3])
    assert restored.snapshot is restored[0] and restored.logits is restored[1]
    same(restored, a)
    assert restored[0].state_basis.valid()
    assert restored[0].state_basis.seal != a[0].state_basis.seal
    assert reopened._store.contract.admissions
    reopened.close()
    # Deliberate caller-selected old head remains a fork; newer disk index ignored.
    old = DecodeCache(tmp_path / "decode", b"k" * 32, head=root_head)
    old.bind(*storage_binding)
    assert old.lookup([1, 2, 3]) is None
    same(old.lookup([1, 2]), root)
    old.close()


def test_restart_reissues_basis_under_new_process_secret(tmp_path, storage_binding, monkeypatch):
    from pllm.runtime import transformer_client
    cache = opened(tmp_path, storage_binding)
    original = state([1, 2])
    head = cache.commit([cache.stage([1, 2], *original)], acknowledged=True)
    cache.close()
    # Storage-only model of interpreter restart: old provenance loses authority.
    # Native subprocess coverage below is reserved for parent execution.
    monkeypatch.setattr(transformer_client, "_STATE_BASIS_KEY", os.urandom(32))
    assert not original[0].state_basis.valid()
    reopened = DecodeCache(tmp_path / "decode", b"k" * 32, head=head)
    reopened.bind(*storage_binding)
    restored = reopened.lookup([1, 2])
    same(restored, original)
    assert restored[0].state_basis.valid()
    reopened.close()


def test_pending_owned_state_and_bounds_fail_before_writes(tmp_path, storage_binding):
    cache = opened(tmp_path, storage_binding, max_nodes=1, max_staged=1)
    snapshot, logits = state([1, 2])
    before = cache.storage_metrics["written_bytes"]
    with pytest.raises(ValueError, match="pending"):
        cache.stage([1, 2, 3], snapshot, logits)
    owned = replace(snapshot, state_basis=snapshot.state_basis._with_owner("response"))
    with pytest.raises(ValueError, match="owner-free"):
        cache.stage([1, 2], owned, logits)
    bad = replace(snapshot, state_basis=replace(snapshot.state_basis, phase="incremental"))
    with pytest.raises(ValueError, match="provenance"):
        cache.stage([1, 2], bad, logits)
    assert cache.storage_metrics["written_bytes"] == before
    ticket = cache.stage([1, 2], snapshot, logits)
    before = cache.storage_metrics["written_bytes"]
    with pytest.raises(ValueError, match="staged batch"):
        cache.stage([1, 2, 3], *state([1, 2, 3]))
    cache.commit([ticket], acknowledged=True)
    with pytest.raises(ValueError, match="node count"):
        cache.stage([1, 2, 3], *state([1, 2, 3]))
    # Only ACK index write occurred after the saved counter.
    assert cache.storage_metrics["written_bytes"] == before + cache.head.bytes
    cache.close()


def test_writer_client_exclusion_wrong_key_context_and_no_disk_latest(tmp_path, storage_binding):
    cache = opened(tmp_path, storage_binding)
    client = object()
    cache.claim(client)
    cache.claim(client)
    with pytest.raises(ValueError, match="another client"):
        cache.claim(object())
    head = cache.commit([cache.stage([1], *state([1]))], acknowledged=True)
    second = DecodeCache(tmp_path / "decode", b"k" * 32, head=head)
    with pytest.raises(Timeout):
        second.bind(*storage_binding)
    second.close()
    cache.close()
    with pytest.raises(FileExistsError):
        DecodeCache(tmp_path / "decode", b"k" * 32).bind(*storage_binding)
    wrong = DecodeCache(tmp_path / "decode", b"w" * 32, head=head)
    with pytest.raises(InvalidTag):
        wrong.bind(*storage_binding)
    wrong.close()
    wrong_context = DecodeCache(tmp_path / "decode", b"k" * 32, head=head)
    with pytest.raises(ValueError, match="context"):
        wrong_context.bind(SimpleNamespace(digest="f" * 64), storage_binding[1])
    wrong_context.close()


def test_failed_publication_preserves_trusted_head_and_retires_attempt(tmp_path, storage_binding, monkeypatch):
    cache = opened(tmp_path, storage_binding)
    root = state([1, 2])
    head = cache.commit([cache.stage([1, 2], *root)], acknowledged=True)
    ticket = cache.stage([1, 2, 3], *state([1, 2, 3]))
    before = cache.storage_metrics["disk_bytes"]
    original = cache._store._write

    def failed_index(raw, metadata):
        result = original(raw, metadata)
        if "index" in metadata:
            raise OSError("injected post-write failure")
        return result

    monkeypatch.setattr(cache._store, "_write", failed_index)
    with pytest.raises(OSError, match="injected"):
        cache.commit([ticket], acknowledged=True)
    assert cache.head == head
    assert cache.storage_metrics["staged_count"] == 0
    assert cache.storage_metrics["disk_bytes"] > before
    assert cache.lookup_child([1, 2], 3) is None
    same(cache.lookup([1, 2]), root)
    cache.close()
    reopened = DecodeCache(tmp_path / "decode", b"k" * 32, head=head)
    reopened.bind(*storage_binding)
    same(reopened.lookup([1, 2]), root)
    reopened.close()


@pytest.mark.parametrize("damage", ["tamper", "truncate"])
def test_numeric_page_damage_returns_no_state_and_leaves_runtime_untouched(tmp_path, storage_binding, damage):
    cache = opened(tmp_path, storage_binding)
    cache.commit([cache.stage([1, 2], *state([1, 2]))], acknowledged=True)
    runtime = storage_binding[1]
    before = runtime.__dict__.copy()
    page = next(iter(cache._store.pages.values()))
    path = tmp_path / "decode" / page["record"]["name"]
    raw = path.read_bytes()
    path.write_bytes(raw[:-1] if damage == "truncate" else bytes([raw[0] ^ 1]) + raw[1:])
    with pytest.raises(ValueError, match="trusted commitment"):
        cache.lookup([1, 2])
    assert runtime.__dict__ == before
    assert cache.storage_metrics["hits"] == 0
    path.write_bytes(raw)
    same(cache.lookup([1, 2]), state([1, 2]))
    cache.close()


def test_authenticated_invalid_index_rejects_before_snapshot_allocation(tmp_path, storage_binding, monkeypatch):
    cache = opened(tmp_path, storage_binding)
    cache.commit([cache.stage([1, 2], *state([1, 2]))], acknowledged=True)
    doc = {"schema": "pllm.decode_index.v2", "context": cache._store.context,
           "nodes": cache._store.nodes, "pages": cache._store.pages}
    doc = msgpack.unpackb(msgpack.packb(doc), raw=False)
    next(iter(doc["pages"].values()))["start"] = 32
    record = cache._store._write(msgpack.packb(doc), {"index": 1})
    bad_head = DecodeCacheHead(1, cache.head.context, **record)
    cache.close()
    bad = DecodeCache(tmp_path / "decode", b"k" * 32, head=bad_head)
    monkeypatch.setattr(np, "empty", lambda *args, **kwargs: pytest.fail("allocated unauthenticated state"))
    with pytest.raises(ValueError, match="geometry"):
        bad.bind(*storage_binding)
    bad.close()


def test_shared_pages_fit_actual_bound_and_orphans_cannot_evade_disk_bound(tmp_path, storage_binding):
    cache = opened(tmp_path, storage_binding, max_pages=12)
    head = cache.commit([cache.stage([1, 2], *state([1, 2]))], acknowledged=True)
    # Only four child pages added; full prefix is not counted twice.
    cache.commit([cache.stage([1, 2, 3], *state([1, 2, 3]))], acknowledged=True)
    assert cache.storage_metrics["page_count"] == 12
    before = cache.storage_metrics["written_bytes"]
    with pytest.raises(ValueError, match="page count"):
        cache.stage([1, 2, 4], *state([1, 2, 4]))
    assert cache.storage_metrics["written_bytes"] == before
    cache.close()
    # Explicit old head still charges all newer and orphan files during reopen.
    extra = tmp_path / "decode" / ("f" * 32)
    extra.write_bytes(b"orphan" * 100)
    reopened = DecodeCache(tmp_path / "decode", b"k" * 32, head=head, max_pages=12)
    reopened.bind(*storage_binding)
    assert reopened.storage_metrics["disk_bytes"] >= extra.stat().st_size + head.bytes
    assert reopened.storage_metrics["unpublished_bytes"] >= extra.stat().st_size
    reopened.close()


def test_index_file_and_snapshot_bounds_preflight_without_mutation(tmp_path, storage_binding):
    cache = opened(tmp_path, storage_binding, max_files=1)
    before = cache.storage_metrics.copy()
    with pytest.raises(ValueError, match="disk ownership"):
        cache.stage([1], *state([1]))
    assert cache.storage_metrics["written_bytes"] == before["written_bytes"]
    cache.close()
    small = DecodeCache(tmp_path / "small", b"k" * 32, max_snapshot_bytes=64)
    small.bind(*storage_binding)
    with pytest.raises(ValueError, match="snapshot/transient"):
        small.stage([1, 2], *state([1, 2]))
    assert small.storage_metrics["written_bytes"] == 0
    small.close()
    for name, limits, error in (("index", {"max_index_bytes": 8192}, "metadata"),
                                ("disk", {"max_disk_bytes": 64}, "disk ownership")):
        bounded = DecodeCache(tmp_path / name, b"k" * 32, **limits)
        bounded.bind(*storage_binding)
        with pytest.raises(ValueError, match=error):
            bounded.stage(list(range(8)), *state(list(range(8))))
        assert bounded.storage_metrics["written_bytes"] == 0
        bounded.close()


def test_failed_stage_keeps_prior_head_and_charges_partial_attempt(tmp_path, storage_binding, monkeypatch):
    cache = opened(tmp_path, storage_binding)
    root = state([1, 2])
    head = cache.commit([cache.stage([1, 2], *root)], acknowledged=True)
    before = cache.storage_metrics["disk_bytes"]
    original = cache._store._write

    def failed_page(raw, metadata):
        original(raw, metadata)
        raise OSError("injected staging failure")

    monkeypatch.setattr(cache._store, "_write", failed_page)
    with pytest.raises(OSError, match="staging"):
        cache.stage([1, 2, 3], *state([1, 2, 3]))
    assert cache.head == head and not cache._store.staged
    assert cache.storage_metrics["disk_bytes"] > before
    assert cache.lookup_child([1, 2], 3) is None
    same(cache.lookup([1, 2]), root)
    cache.close()


def test_private_sdk_rollback_empty_and_prior_retains_all_disk_charges(tmp_path, storage_binding):
    cache = opened(tmp_path, storage_binding)
    root = state([1, 2])
    first = cache.commit([cache.stage([1, 2], *root)], acknowledged=True)
    before = cache.storage_metrics.copy()
    cache._restore_head(None)
    assert cache.head is None and cache.lookup([1, 2]) is None
    assert cache.storage_metrics["disk_bytes"] == before["disk_bytes"]
    assert cache.storage_metrics["disk_files"] == before["disk_files"]
    # A fresh publication after empty rollback may reuse revision zero, but the
    # remembered head's commitment, never revision alone, remains authority.
    root_head = cache.commit([cache.stage([1, 2], *root)], acknowledged=True)
    child = state([1, 2, 3])
    new = cache.commit([cache.stage([1, 2, 3], *child)], acknowledged=True)
    ticket = cache.stage([1, 2, 4], *state([1, 2, 4]))
    before = cache.storage_metrics.copy()
    cache._restore_head(root_head)
    assert cache.head == root_head and cache.head != new
    assert cache.lookup_child([1, 2], 3) is None
    same(cache.lookup([1, 2]), root)
    assert cache.storage_metrics["staged_count"] == 0
    assert cache.storage_metrics["disk_bytes"] == before["disk_bytes"]
    assert cache.storage_metrics["written_bytes"] == before["written_bytes"]
    assert cache.storage_metrics["read_bytes"] > before["read_bytes"]
    with pytest.raises(ValueError, match="tickets"):
        cache.commit([ticket], acknowledged=True)
    with pytest.raises(ValueError, match="trusted prior"):
        cache._restore_head(new)
    # Same-revision alternate head is not an authenticated remembered ancestor.
    assert first.revision == root_head.revision and first != root_head
    with pytest.raises(ValueError, match="trusted prior"):
        cache._restore_head(first)
    cache.close()


def test_private_sdk_rollback_damaged_prior_index_fails_closed(tmp_path, storage_binding):
    cache = opened(tmp_path, storage_binding)
    prior = cache.commit([cache.stage([1, 2], *state([1, 2]))], acknowledged=True)
    current = cache.commit([cache.stage([1, 2, 3], *state([1, 2, 3]))], acknowledged=True)
    cache.stage([1, 2, 4], *state([1, 2, 4]))
    path = tmp_path / "decode" / prior.name
    raw = path.read_bytes()
    path.write_bytes(raw[:-1])
    before = cache.storage_metrics.copy()
    with pytest.raises(ValueError, match="trusted commitment"):
        cache._restore_head(prior)
    assert cache.head == current  # No damaged/partially decoded prior index adopted.
    assert cache.storage_metrics["staged_count"] == 0
    assert cache.storage_metrics["read_bytes"] > before["read_bytes"]
    assert cache.storage_metrics["disk_bytes"] == before["disk_bytes"]
    path.write_bytes(raw)
    cache._restore_head(prior)
    assert cache.lookup_child([1, 2], 3) is None
    cache.close()


def test_conflicting_exact_state_cannot_replace_published_path(tmp_path, storage_binding):
    cache = opened(tmp_path, storage_binding)
    source = state([1, 2])
    head = cache.commit([cache.stage([1, 2], *source)], acknowledged=True)
    wrong = source[1] + np.float32(1)
    with pytest.raises(ValueError, match="conflicting"):
        cache.stage([1, 2], source[0], wrong)
    assert cache.head == head and not cache._store.staged
    same(cache.lookup([1, 2]), source)
    cache.close()


def test_close_releases_runtime_client_key_and_never_erases_caller_state(tmp_path, storage_binding):
    cache = opened(tmp_path, storage_binding)
    cache.claim(object())
    snapshot, logits = state([1, 2])
    before = snapshot.caches[0].key.copy()
    cache.stage([1, 2], snapshot, logits)
    files = set((tmp_path / "decode").iterdir())
    cache.close()
    cache.close()
    assert cache._runtime is cache._client is cache._key is cache._store.aead is None
    assert not cache._store.nodes and not cache._store.pages and not cache._store.staged
    assert set((tmp_path / "decode").iterdir()) == files
    assert np.array_equal(snapshot.caches[0].key, before)
    assert np.array_equal(logits, np.arange(8, dtype=np.float32))


class QwenGeometryContract(StorageContract):
    def to_dict(self):
        return {"token_bound": 128}

    def admit(self, position, count):
        assert 0 < position < 128 and count == 1
        self.admissions.append((position, count))


class QwenGeometryRuntime(StorageRuntime):
    """Real Qwen2.5 geometry, synthetic numeric contents, no model/body weights."""
    default_kv_heads = 2
    default_head_dim = 64
    layers = 24
    cfg = {"vocab_size": 151936}

    def _validate_full_snapshot(self, snapshot):
        assert snapshot.semantic_binding_digest == self._state_binding_digest
        assert not snapshot.shared_kv and len(snapshot.caches) == self.layers
        for cache in snapshot.caches:
            assert cache.length == snapshot.position
            for value in (cache.key, cache.value):
                assert value.dtype == np.float32 and value.shape == (snapshot.position, 2, 64)
                assert np.all(np.isfinite(value))


@pytest.fixture
def qwen_geometry_binding(monkeypatch):
    runtime, compiled = QwenGeometryRuntime(), SimpleNamespace(digest="d" * 64)
    contract = QwenGeometryContract()

    def binding(self, source, bound_runtime):
        assert source is compiled and bound_runtime is runtime
        self._candidate_binding_bytes = 64 << 10
        return hashlib.sha256(_DOMAIN + source.digest.encode() + _canonical(self.limits)).hexdigest(), contract

    monkeypatch.setattr(DecodeStore, "_binding", binding)
    return compiled, runtime, contract


def qwen_geometry_state(ids):
    # Every row causally depends on the complete executed prefix, so divergent
    # suffixes genuinely differ while earlier rows remain bit-identical.
    sums = np.cumsum(np.asarray(ids, np.float32), dtype=np.float32)
    base = sums[:, None, None] + np.arange(128, dtype=np.float32).reshape(1, 2, 64) / np.float32(128)
    caches = [LayerCache((base + np.float32(layer)).copy(),
                         (base - np.float32(layer + 0.5)).copy(), len(ids)) for layer in range(24)]
    basis = _snapshot_state_basis("canonical_incremental", len(ids), "b" * 64, "c" * 64,
                                  "e" * 64, None, len(ids))
    logits = np.arange(151936, dtype=np.float32) / np.float32(256) + sums[-1]
    return RuntimeSnapshot(len(ids), caches, {}, "b" * 64, basis), logits


def qwen_geometry_owner(directory, *, head=None):
    return DecodeCache(directory, b"k" * 32, head=head, max_snapshot_bytes=8 << 20,
                       max_transient_bytes=24 << 20, max_staged=8, max_nodes=24)


def test_qwen_geometry_eight_staged_states_three_branches_fit_and_restore_bits(tmp_path, qwen_geometry_binding):
    compiled, runtime, contract = qwen_geometry_binding
    cache = qwen_geometry_owner(tmp_path / "decode")
    cache.bind(compiled, runtime)
    assert cache._store.contract is contract  # One shared already-admitted owner.
    root = list(range(3, 57))  # 54 executed rows, then 7 decode states per request.
    branches = []
    heads = []
    for branch in range(3):
        path, tickets = root.copy(), []
        for step in range(8):
            if step:
                path.append(100 + branch * 100 + step)
            ticket = cache.stage(path, *qwen_geometry_state(path))
            if ticket is not None:
                tickets.append(ticket)
        branches.append(path.copy())
        assert len(tickets) == (8 if branch == 0 else 7)
        assert cache.storage_metrics["staged_count"] == len(tickets)
        # Resident nodes have no full refs/logits; each introduced page belongs
        # to exactly one staged map. Prefix pages are shared, never copied there.
        assert all(set(node) == {"tokens", "record"} for node, _ in cache._store.staged.values())
        introduced = sum(len(pages) for _, pages in cache._store.staged.values())
        assert introduced == (61 * 48 if branch == 0 else 7 * 48)
        assert cache.lookup(path) is None
        heads.append(cache.commit(tickets, acknowledged=True))
        assert heads[-1].bytes < 2 << 20
        assert cache.storage_metrics["peak_transient_bytes"] <= 24 << 20
    assert cache.storage_metrics["node_count"] == 22
    assert cache.storage_metrics["page_count"] == (54 + 3 * 7) * 48 == 3600
    assert cache.storage_metrics["pending_page_count"] == 0
    # Python metadata estimate is explicit and much larger than serialized index;
    # neither metric pretends that encoded bytes equal a whole-client RSS bound.
    assert cache.storage_metrics["owner_metadata_bytes"] > heads[-1].bytes
    assert cache.storage_metrics["binding_owner_bytes"] == 64 << 10
    before = cache.storage_metrics.copy()
    for ids in [root, *branches]:
        same(cache.lookup(ids), qwen_geometry_state(ids))
        assert cache.stage(ids, *qwen_geometry_state(ids)) is None  # Exact replay: no publication ticket.
    assert cache.storage_metrics["written_bytes"] == before["written_bytes"]
    assert cache.storage_metrics["disk_files"] == before["disk_files"]
    assert cache.storage_metrics["publications"] == 3
    with pytest.raises(ValueError, match="node count"):
        # The 25th unique node rejects before mutation, even when staging room
        # remains. This attempt leaves only the earlier bounded orphan writes.
        path = root.copy()
        for step in range(9):
            path.append(500 + step)
            cache.stage(path, *qwen_geometry_state(path))
    cache.discard()
    before = cache.storage_metrics.copy()
    cache.close()
    reopened = qwen_geometry_owner(tmp_path / "decode", head=heads[-1])
    reopened.bind(compiled, runtime)
    assert reopened.storage_metrics["disk_bytes"] == before["disk_bytes"]  # Orphans remain charged.
    for ids in branches:
        same(reopened.lookup(ids), qwen_geometry_state(ids))
    reopened._restore_head(heads[0])
    same(reopened.lookup(branches[0]), qwen_geometry_state(branches[0]))
    assert reopened.lookup(branches[1]) is None
    assert reopened.storage_metrics["disk_bytes"] == before["disk_bytes"]
    reopened.close()


@pytest.mark.parametrize("damage", ["truncate", "replay", "order", "extent", "path"])
def test_qwen_geometry_manifest_failure_precedes_state_array_allocation(tmp_path, qwen_geometry_binding, monkeypatch, damage):
    compiled, runtime, _ = qwen_geometry_binding
    cache = qwen_geometry_owner(tmp_path / "decode")
    cache.bind(compiled, runtime)
    first, second = list(range(3, 57)) + [100], list(range(3, 57)) + [200]
    tickets = [cache.stage(ids, *qwen_geometry_state(ids)) for ids in (first, second)]
    head = cache.commit(tickets, acknowledged=True)
    path_key = cache._store._path(first)
    node = cache._store.nodes[path_key]
    if damage == "truncate":
        frame = tmp_path / "decode" / node["record"]["name"]
        frame.write_bytes(frame.read_bytes()[:-1])
    elif damage == "replay":
        node["record"] = cache._store.nodes[cache._store._path(second)]["record"]
    else:
        doc = cache._store._node_doc(node, cache._store.pages)
        if damage == "order":
            doc["pages"][0], doc["pages"][1] = doc["pages"][1], doc["pages"][0]
        elif damage == "extent":
            doc["pages"].pop()
        else:
            doc["path"] = cache._store._path(second)
        node["record"] = cache._store._write(msgpack.packb(doc), {"node": path_key})
    before = cache.storage_metrics["written_bytes"]
    monkeypatch.setattr(np, "empty", lambda *args, **kwargs: pytest.fail("allocated invalid manifest state"))
    monkeypatch.setattr(np, "frombuffer", lambda *args, **kwargs: pytest.fail("materialized invalid manifest logits"))
    with pytest.raises((ValueError, InvalidTag)):
        cache.lookup(first)
    assert cache.head == head and cache.storage_metrics["hits"] == 0
    assert cache.storage_metrics["written_bytes"] == before
    cache.close()


def test_disk_manifest_child_can_commit_after_creator_stage_is_discarded(tmp_path, storage_binding):
    cache = opened(tmp_path, storage_binding)
    root = cache.stage([1, 2], *state([1, 2]))
    child = cache.stage([1, 2, 3], *state([1, 2, 3]))
    assert len(cache._store.staged[child][1]) == 4  # Child-only introduced rows.
    cache.discard([root])
    cache.commit([child], acknowledged=True)
    assert cache.lookup([1, 2]) is None
    same(cache.lookup([1, 2, 3]), state([1, 2, 3]))
    assert cache.storage_metrics["page_count"] == 12
    assert cache.storage_metrics["pending_page_count"] == 0
    cache.close()


def test_binding_borrows_admitted_contract_without_constructing_duplicate_owner(tmp_path, monkeypatch):
    import pllm.configuration
    import pllm.runtime.model_binding
    schedule = {"composition_digest": "a" * 64, "steps": []}
    encoded_schedule = json.dumps(schedule, sort_keys=True, separators=(",", ":")).encode()
    document = {"token_bound": 128, "generated_prefix_canonical": True,
                "source_schedule_digest": hashlib.sha256(b"pllm.decoder_runtime_schedule.v2\0" + encoded_schedule).hexdigest()}

    class Admitted(QwenGeometryContract):
        _source_plan = b"public source plan"
        _composition = b"{}"
        _canonical_bytes = json.dumps(document).encode()

        def to_dict(self):
            return document.copy()

    def no_duplicate(*args):
        pytest.fail("constructed a duplicate native continuation/source owner")

    validated = []
    contract = Admitted()
    plan = SimpleNamespace(canonical_bytes=lambda: contract._source_plan, continuation_schedule=no_duplicate)
    compiled = SimpleNamespace(digest="d" * 64, _plan=plan, _bundle=object(),
        _canonical_composition=contract._composition, validate=lambda: validated.append(True))
    runtime = QwenGeometryRuntime()
    runtime._retained = runtime._mlp_tiling = runtime.nonlinear_evaluator = runtime._tiered = None
    runtime._window_contracts, runtime._hybrid_contracts, runtime.shared_kv = {}, {}, {}
    runtime._verification_bits, runtime._causal_reduction = 0, "prefix_f32"
    runtime._generated_prefix_contract, runtime._source_plan, runtime._schedule = contract, plan, schedule
    monkeypatch.setattr(pllm.configuration.Pipeline, "from_spec", staticmethod(lambda _: SimpleNamespace(digest=lambda: "a" * 64)))
    monkeypatch.setattr(pllm.runtime.model_binding, "_state_binding_digest", lambda *args: "b" * 64)
    cache = qwen_geometry_owner(tmp_path / "decode")
    cache.bind(compiled, runtime)
    cache.bind(compiled, runtime)
    assert cache._store.contract is runtime._generated_prefix_contract is contract
    assert len(validated) == 2
    expected = sum(len(v) for v in (contract._source_plan, contract._composition, contract._canonical_bytes))
    assert cache.storage_metrics["binding_owner_bytes"] == expected + 4 * len(encoded_schedule) + 65536
    cache.close()


def test_metadata_decoder_preflights_object_fanout_depth_and_ascii_before_unpack(tmp_path, storage_binding, monkeypatch):
    cache = opened(tmp_path, storage_binding, max_snapshot_bytes=1 << 20, max_transient_bytes=16 << 20)
    monkeypatch.setattr(msgpack, "unpackb", lambda *args, **kwargs: pytest.fail("decoded overbound metadata"))
    # Small ciphertext can still describe many empty Python containers. Four
    # nested batches avoid the per-container 32768 cap but not total object cost.
    wide = b"\x94" + (b"\xdc\x80\x00" + b"\x90" * 32768) * 4
    for raw, error in ((wide, "parsed metadata"), (b"\x91" * 33 + b"\x00", "nesting"),
                       (msgpack.packb({"bad": "\u2603"}), "ASCII")):
        with pytest.raises(ValueError, match=error):
            cache._store._decode(raw)
    assert cache.storage_metrics["written_bytes"] == 0
    assert _preflight_metadata(msgpack.packb({"x": [1, 2], "bytes": b"data"}), 1024, 4096) < 4096
    cache.close()


def native_fixture(tmp_path, family):
    from test_state_reuse_probes import fixture
    from pllm.runtime.model_binding import compile_runtime_model
    from pllm.state import ClientPrefixReuse
    _, remote, plan, bundle, pipeline = fixture(tmp_path / "model", family)
    pipeline = pipeline.with_params(cache=ClientPrefixReuse(
        max_bytes=1 << 20, fixed_input_tokens=32, generated_prefixes=True))
    return compile_runtime_model(plan, bundle, composition=pipeline), remote


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
def test_native_bit_exact_divergence_restart_fresh_continuation_and_retirement(tmp_path, family):
    compiled, remote = native_fixture(tmp_path, family)
    runtime = compiled.runtime(remote)
    cache = DecodeCache(tmp_path / "decode", os.urandom(32))
    key = cache._key  # Caller custody for reopening; never obtained from disk.
    cache.bind(compiled, runtime)
    ids = [3, 8, 7]
    root_logits = runtime.prepare_ids(ids)[1]
    root = runtime.snapshot()
    root_ticket = cache.stage(ids, root, root_logits)
    assert cache.lookup(ids) is None
    a_logits = runtime.decode_step(9, runtime.caches)[0]
    a = runtime.canonical_generated_snapshot()
    a_ticket = cache.stage([*ids, 9], a, a_logits)
    runtime.restore(root)
    b_logits = runtime.decode_step(10, runtime.caches)[0]
    b = runtime.canonical_generated_snapshot()
    b_ticket = cache.stage([*ids, 10], b, b_logits)
    assert cache.lookup_child(ids, 9) is None
    head = cache.commit([root_ticket, a_ticket, b_ticket], acknowledged=True)
    same(cache.lookup_child(ids, 9), (a, a_logits))
    same(cache.lookup_child(ids, 10), (b, b_logits))
    assert cache.storage_metrics["page_count"] == (len(ids) + 2) * runtime.layers * 2
    original_words = a.caches[0].key.copy()
    assert cache.lookup([*ids, 9, int(np.argmax(a_logits))]) is None  # Pending selection excluded.
    cache.close()
    assert np.array_equal(a.caches[0].key, original_words)
    assert not cache._store.staged and cache._store.aead is None
    # Distinct runtime; authenticates every row and remints execution provenance.
    fresh = compiled.runtime(remote)
    reopened = DecodeCache(tmp_path / "decode", key, head=head)
    reopened.bind(compiled, fresh)
    hit = reopened.lookup_child(ids, 9)
    same(hit, (a, a_logits))
    assert hit[0].state_basis.seal != a.state_basis.seal and hit[0].state_basis.valid()
    fresh.restore(hit[0])
    actual_logits = fresh.continue_ids([11])[-1]
    oracle = compiled.runtime(remote)
    oracle_logits = oracle.prepare_ids([*ids, 9, 11])[1]
    same((fresh.snapshot(), actual_logits), (oracle.snapshot(), oracle_logits))
    owned = a.state_basis._with_owner("response")
    before = reopened.storage_metrics["written_bytes"]
    with pytest.raises(ValueError, match="owner-free"):
        reopened.stage([*ids, 9], replace(a, state_basis=owned), a_logits)
    assert reopened.storage_metrics["written_bytes"] == before
    reopened.close()
    # A fresh interpreter reconstructs only the public compiled bundle. No body
    # weights, local-checkpoint executor, provider or old process seal is needed.
    public_bundle = {field.name: getattr(compiled._bundle, field.name)
                     for field in fields(compiled._bundle) if field.init}
    payload = {"plan": compiled._plan.canonical_bytes(), "composition": compiled._canonical_composition,
               "bundle": public_bundle, "root": str(tmp_path / "decode"), "key": key,
               "head": head.to_spec(), "ids": [*ids, 9], "old_seal": a.state_basis.seal}
    code = """
import hashlib, json, pickle, sys
import numpy as np
from pllm.configuration import Pipeline
from pllm.modeling import ModelPlan
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.transformer_client import ClientBundle
from pllm.state.decode import DecodeCache, DecodeCacheHead
p = pickle.loads(sys.stdin.buffer.read())  # trusted synthetic test fixture only
compiled = compile_runtime_model(ModelPlan(p['plan']), ClientBundle(**p['bundle']),
    composition=Pipeline.from_spec(json.loads(p['composition'])))
def unavailable(*args):
    raise AssertionError('decode storage tried to execute a provider')
runtime = compiled.runtime(unavailable)
with DecodeCache(p['root'], p['key'], head=DecodeCacheHead.from_spec(p['head'])) as cache:
    cache.bind(compiled, runtime)
    snapshot, logits = cache.lookup(p['ids'])
    arrays = [logits] + [v for row in snapshot.caches for v in (row.key, row.value)]
    print(json.dumps({'bits': hashlib.sha256(b''.join(v.view(np.uint32).tobytes() for v in arrays)).hexdigest(),
        'valid': snapshot.state_basis.valid(), 'reminted': snapshot.state_basis.seal != p['old_seal']}))
"""
    child = subprocess.run([sys.executable, "-c", code], input=pickle.dumps(payload),
        capture_output=True, check=True, timeout=90,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.resolve()), "OMP_NUM_THREADS": "1",
             "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"})
    arrays = [a_logits] + [v for row in a.caches for v in (row.key, row.value)]
    digest = hashlib.sha256(b"".join(v.view(np.uint32).tobytes() for v in arrays)).hexdigest()
    assert json.loads(child.stdout) == {"bits": digest, "valid": True, "reminted": True}
