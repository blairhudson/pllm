"""Small storage/numeric oracle: no checkpoint values, Torch or native builds.

Fake compiler admission is explicit; real decoder parity is a separate serial gate.
"""
import copy
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest

from pllm.runtime.context_graph import ContextGraphExecution
from pllm.runtime.tiered_execution import TieredExecution, _MAX_MANIFEST
from pllm.runtime.transformer_client import LayerCache, RuntimeSnapshot, _snapshot_state_basis


@pytest.fixture
def owner_factory(monkeypatch):
    from pllm.configuration import Pipeline
    monkeypatch.setattr(Pipeline, "from_spec", lambda spec: None)

    def make(root, head=None, cls=ContextGraphExecution, key=b"k" * 32):
        monkeypatch.setattr(Pipeline, "from_spec", lambda spec: None)
        graph = {"maximum_key_sequence": 80,
            "operations": [{"id": str(i), "layer": i} for i in range(2)],
            "state_outputs": [{"kind": k} for _ in range(2) for k in ("key", "value")]}
        runtime = SimpleNamespace(position=0, _retained=None, _window_contracts={},
            _hybrid_contracts={}, _verification_bits=0, _mlp_tiling=None,
            nonlinear_evaluator=None, _causal_reduction="prefix_f32", default_kv_heads=1,
            default_head_dim=4, layers=2, cfg={"vocab_size": 8},
            _graphs={p: graph for p in ("prefill", "decode")},
            _schedule={p: {"steps": [{"operation_ids": [str(i)]} for i in range(2)]}
                       for p in ("prefill", "decode")},
            caches=[LayerCache(), LayerCache()], _state_binding_digest="b" * 64,
            _snapshot_basis=None)
        contract = SimpleNamespace(digest="c" * 64, admit=lambda *a: None)
        compiled = SimpleNamespace(validate=lambda: None, digest="d" * 64,
            _canonical_composition="{}", _plan=SimpleNamespace(continuation_schedule=lambda p: contract))
        return runtime, cls(compiled, runtime, root, key, head, page_rows=2)
    return make


def advance(runtime, owner, tokens):
    ids = np.array(tokens, np.int64)
    owner.begin(runtime, ids, False)
    old = runtime.position
    scores = np.zeros(8, np.float32)
    for layer in range(2):
        owner.load(runtime, layer)
        cache = runtime.caches[layer]
        for offset, token in enumerate(tokens):
            pos = old + offset
            cache.key[pos] = (np.arange(4, dtype=np.float32) + token + layer) / 16
            cache.value[pos] = (np.arange(4, dtype=np.float32) + pos + layer) / 32
            # Full-prefix attention; independent oracle below uses same explicit
            # public arithmetic but constructs whole KV without persisted pages.
            q = np.full(4, (token + 1) / 8, np.float32)
            weights = np.exp(cache.key[:pos + 1, 0] @ q)
            output = (weights / weights.sum()) @ cache.value[:pos + 1, 0]
            scores[layer * 4:(layer + 1) * 4] = output
        cache.length = old + len(tokens)
        owner.retire_step(runtime, layer, {})
    runtime.position += len(tokens)
    runtime._snapshot_basis = _snapshot_state_basis(
        "completed_prefill" if old == 0 else "incremental", runtime.position,
        "b" * 64, "e" * 64, "f" * 64, "a" * 32, len(tokens) if old == 0 else 3)
    owner.finish(runtime, ids, scores)
    return scores


def oracle(history):
    out = np.zeros(8, np.float32)
    for layer in range(2):
        keys = np.array([(np.arange(4, dtype=np.float32) + t + layer) / 16 for t in history])
        values = np.array([(np.arange(4, dtype=np.float32) + p + layer) / 32
                           for p in range(len(history))])
        weights = np.exp(keys @ np.full(4, (history[-1] + 1) / 8, np.float32))
        out[layer * 4:(layer + 1) * 4] = (weights / weights.sum()) @ values
    return out


def test_shared_forks_restart_anchors_full_attention_and_pending(owner_factory, tmp_path):
    root = tmp_path / "graph"
    runtime, owner = owner_factory(root)
    advance(runtime, owner, [1, 4, 6])
    prefix = copy.deepcopy(owner.records)
    fork = copy.deepcopy(owner.head)
    owner.close(runtime)
    for branch in ([2] * 18, [3] * 18):
        head, history = fork, [1, 4, 6]
        for token in branch:
            runtime, owner = owner_factory(root, head)
            assert owner.pending_token == int(np.argmax(oracle(history)))
            actual = advance(runtime, owner, [token])
            history.append(token)
            np.testing.assert_array_equal(actual.view(np.uint32), oracle(history).view(np.uint32))
            assert all(rows[:len(old)] == old for rows, old in zip(owner.records, prefix))
            assert owner.history == history and all(c.key is None for c in runtime.caches)
            head = copy.deepcopy(owner.head)
            owner.begin(runtime, np.array([0], np.int64), False)
            for layer in range(2):
                owner.load(runtime, layer)
                keys = np.array([(np.arange(4, dtype=np.float32) + t + layer) / 16 for t in history])
                values = np.array([(np.arange(4, dtype=np.float32) + p + layer) / 32
                                   for p in range(len(history))])
                np.testing.assert_array_equal(runtime.caches[layer].key[:len(history), 0].view(np.uint32),
                                              keys.view(np.uint32))
                np.testing.assert_array_equal(runtime.caches[layer].value[:len(history), 0].view(np.uint32),
                                              values.view(np.uint32))
                owner._retire(runtime)
            owner.close(runtime)
    runtime, owner = owner_factory(root, fork)
    assert owner.history == [1, 4, 6]
    owner.close(runtime)


def test_delta_publication_reduces_bytes_against_shared_store_full_manifests(owner_factory, tmp_path):
    totals = {}
    for cls in (TieredExecution, ContextGraphExecution):
        runtime, owner = owner_factory(tmp_path / cls.__name__, cls=cls)
        advance(runtime, owner, [1, 4, 6])
        prefix = copy.deepcopy(owner.records)
        for _ in range(20):
            advance(runtime, owner, [2])
        assert all(rows[:len(old)] == old for rows, old in zip(owner.records, prefix))
        totals[cls] = owner.written_bytes
        owner.close(runtime)
    assert totals[ContextGraphExecution] < totals[TieredExecution]


def test_generated_state_exceeding_resident_cache_budget_restores(owner_factory, tmp_path):
    from pllm.runtime.prefill_cache import ExactPrefillCache
    root = tmp_path / "graph"
    runtime, owner = owner_factory(root)
    advance(runtime, owner, [2] * 20)
    head = copy.deepcopy(owner.head)
    scores = owner.logits.copy()
    owner.begin(runtime, np.array([0], np.int64), False)
    caches = []
    for layer in range(2):
        owner.load(runtime, layer)
        caches.append(runtime.caches[layer].copy_active())
        owner._retire(runtime)
    basis = _snapshot_state_basis("completed_prefill", 20, "b" * 64, "e" * 64, "f" * 64,
                                  prefill_extent=20)
    snapshot = RuntimeSnapshot(20, caches, {}, "b" * 64, basis)
    cache = ExactPrefillCache(256, causal_reduction="prefix_f32")
    assert basis.cache_eligible() and sum(c.key.nbytes + c.value.nbytes for c in caches) > cache.max_bytes
    assert not cache.put("prefix", snapshot, scores)
    owner.close(runtime)
    runtime, owner = owner_factory(root, head)
    np.testing.assert_array_equal(owner.logits.view(np.uint32), scores.view(np.uint32))
    owner.begin(runtime, np.array([0], np.int64), False)
    for layer in range(2):
        owner.load(runtime, layer)
        np.testing.assert_array_equal(runtime.caches[layer].key[:20].view(np.uint32), caches[layer].key.view(np.uint32))
        np.testing.assert_array_equal(runtime.caches[layer].value[:20].view(np.uint32), caches[layer].value.view(np.uint32))
        caches[layer].key.fill(0)
        caches[layer].value.fill(0)
        owner._retire(runtime)
    cache.clear()
    owner.close(runtime)


def test_writer_arbitration_and_corrupted_shared_page_burn_attempt(owner_factory, tmp_path):
    from filelock import Timeout
    root = tmp_path / "graph"
    runtime, owner = owner_factory(root)
    advance(runtime, owner, [1, 4, 6])
    head = copy.deepcopy(owner.head)
    with pytest.raises(Timeout):
        owner_factory(root, head)
    page = owner.records[0][0]["key"]
    path = root / page["name"]
    frame = path.read_bytes()
    path.write_bytes(frame[:-1] + bytes([frame[-1] ^ 1]))
    with pytest.raises(ValueError, match="trusted commitment"):
        try:
            owner.begin(runtime, np.array([2], np.int64), False)
            owner.load(runtime, 0)
        except BaseException:
            owner.abort(runtime)
            raise
    assert owner.failed and owner.head == head and runtime.position == 3
    assert all(c.key is None and c.value is None for c in runtime.caches)
    owner.close(runtime)
    path.write_bytes(frame)
    runtime, restored = owner_factory(root, head)
    np.testing.assert_array_equal(advance(runtime, restored, [2]), oracle([1, 4, 6, 2]))
    restored.close(runtime)


@pytest.mark.parametrize("failure", ["corrupt_parent", "wrong_key", "wrong_context", "wrong_head",
                                     "pending", "parent_revision", "publication", "interrupt"])
def test_failure_rollback_and_cleanup(owner_factory, tmp_path, monkeypatch, failure):
    runtime, owner = owner_factory(tmp_path / "graph")
    advance(runtime, owner, [1, 4, 6])
    prior = copy.deepcopy(owner.head)
    advance(runtime, owner, [2])
    newest = copy.deepcopy(owner.head)
    if failure in {"publication", "interrupt"}:
        before = {p.name for p in owner.root.iterdir()}
        if failure == "publication":
            write = owner._write
            def fail(raw, meta):
                if "manifest" in meta:
                    raise OSError("injected publication failure")
                return write(raw, meta)
            monkeypatch.setattr(owner, "_write", fail)
        else:
            def interrupted(obj, name, value):
                if name == "_committed":
                    raise KeyboardInterrupt
                object.__setattr__(obj, name, value)
            monkeypatch.setattr(ContextGraphExecution, "__setattr__", interrupted)
        with pytest.raises((OSError, KeyboardInterrupt)):
            try:
                advance(runtime, owner, [3])
            except BaseException:
                owner.abort(runtime)
                raise
        assert owner.head == newest and runtime.position == 4
        assert all(c.key is None for c in runtime.caches)
        if failure == "publication":
            assert before == {p.name for p in owner.root.iterdir()}
        monkeypatch.undo()
        owner.close(runtime)
        runtime, restored = owner_factory(tmp_path / "graph", newest)
        np.testing.assert_array_equal(restored.logits, oracle([1, 4, 6, 2]))
        restored.close(runtime)
        return
    if failure in {"pending", "parent_revision"}:
        raw = owner._read_frame(newest["record"], {"manifest": 1}, _MAX_MANIFEST)
        env = msgpack.unpackb(raw, raw=False)
        if failure == "pending":
            env["pending"] = (env["pending"] + 1) % 8
        else:
            env["parent"]["revision"] = 1
        # Authenticated but semantically invalid issuer frame.
        record = TieredExecution._write(owner, msgpack.packb(env, use_bin_type=True), {"manifest": 1})
        newest["record"] = record
        owner.new_files.clear()
    owner.close(runtime)
    key = b"k" * 32
    if failure == "corrupt_parent":
        path = owner.root / prior["record"]["name"]
        frame = path.read_bytes()
        path.write_bytes(frame[:-1] + bytes([frame[-1] ^ 1]))
    elif failure == "wrong_key":
        key = b"z" * 32
    elif failure == "wrong_context":
        monkeypatch.setattr(ContextGraphExecution, "_aad", lambda self, meta: b"wrong")
    elif failure == "wrong_head":
        newest["revision"] = 2
    with pytest.raises(Exception):
        owner_factory(tmp_path / "graph", newest, key=key)
    monkeypatch.undo()
    if failure == "corrupt_parent":
        path.write_bytes(frame)
    runtime, restored = owner_factory(tmp_path / "graph", prior)
    np.testing.assert_array_equal(restored.logits, oracle([1, 4, 6]))
    restored.close(runtime)
    assert restored.aead is None and not restored.lock.is_locked
