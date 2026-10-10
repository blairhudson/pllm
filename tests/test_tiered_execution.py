import copy

import numpy as np
import pytest

from pllm.runtime.tiered_execution import tiered_runtime
from pllm.runtime.transformer_client import TransformerClientError
from test_batched_greedy_verification import fixture
from test_decoder_continuation import runtime_for


def test_exact_layer_paging_append_only_restart_and_residency(fixture, tmp_path):
    engine, compiled, _ = fixture
    serial, _ = runtime_for(compiled, engine)
    _, remote = runtime_for(compiled, engine)
    root, key = tmp_path / "state", b"k" * 32
    candidate = tiered_runtime(compiled, remote, root, key, page_rows=2)
    prompt = [1, 4, 6, 9, 2]
    _, expected, _ = serial.prepare_ids(prompt)
    _, actual, _ = candidate.prepare_ids(prompt)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    for index in range(3):
        old_pages = copy.deepcopy(candidate._tiered.records)
        token = int(np.argmax(expected))
        expected = serial.forward_ids([token])[-1]
        actual = candidate.forward_ids([token])[-1]
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        assert all(cache.key is None and cache.value is None for cache in candidate.caches)
        for old, new in zip(old_pages, candidate._tiered.records, strict=True):
            assert new[:len(old)] == old and len(new) == len(old) + 1
        head = copy.deepcopy(candidate._tiered.head)
        position = candidate.position
        candidate._tiered.close(candidate)
        _, remote = runtime_for(compiled, engine)
        candidate = tiered_runtime(compiled, remote, root, key, head=head, page_rows=2)
        assert candidate.position == position
        np.testing.assert_array_equal(candidate._tiered.logits.view(np.uint32), expected.view(np.uint32))
    candidate._tiered.begin(candidate, np.array([0], np.int64), False)
    for layer in range(candidate.layers):
        candidate._tiered.load(candidate, layer)
        for kind in ("key", "value"):
            np.testing.assert_array_equal(getattr(candidate.caches[layer], kind)[:candidate.position].view(np.uint32),
                getattr(serial.caches[layer], kind)[:serial.position].view(np.uint32))
        candidate._tiered._retire(candidate)
    assert candidate._tiered.peak_live_kv_bytes <= (candidate.position + 1) * np.prod(candidate._tiered.shape) * 8
    candidate._tiered.close(candidate)


@pytest.mark.parametrize("failure", ["remote", "publication", "page"])
def test_failure_preserves_prior_trusted_generation_and_burns_attempt(fixture, tmp_path, monkeypatch, failure):
    engine, compiled, _ = fixture
    _, remote = runtime_for(compiled, engine)
    runtime = tiered_runtime(compiled, remote, tmp_path / "state", b"k" * 32, page_rows=2)
    _, logits, _ = runtime.prepare_ids([1, 4, 7])
    head = copy.deepcopy(runtime._tiered.head)
    files = {p.name for p in runtime._tiered.root.iterdir()}
    write = runtime._tiered._write
    if failure == "remote":
        remote.inject = "nonfinite"
    elif failure == "publication":
        def fail(raw, metadata):
            if "manifest" in metadata:
                raise OSError("injected publication failure")
            return write(raw, metadata)
        monkeypatch.setattr(runtime._tiered, "_write", fail)
    else:
        record = runtime._tiered.records[-1][0]["key"]
        path = runtime._tiered.root / record["name"]
        original = path.read_bytes()
        path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises((ValueError, OSError, TransformerClientError)):
        runtime.forward_ids([int(np.argmax(logits))])
    assert runtime._tiered.failed and runtime._tiered.head == head and runtime.position == 3
    assert {p.name for p in runtime._tiered.root.iterdir()} == files
    assert all(cache.key is None for cache in runtime.caches)
    with pytest.raises(ValueError, match="unavailable"):
        runtime.forward_ids([1])
    if failure == "page":
        path.write_bytes(original)
    runtime._tiered.close(runtime)
    _, remote = runtime_for(compiled, engine)
    restored = tiered_runtime(compiled, remote, tmp_path / "state", b"k" * 32, head=head, page_rows=2)
    restored.forward_ids([int(np.argmax(logits))])
    restored._tiered.close(restored)


def test_head_substitution_wrong_key_and_concurrent_writer_reject(fixture, tmp_path):
    from filelock import Timeout
    engine, compiled, _ = fixture
    _, remote = runtime_for(compiled, engine)
    runtime = tiered_runtime(compiled, remote, tmp_path / "state", b"k" * 32)
    runtime.prepare_ids([1, 4])
    head = copy.deepcopy(runtime._tiered.head)
    with pytest.raises(Timeout):
        tiered_runtime(compiled, remote, tmp_path / "state", b"k" * 32, head=head)
    with pytest.raises(TransformerClientError, match="checkpoint owner"):
        runtime.snapshot()
    runtime._tiered.close(runtime)
    for key, saved in ((b"z" * 32, head), (b"k" * 32, {**head, "revision": 3})):
        with pytest.raises(Exception):
            tiered_runtime(compiled, remote, tmp_path / "state", key, head=saved)


def test_full_state_import_and_reset_cannot_bypass_layer_ownership(fixture, tmp_path):
    engine, compiled, contract = fixture
    original, _ = runtime_for(compiled, engine)
    original.prepare_ids([1, 2])
    snapshot = original.snapshot()
    _, remote = runtime_for(compiled, engine)
    candidate = tiered_runtime(compiled, remote, tmp_path / "state", b"k" * 32)
    with pytest.raises(TransformerClientError, match="checkpoint owner"):
        candidate.restore(snapshot)
    with pytest.raises(TransformerClientError, match="checkpoint owner"):
        candidate.install_continuation(contract)
    candidate.prepare_ids([1, 2])
    head = copy.deepcopy(candidate._tiered.head)
    candidate.reset()
    assert candidate._tiered.closed and all(c.key is None for c in candidate.caches)
    restored = tiered_runtime(compiled, remote, tmp_path / "state", b"k" * 32, head=head)
    restored.forward_ids([3])
    restored.reset()


def test_unaccepted_commit_keeps_previous_head_and_disk_limit_precedes_execution(fixture, tmp_path, monkeypatch):
    import pllm.runtime.tiered_execution as storage
    engine, compiled, _ = fixture
    _, remote = runtime_for(compiled, engine)
    root, key = tmp_path / "state", b"k" * 32
    candidate = tiered_runtime(compiled, remote, root, key)
    _, logits, _ = candidate.prepare_ids([1, 2])
    head = copy.deepcopy(candidate._tiered.head)
    candidate.forward_ids([int(np.argmax(logits))])
    candidate.reset()
    restored = tiered_runtime(compiled, remote, root, key, head=head)
    np.testing.assert_array_equal(restored._tiered.logits, logits)
    monkeypatch.setattr(storage, "_MAX_DISK", restored._tiered.disk_bytes)
    with pytest.raises(ValueError, match="disk"):
        restored.forward_ids([1])
    assert restored._tiered.failed and restored._tiered.head == head
    restored.reset()


def test_fresh_reset_retires_writer_and_key(fixture, tmp_path):
    engine, compiled, _ = fixture
    _, remote = runtime_for(compiled, engine)
    runtime = tiered_runtime(compiled, remote, tmp_path / "state", b"k" * 32)
    runtime.reset()
    assert runtime._tiered.closed and runtime._tiered.aead is None
    assert not runtime._tiered.lock.is_locked


def test_interrupt_at_head_publication_preserves_durable_prior_generation(fixture, tmp_path, monkeypatch):
    from pllm.runtime.tiered_execution import TieredExecution
    engine, compiled, _ = fixture
    reference, _ = runtime_for(compiled, engine)
    _, remote = runtime_for(compiled, engine)
    root, key = tmp_path / "state", b"k" * 32
    candidate = tiered_runtime(compiled, remote, root, key)
    _, logits, _ = candidate.prepare_ids([1, 2])
    reference.prepare_ids([1, 2])
    head = copy.deepcopy(candidate._tiered.head)
    token = int(np.argmax(logits))

    def interrupted(owner, name, value):
        if name == "_committed":
            raise KeyboardInterrupt
        object.__setattr__(owner, name, value)

    with monkeypatch.context() as patch:
        patch.setattr(TieredExecution, "__setattr__", interrupted)
        with pytest.raises(KeyboardInterrupt):
            candidate.forward_ids([token])
    assert candidate._tiered.head == head and candidate.position == 2
    assert all(cache.key is None for cache in candidate.caches)
    candidate.reset()
    restored = tiered_runtime(compiled, remote, root, key, head=head)
    np.testing.assert_array_equal(restored.forward_ids([token]).view(np.uint32),
                                  reference.forward_ids([token]).view(np.uint32))
    restored.reset()
    reference.reset()
