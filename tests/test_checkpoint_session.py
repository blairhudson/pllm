import copy
import pickle

import numpy as np
import pytest

from pllm.state import CheckpointHead, CheckpointSession
from test_batched_greedy_verification import fixture
from test_decoder_continuation import runtime_for


@pytest.mark.parametrize("storage", ["layer_pages", "context_graph"])
def test_checkpoint_public_exact_execution_restart_and_head_ownership(fixture, tmp_path, storage):
    engine, compiled, _ = fixture
    reference, _ = runtime_for(compiled, engine)
    _, remote = runtime_for(compiled, engine)
    root, key = tmp_path / "state", b"k" * 32
    with CheckpointSession(compiled, remote, root, key, storage=storage, page_rows=2) as session:
        prompt = [1, 4, 6]
        expected = reference.prepare_ids(prompt)[1]
        np.testing.assert_array_equal(session.append(prompt).view(np.uint32), expected.view(np.uint32))
        head = session.head
        assert CheckpointHead.from_spec(head.to_spec()) == head
        assert session.position == 3 and session.pending_token == int(np.argmax(expected))
        session.logits.fill(0)
        np.testing.assert_array_equal(session.logits, expected)
        for operation in (copy.copy, copy.deepcopy, pickle.dumps):
            with pytest.raises(TypeError):
                operation(session)
    assert session.head == head
    with pytest.raises(ValueError, match="unavailable"):
        session.advance()
    _, remote = runtime_for(compiled, engine)
    with CheckpointSession(compiled, remote, root, key, storage=storage, head=head, page_rows=2) as restored:
        np.testing.assert_array_equal(restored.logits, expected)
        expected = reference.forward_ids([int(np.argmax(expected))])[-1]
        np.testing.assert_array_equal(restored.advance().view(np.uint32), expected.view(np.uint32))
        assert restored.position == 4 and restored.head.revision == head.revision + 1
        assert restored.storage_metrics["peak_live_kv_bytes"] > 0
    reference.reset()


def test_checkpoint_failed_append_retires_owner_preserves_trusted_head(fixture, tmp_path):
    engine, compiled, _ = fixture
    _, remote = runtime_for(compiled, engine)
    root, key = tmp_path / "state", b"k" * 32
    session = CheckpointSession(compiled, remote, root, key)
    expected = session.append([1, 4, 6])
    head = session.head
    with pytest.raises(ValueError, match="tokens"):
        session.append([True])
    assert session.head == head
    assert session._owner.aead is None and not session._owner.lock.is_locked
    _, remote = runtime_for(compiled, engine)
    with CheckpointSession(compiled, remote, root, key, head=head) as restored:
        np.testing.assert_array_equal(restored.logits, expected)
    with pytest.raises(ValueError, match="geometry"):
        CheckpointSession(compiled, remote, root, key, head=head, page_rows=2)
