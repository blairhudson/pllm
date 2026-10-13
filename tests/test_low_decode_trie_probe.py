"""Independent whole-prefix attention and sampler controls, not native parity."""
import random

import pytest
from cryptography.exceptions import InvalidTag

from scripts.probe_low_decode_trie import (
    DecodeTrie, MAX_NODES, State, break_even, execute, generate,
)


def oracle(history):
    # Recompute entire causal prefix, with no incremental execution or cache calls.
    keys = [token + index + 1 for index, token in enumerate(history)]
    values = [(token + 1) * (token + 1) for token in history]
    q = history[-1] + 2
    weighted = q * sum(key * value for key, value in zip(keys, values, strict=True)) + sum(values)
    logits = tuple((weighted + q * column + len(history) - 1) % 7 + 1
                   for column in (1, 2, 3))
    return State(tuple(history), tuple(zip(keys, values, strict=True)), logits)


def oracle_generation(history, seed, outputs):
    rng = random.Random(seed)
    selected = []
    states = []
    for index in range(outputs):
        state = oracle(history)
        weights = state.logits
        draw = rng.randrange(sum(weights))
        boundaries = (weights[0], weights[0] + weights[1])
        token = 0 if draw < boundaries[0] else 1 if draw < boundaries[1] else 2
        selected.append(token)
        if index < outputs - 1:
            history = (*history, token)
            states.append(oracle(history))
    return selected, states, rng.getstate()


@pytest.mark.parametrize("seed", range(16))
def test_stochastic_exact_branch_oracle(tmp_path, seed):
    root = oracle((0, 1))
    trie = DecodeTrie(tmp_path, b"k" * 32)
    trie.put(root)
    # Seed unrelated branches first; then compare cold/mixed/hot path to independent oracle.
    generate(root, seed=(seed + 5) % 16, trie=trie)
    expected, states, rng = oracle_generation((0, 1), seed, 5)
    for _ in range(2):
        result = generate(root, seed=seed, trie=trie)
        assert list(result["selected"]) == expected
        assert result["states"] == states
        assert result["rng_state"] == rng
        assert result["state"].history == (0, 1, *expected[:-1])
        assert result["pending"] == expected[-1]
    assert result["hits"] == 4
    assert trie.get((0, 1, *expected)) is None  # Pending selection never published.


def test_divergent_forks_share_prefix_pages(tmp_path):
    root = oracle((0, 1))
    trie = DecodeTrie(tmp_path, b"k" * 32)
    trie.put(root)
    for token in (0, 2):
        state = execute(root, token)
        assert state == oracle((0, 1, token))
        trie.put(state)
        assert trie.get(state.history) == oracle(state.history)
    assert trie.ledger()["unique_row_pages"] == 4
    assert trie.ledger()["manifest_writes"] == 3
    assert trie.get(root.history) == root


@pytest.mark.parametrize("fault", ["key", "binding", "ciphertext", "revision"])
def test_authentication_fail_closed(tmp_path, fault):
    root = oracle((0, 1))
    trie = DecodeTrie(tmp_path, b"k" * 32)
    trie.put(root)
    if fault == "key":
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        trie.aead = AESGCM(b"w" * 32)
    elif fault == "binding":
        trie.binding = "changed numeric contract"
    elif fault == "revision":
        rev, record = trie.heads[root.history]
        trie.heads[root.history] = (rev + 1, record)
    else:
        record = trie.pages[(0,)]
        path = tmp_path / record[0]
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        path.write_bytes(raw)
    with pytest.raises((ValueError, InvalidTag)):
        trie.get(root.history)


def test_authenticated_invalid_position_and_pending_rejected(tmp_path):
    trie = DecodeTrie(tmp_path, b"k" * 32)
    root = oracle((0, 1))
    with pytest.raises(ValueError, match="pending"):
        trie.put(State((0, 1, 2), root.rows, root.logits))
    trie.put(root)
    revision, record = trie.heads[root.history]
    doc = trie._read("node", [root.history, revision], record)
    doc["position"] += 1
    bad = trie._write("node", [root.history, revision], doc)
    trie.heads[root.history] = revision, bad
    with pytest.raises(ValueError, match="identity"):
        trie.get(root.history)


def test_failed_publication_retains_prior_head_and_disk_owner(tmp_path):
    root = oracle((0, 1))
    trie = DecodeTrie(tmp_path, b"k" * 32)
    trie.put(root)
    accepted = dict(trie.heads)
    with pytest.raises(OSError, match="publication"):
        trie.put(oracle((0, 1, 2)), fail_before_head=True)
    assert trie.heads == accepted
    assert trie.get(root.history) == root
    assert trie.get((0, 1, 2)) is None
    assert trie.ledger()["orphan_publication_bytes"] > 0
    retained = trie.ledger()["retained_file_bytes"]
    trie.close()
    assert not trie.heads and not trie.pages and trie.aead is None
    assert sum(path.stat().st_size for path in tmp_path.iterdir()) == retained
    with pytest.raises(ValueError, match="closed"):
        trie.get(root.history)


def test_bounds_no_automatic_disk_latest(tmp_path):
    trie = DecodeTrie(tmp_path, b"k" * 32)
    trie.put(oracle((0,)))
    assert trie.get((2,)) is None
    with pytest.raises(ValueError, match="fresh"):
        DecodeTrie(tmp_path, b"k" * 32)
    # Entry admission checked before disk publication; tiny allocation only.
    trie.heads = {tuple([index]): (0, ("", 28)) for index in range(MAX_NODES)}
    before = trie.write_bytes
    with pytest.raises(ValueError, match="node bound"):
        trie.put(oracle((0, 1)))
    assert trie.write_bytes == before


def test_break_even_unknown_and_complete_costs():
    assert break_even(hits=92, per_execution_all_link_bytes=None, extra_all_link_bytes=0) is None
    assert break_even(hits=92, per_execution_all_link_bytes=100, extra_all_link_bytes=None) is None
    assert break_even(hits=92, per_execution_all_link_bytes=100, extra_all_link_bytes=9201) == -1
    assert break_even(hits=92, per_execution_all_link_bytes=100, extra_all_link_bytes=9199) == 1


def test_one_output_never_executes_selected_token(tmp_path):
    root = oracle((0, 1))
    trie = DecodeTrie(tmp_path, b"k" * 32)
    trie.put(root)
    result = generate(root, seed=0, outputs=1, trie=trie)
    assert result["state"] == root
    assert result["hits"] == result["misses"] == 0
    assert trie.ledger()["nodes"] == 1


def test_frame_depth_disk_admission_and_byte_ledger(tmp_path):
    from scripts.probe_low_decode_trie import MAX_DEPTH, MAX_DISK, MAX_FRAME

    trie = DecodeTrie(tmp_path, b"k" * 32)
    root = oracle((0, 1))
    trie.put(root)
    ledger = trie.ledger()
    assert ledger["logical_encrypted_writes"] == (
        ledger["row_plaintext_written_bytes"] + ledger["manifest_plaintext_written_bytes"]
        + ledger["nonce_tag_written_bytes"])
    assert ledger["logical_encrypted_writes"] == ledger["retained_file_bytes"]
    assert ledger["allocated_disk_bytes"] >= ledger["retained_file_bytes"]
    with pytest.raises(ValueError, match="frame bound"):
        trie._write("row", (0,), "x" * MAX_FRAME)
    trie.write_bytes = MAX_DISK
    with pytest.raises(ValueError, match="disk bound"):
        trie._write("row", (0,), [1, 1])
    with pytest.raises(ValueError, match="depth"):
        execute(oracle((0,) * MAX_DEPTH), 1)
