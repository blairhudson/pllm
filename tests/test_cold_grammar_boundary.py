"""Independent small oracles; no installed or checkout SDK execution."""
import importlib.util
import itertools
import math
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "cold_grammar", Path(__file__).resolve().parents[1] / "scripts/probe_cold_grammar_boundary.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_integer_head_and_embedding_row_identity():
    rows, scales, grammar = probe.fixture()
    shard = probe.specialize(rows, scales, grammar, probe.identity(rows, scales))
    x = [1, -2, 3, 4]
    for index, token in enumerate(shard["ids"]):
        expected = (rows[token][0] - 2 * rows[token][1] + 3 * rows[token][2]
                    + 4 * rows[token][3]) * scales[token]
        assert probe.logits(shard["rows"], shard["scales"], x)[index] == expected
        assert shard["rows"][index] == rows[token]
        assert probe.recover_embedding(probe.embedding_record(rows[token], scales[token]), 4) == [
            value * scales[token] for value in rows[token]]
    with pytest.raises(ValueError, match="shape"):
        probe.recover_embedding(b"bad", 4)


def test_independent_softmax_nucleus_and_tie():
    # Huge excluded score must have no effect; mask before normalization.
    scores = [2.0, 1.0, 999.0]
    p = probe.distribution(scores, [0, 1, 2], [0, 1], 1, 1)
    assert p[0] == pytest.approx(math.e / (math.e + 1))
    assert p[1] == pytest.approx(1 / (math.e + 1))
    assert probe.distribution(scores, [0, 1, 2], [0, 1], 1, 0.7) == {0: 1, 1: 0}
    assert probe.distribution([0, 0], [9, 7], [9, 7], 0, 1) == {7: 1, 9: 0}
    assert probe.select({7: 0.25, 9: 0.75}, 0.25) == 9
    # Current SDK excludes crossing token; common nucleus would keep token 1.
    assert probe.distribution([math.log(0.6), math.log(0.3), math.log(0.1)],
                              [0, 1, 2], [0, 1, 2], 1, 0.8) == {0: 1, 1: 0, 2: 0}


def test_all_small_hidden_states_masked_equal():
    rows, scales, grammar = probe.fixture()
    shard = probe.specialize(rows, scales, grammar, probe.identity(rows, scales))
    for x in itertools.product((-1, 0, 1), repeat=4):
        for allowed in grammar["states"]:
            for temp, nucleus in ((0, 1), (0.5, 1), (2, 0.75)):
                full = probe.distribution(probe.logits(rows, scales, x), list(range(12)), allowed, temp, nucleus)
                slim = probe.distribution(probe.logits(shard["rows"], shard["scales"], x),
                                          shard["ids"], allowed, temp, nucleus)
                assert full == slim
                for u in (0, 0.1, 0.9, 0.999):
                    assert probe.select(full, u) == probe.select(slim, u)


@pytest.mark.parametrize("failure", ["identity", "shape", "missing", "private", "eos"])
def test_fail_closed(failure):
    rows, scales, grammar = probe.fixture()
    digest = probe.identity(rows, scales)
    if failure == "identity": digest = "wrong"
    if failure == "shape": rows[0].pop()
    if failure == "missing": grammar["states"][0] = [999]
    if failure == "private": grammar["visibility"] = "PRIVATE"
    if failure == "eos": grammar["states"][-1] = [2]
    with pytest.raises(ValueError):
        probe.specialize(rows, scales, grammar, digest)


def test_accounting_independent_and_cold_gate():
    a = probe.accounting()
    assert a["head_payload_bytes"] == 230400
    assert a["full_tied_raw_i8_and_f32_bytes"] == 136742400
    assert 151936 * 896 == 136134656  # user's raw figure excludes scales
    assert a["query_bytes_each_worker_each_slot"] == 388
    assert a["reply_bytes_each_worker_each_slot"] == 965
    assert a["lookup_online_bytes"] == 24 * 2706
    assert a["client_boundary_total_bytes"] == 296464
    assert probe.accounting(worker_cold=True)["net_saved_bytes"] < 0
    assert probe.accounting(union=151936)["net_saved_bytes"] < 0
    gate = a["strict_saving_max_lookup_slots"]
    # Arithmetic gate exceeds native lifetime limit; not admission permission.
    assert gate == (136742400 - 230400 - 1024 - 96 - 1) // 2706
    with pytest.raises(ValueError): probe.accounting(prompt_slots=4097)


def test_eos_and_exact_replay_extensions():
    e = probe.evidence()
    assert all(case["selections"] == [11] * 4 for case in e["numeric_cases"] if case["state"] == 2)
    e["future_extension"] = {"keep": [1, 2, 3]}
    e["warm_workers"]["future_extension"] = "also retain nested metadata"
    assert probe.replay(e) == e
    e["warm_workers"]["head_payload_bytes"] += 1
    with pytest.raises(ValueError, match="warm_workers"):
        probe.replay(e)


def test_clear_toy_generation_arbitrary_prompt_and_cache_eos():
    rows, scales, _ = probe.fixture()
    for u in (0.125, 0.875):
        full = probe.clear_generation([0, 1, 10], False, u)
        assert full == probe.clear_generation([0, 1, 10], True, u)
        assert full["eos_selected"]
        assert len(full["emitted"]) == 2
        assert full["cache_ids"] == [0, 1, 10] + full["emitted"]
        assert 11 not in full["cache_ids"]
        assert [s["position"] for s in full["steps"]] == [3, 4, 5]
        independent_hidden = [sum(rows[t][d] * scales[t] for t in full["cache_ids"])
                              for d in range(4)]
        assert full["hidden"] == independent_hidden


@pytest.mark.parametrize("temp,p", [(-1, 1), (math.nan, 1), (1, 0), (1, 1.1)])
def test_invalid_sampling(temp, p):
    with pytest.raises(ValueError): probe.distribution([1], [0], [0], temp, p)
