"""Synthetic only: small NumPy arrays, no network, checkpoints or timing."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("cold_shared_core", Path(__file__).resolve().parents[1]/"scripts/probe_cold_shared_core.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.mark.parametrize("tied", [True, False])
@pytest.mark.parametrize("depth", [1, 4, 7])
def test_incremental_independent_full_sequence_oracle(tied, depth):
    m = probe.model(tied, depth=depth)
    tokens = [1, 3, 7, 2, 8]
    owner = probe.Decoder(m, len(tokens))
    actual = np.array([owner.step(t) for t in tokens])
    np.testing.assert_allclose(actual, probe.oracle(m, tokens), rtol=1e-12, atol=1e-12)
    assert owner.calls == depth*len(tokens)


def test_defined_tying_matches_materialized_unrolling():
    m = probe.model()
    unrolled = probe.expanded(m)
    np.testing.assert_array_equal(probe.oracle(m, [1, 2, 3]), probe.oracle(unrolled, [1, 2, 3]))
    assert m["blocks"][0] is m["blocks"][-1]
    assert unrolled["blocks"][0] is not unrolled["blocks"][-1]
    assert probe.artifact(m)["complete_download_bytes"] < probe.artifact(unrolled)["complete_download_bytes"]


def test_autoregressive_greedy_continuation_matches_oracle():
    m = probe.model()
    owner = probe.Decoder(m, 12)
    tokens = [1, 4, 2]
    for token in tokens[:-1]:
        owner.step(token)
    for _ in range(8):
        logits = owner.step(tokens[-1])
        expected = probe.oracle(m, tokens)[-1]
        np.testing.assert_allclose(logits, expected, rtol=1e-12, atol=1e-12)
        assert np.argmax(logits) == np.argmax(expected)
        tokens.append(int(np.argmax(logits)))


@pytest.mark.parametrize("average", [True, False])
def test_posthoc_tie_is_counterexample_not_optimization(average):
    m = probe.model(False)
    error = np.max(np.abs(probe.oracle(m, [1, 2, 3])-probe.oracle(probe.posthoc(m, average), [1, 2, 3])))
    assert error > .01


def test_exact_parameter_ownership_and_depth_extras():
    m = probe.model()
    w, d, v = m["width"], m["depth"], m["vocab"]
    assert probe.artifact(m)["distinct_parameters"] == 6*w*w + 3*d*w + 2*w*v
    assert probe.artifact(probe.expanded(m))["distinct_parameters"] == 6*d*w*w + 3*d*w + 2*w*v


def test_kv_is_per_depth_not_per_distinct_block():
    m = probe.model()
    owner = probe.Decoder(m, 8)
    owner.step(1)
    assert owner.keys.nbytes + owner.values.nbytes == 2*4*8*8*8
    assert not np.array_equal(owner.keys[0, 0], owner.keys[1, 0])
    assert probe.resources(m, 8)["dense_matmul_flops"] == probe.resources(probe.expanded(m), 8)["dense_matmul_flops"]


def test_invalid_input_does_not_advance_state():
    owner = probe.Decoder(probe.model(), 1)
    for token in (-1, 19, .5):
        with pytest.raises(ValueError):
            owner.step(token)
        assert owner.position == 0
        assert not owner.keys.any()
    owner.step(1)
    before = owner.keys.copy()
    with pytest.raises(ValueError):
        owner.step(2)
    np.testing.assert_array_equal(owner.keys, before)


def test_bounds():
    with pytest.raises(ValueError):
        probe.model(width=65)
    with pytest.raises(ValueError):
        probe.Decoder(probe.model(), 513)
    for tokens in ([], [1.5], [-1], [19], [1]*513):
        with pytest.raises(ValueError):
            probe.oracle(probe.model(), tokens)


def test_cold_accounting_includes_download_and_compute():
    result = probe.cold_case(10, 10, 100, 2, 10, 200, 5, 1000)
    assert result["local_seconds"] == 20
    assert result["prepared_seconds"] == 5
    assert not result["local_faster"]


def test_deterministic_evidence_and_lossy_boundary():
    first = probe.evidence()
    assert first == probe.evidence()
    assert first["boundary"]["max_abs_reconstruction_error"] > 0
    assert first["tied_artifact"]["complete_download_bytes"] > first["tied_artifact"]["parameter_payload_bytes"]
    assert first["huginn_estimates"]["full_kv_bf16_at_4096_tokens_32_loops_bytes"] > 10_000_000_000
