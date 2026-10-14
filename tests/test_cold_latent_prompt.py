"""Independent counterexamples and arithmetic controls for synthetic hypothesis."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/probe_cold_latent_prompt.py"
SPEC = importlib.util.spec_from_file_location("cold_latent", PATH)
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


def test_disjoint_and_rare():
    train, test, rare = p.samples()
    assert not set(map(tuple, train)) & set(map(tuple, test))
    assert len(set(map(tuple, train))) == 256
    assert len(set(map(tuple, test))) == 128
    assert np.all(train < 8)
    assert np.all((rare >= 8).sum(axis=1) == 1)


def test_order_collision_independent():
    # Counterexample independent of random corpus and learned weights.
    a = np.array([[0, 1, 0, 1, 0, 1, 0, 1]])
    b = 1 - a
    for slots in (1, 2, 4):
        assert np.array_equal(p.pooled(a, slots), p.pooled(b, slots))
        assert a[0, 0] != b[0, 0]
    assert not np.array_equal(p.pooled(a, 8), p.pooled(b, 8))


def test_learned_common_identity_and_unseen_failure():
    train, test, rare = p.samples()
    basis = p.fit(train, 8)
    assert np.allclose(basis.T @ basis, np.eye(8), atol=1e-12)
    assert np.allclose(basis[8:], 0, atol=1e-12)
    prediction, _ = p.reconstruct(test, 8, basis)
    assert np.array_equal(prediction, test)
    prediction, _ = p.reconstruct(rare, 8, basis)
    assert np.count_nonzero((prediction == rare) & (rare >= 8)) == 0


def test_quality_failure_and_determinism():
    q = p.quality()
    assert q == p.quality()
    assert q["added_arrays_conservative_ceiling_bytes"] <= 128 * (1 << 20)
    for r in q["results"]:
        assert r["splits"]["heldout_common"]["majority_correct"] == 128
        assert r["splits"]["heldout_rare"]["rare_token_correct"] == 0
        if r["slots"] < 8:
            assert r["splits"]["heldout_common"]["exact_copy"] < 128
            assert r["collision_different_copy_answer"]


def test_all_links_and_break_even_independent():
    r = p.ledger(150, 32, 8, 16 * p.MIB)
    assert r["baseline_bytes"] == sum(r["baseline_links"].values())
    assert r["latent_bytes"] == sum(r["latent_links"].values())
    # Independently compute four compiled groups per layer; corrections charged.
    row = 24 * ((896 + 896 + 896 + 4864)
                + 2 * (1152 + 896 + 9728 + 896)) * 4
    expected = 118 * row + 118 * 8 - (16 + 12 + 2) * p.MIB
    assert r["net_saved_bytes"] == expected
    limit = r["encoder_bytes_strict_break_even_ceiling"]
    assert p.ledger(150, 32, 8, limit)["net_saved_bytes"] == 1
    assert p.ledger(150, 32, 8, limit + 1)["net_saved_bytes"] == 0
    large = p.ledger(32, 16, 8, 1024 * p.MIB)
    n = large["strict_positive_saving_requests"]
    assert p.ledger(32, 16, 8, 1024 * p.MIB, n)["net_saved_bytes"] > 0
    assert p.ledger(32, 16, 8, 1024 * p.MIB, n - 1)["net_saved_bytes"] <= 0


def test_zero_saving_equal_rows_and_memory():
    assert p.ledger(32, 32, 8, 0)["net_saved_bytes"] == -14 * p.MIB
    assert p.ledger(32, 32, 8, 0)["strict_positive_saving_requests"] is None
    low = p.owners(16 * p.MIB)
    assert low["logical_total_bytes"] == sum(low["logical_owners"].values())
    assert low["logical_owners"]["client_active_KV_float32"] == 2 * 24 * 2 * 64 * 39 * 4
    assert low["tiers"]["Low256MiB"]["admitted_logical"]
    assert not p.owners(16 * p.MIB, boundary_word_bytes=2)["tiers"]["Low256MiB"]["admitted_logical"]
    assert p.owners(1024 * p.MIB)["tiers"]["Medium2GiB"]["admitted_logical"]
    assert low["not_RSS"]


@pytest.mark.parametrize("args", [(0, 1, 1, 0), (1, 2, 1, 0), (1, 1, 1, -1)])
def test_invalid_ledger(args):
    with pytest.raises(ValueError):
        p.ledger(*args)
