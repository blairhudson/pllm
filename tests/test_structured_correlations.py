"""Meaningful algebra, leakage and one-use checks for the research-only screen."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts/probe_structured_correlations.py"
_SPEC = importlib.util.spec_from_file_location("structured_probe", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
probe = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = probe
_SPEC.loader.exec_module(probe)


def pair(p=probe.P):
    maps = probe.projections(3, 4, 2)
    r0, r1 = probe.random_residues((2, 3), p), probe.random_residues((2, 3), p)
    b0, b1 = probe.random_residues((2,), p), probe.random_residues((2,), p)
    full = probe.functions((r0 + r1) % p, (b0 + b1) % p, maps, p)
    c0 = tuple(probe.random_residues(x.shape, p) for x in full)
    c1 = tuple((x - a) % p for x, a in zip(full, c0, strict=True))
    return probe.material_pair(r0, r1, b0, b1, c0, c1, maps, "tests", p)


@pytest.mark.parametrize("p", [5, 257, probe.P, 1 << 24, 1 << 32])
def test_joint_one_opening_consumer(p):
    assert probe.consume(pair(p))["exact_joint_consumer_parity"]


def test_replay_and_cancel_burn():
    a, b = pair()
    x = np.zeros_like(a.r)
    opening = a.begin_source(x)
    peer = b.begin_source(x)
    a.finish_source(peer)
    b.finish_source(opening)
    with pytest.raises(ValueError, match="spent"):
        a.begin_source(x)
    with pytest.raises(ValueError, match="spent"):
        a.finish_source(peer)
    a.cancel()
    with pytest.raises(ValueError, match="unavailable"):
        a.begin_scalar(np.zeros_like(a.b))
    assert not np.any(a.r) and not np.any(a.b)
    assert all(not np.any(x) for x in a.coefficients)


@pytest.mark.parametrize("fault", ["session", "party", "residue", "truncated"])
def test_bad_source_response_burns(fault):
    a, b = pair()
    x = np.zeros_like(a.r)
    a.begin_source(x)
    body = b.begin_source(x)
    if fault == "session":
        body = probe.frame("wrong", a.digest, "source", 1, probe.words(x))
    elif fault == "party":
        body = probe.frame(a.session, a.digest, "source", 0, probe.words(x))
    elif fault == "residue":
        body = probe.frame(a.session, a.digest, "source", 1, probe.words(x + probe.P))
    else:
        body = body[:-1]
    with pytest.raises((ValueError, probe.msgpack.UnpackException)):
        a.finish_source(body)
    assert a.cancelled
    with pytest.raises(ValueError, match="spent"):
        a.finish_source(body)


def test_malformed_share_burns_before_validation():
    a, _ = pair()
    with pytest.raises(ValueError, match="invalid source"):
        a.begin_source(np.zeros((1, 3), dtype=np.int64))
    assert a.cancelled
    with pytest.raises(ValueError, match="spent"):
        a.begin_source(np.zeros((2, 3), dtype=np.int64))


@pytest.mark.parametrize("fault", ["malformed_share", "malformed_response"])
def test_scalar_fault_burns_entire_joint_block(fault):
    a, b = pair()
    x = np.zeros_like(a.r)
    oa, ob = a.begin_source(x), b.begin_source(x)
    a.finish_source(ob)
    b.finish_source(oa)
    if fault == "malformed_share":
        with pytest.raises(ValueError, match="invalid scalar"):
            a.begin_scalar(np.zeros((3,), dtype=np.int64))
    else:
        a.begin_scalar(np.zeros_like(a.b))
        with pytest.raises(ValueError, match="binding"):
            a.finish_scalar(probe.frame(a.session, a.digest, "scalar", 0, probe.words(a.b)))
    assert a.cancelled and not np.any(a.r)
    with pytest.raises(ValueError, match="unavailable"):
        a.begin_scalar(np.zeros_like(a.b))


def test_actual_linear_image_rank_and_reused_mask_leak():
    # Full-rank public map gives no entropy compression. Rank-deficient mask
    # would disclose a left-nullspace component of the input opening.
    a = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.int64)
    assert probe.rank_mod(a, 5) == 2
    assert probe.rank_mod(np.asarray([[1, 2], [2, 4]], dtype=np.int64), 5) == 1
    r = np.asarray([3, 4])
    x, y = np.asarray([1, 2]), np.asarray([2, 0])
    np.testing.assert_array_equal(((x - r) - (y - r)) % 5, (x - y) % 5)


def test_small_field_joint_distribution_view():
    # Fix party 0's source-mask shares. Enumerate its independent output pads:
    # q1,c1 are uniform conditional on party 1's local r1,b1. This support check
    # does not establish security of any seeded generator.
    p = 5
    r0, b0 = 2, 3
    for r1 in range(p):
        for b1 in range(p):
            seen = set()
            for q0 in range(p):
                for c0 in range(p):
                    q1 = ((r0 + r1) ** 2 - q0) % p
                    c1 = ((r0 + r1) * (b0 + b1) - c0) % p
                    seen.add((q1, c1))
            assert len(seen) == p * p


def test_complete_region_screen_reproduces_norm_veto():
    cohorts = probe.region_screen()["cohorts"]
    expected = {8: (16149248, 31689216), 32: (24717056, 48222720)}
    for row in cohorts:
        full = row["norm_scenarios"][0]
        assert (
            full["norm_plus_boundary_online_floor_bytes"],
            full["explicit_joint_norm_material_storage_bytes"],
        ) == expected[row["output_tokens"]]
        assert full["exceeds_online_budget"]
        assert full["exceeds_all_link_budget_before_material"]
        assert not row["complete_region_admitted"]


def test_projected_cubic_expansion_with_quadratic_features():
    p = probe.P
    g, u, d = probe.projections(3, 4, 2)
    r, delta = probe.random_residues((2, 3)), probe.random_residues((2, 3))
    a, b = r @ g.T % p, r @ u.T % p
    v, w = delta @ g.T % p, delta @ u.T % p
    alpha, beta = 256, 1
    const = (((alpha * a + beta * a * a) % p) * b) % p
    expanded = (
        const
        + (alpha * v + beta * v * v) * w
        + (alpha * v + beta * v * v) * b
        + (alpha * w + 2 * beta * v * w) * a
        + beta * w * ((a * a) % p)
        + 2 * beta * v * ((a * b) % p)
    ) % p
    ag, bu = (a + v) % p, (b + w) % p
    direct = (((alpha * ag + beta * ag * ag) % p) * bu) % p
    np.testing.assert_array_equal((expanded @ d.T) % p, (direct @ d.T) % p)
