"""Independent toy consumer, shortcut and exact finite-budget checks."""
import importlib.util
import copy
from fractions import Fraction
from pathlib import Path
import random

import pytest

SPEC = importlib.util.spec_from_file_location(
    "stream_probe", Path(__file__).resolve().parents[1] / "scripts/probe_streaming_matrix_correlation.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


@pytest.mark.parametrize("bits", [3, 16, 24, 32])
def test_ot_full_ring_endpoints(bits):
    w, r = [[-128, 127], [3, -7]], [0, (1 << bits)-1]
    s, q = probe.ideal_vector_ot(w, r, bits, random.Random(43))
    expected = [sum(a*b for a, b in zip(row, r)) % (1 << bits) for row in w]
    assert [(a+b) % (1 << bits) for a, b in zip(s, q)] == expected


def test_exhaustive_provider_view_and_attacks():
    result = probe.privacy_screen()
    assert result["full_view_independent_masks_tv"] == "0"
    assert result["s_equals_2r_W_equals_3_tv"] == "1"
    assert result["rank_one_mask_tv"] == "1"
    assert result["reused_r_opening_difference"] == 3


def test_public_delta_recovers_other_message_without_sender_input_leakage():
    result = probe.public_delta_screen()
    assert result["public_delta_recovered_other_messages"] == 224
    assert result["full_sender_masks_and_opening_tv"] == "0"


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "ring", "width"])
def test_geometry_rejects_ingress_identity_or_ring_tampering(mutation):
    common = {"stage": "toy", "shape": [1, 1], "weight_sha256": "public-toy"}
    ingress = [{**common, "original_ring_bits": 24, "stage_bits": 23}]
    narrow = [{**common, "stage_bits": 23}]
    demand = [common]
    geometry = [{**common, "ring_bits": 24}]
    probe.validate_geometry(ingress, narrow, demand, geometry, {"toy": {}})
    tampered = copy.deepcopy(ingress)
    if mutation == "duplicate":
        tampered.append(copy.deepcopy(tampered[0]))
    elif mutation == "missing":
        tampered.clear()
    elif mutation == "ring":
        tampered[0]["original_ring_bits"] = 32
    else:
        tampered[0]["stage_bits"] = 24
    with pytest.raises(ValueError):
        probe.validate_geometry(tampered, narrow, demand, geometry, {"toy": {}})


def test_consumer_independent_oracle():
    assert probe.algebra_screen()["ideal_ot_checked_outputs"] == 96


def test_shared_authority_blocks_clones_restart_cancel_expiry():
    authority = probe.TicketBook(4, 10)
    clone = authority  # Every restored worker must consult SAME external authority.
    authority.claim("model/stage/epoch", 0, 1)
    with pytest.raises(ValueError, match="reused"):
        clone.claim("model/stage/epoch", 0, 1)
    for index, now, valid in [(1, 1, False), (2, 10, True)]:
        with pytest.raises(ValueError):
            authority.claim("model/stage/epoch", index, now, valid)
        with pytest.raises(ValueError, match="reused"):
            authority.claim("model/stage/epoch", index, 1)
    with pytest.raises(ValueError):
        authority.claim("model/stage/epoch", 4, 1)
    authority.claim("different-session", 0, 1)
    with pytest.raises(ValueError, match="total-use"):
        authority.claim("third-session", 0, 1)
    # Local rollback reset WOULD accept same ticket: counterexample, not protection.
    probe.TicketBook(4, 10).claim("model/stage/epoch", 0, 1)


def test_first_pair_endpoints_and_exact_headroom():
    for p in (20, 35, 50):
        target = Fraction(p, 100)
        assert probe.budget(1, target, True) == probe.budget(1, target, False)
    assert probe.budget(1, Fraction(1, 5), True) == Fraction(201728198, 5)
    assert probe.budget(1, Fraction(1, 2), True) < 0
    assert probe.budget(16, Fraction(35, 100), True) > 0


def test_renewal_discontinuity_and_finite_tape():
    target = Fraction(1, 5)
    # E=100MB every 4 pairs: headroom falls just after boundary.
    assert probe.budget(5, target, True, renewal=100_000_000, capacity=4) < probe.budget(
        4, target, True, renewal=100_000_000, capacity=4)
    for h in (1, 2, 16):
        assert probe.budget(h, target, True, bootstrap=h*139_060_689) < 0


def test_saved_geometry_and_chosen_message_cost():
    result = probe.cost_screen()
    assert len(result["stages"]) == 96
    assert sum(s["rows"] for s in result["stages"]) == 15297
    totals = result["totals"]
    assert totals["selected_vector_bytes"] > 1_000_000_000_000
    assert totals["two_chosen_message_ciphertext_payload_bytes"] == 2*totals["selected_vector_bytes"]
    assert totals["client_ot_lane_additions"] >= 24*totals["body_macs"]


@pytest.mark.parametrize("args", [(0, Fraction(1, 5), True), (1, Fraction(1), True)])
def test_invalid_budget(args):
    with pytest.raises(ValueError):
        probe.budget(*args)
